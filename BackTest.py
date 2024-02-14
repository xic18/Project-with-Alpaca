import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import time
import os

# 简单的回测逻辑 会在price limit，边界点处理上有小问题
# transaction cost计算存在bug
# return和turnover都归属在T0 时间没对齐
# backtest时间区间没定义


class Backtest():
    '''
    规则：factor表示截止T0收盘后可获得所有数据，生成的因子，
    表示T0+interval[0]买，T0+interval[1]卖，所占仓位
    收益归属到T0
    回测默认overlap，如果不想overlap（按一定周期调仓），自己设置factor让overlap的部分仓位为0
    也就是说输入factor的频率应该和holding_interval配合
    上一期的factor的interval[1] <= 下一期的factor的interval[0]
    例1：
    最典型的daily因子，holding_interval=(1,2)
    T0给出因子值F0(未来holding period的仓位)，对应收益:R0=(lnP2-lenP1)*F0
    计费：T0先算无transaction R0，然后减去cost: C0
    C[0] = costrate * SUM(abs(F[0]-F[-1]))
    TVR[0,i] = pct_change(F[-1],F[0])
    '''
    def __init__(self,start_day=datetime.date(2023,12,4),end_day=datetime.date(2024,2,12),return_type='close',long_only=True,holding_interval=(1,2),CostRate=0.004,mask_overlap=True):
        self.start_day=start_day
        self.end_day=end_day
        self.return_type=return_type
        self.holding_interval=holding_interval
        self.CostRate=CostRate
        self.mask_overlap = mask_overlap
        self.long_only=long_only

        # load data方式 可以随便定义 最后存储结果一定要是一个ordered N*T length的dataframe
        self.raw_data = pd.read_csv('./data/raw_data/raw_data.csv')
        self.raw_data['date']=pd.to_datetime(self.raw_data['date']).dt.date
        with open('./data/stockUniverse.pickle', 'rb') as f:
            self.universe = pickle.load(f)

        self.TradingDay=sorted(np.unique(self.raw_data['date']))
        self.TradingDay=[x for x in self.TradingDay if ((x>=start_day) and (x<=end_day))]
        self.raw_data=self.raw_data.rename(columns={'date':'Date','symbol':'Ticker'})

    def gen_benchmark(self, portfolio_type): # return normalized factor
        if portfolio_type == 'equal_weight_long':
            factor = self.raw_data.copy()[['Date', 'Ticker']]
            factor['factor'] = 1
            factor['factor'] = factor.groupby('Date')['factor'].apply(lambda x: x / np.nansum(x)).droplevel(0)
        elif portfolio_type == 'random_long_short':
            def adjFunc(x, long_max, short_max):
                res = x.copy()
                sump = np.nansum(x[x > 0])
                sumn = np.nansum(x[x < 0])
                res[x > 0] = res[x > 0] * long_max / sump
                res[x < 0] = -res[x < 0] * short_max / sumn
                return res

            factor = self.raw_data.copy()[['Date', 'Ticker']]
            factor['factor'] = np.random.randn(len(factor))
            factor['factor'] = factor.groupby('Date')['factor'].apply(lambda x: adjFunc(x, 1, 1)).droplevel(0)
        elif portfolio_type == 'random_long':
            factor = self.raw_data.copy()[['Date', 'Ticker']]
            factor['factor'] = np.random.random(len(factor))
            factor['factor'] = factor.groupby('Date')['factor'].apply(lambda x: x / np.nansum(x)).droplevel(0)

        else:
            factor = None
        return factor

    def factor_preprocess(self,factor): # preprocess in place
        factor['factor'] = factor['factor'].round(5)

        def adjFunc(x, long_max, short_max):
            res = x.copy()
            sump = np.nansum(x[x > 0])
            sumn = np.nansum(x[x < 0])
            res[x > 0] = res[x > 0] * long_max / sump
            res[x < 0] = -res[x < 0] * short_max / sumn
            return res

        factor['factor'] = factor.groupby('Date')['factor'].apply(lambda x: adjFunc(x, 1, 1)).droplevel(0)

        return factor

    def quick_backtest(self,factor):
        backtest_data = self.raw_data.copy()
        backtest_data = backtest_data[(backtest_data['Date'] >= self.start_day) & (backtest_data['Date'] <= self.end_day)]

        # factor preprocess in place
        # self.factor_preprocess(factor)

        # mask overlap
        if self.mask_overlap:
            rebalance_date = [self.TradingDay[i] for i in range(len(self.TradingDay)) if i % (self.holding_interval[1] - self.holding_interval[0]) == 0]
            factor = factor.set_index('Date').loc[rebalance_date].reset_index()
        else:
            print("do not support overlap")
            raise RuntimeError

        backtest_data['Ret'] = backtest_data[self.return_type].apply(lambda x: np.log(x))
        backtest_data['Ret'] = backtest_data.groupby('Ticker')['Ret'].diff(self.holding_interval[1] - self.holding_interval[0]).shift(-self.holding_interval[1])
        backtest_data = pd.merge(backtest_data, factor, how='left')

        backtest_data['position_change'] = backtest_data.groupby('Ticker')['factor'].diff(self.holding_interval[1] - self.holding_interval[0]).abs()
        cost = backtest_data.groupby('Date')['position_change'].sum() * self.CostRate
        ret = backtest_data.groupby('Date')[['Ret', 'factor']].apply(lambda x: np.sum(x['Ret'] * x['factor']))
        ret = ret - cost

        turnover = backtest_data.groupby('Date')['position_change'].sum()

        # PNL for long and short
        backtest_data['long_profit'] = 0
        backtest_data['short_profit'] = 0
        backtest_data.loc[backtest_data['factor'] >= 0, 'long_profit'] = \
            backtest_data[backtest_data['factor'] >= 0]['Ret'] * backtest_data[backtest_data['factor'] >= 0]['factor']\
            - backtest_data[backtest_data['factor'] >= 0]['position_change'] * self.CostRate
        backtest_data.loc[backtest_data['factor'] < 0, 'short_profit'] = \
            backtest_data[backtest_data['factor'] < 0]['Ret'] * backtest_data[backtest_data['factor'] < 0][ 'factor'] \
            - backtest_data[backtest_data['factor'] < 0]['position_change'] * self.CostRate

        plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(2,2,1)
        ret.cumsum().plot(ax=ax1,label='total',grid=True)
        backtest_data.groupby('Date')['long_profit'].sum().cumsum().plot(ax=ax1,label='long',grid=True)
        backtest_data.groupby('Date')['short_profit'].sum().cumsum().plot(ax=ax1,label='short',grid=True)
        plt.legend()
        plt.title('Cumulative Return')
        # plt.show()

        pnl_long = backtest_data.groupby('Date')['long_profit'].sum()
        pnl_short = backtest_data.groupby('Date')['short_profit'].sum()
        # 这里就是关于中低频因子的收益定义问题，我们假设H天频率的策略收益平均分配到holding period的每一天
        ret=ret.rolling(self.holding_interval[1] - self.holding_interval[0]).mean()
        pnl_long=pnl_long.rolling(self.holding_interval[1] - self.holding_interval[0]).mean()
        pnl_short=pnl_short.rolling(self.holding_interval[1] - self.holding_interval[0]).mean()

        pnl_summary = pd.DataFrame(index=['total', 'long', 'short'],columns=['Annual Return', 'Annual Sharpe', 'Winning Rate'])
        pnl_summary.iloc[0] = ret.mean() * 252, ret.mean() / ret.std() * np.sqrt(252), ret[ret > 0].count() / ret.count()
        pnl_summary.iloc[1] = pnl_long.mean() * 252, pnl_long.mean() / pnl_long.std() * np.sqrt(252), pnl_long[
            pnl_long > 0].count() / pnl_long.count()
        pnl_summary.iloc[2] = pnl_short.mean() * 252, pnl_short.mean() / pnl_short.std() * np.sqrt(252), pnl_short[
            pnl_short > 0].count() / pnl_short.count()
        print(pnl_summary)

        # turn over
        ax2 = plt.subplot(2,2,2)
        turnover.plot(ax=ax2,grid=True)
        plt.title('Turnover')

        # plt.show()

        # long short tickers count
        ax4 = plt.subplot(2,2,4)
        backtest_data.groupby('Date')['factor'].apply(lambda x: x[x > 0].count()).plot(ax=ax4,label='long tickers',grid=True)
        backtest_data.groupby('Date')['factor'].apply(lambda x: x[x < 0].count()).plot(ax=ax4,label='short tickers',grid=True)
        plt.title('Ticker Number')
        plt.legend()
        # plt.show()

        # IC analysis
        IC = backtest_data.groupby('Date')[['Ret', 'factor']].apply(lambda x: x[x['factor'] != 0].corr(method='pearson').iloc[0, 1])
        IC_long = backtest_data.groupby('Date')[['Ret', 'factor']].apply(lambda x: x[x['factor'] > 0].corr(method='pearson').iloc[0, 1])
        IC_short = backtest_data.groupby('Date')[['Ret', 'factor']].apply(lambda x: x[x['factor'] < 0].corr(method='pearson').iloc[0, 1])
        Rank_IC = backtest_data.groupby('Date')[['Ret', 'factor']].apply(lambda x: x[x['factor'] != 0].corr(method='spearman').iloc[0, 1])
        Rank_IC_long = backtest_data.groupby('Date')[['Ret', 'factor']].apply(lambda x: x[x['factor'] > 0].corr(method='spearman').iloc[0, 1])
        Rank_IC_short = backtest_data.groupby('Date')[['Ret', 'factor']].apply(lambda x: x[x['factor'] < 0].corr(method='spearman').iloc[0, 1])
        ax3 = plt.subplot(2,2,3)
        IC.fillna(0).cumsum().plot(ax=ax3,label='total ic', grid=True)
        IC_long.fillna(0).cumsum().plot(ax=ax3,label='long ic', grid=True)
        IC_short.fillna(0).cumsum().plot(ax=ax3,label='short ic', grid=True)
        plt.title('Cumulative IC')
        plt.legend()

        IC_summary = pd.DataFrame(index=['total', 'long', 'short'], columns=['mean IC', 'IR', 'mean rankIC', 'rankIR'])
        IC_summary.iloc[0] = IC.mean(), IC.mean() / IC.std(), Rank_IC.mean(), Rank_IC.mean() / Rank_IC.std()
        IC_summary.iloc[1] = IC_long.mean(), IC_long.mean() / IC_long.std(), Rank_IC_long.mean(), Rank_IC_long.mean() / Rank_IC_long.std()
        IC_summary.iloc[2] = IC_short.mean(), IC_short.mean() / IC_short.std(), Rank_IC_short.mean(), Rank_IC_short.mean() / Rank_IC_short.std()
        print(IC_summary)

        plt.show()
        return {'pnl_summary': pnl_summary, 'IC_summary': IC_summary}

    def quick_backtest_2(self,factor):
        backtest_data = self.raw_data.copy()
        backtest_data = backtest_data[(backtest_data['Date'] >= self.start_day) & (backtest_data['Date'] <= self.end_day)]

        # factor preprocess in place
        # self.factor_preprocess(factor)

        # mask overlap
        if self.mask_overlap:
            rebalance_date = [self.TradingDay[i] for i in range(len(self.TradingDay)) if i % (self.holding_interval[1] - self.holding_interval[0]) == 0]
            factor = factor.set_index('Date').loc[rebalance_date].reset_index()

        else:
            print("do not support overlap")
            raise RuntimeError

        backtest_data['Ret'] = backtest_data[self.return_type].apply(lambda x: np.log(x))
        backtest_data['Ret'] = backtest_data.groupby('Ticker')['Ret'].diff(1).shift(-self.holding_interval[0])
        backtest_data = pd.merge(backtest_data, factor, how='left')
        if self.holding_interval[1] - self.holding_interval[0]-1>0:
            backtest_data['factor_filled']=backtest_data.groupby('Ticker')['factor'].fillna(method='ffill',limit=self.holding_interval[1] - self.holding_interval[0]-1)
        else:
            backtest_data['factor_filled'] =backtest_data['factor']
        backtest_data['position_change'] = backtest_data.groupby('Ticker')['factor'].diff(self.holding_interval[1] - self.holding_interval[0]).abs()
        cost = backtest_data.groupby('Date')['position_change'].sum() * self.CostRate
        ret = backtest_data.groupby('Date')[['Ret', 'factor_filled']].apply(lambda x: np.sum(x['Ret'] * x['factor_filled']))
        ret = ret - cost

        turnover = backtest_data.groupby('Date')['position_change'].sum()

        # PNL for long and short
        backtest_data['long_profit'] = 0
        backtest_data['short_profit'] = 0
        backtest_data.loc[backtest_data['factor_filled'] >= 0, 'long_profit'] = backtest_data[backtest_data['factor_filled'] >= 0]['Ret'] * backtest_data[backtest_data['factor_filled'] >= 0]['factor_filled']\
            - backtest_data[backtest_data['factor_filled'] >= 0]['position_change'].fillna(0) * self.CostRate
        backtest_data.loc[backtest_data['factor_filled'] < 0, 'short_profit'] = \
            backtest_data[backtest_data['factor_filled'] < 0]['Ret'] * backtest_data[backtest_data['factor_filled'] < 0][ 'factor_filled'] \
            - backtest_data[backtest_data['factor_filled'] < 0]['position_change'].fillna(0) * self.CostRate

        plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(2,2,1)
        ret.cumsum().plot(ax=ax1,label='total',grid=True)
        backtest_data.groupby('Date')['long_profit'].sum().cumsum().plot(ax=ax1,label='long',grid=True)
        backtest_data.groupby('Date')['short_profit'].sum().cumsum().plot(ax=ax1,label='short',grid=True)
        plt.legend()
        plt.title('Cumulative Return')

        pnl_long = backtest_data.groupby('Date')['long_profit'].sum()
        pnl_short = backtest_data.groupby('Date')['short_profit'].sum()

        pnl_summary = pd.DataFrame(index=['total', 'long', 'short'],columns=['Annual Return', 'Annual Sharpe', 'Winning Rate'])
        pnl_summary.iloc[0] = ret.mean() * 252, ret.mean() / ret.std() * np.sqrt(252), ret[ret > 0].count() / ret.count()
        pnl_summary.iloc[1] = pnl_long.mean() * 252, pnl_long.mean() / pnl_long.std() * np.sqrt(252), pnl_long[
            pnl_long > 0].count() / pnl_long.count()
        pnl_summary.iloc[2] = pnl_short.mean() * 252, pnl_short.mean() / pnl_short.std() * np.sqrt(252), pnl_short[
            pnl_short > 0].count() / pnl_short.count()
        print(pnl_summary)

        # turn over
        ax2 = plt.subplot(2,2,2)
        turnover.plot(ax=ax2,grid=True)
        plt.title('Turnover')

        # plt.show()

        # long short tickers count
        ax4 = plt.subplot(2,2,4)
        backtest_data.groupby('Date')['factor_filled'].apply(lambda x: x[x > 0].count()).plot(ax=ax4,label='long tickers',grid=True)
        backtest_data.groupby('Date')['factor_filled'].apply(lambda x: x[x < 0].count()).plot(ax=ax4,label='short tickers',grid=True)
        plt.title('Ticker Number')
        plt.legend()
        # plt.show()

        # IC analysis no fee
        IC = backtest_data.groupby('Date')[['Ret', 'factor_filled']].apply(lambda x: x[x['factor_filled'] != 0].corr(method='pearson').iloc[0, 1])
        IC_long = backtest_data.groupby('Date')[['Ret', 'factor_filled']].apply(lambda x: x[x['factor_filled'] > 0].corr(method='pearson').iloc[0, 1])
        IC_short = backtest_data.groupby('Date')[['Ret', 'factor_filled']].apply(lambda x: x[x['factor_filled'] < 0].corr(method='pearson').iloc[0, 1])
        Rank_IC = backtest_data.groupby('Date')[['Ret', 'factor_filled']].apply(lambda x: x[x['factor_filled'] != 0].corr(method='spearman').iloc[0, 1])
        Rank_IC_long = backtest_data.groupby('Date')[['Ret', 'factor_filled']].apply(lambda x: x[x['factor_filled'] > 0].corr(method='spearman').iloc[0, 1])
        Rank_IC_short = backtest_data.groupby('Date')[['Ret', 'factor_filled']].apply(lambda x: x[x['factor_filled'] < 0].corr(method='spearman').iloc[0, 1])
        ax3 = plt.subplot(2,2,3)
        IC.fillna(0).cumsum().plot(ax=ax3,label='total ic', grid=True)
        IC_long.fillna(0).cumsum().plot(ax=ax3,label='long ic', grid=True)
        IC_short.fillna(0).cumsum().plot(ax=ax3,label='short ic', grid=True)
        plt.title('Cumulative IC')
        plt.legend()

        IC_summary = pd.DataFrame(index=['total', 'long', 'short'], columns=['mean IC', 'IR', 'mean rankIC', 'rankIR'])
        IC_summary.iloc[0] = IC.mean(), IC.mean() / IC.std(), Rank_IC.mean(), Rank_IC.mean() / Rank_IC.std()
        IC_summary.iloc[1] = IC_long.mean(), IC_long.mean() / IC_long.std(), Rank_IC_long.mean(), Rank_IC_long.mean() / Rank_IC_long.std()
        IC_summary.iloc[2] = IC_short.mean(), IC_short.mean() / IC_short.std(), Rank_IC_short.mean(), Rank_IC_short.mean() / Rank_IC_short.std()
        print(IC_summary)

        plt.show()
        return {'pnl_summary': pnl_summary, 'IC_summary': IC_summary}


if __name__=='__main__':

    backtest=Backtest(holding_interval=(1,2),CostRate=0.00,mask_overlap=True)
    # factor=backtest.gen_benchmark('equal_weight_long')
    # factor=backtest.gen_benchmark('random_long')
    factor=backtest.gen_benchmark('random_long_short')

    # backtest.quick_backtest(factor)
    backtest.quick_backtest_2(factor)
