# README

## 1. DataDownload

In this project we assume we can assign a given stock universe to trade. Therefore, we need to download market data from Alpaca API. The methods have been implemented and wrapped into class `DataDownload` , which supports `first_download` and `update_to_today`.

## 2. Backtest

This class wraps several methods to get data and generate factors, as well as a `Backtest` framework to evaluate the result of a factor, so that we can judge if it is effective. This toolkit provides 2 backtest methods:

`quick_backtest` is a backtest with very simple logic, which runs fast.

`quick_backtest_2` is a modified version, which is more robust and supports more modes to backtest.  Like we can choose if considering transaction cost, if long only strategy or long-short strategy, what holding period we take.

The analysis results includes: annual return / annual sharpe / winning rate / ICIR.

Output graphs show the accumulative return for long/short, turnover rate, ticker number for long/short, cumulative IC for long/short.    

## 3. PaperTrding

A script to conduct daily strategy in paper trading. Run code in daily frequency.