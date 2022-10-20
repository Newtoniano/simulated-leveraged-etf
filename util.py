"""Quantitative investing utility routines."""

import numpy as np
import pandas as pd
import yfinance

TRADING_DAYS = 252


def yget(tickers, start_date=None):
    """Download adjusted close prices for a sequence of `tickers` from Yahoo Finance."""
    tickers = tickers.split() if isinstance(tickers, str) else list(tickers)
    data = yfinance.download(
        tickers,
        start=start_date,
        period="max",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False # keep disabled if throttling or other issues with retrieving data
    )["Close"]
    if len(tickers) > 1:
        # yfinance may not return columns in the order asked, so we always re-order
        data = data[tickers]
    else:
        data.name = tickers[0]
    data.index = data.index.tz_localize(None)
    return data


def read_fred(name):
    """:Return: FRED series `name` as a pd.Series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}"
    return pd.to_numeric(
        pd.read_csv(url, index_col="DATE", parse_dates=["DATE"])
        .squeeze("columns")
        .replace(".", np.nan)
        .dropna()
    )


def cumret(prices):
    """:Return: The cumulative return of a Series or DataFrame of `prices`, 1.0 == 100%."""
    return prices.iloc[-1] / prices.iloc[0] - 1


def annret(prices, ann_periods=TRADING_DAYS):
    """:Return: The annualized returns of a DataFrame `prices`, one security per column.

    Note that to be strictly correct, if you are e.g. computing returns for a year of 252 days, you will need 253 prices
    including the last day of the previous year.
    """
    ret = (prices.iloc[-1] / prices.iloc[0]) ** (ann_periods / (len(prices) - 1)) - 1
    # len(prices) - 1 because for two prices, there's only one return (and only one price should fail)
    if isinstance(ret, pd.Series):  # Otherwise it's a scalar
        ret.rename("return", inplace=True)
    return ret


def annvol(prices, ann_periods=TRADING_DAYS):
    """:Return: The annualized volatility of a DataFrame of `prices`, one security per column."""
    vol = prices.pct_change().std() * np.sqrt(ann_periods)
    if isinstance(vol, pd.Series):  # Result is scalar for Series input
        vol.rename("volatility", inplace=True)
    return vol
