import pandas as pd
import yfinance as yf
import numpy as np
import math
from datetime import datetime, timedelta
from Yield_curve.curve_classes import CurveBootstrapper

def estimate_vol(prices: pd.Series) -> float:
    """
    Estimate annualised volatility from daily log-returns.
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    return float(log_rets.std() * np.sqrt(252))  # 252 trading days per year

def estimate_div_yield(div_series: pd.Series, price_series) -> float:
    """
    Estimate dividend yield as: total dividends / average price.
    Handles both Series and one-column DataFrames for price_series.
    """
    if isinstance(price_series, pd.DataFrame) and price_series.shape[1] == 1:
        price_series = price_series.iloc[:, 0]
    elif isinstance(price_series, pd.DataFrame):
        price_series = price_series["Close"]

    divs, prices = div_series.align(price_series, join="inner")
    total_div = divs.sum()
    avg_price = prices.mean()
    if isinstance(avg_price, pd.Series):
        avg_price = float(avg_price.iloc[0])
    if avg_price <= 0 or pd.isna(avg_price):
        return 0.0
    return float(total_div / avg_price)

def get_rf_rate(val_date: datetime, expiry: datetime) -> float:
    """
    Placeholder for risk-free rate.
    Could later be replaced with Bloomberg-derived curve from curve_classes.py.
    """
    return 0.035  # 3.5% p.a.
