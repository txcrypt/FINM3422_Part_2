import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

class DataFetcher:
    def __init__(self, tickers, start_date, end_date, csv_path: str = None):
        """
        tickers: list of ticker symbols
        start_date, end_date: str or datetime-like for date filtering
        csv_path: optional path to CSV containing closing prices (dates as index, tickers as columns)
        """
        # Ensure dates are Timestamps
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.tickers = tickers
        self.data = {}  # { ticker: pd.Series of Close prices }

        # Pre-defined volatility and dividend data (optional)
        self.implied_volatility = {
            "BHP.AX": 0.2513,
            "CBA.AX": 0.19108,
            "WES.AX": 0.19287,
            "CSL.AX": 0.22456,
            "WDS.AX": 0.27865,
            "MQG.AX": 0.22412,
        }
        self.historical_div = {
            "BHP.AX": 0.0492,
            "CBA.AX": 0.0273,
            "WES.AX": 0.0244,
            "CSL.AX": 0.0172,
            "WDS.AX": 0.0841,
            "MQG.AX": 0.031,
        }

        if csv_path:
            self.load_data_from_csv(csv_path)
        else:
            self.fetch_data()

    def load_data_from_csv(self, csv_path: str):
        """
        Load historical closing prices from a CSV file.
        The CSV should have dates as the index and tickers as columns.
        """
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        # Filter by date range
        df = df.loc[self.start_date : self.end_date]
        for t in self.tickers:
            if t not in df.columns:
                raise KeyError(f"Ticker {t} not found in CSV columns")
            self.data[t] = df[t].tz_localize(None)

    def fetch_data(self):
        """Populate self.data[ticker] with a pd.Series of Close prices from yfinance."""
        for t in self.tickers:
            df = yf.download(
                t,
                start=self.start_date.strftime("%Y-%m-%d"),
                end=(self.end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
            if df.empty:
                raise RuntimeError(f"No price data for {t}")
            self.data[t] = df["Close"].tz_localize(None)

    def get_returns(self, ticker) -> float:
        """
        Geometric annualized return over the period:
        (1 + total_return) ** (1/years) - 1
        """
        prices = self.data[ticker]
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        total_ret = end_price / start_price - 1.0
        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        if years <= 0:
            return np.nan
        return (1 + total_ret) ** (1 / years) - 1

    def get_stock_data(self, ticker) -> dict:
        """
        Returns a dict with:
          - annual_return       (float)
          - implied_volatility  (float)
          - spot                (float)
          - dividend_yield      (float)
        """
        prices = self.data[ticker]
        # 1) annualized return
        ann_ret = self.get_returns(ticker)
        # 2) historical vol (proxy for implied vol)
        log_rets = np.log(prices / prices.shift(1)).dropna()
        vol = float(log_rets.std() * np.sqrt(252))
        self.implied_volatility[ticker] = vol
        # 3) spot price
        spot = float(prices.iloc[-1])
        # 4) dividend yield (fallback to historical_div if no dividends in period)
        div_yield = self.historical_div.get(ticker, np.nan)
        return {
            "annual_return": ann_ret,
            "implied_volatility": vol,
            "spot": spot,
            "dividend_yield": div_yield
        }

    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        Builds the annualized covariance matrix of mean-centered daily pct-change returns
        for all tickers in self.tickers.
        """
        # Build DataFrame of daily percent returns
        df_rets = pd.DataFrame({
            t: self.data[t].pct_change().dropna()
            for t in self.tickers
        })
        # Mean-center returns
        df_centered = df_rets - df_rets.mean()
        # Unbiased covariance and annualize
        cov_ann = df_centered.cov() * 252
        return cov_ann
