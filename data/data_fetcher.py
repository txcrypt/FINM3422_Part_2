import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

class DataFetcher:
    def __init__(self, tickers=None, start_date=None, end_date=None, csv_path: str = None):
        """
        tickers: optional list of ticker symbols; if None, uses keys of implied_volatility
        start_date, end_date: optional str or datetime-like for date filtering when fetching from yfinance
        csv_path: optional path to CSV containing closing prices (dates as index, tickers as columns)
        """
        # Pre-defined volatility and dividend data
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

        # Set tickers to provided list or default to implied_volatility keys
        self.tickers = tickers if tickers is not None else list(self.implied_volatility.keys())
        self.data = {}  # { ticker: pd.Series of Close prices }

        # Optional date filters for yfinance fetch
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # Load data from CSV or fetch via yfinance
        if csv_path:
            self.load_data_from_csv(csv_path)
        else:
            self.fetch_data()

    def load_data_from_csv(self, csv_path: str):
        """
        Load historical closing prices from a CSV file.
        Uses all timestamps in the CSV without date filtering.
        """
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        for t in self.tickers:
            if t not in df.columns:
                raise KeyError(f"Ticker {t} not found in CSV columns")
            self.data[t] = df[t].tz_localize(None)

    def fetch_data(self):
        """Populate self.data[ticker] with a pd.Series of Close prices from yfinance."""
        for t in self.tickers:
            df = yf.download(
                t,
                start=self.start_date.strftime("%Y-%m-%d") if self.start_date else None,
                end=(self.end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if self.end_date else None,
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
        return (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan

    def get_stock_data(self, ticker) -> dict:
        """
        Returns a dict with:
          - annual_return       (float)
          - implied_volatility  (float)
          - spot                (float)
          - dividend_yield      (float)
        """
        prices = self.data[ticker]
        ann_ret = self.get_returns(ticker)
        log_rets = np.log(prices / prices.shift(1)).dropna()
        vol = float(log_rets.std() * np.sqrt(252))
        self.implied_volatility[ticker] = vol
        spot = float(prices.iloc[-1])
        div_yield = self.historical_div.get(ticker, np.nan)
        return {
            "annual_return": ann_ret,
            "implied_volatility": vol,
            "spot": spot,
            "dividend_yield": div_yield
        }

    def get_covariance_matrix(self, tickers: list = None) -> pd.DataFrame:
        """
        Builds the annualized covariance matrix of mean-centered daily pct-change returns
        for the specified tickers.
        
        :param tickers: list of ticker strings to include; defaults to self.tickers
        :return: DataFrame of annualized covariances
        """
        # default to all
        tickers = tickers or self.tickers

        # build daily-return DataFrame only for the requested tickers
        df_rets = pd.DataFrame({
            t: self.data[t].pct_change().dropna()
            for t in tickers
        })

        # mean-center and annualize
        df_centered = df_rets - df_rets.mean()
        cov_ann = df_centered.cov() * 252
        return cov_ann
