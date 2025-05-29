import pandas as pd
import yfinance as yf
import numpy as np
import math
from datetime import datetime, timedelta
from Yield_curve.curve_classes import CurveBootstrapper

# --- Helper Functions ---

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

# --- American Option Class ---

from option_models.option import Option

class AmericanOption(Option):
    def __init__(self, S0, K, r, sigma, q, T, steps=200, opt_type="put"):
        super().__init__(S0, K, T, q, r, sigma)
        self.N = steps
        self.opt_type = opt_type  # "put" or "call"

    def price(self) -> float:
        """
        Binomial tree (CRR) pricing of American options.
        Supports early exercise — most suitable for American puts.
        """
        N = self.N
        dt = self.T / N
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1.0 / u
        pu = (math.exp((self.r - self.q) * dt) - d) / (u - d)
        pd = 1.0 - pu
        disc = math.exp(-self.r * dt)

        # Step 1: Terminal stock prices
        ST = [self.S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

        # Step 2: Terminal payoffs
        if self.opt_type == "put":
            V = [max(self.K - s, 0.0) for s in ST]
        else:
            V = [max(s - self.K, 0.0) for s in ST]

        # Step 3: Backward induction for early exercise
        for i in range(N - 1, -1, -1):
            ST = [s / d for s in ST[: i + 1]]  # Recompute underlying prices
            V_next = []
            for j in range(i + 1):
                cont = disc * (pu * V[j + 1] + pd * V[j])
                if self.opt_type == "put":
                    exercise = max(self.K - ST[j], 0.0)
                else:
                    exercise = max(ST[j] - self.K, 0.0)
                V_next.append(max(cont, exercise))
            V = V_next

        return float(V[0])

# --- Main Script ---

if __name__ == "__main__":
    # 1. Setup parameters
    val_date = datetime(2025, 5, 16)
    expiry = datetime(2026, 5, 15)
    ticker = "CBA.AX"
    K = 170.0

    # 2. Pull 1 year of historical data
    start = (val_date - timedelta(days=365)).strftime("%Y-%m-%d")
    end = (val_date + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=False)
    df.index = df.index.tz_localize(None)
    if df.empty:
        raise RuntimeError("No price data returned")

    # 3. Get last closing price
    S0 = df["Close"].iloc[-1].item()

    # 4. Download dividend history
    tk = yf.Ticker(ticker)
    divs = tk.dividends.copy()
    divs.index = divs.index.tz_localize(None)
    divs = divs[(divs.index >= df.index.min()) & (divs.index <= df.index.max())]

    # 5. Calibrate key parameters
    sigma = estimate_vol(df["Close"])
    q = estimate_div_yield(divs, df["Close"])
    r = get_rf_rate(val_date, expiry)
    T = (expiry - val_date).days / 365

    # 6. Price American put
    am_put = AmericanOption(S0, K, r, sigma, q, T,
                            steps=200, opt_type="put")
    price = am_put.price()

    # 7. Output
    print(f"Valuation date: {val_date.date()}")
    print(f"Spot S0         = {S0:.2f}")
    print(f"σ (vol)         = {sigma:.4f}")
    print(f"q (div yield)   = {q:.4f}")
    print(f"r (rf rate)     = {r:.4f}")
    print(f"T (years)       = {T:.4f}")
    print(f"American Put    = ${price:.4f}")
