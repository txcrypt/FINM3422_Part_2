import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class BasketOption:
    def __init__(
        self,
        basket_weights: dict,
        strike: float,
        expiry: datetime,
        valuation_date: datetime,
        rate: float,
        data_fetcher,
        M: int = 10000,
        N: int = 252,
        seed: int = None
    ):
        """
        Monte Carlo pricer for a European basket call option using DataFetcher.

        Parameters:
         - basket_weights: dict of {ticker: weight}
         - strike: strike price K
         - expiry: option expiry date (datetime)
         - valuation_date: pricing date (datetime)
         - rate: annual risk-free rate r
         - data_fetcher: instance of DataFetcher with .data loaded
         - M: number of Monte Carlo paths
         - N: number of time steps per path
         - seed: random seed for reproducibility
        """
        self.weights = pd.Series(basket_weights)
        self.K = strike
        self.expiry = pd.to_datetime(expiry)
        self.valuation_date = pd.to_datetime(valuation_date)
        self.r = rate
        self.M = M
        self.N = N
        self.seed = seed
        self.fetcher = data_fetcher

        # Fetch market data via DataFetcher
        # Ensure data_fetcher already loaded prices in its .data dict
        cov_ann = self.fetcher.get_covariance_matrix()
        # Annualized covariance matrix of returns
        self.cov_matrix = cov_ann

        # Spot prices and dividend yields per ticker
        spots = {}
        div_yields = {}
        for t in self.weights.index:
            sd = self.fetcher.get_stock_data(t)
            spots[t] = sd['spot']
            div_yields[t] = sd['dividend_yield']
        self.spot_prices = pd.Series(spots)
        self.div_yield = pd.Series(div_yields)

        # Basket spot and volatility
        self.S0 = (self.weights * self.spot_prices).sum()
        w = self.weights.values
        cov = self.cov_matrix.loc[self.weights.index, self.weights.index].values
        self.sigma = np.sqrt(w @ cov @ w)

        # Time to expiry
        delta = self.expiry - self.valuation_date
        self.T = delta.days / 365

    def _simulate_basket(self, M_sim: int = None):
        """Simulate basket price paths under risk-neutral drift r and no dividends."""
        M_sim = M_sim or self.M
        if self.seed is not None:
            np.random.seed(self.seed)
        dt = self.T / self.N
        drift = (self.r - 0.5 * self.sigma**2) * dt
        vol_step = self.sigma * np.sqrt(dt)

        S = np.empty((self.N + 1, M_sim))
        S[0] = self.S0
        for i in range(1, self.N + 1):
            Z = np.random.randn(M_sim)
            S[i] = S[i-1] * np.exp(drift + vol_step * Z)
        return S

    def price(self, M_sim: int = None) -> float:
        """Compute Monte Carlo price of the European basket call."""
        S = self._simulate_basket(M_sim)
        payoffs = np.maximum(S[-1] - self.K, 0)
        return np.exp(-self.r * self.T) * payoffs.mean()

    def delta(self, eps: float = 1.0, **mc_kwargs) -> float:
        """Finite-difference approximation of delta."""
        orig_S0 = self.S0
        self.S0 = orig_S0 + eps
        up = self.price(**mc_kwargs)
        self.S0 = orig_S0 - eps
        down = self.price(**mc_kwargs)
        self.S0 = orig_S0
        return (up - down) / (2 * eps)

    def vega(self, eps: float = 0.01, **mc_kwargs) -> float:
        """Finite-difference approximation of vega."""
        orig_sigma = self.sigma
        self.sigma = orig_sigma + eps
        up = self.price(**mc_kwargs)
        self.sigma = orig_sigma - eps
        down = self.price(**mc_kwargs)
        self.sigma = orig_sigma
        return (up - down) / (2 * eps)

    def gamma(self, eps: float = 1.0, **mc_kwargs) -> float:
        """Finite-difference approximation of gamma."""
        orig_S0 = self.S0
        self.S0 = orig_S0 + eps
        up = self.price(**mc_kwargs)
        self.S0 = orig_S0
        mid = self.price(**mc_kwargs)
        self.S0 = orig_S0 - eps
        down = self.price(**mc_kwargs)
        self.S0 = orig_S0
        return (up - 2*mid + down) / (eps**2)

    def rho(self, eps: float = 1e-4, **mc_kwargs) -> float:
        """Finite-difference approximation of rho."""
        orig_r = self.r
        self.r = orig_r + eps
        up = self.price(**mc_kwargs)
        self.r = orig_r - eps
        down = self.price(**mc_kwargs)
        self.r = orig_r
        return (up - down) / (2 * eps)

    def plot_paths(self, M_plot: int = 200):
        """Plot simulated basket paths."""
        S = self._simulate_basket(M_plot)
        t = np.linspace(0, self.T, self.N + 1)
        plt.figure(figsize=(10,6))
        for j in range(M_plot):
            plt.plot(t, S[:,j], color='grey', alpha=0.5)
        plt.title('Simulated Basket Price Paths')
        plt.xlabel('Time (Years)')
        plt.ylabel('Basket Value')
        plt.grid(True)
        plt.show()