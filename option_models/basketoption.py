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
        Monte Carlo pricer for a European basket call option.

        :param basket_weights: dict of {ticker: weight}
        :param strike: strike price K
        :param expiry: option expiry date
        :param valuation_date: pricing date
        :param rate: annual risk-free rate r
        :param data_fetcher: Dataloader instance providing get_stock_data and get_covariance_matrix
        :param M: number of Monte Carlo paths
        :param N: number of time steps per path
        :param seed: random seed for reproducibility
        """
        # store inputs
        self.weights = pd.Series(basket_weights)
        self.K = strike
        self.expiry = pd.to_datetime(expiry)
        self.valuation_date = pd.to_datetime(valuation_date)
        self.r = rate
        self.M = M
        self.N = N
        self.seed = seed
        self.fetcher = data_fetcher

        # time to expiry in years
        delta = self.expiry - self.valuation_date
        self.T = delta.days / 365.0

        spots = {}
        divs = {}
        vols = {}
        avg_rets = {}
        for t in self.weights.index:
            sd = self.fetcher.get_stock_data(t)
            # extract numeric values by key
            spot    = sd["spot"]
            div     = sd["dividend_yield"]
            vol     = sd["implied_volatility"]
            avg_ret = sd["annual_return"]
            spots[t]    = spot
            divs[t]     = div
            vols[t]     = vol
            avg_rets[t] = avg_ret

        self.spot_prices = pd.Series(spots, dtype=float)
        self.div_yield   = pd.Series(divs, dtype=float)
        self.vols        = pd.Series(vols, dtype=float)
        self.avg_returns = pd.Series(avg_rets, dtype=float)

        tickers = self.weights.index.tolist()
        self.cov_matrix = self.fetcher.get_covariance_matrix(tickers)

        w = self.weights.values
        s = self.spot_prices.values
        self.S0 = np.dot(w, s)

        cov_vals = self.cov_matrix.values
        self.sigma = np.sqrt(np.dot(w, cov_vals.dot(w)))
        ar = self.avg_returns.values
        self.mu = np.dot(w, ar)
        self.drift_rate = self.r

    def get_return(self):
        return self.mu

    def _simulate_basket(self, M_sim: int = None):
        """Simulate basket paths under current drift_rate and volatility."""
        M_sim = M_sim or self.M
        if self.seed is not None:
            np.random.seed(self.seed)
        dt = self.T / self.N
        drift = (self.drift_rate - 0.5 * self.sigma**2) * dt
        vol_step = self.sigma * np.sqrt(dt)

        S = np.empty((self.N + 1, M_sim))
        S[0] = self.S0
        for i in range(1, self.N + 1):
            Z = np.random.randn(M_sim)
            S[i] = S[i-1] * np.exp(drift + vol_step * Z)
        return S

    def price(self, M_sim: int = None) -> float:
        """Monte Carlo price of the European basket call."""
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

    def price_risk_free(self, M_sim: int = None) -> float:
        """Price using risk-free drift."""
        self.drift_rate = self.r
        return self.price(M_sim)

    def price_historical(self, M_sim: int = None) -> float:
        """Price using historical basket return as drift."""
        self.drift_rate = self.mu
        return self.price(M_sim)

    def plot_diffusion_risk_free(self, M_plot: int = 200):
        """Plot diffusion with risk-free drift."""
        self.drift_rate = self.r
        self.plot_paths(M_plot)

    def plot_diffusion_historical(self, M_plot: int = 200):
        """Plot diffusion with historical return drift."""
        self.drift_rate = self.mu
        self.plot_paths(M_plot)
