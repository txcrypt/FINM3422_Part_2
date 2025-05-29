import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Option:
    """
    Abstract base class for all option types.
    Provides core attributes, Greeks, and plotting utilities.
    """

    def __init__(self, S0, K, T, q, r, sigma, valuation_date=None):
        """
        Initialize common parameters.

        Parameters:
        - S0 (float): Spot price of underlying
        - K (float): Strike price
        - T (float): Time to maturity (in years)
        - r (float): Risk-free rate (continuously compounded)
        - sigma (float): Volatility (annualized)
        - valuation_date (datetime): Pricing date (default = 16 May 2025)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.valuation_date = valuation_date or datetime(2025, 5, 16)

    def price(self):
        """
        Return the option price.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must override price()")

    def _extra_args(self):
        """
        Provide extra arguments required by subclasses (e.g. barrier level).
        Override in subclass if needed.
        """
        return []

    def delta(self, epsilon=0.01, *args, **kwargs):
        """
        Finite difference Delta approximation.
        For use with smooth analytic models (not Monte Carlo).
        """
        up = self.__class__(self.S0 + epsilon, self.K, self.T, self.r, self.sigma,
                            valuation_date=self.valuation_date, *self._extra_args())
        down = self.__class__(self.S0 - epsilon, self.K, self.T, self.r, self.sigma,
                              valuation_date=self.valuation_date, *self._extra_args())
        return (up.price(*args, **kwargs) - down.price(*args, **kwargs)) / (2 * epsilon)

    def vega(self, epsilon=0.01, *args, **kwargs):
        """
        Finite difference Vega approximation.
        For use with smooth analytic models (not Monte Carlo).
        """
        up = self.__class__(self.S0, self.K, self.T, self.r, self.sigma + epsilon,
                            valuation_date=self.valuation_date, *self._extra_args())
        down = self.__class__(self.S0, self.K, self.T, self.r, self.sigma - epsilon,
                              valuation_date=self.valuation_date, *self._extra_args())
        return (up.price(*args, **kwargs) - down.price(*args, **kwargs)) / (2 * epsilon)
    
    @staticmethod
    def calculate_T(expiry_date, valuation_date=datetime(2025, 5, 16)):
        """
        Calculates time to maturity (T) from expiry date.
        Uses default valuation date unless overridden.
        """
        return (expiry_date - valuation_date).days / 365
    
    def _setup_payoff_plot(self, title="Payoff Profile", xlabel="Stock Price at Expiry", ylabel="Payoff"):
        """
        Sets up a standard payoff plot layout used across different option types.
        
        This helper method centralizes consistent visual styling for all option payoff plots,
        including:
        - Figure size and resolution
        - Title and axis labels
        - Grid formatting

        Subclasses can call this before plotting their specific payoff logic,
        ensuring visual consistency while avoiding duplicated formatting code.
        """
        plt.figure(figsize=(10, 6), dpi=120)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle=':')
    