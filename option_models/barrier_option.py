# option_models/barrier_option.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from option_models.option import Option          # <--  S0, K, T, q, r, sigma
from utilities.monte_carlo import simulate_paths

class BarrierOption(Option):
    """
    European up-and-in barrier *call* (easy to flip to put if desired).
    Valued by Monte-Carlo; includes Delta & Vega via finite-difference;
    basic path & payoff visualisers.
    """

    DEFAULT_PATHS = 500_000
    DEFAULT_STEPS = 252

    # ---------- constructor -------------------------------------------------
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        B: float,
        q: float = 0.0,
        valuation_date=None
    ):
        """
        S0  : spot price
        K   : strike
        T   : maturity (yrs)
        r   : risk-free CC rate
        sigma: vol (annualised)
        B   : barrier level (up-and-in)
        q   : continuous dividend yield
        """
        super().__init__(S0, K, T, q, r, sigma, valuation_date=valuation_date)
        self.B = B                     # barrier level

    # ---------- pricing -----------------------------------------------------
    def price(self,
              M: int = DEFAULT_PATHS,
              N: int = DEFAULT_STEPS,
              seed: int = None,
              show_diagnostics: bool = True) -> float:
        """
        Monte-Carlo estimator for an up-and-in barrier call.
        """
        # simulate GBM with dividend yield q (drift = (r-q))
        S = simulate_paths(
            self.S0, self.T, self.r, self.sigma,
            q=self.q, M=M, N=N, seed=seed
        )

        breached = np.any(S >= self.B, axis=0)            # path hits barrier?
        payoffs  = np.where(breached,
                            np.maximum(S[-1] - self.K, 0.0),
                            0.0)

        if show_diagnostics:
            print(f"🔍 P(barrier breached): {breached.mean():.4%}")
            print(f"🔍 P(in-the-money @T) : {((S[-1] > self.K) & breached).mean():.4%}")

        return np.exp(-self.r * self.T) * payoffs.mean()

    # ---------- Greeks (FD) --------------------------------------------------
    def delta(self, eps: float = 0.50, **mc_kwargs) -> float:
        up   = self.__class__(self.S0 + eps, self.K, self.T,
                              self.r, self.sigma, self.B, self.q)
        down = self.__class__(self.S0 - eps, self.K, self.T,
                              self.r, self.sigma, self.B, self.q)

        return (up.price(show_diagnostics=False, **mc_kwargs)
                - down.price(show_diagnostics=False, **mc_kwargs)) / (2 * eps)

    def vega(self, eps: float = 0.01, **mc_kwargs) -> float:
        up   = self.__class__(self.S0, self.K, self.T,
                              self.r, self.sigma + eps, self.B, self.q)
        down = self.__class__(self.S0, self.K, self.T,
                              self.r, self.sigma - eps, self.B, self.q)

        return (up.price(show_diagnostics=False, **mc_kwargs)
                - down.price(show_diagnostics=False, **mc_kwargs)) / (2 * eps)

    def visualize_paths(self, M=200, N=252, seed=None):
        """
        Visualize sample simulated paths and color-code their outcomes.

        Green = breached & in-the-money  
        Red = breached & out-of-the-money  
        Grey = never breached
        """
        S = simulate_paths(self.S0, self.T, self.r, self.sigma, M=M, N=N, seed=seed)
        t = np.linspace(0, self.T, N + 1)

        # Determine barrier breach and payoff condition
        breached = np.any(S >= self.B, axis=0)
        in_money = breached & (S[-1] > self.K)

        plt.figure(figsize=(12, 6), dpi=120)
        for j in range(M):
            color = 'green' if in_money[j] else ('red' if breached[j] else 'grey')
            plt.plot(t, S[:, j], color=color, alpha=0.4)

        plt.axhline(self.B, color='blue', linestyle='--', linewidth=1.5, label='Barrier (B)')
        legend = [
            Line2D([0], [0], color='green', label='Breached & Payoff'),
            Line2D([0], [0], color='red', label='Breached No Payoff'),
            Line2D([0], [0], color='grey', label='Not Breached'),
            Line2D([0], [0], color='blue', linestyle='--', label='Barrier Level (B)')
        ]
        plt.legend(handles=legend, loc='upper left')
        plt.title(f"Simulated Paths — Up-and-In Barrier Call (B={self.B})")
        plt.xlabel("Time (Years)")
        plt.ylabel("Underlying Price")
        plt.grid(True)
        plt.show()

    def plot_payoff_region(self, resolution=300):
        """
        Plots the expiry payoff profile of a European up-and-in barrier call option.

        This method:
        - Calculates theoretical payoffs at expiry across a range of final prices
        - Uses the base class's _setup_payoff_plot() to apply consistent plot styling
        - Highlights the barrier and strike levels

        The result shows how the option only activates (pays off) if the barrier is breached
        and the final price ends above the strike.
        """
        S_range = np.linspace(0.5 * self.B, 1.5 * self.B, resolution)
        S_range = np.sort(np.unique(np.append(S_range, [self.B - 1e-6, self.B])))

        payoff = np.where(S_range < self.B, 0, np.maximum(S_range - self.K, 0))

        self._setup_payoff_plot(title="Payoff Profile — Up-and-In Barrier Call")
        plt.plot(S_range, payoff, label="Barrier Call Payoff", linewidth=2)
        plt.axvline(self.K, color='red', linestyle='--', linewidth=1.2, label='Strike (K)')
        plt.axvline(self.B, color='blue', linestyle='--', linewidth=1.2, label='Barrier (B)')
        plt.legend()
        plt.tight_layout()
        plt.show()
