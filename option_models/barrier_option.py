import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from option_models.option import Option
from utilities.monte_carlo import simulate_paths

class BarrierOption(Option):
    """
    Represents a European up-and-in barrier call option, valued using Monte Carlo.
    Inherits common behavior from the Option base class.
    """

    DEFAULT_PATHS = 500_000
    DEFAULT_STEPS = 252

    def __init__(self, S0, K, T, r, sigma, B, valuation_date=None):
        """
        Initialize the barrier option with standard option inputs and barrier level.

        Parameters:
        - S0: Spot price of the underlying
        - K: Strike price
        - T: Time to maturity (years)
        - r: Risk-free interest rate
        - sigma: Volatility (annualized)
        - B: Barrier level
        - valuation_date: Valuation date (default: 16 May 2025 if None)
        """
        super().__init__(S0, K, T, r, sigma, valuation_date=valuation_date)
        self.B = B  # Barrier level

    def _extra_args(self):
        """
        Pass barrier level to superclass for delta/vega finite difference calls.
        """
        return [self.B]

    def price(self, M=DEFAULT_PATHS, N=DEFAULT_STEPS, seed=None, show_diagnostics=True):
        """
        Estimate the option price using Monte Carlo simulation.

        Parameters:
        - M: Number of Monte Carlo paths
        - N: Number of time steps per path
        - seed: Optional random seed for reproducibility
        - show_diagnostics: Whether to print probability stats (default True)

        Returns:
        - Discounted expected payoff of the barrier option
        """
        # Simulate price paths under GBM
        S = simulate_paths(self.S0, self.T, self.r, self.sigma, M=M, N=N, seed=seed)

        # Check if each path ever crosses the barrier
        breached = np.any(S >= self.B, axis=0)

        # Compute payoff only if barrier was breached
        payoffs = np.where(breached, np.maximum(S[-1] - self.K, 0), 0)

        if show_diagnostics:
            prob_breached = np.mean(breached)
            prob_not_breached = 1 - prob_breached
            prob_in_the_money = np.mean((S[-1] > self.K) & (np.max(S, axis=0) >= self.B))
            prob_out_of_the_money = 1 - prob_in_the_money

            print(f"🔍 P(Barrier Breached):         {prob_breached:.4%}")
            print(f"🔍 P(Barrier Not Breached):     {prob_not_breached:.4%}")
            print(f"🔍 P(In-the-money at expiry):   {prob_in_the_money:.4%}")
            print(f"🔍 P(Out-of-the-money at expiry): {prob_out_of_the_money:.4%}")

        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def delta(self, epsilon=0.5, M=DEFAULT_PATHS, N=DEFAULT_STEPS, seed=None):
        """
        Monte Carlo-based Delta for barrier options using central finite difference.
        Includes control over number of paths and random seed for reproducibility.
        """
        up = self.__class__(self.S0 + epsilon, self.K, self.T, self.r, self.sigma, self.B)
        down = self.__class__(self.S0 - epsilon, self.K, self.T, self.r, self.sigma, self.B)
        return (up.price(M=M, N=N, seed=seed, show_diagnostics=False) - down.price(M=M, N=N, seed=seed,show_diagnostics=False)) / (2 * epsilon)

    def vega(self, epsilon=0.01, M=DEFAULT_PATHS, N=DEFAULT_STEPS, seed=None):
        """
        Monte Carlo-based Vega for barrier options using central finite difference.
        Measures sensitivity to volatility changes with consistent simulation parameters.
        """
        up = self.__class__(self.S0, self.K, self.T, self.r, self.sigma + epsilon, self.B)
        down = self.__class__(self.S0, self.K, self.T, self.r, self.sigma - epsilon, self.B)
        return (up.price(M=M, N=N, seed=seed, show_diagnostics=False) - down.price(M=M, N=N, seed=seed,show_diagnostics=False)) / (2 * epsilon)


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
