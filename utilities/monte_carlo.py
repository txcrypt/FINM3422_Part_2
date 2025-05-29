import numpy as np

def simulate_paths(S0, T, r, sigma, M=10000, N=252, seed=None):
    """
    Simulate geometric Brownian motion price paths.

    Parameters:
    - S0: Initial stock price
    - T: Time to maturity (years)
    - r: Risk-free rate
    - sigma: Volatility (annualized)
    - M: Number of Monte Carlo paths
    - N: Number of time steps
    - seed: Random seed (optional)

    Returns:
    - S: Array of shape (N+1, M) representing simulated price paths
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    S = np.zeros((N + 1, M))
    S[0] = S0

    for t in range(1, N + 1):
        Z = np.random.normal(0, 1, M)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return S