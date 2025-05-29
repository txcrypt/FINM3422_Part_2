# option_models/american_put_greeks.py

from option_models.americanput import AmericanOption

def compute_american_put_greeks(
    option: AmericanOption,
    h: float = 1.0,
    vol_bump: float = 0.01
 ) -> dict:
    """
    Delta, Gamma, Vega for an American PUT via central finite differences.
    - Delta: first‐derivative wrt S0; vital for primary hedge.
    - Gamma: convexity; critical near early‐exercise boundaries.
    - Vega: volatility sensitivity; American puts gain more from vol rises.
    """
    S0, K, r, sigma, q, T, N = (
        option.S0, option.K, option.r, option.sigma, option.q, option.T, option.N
    )
    typ = option.optype = "put"

    base = option.price()

    # Delta
    up   = AmericanOption(S0 + h, K, r, sigma, q, T, N, typ).price()
    down = AmericanOption(S0 - h, K, r, sigma, q, T, N, typ).price()
    delta = (up - down) / (2*h)

    # Gamma
    gamma = (up - 2*base + down) / (h*h)

    # Vega
    upv   = AmericanOption(S0, K, r, sigma + vol_bump, q, T, N, typ).price()
    downv = AmericanOption(S0, K, r, sigma - vol_bump, q, T, N, typ).price()
    vega  = (upv - downv) / (2*vol_bump)

    return {"Delta": delta, "Gamma": gamma, "Vega": vega}
