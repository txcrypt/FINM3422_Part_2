# option_models/americanput.py
import math
from option_models.option import Option

class AmericanOption(Option):
    """
    American option (put or call) priced with a CRR binomial tree.
    Now includes Greeks and delta-hedge utilities.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        q: float,
        T: float,
        steps: int = 200,
        opt_type: str = "put",
    ):
        super().__init__(S0, K, T, q, r, sigma)
        self.N = steps
        self.opt_type = opt_type.lower()  # 'put' or 'call'

    # ---------- Core Pricing -------------------------------------------------
    def price(self) -> float:
        """CRR binomial price with early-exercise logic (American)."""
        N = self.N
        dt = self.T / N
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1.0 / u
        pu = (math.exp((self.r - self.q) * dt) - d) / (u - d)
        pd = 1.0 - pu
        disc = math.exp(-self.r * dt)

        # terminal stock prices
        ST = [self.S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

        # terminal payoffs
        if self.opt_type == "put":
            V = [max(self.K - s, 0.0) for s in ST]
        else:
            V = [max(s - self.K, 0.0) for s in ST]

        # backward induction with early exercise
        for i in range(N - 1, -1, -1):
            ST = [s / d for s in ST[: i + 1]]     # roll back one layer
            V_next = []
            for j in range(i + 1):
                cont = disc * (pu * V[j + 1] + pd * V[j])
                exercise = (
                    max(self.K - ST[j], 0.0)
                    if self.opt_type == "put"
                    else max(ST[j] - self.K, 0.0)
                )
                V_next.append(max(cont, exercise))
            V = V_next
        return float(V[0])

    # ---------- Greeks -------------------------------------------------------
    def greeks(self, h: float = 1.0, vol_bump: float = 0.01) -> dict[str, float]:
        """
        Central finite-difference Greeks (Delta, Gamma, Vega).
        h: spot bump in currency units
        vol_bump: absolute σ bump (e.g. 0.01 = +/-1 vol pt)
        """
        base = self.price()

        # Delta
        up   = AmericanOption(self.S0 + h, self.K, self.r, self.sigma,
                              self.q, self.T, self.N, self.opt_type).price()
        down = AmericanOption(self.S0 - h, self.K, self.r, self.sigma,
                              self.q, self.T, self.N, self.opt_type).price()
        delta = (up - down) / (2 * h)

        # Gamma
        gamma = (up - 2 * base + down) / (h * h)

        # Vega
        upv   = AmericanOption(self.S0, self.K, self.r, self.sigma + vol_bump,
                               self.q, self.T, self.N, self.opt_type).price()
        downv = AmericanOption(self.S0, self.K, self.r, self.sigma - vol_bump,
                               self.q, self.T, self.N, self.opt_type).price()
        vega  = (upv - downv) / (2 * vol_bump)

        return {"Delta": delta, "Gamma": gamma, "Vega": vega}

    # ---------- Hedge --------------------------------------------------------
    def hedge(self, delta_bump: float = 1e-4) -> dict[str, float]:
        """
        Compute a delta-neutral hedge:
          * Long 1 option
          * Short Δ shares
          * Cash position to finance hedge
        Returns a dict with position sizes.
        """
        # Delta via small bump
        up_price   = AmericanOption(self.S0 + delta_bump, self.K, self.r,
                                    self.sigma, self.q, self.T, self.N,
                                    self.opt_type).price()
        down_price = AmericanOption(self.S0 - delta_bump, self.K, self.r,
                                    self.sigma, self.q, self.T, self.N,
                                    self.opt_type).price()
        delta = (up_price - down_price) / (2 * delta_bump)

        # Cash to offset value: (Δ * S0) − option price
        cash = delta * self.S0 - self.price()

        return {
            "Long Put" if self.opt_type == "put" else "Long Call": 1.0,
            "Short Stock": -delta,
            "Cash Position": cash
        }
