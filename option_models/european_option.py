import numpy as np
from scipy.stats import norm

class EuropeanOption:
    """
    Represents a European option (call or put) and calculates its price
    using the Black-Scholes-Merton model.
    """

    def __init__(self, S0: float, K: float, T: float, r: float, 
                 sigma: float, option_type: str, q: float = 0.0):
        """
        Initializes the EuropeanOption.

        Args:
            S0 (float): Current price of the underlying asset.
            K (float): Strike price of the option.
            T (float): Time to maturity (in years).
            r (float): Annualized continuously compounded risk-free interest rate.
            sigma (float): Annualized volatility of the underlying asset's returns.
            option_type (str): Type of the option, 'call' or 'put'.
            q (float, optional): Annualized continuous dividend yield. Defaults to 0.0.
        """
        if not all(isinstance(arg, (int, float)) for arg in [S0, K, T, r, sigma, q]):
            raise ValueError("S0, K, T, r, sigma, and q must be numeric.")
        if S0 <= 0:
            raise ValueError("Underlying price (S0) must be positive.")
        if K < 0: # Strike can be zero in some theoretical cases, but typically positive
            raise ValueError("Strike price (K) must be non-negative.")
        if T < 0: # Time to maturity cannot be negative
            raise ValueError("Time to maturity (T) must be non-negative.")
        if sigma < 0: # Volatility cannot be negative
            raise ValueError("Volatility (sigma) must be non-negative.")
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'.")

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q # Continuous dividend yield
        self.option_type = option_type.lower()

        # Pre-calculate d1 and d2 if T > 0 and sigma > 0 for efficiency
        # These are fundamental components of the Black-Scholes formula.
        if self.T > 0 and self.sigma > 0:
            self._d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
                        (self.sigma * np.sqrt(self.T))
            self._d2 = self._d1 - self.sigma * np.sqrt(self.T)
        elif self.T == 0: # Option is at expiry
            # Set d1/d2 to +/- infinity based on moneyness to ensure correct cdf values (0 or 1)
            if self.S0 > self.K:
                self._d1, self._d2 = np.inf, np.inf
            elif self.S0 < self.K:
                self._d1, self._d2 = -np.inf, -np.inf
            else: # S0 == K
                self._d1, self._d2 = 0, 0 # cdf(0) = 0.5, though price is intrinsic value
        else: # T > 0 but sigma = 0 (deterministic price path)
            # Price will be max(0, S0*exp((r-q)T) - K)exp(-rT) for call
            # d1/d2 effectively go to +/- infinity based on comparison of S0*exp(-qT) vs K*exp(-rT)
            # For simplicity in d1/d2, can set to handle in price logic, or:
            future_S = self.S0 * np.exp((self.r - self.q) * self.T) # Effective stock price at T under risk-neutral measure if sigma=0
            if future_S > self.K:
                 self._d1, self._d2 = np.inf, np.inf
            else:
                 self._d1, self._d2 = -np.inf, -np.inf


    def delta(self) -> float:
        """
        Calculates the Delta of the European option.
        Delta for call: exp(-qT) * N(d1)
        Delta for put: exp(-qT) * (N(d1) - 1)
        """
        if self.T == 0: # At expiry
            if self.option_type == 'call':
                return 1.0 if self.S0 > self.K else (0.5 if self.S0 == self.K else 0.0)
        
        if self.sigma == 0: # Deterministic case
            if self.option_type == 'call':
                # If S0*exp((r-q)T) > K, option is ITM, delta is effectively exp(-qT) * discount factor for S0
                # This simplifies to exp(-qT) as S0's change directly impacts option value if ITM
                return np.exp(-self.q * self.T) if self.S0 * np.exp((self.r - self.q) * self.T) > self.K else 0.0

        # Standard Black-Scholes delta
        if self.option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(self._d1)

    def gamma(self) -> float:
        """
        Calculates the Gamma of the European option.
        Gamma = exp(-qT) * N'(d1) / (S0 * sigma * sqrt(T))
        Same for call and put.
        """
        if self.T == 0 or self.sigma == 0 or self.S0 == 0:
            # Gamma is undefined or effectively zero at expiry or if no volatility/spot price
            return 0.0
        
        n_prime_d1 = norm.pdf(self._d1) # N'(d1) is the PDF of standard normal at d1
        gamma_val = np.exp(-self.q * self.T) * n_prime_d1 / \
                    (self.S0 * self.sigma * np.sqrt(self.T))
        return gamma_val

    def vega(self) -> float:
        """
        Calculates the Vega of the European option (points per 1% change in volatility).
        Vega = S0 * exp(-qT) * N'(d1) * sqrt(T)
        Same for call and put.
        """
        if self.T == 0 or self.sigma == 0 or self.S0 == 0:
            # Vega is zero at expiry or if no volatility/spot price
            return 0.0
            
        n_prime_d1 = norm.pdf(self._d1)
        vega_val = self.S0 * np.exp(-self.q * self.T) * n_prime_d1 * np.sqrt(self.T)
        return vega_val / 100 # Per 1% change in sigma (standard convention)

    def theta(self) -> float:
        """
        Calculates the Theta of the European option (points per calendar day).
        Theta_call = -S0*exp(-qT)*N'(d1)*sigma/(2*sqrt(T)) - r*K*exp(-rT)*N(d2) + q*S0*exp(-qT)*N(d1)
        Theta_put  = -S0*exp(-qT)*N'(d1)*sigma/(2*sqrt(T)) + r*K*exp(-rT)*N(-d2) - q*S0*exp(-qT)*N(-d1)
        """
        if self.T == 0:
            return 0.0 # No time decay at expiry

        if self.sigma == 0: # Deterministic case
            # Theta reflects change in PV of deterministic payoff due to time passing
            # For a call: d/dT [ (S0*exp((r-q)T) - K)*exp(-rT) ]
            # For simplicity, often considered near zero or handled by intrinsic value change.
            # A more precise calculation for sigma=0:
            if self.option_type == 'call':
                # Change in value if option is ITM based on forward
                if self.S0 * np.exp((self.r - self.q) * self.T) > self.K:
                    # d/dT ( S0*exp(-qT) - K*exp(-rT) )
                    theta_val_yr = -self.q * self.S0 * np.exp(-self.q * self.T) + self.r * self.K * np.exp(-self.r * self.T)
                else: # OTM, value is 0, theta is 0
                    theta_val_yr = 0.0
            return theta_val_yr / 365.25


        n_prime_d1 = norm.pdf(self._d1)
        
        common_term = -self.S0 * np.exp(-self.q * self.T) * n_prime_d1 * self.sigma / \
                      (2 * np.sqrt(self.T))

        if self.option_type == 'call':
            n_d1 = norm.cdf(self._d1)
            n_d2 = norm.cdf(self._d2)
            theta_val_yr = common_term - self.r * self.K * np.exp(-self.r * self.T) * n_d2 + \
                           self.q * self.S0 * np.exp(-self.q * self.T) * n_d1
                           
        return theta_val_yr / 365.25 # Per calendar day

    def rho(self) -> float:
        """
        Calculates the Rho of the European option (points per 1% change in risk-free rate r).
        Rho_call = K*T*exp(-rT)*N(d2)
        Rho_put  = -K*T*exp(-rT)*N(-d2)
        (Assuming q, the dividend yield, is not directly a function of r for this partial derivative)
        """
        if self.T == 0:
            return 0.0 # No interest rate sensitivity at expiry

        if self.option_type == 'call':
            n_d2 = norm.cdf(self._d2)
            rho_val = self.K * self.T * np.exp(-self.r * self.T) * n_d2
            
        return rho_val / 100 # Per 1% change in r


    def price(self) -> float:
        """
        Calculates the price of the European option using the Black-Scholes-Merton formula.
        Handles cases for T=0 (at expiry) and sigma=0 (no volatility).
        """
        # At expiry, price is intrinsic value
        if self.T == 0:
            if self.option_type == 'call':
                return max(0.0, self.S0 - self.K)

        # If volatility is zero, option value is deterministic
        if self.sigma == 0:
            # Payoff at maturity under risk-neutral measure (S0 grows at r-q)
            # Discounted back to present
            if self.option_type == 'call':
                price_at_expiry = max(0.0, self.S0 * np.exp((self.r - self.q) * self.T) - self.K)
                return price_at_expiry * np.exp(-self.r * self.T)


        # Standard Black-Scholes calculation
        n_d1 = norm.cdf(self._d1)
        n_d2 = norm.cdf(self._d2)

        if self.option_type == 'call':
            option_price = self.S0 * np.exp(-self.q * self.T) * n_d1 - \
                           self.K * np.exp(-self.r * self.T) * n_d2
        
        return option_price
    
