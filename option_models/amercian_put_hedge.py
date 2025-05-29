# option_models/american_put_hedge.py

from .americanput import AmericanOption

def hedge_american_put(option: AmericanOption, h: float = 1e-4) -> dict:
    """
    Compute a delta‐hedge for an American put.

    Returns a dictionary with:
      * 'put':        +1.0  # long one put
      * 'stock':     -Delta # short Δ shares
      * 'cash':       amount invested at r to finance the hedge

    This provides the runner with exact quantities to trade.
    """
    # 1) Compute Delta via tiny bump
    base_price = option.price()
    up_price   = AmericanOption(option.S0 + h, option.K, option.r,
                                option.sigma, option.q, option.T,
                                option.N, option.opt_type).price()
    down_price = AmericanOption(option.S0 - h, option.K, option.r,
                                option.sigma, option.q, option.T,
                                option.N, option.opt_type).price()
    delta = (up_price - down_price) / (2 * h)

    # 2) The risk‐free bond position = (Δ * S0) − P
    #    so that combined stock + bond + option is locally flat
    cash = delta * option.S0 - base_price

    return {
        "Long Put":     1.0,
        "Short Stock": -delta,
        "Cash Position": cash
    }
