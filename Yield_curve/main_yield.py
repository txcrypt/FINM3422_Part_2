# --- Step 1: Create Instruments ---
from .instrument_classes import Bank_bill, Bond, Portfolio
from .curve_classes import YieldCurve
import matplotlib.pyplot as plt
import numpy as np
import math


def build_yield_curve():
    # Bank Bills
    bill_1m = Bank_bill(maturity=1/12)  # ≈ 1 month
    bill_1m.set_ytm(0.037925)  # from CAPIQ
    bill_1m.set_cash_flows()

    bill_3m = Bank_bill(maturity=0.25)  # ≈ 3 months
    bill_3m.set_ytm(0.037972)  # from CAPIQ
    bill_3m.set_cash_flows()

    bill_6m = Bank_bill(maturity=0.5)  # ≈ 6 months
    bill_6m.set_ytm(0.03855)  # from CAPIQ
    bill_6m.set_cash_flows()

    # Bonds
    bond_2y = Bond(maturity=2, coupon=0.06, frequency=2)
    bond_2y.set_ytm(0.03567)  # from CAPIQ
    bond_2y.set_cash_flows()

    bond_3y = Bond(maturity=3, coupon=0.062, frequency=2)
    bond_3y.set_ytm(0.03602)  # from CAPIQ
    bond_3y.set_cash_flows()

    bond_5y = Bond(maturity=5, coupon=0.065, frequency=2)
    bond_5y.set_ytm(0.03802)  # from CAPIQ
    bond_5y.set_cash_flows()


    # --- Step 2: Build Portfolio with All Instruments ---
    portfolio = Portfolio()
    # Add bank bills
    portfolio.add_bank_bill(bill_1m)
    portfolio.add_bank_bill(bill_3m)
    portfolio.add_bank_bill(bill_6m)
    # Add bonds
    portfolio.add_bond(bond_2y)
    portfolio.add_bond(bond_3y)
    portfolio.add_bond(bond_5y)
    # Aggregate cash flows
    portfolio.set_cash_flows()

    # --- Step 3: Bootstrap Yield Curve Once ---
    yield_curve = YieldCurve()
    yield_curve.set_constituent_portfolio(portfolio)
    yield_curve.bootstrap()

    return yield_curve