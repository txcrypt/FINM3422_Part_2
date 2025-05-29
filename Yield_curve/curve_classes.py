
import numpy as np
import math

class ZeroCurve:
    def __init__(self):
        # set up empty list
        self.maturities = []
        self.zero_rates = []
        self.AtMats = []
        self.discount_factors = []
    
    def add_zero_rate(self, maturity, zero_rate):
        self.maturities.append(maturity)
        self.zero_rates.append(zero_rate)
        self.AtMats.append(math.exp(zero_rate*maturity))
        self.discount_factors.append(1/self.AtMats[-1])

    def add_discount_factor(self, maturity, discount_factor):
        self.maturities.append(maturity)
        self.discount_factors.append(discount_factor)
        self.AtMats.append(1/discount_factor)
        self.zero_rates.append(math.log(1/discount_factor)/maturity)
    
    def get_AtMat(self, maturity):
        if maturity in self.maturities:
            return self.AtMats[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.AtMats, maturity)

    def get_discount_factor(self, maturity):
        if maturity in self.maturities:
            return self.discount_factors[self.maturities.index(maturity)]
    # debug print:
        print(f"[ZeroCurve] interpolating DF at t={maturity:.4f} from points {self.maturities}")
        return exp_interp(self.maturities, self.discount_factors, maturity)


    def get_zero_rate(self, maturity):
        if maturity in self.maturities:
            return self.zero_rates[self.maturities.index(maturity)]
        else:
            return math.log(self.get_AtMat(maturity))/maturity
        
    def get_zero_curve(self):
        return self.maturities, self.discount_factors
    
    def npv(self, cash_flows):
        npv = 0
        for maturity in cash_flows.get_maturities():
            npv += cash_flows.get_cash_flow(maturity)*self.get_discount_factor(maturity)
        return npv
        
def exp_interp(xs, ys, x):
    """
    Exponential interpolation with safe bounds for extrapolation.
    """
    xs = np.array(xs)
    ys = np.array(ys)

    # If x is before the first known point, return the first y
    if x <= xs[0]:
        return ys[0]
    # If x is after the last known point, return the last y
    if x >= xs[-1]:
        return ys[-1]

    # Otherwise, find the bracketing interval
    idx = np.searchsorted(xs, x)
    x0, x1 = xs[idx-1], xs[idx]
    y0, y1 = ys[idx-1], ys[idx]

    rate = (np.log(y1) - np.log(y0)) / (x1 - x0)
    return y0 * np.exp(rate * (x - x0))


class YieldCurve(ZeroCurve):
    def __init__(self):
        super().__init__()
        self.portfolio = []

    # set the constituent portfolio
    # the portfolio must contain bills and bonds in order of maturity
    # where all each successive bond only introduces one new cashflow beyond 
    #       the longest maturity to that point (being the maturity cashflow)
    def set_constituent_portfolio(self, portfolio):
        self.portfolio = portfolio

    def bootstrap(self):
        bank_bills = self.portfolio.get_bank_bills()
        bonds = self.portfolio.get_bonds()
        self.add_zero_rate(0,0)
        for bank_bill in bank_bills:
            self.add_discount_factor(bank_bill.get_maturity(),bank_bill.get_price()/bank_bill.get_face_value())
        for bond in bonds:
            # calculate the PV of the bond cashflows excluding the maturity cashflow
            pv = 0
            bond_dates = bond.get_maturities()
            bond_amounts = bond.get_amounts()
            for i in range(1, len(bond_amounts)-1):
                pv += bond_amounts[i]*self.get_discount_factor(bond_dates[i])
            print("PV of all the cashflows except maturity is: ", pv)
            print("The bond price is: ", bond.get_price())
            print("The last cashflow is: ", bond_amounts[-1])
            self.add_discount_factor(bond.get_maturity(),(bond.get_price()-pv)/bond.get_amounts()[-1])