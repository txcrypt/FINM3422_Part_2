


class data_fetcher:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.implplied_volatility = {}
        self.fetch_data()

    def fetch_data(self):
        """Get data from yfiannce for stocks for from a given start date to a given end date and save it as a dataframe"""

    def get_returns(self,ticker):
        """" Get teh stock returns for the given ticker by averaging the daily returns over the period of interest"""

    def get_stock_data(self, ticker)->:
        """Get the stock data for the given ticker and return it as a dataframe"""

    def get_covariance_matrix(self,tickers):
        """Get the covariance matrix for the stock data of all tickers"""