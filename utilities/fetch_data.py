import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# your tickers
tickers = [
    "BHP.AX", "CBA.AX", "WES.AX",
    "CSL.AX", "WDS.AX", "MQG.AX"
]

# define date range: today back 365 days
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# download only the 'Close' prices
df_close = yf.download(
    tickers,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    progress=False,
    auto_adjust=False
)["Close"]

# (Optional) If you get a MultiIndex (when >1 ticker), df_close will have columns per ticker.
# If only one ticker, it'll be a Series—so make it a DataFrame:
if isinstance(df_close, pd.Series):
    df_close = df_close.to_frame(tickers[0])

# save to CSV
output_path = "closing_prices_last_year.csv"
df_close.to_csv(output_path)

print(f"Saved closing prices to {output_path}")
