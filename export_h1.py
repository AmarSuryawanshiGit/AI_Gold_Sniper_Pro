
import MetaTrader5 as mt5
import pandas as pd

SYMBOL_CANDIDATES = [
    "GOLD", "XAUUSD", "GOLD.",
    "XAUUSD.", "GOLDm", "XAUUSDm"
]

print("Connecting to XM MT5...")

if not mt5.initialize():
    print("MT5 connection failed:", mt5.last_error())
    raise SystemExit

symbol = None

for candidate in SYMBOL_CANDIDATES:
    if mt5.symbol_select(candidate, True):
        symbol = candidate
        break

if symbol is None:
    print("Gold symbol not found.")
    mt5.shutdown()
    raise SystemExit

print("Selected symbol:", symbol)

# Fetch 10,000 H1 candles, or all currently available
rates = mt5.copy_rates_from_pos(
    symbol,
    mt5.TIMEFRAME_H1,
    0,
    10000
)

if rates is None or len(rates) == 0:
    print("No H1 data:", mt5.last_error())
    mt5.shutdown()
    raise SystemExit

df = pd.DataFrame(rates)

df["time"] = pd.to_datetime(
    df["time"], unit="s", utc=True
)

df = df.sort_values("time")
df = df.drop_duplicates(subset=["time"])
df = df.reset_index(drop=True)

filename = "GOLD_H1_history.csv"
df.to_csv(filename, index=False)

print("\nExport successful!")
print("Symbol:", symbol)
print("Total candles:", len(df))
print("First candle:", df["time"].iloc[0])
print("Last candle:", df["time"].iloc[-1])
print("Saved file:", filename)

mt5.shutdown()
