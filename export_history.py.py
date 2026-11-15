
import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_M1
BARS = 100000

if not mt5.initialize():
    print("MT5 connection failed:", mt5.last_error())
    raise SystemExit

if not mt5.symbol_select(SYMBOL, True):
    print("Symbol not found:", SYMBOL)
    mt5.shutdown()
    raise SystemExit

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

if rates is None or len(rates) == 0:
    print("No historical data:", mt5.last_error())
else:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time", "open", "high", "low", "close",
             "tick_volume", "spread", "real_volume"]]
    df.to_csv("GOLD_M1.csv", index=False)
    print("Export successful:", len(df), "candles")
    print("Saved: GOLD_M1.csv")

mt5.shutdown()
