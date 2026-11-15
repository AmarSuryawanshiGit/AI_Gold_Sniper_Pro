
import MetaTrader5 as mt5
from datetime import datetime

if not mt5.initialize():
    print("MT5 connection failed:", mt5.last_error())
    raise SystemExit

symbol = "GOLD"

if not mt5.symbol_select(symbol, True):
    print("Symbol selection failed:", mt5.last_error())
    mt5.shutdown()
    raise SystemExit

timeframes = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1
}

for name, timeframe in timeframes.items():
    print(f"\n--- {name} DATA ---")

    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        0,
        10
    )

    if rates is None or len(rates) == 0:
        print("No candle data:", mt5.last_error())
        continue

    print("Candles received:", len(rates))

    for candle in rates[-3:]:
        print({
            "time": datetime.fromtimestamp(
                candle["time"]
            ),
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "tick_volume": candle["tick_volume"],
            "real_volume": candle["real_volume"]
        })

mt5.shutdown()
