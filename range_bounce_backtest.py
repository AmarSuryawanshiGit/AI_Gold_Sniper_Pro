
import pandas as pd
import numpy as np

# ==========================================================
# H1 RANGE BOUNCE BACKTEST - RESEARCH VERSION
# Broker model: XM GOLD, 0.01 lot, contract size 100
# No live orders. No V2 signals.
# ==========================================================

M5_FILE = "M5.csv"
H1_FILE = "H1.csv"

LOT_SIZE = 0.01
CONTRACT_SIZE = 100
POINT_SIZE = 0.01

# 0.01 lot x 100 contract = $1 per $1 GOLD movement
USD_PER_PRICE_MOVE = LOT_SIZE * CONTRACT_SIZE

H1_LOOKBACK = 8
H1_ATR_PERIOD = 14
MAX_RANGE_ATR = 4.0

MIN_TOP_TOUCHES = 2
MIN_BOTTOM_TOUCHES = 2
TOUCH_TOLERANCE_ATR = 0.25

M15_SWING_LOOKBACK = 3
MAX_HOLD_M15_BARS = 48

# Historical spread assumption:
# CSV spread is assumed to be in MT5 points.
# Verify this convention against your XM export.
USE_SPREAD = True

# Commission and slippage are NOT verified.
COMMISSION_PER_TRADE_USD = 0.0
SLIPPAGE_PRICE = 0.0

# Test both target styles independently.
TARGET_MODES = ["MIDPOINT", "OPPOSITE_BOUNDARY"]


def load_data(path):
    df = pd.read_csv(path)

    df["time"] = pd.to_datetime(
        df["time"], utc=True, errors="coerce"
    )

    for col in [
        "open", "high", "low", "close",
        "tick_volume", "spread"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0

    if "spread" not in df.columns:
        df["spread"] = 0

    return df.sort_values("time").reset_index(drop=True)


def make_m15(m5):
    x = m5.set_index("time")

    m15 = x.resample(
        "15min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "spread": "mean"
    }).dropna(subset=["open", "high", "low", "close"])

    return m15.reset_index()


def atr(df, period=14):
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def build_h1_ranges(h1):
    h1 = h1.copy()
    h1["atr"] = atr(h1, H1_ATR_PERIOD)

    h1["support"] = np.nan
    h1["resistance"] = np.nan
    h1["width_atr"] = np.nan
    h1["top_touches"] = 0
    h1["bottom_touches"] = 0
    h1["range_valid"] = False

    # At row i, use only the preceding completed H1 candles.
    for i in range(H1_LOOKBACK, len(h1)):
        window = h1.iloc[i - H1_LOOKBACK:i]

        current_atr = h1.iloc[i]["atr"]

        if pd.isna(current_atr) or current_atr <= 0:
            continue

        resistance = window["high"].max()
        support = window["low"].min()

        tolerance = current_atr * TOUCH_TOLERANCE_ATR

        top_touches = int(
            (window["high"] >= resistance - tolerance).sum()
        )

        bottom_touches = int(
            (window["low"] <= support + tolerance).sum()
        )

        width = resistance - support
        width_atr = width / current_atr

        h1.loc[i, "support"] = support
        h1.loc[i, "resistance"] = resistance
        h1.loc[i, "width_atr"] = width_atr
        h1.loc[i, "top_touches"] = top_touches
        h1.loc[i, "bottom_touches"] = bottom_touches

        h1.loc[i, "range_valid"] = (
            width > 0
            and width_atr <= MAX_RANGE_ATR
            and top_touches >= MIN_TOP_TOUCHES
            and bottom_touches >= MIN_BOTTOM_TOUCHES
        )

    # H1 candle is available only after its close.
    h1["available_time"] = (
        h1["time"] + pd.Timedelta(hours=1)
    )

    return h1


def candle_patterns(df):
    x = df.copy()

    x["body"] = (x["close"] - x["open"]).abs()

    x["lower_wick"] = (
        np.minimum(x["open"], x["close"]) - x["low"]
    )

    x["upper_wick"] = (
        x["high"] - np.maximum(x["open"], x["close"])
    )

    candle_range = (x["high"] - x["low"]).replace(0, np.nan)

    x["close_location"] = (
        (x["close"] - x["low"]) / candle_range
    )

    prev_open = x["open"].shift(1)
    prev_close = x["close"].shift(1)

    bullish_engulfing = (
        (prev_close < prev_open)
        & (x["close"] > x["open"])
        & (x["open"] <= prev_close)
        & (x["close"] >= prev_open)
    )

    bearish_engulfing = (
        (prev_close > prev_open)
        & (x["close"] < x["open"])
        & (x["open"] >= prev_close)
        & (x["close"] <= prev_open)
    )

    bullish_pin = (
        (x["close"] > x["open"])
        & (x["lower_wick"] >= 1.5 * x["body"])
        & (x["close_location"] >= 0.65)
    )

    bearish_pin = (
        (x["close"] < x["open"])
        & (x["upper_wick"] >= 1.5 * x["body"])
        & (x["close_location"] <= 0.35)
    )

    x["bullish_pattern"] = (
        bullish_engulfing | bullish_pin
    )

    x["bearish_pattern"] = (
        bearish_engulfing | bearish_pin
    )

    # Volume is diagnostic in this first version.
    x["volume_avg"] = (
        x["tick_volume"].rolling(20, min_periods=20).mean()
    )

    x["volume_ratio"] = (
        x["tick_volume"] / x["volume_avg"]
    )

    return x


def attach_h1_levels(m15, h1):
    left = m15.sort_values("time").copy()
    right = h1[[
        "available_time",
        "support",
        "resistance",
        "atr",
        "width_atr",
        "top_touches",
        "bottom_touches",
        "range_valid"
    ]].copy()

    right = right.rename(columns={
        "support": "h1_support",
        "resistance": "h1_resistance",
        "atr": "h1_atr",
        "width_atr": "h1_width_atr",
        "top_touches": "h1_top_touches",
        "bottom_touches": "h1_bottom_touches",
        "range_valid": "h1_range_valid"
    })

    return pd.merge_asof(
        left.sort_values("time"),
        right.sort_values("available_time"),
        left_on="time",
        right_on="available_time",
        direction="backward"
    )


def get_spread_price(row):
    if not USE_SPREAD:
        return 0.0

    value = row.get("spread", 0)

    if pd.isna(value):
        return 0.0

    return max(0.0, float(value) * POINT_SIZE)


def find_signals(df):
    signals = []

    for i in range(20, len(df) - 1):
        row = df.iloc[i]

        if not bool(row["h1_range_valid"]):
            continue

        support = row["h1_support"]
        resistance = row["h1_resistance"]
        h1_atr = row["h1_atr"]

        if pd.isna(support) or pd.isna(resistance):
            continue

        if pd.isna(h1_atr) or h1_atr <= 0:
            continue

        zone = h1_atr * TOUCH_TOLERANCE_ATR

        near_support = (
            row["low"] <= support + zone
            and row["close"] >= support
            and row["close"] <= support + zone * 2
        )

        near_resistance = (
            row["high"] >= resistance - zone
            and row["close"] <= resistance
            and row["close"] >= resistance - zone * 2
        )

        side = None

        if near_support and bool(row["bullish_pattern"]):
            side = "BUY"

        elif near_resistance and bool(row["bearish_pattern"]):
            side = "SELL"

        if side is None:
            continue

        # Swing SL: previous 3 completed M15 candles,
        # including the signal candle.
        swing_window = df.iloc[
            max(0, i - M15_SWING_LOOKBACK + 1):i + 1
        ]

        if side == "BUY":
            sl = float(swing_window["low"].min())
            target_mid = (support + resistance) / 2
            target_far = resistance

            # Do not enter if signal candle already invalidated
            # the support-to-midpoint setup.
            if target_mid <= row["close"]:
                continue

        else:
            sl = float(swing_window["high"].max())
            target_mid = (support + resistance) / 2
            target_far = support

            if target_mid >= row["close"]:
                continue

        signals.append({
            "signal_index": i,
            "signal_time": row["time"] + pd.Timedelta(minutes=15),
            "side": side,
            "support": float(support),
            "resistance": float(resistance),
            "range_width": float(resistance - support),
            "h1_width_atr": float(row["h1_width_atr"]),
            "sl": sl,
            "target_mid": float(target_mid),
            "target_far": float(target_far),
            "volume_ratio": row["volume_ratio"]
        })

    return signals


def simulate(signals, df, target_mode):
    trades = []
    next_allowed_index = 0

    for signal in signals:
        entry_index = signal["signal_index"] + 1

        if entry_index < next_allowed_index:
            continue

        if entry_index >= len(df):
            continue

        entry_bar = df.iloc[entry_index]
        side = signal["side"]

        spread = get_spread_price(entry_bar)

        if side == "BUY":
            entry = (
                float(entry_bar["open"])
                + spread
                + SLIPPAGE_PRICE
            )
            sl = signal["sl"]
            tp = (
                signal["target_mid"]
                if target_mode == "MIDPOINT"
                else signal["target_far"]
            )

            if sl >= entry or tp <= entry:
                continue

        else:
            entry = (
                float(entry_bar["open"])
                - SLIPPAGE_PRICE
            )
            sl = signal["sl"]
            tp = (
                signal["target_mid"]
                if target_mode == "MIDPOINT"
                else signal["target_far"]
            )

            if sl <= entry or tp >= entry:
                continue

        exit_price = None
        exit_reason = "TIME"
        exit_index = None

        last_index = min(
            len(df) - 1,
            entry_index + MAX_HOLD_M15_BARS - 1
        )

        for j in range(entry_index, last_index + 1):
            bar = df.iloc[j]
            bar_spread = get_spread_price(bar)

            if side == "BUY":
                sl_hit = float(bar["low"]) <= sl
                tp_hit = float(bar["high"]) >= tp

                # Conservative: stop first if both touched.
                if sl_hit:
                    exit_price = sl
                    exit_reason = "SL"
                    exit_index = j
                    break

                if tp_hit:
                    exit_price = tp
                    exit_reason = "TP"
                    exit_index = j
                    break

            else:
                # Approximate Ask OHLC using Bid + spread.
                ask_high = float(bar["high"]) + bar_spread
                ask_low = float(bar["low"]) + bar_spread

                sl_hit = ask_high >= sl
                tp_hit = ask_low <= tp

                if sl_hit:
                    exit_price = sl
                    exit_reason = "SL"
                    exit_index = j
                    break

                if tp_hit:
                    exit_price = tp
                    exit_reason = "TP"
                    exit_index = j
                    break

        if exit_price is None:
            exit_index = last_index
            last_bar = df.iloc[last_index]

            if side == "BUY":
                exit_price = float(last_bar["close"])
            else:
                exit_price = (
                    float(last_bar["close"])
                    + get_spread_price(last_bar)
                )

        if side == "BUY":
            pnl = (
                exit_price - entry
            ) * USD_PER_PRICE_MOVE

        else:
            pnl = (
                entry - exit_price
            ) * USD_PER_PRICE_MOVE

        pnl -= COMMISSION_PER_TRADE_USD

        exit_time = (
            df.iloc[exit_index]["time"]
            + pd.Timedelta(minutes=15)
        )

        trades.append({
            "signal_time": signal["signal_time"],
            "entry_time": entry_bar["time"],
            "exit_time": exit_time,
            "side": side,
            "target_mode": target_mode,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit": exit_price,
            "exit_reason": exit_reason,
            "net_profit_usd": pnl,
            "support": signal["support"],
            "resistance": signal["resistance"],
            "range_width": signal["range_width"],
            "h1_width_atr": signal["h1_width_atr"],
            "volume_ratio": signal["volume_ratio"],
            "holding_bars": exit_index - entry_index + 1
        })

        # One open position at a time.
        next_allowed_index = exit_index + 1

    return pd.DataFrame(trades)


def report(trades, name):
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    if trades.empty:
        print("No trades generated.")
        return

    pnl = trades["net_profit_usd"]

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0 else float("inf")
    )

    equity = pnl.cumsum()
    dd = equity - equity.cummax().clip(lower=0)

    print("Trades       :", len(trades))
    print("Wins         :", wins)
    print("Losses       :", losses)
    print("Win rate     :", round(wins / len(trades) * 100, 2), "%")
    print("Net P/L      : $", round(pnl.sum(), 2))
    print("Profit Factor:", round(pf, 2))
    print("Max Drawdown : $", round(dd.min(), 2))
    print("Average P/L  : $", round(pnl.mean(), 2))
    print("Average hold :", round(trades["holding_bars"].mean(), 2), "M15 bars")

    print("\nExit reasons:")
    print(trades["exit_reason"].value_counts().to_string())

    print("\nBUY / SELL:")
    print(
        trades.groupby("side")["net_profit_usd"]
        .agg(["count", "sum", "mean"])
        .to_string()
    )


def main():
    print("Loading M5 and H1 data...")

    m5 = load_data(M5_FILE)
    h1 = load_data(H1_FILE)

    print("Building M15 candles...")
    m15 = make_m15(m5)

    print("Calculating H1 range levels...")
    h1 = build_h1_ranges(h1)

    print("Attaching completed H1 levels to M15...")
    df = attach_h1_levels(m15, h1)

    print("Detecting M15 rejection patterns...")
    df = candle_patterns(df)

    print("\n========== H1 RANGE STATUS ==========")
    print(
        h1["range_valid"]
        .value_counts(dropna=False)
        .rename(index={True: "RANGE", False: "NOT RANGE"})
        .to_string()
    )

    print("\n========== M15 DATA ==========")
    print("M15 candles:", len(df))
    print("H1 range candles:", int(df["h1_range_valid"].fillna(False).sum()))

    signals = find_signals(df)

    print("\nPotential bounce signals:", len(signals))

    signal_df = pd.DataFrame(signals)
    signal_df.to_csv(
        "range_bounce_signals.csv",
        index=False
    )

    all_results = []

    for mode in TARGET_MODES:
        trades = simulate(signals, df, mode)

        report(trades, f"RANGE BOUNCE - TARGET {mode}")

        if not trades.empty:
            trades.to_csv(
                f"range_bounce_{mode.lower()}_trades.csv",
                index=False
            )

            all_results.append(trades)

    if all_results:
        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        combined.to_csv(
            "range_bounce_all_results.csv",
            index=False
        )

    print("\n========== FILES SAVED ==========")
    print("range_bounce_signals.csv")
    print("range_bounce_midpoint_trades.csv")
    print("range_bounce_opposite_boundary_trades.csv")
    print("range_bounce_all_results.csv")

    print("\nIMPORTANT:")
    print("1. This is a research backtest, not a live trading system.")
    print("2. It uses H1 levels known only after H1 candle close.")
    print("3. Entry is at the next M15 candle open.")
    print("4. SL uses the preceding M15 swing window.")
    print("5. Same-bar SL and TP assumes SL first.")
    print("6. Spread is assumed to be CSV points x 0.01.")
    print("7. Commission and slippage are not verified.")
    print("8. No breakout strategy is included in this test.")
    print("9. No automatic trading is performed.")


if __name__ == "__main__":
    main()
