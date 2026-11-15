
import pandas as pd
import numpy as np

M5_FILE = "M5.csv"
H1_FILE = "H1.csv"
TRADES_FILE = "backtest_trades_v2.csv"

# Initial experiment parameters; not optimized.
M15_LOOKBACK = 16
H1_LOOKBACK = 8

M15_MAX_WIDTH_ATR = 5.0
H1_MAX_WIDTH_ATR = 4.0

TOUCH_TOLERANCE_ATR = 0.25
BREAKOUT_BUFFER_ATR = 0.10
MIN_TOUCHES = 2


def load_csv(path):
    df = pd.read_csv(path)

    df["time"] = pd.to_datetime(
        df["time"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    return df.sort_values("time").reset_index(drop=True)


def make_m15(m5):
    data = m5.set_index("time")

    m15 = data.resample(
        "15min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    return m15.reset_index()


def calculate_atr(df, period=14):
    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def detect_ranges(df, timeframe, lookback, max_width_atr):
    df = df.copy()

    df["atr"] = calculate_atr(df)

    # Boundaries are calculated from PREVIOUS completed candles.
    df["resistance"] = (
        df["high"].shift(1)
        .rolling(lookback, min_periods=lookback)
        .max()
    )

    df["support"] = (
        df["low"].shift(1)
        .rolling(lookback, min_periods=lookback)
        .min()
    )

    df["range_width"] = (
        df["resistance"] - df["support"]
    )

    df["width_atr"] = (
        df["range_width"] / df["atr"]
    )

    # Count how often previous candles tested both boundaries.
    highs = df["high"].shift(1)
    lows = df["low"].shift(1)

    tolerance = df["atr"] * TOUCH_TOLERANCE_ATR

    df["top_touches"] = (
        (highs >= df["resistance"] - tolerance)
        .rolling(lookback, min_periods=lookback)
        .sum()
    )

    df["bottom_touches"] = (
        (lows <= df["support"] + tolerance)
        .rolling(lookback, min_periods=lookback)
        .sum()
    )

    valid = (
        df["atr"].notna()
        & df["resistance"].notna()
        & df["support"].notna()
        & df["width_atr"].notna()
    )

    narrow = df["width_atr"] <= max_width_atr

    enough_touches = (
        (df["top_touches"] >= MIN_TOUCHES)
        & (df["bottom_touches"] >= MIN_TOUCHES)
    )

    inside = (
        (df["close"] <= df["resistance"])
        & (df["close"] >= df["support"])
    )

    # A close outside the old boundary is a breakout/breakdown.
    buffer = df["atr"] * BREAKOUT_BUFFER_ATR

    breakout = df["close"] > (
        df["resistance"] + buffer
    )

    breakdown = df["close"] < (
        df["support"] - buffer
    )

    df["status"] = "TRENDING"

    df.loc[
        valid & narrow & enough_touches & inside,
        "status"
    ] = "RANGING"

    df.loc[valid & breakout, "status"] = "BREAKOUT"
    df.loc[valid & breakdown, "status"] = "BREAKDOWN"

    df.loc[~valid, "status"] = "UNKNOWN"

    # Candle becomes usable only after it closes.
    minutes = 15 if timeframe == "M15" else 60

    df["available_time"] = (
        df["time"] + pd.Timedelta(minutes=minutes)
    )

    return df[[
        "available_time",
        "status",
        "support",
        "resistance",
        "width_atr",
        "top_touches",
        "bottom_touches",
        "atr"
    ]].copy()


def attach_timeframe(trades, levels, prefix):
    left = trades.sort_values("entry_time").copy()
    right = levels.sort_values("available_time").copy()

    right = right.rename(columns={
        col: f"{prefix}_{col}"
        for col in right.columns
        if col != "available_time"
    })

    return pd.merge_asof(
        left,
        right,
        left_on="entry_time",
        right_on="available_time",
        direction="backward"
    )


def metrics(df, name):
    if len(df) == 0:
        print(f"\n{name}: No trades")
        return

    pnl = df["net_profit_usd"].astype(float)

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0 else float("inf")
    )

    equity = pnl.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0)

    print(f"\n{name}")
    print("-" * 40)
    print("Trades       :", len(df))
    print("Wins         :", wins)
    print("Losses       :", losses)
    print("Win rate     :", round(wins / len(df) * 100, 2), "%")
    print("Net P/L      : $", round(pnl.sum(), 2))
    print("Profit Factor:", round(pf, 2))
    print("Max Drawdown : $", round(drawdown.min(), 2))


def main():
    print("Loading CSV files...")

    m5 = load_csv(M5_FILE)
    h1 = load_csv(H1_FILE)

    trades = pd.read_csv(TRADES_FILE)

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True
    )

    trades["net_profit_usd"] = pd.to_numeric(
        trades["net_profit_usd"], errors="coerce"
    )

    trades = trades.dropna(
        subset=["entry_time", "net_profit_usd"]
    )

    print("Creating M15 candles from M5...")
    m15 = make_m15(m5)

    print("Detecting M15 ranges...")
    m15_levels = detect_ranges(
        m15,
        "M15",
        M15_LOOKBACK,
        M15_MAX_WIDTH_ATR
    )

    print("Detecting H1 ranges...")
    h1_levels = detect_ranges(
        h1,
        "H1",
        H1_LOOKBACK,
        H1_MAX_WIDTH_ATR
    )

    print("Matching completed candles with entries...")

    result = attach_timeframe(
        trades, m15_levels, "m15"
    )

    result = attach_timeframe(
        result, h1_levels, "h1"
    )

    # Three separate filtering experiments.
    result["m15_block"] = (
        result["m15_status"] == "RANGING"
    )

    result["h1_block"] = (
        result["h1_status"] == "RANGING"
    )

    result["both_block"] = (
        result["m15_block"] | result["h1_block"]
    )

    print("\n========== TIMEFRAME STATUS ==========")

    print("\nM15:")
    print(result["m15_status"].value_counts().to_string())

    print("\nH1:")
    print(result["h1_status"].value_counts().to_string())

    print("\n========== FILTER COMPARISON ==========")

    metrics(result, "BASELINE - NO FILTER")

    metrics(
        result[~result["m15_block"]],
        "M15 RANGE FILTER"
    )

    metrics(
        result[~result["h1_block"]],
        "H1 RANGE FILTER"
    )

    metrics(
        result[~result["both_block"]],
        "M15 + H1 RANGE FILTER"
    )

    print("\n========== BLOCKED TRADE ANALYSIS ==========")

    for label, mask in [
        ("M15", result["m15_block"]),
        ("H1", result["h1_block"]),
        ("M15 + H1", result["both_block"])
    ]:
        blocked = result[mask]

        print(f"\n{label} blocked:")
        print("Trades:", len(blocked))
        print("Blocked P/L:", round(
            blocked["net_profit_usd"].sum(), 2
        ))
        print("Profitable trades blocked:", int(
            (blocked["net_profit_usd"] > 0).sum()
        ))
        print("Losing trades avoided:", int(
            (blocked["net_profit_usd"] < 0).sum()
        ))

    output_cols = [
        "signal_time", "entry_time", "exit_time",
        "side", "entry", "exit",
        "exit_reason", "net_profit_usd",
        "m15_status", "m15_support", "m15_resistance",
        "m15_width_atr",
        "h1_status", "h1_support", "h1_resistance",
        "h1_width_atr",
        "m15_block", "h1_block", "both_block"
    ]

    output_cols = [
        c for c in output_cols if c in result.columns
    ]

    result[output_cols].to_csv(
        "htf_range_analysis_trades.csv",
        index=False
    )

    result.groupby(
        ["m15_status", "h1_status"],
        dropna=False
    ).agg(
        trades=("net_profit_usd", "size"),
        net_pnl=("net_profit_usd", "sum"),
        avg_pnl=("net_profit_usd", "mean")
    ).reset_index().to_csv(
        "htf_range_analysis_summary.csv",
        index=False
    )

    print("\nFiles saved:")
    print("htf_range_analysis_trades.csv")
    print("htf_range_analysis_summary.csv")

    print("\nIMPORTANT:")
    print("This is a historical filter study of existing V2 trades.")
    print("It does not change entry, SL or TP.")
    print("It does not independently rerun the strategy.")
    print("Costs and spread assumptions are inherited from V2.")


if __name__ == "__main__":
    main()
