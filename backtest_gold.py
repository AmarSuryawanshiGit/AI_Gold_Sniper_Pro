
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

M1_FILE = BASE / "M1.csv"
M5_FILE = BASE / "M5.csv"
H1_FILE = BASE / "H1.csv"

POINT = 0.01
CONTRACT_SIZE = 100
LOT_SIZE = 0.01

# Actual XM Zero commission verify karke yahan set karna.
COMMISSION_PER_LOT_ROUND_TURN_USD = 0.0
SLIPPAGE = 0.0

SPREAD_IS_POINTS = True

MIN_RR = 1.5
MAX_HOLD = 60
SL_LOOKBACK = 6

# Original SL settings - unchanged
SL_ATR_MULTIPLIER = 1.2
SL_STRUCTURE_BUFFER = 0.2
MIN_SL_DISTANCE = 0.80
MAX_SL_DISTANCE = 8.00

# Confirmed swing settings
M5_LEFT, M5_RIGHT = 3, 3
H1_LEFT, H1_RIGHT = 2, 2

TARGET_BUFFER_ATR = 0.05
TARGET_BUFFER_MIN_POINTS = 2

M5_LEVEL_MAX_AGE_HOURS = 72
H1_LEVEL_MAX_AGE_HOURS = 720


def load_csv(filename, minutes):
    df = pd.read_csv(filename)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for col in ["open", "high", "low", "close", "spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["time", "open", "high", "low", "close", "spread"]
    )

    df = (
        df.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    df["available_at"] = (
        df["time"] + pd.Timedelta(minutes=minutes)
    )

    return df


def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    return df


def find_pivots(df, timeframe, left, right, max_age):
    events = []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for i in range(left, len(df) - right):

        left_high = highs[i-left:i]
        right_high = highs[i+1:i+right+1]

        left_low = lows[i-left:i]
        right_low = lows[i+1:i+right+1]

        pivot_high = (
            highs[i] > left_high.max()
            and highs[i] >= right_high.max()
        )

        pivot_low = (
            lows[i] < left_low.min()
            and lows[i] <= right_low.min()
        )

        # Pivot is available only after right-side candles close.
        confirmation_time = df.iloc[i + right]["available_at"]

        if pivot_high:
            events.append({
                "confirm_time": confirmation_time,
                "pivot_time": df.iloc[i]["time"],
                "level": float(highs[i]),
                "kind": "R",
                "max_age": max_age
            })

        if pivot_low:
            events.append({
                "confirm_time": confirmation_time,
                "pivot_time": df.iloc[i]["time"],
                "level": float(lows[i]),
                "kind": "S",
                "max_age": max_age
            })

    return pd.DataFrame(events)


def get_spread(value):
    value = max(0.0, float(value))

    if SPREAD_IS_POINTS:
        return value * POINT

    return value


def prepare_data():
    m1 = add_indicators(load_csv(M1_FILE, 1))
    m5 = add_indicators(load_csv(M5_FILE, 5))
    h1 = add_indicators(load_csv(H1_FILE, 60))

    m5_join = m5[
        ["available_at", "ema20", "ema50"]
    ].rename(columns={
        "ema20": "m5_ema20",
        "ema50": "m5_ema50"
    })

    h1_join = h1[
        ["available_at", "ema50", "ema200"]
    ].rename(columns={
        "ema50": "h1_ema50",
        "ema200": "h1_ema200"
    })

    data = pd.merge_asof(
        m1.sort_values("available_at"),
        m5_join.sort_values("available_at"),
        on="available_at",
        direction="backward"
    )

    data = pd.merge_asof(
        data.sort_values("available_at"),
        h1_join.sort_values("available_at"),
        on="available_at",
        direction="backward"
    )

    m5_levels = find_pivots(
        m5, "M5", M5_LEFT, M5_RIGHT,
        M5_LEVEL_MAX_AGE_HOURS
    )

    h1_levels = find_pivots(
        h1, "H1", H1_LEFT, H1_RIGHT,
        H1_LEVEL_MAX_AGE_HOURS
    )

    levels = pd.concat(
        [m5_levels, h1_levels],
        ignore_index=True
    )

    if not levels.empty:
        levels = levels.sort_values(
            "confirm_time"
        ).reset_index(drop=True)

    return data.reset_index(drop=True), levels


def get_known_levels(levels, signal_time, entry):
    known = levels[
        levels["confirm_time"] <= signal_time
    ].copy()

    if known.empty:
        return [], []

    age_hours = (
        signal_time - known["pivot_time"]
    ).dt.total_seconds() / 3600

    known = known[
        age_hours <= known["max_age"]
    ]

    resistance = known[
        (known["kind"] == "R")
        & (known["level"] > entry)
    ]["level"].tolist()

    support = known[
        (known["kind"] == "S")
        & (known["level"] < entry)
    ]["level"].tolist()

    return resistance, support


def calculate_stop(side, entry, recent, atr):
    # Original small SL logic is unchanged.
    if not np.isfinite(atr) or atr <= 0:
        return None

    if side == "BUY":
        structure_sl = (
            recent["low"].min()
            - SL_STRUCTURE_BUFFER * atr
        )

        atr_sl = entry - SL_ATR_MULTIPLIER * atr

        sl = min(structure_sl, atr_sl)
        distance = entry - sl

    else:
        structure_sl = (
            recent["high"].max()
            + SL_STRUCTURE_BUFFER * atr
        )

        atr_sl = entry + SL_ATR_MULTIPLIER * atr

        sl = max(structure_sl, atr_sl)
        distance = sl - entry

    if distance < MIN_SL_DISTANCE:
        sl = (
            entry - MIN_SL_DISTANCE
            if side == "BUY"
            else entry + MIN_SL_DISTANCE
        )
        distance = MIN_SL_DISTANCE

    if distance > MAX_SL_DISTANCE:
        return None

    return round(sl / POINT) * POINT


def get_signal(row, previous):
    h1_buy = row["h1_ema50"] > row["h1_ema200"]
    h1_sell = row["h1_ema50"] < row["h1_ema200"]

    m5_buy = row["m5_ema20"] > row["m5_ema50"]
    m5_sell = row["m5_ema20"] < row["m5_ema50"]

    m1_buy = row["ema20"] > row["ema50"]
    m1_sell = row["ema20"] < row["ema50"]

    if (
        h1_buy and m5_buy and m1_buy
        and row["close"] > previous["high"]
    ):
        return "BUY"

    if (
        h1_sell and m5_sell and m1_sell
        and row["close"] < previous["low"]
    ):
        return "SELL"

    return None


def simulate_exit(data, start, side, entry, sl, tp):
    end = min(len(data) - 1, start + MAX_HOLD - 1)

    for j in range(start, end + 1):
        candle = data.iloc[j]
        spread = get_spread(candle["spread"])

        if side == "BUY":
            stop_hit = candle["low"] <= sl
            target_hit = candle["high"] >= tp

            # Conservative: SL first if both touched.
            if stop_hit:
                return j, sl - SLIPPAGE, "SL"

            if target_hit:
                return j, tp - SLIPPAGE, "TP"

        else:
            ask_high = candle["high"] + spread
            ask_low = candle["low"] + spread

            stop_hit = ask_high >= sl
            target_hit = ask_low <= tp

            if stop_hit:
                return j, sl + SLIPPAGE, "SL"

            if target_hit:
                return j, tp + SLIPPAGE, "TP"

    candle = data.iloc[end]

    if side == "BUY":
        exit_price = candle["close"] - SLIPPAGE
    else:
        exit_price = (
            candle["close"]
            + get_spread(candle["spread"])
            + SLIPPAGE
        )

    return end, exit_price, "TIME"


def run_backtest(data, levels):
    trades = []
    i = 202

    while i < len(data) - 1:
        row = data.iloc[i]
        previous = data.iloc[i - 1]

        required = [
            "atr", "m5_ema20", "m5_ema50",
            "h1_ema50", "h1_ema200"
        ]

        if any(pd.isna(row[x]) for x in required):
            i += 1
            continue

        side = get_signal(row, previous)

        if side is None:
            i += 1
            continue

        # Entry on next M1 candle open.
        entry_row = data.iloc[i + 1]
        spread = get_spread(entry_row["spread"])

        if side == "BUY":
            entry = (
                entry_row["open"] + spread + SLIPPAGE
            )
        else:
            entry = entry_row["open"] - SLIPPAGE

        recent = data.iloc[
            max(0, i - SL_LOOKBACK + 1):i + 1
        ]

        sl = calculate_stop(
            side, entry, recent, row["atr"]
        )

        if sl is None:
            i += 1
            continue

        risk = (
            entry - sl if side == "BUY"
            else sl - entry
        )

        if risk <= 0:
            i += 1
            continue

        resistances, supports = get_known_levels(
            levels, row["available_at"], entry
        )

        buffer_price = max(
            TARGET_BUFFER_MIN_POINTS * POINT,
            TARGET_BUFFER_ATR * row["atr"]
        )

        if side == "BUY":
            targets = [
                level - buffer_price
                for level in resistances
                if level - buffer_price > entry
            ]

            if not targets:
                i += 1
                continue

            tp = min(targets)
            reward = tp - entry

        else:
            targets = [
                level + buffer_price
                for level in supports
                if level + buffer_price < entry
            ]

            if not targets:
                i += 1
                continue

            tp = max(targets)
            reward = entry - tp

        rr = reward / risk

        if reward <= 0 or rr < MIN_RR:
            i += 1
            continue

        exit_index, exit_price, reason = simulate_exit(
            data, i + 1, side, entry, sl, tp
        )

        if side == "BUY":
            gross = (
                (exit_price - entry)
                * CONTRACT_SIZE * LOT_SIZE
            )
        else:
            gross = (
                (entry - exit_price)
                * CONTRACT_SIZE * LOT_SIZE
            )

        commission = (
            COMMISSION_PER_LOT_ROUND_TURN_USD
            * LOT_SIZE
        )

        net = gross - commission

        trades.append({
            "signal_time": row["available_at"],
            "entry_time": entry_row["time"],
            "exit_time": data.iloc[exit_index]["time"],
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "reward": reward,
            "planned_rr": rr,
            "exit": exit_price,
            "exit_reason": reason,
            "gross_profit_usd": gross,
            "commission_usd": commission,
            "net_profit_usd": net
        })

        # One trade at a time.
        i = exit_index + 1

    return pd.DataFrame(trades)


def main():
    print("Loading M1, M5 and H1 data...")

    data, levels = prepare_data()

    print("M1 candles:", len(data))
    print("Confirmed S/R levels:", len(levels))

    trades = run_backtest(data, levels)

    if trades.empty:
        print("No trades found.")
        return

    trades.to_csv(
        BASE / "backtest_trades_dynamic_sr.csv",
        index=False
    )

    total = len(trades)
    wins = (trades["net_profit_usd"] > 0).sum()
    losses = (trades["net_profit_usd"] <= 0).sum()

    win_rate = wins / total * 100

    net = trades["net_profit_usd"].sum()

    gross_profit = trades.loc[
        trades["net_profit_usd"] > 0,
        "net_profit_usd"
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["net_profit_usd"] < 0,
            "net_profit_usd"
        ].sum()
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0 else float("inf")
    )

    equity = trades["net_profit_usd"].cumsum()
    drawdown = equity - equity.cummax()
    max_dd = drawdown.min()

    print("\n====== DYNAMIC S/R BACKTEST ======")
    print("Total trades :", total)
    print("Wins         :", wins)
    print("Losses       :", losses)
    print(f"Win rate     : {win_rate:.2f}%")
    print(f"Net P/L      : ${net:.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Max Drawdown : ${max_dd:.2f}")
    print("TP           :", (trades.exit_reason == "TP").sum())
    print("SL           :", (trades.exit_reason == "SL").sum())
    print("TIME         :", (trades.exit_reason == "TIME").sum())

    summary = pd.DataFrame([{
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "net_profit_usd": net,
        "profit_factor": pf,
        "max_drawdown_usd": max_dd
    }])

    summary.to_csv(
        BASE / "backtest_summary_dynamic_sr.csv",
        index=False
    )

    print("\nFiles saved:")
    print("backtest_trades_dynamic_sr.csv")
    print("backtest_summary_dynamic_sr.csv")


if __name__ == "__main__":
    main()
