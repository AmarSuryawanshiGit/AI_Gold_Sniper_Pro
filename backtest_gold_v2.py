
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================================
# AI GOLD SNIPER PRO - BACKTEST V2
# M1 breakout + strong closed M5 momentum + H1 trend
# Nearest confirmed M5/H1 S/R + MFE/MAE
# ==========================================================

BASE = Path(__file__).resolve().parent
M1_FILE = BASE / "M1.csv"
M5_FILE = BASE / "M5.csv"
H1_FILE = BASE / "H1.csv"

# XM GOLD specification supplied by user
POINT = 0.01
CONTRACT_SIZE = 100
LOT_SIZE = 0.01

# Placeholder: verify actual XM Zero round-turn commission.
# USD per 1.00 lot, round turn. Zero is NOT confirmed commission-free.
COMMISSION_PER_LOT_ROUND_TURN_USD = 0.0
SLIPPAGE = 0.0

# Assumes CSV spread is in points: 20 = $0.20.
# Verify this against your MT5 CSV export.
SPREAD_IS_POINTS = True

MIN_RR = 1.5
MAX_HOLD = 60
SL_LOOKBACK = 6

# ORIGINAL SL SETTINGS - UNCHANGED
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

# Strong M5 momentum settings
M5_EMA_SLOPE_LOOKBACK = 3
M5_MIN_BODY_RATIO = 0.55
M5_MIN_CLOSE_LOCATION_BUY = 0.65
M5_MAX_CLOSE_LOCATION_SELL = 0.35


def load_csv(filename, minutes):
    if not filename.exists():
        raise FileNotFoundError(
            f"Missing {filename.name}. Put it in {BASE}"
        )

    df = pd.read_csv(filename)

    required = [
        "time", "open", "high",
        "low", "close", "spread"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{filename.name} missing columns: {missing}"
        )

    df["time"] = pd.to_datetime(
        df["time"], errors="coerce"
    )

    for col in [
        "open", "high", "low", "close", "spread"
    ]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = (
        df.dropna(subset=required)
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    # OHLC is available only after candle close.
    df["available_at"] = (
        df["time"] + pd.Timedelta(minutes=minutes)
    )

    return df


def add_indicators(df):
    df = df.copy()

    for span in [20, 50, 200]:
        df[f"ema{span}"] = (
            df["close"].ewm(
                span=span, adjust=False
            ).mean()
        )

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    return df


def add_m5_momentum(df):
    df = df.copy()

    candle_range = (
        df["high"] - df["low"]
    ).replace(0, np.nan)

    body = (
        df["close"] - df["open"]
    ).abs()

    df["m5_body_ratio"] = (
        body / candle_range
    ).fillna(0.0)

    df["m5_close_location"] = (
        (df["close"] - df["low"]) / candle_range
    ).fillna(0.5)

    df["m5_ema20_slope"] = (
        df["ema20"] -
        df["ema20"].shift(M5_EMA_SLOPE_LOOKBACK)
    )

    buy_checks = [
        df["ema20"] > df["ema50"],
        df["m5_ema20_slope"] > 0,
        df["close"] > df["ema20"],
        df["close"] > df["open"],
        df["m5_body_ratio"] >= M5_MIN_BODY_RATIO,
        df["m5_close_location"] >= M5_MIN_CLOSE_LOCATION_BUY
    ]

    sell_checks = [
        df["ema20"] < df["ema50"],
        df["m5_ema20_slope"] < 0,
        df["close"] < df["ema20"],
        df["close"] < df["open"],
        df["m5_body_ratio"] >= M5_MIN_BODY_RATIO,
        df["m5_close_location"] <= M5_MAX_CLOSE_LOCATION_SELL
    ]

    df["m5_buy_score"] = sum(
        x.astype(int) for x in buy_checks
    )

    df["m5_sell_score"] = sum(
        x.astype(int) for x in sell_checks
    )

    df["m5_buy_confirm"] = np.logical_and.reduce(
        buy_checks
    )

    df["m5_sell_confirm"] = np.logical_and.reduce(
        sell_checks
    )

    return df


def find_pivots(df, timeframe, left, right, max_age):
    events = []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for i in range(left, len(df) - right):

        pivot_high = (
            highs[i] > highs[i-left:i].max()
            and highs[i] >= highs[i+1:i+right+1].max()
        )

        pivot_low = (
            lows[i] < lows[i-left:i].min()
            and lows[i] <= lows[i+1:i+right+1].min()
        )

        # Pivot is known only after right candles close.
        confirm_time = (
            df.iloc[i + right]["available_at"]
        )

        if pivot_high:
            events.append({
                "confirm_time": confirm_time,
                "pivot_time": df.iloc[i]["time"],
                "level": float(highs[i]),
                "kind": "R",
                "source": f"{timeframe}_RESISTANCE",
                "max_age": max_age
            })

        if pivot_low:
            events.append({
                "confirm_time": confirm_time,
                "pivot_time": df.iloc[i]["time"],
                "level": float(lows[i]),
                "kind": "S",
                "source": f"{timeframe}_SUPPORT",
                "max_age": max_age
            })

    return pd.DataFrame(events)


def get_spread(value):
    value = max(0.0, float(value))

    if SPREAD_IS_POINTS:
        return value * POINT

    return value


def prepare_data():
    m1 = add_indicators(
        load_csv(M1_FILE, 1)
    )

    m5 = add_m5_momentum(
        add_indicators(load_csv(M5_FILE, 5))
    )

    h1 = add_indicators(
        load_csv(H1_FILE, 60)
    )

    m5_join = m5[
        [
            "available_at",
            "ema20",
            "ema50",
            "m5_ema20_slope",
            "m5_body_ratio",
            "m5_close_location",
            "m5_buy_score",
            "m5_sell_score",
            "m5_buy_confirm",
            "m5_sell_confirm"
        ]
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

    # Only completed M5 and H1 candles are merged.
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
        m5, "M5", M5_LEFT,
        M5_RIGHT, M5_LEVEL_MAX_AGE_HOURS
    )

    h1_levels = find_pivots(
        h1, "H1", H1_LEFT,
        H1_RIGHT, H1_LEVEL_MAX_AGE_HOURS
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
    if levels.empty:
        return []

    known = levels.loc[
        levels["confirm_time"] <= signal_time
    ].copy()

    if known.empty:
        return []

    age_hours = (
        signal_time - known["pivot_time"]
    ).dt.total_seconds() / 3600

    known = known.loc[
        age_hours <= known["max_age"]
    ]

    result = []

    for r in known.itertuples(index=False):

        if r.kind == "R" and r.level > entry:
            result.append({
                "level": float(r.level),
                "source": r.source,
                "kind": "R",
                "confirm_time": r.confirm_time
            })

        elif r.kind == "S" and r.level < entry:
            result.append({
                "level": float(r.level),
                "source": r.source,
                "kind": "S",
                "confirm_time": r.confirm_time
            })

    return result


def calculate_stop(side, entry, recent, atr):
    # ORIGINAL SL FORMULA - UNCHANGED

    if not np.isfinite(atr) or atr <= 0:
        return None

    if side == "BUY":

        structure_sl = (
            recent["low"].min()
            - SL_STRUCTURE_BUFFER * atr
        )

        atr_sl = (
            entry - SL_ATR_MULTIPLIER * atr
        )

        sl = min(structure_sl, atr_sl)
        distance = entry - sl

    else:

        structure_sl = (
            recent["high"].max()
            + SL_STRUCTURE_BUFFER * atr
        )

        atr_sl = (
            entry + SL_ATR_MULTIPLIER * atr
        )

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

    h1_buy = (
        row["h1_ema50"] > row["h1_ema200"]
    )

    h1_sell = (
        row["h1_ema50"] < row["h1_ema200"]
    )

    m1_buy = row["ema20"] > row["ema50"]
    m1_sell = row["ema20"] < row["ema50"]

    if (
        h1_buy
        and bool(row["m5_buy_confirm"])
        and m1_buy
        and row["close"] > previous["high"]
    ):
        return "BUY"

    if (
        h1_sell
        and bool(row["m5_sell_confirm"])
        and m1_sell
        and row["close"] < previous["low"]
    ):
        return "SELL"

    return None


def choose_nearest_target(
    side, levels, entry, buffer_price
):
    # Select nearest valid level across M5 and H1.
    # Never skip a nearer level just to force higher RR.

    if side == "BUY":

        candidates = [
            x for x in levels
            if x["kind"] == "R"
            and x["level"] - buffer_price > entry
        ]

        if not candidates:
            return None

        selected = min(
            candidates,
            key=lambda x: x["level"] - buffer_price
        )

        tp = selected["level"] - buffer_price

    else:

        candidates = [
            x for x in levels
            if x["kind"] == "S"
            and x["level"] + buffer_price < entry
        ]

        if not candidates:
            return None

        selected = max(
            candidates,
            key=lambda x: x["level"] + buffer_price
        )

        tp = selected["level"] + buffer_price

    return {
        "tp": round(tp / POINT) * POINT,
        "target_level": selected["level"],
        "target_source": selected["source"],
        "target_confirm_time": selected["confirm_time"]
    }


def simulate_exit(data, start, side, entry, sl, tp):
    """
    MFE/MAE include the exit candle OHLC extremes.

    BUY:
      Executable exit side = Bid.

    SELL:
      Executable exit side = Ask = Bid + spread.

    If SL and TP both touch in one candle, SL is assumed first.
    OHLC cannot establish the exact intrabar sequence.
    """

    end = min(
        len(data) - 1,
        start + MAX_HOLD - 1
    )

    mfe = 0.0
    mae = 0.0
    both_hit_same_bar = False

    for j in range(start, end + 1):

        candle = data.iloc[j]
        spread = get_spread(candle["spread"])

        if side == "BUY":

            favourable = (
                float(candle["high"]) - entry
            )

            adverse = (
                float(candle["low"]) - entry
            )

            mfe = max(mfe, favourable)
            mae = min(mae, adverse)

            stop_hit = candle["low"] <= sl
            target_hit = candle["high"] >= tp

            both_hit_same_bar = bool(
                stop_hit and target_hit
            )

            if stop_hit:
                return (
                    j, sl - SLIPPAGE, "SL",
                    mfe, mae, both_hit_same_bar
                )

            if target_hit:
                return (
                    j, tp - SLIPPAGE, "TP",
                    mfe, mae, both_hit_same_bar
                )

        else:

            ask_high = (
                float(candle["high"]) + spread
            )

            ask_low = (
                float(candle["low"]) + spread
            )

            favourable = entry - ask_low
            adverse = entry - ask_high

            mfe = max(mfe, favourable)
            mae = min(mae, adverse)

            stop_hit = ask_high >= sl
            target_hit = ask_low <= tp

            both_hit_same_bar = bool(
                stop_hit and target_hit
            )

            if stop_hit:
                return (
                    j, sl + SLIPPAGE, "SL",
                    mfe, mae, both_hit_same_bar
                )

            if target_hit:
                return (
                    j, tp + SLIPPAGE, "TP",
                    mfe, mae, both_hit_same_bar
                )

    candle = data.iloc[end]

    if side == "BUY":

        exit_price = (
            float(candle["close"]) - SLIPPAGE
        )

    else:

        exit_price = (
            float(candle["close"])
            + get_spread(candle["spread"])
            + SLIPPAGE
        )

    return (
        end, exit_price, "TIME",
        mfe, mae, both_hit_same_bar
    )


def run_backtest(data, levels):

    trades = []
    i = 202

    while i < len(data) - 1:

        row = data.iloc[i]
        previous = data.iloc[i - 1]

        required = [
            "atr",
            "m5_ema20",
            "m5_ema50",
            "m5_ema20_slope",
            "m5_body_ratio",
            "m5_close_location",
            "m5_buy_score",
            "m5_sell_score",
            "h1_ema50",
            "h1_ema200"
        ]

        if any(pd.isna(row[x]) for x in required):
            i += 1
            continue

        side = get_signal(row, previous)

        if side is None:
            i += 1
            continue

        # Entry at next M1 candle open.
        entry_row = data.iloc[i + 1]
        spread = get_spread(entry_row["spread"])

        if side == "BUY":

            entry = (
                float(entry_row["open"])
                + spread + SLIPPAGE
            )

        else:

            entry = (
                float(entry_row["open"])
                - SLIPPAGE
            )

        recent = data.iloc[
            max(0, i - SL_LOOKBACK + 1):i + 1
        ]

        # Existing SL calculation unchanged.
        sl = calculate_stop(
            side, entry, recent, row["atr"]
        )

        if sl is None:
            i += 1
            continue

        risk = (
            entry - sl
            if side == "BUY"
            else sl - entry
        )

        if risk <= 0:
            i += 1
            continue

        known_levels = get_known_levels(
            levels, row["available_at"], entry
        )

        buffer_price = max(
            TARGET_BUFFER_MIN_POINTS * POINT,
            TARGET_BUFFER_ATR * row["atr"]
        )

        target = choose_nearest_target(
            side, known_levels, entry, buffer_price
        )

        if target is None:
            i += 1
            continue

        tp = target["tp"]

        reward = (
            tp - entry
            if side == "BUY"
            else entry - tp
        )

        rr = reward / risk if risk > 0 else 0.0

        # If nearest valid target fails minimum RR, skip.
        if reward <= 0 or rr < MIN_RR:
            i += 1
            continue

        (
            exit_index,
            exit_price,
            reason,
            mfe,
            mae,
            both_touched
        ) = simulate_exit(
            data, i + 1, side, entry, sl, tp
        )

        multiplier = CONTRACT_SIZE * LOT_SIZE

        if side == "BUY":

            gross = (
                (exit_price - entry) * multiplier
            )

        else:

            gross = (
                (entry - exit_price) * multiplier
            )

        commission = (
            COMMISSION_PER_LOT_ROUND_TURN_USD
            * LOT_SIZE
        )

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

            "m5_momentum_score": (
                row["m5_buy_score"]
                if side == "BUY"
                else row["m5_sell_score"]
            ),

            "m5_ema20_slope": row["m5_ema20_slope"],
            "m5_body_ratio": row["m5_body_ratio"],
            "m5_close_location": row["m5_close_location"],

            "target_source": target["target_source"],
            "target_level": target["target_level"],
            "target_confirm_time": target["target_confirm_time"],

            "exit": exit_price,
            "exit_reason": reason,

            "mfe_price": mfe,
            "mae_price": mae,

            "mfe_points": mfe / POINT,
            "mae_points": mae / POINT,

            "mfe_usd": mfe * multiplier,
            "mae_usd": mae * multiplier,

            "both_sl_tp_touched_same_bar": both_touched,

            "gross_profit_usd": gross,
            "commission_usd": commission,
            "net_profit_usd": gross - commission
        })

        # One trade at a time.
        i = exit_index + 1

    return pd.DataFrame(trades)


def profit_factor(trades):

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

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss


def group_report(trades, column, title):

    print(f"\n--- {title} ---")

    for key, group in trades.groupby(
        column, dropna=False
    ):

        total = len(group)

        win_rate = (
            group["net_profit_usd"] > 0
        ).mean() * 100

        print(
            f"{key}: "
            f"trades={total}, "
            f"win_rate={win_rate:.2f}%, "
            f"net=${group.net_profit_usd.sum():.2f}, "
            f"PF={profit_factor(group):.2f}, "
            f"avg_MFE=${group.mfe_usd.mean():.2f}, "
            f"avg_MAE=${group.mae_usd.mean():.2f}"
        )


def main():

    print("Loading M1, M5 and H1 data...")

    data, levels = prepare_data()

    print("M1 candles:", len(data))
    print("Confirmed S/R levels:", len(levels))

    if not levels.empty:

        print(
            "M5 levels:",
            int(levels["source"].str.startswith("M5").sum())
        )

        print(
            "H1 levels:",
            int(levels["source"].str.startswith("H1").sum())
        )

    trades = run_backtest(data, levels)

    if trades.empty:
        print("No trades found. Check CSV data and filters.")
        return

    trades.to_csv(
        BASE / "backtest_trades_v2.csv",
        index=False
    )

    total = len(trades)

    wins = int(
        (trades["net_profit_usd"] > 0).sum()
    )

    losses = total - wins

    net = float(
        trades["net_profit_usd"].sum()
    )

    equity = trades["net_profit_usd"].cumsum()

    max_dd = float(
        (equity - equity.cummax()).min()
    )

    pf = profit_factor(trades)

    summary = pd.DataFrame([{

        "total_trades": total,
        "wins": wins,
        "losses": losses,

        "win_rate_pct": wins / total * 100,
        "net_profit_usd": net,
        "profit_factor": pf,
        "max_drawdown_usd": max_dd,

        "average_mfe_usd": trades["mfe_usd"].mean(),
        "average_mae_usd": trades["mae_usd"].mean(),

        "same_bar_sl_tp_count": int(
            trades["both_sl_tp_touched_same_bar"].sum()
        )

    }])

    summary.to_csv(
        BASE / "backtest_summary_v2.csv",
        index=False
    )

    # Separate performance report for each target source.
    source_rows = []

    for source, group in trades.groupby("target_source"):

        source_rows.append({

            "target_source": source,
            "total_trades": len(group),

            "wins": int(
                (group["net_profit_usd"] > 0).sum()
            ),

            "win_rate_pct": (
                group["net_profit_usd"] > 0
            ).mean() * 100,

            "net_profit_usd": group["net_profit_usd"].sum(),

            "profit_factor": profit_factor(group),

            "average_mfe_usd": group["mfe_usd"].mean(),
            "average_mae_usd": group["mae_usd"].mean()

        })

    pd.DataFrame(source_rows).to_csv(
        BASE / "backtest_target_source_v2.csv",
        index=False
    )

    print("\n========== GOLD BACKTEST V2 ==========")

    print("Total trades :", total)
    print("Wins         :", wins)
    print("Losses       :", losses)

    print(f"Win rate     : {wins / total * 100:.2f}%")
    print(f"Net P/L      : ${net:.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Max Drawdown : ${max_dd:.2f}")

    print(
        f"Average MFE  : ${trades.mfe_usd.mean():.2f}"
    )

    print(
        f"Average MAE  : ${trades.mae_usd.mean():.2f}"
    )

    print(
        "TP           :",
        int((trades.exit_reason == "TP").sum())
    )

    print(
        "SL           :",
        int((trades.exit_reason == "SL").sum())
    )

    print(
        "TIME         :",
        int((trades.exit_reason == "TIME").sum())
    )

    print(
        "Same-bar SL+TP:",
        int(trades.both_sl_tp_touched_same_bar.sum()),
        "(OHLC order unknown; SL-first assumed)"
    )

    group_report(trades, "side", "BUY vs SELL")
    group_report(
        trades, "target_source",
        "M5 vs H1 target source"
    )
    group_report(
        trades, "exit_reason",
        "Exit reason"
    )

    print("\nFiles saved:")

    print("backtest_trades_v2.csv")
    print("backtest_summary_v2.csv")
    print("backtest_target_source_v2.csv")

    print("\nIMPORTANT:")
    print("1. Verify CSV spread units.")
    print("2. Set actual XM Zero commission.")
    print("3. Slippage is currently zero.")
    print("4. MFE/MAE use M1 OHLC; intrabar sequence is unknown.")
    print("5. Validate on unseen chronological data before live use.")


if __name__ == "__main__":
    main()
