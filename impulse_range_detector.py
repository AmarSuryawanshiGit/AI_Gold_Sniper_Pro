
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# IMPULSE -> PULLBACK -> RETEST -> RANGE DETECTOR
# Research only: no BUY/SELL signals and no auto trading.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

M5_FILE = BASE_DIR / "M5.csv"
H1_FILE = BASE_DIR / "H1.csv"

# ---------------- STRATEGY SETTINGS ----------------

IMPULSE_LOOKBACK = 8
IMPULSE_ATR_MULTIPLE = 2.0

MIN_PULLBACK_RETRACE = 0.20
MAX_PULLBACK_RETRACE = 0.75
MAX_PULLBACK_BARS = 12

RETEST_TOLERANCE_ATR = 0.25
MAX_RETEST_BARS = 12

RANGE_CONFIRM_BARS = 8
MAX_RANGE_FORMATION_BARS = 20

MIN_TOUCHES_UPPER = 2
MIN_TOUCHES_LOWER = 2

MIN_RANGE_ATR = 0.80
MAX_RANGE_ATR = 10.0

INSIDE_RATIO_REQUIRED = 0.80
BREAKOUT_BUFFER_ATR = 0.15

ATR_PERIOD = 14


# ---------------- DATA LOADING ----------------

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    # Handle normal XM date strings and numeric Unix timestamps.
    if pd.api.types.is_numeric_dtype(df["time"]):
        sample = df["time"].dropna().iloc[0]
        unit = "ms" if sample > 10**12 else "s"
        df["time"] = pd.to_datetime(df["time"], unit=unit, utc=True)
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df.sort_values("time").drop_duplicates("time")
    df = df.reset_index(drop=True)

    return df


def make_m15(m5):
    x = m5.copy().set_index("time")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }

    if "tick_volume" in x.columns:
        agg["tick_volume"] = "sum"

    if "spread" in x.columns:
        agg["spread"] = "mean"

    if "real_volume" in x.columns:
        agg["real_volume"] = "sum"

    # Label M15 candles by their opening time.
    m15 = x.resample("15min", label="left", closed="left").agg(agg)

    # Only retain complete M15 candles made from 3 M5 candles.
    count = x["close"].resample("15min", label="left", closed="left").count()
    m15 = m15[count == 3]

    m15 = m15.dropna(subset=["open", "high", "low", "close"])
    return m15.reset_index()


# ---------------- ATR ----------------

def add_atr(df):
    x = df.copy()

    previous_close = x["close"].shift(1)

    tr1 = x["high"] - x["low"]
    tr2 = (x["high"] - previous_close).abs()
    tr3 = (x["low"] - previous_close).abs()

    x["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    x["atr"] = x["tr"].rolling(ATR_PERIOD).mean()

    return x


# ---------------- IMPULSE CHECK ----------------

def detect_impulse(df, end):
    start = end - IMPULSE_LOOKBACK

    if start < 0:
        return None

    window = df.iloc[start:end]

    atr = df.iloc[end - 1]["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    displacement = (
        df.iloc[end - 1]["close"] - df.iloc[start]["open"]
    )

    impulse_high = window["high"].max()
    impulse_low = window["low"].min()
    impulse_size = impulse_high - impulse_low

    if impulse_size <= 0:
        return None

    if abs(displacement) < IMPULSE_ATR_MULTIPLE * atr:
        return None

    bullish_count = (window["close"] > window["open"]).sum()
    bearish_count = (window["close"] < window["open"]).sum()

    required_directional = int(np.ceil(IMPULSE_LOOKBACK * 0.60))

    if displacement > 0 and bullish_count >= required_directional:
        direction = "BULLISH"

    elif displacement < 0 and bearish_count >= required_directional:
        direction = "BEARISH"

    else:
        return None

    if impulse_size / atr > 8:
        return None

    return {
        "direction": direction,
        "start_index": start,
        "end_index": end - 1,
        "impulse_high": float(impulse_high),
        "impulse_low": float(impulse_low),
        "impulse_size": float(impulse_size),
        "atr": float(atr)
    }


# ---------------- PULLBACK CHECK ----------------

def find_pullback(df, impulse):
    start = impulse["end_index"] + 1
    end = min(len(df), start + MAX_PULLBACK_BARS)

    if start >= end:
        return None

    window = df.iloc[start:end]

    high = impulse["impulse_high"]
    low = impulse["impulse_low"]
    size = impulse["impulse_size"]

    if impulse["direction"] == "BULLISH":
        extreme_index = window["low"].idxmin()
        pullback_price = float(df.loc[extreme_index, "low"])
        retracement = (high - pullback_price) / size

    else:
        extreme_index = window["high"].idxmax()
        pullback_price = float(df.loc[extreme_index, "high"])
        retracement = (pullback_price - low) / size

    if not MIN_PULLBACK_RETRACE <= retracement <= MAX_PULLBACK_RETRACE:
        return None

    return {
        "pullback_index": int(extreme_index),
        "pullback_price": pullback_price,
        "retracement": float(retracement)
    }


# ---------------- RETEST CHECK ----------------

def find_retest(df, impulse, pullback):
    start = pullback["pullback_index"] + 1
    end = min(len(df), start + MAX_RETEST_BARS)

    if start >= end:
        return None

    atr = impulse["atr"]
    tolerance = RETEST_TOLERANCE_ATR * atr
    breakout_buffer = BREAKOUT_BUFFER_ATR * atr

    window = df.iloc[start:end]

    if impulse["direction"] == "BULLISH":
        level = impulse["impulse_high"]

        candidates = window[
            (window["high"] >= level - tolerance) &
            (window["close"] <= level + breakout_buffer)
        ]

    else:
        level = impulse["impulse_low"]

        candidates = window[
            (window["low"] <= level + tolerance) &
            (window["close"] >= level - breakout_buffer)
        ]

    if candidates.empty:
        return None

    return int(candidates.index[0])


# ---------------- RANGE CONFIRMATION ----------------

def confirm_range(df, impulse, pullback, retest_index):
    start = retest_index
    max_end = min(
        len(df),
        retest_index + MAX_RANGE_FORMATION_BARS
    )

    upper = impulse["impulse_high"] if impulse["direction"] == "BULLISH" else pullback["pullback_price"]
    lower = pullback["pullback_price"] if impulse["direction"] == "BULLISH" else impulse["impulse_low"]

    width = upper - lower
    atr = impulse["atr"]

    if width <= 0:
        return None

    width_atr = width / atr

    if not MIN_RANGE_ATR <= width_atr <= MAX_RANGE_ATR:
        return None

    tolerance = RETEST_TOLERANCE_ATR * atr
    breakout_buffer = BREAKOUT_BUFFER_ATR * atr

    earliest_end = start + RANGE_CONFIRM_BARS - 1

    for end in range(earliest_end, max_end):
        window = df.iloc[start:end + 1]

        upper_touches = int(
            (window["high"] >= upper - tolerance).sum()
        )

        lower_touches = int(
            (window["low"] <= lower + tolerance).sum()
        )

        inside = (
            (window["close"] <= upper + breakout_buffer) &
            (window["close"] >= lower - breakout_buffer)
        )

        inside_ratio = float(inside.mean())

        if (
            upper_touches >= MIN_TOUCHES_UPPER
            and lower_touches >= MIN_TOUCHES_LOWER
            and inside_ratio >= INSIDE_RATIO_REQUIRED
        ):
            return {
                "confirm_index": end,
                "upper_touches": upper_touches,
                "lower_touches": lower_touches,
                "inside_ratio": inside_ratio,
                "width": width,
                "width_atr": width_atr,
                "upper": upper,
                "lower": lower
            }

    return None


# ---------------- BREAKOUT CHECK ----------------

def find_range_breakout(df, upper, lower, atr, start_index):
    buffer = BREAKOUT_BUFFER_ATR * atr

    for i in range(start_index + 1, len(df)):
        close = float(df.iloc[i]["close"])

        if close > upper + buffer:
            return i, "BREAKOUT"

        if close < lower - buffer:
            return i, "BREAKDOWN"

    return None, "STILL_ACTIVE"


# ---------------- MAIN RANGE DETECTOR ----------------

def detect_ranges(source_df, timeframe):
    df = add_atr(source_df)

    ranges = []
    cursor = max(ATR_PERIOD, IMPULSE_LOOKBACK)

    while cursor < len(df) - MAX_PULLBACK_BARS - MAX_RETEST_BARS - RANGE_CONFIRM_BARS:

        impulse = detect_impulse(df, cursor)

        if impulse is None:
            cursor += 1
            continue

        pullback = find_pullback(df, impulse)

        if pullback is None:
            cursor += 1
            continue

        retest_index = find_retest(df, impulse, pullback)

        if retest_index is None:
            cursor += 1
            continue

        confirmed = confirm_range(
            df, impulse, pullback, retest_index
        )

        if confirmed is None:
            cursor += 1
            continue

        confirm_index = confirmed["confirm_index"]

        breakout_index, status = find_range_breakout(
            df,
            confirmed["upper"],
            confirmed["lower"],
            impulse["atr"],
            confirm_index
        )

        if breakout_index is not None:
            end_index = breakout_index
            end_time = df.iloc[breakout_index]["time"]
        else:
            end_index = len(df) - 1
            end_time = df.iloc[end_index]["time"]

        ranges.append({
            "timeframe": timeframe,
            "direction_before_range": impulse["direction"],

            "impulse_start": df.iloc[impulse["start_index"]]["time"],
            "impulse_end": df.iloc[impulse["end_index"]]["time"],

            "impulse_high": impulse["impulse_high"],
            "impulse_low": impulse["impulse_low"],

            "pullback_time": df.iloc[pullback["pullback_index"]]["time"],
            "pullback_price": pullback["pullback_price"],
            "pullback_retracement": pullback["retracement"],

            "retest_time": df.iloc[retest_index]["time"],

            "range_confirm_time": df.iloc[confirm_index]["time"],
            "range_upper": confirmed["upper"],
            "range_lower": confirmed["lower"],
            "range_width": confirmed["width"],
            "range_width_atr": confirmed["width_atr"],

            "upper_touches": confirmed["upper_touches"],
            "lower_touches": confirmed["lower_touches"],
            "inside_ratio": confirmed["inside_ratio"],

            "range_end_time": end_time,
            "range_status": status
        })

        # Avoid detecting the same structure repeatedly.
        cursor = max(confirm_index + 1, cursor + 1)

        if breakout_index is not None:
            cursor = max(cursor, breakout_index + 1)

    return pd.DataFrame(ranges)


# ---------------- REPORT ----------------

def print_report(name, result):
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Confirmed ranges:", len(result))

    if result.empty:
        print("No ranges detected. Parameters may need adjustment.")
        return

    print("\nRange status:")
    print(result["range_status"].value_counts().to_string())

    print("\nDirection before range:")
    print(result["direction_before_range"].value_counts().to_string())

    print("\nAverage range width:", round(result["range_width"].mean(), 2))
    print("Average width in ATR:", round(result["range_width_atr"].mean(), 2))

    print("\nLast 10 detected ranges:")
    print(
        result[
            [
                "direction_before_range",
                "range_confirm_time",
                "range_upper",
                "range_lower",
                "range_status"
            ]
        ].tail(10).to_string(index=False)
    )


# ---------------- EXECUTION ----------------

def main():
    print("Loading M5 and H1 data...")

    m5 = load_csv(M5_FILE)
    h1 = load_csv(H1_FILE)

    print("M5 candles:", len(m5))
    print("H1 candles:", len(h1))

    print("\nBuilding M15 candles...")
    m15 = make_m15(m5)

    print("M15 candles:", len(m15))

    print("\nDetecting H1 impulse-pullback-retest ranges...")
    h1_ranges = detect_ranges(h1, "H1")

    print("\nDetecting M15 impulse-pullback-retest ranges...")
    m15_ranges = detect_ranges(m15, "M15")

    print_report("H1 RANGE RESULTS", h1_ranges)
    print_report("M15 RANGE RESULTS", m15_ranges)

    h1_ranges.to_csv(
        BASE_DIR / "impulse_ranges_h1.csv", index=False
    )

    m15_ranges.to_csv(
        BASE_DIR / "impulse_ranges_m15.csv", index=False
    )

    all_ranges = pd.concat(
        [h1_ranges, m15_ranges],
        ignore_index=True
    )

    all_ranges.to_csv(
        BASE_DIR / "impulse_ranges_all.csv", index=False
    )

    print("\n========== FILES SAVED ==========")
    print("impulse_ranges_h1.csv")
    print("impulse_ranges_m15.csv")
    print("impulse_ranges_all.csv")

    print("\nIMPORTANT:")
    print("1. This script detects ranges only.")
    print("2. No entry, SL, TP or trading signals are generated.")
    print("3. M15 candles are created from complete M5 candles.")
    print("4. Range is confirmed only after retest and consolidation.")
    print("5. This is an initial rule-based detector; inspect its zones on charts.")
    print("6. It is not a validated profitable strategy.")


if __name__ == "__main__":
    main()
