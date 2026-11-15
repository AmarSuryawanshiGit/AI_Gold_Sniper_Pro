
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# IMPULSE -> PULLBACK -> RETEST -> RANGE DETECTOR V2
# Range detection only. No trade entries or auto trading.
# ============================================================

BASE = Path(__file__).resolve().parent
M5_FILE = BASE / "M5.csv"
H1_FILE = BASE / "H1.csv"

# ---------------- PARAMETERS ----------------

ATR_PERIOD = 14
IMPULSE_LOOKBACK = 8
IMPULSE_ATR_MULTIPLE = 2.0
MIN_DIRECTIONAL_CANDLES = 0.60

MIN_PULLBACK_RETRACE = 0.20
MAX_PULLBACK_RETRACE = 0.75
MAX_PULLBACK_BARS = 12

RETEST_TOLERANCE_ATR = 0.25
MAX_RETEST_BARS = 12

MIN_RANGE_BARS = 8
MAX_RANGE_BARS = 24
MIN_UPPER_SWINGS = 2
MIN_LOWER_SWINGS = 2
MIN_SWING_GAP_BARS = 3

MIN_INSIDE_RATIO = 0.80
MIN_RANGE_ATR = 0.50
MAX_RANGE_ATR = 5.00

BREAKOUT_BUFFER_ATR = 0.15
SWING_LEFT = 2
SWING_RIGHT = 2

# Chart output
SAVE_CHARTS = True
MAX_CHARTS_PER_TIMEFRAME = 20
CHART_BARS_BEFORE = 35
CHART_BARS_AFTER = 35


# ---------------- DATA ----------------

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    if pd.api.types.is_numeric_dtype(df["time"]):
        first = df["time"].dropna().iloc[0]
        unit = "ms" if first > 10**12 else "s"
        df["time"] = pd.to_datetime(
            df["time"], unit=unit, utc=True
        )
    else:
        df["time"] = pd.to_datetime(
            df["time"], errors="coerce", utc=True
        )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
          .sort_values("time")
          .drop_duplicates("time")
          .reset_index(drop=True)
    )
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

    result = x.resample(
        "15min", label="left", closed="left"
    ).agg(agg)

    # Keep only complete M15 candles (3 M5 candles).
    counts = x["close"].resample(
        "15min", label="left", closed="left"
    ).count()

    result = result[counts == 3]
    return result.dropna().reset_index()


def add_atr(df):
    x = df.copy()
    prev_close = x["close"].shift(1)

    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    x["atr"] = tr.rolling(ATR_PERIOD).mean()
    return x


# ---------------- SWING POINTS ----------------

def find_swings(df):
    """
    A swing is counted only after SWING_RIGHT candles have closed.
    This avoids using an unconfirmed pivot.
    """
    x = df.copy()
    x["swing_high"] = False
    x["swing_low"] = False

    for i in range(SWING_LEFT, len(x) - SWING_RIGHT):
        left = x.iloc[i - SWING_LEFT:i]
        right = x.iloc[i + 1:i + 1 + SWING_RIGHT]
        current = x.iloc[i]

        if (
            current["high"] > left["high"].max()
            and current["high"] >= right["high"].max()
        ):
            x.loc[i, "swing_high"] = True

        if (
            current["low"] < left["low"].min()
            and current["low"] <= right["low"].min()
        ):
            x.loc[i, "swing_low"] = True

    return x


def count_distinct_touches(values, level, tolerance, min_gap):
    """
    Count separate swing points near a boundary.
    Nearby consecutive swing points are merged into one touch.
    """
    selected = []

    for index, price in values:
        if abs(price - level) <= tolerance:
            if not selected or index - selected[-1][0] >= min_gap:
                selected.append((index, price))

    return len(selected), selected


# ---------------- IMPULSE ----------------

def detect_impulse(df, end):
    start = end - IMPULSE_LOOKBACK
    if start < 0:
        return None

    window = df.iloc[start:end]
    atr = df.iloc[end - 1]["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    displacement = (
        float(df.iloc[end - 1]["close"])
        - float(df.iloc[start]["open"])
    )

    impulse_high = float(window["high"].max())
    impulse_low = float(window["low"].min())
    impulse_size = impulse_high - impulse_low

    if impulse_size <= 0:
        return None

    if abs(displacement) < IMPULSE_ATR_MULTIPLE * atr:
        return None

    required = int(np.ceil(IMPULSE_LOOKBACK * MIN_DIRECTIONAL_CANDLES))
    bullish = int((window["close"] > window["open"]).sum())
    bearish = int((window["close"] < window["open"]).sum())

    if displacement > 0 and bullish >= required:
        direction = "BULLISH"
    elif displacement < 0 and bearish >= required:
        direction = "BEARISH"
    else:
        return None

    if impulse_size / atr > 8:
        return None

    return {
        "direction": direction,
        "start": start,
        "end": end - 1,
        "high": impulse_high,
        "low": impulse_low,
        "size": impulse_size,
        "atr": float(atr)
    }


# ---------------- PULLBACK ----------------

def find_pullback(df, impulse):
    start = impulse["end"] + 1
    end = min(len(df), start + MAX_PULLBACK_BARS)

    if start >= end:
        return None

    window = df.iloc[start:end]
    size = impulse["size"]

    if impulse["direction"] == "BULLISH":
        idx = int(window["low"].idxmin())
        price = float(df.loc[idx, "low"])
        retracement = (impulse["high"] - price) / size
    else:
        idx = int(window["high"].idxmax())
        price = float(df.loc[idx, "high"])
        retracement = (price - impulse["low"]) / size

    if not MIN_PULLBACK_RETRACE <= retracement <= MAX_PULLBACK_RETRACE:
        return None

    return {
        "index": idx,
        "price": price,
        "retracement": float(retracement)
    }


# ---------------- RETEST ----------------

def find_retest(df, impulse, pullback):
    start = pullback["index"] + 1
    end = min(len(df), start + MAX_RETEST_BARS)

    if start >= end:
        return None

    tol = RETEST_TOLERANCE_ATR * impulse["atr"]
    buffer = BREAKOUT_BUFFER_ATR * impulse["atr"]
    window = df.iloc[start:end]

    if impulse["direction"] == "BULLISH":
        level = impulse["high"]
        mask = (
            (window["high"] >= level - tol)
            & (window["close"] <= level + buffer)
        )
    else:
        level = impulse["low"]
        mask = (
            (window["low"] <= level + tol)
            & (window["close"] >= level - buffer)
        )

    matches = window[mask]
    if matches.empty:
        return None

    return int(matches.index[0])


# ---------------- RANGE CONFIRMATION ----------------

def confirm_range(df, impulse, pullback, retest_idx):
    if impulse["direction"] == "BULLISH":
        upper = impulse["high"]
        lower = pullback["price"]
    else:
        upper = pullback["price"]
        lower = impulse["low"]

    width = upper - lower
    atr = impulse["atr"]

    if width <= 0:
        return None

    width_atr = width / atr
    if not MIN_RANGE_ATR <= width_atr <= MAX_RANGE_ATR:
        return None

    tolerance = RETEST_TOLERANCE_ATR * atr
    breakout_buffer = BREAKOUT_BUFFER_ATR * atr

    max_end = min(len(df), retest_idx + MAX_RANGE_BARS)

    # Test each possible consolidation window.
    for end in range(
        retest_idx + MIN_RANGE_BARS - 1,
        max_end
    ):
        window = df.iloc[retest_idx:end + 1]

        # A close outside the range invalidates this candidate.
        close_inside = (
            (window["close"] <= upper + breakout_buffer)
            & (window["close"] >= lower - breakout_buffer)
        )

        if not close_inside.all():
            continue

        # Use confirmed swing points only.
        upper_values = [
            (i, float(df.iloc[i]["high"]))
            for i in range(retest_idx, end + 1)
            if bool(df.iloc[i]["swing_high"])
        ]

        lower_values = [
            (i, float(df.iloc[i]["low"]))
            for i in range(retest_idx, end + 1)
            if bool(df.iloc[i]["swing_low"])
        ]

        upper_count, upper_points = count_distinct_touches(
            upper_values, upper, tolerance, MIN_SWING_GAP_BARS
        )
        lower_count, lower_points = count_distinct_touches(
            lower_values, lower, tolerance, MIN_SWING_GAP_BARS
        )

        inside_ratio = float(close_inside.mean())

        if (
            upper_count >= MIN_UPPER_SWINGS
            and lower_count >= MIN_LOWER_SWINGS
            and inside_ratio >= MIN_INSIDE_RATIO
        ):
            return {
                "confirm_index": end,
                "upper": upper,
                "lower": lower,
                "width": width,
                "width_atr": width_atr,
                "upper_touches": upper_count,
                "lower_touches": lower_count,
                "inside_ratio": inside_ratio,
                "upper_points": upper_points,
                "lower_points": lower_points
            }

    return None


# ---------------- RANGE LIFE-CYCLE ----------------

def classify_range(df, confirmed, atr):
    """
    Starts checking only AFTER confirmation.
    First close outside the buffered boundary defines the break.
    """
    upper = confirmed["upper"]
    lower = confirmed["lower"]
    start = confirmed["confirm_index"]

    buffer = BREAKOUT_BUFFER_ATR * atr

    for i in range(start + 1, len(df)):
        close = float(df.iloc[i]["close"])

        if close > upper + buffer:
            return i, "BREAKOUT"

        if close < lower - buffer:
            return i, "BREAKDOWN"

    return len(df) - 1, "ACTIVE"


# ---------------- DETECTOR ----------------

def detect_ranges(source, timeframe):
    df = add_atr(source)
    df = find_swings(df)

    results = []
    cursor = max(ATR_PERIOD + IMPULSE_LOOKBACK, SWING_LEFT + 1)

    while cursor < len(df) - (
        MAX_PULLBACK_BARS + MAX_RETEST_BARS + MIN_RANGE_BARS
    ):
        impulse = detect_impulse(df, cursor)

        if impulse is None:
            cursor += 1
            continue

        pullback = find_pullback(df, impulse)
        if pullback is None:
            cursor += 1
            continue

        retest_idx = find_retest(df, impulse, pullback)
        if retest_idx is None:
            cursor += 1
            continue

        confirmed = confirm_range(
            df, impulse, pullback, retest_idx
        )
        if confirmed is None:
            cursor += 1
            continue

        end_idx, status = classify_range(
            df, confirmed, impulse["atr"]
        )

        results.append({
            "timeframe": timeframe,
            "direction_before_range": impulse["direction"],

            "impulse_start_time": df.iloc[impulse["start"]]["time"],
            "impulse_end_time": df.iloc[impulse["end"]]["time"],
            "impulse_high": impulse["high"],
            "impulse_low": impulse["low"],

            "pullback_time": df.iloc[pullback["index"]]["time"],
            "pullback_price": pullback["price"],
            "pullback_retracement": pullback["retracement"],

            "retest_time": df.iloc[retest_idx]["time"],
            "range_confirm_time": df.iloc[confirmed["confirm_index"]]["time"],

            "range_upper": confirmed["upper"],
            "range_lower": confirmed["lower"],
            "range_width": confirmed["width"],
            "range_width_atr": confirmed["width_atr"],

            "upper_swing_touches": confirmed["upper_touches"],
            "lower_swing_touches": confirmed["lower_touches"],
            "inside_ratio": confirmed["inside_ratio"],

            "range_end_time": df.iloc[end_idx]["time"],
            "range_status": status,

            # Internal indices used for chart generation.
            "_impulse_start": impulse["start"],
            "_impulse_end": impulse["end"],
            "_pullback_index": pullback["index"],
            "_retest_index": retest_idx,
            "_confirm_index": confirmed["confirm_index"],
            "_end_index": end_idx
        })

        # Avoid repeatedly detecting the same structure.
        cursor = max(confirmed["confirm_index"] + 1, cursor + 1)

    return df, pd.DataFrame(results)


# ---------------- CHARTS ----------------

def save_range_charts(df, ranges, timeframe):
    if not SAVE_CHARTS or ranges.empty:
        return

    folder = BASE / f"range_charts_{timeframe.lower()}"
    folder.mkdir(exist_ok=True)

    selected = ranges.tail(MAX_CHARTS_PER_TIMEFRAME)

    for number, (_, r) in enumerate(selected.iterrows(), 1):
        confirm_idx = int(r["_confirm_index"])
        end_idx = int(r["_end_index"])

        left = max(0, int(r["_impulse_start"]) - CHART_BARS_BEFORE)
        right = min(len(df) - 1, end_idx + CHART_BARS_AFTER)

        view = df.iloc[left:right + 1].copy()
        x = np.arange(len(view))

        fig, ax = plt.subplots(figsize=(15, 7))

        # Candlestick bodies and wicks.
        for j, (_, candle) in enumerate(view.iterrows()):
            up = candle["close"] >= candle["open"]
            color = "green" if up else "red"

            ax.vlines(
                j, candle["low"], candle["high"],
                color=color, linewidth=1
            )

            body_bottom = min(candle["open"], candle["close"])
            body_height = abs(candle["close"] - candle["open"])
            body_height = max(body_height, 0.01)

            ax.bar(
                j, body_height, bottom=body_bottom,
                width=0.65, color=color, alpha=0.75
            )

        def local_x(index):
            return index - left

        ax.axhline(
            r["range_upper"], color="blue",
            linestyle="--", linewidth=1.5, label="Range upper"
        )
        ax.axhline(
            r["range_lower"], color="purple",
            linestyle="--", linewidth=1.5, label="Range lower"
        )

        ax.axvspan(
            local_x(confirm_idx),
            local_x(end_idx),
            alpha=0.10, color="orange",
            label="Confirmed range period"
        )

        markers = [
            ("Impulse start", int(r["_impulse_start"]), "black"),
            ("Impulse end", int(r["_impulse_end"]), "green"),
            ("Pullback", int(r["_pullback_index"]), "orange"),
            ("Retest", int(r["_retest_index"]), "purple"),
            ("Range confirmed", confirm_idx, "blue"),
            ("Range end/break", end_idx, "red")
        ]

        for label, idx, color in markers:
            xpos = local_x(idx)
            if 0 <= xpos < len(view):
                ax.axvline(
                    xpos, color=color, linestyle=":",
                    alpha=0.8, label=label
                )

        tick_step = max(1, len(view) // 10)
        ticks = list(range(0, len(view), tick_step))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [view.iloc[i]["time"].strftime("%m-%d %H:%M") for i in ticks],
            rotation=35, ha="right"
        )

        ax.set_title(
            f"{timeframe} | {r['direction_before_range']} | "
            f"{r['range_status']} | "
            f"Upper {r['range_upper']:.2f} / Lower {r['range_lower']:.2f}"
        )
        ax.set_ylabel("Gold price")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        filename = folder / f"{timeframe.lower()}_range_{number:02d}.png"
        fig.savefig(filename, dpi=120)
        plt.close(fig)


# ---------------- REPORT ----------------

def report(name, ranges):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Confirmed ranges:", len(ranges))

    if ranges.empty:
        print("No ranges found.")
        return

    print("\nRange status:")
    print(ranges["range_status"].value_counts().to_string())

    print("\nDirection before range:")
    print(
        ranges["direction_before_range"].value_counts().to_string()
    )

    print("\nAverage range width:",
          round(ranges["range_width"].mean(), 2))
    print("Average width in ATR:",
          round(ranges["range_width_atr"].mean(), 2))

    print("\nLast 10 ranges:")
    cols = [
        "direction_before_range",
        "range_confirm_time",
        "range_upper",
        "range_lower",
        "upper_swing_touches",
        "lower_swing_touches",
        "range_status"
    ]
    print(ranges[cols].tail(10).to_string(index=False))


# ---------------- MAIN ----------------

def main():
    print("Loading M5 and H1 data...")
    m5 = load_csv(M5_FILE)
    h1 = load_csv(H1_FILE)

    print("M5 candles:", len(m5))
    print("H1 candles:", len(h1))

    print("\nBuilding M15 candles...")
    m15 = make_m15(m5)
    print("M15 candles:", len(m15))

    print("\nDetecting H1 ranges...")
    h1_df, h1_ranges = detect_ranges(h1, "H1")

    print("\nDetecting M15 ranges...")
    m15_df, m15_ranges = detect_ranges(m15, "M15")

    # Remove internal chart-only columns from CSV.
    internal = [c for c in h1_ranges.columns if c.startswith("_")]
    h1_export = h1_ranges.drop(columns=internal, errors="ignore")
    m15_export = m15_ranges.drop(columns=internal, errors="ignore")

    h1_export.to_csv(BASE / "impulse_ranges_h1_v2.csv", index=False)
    m15_export.to_csv(BASE / "impulse_ranges_m15_v2.csv", index=False)

    all_export = pd.concat(
        [h1_export, m15_export], ignore_index=True
    )
    all_export.to_csv(
        BASE / "impulse_ranges_all_v2.csv", index=False
    )

    report("H1 RANGE RESULTS V2", h1_ranges)
    report("M15 RANGE RESULTS V2", m15_ranges)

    print("\nSaving chart images...")
    save_range_charts(h1_df, h1_ranges, "H1")
    save_range_charts(m15_df, m15_ranges, "M15")

    print("\n========== FILES SAVED ==========")
    print("impulse_ranges_h1_v2.csv")
    print("impulse_ranges_m15_v2.csv")
    print("impulse_ranges_all_v2.csv")
    print("range_charts_h1 folder")
    print("range_charts_m15 folder")

    print("\nIMPORTANT:")
    print("1. No BUY/SELL signals are generated.")
    print("2. Swing touches are separated by minimum bar distance.")
    print("3. Breakout is checked only after range confirmation.")
    print("4. ACTIVE means no confirmed close outside the range yet.")
    print("5. This is a research detector, not a validated strategy.")


if __name__ == "__main__":
    main()
