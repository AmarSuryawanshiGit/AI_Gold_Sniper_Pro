
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent

ATR_PERIOD = 14
SWING_LEFT = 2
SWING_RIGHT = 2
BREAK_BUFFER_ATR = 0.15
FAILURE_BARS = 3

SETTINGS = {
    "H1": {
        "min_bars": 10,
        "max_bars": 36,
        "min_touches": 3,
        "touch_gap": 2,
        "touch_atr": 0.30,
        "min_width_atr": 0.35,
        "max_width_atr": 5.5,
        "inside_ratio": 0.65
    },
    "M15": {
        "min_bars": 10,
        "max_bars": 42,
        "min_touches": 3,
        "touch_gap": 2,
        "touch_atr": 0.35,
        "min_width_atr": 0.30,
        "max_width_atr": 6.0,
        "inside_ratio": 0.60
    }
}


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

def load_data(filename):
    path = BASE / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = ["time", "open", "high", "low", "close"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["time"] = pd.to_datetime(
        df["time"], utc=True, errors="coerce"
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


# --------------------------------------------------
# 2. BUILD M15 FROM COMPLETE M5 CANDLES
# --------------------------------------------------

def build_m15(m5):
    x = m5.set_index("time")

    grouped = x.resample(
        "15min", label="left", closed="left"
    )

    candles = grouped.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    counts = grouped["close"].count()

    candles = candles[counts == 3].dropna()

    return candles.reset_index()


# --------------------------------------------------
# 3. ATR
# --------------------------------------------------

def add_atr(df):
    df = df.copy()

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    return df


# --------------------------------------------------
# 4. CAUSALLY CONFIRMED SWINGS
# --------------------------------------------------

def add_swings(df):
    df = df.copy()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing_high = np.full(len(df), np.nan)
    swing_low = np.full(len(df), np.nan)

    for pivot in range(
        SWING_LEFT,
        len(df) - SWING_RIGHT
    ):
        left = pivot - SWING_LEFT
        right = pivot + SWING_RIGHT + 1

        h = highs[left:right]
        l = lows[left:right]

        if (
            highs[pivot] == np.max(h)
            and np.sum(h == highs[pivot]) == 1
        ):
            # Store swing at confirmation candle.
            swing_high[pivot + SWING_RIGHT] = highs[pivot]

        if (
            lows[pivot] == np.min(l)
            and np.sum(l == lows[pivot]) == 1
        ):
            swing_low[pivot + SWING_RIGHT] = lows[pivot]

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low

    return df


# --------------------------------------------------
# 5. FAST REGRESSION TRENDLINE
# --------------------------------------------------

def fit_line(pivots, tolerance, min_touches, gap):
    """
    Regression-based line fitting.
    No repeated pair-combination search.
    """

    if len(pivots) < min_touches:
        return None

    x = np.asarray([p[0] for p in pivots], dtype=float)
    y = np.asarray([p[1] for p in pivots], dtype=float)

    # Initial linear regression.
    x_mean = x.mean()
    y_mean = y.mean()

    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return None

    slope = np.sum(
        (x - x_mean) * (y - y_mean)
    ) / denominator

    intercept = y_mean - slope * x_mean

    # Identify points close to the initial line.
    errors = np.abs(y - (slope * x + intercept))
    inliers = errors <= tolerance

    if inliers.sum() < min_touches:
        return None

    # Refine regression using inliers.
    xi = x[inliers]
    yi = y[inliers]

    xm = xi.mean()
    ym = yi.mean()

    den = np.sum((xi - xm) ** 2)

    if den == 0:
        return None

    slope = np.sum((xi - xm) * (yi - ym)) / den
    intercept = ym - slope * xm

    errors = np.abs(y - (slope * x + intercept))
    candidates = np.where(errors <= tolerance)[0]

    # Count separate touches, not consecutive candles.
    selected = []

    for idx in candidates:
        if (
            not selected
            or x[idx] - x[selected[-1]] >= gap
        ):
            selected.append(int(idx))

    if len(selected) < min_touches:
        return None

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "touches": len(selected),
        "indices": [int(x[i]) for i in selected]
    }


def line_at(line, index):
    return line["slope"] * index + line["intercept"]


# --------------------------------------------------
# 6. RANGE SHAPE
# --------------------------------------------------

def get_shape(upper, lower, length, atr):
    up_move = upper["slope"] * length / atr
    low_move = lower["slope"] * length / atr

    if abs(up_move) < 0.35 and abs(low_move) < 0.35:
        return "HORIZONTAL"

    if up_move > 0.25 and low_move > 0.25:
        return "ASCENDING_CHANNEL"

    if up_move < -0.25 and low_move < -0.25:
        return "DESCENDING_CHANNEL"

    if up_move < -0.20 and low_move > 0.20:
        return "CONVERGING_TRIANGLE"

    if up_move > 0.20 and low_move < -0.20:
        return "DIVERGING_WEDGE"

    return "SLOPING_OR_MIXED"


# --------------------------------------------------
# 7. BREAKOUT / BREAKDOWN / FAILED
# --------------------------------------------------

def classify_status(df, confirm, upper, lower, atr):
    buffer = BREAK_BUFFER_ATR * atr

    first_break = None
    direction = None

    for i in range(confirm + 1, len(df)):
        top = line_at(upper, i) + buffer
        bottom = line_at(lower, i) - buffer
        close = df.iloc[i]["close"]

        if close > top:
            first_break = i
            direction = "UP"
            break

        if close < bottom:
            first_break = i
            direction = "DOWN"
            break

    if first_break is None:
        return "ACTIVE", None, None

    # Check whether price returns inside within failure window.
    check_end = min(
        len(df),
        first_break + FAILURE_BARS + 1
    )

    for i in range(first_break + 1, check_end):
        top = line_at(upper, i) + buffer
        bottom = line_at(lower, i) - buffer
        close = df.iloc[i]["close"]

        if bottom <= close <= top:
            return (
                "FAILED",
                direction,
                df.iloc[first_break]["time"]
            )

    # If dataset ends before failure window is complete,
    # do not prematurely call it a confirmed breakout.
    if len(df) <= first_break + FAILURE_BARS:
        return (
            "PENDING_BREAK",
            direction,
            df.iloc[first_break]["time"]
        )

    status = "BREAKOUT" if direction == "UP" else "BREAKDOWN"

    return (
        status,
        direction,
        df.iloc[first_break]["time"]
    )


# --------------------------------------------------
# 8. DETECT RANGES
# --------------------------------------------------

def detect_ranges(source, timeframe):
    cfg = SETTINGS[timeframe]

    df = add_swings(add_atr(source))

    high_swings = [
        (i, value)
        for i, value in enumerate(df["swing_high"].to_numpy())
        if np.isfinite(value)
    ]

    low_swings = [
        (i, value)
        for i, value in enumerate(df["swing_low"].to_numpy())
        if np.isfinite(value)
    ]

    high_indices = np.array(
        [p[0] for p in high_swings], dtype=int
    )

    low_indices = np.array(
        [p[0] for p in low_swings], dtype=int
    )

    results = []

    total = len(df)

    print(
        f"{timeframe}: scanning {total} candles...",
        flush=True
    )

    for end in range(
        ATR_PERIOD + cfg["min_bars"],
        total
    ):
        atr = df.iloc[end]["atr"]

        if not np.isfinite(atr) or atr <= 0:
            continue

        found = None

        # Try longest range first.
        for length in range(
            cfg["max_bars"],
            cfg["min_bars"] - 1,
            -1
        ):
            start = end - length + 1

            # Fast swing lookup using precomputed arrays.
            hi_left = np.searchsorted(high_indices, start)
            hi_right = np.searchsorted(high_indices, end + 1)

            lo_left = np.searchsorted(low_indices, start)
            lo_right = np.searchsorted(low_indices, end + 1)

            hp = high_swings[hi_left:hi_right]
            lp = low_swings[lo_left:lo_right]

            section = df.iloc[start:end + 1]

            avg_atr = section["atr"].median()

            if not np.isfinite(avg_atr) or avg_atr <= 0:
                continue

            tolerance = avg_atr * cfg["touch_atr"]

            upper = fit_line(
                hp, tolerance,
                cfg["min_touches"], cfg["touch_gap"]
            )

            if upper is None:
                continue

            lower = fit_line(
                lp, tolerance,
                cfg["min_touches"], cfg["touch_gap"]
            )

            if lower is None:
                continue

            top_start = line_at(upper, start)
            top_end = line_at(upper, end)

            bottom_start = line_at(lower, start)
            bottom_end = line_at(lower, end)

            width_start = top_start - bottom_start
            width_end = top_end - bottom_end

            if min(width_start, width_end) <= 0:
                continue

            avg_width = (width_start + width_end) / 2
            width_atr = avg_width / avg_atr

            if not (
                cfg["min_width_atr"]
                <= width_atr
                <= cfg["max_width_atr"]
            ):
                continue

            # Avoid excessively steep or distorted boundaries.
            if abs(upper["slope"] * length / avg_atr) > 1.5:
                continue

            if abs(lower["slope"] * length / avg_atr) > 1.5:
                continue

            x = np.arange(start, end + 1)

            top_line = (
                upper["slope"] * x + upper["intercept"]
            )

            bottom_line = (
                lower["slope"] * x + lower["intercept"]
            )

            closes = section["close"].to_numpy()

            inside = np.mean(
                (closes <= top_line + tolerance)
                & (closes >= bottom_line - tolerance)
            )

            if inside < cfg["inside_ratio"]:
                continue

            shape = get_shape(
                upper, lower, length, avg_atr
            )

            status, direction, break_time = classify_status(
                df, end, upper, lower, avg_atr
            )

            found = {
                "timeframe": timeframe,
                "range_start_time": df.iloc[start]["time"],
                "range_confirm_time": df.iloc[end]["time"],
                "range_start_index": start,
                "range_confirm_index": end,
                "range_type": shape,
                "range_upper": top_end,
                "range_lower": bottom_end,
                "upper_slope": upper["slope"],
                "lower_slope": lower["slope"],
                "upper_touches": upper["touches"],
                "lower_touches": lower["touches"],
                "range_width": avg_width,
                "range_width_atr": width_atr,
                "inside_ratio": inside,
                "range_status": status,
                "break_direction": direction,
                "first_break_time": break_time
            }

            break

        if found:
            results.append(found)

    if not results:
        return df, pd.DataFrame()

    result = pd.DataFrame(results)

    # Remove duplicate detections of substantially
    # overlapping ranges of the same shape.
    kept = []

    for _, row in result.iterrows():
        duplicate = False

        for old in kept[-20:]:
            a0 = int(row["range_start_index"])
            a1 = int(row["range_confirm_index"])

            b0 = int(old["range_start_index"])
            b1 = int(old["range_confirm_index"])

            overlap = max(
                0,
                min(a1, b1) - max(a0, b0) + 1
            )

            shorter = max(
                1,
                min(a1 - a0 + 1, b1 - b0 + 1)
            )

            if (
                overlap / shorter >= 0.55
                and row["range_type"] == old["range_type"]
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(row)

    return df, pd.DataFrame(kept).reset_index(drop=True)


# --------------------------------------------------
# 9. SAVE CHARTS
# --------------------------------------------------

def save_charts(df, ranges, timeframe):
    folder = BASE / f"range_charts_{timeframe.lower()}_v5"
    folder.mkdir(exist_ok=True)

    for number, (_, r) in enumerate(
        ranges.tail(30).iterrows(), start=1
    ):
        start = int(r["range_start_index"])
        confirm = int(r["range_confirm_index"])

        end = min(len(df) - 1, confirm + 50)
        left = max(0, start - 20)

        fig, ax = plt.subplots(figsize=(15, 6))

        for i in range(left, end + 1):
            candle = df.iloc[i]

            color = (
                "green"
                if candle["close"] >= candle["open"]
                else "red"
            )

            ax.vlines(
                i,
                candle["low"],
                candle["high"],
                color=color,
                linewidth=0.8
            )

            body = max(
                abs(candle["close"] - candle["open"]),
                0.01
            )

            ax.bar(
                i,
                body,
                bottom=min(candle["open"], candle["close"]),
                color=color,
                width=0.6
            )

        x = np.arange(start, end + 1)

        upper = (
            r["upper_slope"] * x
            + r["range_upper"]
            - r["upper_slope"] * confirm
        )

        lower = (
            r["lower_slope"] * x
            + r["range_lower"]
            - r["lower_slope"] * confirm
        )

        ax.plot(x, upper, "b--", label="Upper resistance")
        ax.plot(x, lower, "m--", label="Lower support")

        ax.axvspan(
            start, confirm,
            color="orange", alpha=0.12,
            label="Detected range"
        )

        ax.axvline(
            confirm,
            color="black",
            linestyle=":",
            label="Confirmation"
        )

        ax.set_title(
            f"{timeframe} | {r['range_type']} | "
            f"{r['range_status']} | "
            f"Upper {r['range_upper']:.2f} / "
            f"Lower {r['range_lower']:.2f}"
        )

        ticks = np.linspace(
            left, end,
            min(9, end - left + 1),
            dtype=int
        )

        ax.set_xticks(ticks)

        ax.set_xticklabels(
            [
                df.iloc[i]["time"].strftime("%m-%d %H:%M")
                for i in ticks
            ],
            rotation=35,
            ha="right"
        )

        ax.set_ylabel("Gold price")
        ax.grid(alpha=0.2)
        ax.legend()

        fig.tight_layout()

        stamp = pd.Timestamp(
            r["range_confirm_time"]
        ).strftime("%Y%m%d_%H%M")

        fig.savefig(
            folder / f"{timeframe}_{number}_{stamp}.png",
            dpi=110
        )

        plt.close(fig)

    return folder


# --------------------------------------------------
# 10. RESULTS
# --------------------------------------------------

def show_results(name, ranges):
    print("\n" + "=" * 55)
    print(name + " RANGE RESULTS V5")
    print("=" * 55)

    print("Confirmed ranges:", len(ranges))

    if ranges.empty:
        print("No ranges detected.")
        return

    print("\nRange status:")
    print(ranges["range_status"].value_counts())

    print("\nRange types:")
    print(ranges["range_type"].value_counts())

    print(
        "\nAverage width:",
        round(ranges["range_width"].mean(), 2)
    )

    print(
        "Average width ATR:",
        round(ranges["range_width_atr"].mean(), 2)
    )

    columns = [
        "range_confirm_time",
        "range_type",
        "range_upper",
        "range_lower",
        "upper_touches",
        "lower_touches",
        "range_status",
        "break_direction"
    ]

    print("\nLast 15 ranges:")
    print(
        ranges[columns].tail(15).to_string(index=False)
    )


# --------------------------------------------------
# 11. MAIN
# --------------------------------------------------

def main():
    print("Loading M5 and H1 data...", flush=True)

    m5 = load_data("M5.csv")
    h1 = load_data("H1.csv")

    print("M5 candles:", len(m5))
    print("H1 candles:", len(h1))

    print("\nBuilding M15...", flush=True)

    m15 = build_m15(m5)

    print("M15 candles:", len(m15))

    print("\nDetecting H1 ranges...", flush=True)

    h1_df, h1_ranges = detect_ranges(h1, "H1")

    show_results("H1", h1_ranges)

    print("\nDetecting M15 ranges...", flush=True)

    m15_df, m15_ranges = detect_ranges(m15, "M15")

    show_results("M15", m15_ranges)

    h1_ranges.to_csv(
        BASE / "impulse_ranges_h1_v5.csv",
        index=False
    )

    m15_ranges.to_csv(
        BASE / "impulse_ranges_m15_v5.csv",
        index=False
    )

    pd.concat(
        [h1_ranges, m15_ranges],
        ignore_index=True
    ).to_csv(
        BASE / "impulse_ranges_all_v5.csv",
        index=False
    )

    print("\nSaving charts...", flush=True)

    save_charts(h1_df, h1_ranges, "H1")
    save_charts(m15_df, m15_ranges, "M15")

    print("\n========== FILES SAVED ==========")
    print("impulse_ranges_h1_v5.csv")
    print("impulse_ranges_m15_v5.csv")
    print("impulse_ranges_all_v5.csv")
    print("range_charts_h1_v5/")
    print("range_charts_m15_v5/")

    print("\nNo BUY/SELL signals generated.")
    print("Horizontal and sloping ranges supported.")
    print("Confirmed swing timing enabled.")
    print("Research only - not validated for live trading.")


if __name__ == "__main__":
    main()
