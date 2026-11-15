
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent

# Independent H1 and M15 settings
SETTINGS = {
    "H1": {
        "min_bars": 10, "max_bars": 36,
        "min_touches": 3, "touch_gap": 2,
        "touch_atr": 0.30, "min_width": 0.35,
        "max_width": 5.5, "inside_ratio": 0.65
    },
    "M15": {
        "min_bars": 10, "max_bars": 42,
        "min_touches": 3, "touch_gap": 2,
        "touch_atr": 0.35, "min_width": 0.30,
        "max_width": 6.0, "inside_ratio": 0.60
    }
}

ATR_PERIOD = 14
SWING_LEFT = 2
SWING_RIGHT = 2
BREAK_BUFFER = 0.15
FAILURE_BARS = 3


def load_data(filename):
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found in {BASE}")

    df = pd.read_csv(path)

    required = ["time", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=required)
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


def build_m15(m5):
    x = m5.set_index("time")

    candles = x.resample(
        "15min", label="left", closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    counts = x["close"].resample(
        "15min", label="left", closed="left"
    ).count()

    return candles[counts == 3].dropna().reset_index()


def add_atr(df):
    df = df.copy()

    previous = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous).abs(),
        (df["low"] - previous).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df


def add_confirmed_swings(df):
    """
    Swing is stored at confirmation candle, NOT at pivot candle.
    Therefore future candles are not used before confirmation.
    """
    df = df.copy()

    df["swing_high"] = np.nan
    df["swing_low"] = np.nan

    for pivot in range(
        SWING_LEFT,
        len(df) - SWING_RIGHT
    ):
        confirm = pivot + SWING_RIGHT

        high_window = df["high"].iloc[
            pivot-SWING_LEFT:pivot+SWING_RIGHT+1
        ]

        low_window = df["low"].iloc[
            pivot-SWING_LEFT:pivot+SWING_RIGHT+1
        ]

        if (
            df.iloc[pivot]["high"] == high_window.max()
            and (high_window == high_window.max()).sum() == 1
        ):
            df.loc[confirm, "swing_high"] = df.iloc[pivot]["high"]

        if (
            df.iloc[pivot]["low"] == low_window.min()
            and (low_window == low_window.min()).sum() == 1
        ):
            df.loc[confirm, "swing_low"] = df.iloc[pivot]["low"]

    return df


def get_separated_touches(values, line, tolerance, gap):
    hits = np.where(np.abs(values - line) <= tolerance)[0]
    selected = []

    for idx in hits:
        if not selected or idx - selected[-1] >= gap:
            selected.append(int(idx))
        elif abs(values[idx] - line[idx]) < abs(
            values[selected[-1]] - line[selected[-1]]
        ):
            selected[-1] = int(idx)

    return selected


def fit_trendline(pivots, tolerance, min_touches, gap):
    """
    Find a trendline supported by at least min_touches
    independently spaced confirmed swing points.
    """
    if len(pivots) < min_touches:
        return None

    x = np.array([p[0] for p in pivots], dtype=float)
    y = np.array([p[1] for p in pivots], dtype=float)

    best = None

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[j] == x[i]:
                continue

            slope = (y[j] - y[i]) / (x[j] - x[i])
            intercept = y[i] - slope * x[i]

            predicted = slope * x + intercept
            errors = np.abs(y - predicted)

            hits = np.where(errors <= tolerance)[0]
            selected = []

            for h in hits:
                if not selected or x[h] - x[selected[-1]] >= gap:
                    selected.append(int(h))

            if len(selected) < min_touches:
                continue

            score = (len(selected), -np.median(errors[selected]))

            if best is None or score > best[0]:
                best = (score, selected, slope, intercept)

    if best is None:
        return None

    _, selected, _, _ = best

    # Refine using only the selected confirmed swing points.
    slope, intercept = np.polyfit(x[selected], y[selected], 1)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "touches": len(selected),
        "indices": [int(x[i]) for i in selected]
    }


def line_at(line, index):
    return line["slope"] * index + line["intercept"]


def get_shape(upper, lower, bars, atr):
    up_move = upper["slope"] * bars / atr
    low_move = lower["slope"] * bars / atr

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


def classify_status(df, confirm, upper, lower, atr):
    """
    First close outside range is checked after confirmation.
    Return inside within failure window = FAILED.
    """
    first_break = None
    direction = None

    buffer = BREAK_BUFFER * atr

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

    end = min(
        len(df),
        first_break + FAILURE_BARS + 1
    )

    for i in range(first_break + 1, end):
        top = line_at(upper, i) + buffer
        bottom = line_at(lower, i) - buffer
        close = df.iloc[i]["close"]

        if bottom <= close <= top:
            return (
                "FAILED",
                direction,
                df.iloc[first_break]["time"]
            )

    status = "BREAKOUT" if direction == "UP" else "BREAKDOWN"

    return (
        status,
        direction,
        df.iloc[first_break]["time"]
    )


def detect_ranges(source, timeframe):
    cfg = SETTINGS[timeframe]

    df = add_confirmed_swings(add_atr(source))
    results = []

    for end in range(
        ATR_PERIOD + cfg["max_bars"],
        len(df)
    ):
        atr = df.iloc[end]["atr"]

        if not np.isfinite(atr) or atr <= 0:
            continue

        found = None

        for length in range(
            cfg["max_bars"],
            cfg["min_bars"] - 1,
            -1
        ):
            start = end - length + 1

            section = df.iloc[start:end+1]

            upper_pivots = []
            lower_pivots = []

            for idx in range(start, end + 1):
                h = df.iloc[idx]["swing_high"]
                l = df.iloc[idx]["swing_low"]

                if np.isfinite(h):
                    upper_pivots.append((idx, h))

                if np.isfinite(l):
                    lower_pivots.append((idx, l))

            tolerance = section["atr"].median() * cfg["touch_atr"]

            upper = fit_trendline(
                upper_pivots, tolerance,
                cfg["min_touches"], cfg["touch_gap"]
            )

            lower = fit_trendline(
                lower_pivots, tolerance,
                cfg["min_touches"], cfg["touch_gap"]
            )

            if upper is None or lower is None:
                continue

            top_start = line_at(upper, start)
            top_end = line_at(upper, end)
            bottom_start = line_at(lower, start)
            bottom_end = line_at(lower, end)

            widths = [
                top_start - bottom_start,
                top_end - bottom_end
            ]

            if min(widths) <= 0:
                continue

            avg_width = np.mean(widths)
            avg_atr = section["atr"].median()
            width_atr = avg_width / avg_atr

            if not (
                cfg["min_width"] <= width_atr <= cfg["max_width"]
            ):
                continue

            # Boundary movement must remain within reasonable limits.
            if abs(upper["slope"] * length / avg_atr) > 1.5:
                continue

            if abs(lower["slope"] * length / avg_atr) > 1.5:
                continue

            x = np.arange(start, end + 1)
            top_line = upper["slope"] * x + upper["intercept"]
            bottom_line = lower["slope"] * x + lower["intercept"]

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

    # Remove heavily overlapping detections of the same structure.
    kept = []

    for _, row in result.iterrows():
        duplicate = False

        for old in kept[-15:]:
            a0, a1 = row["range_start_index"], row["range_confirm_index"]
            b0, b1 = old["range_start_index"], old["range_confirm_index"]

            overlap = max(
                0, min(a1, b1) - max(a0, b0) + 1
            )

            shorter = max(
                1, min(a1-a0+1, b1-b0+1)
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


def save_charts(df, ranges, timeframe):
    folder = BASE / f"range_charts_{timeframe.lower()}_v4"
    folder.mkdir(exist_ok=True)

    for _, r in ranges.tail(30).iterrows():
        start = int(r["range_start_index"])
        confirm = int(r["range_confirm_index"])
        end = min(len(df)-1, confirm + 50)
        left = max(0, start - 20)

        fig, ax = plt.subplots(figsize=(15, 6))

        for i in range(left, end + 1):
            candle = df.iloc[i]
            color = "green" if candle["close"] >= candle["open"] else "red"

            ax.vlines(
                i, candle["low"], candle["high"],
                color=color, linewidth=0.8
            )

            ax.bar(
                i,
                max(abs(candle["close"] - candle["open"]), 0.01),
                bottom=min(candle["open"], candle["close"]),
                color=color, width=0.6
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

        ax.plot(x, upper, "b--", label="Upper trendline")
        ax.plot(x, lower, "m--", label="Lower trendline")
        ax.axvspan(start, confirm, color="orange", alpha=0.12)
        ax.axvline(confirm, color="black", linestyle=":")

        ax.set_title(
            f"{timeframe} | {r['range_type']} | "
            f"{r['range_status']} | "
            f"Upper {r['range_upper']:.2f} / "
            f"Lower {r['range_lower']:.2f}"
        )

        ticks = np.linspace(left, end, min(9, end-left+1), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [df.iloc[i]["time"].strftime("%m-%d %H:%M") for i in ticks],
            rotation=35, ha="right"
        )

        ax.set_ylabel("Gold price")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()

        stamp = pd.Timestamp(r["range_confirm_time"]).strftime("%Y%m%d_%H%M")
        fig.savefig(folder / f"{timeframe}_{stamp}.png", dpi=120)
        plt.close(fig)

    return folder


def show_results(name, ranges):
    print("\n" + "=" * 55)
    print(name, "RANGE RESULTS V4")
    print("=" * 55)

    print("Confirmed ranges:", len(ranges))

    if ranges.empty:
        print("No ranges detected.")
        return

    print("\nRange status:")
    print(ranges["range_status"].value_counts())

    print("\nRange types:")
    print(ranges["range_type"].value_counts())

    print("\nAverage width:", round(ranges["range_width"].mean(), 2))
    print("Average width ATR:", round(ranges["range_width_atr"].mean(), 2))

    columns = [
        "range_confirm_time", "range_type",
        "range_upper", "range_lower",
        "upper_touches", "lower_touches",
        "range_status", "break_direction"
    ]

    print("\nLast 15 ranges:")
    print(ranges[columns].tail(15).to_string(index=False))


def main():
    print("Loading M5 and H1 data...")

    m5 = load_data("M5.csv")
    h1 = load_data("H1.csv")

    print("M5 candles:", len(m5))
    print("H1 candles:", len(h1))

    print("\nBuilding M15...")
    m15 = build_m15(m5)
    print("M15 candles:", len(m15))

    print("\nDetecting H1 ranges...")
    h1_df, h1_ranges = detect_ranges(h1, "H1")
    show_results("H1", h1_ranges)

    print("\nDetecting M15 ranges...")
    m15_df, m15_ranges = detect_ranges(m15, "M15")
    show_results("M15", m15_ranges)

    h1_ranges.to_csv(BASE / "impulse_ranges_h1_v4.csv", index=False)
    m15_ranges.to_csv(BASE / "impulse_ranges_m15_v4.csv", index=False)

    pd.concat(
        [h1_ranges, m15_ranges],
        ignore_index=True
    ).to_csv(BASE / "impulse_ranges_all_v4.csv", index=False)

    print("\nSaving charts...")
    save_charts(h1_df, h1_ranges, "H1")
    save_charts(m15_df, m15_ranges, "M15")

    print("\n========== FILES SAVED ==========")
    print("impulse_ranges_h1_v4.csv")
    print("impulse_ranges_m15_v4.csv")
    print("impulse_ranges_all_v4.csv")
    print("range_charts_h1_v4/")
    print("range_charts_m15_v4/")

    print("\nNo BUY/SELL signals generated.")
    print("Causal swing confirmation enabled.")
    print("Horizontal and sloping ranges supported.")
    print("Research only - not validated for live trading.")


if __name__ == "__main__":
    main()
