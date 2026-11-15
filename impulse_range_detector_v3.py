
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    raise SystemExit("Install matplotlib: pip install matplotlib")

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent

# ---------------- SETTINGS ----------------
ATR_PERIOD = 14

IMPULSE_BARS = 8
IMPULSE_ATR = 2.0
MIN_DIRECTIONAL_RATIO = 0.60

PULLBACK_MIN = 0.20
PULLBACK_MAX = 0.75
PULLBACK_MAX_BARS = 12
RETEST_MAX_BARS = 12
RETEST_TOLERANCE_ATR = 0.25

RANGE_MIN_BARS = 8
RANGE_MAX_BARS = 24

SWING_LEFT = 2
SWING_RIGHT = 2
MIN_TOUCHES = 2
MIN_TOUCH_GAP = 3

MIN_INSIDE_RATIO = 0.70
MIN_RANGE_ATR = 0.40
MAX_RANGE_ATR = 5.0
BREAK_BUFFER_ATR = 0.15
FAILED_BREAK_WINDOW = 3

MAX_CHARTS = 30
# ------------------------------------------


def load_data(filename):
    path = BASE / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found. M5.csv and H1.csv "
            "must be in the same folder as this script."
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

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
    x = m5.set_index("time")

    result = x.resample(
        "15min", label="left", closed="left"
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        count=("close", "count")
    )

    # Only complete groups of three M5 candles.
    result = result[result["count"] == 3]
    result = result.drop(columns="count")
    return result.dropna().reset_index()


def add_atr(df):
    df = df.copy()

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(
        ATR_PERIOD, min_periods=ATR_PERIOD
    ).mean()

    return df


def find_confirmed_swings(df):
    """
    A pivot at index P becomes usable only at:
        P + SWING_RIGHT

    This avoids using a swing before it is confirmed.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    events = {
        i: {"highs": [], "lows": []}
        for i in range(len(df))
    }

    for p in range(SWING_LEFT, len(df) - SWING_RIGHT):

        left_high = high[p-SWING_LEFT:p]
        right_high = high[p+1:p+SWING_RIGHT+1]

        left_low = low[p-SWING_LEFT:p]
        right_low = low[p+1:p+SWING_RIGHT+1]

        is_high = (
            high[p] > left_high.max()
            and high[p] >= right_high.max()
        )

        is_low = (
            low[p] < left_low.min()
            and low[p] <= right_low.min()
        )

        confirmed_at = p + SWING_RIGHT

        if is_high:
            events[confirmed_at]["highs"].append(
                (p, float(high[p]))
            )

        if is_low:
            events[confirmed_at]["lows"].append(
                (p, float(low[p]))
            )

    return events


def count_touches(events, start, end, level, side, tolerance):
    key = "highs" if side == "upper" else "lows"

    candidates = []

    # Only swing events confirmed by end are considered.
    for confirmed_at in range(start, end + 1):
        for pivot, price in events[confirmed_at][key]:
            if abs(price - level) <= tolerance:
                candidates.append(pivot)

    candidates.sort()

    selected = []
    last_pivot = -999999

    for pivot in candidates:
        if pivot - last_pivot >= MIN_TOUCH_GAP:
            selected.append(pivot)
            last_pivot = pivot

    return len(selected)


def classify_break(df, confirm_idx, upper, lower, atr):
    buffer = BREAK_BUFFER_ATR * atr
    first_break = None

    for i in range(confirm_idx + 1, len(df)):

        close = float(df.at[i, "close"])

        if close > upper + buffer:
            direction = "UP"
        elif close < lower - buffer:
            direction = "DOWN"
        else:
            direction = None

        if first_break is None:
            if direction:
                first_break = (i, direction)
            continue

        break_idx, break_direction = first_break

        # Price returned inside the range.
        if i <= break_idx + FAILED_BREAK_WINDOW:
            if lower <= close <= upper:
                return "FAILED", break_idx, break_direction, i

            # Failure window has completed.
            if i == break_idx + FAILED_BREAK_WINDOW:
                status = (
                    "BREAKOUT" if break_direction == "UP"
                    else "BREAKDOWN"
                )
                return status, break_idx, break_direction, None

    if first_break is None:
        return "ACTIVE", None, None, None

    break_idx, break_direction = first_break

    # Not enough candles to decide if this break failed.
    if len(df) - 1 <= break_idx + FAILED_BREAK_WINDOW:
        return "PENDING_BREAK", break_idx, break_direction, None

    status = (
        "BREAKOUT" if break_direction == "UP"
        else "BREAKDOWN"
    )

    return status, break_idx, break_direction, None


def detect_ranges(source_df, timeframe):
    df = add_atr(source_df).reset_index(drop=True)
    swings = find_confirmed_swings(df)

    results = []
    used_until = -1
    n = len(df)

    i = ATR_PERIOD + SWING_LEFT + 2

    while i + IMPULSE_BARS + 2 < n:

        if i <= used_until:
            i += 1
            continue

        atr = df.at[i, "atr"]

        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        impulse_end = i + IMPULSE_BARS - 1
        impulse = df.iloc[i:impulse_end + 1]

        net_move = (
            impulse.iloc[-1]["close"]
            - impulse.iloc[0]["open"]
        )

        direction = "BULLISH" if net_move > 0 else "BEARISH"

        if direction == "BULLISH":
            directional_count = (
                impulse["close"] > impulse["open"]
            ).sum()
        else:
            directional_count = (
                impulse["close"] < impulse["open"]
            ).sum()

        ratio = directional_count / IMPULSE_BARS

        if (
            abs(net_move) < IMPULSE_ATR * atr
            or ratio < MIN_DIRECTIONAL_RATIO
        ):
            i += 1
            continue

        if direction == "BULLISH":
            impulse_extreme = float(impulse["high"].max())
            impulse_base = float(impulse["low"].min())
        else:
            impulse_extreme = float(impulse["low"].min())
            impulse_base = float(impulse["high"].max())

        impulse_size = abs(impulse_extreme - impulse_base)

        if impulse_size <= 0:
            i += 1
            continue

        # -------- PULLBACK --------
        pullback_idx = None
        pullback_extreme = None

        pullback_limit = min(
            impulse_end + PULLBACK_MAX_BARS,
            n - 1
        )

        for p in range(impulse_end + 1, pullback_limit + 1):

            if direction == "BULLISH":
                candidate = float(df.at[p, "low"])
                retracement = (
                    impulse_extreme - candidate
                ) / impulse_size
            else:
                candidate = float(df.at[p, "high"])
                retracement = (
                    candidate - impulse_extreme
                ) / impulse_size

            if PULLBACK_MIN <= retracement <= PULLBACK_MAX:
                pullback_idx = p
                pullback_extreme = candidate
                break

        if pullback_idx is None:
            i += 1
            continue

        # -------- RETEST --------
        retest_idx = None
        retest_limit = min(
            pullback_idx + RETEST_MAX_BARS,
            n - 1
        )

        for r in range(pullback_idx + 1, retest_limit + 1):

            if direction == "BULLISH":
                pullback_extreme = min(
                    pullback_extreme,
                    float(df.at[r, "low"])
                )

                touched = (
                    df.at[r, "high"]
                    >= impulse_extreme - RETEST_TOLERANCE_ATR * atr
                )

                retracement = (
                    impulse_extreme - pullback_extreme
                ) / impulse_size

            else:
                pullback_extreme = max(
                    pullback_extreme,
                    float(df.at[r, "high"])
                )

                touched = (
                    df.at[r, "low"]
                    <= impulse_extreme + RETEST_TOLERANCE_ATR * atr
                )

                retracement = (
                    pullback_extreme - impulse_extreme
                ) / impulse_size

            if (
                touched
                and PULLBACK_MIN <= retracement <= PULLBACK_MAX
            ):
                retest_idx = r
                break

        if retest_idx is None:
            i += 1
            continue

        upper = max(impulse_extreme, pullback_extreme)
        lower = min(impulse_extreme, pullback_extreme)
        width = upper - lower

        retest_atr = df.at[retest_idx, "atr"]

        if not np.isfinite(retest_atr) or retest_atr <= 0:
            i += 1
            continue

        width_atr = width / retest_atr

        if not MIN_RANGE_ATR <= width_atr <= MAX_RANGE_ATR:
            i += 1
            continue

        # -------- CAUSAL RANGE CONFIRMATION --------
        confirm_idx = None
        upper_touches = 0
        lower_touches = 0

        last_end = min(
            retest_idx + RANGE_MAX_BARS,
            n - 1
        )

        for end_idx in range(
            retest_idx + RANGE_MIN_BARS - 1,
            last_end + 1
        ):

            start_idx = max(
                retest_idx,
                end_idx - RANGE_MAX_BARS + 1
            )

            sample = df.iloc[start_idx:end_idx + 1]

            inside = (
                (sample["high"] <= upper + 0.10 * retest_atr)
                & (sample["low"] >= lower - 0.10 * retest_atr)
            )

            inside_ratio = inside.mean()
            tolerance = max(0.10 * retest_atr, 0.01)

            up_count = count_touches(
                swings, start_idx, end_idx,
                upper, "upper", tolerance
            )

            low_count = count_touches(
                swings, start_idx, end_idx,
                lower, "lower", tolerance
            )

            if (
                inside_ratio >= MIN_INSIDE_RATIO
                and up_count >= MIN_TOUCHES
                and low_count >= MIN_TOUCHES
            ):
                confirm_idx = end_idx
                upper_touches = up_count
                lower_touches = low_count
                break

        if confirm_idx is None:
            i += 1
            continue

        status, break_idx, break_dir, failed_idx = classify_break(
            df,
            confirm_idx,
            upper,
            lower,
            float(df.at[confirm_idx, "atr"])
        )

        end_idx = (
            break_idx if break_idx is not None
            else n - 1
        )

        results.append({
            "timeframe": timeframe,
            "direction_before_range": direction,
            "impulse_start_time": df.at[i, "time"],
            "impulse_end_time": df.at[impulse_end, "time"],
            "pullback_time": df.at[pullback_idx, "time"],
            "retest_time": df.at[retest_idx, "time"],
            "range_confirm_time": df.at[confirm_idx, "time"],
            "break_time": (
                df.at[break_idx, "time"]
                if break_idx is not None else pd.NaT
            ),
            "failed_return_time": (
                df.at[failed_idx, "time"]
                if failed_idx is not None else pd.NaT
            ),
            "range_upper": round(upper, 2),
            "range_lower": round(lower, 2),
            "range_width": round(width, 2),
            "range_width_atr": round(width_atr, 3),
            "upper_swing_touches": upper_touches,
            "lower_swing_touches": lower_touches,
            "range_status": status,
            "break_direction": break_dir or "",
            "impulse_start_idx": i,
            "impulse_end_idx": impulse_end,
            "pullback_idx": pullback_idx,
            "retest_idx": retest_idx,
            "range_confirm_idx": confirm_idx,
            "range_end_idx": end_idx
        })

        used_until = max(
            confirm_idx,
            break_idx if break_idx is not None else confirm_idx
        )

        i = used_until + 1

    return pd.DataFrame(results)


def draw_chart(df, ranges, timeframe, folder):
    if ranges.empty:
        return

    folder.mkdir(exist_ok=True)

    for number, row in enumerate(
        ranges.tail(MAX_CHARTS).itertuples(index=False), 1
    ):

        left = max(0, row.impulse_start_idx - 80)
        right = min(len(df) - 1, row.range_end_idx + 40)

        view = df.iloc[left:right + 1].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(16, 7))

        for x, candle in enumerate(view.itertuples(index=False)):
            color = "green" if candle.close >= candle.open else "red"

            ax.vlines(
                x, candle.low, candle.high,
                color=color, linewidth=0.8
            )

            body_low = min(candle.open, candle.close)
            body_height = max(
                abs(candle.close - candle.open), 0.01
            )

            ax.add_patch(Rectangle(
                (x - 0.30, body_low),
                0.60,
                body_height,
                facecolor=color,
                edgecolor=color
            ))

        def local(idx):
            return idx - left

        confirm_x = local(row.range_confirm_idx)
        end_x = local(row.range_end_idx)

        ax.axhline(
            row.range_upper,
            color="blue",
            linestyle="--",
            label="Range upper"
        )

        ax.axhline(
            row.range_lower,
            color="purple",
            linestyle="--",
            label="Range lower"
        )

        ax.axvspan(
            confirm_x, end_x,
            color="orange", alpha=0.10,
            label="Confirmed range"
        )

        markers = [
            (row.impulse_start_idx, "black", "Impulse start"),
            (row.impulse_end_idx, "green", "Impulse end"),
            (row.pullback_idx, "orange", "Pullback"),
            (row.retest_idx, "magenta", "Retest"),
            (row.range_confirm_idx, "blue", "Range confirmed")
        ]

        for idx, color, label in markers:
            ax.axvline(
                local(idx), color=color,
                linestyle=":", label=label
            )

        if row.break_time is not pd.NaT and pd.notna(row.break_time):
            ax.axvline(
                local(row.range_end_idx),
                color="red", linestyle=":",
                label="First break"
            )

        positions = np.linspace(
            0, len(view) - 1,
            min(10, len(view))
        ).astype(int)

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [
                view.at[p, "time"].strftime("%m-%d %H:%M")
                for p in positions
            ],
            rotation=35,
            ha="right"
        )

        ax.set_title(
            f"{timeframe} | {row.direction_before_range} | "
            f"{row.range_status} | Upper {row.range_upper:.2f} "
            f"/ Lower {row.range_lower:.2f}"
        )

        ax.set_ylabel("Gold price")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()

        fig.savefig(
            folder / f"{timeframe.lower()}_range_{number:02d}.png",
            dpi=120
        )

        plt.close(fig)


def print_results(ranges, timeframe):
    print("\n" + "=" * 55)
    print(f"{timeframe} RANGE RESULTS V3")
    print("=" * 55)

    print(f"Confirmed ranges: {len(ranges)}")

    if ranges.empty:
        print("No ranges detected.")
        return

    print("\nRange status:")
    print(ranges["range_status"].value_counts().to_string())

    print("\nDirection before range:")
    print(
        ranges["direction_before_range"]
        .value_counts().to_string()
    )

    print(
        f"\nAverage range width: "
        f"{ranges['range_width'].mean():.2f}"
    )

    print(
        f"Average width in ATR: "
        f"{ranges['range_width_atr'].mean():.2f}"
    )

    columns = [
        "direction_before_range",
        "range_confirm_time",
        "range_upper",
        "range_lower",
        "upper_swing_touches",
        "lower_swing_touches",
        "range_status",
        "break_direction"
    ]

    print("\nLast 15 ranges:")
    print(ranges[columns].tail(15).to_string(index=False))


def main():
    print("Loading M5 and H1 data...")

    m5 = load_data("M5.csv")
    h1 = load_data("H1.csv")

    print(f"M5 candles: {len(m5)}")
    print(f"H1 candles: {len(h1)}")

    print("\nBuilding M15 candles...")
    m15 = make_m15(m5)

    print(f"M15 candles: {len(m15)}")

    print("\nDetecting H1 ranges...")
    h1_ranges = detect_ranges(h1, "H1")

    print("\nDetecting M15 ranges...")
    m15_ranges = detect_ranges(m15, "M15")

    h1_ranges.to_csv(
        BASE / "impulse_ranges_h1_v3.csv",
        index=False
    )

    m15_ranges.to_csv(
        BASE / "impulse_ranges_m15_v3.csv",
        index=False
    )

    pd.concat(
        [h1_ranges, m15_ranges],
        ignore_index=True
    ).to_csv(
        BASE / "impulse_ranges_all_v3.csv",
        index=False
    )

    print_results(h1_ranges, "H1")
    print_results(m15_ranges, "M15")

    print("\nSaving charts...")

    draw_chart(
        h1, h1_ranges, "H1",
        BASE / "range_charts_h1_v3"
    )

    draw_chart(
        m15, m15_ranges, "M15",
        BASE / "range_charts_m15_v3"
    )

    print("\n========== FILES SAVED ==========")
    print("impulse_ranges_h1_v3.csv")
    print("impulse_ranges_m15_v3.csv")
    print("impulse_ranges_all_v3.csv")
    print("range_charts_h1_v3/")
    print("range_charts_m15_v3/")

    print("\nNo BUY/SELL signals generated.")
    print("Swing confirmation timing is causal.")
    print("Breaks are checked only after range confirmation.")
    print("A return inside within the failure window is FAILED.")
    print("Research detector only; not validated for live trading.")


if __name__ == "__main__":
    main()
