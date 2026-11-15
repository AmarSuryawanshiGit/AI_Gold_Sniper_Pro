
from pathlib import Path
import pandas as pd

# ============================================================
# V6 - CLEANUP FOR ACTUAL V5 CSV STRUCTURE
# No BUY/SELL signals. Research only.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "impulse_ranges_all_v5.csv"
OUTPUT_FILE = BASE_DIR / "impulse_ranges_v6_cleaned.csv"

TIME_OVERLAP_MIN = 0.50
PRICE_OVERLAP_MIN = 0.60


def normalize_status(status, direction):
    s = str(status).upper().strip()
    d = str(direction).upper().strip()

    if "FAILED" in s:
        return "FAILED"

    if "BREAKDOWN" in s:
        return "BREAKOUT_DOWN"

    if "BREAKOUT_UP" in s:
        return "BREAKOUT_UP"

    if "BREAKOUT_DOWN" in s:
        return "BREAKOUT_DOWN"

    if "BREAKOUT" in s:
        if d in ["UP", "UPWARD", "BULLISH", "BUY"]:
            return "BREAKOUT_UP"
        if d in ["DOWN", "DOWNWARD", "BEARISH", "SELL"]:
            return "BREAKOUT_DOWN"
        return "BREAKOUT_UNVERIFIED"

    return "UNCLASSIFIED"


def classify_pattern(value):
    text = str(value).upper()

    if "HORIZONTAL" in text:
        return "HORIZONTAL"
    if "ASCENDING" in text:
        return "ASCENDING_CHANNEL"
    if "DESCENDING" in text:
        return "DESCENDING_CHANNEL"
    if "CONVERGING" in text or "TRIANGLE" in text:
        return "CONVERGING_TRIANGLE"
    if "DIVERGING" in text or "WEDGE" in text:
        return "DIVERGING_WEDGE"

    return "SLOPING_OR_MIXED"


def time_overlap_ratio(a, b):
    start = max(a["_start"], b["_start"])
    end = min(a["_confirm"], b["_confirm"])

    intersection = max(
        0.0, (end - start).total_seconds()
    )

    duration_a = max(
        1.0, (a["_confirm"] - a["_start"]).total_seconds()
    )
    duration_b = max(
        1.0, (b["_confirm"] - b["_start"]).total_seconds()
    )

    return intersection / min(duration_a, duration_b)


def price_overlap_ratio(a, b):
    intersection = max(
        0.0,
        min(a["range_upper"], b["range_upper"])
        - max(a["range_lower"], b["range_lower"])
    )

    width_a = max(
        0.000001, a["range_upper"] - a["range_lower"]
    )
    width_b = max(
        0.000001, b["range_upper"] - b["range_lower"]
    )

    return intersection / min(width_a, width_b)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nCSV not found: {INPUT_FILE}\n"
            "Make sure impulse_ranges_all_v5.csv is in this folder."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "timeframe",
        "range_start_time",
        "range_confirm_time",
        "range_type",
        "range_upper",
        "range_lower",
        "range_status",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Preserve original CSV fields.
    df = df.copy()
    df["_source_row"] = df.index

    df["_start"] = pd.to_datetime(
        df["range_start_time"], errors="coerce"
    )
    df["_confirm"] = pd.to_datetime(
        df["range_confirm_time"], errors="coerce"
    )

    df["range_upper"] = pd.to_numeric(
        df["range_upper"], errors="coerce"
    )
    df["range_lower"] = pd.to_numeric(
        df["range_lower"], errors="coerce"
    )

    df = df.dropna(subset=[
        "timeframe",
        "_start",
        "_confirm",
        "range_upper",
        "range_lower",
    ]).copy()

    df = df[
        (df["range_upper"] > df["range_lower"])
        & (df["_confirm"] >= df["_start"])
    ].copy()

    if df.empty:
        raise ValueError("No valid ranges found in CSV.")

    if "break_direction" not in df.columns:
        df["break_direction"] = ""

    df["pattern_clean"] = df["range_type"].apply(
        classify_pattern
    )

    df["status_clean"] = df.apply(
        lambda r: normalize_status(
            r["range_status"], r["break_direction"]
        ),
        axis=1
    )

    df = df.sort_values(
        ["timeframe", "_confirm"]
    ).reset_index(drop=True)

    final_rows = []

    for timeframe, tf_data in df.groupby("timeframe"):
        groups = []

        for _, row in tf_data.iterrows():
            matched_group = None

            for group in reversed(groups):
                representative = group[-1]

                t_ratio = time_overlap_ratio(
                    row, representative
                )
                p_ratio = price_overlap_ratio(
                    row, representative
                )

                if (
                    t_ratio >= TIME_OVERLAP_MIN
                    and p_ratio >= PRICE_OVERLAP_MIN
                ):
                    matched_group = group
                    break

            if matched_group is None:
                groups.append([row])
            else:
                matched_group.append(row)

        for group in groups:
            # Keep latest confirmed detection as representative.
            representative = max(
                group, key=lambda r: r["_confirm"]
            )

            result = representative.drop(
                labels=["_start", "_confirm"],
                errors="ignore"
            ).to_dict()

            result["merged_count"] = len(group)

            result["source_rows"] = ",".join(
                str(int(r["_source_row"])) for r in group
            )

            patterns = sorted(set(
                r["pattern_clean"] for r in group
            ))

            statuses = sorted(set(
                r["status_clean"] for r in group
            ))

            result["pattern_clean"] = (
                patterns[0]
                if len(patterns) == 1
                else "MIXED_STRUCTURE"
            )

            result["status_clean"] = (
                statuses[0]
                if len(statuses) == 1
                else "CONFLICT_REVIEW"
            )

            result["review_required"] = (
                result["status_clean"] in [
                    "BREAKOUT_UNVERIFIED",
                    "CONFLICT_REVIEW",
                    "UNCLASSIFIED",
                ]
            )

            final_rows.append(result)

    output = pd.DataFrame(final_rows)

    # Keep original fields plus V6 review fields.
    output = output.drop(
        columns=["_source_row"], errors="ignore"
    )

    output = output.sort_values(
        ["timeframe", "range_confirm_time"]
    )

    output.to_csv(OUTPUT_FILE, index=False)

    print("\n========== V6 RANGE CLEANUP ==========")
    print("Original rows :", len(pd.read_csv(INPUT_FILE)))
    print("Valid rows    :", len(df))
    print("Final ranges  :", len(output))

    print("\nPattern summary:")
    print(output["pattern_clean"].value_counts().to_string())

    print("\nStatus summary:")
    print(output["status_clean"].value_counts().to_string())

    print(
        "\nReview required:",
        int(output["review_required"].sum())
    )

    print("\nSaved successfully:")
    print(OUTPUT_FILE)

    print("\nResearch only - no BUY/SELL signals generated.")


if __name__ == "__main__":
    main()
