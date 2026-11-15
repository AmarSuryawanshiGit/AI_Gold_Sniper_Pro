
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import hashlib

import numpy as np
import pandas as pd

from impulse_range_detector_v5 import (
    detect_ranges,
    SETTINGS,
)


# =========================================================
# RANGE ENGINE
# Live range detection for H1, M15 and M5
# =========================================================

@dataclass
class RangeRecord:
    range_id: str
    timeframe: str
    start_time: int
    confirm_time: int
    upper_slope: float
    lower_slope: float
    upper_intercept: float
    lower_intercept: float
    upper_start_price: float
    lower_start_price: float
    range_type: str
    upper_price: float
    lower_price: float
    status: str = "ACTIVE"


class RangeEngine:

    TIMEFRAMES = ("H1", "M15", "M5")

    # M5 settings are provisional.
    # Validate these against M5 historical data.
    M5_SETTINGS = {
        "min_bars": 10,
        "max_bars": 42,
        "min_touches": 3,
        "touch_gap": 2,
        "touch_atr": 0.35,
        "min_width_atr": 0.30,
        "max_width_atr": 6.0,
        "inside_ratio": 0.60,
    }

    def __init__(self, max_history: int = 500):
        self.max_history = max_history

        # Persistent range storage.
        self.ranges: Dict[str, RangeRecord] = {}

        # Add M5 configuration if V5 does not define it.
        if "M5" not in SETTINGS:
            SETTINGS["M5"] = dict(self.M5_SETTINGS)

    # -----------------------------------------------------
    # Normalize MT5 dataframe
    # -----------------------------------------------------

    @staticmethod
    def normalize_frame(
        source: Optional[pd.DataFrame]
    ) -> pd.DataFrame:

        if source is None or source.empty:
            return pd.DataFrame()

        df = source.copy()

        df.columns = [str(col).lower() for col in df.columns]

        if "time" not in df.columns:
            df = df.reset_index()

            df.columns = [str(col).lower() for col in df.columns]

            if "time" not in df.columns:
                df = df.rename(
                    columns={df.columns[0]: "time"}
                )

        required = [
            "time",
            "open",
            "high",
            "low",
            "close",
        ]

        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        # MT5 time may be datetime or Unix seconds.
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(
                df["time"],
                unit="s",
                utc=True,
                errors="coerce",
            )
        else:
            df["time"] = pd.to_datetime(
                df["time"],
                utc=True,
                errors="coerce",
            )

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df = (
            df.dropna(subset=required)
            .sort_values("time")
            .drop_duplicates("time", keep="last")
            .reset_index(drop=True)
        )

        return df.tail(5000).reset_index(drop=True)

    # -----------------------------------------------------
    # Stable range ID
    # -----------------------------------------------------

    @staticmethod
    def make_range_id(
        timeframe: str,
        start_time: int,
        confirm_time: int,
    ) -> str:

        raw = (
            f"{timeframe}|"
            f"{start_time}|"
            f"{confirm_time}"
        )

        return hashlib.sha1(
            raw.encode("utf-8")
        ).hexdigest()[:16]

    # -----------------------------------------------------
    # Convert detector output into persistent records
    # -----------------------------------------------------

    def scan_timeframe(
        self,
        timeframe: str,
        source: pd.DataFrame,
    ) -> None:

        timeframe = str(timeframe).upper()

        if timeframe not in self.TIMEFRAMES:
            return

        df = self.normalize_frame(source)

        if df.empty:
            return

        min_bars = SETTINGS[timeframe]["min_bars"]

        if len(df) < min_bars + 20:
            return

        # V5 detector scans supplied historical candles.
        _, detected = detect_ranges(df, timeframe)

        if detected is None or detected.empty:
            return

        for _, row in detected.iterrows():

            try:
                start_index = int(row["range_start_index"])
                confirm_index = int(row["range_confirm_index"])

                if (
                    start_index < 0
                    or confirm_index >= len(df)
                    or confirm_index <= start_index
                ):
                    continue

                start_time = int(
                    pd.Timestamp(
                        df.iloc[start_index]["time"]
                    ).timestamp()
                )

                confirm_time = int(
                    pd.Timestamp(
                        df.iloc[confirm_index]["time"]
                    ).timestamp()
                )

                if confirm_time <= start_time:
                    continue

                range_id = self.make_range_id(
                    timeframe,
                    start_time,
                    confirm_time,
                )

                upper_slope = float(row["upper_slope"])
                lower_slope = float(row["lower_slope"])

                upper_price = float(row["range_upper"])
                lower_price = float(row["range_lower"])

                # Calculate line intercepts at confirmation index.
                upper_intercept = (
                    upper_price
                    - upper_slope * confirm_index
                )

                lower_intercept = (
                    lower_price
                    - lower_slope * confirm_index
                )

                # Calculate trendline prices at original start index.
                upper_start_price = (
                    upper_slope * start_index
                    + upper_intercept
                )

                lower_start_price = (
                    lower_slope * start_index
                    + lower_intercept
                )

                # Reject invalid numerical values.
                values = [
                    upper_slope,
                    lower_slope,
                    upper_intercept,
                    lower_intercept,
                    upper_start_price,
                    lower_start_price,
                    upper_price,
                    lower_price,
                ]

                if not np.all(np.isfinite(values)):
                    continue

                # Preserve original range after breakout/breakdown.
                # Do not delete historical ranges based on status.
                record = RangeRecord(
                    range_id=range_id,
                    timeframe=timeframe,
                    start_time=start_time,
                    confirm_time=confirm_time,
                    upper_slope=upper_slope,
                    lower_slope=lower_slope,
                    upper_intercept=upper_intercept,
                    lower_intercept=lower_intercept,
                    upper_start_price=upper_start_price,
                    lower_start_price=lower_start_price,
                    range_type=str(row["range_type"]),
                    upper_price=upper_price,
                    lower_price=lower_price,
                    status=str(row.get("range_status", "ACTIVE")).upper(),
                )

                self.ranges[range_id] = record

            except (
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                continue

    # -----------------------------------------------------
    # Update all supported timeframe ranges
    # -----------------------------------------------------

    def update(
        self,
        frames: Dict[str, pd.DataFrame],
    ) -> List[dict]:

        for timeframe in self.TIMEFRAMES:

            source = frames.get(timeframe)

            if source is None:
                continue

            self.scan_timeframe(
                timeframe,
                source,
            )

        # Keep historical ranges.
        # Never remove a range after breakout.
        return self.get_ranges()

    # -----------------------------------------------------
    # Return all ranges for chart payload
    # -----------------------------------------------------

    def get_ranges(self) -> List[dict]:

        output = []

        for record in self.ranges.values():

            output.append({
                "id": record.range_id,
                "timeframe": record.timeframe,

                "startTime": record.start_time,
                "confirmTime": record.confirm_time,

                "upperSlope": record.upper_slope,
                "lowerSlope": record.lower_slope,

                "upperIntercept": record.upper_intercept,
                "lowerIntercept": record.lower_intercept,

                # Required by chart_bridge.py
                "upperStartPrice": record.upper_start_price,
                "lowerStartPrice": record.lower_start_price,

                "upperPrice": record.upper_price,
                "lowerPrice": record.lower_price,

                "rangeType": record.range_type,
                "status": record.status,
            })

        output.sort(
            key=lambda item: (
                item["startTime"],
                item["timeframe"],
            )
        )

        return output

    # -----------------------------------------------------
    # Get only ranges applicable to selected chart
    # -----------------------------------------------------

    def get_ranges_for_chart(
        self,
        chart_timeframe: str,
    ) -> List[dict]:

        chart_timeframe = str(chart_timeframe).upper()

        visibility = {
            "H1": {"H1"},
            "M30": {"H1"},
            "M15": {"H1", "M15"},
            "M5": {"H1", "M15", "M5"},
            "M1": {"H1", "M15", "M5"},
        }

        allowed = visibility.get(
            chart_timeframe,
            set(),
        )

        return [
            item
            for item in self.get_ranges()
            if item["timeframe"] in allowed
        ]
