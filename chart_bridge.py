from __future__ import annotations

import os

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

CHART_CANDLES = max(
    500,
    int(os.getenv("CHART_CANDLES", "2000"))
)

PATTERN_BODY_MIN_RATIO = 0.03
PATTERN_BODY_MAX_RATIO = 0.40


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]

    required = ["open", "high", "low", "close"]

    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data.dropna(subset=required, inplace=True)

    close = data["close"]
    high = data["high"]
    low = data["low"]

    # EMA
    for period in [9, 20, 50, 200]:
        data[f"ema{period}"] = close.ewm(
            span=period,
            adjust=False
        ).mean()

    # Bollinger Bands
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    data["bb_upper"] = middle + 2 * std
    data["bb_lower"] = middle - 2 * std
    data["bb_middle"] = middle

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["rsi"] = 100 - (100 / (1 + rs))

    data.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi"
    ] = 100

    data.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "rsi"
    ] = 0

    data["rsi"] = data["rsi"].fillna(50)

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["macd"] = ema12 - ema26

    data["macd_signal"] = data["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["macd_histogram"] = (
        data["macd"] - data["macd_signal"]
    )

    # ATR
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    return data


# =========================================================
# TIME HELPERS
# =========================================================

def to_unix_seconds(value) -> int:

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return int(timestamp.timestamp())


def safe_number(value):

    if value is None or pd.isna(value):
        return None

    result = float(value)

    if not np.isfinite(result):
        return None

    return result


def make_line_data(data, column):

    result = []

    for timestamp, value in zip(
        data.index,
        data[column]
    ):

        number = safe_number(value)

        if number is None:
            continue

        result.append({
            "time": to_unix_seconds(timestamp),
            "value": number
        })

    return result


# =========================================================
# RANGE LINES
# =========================================================

def build_range_lines(ranges):

    output = []

    if not ranges:
        return output

    for item in ranges:

        try:

            start_time = int(item["startTime"])
            confirm_time = int(item["confirmTime"])

            upper_start = safe_number(
                item.get("upperStartPrice")
            )

            lower_start = safe_number(
                item.get("lowerStartPrice")
            )

            upper_end = safe_number(
                item.get("upperPrice")
            )

            lower_end = safe_number(
                item.get("lowerPrice")
            )

            if None in (
                upper_start,
                lower_start,
                upper_end,
                lower_end
            ):
                continue

            if confirm_time <= start_time:
                continue

            output.append({
                "id": str(item.get("id", "")),
                "timeframe": str(
                    item.get("timeframe", "")
                ).upper(),
                "rangeType": str(
                    item.get("rangeType", "RANGE")
                ),
                "status": str(
                    item.get("status", "ACTIVE")
                ),
                "upper": [
                    {
                        "time": start_time,
                        "value": upper_start
                    },
                    {
                        "time": confirm_time,
                        "value": upper_end
                    }
                ],
                "lower": [
                    {
                        "time": start_time,
                        "value": lower_start
                    },
                    {
                        "time": confirm_time,
                        "value": lower_end
                    }
                ]
            })

        except (KeyError, TypeError, ValueError):
            continue

    return output


# =========================================================
# CANDLESTICK PATTERN DETECTION
# Raw pattern detection helper is retained for analysis.
# It will NOT be used to draw chart markers.
# =========================================================

def detect_single_candle_pattern(row):

    o = safe_number(row.open)
    h = safe_number(row.high)
    l = safe_number(row.low)
    c = safe_number(row.close)

    if None in (o, h, l, c):
        return []

    candle_range = h - l

    if candle_range <= 0:
        return []

    body = abs(c - o)

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    body_ratio = body / candle_range
    upper_ratio = upper_wick / candle_range
    lower_ratio = lower_wick / candle_range

    patterns = []

    # HAMMER
    if (
        lower_ratio >= 0.55
        and upper_ratio <= 0.20
        and body_ratio <= PATTERN_BODY_MAX_RATIO
        and c > l + candle_range * 0.60
    ):
        patterns.append({
            "name": "HAMMER",
            "direction": "BUY",
            "position": "belowBar",
            "color": "#2196f3",
            "shape": "arrowUp"
        })

    # SHOOTING STAR
    if (
        upper_ratio >= 0.55
        and lower_ratio <= 0.20
        and body_ratio <= PATTERN_BODY_MAX_RATIO
        and c < l + candle_range * 0.40
    ):
        patterns.append({
            "name": "SHOOTING STAR",
            "direction": "SELL",
            "position": "aboveBar",
            "color": "#ff9800",
            "shape": "arrowDown"
        })

    # BULLISH PIN BAR
    if (
        lower_ratio >= 0.60
        and upper_ratio <= 0.20
        and body_ratio <= 0.30
        and c > l + candle_range * 0.60
    ):
        patterns.append({
            "name": "BULLISH PIN BAR",
            "direction": "BUY",
            "position": "belowBar",
            "color": "#26a69a",
            "shape": "arrowUp"
        })

    # BEARISH PIN BAR
    if (
        upper_ratio >= 0.60
        and lower_ratio <= 0.20
        and body_ratio <= 0.30
        and c < l + candle_range * 0.40
    ):
        patterns.append({
            "name": "BEARISH PIN BAR",
            "direction": "SELL",
            "position": "aboveBar",
            "color": "#ef5350",
            "shape": "arrowDown"
        })

    return patterns



def detect_chart_patterns(
    data: pd.DataFrame,
    timeframe: str = "M1",
    drop_forming_candle: bool = True
):

    markers = []

    if data is None or len(data) < 3:
        return markers

    timeframe = str(timeframe).upper()

    # Exclude current unfinished candle.
    # Pattern detection uses only completed candles.
    scan_data = data.copy()

    if drop_forming_candle:
        scan_data = scan_data.iloc[:-1]

    if len(scan_data) < 2:
        return markers

    for i in range(1, len(scan_data)):

        row = scan_data.iloc[i]
        previous = scan_data.iloc[i - 1]

        timestamp = to_unix_seconds(
            scan_data.index[i]
        )

        # -----------------------------------------
        # Single-candle patterns
        # -----------------------------------------

        single_patterns = detect_single_candle_pattern(
            row
        )

        for pattern in single_patterns:

            markers.append({
                "time": timestamp,
                "position": pattern["position"],
                "color": pattern["color"],
                "shape": pattern["shape"],
                "text": (
                    f"{timeframe} {pattern['name']}"
                ),
                "pattern": pattern["name"],
                "direction": pattern["direction"],
                "timeframe": timeframe
            })

        # -----------------------------------------
        # Two-candle engulfing patterns
        # -----------------------------------------

        bullish = row.close > row.open
        bearish = row.close < row.open

        previous_bullish = (
            previous.close > previous.open
        )

        previous_bearish = (
            previous.close < previous.open
        )

        # Bullish Engulfing
        bullish_engulfing = (
            previous_bearish
            and bullish
            and row.open <= previous.close
            and row.close >= previous.open
        )

        if bullish_engulfing:

            markers.append({
                "time": timestamp,
                "position": "belowBar",
                "color": "#26a69a",
                "shape": "arrowUp",
                "text": f"{timeframe} BULLISH ENGULFING",
                "pattern": "BULLISH ENGULFING",
                "direction": "BUY",
                "timeframe": timeframe
            })

        # Bearish Engulfing
        bearish_engulfing = (
            previous_bullish
            and bearish
            and row.open >= previous.close
            and row.close <= previous.open
        )

        if bearish_engulfing:

            markers.append({
                "time": timestamp,
                "position": "aboveBar",
                "color": "#ef5350",
                "shape": "arrowDown",
                "text": f"{timeframe} BEARISH ENGULFING",
                "pattern": "BEARISH ENGULFING",
                "direction": "SELL",
                "timeframe": timeframe
            })

    return markers


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def calculate_support_resistance(
    data: pd.DataFrame,
    lookback: int = 60
):

    recent = data.tail(lookback)

    if recent.empty:
        return None, None

    support = safe_number(
        recent["low"].min()
    )

    resistance = safe_number(
        recent["high"].max()
    )

    return support, resistance


# =========================================================
# VERIFIED SIGNAL MARKER
# =========================================================

def build_verified_signal_marker(signal, data):

    if not isinstance(signal, dict):
        return []

    if data is None or data.empty:
        return []

    if signal.get("status") != "SIGNAL":
        return []

    direction = str(
        signal.get("direction", "")
    ).upper()

    if direction not in ("BUY", "SELL"):
        return []

    try:
        score = float(signal.get("score", 0))
    except (TypeError, ValueError):
        return []

    if not np.isfinite(score):
        return []

    try:
        entry = safe_number(signal.get("entry"))
    except (TypeError, ValueError):
        entry = None

    if entry is None:
        return []

    # The marker is anchored to the latest completed
    # candle of the displayed chart timeframe.
    if len(data) < 2:
        return []

    candle = data.iloc[-2]
    candle_time = data.index[-2]

    if direction == "BUY":

        position = "belowBar"
        shape = "arrowUp"
        color = "#00c853"

    else:

        position = "aboveBar"
        shape = "arrowDown"
        color = "#ff1744"

    setup = str(
        signal.get("setup", "VERIFIED SETUP")
    )

    return [{
        "time": to_unix_seconds(candle_time),
        "position": position,
        "color": color,
        "shape": shape,
        "text": f"{direction} | {setup} | {score:.0f}",
        "pattern": setup,
        "direction": direction,
        "score": score,
        "entry": entry,
        "timeframe": "SIGNAL"
    }]


# =========================================================
# CHART PAYLOAD
# =========================================================

def build_chart_payload(
    df: pd.DataFrame,
    symbol="GOLD",
    timeframe="M1",
    bid=None,
    ask=None,
    ranges=None,
    signal=None,
    pattern_scan=None
):

    timeframe = str(timeframe).upper()

    if df is None or df.empty:

        return {
            "candles": [],
            "markers": [],
            "ranges": build_range_lines(ranges),
            "status": "No market data"
        }

    data = df.copy()

    data.columns = [
        str(column).lower()
        for column in data.columns
    ]

    # ---------------------------------------------
    # Timestamp handling
    # ---------------------------------------------

    if "time" in data.columns:

        if pd.api.types.is_numeric_dtype(
            data["time"]
        ):

            data["time"] = pd.to_datetime(
                data["time"],
                unit="s",
                utc=True,
                errors="coerce"
            )

        else:

            data["time"] = pd.to_datetime(
                data["time"],
                utc=True,
                errors="coerce"
            )

        data.set_index("time", inplace=True)

    if not isinstance(
        data.index,
        pd.DatetimeIndex
    ):

        raise ValueError(
            "Chart data must have a DatetimeIndex or time column"
        )

    if data.index.tz is None:

        data.index = data.index.tz_localize("UTC")

    else:

        data.index = data.index.tz_convert("UTC")

    data = data.sort_index()

    data = data[
        ~data.index.duplicated(keep="last")
    ]

    # ---------------------------------------------
    # OHLC validation
    # ---------------------------------------------

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

    for col in required:

        if col not in data.columns:

            raise ValueError(
                f"Chart data missing column: {col}"
            )

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data.dropna(
        subset=required,
        inplace=True
    )

    if data.empty:

        return {
            "candles": [],
            "markers": [],
            "ranges": build_range_lines(ranges),
            "status": "No valid OHLC data"
        }

    # ---------------------------------------------
    # Indicator calculation
    # ---------------------------------------------

    data = calculate_indicators(data)

    data = data.tail(CHART_CANDLES).copy()

    # ---------------------------------------------
    # Candle payload
    # ---------------------------------------------

    candles = []

    for timestamp, row in data.iterrows():

        candle = {
            "time": to_unix_seconds(timestamp),
            "open": safe_number(row.open),
            "high": safe_number(row.high),
            "low": safe_number(row.low),
            "close": safe_number(row.close)
        }

        if all(
            candle[key] is not None
            for key in required
        ):

            candles.append(candle)

    # ---------------------------------------------
    # Moving averages
    # ---------------------------------------------

    ema9 = make_line_data(data, "ema9")
    ema20 = make_line_data(data, "ema20")
    ema50 = make_line_data(data, "ema50")
    ema200 = make_line_data(data, "ema200")

    # ---------------------------------------------
    # Bollinger Bands
    # ---------------------------------------------

    bb_upper = make_line_data(
        data,
        "bb_upper"
    )

    bb_lower = make_line_data(
        data,
        "bb_lower"
    )

    bb_middle = make_line_data(
        data,
        "bb_middle"
    )

    # ---------------------------------------------
    # RSI / MACD
    # ---------------------------------------------

    rsi = make_line_data(data, "rsi")

    macd = make_line_data(data, "macd")

    macd_signal = make_line_data(
        data,
        "macd_signal"
    )

    macd_histogram = make_line_data(
        data,
        "macd_histogram"
    )

    # ---------------------------------------------
    # Volume
    # ---------------------------------------------

    volume_data = []

    volume_column = next(
        (
            column
            for column in [
                "tick_volume",
                "real_volume",
                "volume"
            ]
            if column in data.columns
        ),
        None
    )

    if volume_column:

        for timestamp, row in data.iterrows():

            value = safe_number(
                row[volume_column]
            )

            if value is None:
                continue

            volume_data.append({
                "time": to_unix_seconds(timestamp),
                "value": max(0, value),
                "color": (
                    "rgba(38,166,154,0.55)"
                    if row.close >= row.open
                    else "rgba(239,83,80,0.55)"
                )
            })

    # ---------------------------------------------
    # CANDLESTICK PATTERN + VERIFIED SIGNAL MARKERS
    # ---------------------------------------------

    # Detect raw candlestick patterns on displayed timeframe
    markers = detect_chart_patterns(
        data,
        timeframe=timeframe,
        drop_forming_candle=True
    )

    # Add verified BUY / SELL signal marker
    verified_markers = build_verified_signal_marker(
        signal,
        data
    )

    markers.extend(verified_markers)

    # ---------------------------------------------
    # Support / Resistance
    # ---------------------------------------------

    support, resistance = (
        calculate_support_resistance(data, 60)
    )

    # ---------------------------------------------
    # Status
    # ---------------------------------------------

    status = (
        f"{symbol} {timeframe} | Bid: {bid} | Ask: {ask}"
        if bid is not None and ask is not None
        else f"{symbol} {timeframe}"
    )

    # ---------------------------------------------
    # Final payload
    # ---------------------------------------------

    return {
        "symbol": symbol,
        "timeframe": timeframe,

        "candles": candles,

        "ema9": ema9,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,

        "bbUpper": bb_upper,
        "bbLower": bb_lower,
        "bbMiddle": bb_middle,

        "rsi": rsi,

        "macd": macd,
        "macdSignal": macd_signal,
        "macdHistogram": macd_histogram,

        "volume": volume_data,

        "markers": markers,

        "support": support,
        "resistance": resistance,

        "ranges": build_range_lines(ranges),

        "status": status
    }
