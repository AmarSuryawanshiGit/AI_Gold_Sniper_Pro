
import numpy as np
import pandas as pd

MIN_SIGNAL_SCORE = 70
MAX_SPREAD_POINTS = 50

def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError, OverflowError):
        return default

class SignalAnalyzer:

    def __init__(self):

        self.min_score = MIN_SIGNAL_SCORE
        self.max_spread = MAX_SPREAD_POINTS
        self.min_rr = 1.5

        self.timeframes = [
            "M1",
            "M5",
            "M15",
            "M30",
            "H1"
        ]

        # Prevent repeated analysis
        # of the same closed candle.
        self.last_processed_candle = None

    # =====================================================
    # MULTI-TIMEFRAME TREND ANALYSIS
    # =====================================================

    def analyze_trends(self, frames):

        trends = {}

        for timeframe in [
            "H1",
            "M30",
            "M15",
            "M5"
        ]:

            trends[timeframe] = _trend(
                frames.get(timeframe)
            )

        bullish_votes = sum(
            value == "BULLISH"
            for value in trends.values()
        )

        bearish_votes = sum(
            value == "BEARISH"
            for value in trends.values()
        )

        sideways_votes = sum(
            value == "SIDEWAYS"
            for value in trends.values()
        )

        if bullish_votes >= 3:
            market_trend = "BULLISH"

        elif bearish_votes >= 3:
            market_trend = "BEARISH"

        elif sideways_votes >= 2:
            market_trend = "SIDEWAYS"

        else:
            market_trend = "TRANSITION"

        return trends, market_trend

    # =====================================================
    # SINGLE CANDLE PATTERN DETECTION
    # =====================================================

    def detect_patterns(self, current, previous):

        o = _safe_float(current.open)
        h = _safe_float(current.high)
        l = _safe_float(current.low)
        c = _safe_float(current.close)

        po = _safe_float(previous.open)
        pc = _safe_float(previous.close)

        body = abs(c - o)

        candle_range = max(
            h - l,
            1e-9
        )

        upper_wick = max(
            0.0,
            h - max(o, c)
        )

        lower_wick = max(
            0.0,
            min(o, c) - l
        )

        body_ratio = body / candle_range
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range

        bullish = c > o
        bearish = c < o

        # Bullish Engulfing
        bullish_engulfing = (
            bullish
            and pc < po
            and o <= pc
            and c >= po
        )

        # Bearish Engulfing
        bearish_engulfing = (
            bearish
            and pc > po
            and o >= pc
            and c <= po
        )

        # Hammer
        hammer = (
            lower_wick >= max(
                body * 2,
                candle_range * 0.35
            )
            and upper_wick <= candle_range * 0.20
            and body_ratio <= 0.40
        )

        # Shooting Star
        shooting_star = (
            upper_wick >= max(
                body * 2,
                candle_range * 0.35
            )
            and lower_wick <= candle_range * 0.20
            and body_ratio <= 0.40
        )

        # Bullish Pin Bar
        bullish_pin_bar = (
            lower_ratio >= 0.60
            and upper_ratio <= 0.20
            and body_ratio <= 0.30
            and c > l + candle_range * 0.60
        )

        # Bearish Pin Bar
        bearish_pin_bar = (
            upper_ratio >= 0.60
            and lower_ratio <= 0.20
            and body_ratio <= 0.30
            and c < l + candle_range * 0.40
        )

        return {

            "bullish_engulfing": bool(
                bullish_engulfing
            ),

            "bearish_engulfing": bool(
                bearish_engulfing
            ),

            "hammer": bool(hammer),

            "shooting_star": bool(
                shooting_star
            ),

            "bullish_pin_bar": bool(
                bullish_pin_bar
            ),

            "bearish_pin_bar": bool(
                bearish_pin_bar
            )
        }

    def scan_timeframe_patterns(self, df, timeframe):
        """
        Scan recent CLOSED candles for candlestick patterns.
        Returns pattern names in a consistent format.
        """
        if df is None or len(df) < 5:
            return []

        required_columns = {"open", "high", "low", "close"}
        if not required_columns.issubset(df.columns):
            return []

        patterns_found = []

        # Last candle may still be forming, so use -2.
        current = df.iloc[-2]
        previous = df.iloc[-3]

        detected = self.detect_patterns(current, previous)

        for pattern_name, is_detected in detected.items():
            if is_detected:
                patterns_found.append(pattern_name.upper())

        return patterns_found

    def scan_all_timeframe_patterns(self, frames):
        """
        Scan patterns across available timeframes.
        Output format is compatible with sniper_engine.py.
        """
        pattern_scan = {}

        for timeframe in self.timeframes:
            df = frames.get(timeframe)

            patterns = self.scan_timeframe_patterns(
                df,
                timeframe
            )

            pattern_scan[timeframe] = {
                "patterns": patterns,
                "count": len(patterns)
            }

        return pattern_scan

    def get_pattern_labels(self, pattern_scan):
        """
        Convert detected patterns into chart-friendly labels.
        """
        labels = []

        if not isinstance(pattern_scan, dict):
            return labels

        for timeframe, details in pattern_scan.items():
            if not isinstance(details, dict):
                continue

            patterns = details.get("patterns", [])

            if not isinstance(patterns, (list, tuple)):
                continue

            for pattern in patterns:
                labels.append(
                    f"{timeframe} {str(pattern).upper()}"
                )

        return labels

    def find_support_resistance(self, df, lookback=20):
        """
        Find recent support and resistance from closed candles.
        """
        if df is None or len(df) < lookback + 2:
            return None, None

        closed_data = df.iloc[:-1].tail(lookback)

        if closed_data.empty:
            return None, None

        support = float(closed_data["low"].min())
        resistance = float(closed_data["high"].max())

        if support <= 0 or resistance <= 0:
            return None, None

        if support >= resistance:
            return None, None

        return support, resistance

    def detect_sr_rejection(self, df, support, resistance):
        """
        Detect bullish rejection near support
        and bearish rejection near resistance.
        """
        if df is None or len(df) < 5:
            return None

        if support is None or resistance is None:
            return None

        current = df.iloc[-2]
        previous = df.iloc[-3]

        patterns = self.detect_patterns(current, previous)

        open_price = float(current["open"])
        high_price = float(current["high"])
        low_price = float(current["low"])
        close_price = float(current["close"])

        candle_range = high_price - low_price

        if candle_range <= 0:
            return None

        # Price tolerance: 15% of recent average candle range.
        recent = df.iloc[:-1].tail(14)
        average_range = float(
            (recent["high"] - recent["low"]).mean()
        )

        tolerance = max(average_range * 0.15, candle_range * 0.10)

        bullish_pattern = any(
            patterns.get(name, False)
            for name in [
                "bullish_engulfing",
                "hammer",
                "bullish_pin_bar"
            ]
        )

        bearish_pattern = any(
            patterns.get(name, False)
            for name in [
                "bearish_engulfing",
                "shooting_star",
                "bearish_pin_bar"
            ]
        )

        # Bullish rejection near support.
        support_rejection = (
            abs(low_price - support) <= tolerance
            or (low_price <= support and close_price > support)
        )

        if support_rejection and bullish_pattern and close_price > open_price:
            return {
                "direction": "BUY",
                "setup": "SUPPORT_REVERSAL",
                "level": support,
                "pattern": [
                    name.upper()
                    for name, detected in patterns.items()
                    if detected
                ]
            }

        # Bearish rejection near resistance.
        resistance_rejection = (
            abs(high_price - resistance) <= tolerance
            or (high_price >= resistance and close_price < resistance)
        )

        if resistance_rejection and bearish_pattern and close_price < open_price:
            return {
                "direction": "SELL",
                "setup": "RESISTANCE_REVERSAL",
                "level": resistance,
                "pattern": [
                    name.upper()
                    for name, detected in patterns.items()
                    if detected
                ]
            }

        return None

    def detect_liquidity_sweep(self, df, support, resistance):
        """
        Detect a sweep below support or above resistance
        followed by a close back inside the level.
        """
        if df is None or len(df) < 5:
            return None

        if support is None or resistance is None:
            return None

        current = df.iloc[-2]
        previous = df.iloc[-3]

        current_low = float(current["low"])
        current_high = float(current["high"])
        current_close = float(current["close"])
        previous_close = float(previous["close"])

        # Sweep below support and reclaim.
        if current_low < support and current_close > support:
            return {
                "direction": "BUY",
                "setup": "LIQUIDITY_SWEEP_BUY",
                "level": support
            }

        # Sweep above resistance and reject.
        if current_high > resistance and current_close < resistance:
            return {
                "direction": "SELL",
                "setup": "LIQUIDITY_SWEEP_SELL",
                "level": resistance
            }

        return None

    def analyze_market(self, frames, bid, ask, point):
        """
        Main analysis:
        1. Validate market data.
        2. Scan candlestick patterns.
        3. Check support/resistance rejection.
        4. Check liquidity sweep.
        5. Calculate entry, SL, TP and score.
        """

        pattern_scan = self.scan_all_timeframe_patterns(frames)
        pattern_labels = self.get_pattern_labels(pattern_scan)

        def no_signal(reason, extra=None):
            result = {
                "status": "NO_SIGNAL",
                "action": "NO_SIGNAL",
                "setup": None,
                "reason": reason,
                "pattern_scan": pattern_scan,
                "pattern_labels": pattern_labels
            }

            if extra:
                result.update(extra)

            return result

        # 1. Validate prices.
        try:
            bid = float(bid)
            ask = float(ask)
            point = float(point)
        except (TypeError, ValueError):
            return no_signal("Invalid bid, ask or point value")

        if bid <= 0 or ask <= 0 or point <= 0 or ask < bid:
            return no_signal("Invalid market prices")

        spread_points = (ask - bid) / point

        if spread_points > self.max_spread:
            return no_signal(
                "Spread is above allowed limit",
                {"spread_points": spread_points}
            )

        # 2. Validate M1 data.
        df = frames.get("M1")

        if df is None or len(df) < 25:
            return no_signal("Insufficient M1 candle data")

        required_columns = {"open", "high", "low", "close"}

        if not required_columns.issubset(df.columns):
            return no_signal("M1 candle columns are missing")

        # 3. Use only closed candles for analysis.
        closed_df = df.iloc[:-1].copy()

        if len(closed_df) < 20:
            return no_signal("Insufficient closed M1 candles")

        current = closed_df.iloc[-1]
        current_price = float(current["close"])

        # 4. Calculate ATR using closed candles.
        previous_close = closed_df["close"].shift(1)

        true_range = pd.concat(
            [
                closed_df["high"] - closed_df["low"],
                (closed_df["high"] - previous_close).abs(),
                (closed_df["low"] - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        atr = float(true_range.tail(14).mean())

        if not np.isfinite(atr) or atr <= 0:
            return no_signal("ATR calculation failed")

        # 5. Calculate trend and support/resistance.
        trends, market_trend = self.analyze_trends(frames)

        support, resistance = self.find_support_resistance(
            closed_df,
            lookback=20
        )

        if support is None or resistance is None:
            return no_signal(
                "Support or resistance not available",
                {"trends": trends, "market_trend": market_trend}
            )

        # 6. Check reversal and liquidity sweep setups.
        rejection = self.detect_sr_rejection(
            closed_df,
            support,
            resistance
        )

        sweep = self.detect_liquidity_sweep(
            closed_df,
            support,
            resistance
        )

        setup_signal = rejection or sweep

        if setup_signal is None:
            return no_signal(
                "No confirmed support/resistance rejection or liquidity sweep",
                {
                    "trends": trends,
                    "market_trend": market_trend,
                    "support": support,
                    "resistance": resistance,
                    "atr": atr
                }
            )

        direction = setup_signal["direction"]
        setup = setup_signal["setup"]

        # 7. Calculate entry, stop loss and target.
        if direction == "BUY":
            entry = ask
            stop_loss = min(
                support - (atr * 0.20),
                entry - (atr * 1.20)
            )

            risk = entry - stop_loss
            take_profit = entry + (risk * self.min_rr)

        else:
            entry = bid
            stop_loss = max(
                resistance + (atr * 0.20),
                entry + (atr * 1.20)
            )

            risk = stop_loss - entry
            take_profit = entry - (risk * self.min_rr)

        if risk <= 0:
            return no_signal("Invalid risk calculation")

        if direction == "BUY" and not (
            stop_loss < entry < take_profit
        ):
            return no_signal("Invalid BUY price levels")

        if direction == "SELL" and not (
            take_profit < entry < stop_loss
        ):
            return no_signal("Invalid SELL price levels")

        # 8. Calculate an internal setup score.
        score = 70

        if rejection:
            score += 5

        if sweep:
            score += 5

        if (
            direction == "BUY"
            and market_trend == "BULLISH"
        ) or (
            direction == "SELL"
            and market_trend == "BEARISH"
        ):
            score += 10

        score = min(score, 95)

        if score < self.min_score:
            return no_signal(
                "Setup score below minimum requirement",
                {
                    "score": score,
                    "trends": trends,
                    "market_trend": market_trend
                }
            )

        # 9. Return signal with diagnostics.
        return {
            "status": "SIGNAL",
            "action": direction,
            "direction": direction,
            "setup": setup,
            "entry": round(entry, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "score": score,
            "risk_reward": self.min_rr,
            "atr": round(atr, 5),
            "spread_points": round(spread_points, 2),
            "support": round(support, 5),
            "resistance": round(resistance, 5),
            "trends": trends,
            "market_trend": market_trend,
            "pattern_scan": pattern_scan,
            "pattern_labels": pattern_labels,
            "reason": setup
        }