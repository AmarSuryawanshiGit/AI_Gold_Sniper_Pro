
from __future__ import annotations

import pandas as pd
import numpy as np


class CandlestickStrategy:
    """
    Multi-timeframe Support/Resistance Reversal Strategy.

    Timeframes:
        M1, M5, M15, M30, H1

    Verified pattern requirements:
        1. Confirmed support/resistance location
        2. Valid candlestick formation
        3. Next closed candle confirmation
        4. Indicator confluence score

    H1 -> Retest -> M1 entry flow is preserved.
    """

    BULLISH_PATTERNS = [
        "HAMMER",
        "BULLISH_PIN_BAR",
        "BULLISH_ENGULFING",
        "MORNING_STAR"
    ]

    BEARISH_PATTERNS = [
        "SHOOTING_STAR",
        "BEARISH_PIN_BAR",
        "BEARISH_ENGULFING",
        "EVENING_STAR"
    ]

    TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1"]

    TF_CONFIG = {
        "M1":  {"score": 4, "sr_tolerance": 0.45},
        "M5":  {"score": 4, "sr_tolerance": 0.40},
        "M15": {"score": 4, "sr_tolerance": 0.35},
        "M30": {"score": 4, "sr_tolerance": 0.35},
        "H1":  {"score": 4, "sr_tolerance": 0.30},
    }

    def __init__(
        self,
        bb_period=20,
        bb_std=2.0,
        retest_atr=0.20,
        max_wait_h1=12,
        sr_lookback=120,
        sr_left=2,
        sr_right=2,
        sr_atr_tolerance=0.35,
        ema_max_wait=30,
        ema_cooldown=3
    ):

        self.bb_period = bb_period
        self.bb_std = bb_std

        self.retest_atr = retest_atr
        self.max_wait_h1 = max_wait_h1

        self.sr_lookback = sr_lookback
        self.sr_left = sr_left
        self.sr_right = sr_right
        self.sr_atr_tolerance = sr_atr_tolerance

        self.ema_max_wait = ema_max_wait
        self.ema_cooldown = ema_cooldown

        self.h1_setup = None
        self.last_h1_time = None
        self.h1_last_m1_time = None

        self.ema_direction = None
        self.ema_retest_seen = False
        self.ema_setup_age = 0
        self.ema_cooldown_left = 0
        self.ema_last_m1_time = None

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.h1_setup = None
        self.last_h1_time = None
        self.h1_last_m1_time = None

        self.ema_direction = None
        self.ema_retest_seen = False
        self.ema_setup_age = 0
        self.ema_cooldown_left = 0
        self.ema_last_m1_time = None

    # =====================================================
    # SAFE NUMBER
    # =====================================================

    @staticmethod
    def number(value):

        try:
            result = float(value)

            if np.isfinite(result):
                return result

        except (TypeError, ValueError):
            pass

        return None

    # =====================================================
    # CLOSED CANDLES
    # =====================================================

    def closed(self, df):

        if df is None or len(df) < 3:
            return pd.DataFrame()

        x = df.copy()

        required = ["open", "high", "low", "close"]

        if not all(col in x.columns for col in required):
            return pd.DataFrame()

        for col in required:
            x[col] = pd.to_numeric(
                x[col],
                errors="coerce"
            )

        x = x.dropna(
            subset=required
        ).sort_index()

        x = x[
            ~x.index.duplicated(keep="last")
        ]

        # MT5 last candle is assumed to be forming.
        return x.iloc[:-1].copy()

    # =====================================================
    # INDICATORS
    # =====================================================

    def indicators(self, df):

        x = df.copy()

        close = x["close"]
        high = x["high"]
        low = x["low"]

        x["ema9"] = close.ewm(
            span=9,
            adjust=False
        ).mean()

        x["ema20"] = close.ewm(
            span=20,
            adjust=False
        ).mean()

        previous_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        x["atr"] = tr.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        # RSI 14
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        x["rsi"] = 100 - (100 / (1 + rs))

        middle = close.rolling(
            self.bb_period
        ).mean()

        std = close.rolling(
            self.bb_period
        ).std(ddof=0)

        x["bb_middle"] = middle
        x["bb_upper"] = middle + self.bb_std * std
        x["bb_lower"] = middle - self.bb_std * std

        return x

    # =====================================================
    # CANDLE PATTERN DETECTION
    # =====================================================

    @staticmethod
    def detect_pattern(cur, prev=None, prev2=None):

        o = float(cur.open)
        h = float(cur.high)
        l = float(cur.low)
        c = float(cur.close)

        candle_range = h - l

        if candle_range <= 0:
            return []

        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l

        body_ratio = body / candle_range

        patterns = []

        # Hammer
        if (
            lower / candle_range >= 0.50
            and upper / candle_range <= 0.20
            and body_ratio <= 0.35
            and c >= l + candle_range * 0.55
        ):
            patterns.append("HAMMER")

        # Shooting Star
        if (
            upper / candle_range >= 0.50
            and lower / candle_range <= 0.20
            and body_ratio <= 0.35
            and c <= l + candle_range * 0.45
        ):
            patterns.append("SHOOTING_STAR")

        # Bullish Pin Bar
        if (
            lower / candle_range >= 0.60
            and upper / candle_range <= 0.18
            and body_ratio <= 0.30
            and c >= l + candle_range * 0.65
        ):
            patterns.append("BULLISH_PIN_BAR")

        # Bearish Pin Bar
        if (
            upper / candle_range >= 0.60
            and lower / candle_range <= 0.18
            and body_ratio <= 0.30
            and c <= l + candle_range * 0.35
        ):
            patterns.append("BEARISH_PIN_BAR")

        if prev is not None:

            po = float(prev.open)
            pc = float(prev.close)

            # Bullish Engulfing
            if (
                c > o
                and pc < po
                and o <= pc
                and c >= po
                and body_ratio >= 0.35
            ):
                patterns.append("BULLISH_ENGULFING")

            # Bearish Engulfing
            if (
                c < o
                and pc > po
                and o >= pc
                and c <= po
                and body_ratio >= 0.35
            ):
                patterns.append("BEARISH_ENGULFING")

        if prev is not None and prev2 is not None:

            p2 = prev2
            p1 = prev

            p2_body = abs(
                float(p2.close) - float(p2.open)
            )

            p1_body = abs(
                float(p1.close) - float(p1.open)
            )

            p2_range = max(
                float(p2.high) - float(p2.low),
                1e-10
            )

            p1_range = max(
                float(p1.high) - float(p1.low),
                1e-10
            )

            # Morning Star
            if (
                float(p2.close) < float(p2.open)
                and p2_body / p2_range >= 0.50
                and p1_body / p1_range <= 0.35
                and c > o
                and body_ratio >= 0.40
                and c > (
                    float(p2.open) + float(p2.close)
                ) / 2
            ):
                patterns.append("MORNING_STAR")

            # Evening Star
            if (
                float(p2.close) > float(p2.open)
                and p2_body / p2_range >= 0.50
                and p1_body / p1_range <= 0.35
                and c < o
                and body_ratio >= 0.40
                and c < (
                    float(p2.open) + float(p2.close)
                ) / 2
            ):
                patterns.append("EVENING_STAR")

        return patterns

    @staticmethod
    def is_bullish_pattern(patterns):

        return any(
            p in CandlestickStrategy.BULLISH_PATTERNS
            for p in patterns
        )

    @staticmethod
    def is_bearish_pattern(patterns):

        return any(
            p in CandlestickStrategy.BEARISH_PATTERNS
            for p in patterns
        )

    # =====================================================
    # SUPPORT / RESISTANCE PIVOTS
    # =====================================================

    def find_sr_levels(self, df, lookback=None):

        if df is None:
            return [], []

        lookback = (
            lookback
            if lookback is not None
            else self.sr_lookback
        )

        x = df.tail(lookback).copy()

        left = self.sr_left
        right = self.sr_right

        if len(x) < left + right + 5:
            return [], []

        lows = x["low"].astype(float).to_numpy()
        highs = x["high"].astype(float).to_numpy()

        supports = []
        resistances = []

        for i in range(left, len(x) - right):

            low_window = lows[
                i - left:i + right + 1
            ]

            high_window = highs[
                i - left:i + right + 1
            ]

            if lows[i] <= np.min(low_window):
                supports.append(float(lows[i]))

            if highs[i] >= np.max(high_window):
                resistances.append(float(highs[i]))

        return supports, resistances

    # =====================================================
    # CLUSTER SUPPORT / RESISTANCE
    # =====================================================

    def cluster_levels(self, levels, tolerance):

        if not levels:
            return []

        levels = sorted(float(v) for v in levels)

        clusters = []
        current = [levels[0]]

        for level in levels[1:]:

            center = float(np.mean(current))

            if abs(level - center) <= tolerance:
                current.append(level)

            else:
                clusters.append(float(np.mean(current)))
                current = [level]

        clusters.append(float(np.mean(current)))

        return clusters

    # =====================================================
    # SUPPORT / RESISTANCE LOCATION
    # =====================================================

    def get_sr_location(self, history, candle, timeframe=None):

        empty = {
            "at_support": False,
            "at_resistance": False,
            "support": None,
            "resistance": None,
            "tolerance": None
        }

        if history is None or len(history) < 10:
            return empty

        atr = self.number(candle.get("atr"))

        if atr is None or atr <= 0:
            return empty

        config = self.TF_CONFIG.get(
            timeframe,
            {"sr_tolerance": self.sr_atr_tolerance}
        )

        tolerance = max(
            atr * config["sr_tolerance"],
            1e-8
        )

        supports, resistances = self.find_sr_levels(history)

        supports = self.cluster_levels(supports, tolerance)
        resistances = self.cluster_levels(resistances, tolerance)

        candle_low = float(candle["low"])
        candle_high = float(candle["high"])

        nearby_supports = [
            level for level in supports
            if abs(candle_low - level) <= tolerance
        ]

        nearby_resistances = [
            level for level in resistances
            if abs(candle_high - level) <= tolerance
        ]

        support = (
            min(
                nearby_supports,
                key=lambda v: abs(candle_low - v)
            )
            if nearby_supports else None
        )

        resistance = (
            min(
                nearby_resistances,
                key=lambda v: abs(candle_high - v)
            )
            if nearby_resistances else None
        )

        return {
            "at_support": support is not None,
            "at_resistance": resistance is not None,
            "support": support,
            "resistance": resistance,
            "tolerance": tolerance
        }

    # =====================================================
    # BOLLINGER BAND CONFLUENCE
    # =====================================================

    def bb_confluence(self, candle, direction):

        atr = self.number(candle.get("atr"))

        if atr is None or atr <= 0:
            return False

        tolerance = atr * self.retest_atr

        if direction == "BUY":
            return (
                float(candle["low"])
                <= float(candle["bb_lower"]) + tolerance
            )

        return (
            float(candle["high"])
            >= float(candle["bb_upper"]) - tolerance
        )

    # =====================================================
    # INDICATOR CONFLUENCE SCORE
    # =====================================================

    def confluence_score(
        self,
        candidate,
        confirmation,
        direction,
        volume_col=None
    ):

        score = 0
        reasons = []

        rsi = self.number(candidate.get("rsi"))
        previous_rsi = self.number(confirmation.get("rsi"))

        ema9 = self.number(candidate.get("ema9"))
        ema20 = self.number(candidate.get("ema20"))

        close = float(candidate["close"])
        confirm_close = float(confirmation["close"])

        # RSI momentum
        if rsi is not None and previous_rsi is not None:

            if (
                direction == "BUY"
                and rsi <= 50
                and previous_rsi > rsi
            ):
                score += 1
                reasons.append("RSI_RECOVERY")

            elif (
                direction == "SELL"
                and rsi >= 50
                and previous_rsi < rsi
            ):
                score += 1
                reasons.append("RSI_WEAKNESS")

        # EMA trend
        if ema9 is not None and ema20 is not None:

            if (
                direction == "BUY"
                and close > ema9
                and ema9 >= ema20
            ):
                score += 1
                reasons.append("EMA_BULLISH")

            elif (
                direction == "SELL"
                and close < ema9
                and ema9 <= ema20
            ):
                score += 1
                reasons.append("EMA_BEARISH")

        # Bollinger Band
        if self.bb_confluence(candidate, direction):
            score += 1
            reasons.append("BB_CONFLUENCE")

        # Price confirmation
        if (
            direction == "BUY"
            and confirm_close > close
        ):
            score += 1
            reasons.append("PRICE_CONFIRMATION")

        elif (
            direction == "SELL"
            and confirm_close < close
        ):
            score += 1
            reasons.append("PRICE_CONFIRMATION")

        # Volume spike
        if volume_col:

            current_volume = self.number(
                candidate.get(volume_col)
            )

            average_volume = self.number(
                candidate.get("volume_average")
            )

            if (
                current_volume is not None
                and average_volume is not None
                and average_volume > 0
                and current_volume >= average_volume * 1.20
            ):
                score += 1
                reasons.append("VOLUME_SPIKE")

        return score, reasons

    # =====================================================
    # MULTI-TIMEFRAME VERIFIED SCANNER
    # =====================================================

    def scan_patterns(self, frames):

        result = {}

        for tf in self.TIMEFRAMES:

            empty = {
                "patterns": [],
                "labels": [],
                "verified": False,
                "status": "NO_VERIFIED_PATTERN",
                "direction": None,
                "location": None,
                "level": None,
                "score": 0,
                "reasons": [],
                "bb_confluence": False,
                "candle_time": None,
                "confirmation_time": None
            }

            raw = self.closed(
                frames.get(tf) if frames else None
            )

            if len(raw) < self.bb_period + 30:
                result[tf] = empty.copy()
                continue

            x = self.indicators(raw)

            candidate_pos = len(x) - 2

            candidate = x.iloc[candidate_pos]
            confirmation = x.iloc[candidate_pos + 1]

            previous = x.iloc[candidate_pos - 1]

            previous2 = (
                x.iloc[candidate_pos - 2]
                if candidate_pos >= 2 else None
            )

            history = x.iloc[
                max(0, candidate_pos - self.sr_lookback):
                candidate_pos
            ]

            if len(history) < 20:
                result[tf] = empty.copy()
                continue

            patterns = self.detect_pattern(
                candidate,
                previous,
                previous2
            )

            if not patterns:
                result[tf] = empty.copy()
                continue

            location = self.get_sr_location(
                history,
                candidate,
                tf
            )

            bullish = self.is_bullish_pattern(patterns)
            bearish = self.is_bearish_pattern(patterns)

            buy_confirmed = (
                bullish
                and location["at_support"]
                and float(confirmation.close) > float(candidate.high)
                and float(confirmation.close) > float(confirmation.open)
            )

            sell_confirmed = (
                bearish
                and location["at_resistance"]
                and float(confirmation.close) < float(candidate.low)
                and float(confirmation.close) < float(confirmation.open)
            )

            if not buy_confirmed and not sell_confirmed:
                result[tf] = empty.copy()
                continue

            direction = (
                "BUY" if buy_confirmed else "SELL"
            )

            selected = [
                p for p in patterns
                if (
                    p in self.BULLISH_PATTERNS
                    if direction == "BUY"
                    else p in self.BEARISH_PATTERNS
                )
            ]

            volume_col = next(
                (
                    col for col in [
                        "tick_volume",
                        "real_volume",
                        "volume"
                    ]
                    if col in x.columns
                ),
                None
            )

            if volume_col:

                x["volume_average"] = pd.to_numeric(
                    x[volume_col],
                    errors="coerce"
                ).rolling(20).mean()

                candidate = x.iloc[candidate_pos]

            score, reasons = self.confluence_score(
                candidate,
                confirmation,
                direction,
                volume_col
            )

            minimum_score = self.TF_CONFIG[tf]["score"]

            if score < minimum_score:

                result[tf] = empty.copy()
                result[tf]["status"] = "FILTERED_LOW_SCORE"
                result[tf]["score"] = score
                result[tf]["reasons"] = reasons

                continue

            market_location = (
                "SUPPORT" if direction == "BUY"
                else "RESISTANCE"
            )

            level = (
                location["support"]
                if direction == "BUY"
                else location["resistance"]
            )

            labels = [
                f"{tf} {direction} {p} @ {market_location}"
                for p in selected
            ]

            result[tf] = {
                "patterns": selected,
                "labels": labels,
                "verified": True,
                "status": "SR_CONFIRMED",
                "direction": direction,
                "location": market_location,
                "level": level,
                "score": score,
                "reasons": reasons,
                "bb_confluence": self.bb_confluence(
                    candidate,
                    direction
                ),
                "candle_time": str(x.index[candidate_pos]),
                "confirmation_time": str(x.index[candidate_pos + 1])
            }

        return result


    # =====================================================
    # H1 SETUP DETECTION
    # =====================================================

    def find_h1_setup(self, frames):

        h1 = self.closed(
            frames.get("H1") if frames else None
        )

        if len(h1) < self.bb_period + 30:
            return

        x = self.indicators(h1)

        cur = x.iloc[-1]
        prev = x.iloc[-2]
        prev2 = x.iloc[-3]

        stamp = str(x.index[-1])

        if stamp == self.last_h1_time:
            return

        self.last_h1_time = stamp

        if self.h1_setup is not None:
            return

        patterns = self.detect_pattern(
            cur,
            prev,
            prev2
        )

        history = x.iloc[:-1]

        location = self.get_sr_location(
            history,
            cur,
            "H1"
        )

        buy_valid = (
            self.is_bullish_pattern(patterns)
            and location["at_support"]
        )

        sell_valid = (
            self.is_bearish_pattern(patterns)
            and location["at_resistance"]
        )

        if buy_valid:

            selected = [
                p for p in patterns
                if p in self.BULLISH_PATTERNS
            ]

            self.h1_setup = {
                "direction": "BUY",
                "patterns": selected,
                "pattern": selected[0],
                "pattern_time": stamp,
                "pattern_high": float(cur.high),
                "pattern_low": float(cur.low),
                "support_level": location["support"],
                "sr_tolerance": location["tolerance"],
                "bb_confluence": self.bb_confluence(
                    cur,
                    "BUY"
                ),
                "state": "WAIT_CONFIRMATION"
            }

        elif sell_valid:

            selected = [
                p for p in patterns
                if p in self.BEARISH_PATTERNS
            ]

            self.h1_setup = {
                "direction": "SELL",
                "patterns": selected,
                "pattern": selected[0],
                "pattern_time": stamp,
                "pattern_high": float(cur.high),
                "pattern_low": float(cur.low),
                "resistance_level": location["resistance"],
                "sr_tolerance": location["tolerance"],
                "bb_confluence": self.bb_confluence(
                    cur,
                    "SELL"
                ),
                "state": "WAIT_CONFIRMATION"
            }

    # =====================================================
    # H1 NEXT CANDLE CONFIRMATION
    # =====================================================

    def process_h1_setup(self, frames):

        if self.h1_setup is None:
            return {"status": "NO_SETUP"}

        h1 = self.closed(
            frames.get("H1") if frames else None
        )

        if len(h1) < 3:
            return {"status": "WAITING"}

        setup = self.h1_setup

        try:
            pos = list(map(str, h1.index)).index(
                setup["pattern_time"]
            )

        except ValueError:

            self.h1_setup = None

            return {"status": "EXPIRED"}

        age = len(h1) - 1 - pos

        if age > self.max_wait_h1:

            self.h1_setup = None

            return {"status": "EXPIRED"}

        if setup["state"] == "WAIT_CONFIRMATION":

            if age < 1:
                return {
                    "status": "WAIT_CONFIRMATION",
                    **setup
                }

            # Immediate next H1 candle only.
            next_h1 = h1.iloc[pos + 1]

            if setup["direction"] == "BUY":

                confirmed = (
                    float(next_h1.close)
                    > setup["pattern_high"]
                    and float(next_h1.close)
                    > float(next_h1.open)
                )

            else:

                confirmed = (
                    float(next_h1.close)
                    < setup["pattern_low"]
                    and float(next_h1.close)
                    < float(next_h1.open)
                )

            if not confirmed:

                self.h1_setup = None

                return {
                    "status": "INVALIDATED",
                    "reason": (
                        "Next H1 candle did not confirm pattern"
                    )
                }

            setup["state"] = "WAIT_RETEST"

            setup["confirmation_time"] = str(
                h1.index[pos + 1]
            )

        return {
            "status": setup["state"],
            **setup
        }

    # =====================================================
    # H1 RETEST + M1 ENTRY
    # =====================================================

    def h1_m1_entry(self, frames):

        setup = self.h1_setup

        if (
            not setup
            or setup["state"] != "WAIT_RETEST"
        ):
            return {"status": "WAITING"}

        m1 = self.closed(
            frames.get("M1") if frames else None
        )

        if len(m1) < 35:
            return {"status": "WAITING"}

        x = self.indicators(m1)

        cur = x.iloc[-1]
        prev = x.iloc[-2]
        prev2 = x.iloc[-3]

        stamp = str(x.index[-1])

        if stamp == self.h1_last_m1_time:
            return {"status": "WAITING"}

        self.h1_last_m1_time = stamp

        atr = self.number(cur.get("atr"))

        if atr is None or atr <= 0:
            return {"status": "WAITING"}

        # Use M1 volatility for M1 retest tolerance.
        tolerance = atr * self.retest_atr

        if setup["direction"] == "BUY":
            level = setup.get("support_level")
        else:
            level = setup.get("resistance_level")

        if level is None:

            self.h1_setup = None

            return {
                "status": "INVALIDATED",
                "reason": "Missing Support/Resistance level"
            }

        level = float(level)

        touched = (
            float(cur.low) <= level + tolerance
            and float(cur.high) >= level - tolerance
        )

        if "retest_time" not in setup:

            if touched:

                setup["retest_time"] = stamp

                return {
                    "status": "RETEST_SEEN",
                    "retest_time": stamp,
                    "level": level
                }

            return {
                "status": "WAIT_RETEST",
                "level": level
            }

        if stamp <= setup["retest_time"]:
            return {"status": "WAIT_M1_CONFIRMATION"}

        near_zone = (
            float(cur.low) <= level + tolerance
            and float(cur.high) >= level - tolerance
        )

        if not near_zone:

            return {
                "status": "WAIT_M1_CONFIRMATION",
                "reason": "Waiting for M1 rejection at S/R"
            }

        patterns = self.detect_pattern(
            cur,
            prev,
            prev2
        )

        # M1 indicator confirmation.
        ema9 = self.number(cur.get("ema9"))
        ema20 = self.number(cur.get("ema20"))
        rsi = self.number(cur.get("rsi"))

        if setup["direction"] == "BUY":

            indicator_ok = (
                ema9 is not None
                and ema20 is not None
                and rsi is not None
                and ema9 >= ema20
                and rsi >= 40
            )

            valid = (
                float(cur.close) > float(cur.open)
                and float(cur.close) > float(prev.close)
                and float(cur.close) > level
                and self.is_bullish_pattern(patterns)
                and indicator_ok
            )

        else:

            indicator_ok = (
                ema9 is not None
                and ema20 is not None
                and rsi is not None
                and ema9 <= ema20
                and rsi <= 60
            )

            valid = (
                float(cur.close) < float(cur.open)
                and float(cur.close) < float(prev.close)
                and float(cur.close) < level
                and self.is_bearish_pattern(patterns)
                and indicator_ok
            )

        if not valid:

            return {
                "status": "WAIT_M1_CONFIRMATION",
                "patterns": patterns
            }

        signal = {
            "status": "SIGNAL",
            "strategy": "H1_SR_RETEST_M1",
            "direction": setup["direction"],
            "entry_price": float(cur.close),
            "entry_time": stamp,
            "h1_pattern": setup["pattern"],
            "h1_patterns": setup.get("patterns", []),
            "h1_pattern_time": setup["pattern_time"],
            "h1_confirmation_time": setup["confirmation_time"],
            "retest_time": setup["retest_time"],
            "support_level": setup.get("support_level"),
            "resistance_level": setup.get("resistance_level"),
            "bb_confluence": setup.get("bb_confluence", False),
            "m1_patterns": patterns,
            "m1_rsi": rsi,
            "m1_ema9": ema9,
            "m1_ema20": ema20
        }

        self.h1_setup = None

        return signal

    # =====================================================
    # EMA9 STRATEGY - DISABLED
    # =====================================================

    def ema9_strategy(self, frames):

        return {
            "status": "DISABLED",
            "reason": (
                "Standalone EMA signals disabled. "
                "EMA is used only as confluence."
            )
        }

    # =====================================================
    # MAIN EVALUATION
    # =====================================================

    def evaluate(self, frames):

        if not isinstance(frames, dict):
            frames = {}

        # Scan all five timeframes.
        pattern_scan = self.scan_patterns(frames)

        # H1 setup lifecycle.
        self.find_h1_setup(frames)

        h1_status = self.process_h1_setup(
            frames
        )

        h1_entry = self.h1_m1_entry(
            frames
        )

        ema_entry = self.ema9_strategy(
            frames
        )

        return {
            "pattern_scan": pattern_scan,

            "h1_strategy": {
                "setup": h1_status,
                "entry": h1_entry
            },

            "ema9_strategy": ema_entry
        }
