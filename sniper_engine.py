import logging
import numpy as np
import pandas as pd

from logic import SignalAnalyzer

logger = logging.getLogger("GoldSniper.SignalEngine")


class SniperEngine:

    REQUIRED_TIMEFRAMES = [
        "M1",
        "M5",
        "M15",
        "M30",
        "H1"
    ]

    ALLOWED_SETUPS = {
        "SUPPORT_REVERSAL": "BUY",
        "RESISTANCE_REVERSAL": "SELL",
        "BULLISH_LIQUIDITY_SWEEP": "BUY",
        "BEARISH_LIQUIDITY_SWEEP": "SELL"
    }

    def __init__(self):

        self.analyzer = SignalAnalyzer()

        logger.info(
            "Filtered Sniper Engine initialized"
        )

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    def _validate_frames(self, frames):

        if not isinstance(frames, dict):
            return False, "Invalid market data"

        for timeframe in self.REQUIRED_TIMEFRAMES:

            df = frames.get(timeframe)

            if df is None:
                return False, (
                    f"Missing market data: {timeframe}"
                )

            if not isinstance(df, pd.DataFrame):
                return False, (
                    f"Invalid dataframe: {timeframe}"
                )

            if df.empty:
                return False, (
                    f"Empty market data: {timeframe}"
                )

            required_columns = [
                "open",
                "high",
                "low",
                "close"
            ]

            missing = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing:
                return False, (
                    f"{timeframe} missing columns: "
                    f"{', '.join(missing)}"
                )

            if len(df) < 3:
                return False, (
                    f"Insufficient candles: {timeframe}"
                )

        return True, ""

    # =====================================================
    # PRICE VALIDATION
    # =====================================================

    def _validate_prices(
        self,
        bid,
        ask,
        point
    ):

        try:
            bid = float(bid)
            ask = float(ask)
            point = float(point)

        except (TypeError, ValueError):
            return False, "Invalid price values"

        if not all(
            np.isfinite(value)
            for value in [bid, ask, point]
        ):
            return False, "Non-finite market prices"

        if bid <= 0:
            return False, "Invalid bid price"

        if ask < bid:
            return False, "Ask price is below bid"

        if point <= 0:
            return False, "Invalid point value"

        return True, ""

    # =====================================================
    # PATTERN INFORMATION
    # =====================================================

    def _extract_pattern_labels(self, pattern_scan):

        labels = []

        if not isinstance(pattern_scan, dict):
            return labels

        for timeframe, details in pattern_scan.items():

            if not isinstance(details, dict):
                continue

            patterns = details.get(
                "patterns",
                []
            )

            if not isinstance(patterns, (list, tuple)):
                continue

            for pattern in patterns:

                pattern_name = str(
                    pattern
                ).strip().upper()

                if not pattern_name:
                    continue

                labels.append(
                    f"{timeframe} {pattern_name}"
                )

        return labels

    def _preserve_diagnostics(self, result):

        if not isinstance(result, dict):
            return {}

        clean = dict(result)

        pattern_scan = clean.get(
            "pattern_scan",
            {}
        )

        if not isinstance(pattern_scan, dict):
            pattern_scan = {}

        clean["pattern_scan"] = pattern_scan

        # Preserve detected patterns even when
        # no BUY/SELL signal is generated.
        clean["pattern_labels"] = (
            self._extract_pattern_labels(
                pattern_scan
            )
        )

        return clean

    # =====================================================
    # NO SIGNAL RESPONSE
    # =====================================================

    def _no_signal(
        self,
        reason,
        **extra
    ):

        result = {
            "status": "NO_SIGNAL",
            "direction": None,
            "setup": None,
            "reason": reason,
            "pattern_labels": [],
            "pattern_scan": {}
        }

        result.update(extra)

        return self._preserve_diagnostics(
            result
        )

    # =====================================================
    # SIGNAL VALIDATION
    # =====================================================

    def _filter_signal(self, result):

        if not isinstance(result, dict):

            return self._no_signal(
                "Invalid analyzer response"
            )

        # Preserve diagnostic information first.
        result = self._preserve_diagnostics(
            result
        )

        if result.get("status") != "SIGNAL":

            return result

        setup = str(
            result.get("setup", "")
        ).strip().upper()

        direction = str(
            result.get("direction", "")
        ).strip().upper()

        expected_direction = (
            self.ALLOWED_SETUPS.get(setup)
        )

        # Reject unsupported setups.
        if expected_direction is None:

            logger.info(
                "Signal rejected: unsupported setup %s",
                setup
            )

            return self._no_signal(
                "Rejected: setup is not Support/Resistance "
                "or Liquidity Sweep reversal",
                rejected_setup=setup,
                pattern_scan=result.get(
                    "pattern_scan", {}
                )
            )

        # Validate direction.
        if direction != expected_direction:

            logger.warning(
                "Signal rejected: setup/direction mismatch"
            )

            return self._no_signal(
                "Rejected: setup and direction mismatch",
                pattern_scan=result.get(
                    "pattern_scan", {}
                )
            )

        return self._validate_signal_values(
            result,
            setup,
            direction
        )

    # =====================================================
    # SIGNAL PRICE / SCORE VALIDATION
    # =====================================================

    def _validate_signal_values(
        self,
        result,
        setup,
        direction
    ):

        try:
            entry = float(result.get("entry"))
            sl = float(result.get("sl"))
            tp = float(result.get("tp"))
            score = float(result.get("score"))

        except (TypeError, ValueError):

            return self._no_signal(
                "Rejected: missing or invalid signal values",
                pattern_scan=result.get(
                    "pattern_scan", {}
                )
            )

        values = [
            entry,
            sl,
            tp,
            score
        ]

        if not all(
            np.isfinite(value)
            for value in values
        ):

            return self._no_signal(
                "Rejected: non-finite signal values",
                pattern_scan=result.get(
                    "pattern_scan", {}
                )
            )

        if entry <= 0 or sl <= 0 or tp <= 0:

            return self._no_signal(
                "Rejected: invalid entry, SL or TP",
                pattern_scan=result.get(
                    "pattern_scan", {}
                )
            )

        # BUY validation
        if direction == "BUY":

            if not (sl < entry < tp):

                return self._no_signal(
                    "Rejected: invalid BUY SL/TP placement",
                    pattern_scan=result.get(
                        "pattern_scan", {}
                    )
                )

        # SELL validation
        elif direction == "SELL":

            if not (tp < entry < sl):

                return self._no_signal(
                    "Rejected: invalid SELL SL/TP placement",
                    pattern_scan=result.get(
                        "pattern_scan", {}
                    )
                )

        else:

            return self._no_signal(
                "Rejected: invalid signal direction",
                pattern_scan=result.get(
                    "pattern_scan", {}
                )
            )

        if score < 0 or score > 100:

            return self._no_signal(
                "Rejected: invalid signal score",
                pattern_scan=result.get(
                    "pattern_scan", {}
                )
            )

        # Keep verified signal information.
        clean_result = dict(result)

        clean_result.update({

            "status": "SIGNAL",
            "direction": direction,
            "setup": setup,

            "entry": entry,
            "sl": sl,
            "tp": tp,
            "score": score

        })

        # Keep diagnostic pattern information.
        clean_result = self._preserve_diagnostics(
            clean_result
        )

        return clean_result

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(
        self,
        frames,
        bid,
        ask,
        point=0.01
    ):

        try:

            # Validate market data
            valid, error = self._validate_frames(
                frames
            )

            if not valid:

                logger.warning(
                    "Market data validation failed: %s",
                    error
                )

                return self._no_signal(error)

            # Validate prices
            valid, error = self._validate_prices(
                bid,
                ask,
                point
            )

            if not valid:

                logger.warning(
                    "Price validation failed: %s",
                    error
                )

                return self._no_signal(error)

            # Run actual strategy analyzer
            result = self.analyzer.analyze_market(
                frames=frames,
                bid=float(bid),
                ask=float(ask),
                point=float(point)
            )

            if not isinstance(result, dict):

                logger.error(
                    "Invalid analyzer response"
                )

                return self._no_signal(
                    "Invalid analyzer response"
                )

            # Filter signal without deleting
            # detected pattern diagnostics.
            result = self._filter_signal(result)

            if result.get("status") == "SIGNAL":

                logger.info(
                    "VERIFIED SIGNAL | "
                    "Direction: %s | "
                    "Setup: %s | "
                    "Score: %s | "
                    "Entry: %s | "
                    "SL: %s | "
                    "TP: %s",

                    result.get("direction"),
                    result.get("setup"),
                    result.get("score"),
                    result.get("entry"),
                    result.get("sl"),
                    result.get("tp")
                )

            else:

                logger.debug(
                    "No verified signal: %s",
                    result.get("reason")
                )

            return result

        except Exception as exc:

            logger.exception(
                "Signal analysis failed"
            )

            return {
                "status": "ERROR",
                "direction": None,
                "setup": None,
                "reason": str(exc),
                "pattern_labels": [],
                "pattern_scan": {}
            }


# =========================================================
# SINGLE SHARED ENGINE INSTANCE
# =========================================================

_engine = SniperEngine()


# =========================================================
# BACKWARD COMPATIBILITY
# Preserve existing main.py import interface
# =========================================================

def analyse_market(
    frames,
    bid,
    ask,
    point=0.01
):

    return _engine.analyze(
        frames=frames,
        bid=bid,
        ask=ask,
        point=point
    )