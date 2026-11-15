
from __future__ import annotations

import os
from pathlib import Path


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_FILE, override=False)

except ImportError:
    print(
        "WARNING: python-dotenv is not installed. "
        "Reading environment variables from Windows."
    )


def get_env(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return str(value).strip()


def get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)

    if value is None:
        return default

    return value.strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def get_int(key: str, default: int) -> int:
    try:
        return int(get_env(key, str(default)))
    except (TypeError, ValueError):
        return default


def get_float(key: str, default: float) -> float:
    try:
        value = float(get_env(key, str(default)))
        return value if value == value and abs(value) != float("inf") else default
    except (TypeError, ValueError):
        return default


# =========================================================
# APPLICATION
# =========================================================

APP_NAME = "AI Gold Sniper Pro"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "sniper.db"
LOG_PATH = DATA_DIR / "sniper.log"


# =========================================================
# METATRADER 5 / XM CONFIGURATION
# =========================================================

MT5_SYMBOL = get_env("MT5_SYMBOL", "GOLD")

SYMBOL_CANDIDATES = [
    MT5_SYMBOL,
    "GOLD",
    "XAUUSD",
    "GOLD.",
    "XAUUSD.",
    "GOLDm",
    "XAUUSDm",
]

SYMBOL_CANDIDATES = list(dict.fromkeys(SYMBOL_CANDIDATES))


# =========================================================
# TIMEFRAME CONFIGURATION
# =========================================================

TIMEFRAMES = [
    "M1",
    "M5",
    "M15",
    "M30",
    "H1",
]

# Timeframes used by the candlestick scanner
CANDLE_PATTERN_TIMEFRAMES = [
    "H1",
    "M15",
    "M5",
    "M1",
]

BARS_TO_LOAD = get_int("BARS_TO_LOAD", 500)

SCAN_INTERVAL_SECONDS = get_int(
    "SCAN_INTERVAL_SECONDS",
    2
)


# =========================================================
# SIGNAL CONFIGURATION
# =========================================================

MIN_SIGNAL_SCORE = get_int(
    "MIN_SIGNAL_SCORE",
    70
)

MAX_SPREAD_POINTS = get_int(
    "MAX_SPREAD_POINTS",
    100
)

COOLDOWN_CANDLES = get_int(
    "COOLDOWN_CANDLES",
    3
)

MAX_SIGNALS_PER_HOUR = get_int(
    "MAX_SIGNALS_PER_HOUR",
    8
)


# =========================================================
# EMA CONFIGURATION
# =========================================================

EMA_PERIOD = get_int(
    "EMA_PERIOD",
    9
)

EMA_TREND_FAST = get_int(
    "EMA_TREND_FAST",
    20
)

EMA_TREND_MEDIUM = get_int(
    "EMA_TREND_MEDIUM",
    50
)

EMA_TREND_SLOW = get_int(
    "EMA_TREND_SLOW",
    200
)

# Entry confirmation using 9 EMA
ENABLE_EMA_RETEST = get_bool(
    "ENABLE_EMA_RETEST",
    True
)


# =========================================================
# BOLLINGER BANDS CONFIGURATION
# =========================================================

BB_PERIOD = get_int(
    "BB_PERIOD",
    20
)

BB_STD_DEV = get_float(
    "BB_STD_DEV",
    2.0
)

ENABLE_BB_CONFLUENCE = get_bool(
    "ENABLE_BB_CONFLUENCE",
    True
)

BB_TOUCH_TOLERANCE_ATR = get_float(
    "BB_TOUCH_TOLERANCE_ATR",
    0.20
)


# =========================================================
# CANDLESTICK PATTERN CONFIGURATION
# =========================================================

ENABLE_CANDLE_PATTERN_SCANNER = get_bool(
    "ENABLE_CANDLE_PATTERN_SCANNER",
    True
)

ENABLE_HAMMER = get_bool(
    "ENABLE_HAMMER",
    True
)

ENABLE_SHOOTING_STAR = get_bool(
    "ENABLE_SHOOTING_STAR",
    True
)

ENABLE_ENGULFING = get_bool(
    "ENABLE_ENGULFING",
    True
)

ENABLE_PIN_BAR = get_bool(
    "ENABLE_PIN_BAR",
    True
)

# Pattern detection uses completed candles only
PATTERN_CLOSED_CANDLE_ONLY = get_bool(
    "PATTERN_CLOSED_CANDLE_ONLY",
    True
)


# =========================================================
# H1 CONFIRMATION AND RETEST CONFIGURATION
# =========================================================

ENABLE_H1_CONFIRMATION = get_bool(
    "ENABLE_H1_CONFIRMATION",
    True
)

ENABLE_H1_RETEST_ENTRY = get_bool(
    "ENABLE_H1_RETEST_ENTRY",
    True
)

ENABLE_M1_ENTRY_CONFIRMATION = get_bool(
    "ENABLE_M1_ENTRY_CONFIRMATION",
    True
)

RETEST_TOLERANCE_ATR = get_float(
    "RETEST_TOLERANCE_ATR",
    0.20
)

RETEST_MAX_WAIT_CANDLES = get_int(
    "RETEST_MAX_WAIT_CANDLES",
    60
)


# =========================================================
# SUPPORT / RESISTANCE CONFIGURATION
# =========================================================

SR_LOOKBACK = get_int(
    "SR_LOOKBACK",
    20
)

SR_TOLERANCE_ATR = get_float(
    "SR_TOLERANCE_ATR",
    0.20
)

ENABLE_SR_CONFLUENCE = get_bool(
    "ENABLE_SR_CONFLUENCE",
    True
)


# =========================================================
# RISK / TARGET CONFIGURATION
# Distances are in Gold price units, not INR.
# =========================================================

ATR_SL_MULTIPLIER = get_float(
    "ATR_SL_MULTIPLIER",
    1.2
)

ATR_TP_MULTIPLIER = get_float(
    "ATR_TP_MULTIPLIER",
    2.0
)

MIN_SL_DISTANCE = get_float(
    "MIN_SL_DISTANCE",
    0.80
)

MAX_SL_DISTANCE = get_float(
    "MAX_SL_DISTANCE",
    8.00
)

MIN_TP_DISTANCE = get_float(
    "MIN_TP_DISTANCE",
    1.00
)

MAX_TP_DISTANCE = get_float(
    "MAX_TP_DISTANCE",
    12.00
)

MIN_RISK_REWARD = get_float(
    "MIN_RISK_REWARD",
    1.5
)


# =========================================================
# STRATEGY CONFIGURATION
# =========================================================

SIGNAL_ON_CLOSED_M1 = get_bool(
    "SIGNAL_ON_CLOSED_M1",
    True
)

ALLOW_COUNTER_TREND_REVERSAL = get_bool(
    "ALLOW_COUNTER_TREND_REVERSAL",
    True
)

ALLOW_RANGE_REVERSAL = get_bool(
    "ALLOW_RANGE_REVERSAL",
    True
)

ALLOW_BREAKOUT = get_bool(
    "ALLOW_BREAKOUT",
    True
)


# =========================================================
# TELEGRAM CONFIGURATION
# Never hardcode bot token in Python files.
# =========================================================

TELEGRAM_BOT_TOKEN = get_env(
    "TELEGRAM_BOT_TOKEN",
    ""
)

TELEGRAM_CHAT_ID = get_env(
    "TELEGRAM_CHAT_ID",
    ""
)

TELEGRAM_ENABLED = (
    get_bool("TELEGRAM_ENABLED", True)
    and bool(TELEGRAM_BOT_TOKEN)
    and bool(TELEGRAM_CHAT_ID)
)


# =========================================================
# APPLICATION MODE
# Signals only. No automatic order execution.
# =========================================================

TRADING_MODE = get_env(
    "TRADING_MODE",
    "SIGNAL_ONLY"
).upper()

AUTO_TRADING_ENABLED = False


# =========================================================
# CONFIGURATION VALIDATION
# =========================================================

BARS_TO_LOAD = max(250, BARS_TO_LOAD)

SCAN_INTERVAL_SECONDS = max(
    1,
    SCAN_INTERVAL_SECONDS
)

MIN_SIGNAL_SCORE = (
    MIN_SIGNAL_SCORE
    if 0 <= MIN_SIGNAL_SCORE <= 100
    else 70
)

MAX_SPREAD_POINTS = max(
    1,
    MAX_SPREAD_POINTS
)

COOLDOWN_CANDLES = max(
    0,
    COOLDOWN_CANDLES
)

MAX_SIGNALS_PER_HOUR = max(
    1,
    MAX_SIGNALS_PER_HOUR
)

EMA_PERIOD = max(1, EMA_PERIOD)
EMA_TREND_FAST = max(1, EMA_TREND_FAST)
EMA_TREND_MEDIUM = max(1, EMA_TREND_MEDIUM)
EMA_TREND_SLOW = max(1, EMA_TREND_SLOW)

BB_PERIOD = max(2, BB_PERIOD)
BB_STD_DEV = max(0.1, BB_STD_DEV)

SR_LOOKBACK = max(5, SR_LOOKBACK)

RETEST_MAX_WAIT_CANDLES = max(
    1,
    RETEST_MAX_WAIT_CANDLES
)

ATR_SL_MULTIPLIER = max(
    0.1,
    ATR_SL_MULTIPLIER
)

ATR_TP_MULTIPLIER = max(
    0.1,
    ATR_TP_MULTIPLIER
)

MIN_SL_DISTANCE = max(
    0.01,
    MIN_SL_DISTANCE
)

MAX_SL_DISTANCE = max(
    MIN_SL_DISTANCE,
    MAX_SL_DISTANCE
)

MIN_TP_DISTANCE = max(
    0.01,
    MIN_TP_DISTANCE
)

MAX_TP_DISTANCE = max(
    MIN_TP_DISTANCE,
    MAX_TP_DISTANCE
)

MIN_RISK_REWARD = max(
    0.1,
    MIN_RISK_REWARD
)


# =========================================================
# CONFIGURATION STATUS
# =========================================================

print("=" * 55)
print(APP_NAME)
print("=" * 55)

print(f"MT5 Symbol       : {MT5_SYMBOL}")
print(f"Timeframes       : {', '.join(TIMEFRAMES)}")
print(f"Pattern TFs      : {', '.join(CANDLE_PATTERN_TIMEFRAMES)}")

print(f"Minimum Score    : {MIN_SIGNAL_SCORE}")
print(f"EMA Entry        : {EMA_PERIOD} EMA")
print(f"Bollinger Bands  : {BB_PERIOD}, {BB_STD_DEV} STD")

print(f"H1 Confirmation  : {ENABLE_H1_CONFIRMATION}")
print(f"H1 Retest        : {ENABLE_H1_RETEST_ENTRY}")
print(f"M1 Confirmation  : {ENABLE_M1_ENTRY_CONFIRMATION}")

print(f"Scan Interval    : {SCAN_INTERVAL_SECONDS} seconds")
print(f"Telegram Enabled : {TELEGRAM_ENABLED}")
print(f"Trading Mode     : {TRADING_MODE}")
print(f"Auto Trading     : {AUTO_TRADING_ENABLED}")

print(f"Data Directory   : {DATA_DIR}")

print("=" * 55)
