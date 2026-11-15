
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from config import (
    DATA_DIR,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ENABLED,
)


# =========================================================
# LOGGING
# =========================================================

LOG_FILE = DATA_DIR / "telegram.log"

logger = logging.getLogger("gold_sniper.telegram")

if not logger.handlers:
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(handler)


# =========================================================
# TELEGRAM SETTINGS
# =========================================================

API_BASE_URL = "https://api.telegram.org"

REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Prevent duplicate alerts for the same closed candle.
SENT_IDS_FILE = DATA_DIR / "telegram_sent_ids.json"

_session = requests.Session()


# =========================================================
# DUPLICATE ALERT PROTECTION
# =========================================================

def _load_sent_ids() -> set[str]:

    try:
        if not SENT_IDS_FILE.exists():
            return set()

        with SENT_IDS_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return {str(item) for item in data}

    except Exception as exc:
        logger.exception(
            "Unable to load sent alert IDs: %s",
            exc
        )

    return set()


def _save_sent_ids(ids: set[str]) -> None:

    try:
        # Keep the latest 1000 IDs.
        recent_ids = list(ids)[-1000:]

        temp_file = SENT_IDS_FILE.with_suffix(".tmp")

        with temp_file.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                recent_ids,
                file,
                indent=2
            )

        temp_file.replace(SENT_IDS_FILE)

    except Exception as exc:
        logger.exception(
            "Unable to save sent alert IDs: %s",
            exc
        )


def _signal_id(signal: dict[str, Any]) -> str:

    candle_time = signal.get(
        "candle_time",
        ""
    )

    symbol = signal.get(
        "symbol",
        "GOLD"
    )

    setup = signal.get(
        "setup",
        "UNKNOWN"
    )

    direction = signal.get(
        "direction",
        "UNKNOWN"
    )

    return "|".join([
        str(symbol),
        str(candle_time),
        str(setup),
        str(direction),
    ])


# =========================================================
# MESSAGE FORMATTER
# =========================================================

def format_signal_message(
    signal: dict[str, Any]
) -> str:

    direction = str(
        signal.get("direction", "UNKNOWN")
    ).upper()

    emoji = "🟢" if direction == "BUY" else "🔴"

    symbol = signal.get(
        "symbol",
        "GOLD"
    )

    timeframe = signal.get(
        "timeframe",
        "M1"
    )

    setup = signal.get(
        "setup",
        "N/A"
    )

    score = signal.get(
        "score",
        0
    )

    entry = signal.get(
        "entry",
        0
    )

    sl = signal.get(
        "sl",
        0
    )

    tp = signal.get(
        "tp",
        0
    )

    rr = signal.get(
        "rr",
        0
    )

    trend = signal.get(
        "market_trend",
        "UNKNOWN"
    )

    rsi = signal.get(
        "rsi",
        "N/A"
    )

    atr = signal.get(
        "atr",
        "N/A"
    )

    reason = signal.get(
        "reason",
        "No details available"
    )

    candle_time = signal.get(
        "candle_time",
        datetime.now().isoformat()
    )

    message = (
        "🚀 <b>AI GOLD SNIPER PRO</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} <b>{direction} SIGNAL</b>\n"
        "\n"
        f"📊 Symbol: <b>{symbol}</b>\n"
        f"⏱ Timeframe: <b>{timeframe}</b>\n"
        f"🎯 Setup: <b>{setup}</b>\n"
        f"⭐ Score: <b>{score}/100</b>\n"
        "\n"
        "💰 <b>TRADE LEVELS</b>\n"
        f"Entry: <code>{entry}</code>\n"
        f"Stop Loss: <code>{sl}</code>\n"
        f"Take Profit: <code>{tp}</code>\n"
        f"Risk/Reward: <b>1:{rr}</b>\n"
        "\n"
        "📈 <b>MARKET DETAILS</b>\n"
        f"Trend: {trend}\n"
        f"RSI: {rsi}\n"
        f"ATR: {atr}\n"
        "\n"
        f"🧠 Reason: {reason}\n"
        f"🕒 Candle: {candle_time}\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "⚠️ Signal only. Manual verification required.\n"
        "Use your own risk management."
    )

    return message


# =========================================================
# TELEGRAM API REQUEST
# =========================================================

def _send_telegram_message(
    message: str
) -> bool:

    if not TELEGRAM_ENABLED:
        logger.warning(
            "Telegram disabled. Check .env credentials."
        )
        return False

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error(
            "Telegram token or chat ID is missing."
        )
        return False

    url = (
        f"{API_BASE_URL}/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = _session.post(
                url,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            result = response.json()

            if result.get("ok") is True:

                logger.info(
                    "Telegram message sent successfully."
                )

                return True

            logger.error(
                "Telegram API returned error: %s",
                result
            )

        except requests.RequestException as exc:

            logger.warning(
                "Telegram attempt %s failed: %s",
                attempt,
                exc
            )

        except ValueError as exc:

            logger.error(
                "Invalid Telegram response: %s",
                exc
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    logger.error(
        "Telegram message failed after all retries."
    )

    return False


# =========================================================
# PUBLIC FUNCTION USED BY main.py
# =========================================================

def send_signal(
    signal: dict[str, Any]
) -> bool:

    if not isinstance(signal, dict):
        logger.error("Invalid signal object.")
        return False

    if signal.get("status") != "SIGNAL":
        return False

    signal_id = _signal_id(signal)

    sent_ids = _load_sent_ids()

    if signal_id in sent_ids:

        logger.info(
            "Duplicate signal ignored: %s",
            signal_id
        )

        return False

    try:

        message = format_signal_message(
            signal
        )

        success = _send_telegram_message(
            message
        )

        if success:

            sent_ids.add(signal_id)
            _save_sent_ids(sent_ids)

        return success

    except Exception as exc:

        logger.exception(
            "Signal notification error: %s",
            exc
        )

        return False


# =========================================================
# TEST TELEGRAM CONNECTION
# =========================================================

def test_telegram() -> bool:

    message = (
        "🚀 <b>AI GOLD SNIPER PRO</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Telegram connection successful!\n"
        "Your Gold signal alerts are configured.\n"
        "\n"
        "Mode: SIGNAL ONLY\n"
        "Broker: XM MT5\n"
        "Symbol: GOLD\n"
        "\n"
        "⚠️ Manual trade execution only."
    )

    return _send_telegram_message(
        message
    )


# =========================================================
# CLEANUP
# =========================================================

def close_telegram_session() -> None:

    try:
        _session.close()

    except Exception as exc:
        logger.warning(
            "Telegram session cleanup failed: %s",
            exc
        )
