
from __future__ import annotations

from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from config import MT5_SYMBOL, SYMBOL_CANDIDATES, TIMEFRAMES


TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}

BATCH_SIZE = 5000

REQUIRED_COLUMNS = {
    "time", "open", "high", "low", "close"
}


class Feed:

    def __init__(self):
        self.connected = False
        self.symbol = None
        self.last_error = None

    def connect(self) -> bool:

        if not mt5.initialize():
            self.last_error = (
                f"MT5 initialize failed: {mt5.last_error()}"
            )
            return False

        self.connected = True

        if not self.select_symbol():
            mt5.shutdown()
            self.connected = False
            return False

        return True

    def select_symbol(self) -> bool:

        if not self.connected:
            self.last_error = "MT5 is not connected"
            return False

        symbols = mt5.symbols_get()

        available = {
            item.name for item in (symbols or [])
        }

        candidates = [MT5_SYMBOL] + [
            name for name in SYMBOL_CANDIDATES
            if name != MT5_SYMBOL
        ]

        for name in candidates:

            if (
                name in available
                and mt5.symbol_select(name, True)
            ):
                self.symbol = name
                self.last_error = None
                return True

        self.last_error = (
            "Gold symbol not found. Tried: "
            + ", ".join(candidates)
        )

        return False

    def _fetch_batch(
        self,
        timeframe: str,
        start_position: int,
        count: int
    ):

        tf = TF_MAP.get(timeframe)

        if tf is None:
            self.last_error = (
                f"Unsupported timeframe: {timeframe}"
            )
            return None

        rates = mt5.copy_rates_from_pos(
            self.symbol,
            tf,
            int(start_position),
            int(count)
        )

        if rates is None:
            self.last_error = (
                f"MT5 history error: {mt5.last_error()}"
            )
            return None

        if len(rates) == 0:
            return None

        return rates

    def candles(
        self,
        timeframe: str,
        count: int | None = None
    ) -> pd.DataFrame | None:

        if not self.connected or not self.symbol:
            self.last_error = (
                "MT5 is not connected or symbol not selected"
            )
            return None

        timeframe = timeframe.upper()

        if timeframe not in TF_MAP:
            self.last_error = (
                f"Unsupported timeframe: {timeframe}"
            )
            return None

        requested = (
            None if count is None
            else max(1, int(count))
        )

        frames = []
        total_fetched = 0
        start_position = 0

        while True:

            if requested is None:
                batch_count = BATCH_SIZE
            else:
                remaining = requested - total_fetched

                if remaining <= 0:
                    break

                batch_count = min(
                    BATCH_SIZE,
                    remaining
                )

            rates = self._fetch_batch(
                timeframe,
                start_position,
                batch_count
            )

            if rates is None:
                break

            try:
                # Preserve MT5 structured-array field names.
                field_names = rates.dtype.names

                if not field_names:
                    self.last_error = (
                        "MT5 returned data without named fields"
                    )
                    return None

                batch_df = pd.DataFrame.from_records(
                    rates,
                    columns=list(field_names)
                )

                batch_df.columns = [
                    str(column).lower()
                    for column in batch_df.columns
                ]

                missing = (
                    REQUIRED_COLUMNS - set(batch_df.columns)
                )

                if missing:
                    self.last_error = (
                        "MT5 batch missing columns: "
                        + ", ".join(sorted(missing))
                    )
                    return None

                frames.append(batch_df)

            except Exception as exc:
                self.last_error = (
                    f"MT5 batch conversion failed: {exc}"
                )
                return None

            fetched = len(rates)
            total_fetched += fetched
            start_position += fetched

            if fetched < batch_count:
                break

        if not frames:
            self.last_error = (
                f"No candles available for "
                f"{self.symbol} {timeframe}. "
                f"MT5 error: {mt5.last_error()}"
            )
            return None

        try:
            df = pd.concat(
                frames,
                ignore_index=True
            )

            if df.empty:
                self.last_error = "MT5 returned empty candle data"
                return None

            df["time"] = pd.to_datetime(
                pd.to_numeric(
                    df["time"],
                    errors="coerce"
                ),
                unit="s",
                utc=True,
                errors="coerce"
            )

            for column in [
                "open", "high", "low", "close"
            ]:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

            df.dropna(
                subset=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close"
                ],
                inplace=True
            )

            if df.empty:
                self.last_error = (
                    "No valid candles after conversion"
                )
                return None

            df.set_index("time", inplace=True)

            df = df[
                ~df.index.duplicated(keep="last")
            ]

            df.sort_index(inplace=True)

            if requested is not None:
                df = df.tail(requested)

            if df.empty:
                self.last_error = "No candles available after filtering"
                return None

            self.last_error = None
            return df

        except Exception as exc:
            self.last_error = (
                f"Candle conversion failed: {exc}"
            )
            return None

    def all_frames(
        self,
        count: int | None = None
    ) -> dict:

        frames = {}

        for timeframe in TIMEFRAMES:

            timeframe = timeframe.upper()

            if timeframe not in TF_MAP:
                continue

            data = self.candles(
                timeframe,
                count
            )

            if data is not None and not data.empty:
                frames[timeframe] = data

        if not frames and not self.last_error:
            self.last_error = "No timeframe data received from MT5"

        return frames

    def tick(self):

        if not self.connected or not self.symbol:
            return None

        value = mt5.symbol_info_tick(self.symbol)

        if value is None:
            self.last_error = (
                f"Unable to get tick: {mt5.last_error()}"
            )
            return None

        return {
            "bid": float(value.bid),
            "ask": float(value.ask),
            "last": float(value.last),
            "time": datetime.fromtimestamp(
                value.time,
                tz=timezone.utc
            )
        }

    def symbol_info(self):

        if not self.connected or not self.symbol:
            return None

        info = mt5.symbol_info(self.symbol)

        if info is None:
            return None

        return {
            "name": info.name,
            "digits": int(info.digits),
            "point": float(info.point),
            "spread": int(info.spread),
            "trade_tick_size": float(
                info.trade_tick_size
            ),
            "trade_contract_size": float(
                info.trade_contract_size
            ),
            "volume_min": float(info.volume_min),
            "volume_step": float(info.volume_step),
        }

    def close(self):

        if self.connected:
            mt5.shutdown()

        self.connected = False
        self.symbol = None
        self.last_error = None
