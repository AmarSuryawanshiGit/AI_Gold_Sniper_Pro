
from __future__ import annotations

import json
import sqlite3
import uuid

from datetime import datetime, timezone
from pathlib import Path

from config import DB_PATH


class Store:

    def __init__(self, db_path=None):
        self.db_path = Path(db_path or DB_PATH)

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # --------------------------------------------------
    # DATABASE CONNECTION
    # --------------------------------------------------

    def _connect(self):

        con = sqlite3.connect(
            str(self.db_path),
            timeout=20
        )

        con.row_factory = sqlite3.Row

        con.execute("PRAGMA busy_timeout = 20000")
        con.execute("PRAGMA foreign_keys = ON")

        return con

    # --------------------------------------------------
    # DATABASE INITIALIZATION
    # --------------------------------------------------

    def _init_db(self):

        with self._connect() as con:

            con.execute("PRAGMA journal_mode = WAL")

            con.execute("""
                CREATE TABLE IF NOT EXISTS signals (

                    id TEXT PRIMARY KEY,

                    created_at TEXT NOT NULL,

                    symbol TEXT NOT NULL,

                    timeframe TEXT NOT NULL,

                    setup TEXT NOT NULL,

                    direction TEXT NOT NULL,

                    score INTEGER NOT NULL,

                    entry REAL NOT NULL,

                    sl REAL NOT NULL,

                    tp REAL NOT NULL,

                    status TEXT NOT NULL DEFAULT 'OPEN',

                    exit_price REAL,

                    closed_at TEXT,

                    reason TEXT,

                    payload TEXT

                )
            """)

            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_created
                ON signals(created_at)
            """)

            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_status
                ON signals(status)
            """)

            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol
                ON signals(symbol)
            """)

            con.commit()

    # --------------------------------------------------
    # ADD NEW SIGNAL
    # --------------------------------------------------

    def add_signal(self, signal: dict) -> str:

        if not isinstance(signal, dict):
            raise TypeError("Signal must be a dictionary")

        required = [
            "symbol",
            "direction",
            "entry",
            "sl",
            "tp"
        ]

        for field in required:

            if signal.get(field) is None:
                raise ValueError(
                    f"Missing signal field: {field}"
                )

        sid = str(uuid.uuid4())

        created = datetime.now(
            timezone.utc
        ).isoformat()

        payload = json.dumps(
            signal,
            default=str
        )

        values = (
            sid,
            created,
            str(signal.get("symbol", "")),
            str(signal.get("timeframe", "M1")),
            str(signal.get("setup", "")),
            str(signal.get("direction", "")).upper(),
            int(signal.get("score", 0)),
            float(signal.get("entry")),
            float(signal.get("sl")),
            float(signal.get("tp")),
            str(signal.get("reason", "")),
            payload
        )

        with self._connect() as con:

            con.execute("""
                INSERT INTO signals (
                    id,
                    created_at,
                    symbol,
                    timeframe,
                    setup,
                    direction,
                    score,
                    entry,
                    sl,
                    tp,
                    reason,
                    payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)

            con.commit()

        return sid

    # --------------------------------------------------
    # CLOSE SIGNAL
    # --------------------------------------------------

    def close_signal(
        self,
        signal_id: str,
        status: str,
        exit_price: float = None
    ) -> bool:

        allowed = {
            "TP",
            "SL",
            "EXPIRED",
            "CANCELLED",
            "MANUAL"
        }

        status = str(status).upper()

        if status not in allowed:
            raise ValueError(
                f"Invalid status: {status}"
            )

        if exit_price is not None:
            exit_price = float(exit_price)

        closed_at = datetime.now(
            timezone.utc
        ).isoformat()

        with self._connect() as con:

            cursor = con.execute("""
                UPDATE signals

                SET
                    status = ?,
                    exit_price = ?,
                    closed_at = ?

                WHERE
                    id = ?
                    AND status = 'OPEN'

            """, (
                status,
                exit_price,
                closed_at,
                signal_id
            ))

            con.commit()

            return cursor.rowcount > 0

    # --------------------------------------------------
    # GET SIGNAL ROWS
    # --------------------------------------------------

    def rows(self, limit: int = 200):

        limit = max(1, min(int(limit), 10000))

        with self._connect() as con:

            data = con.execute("""
                SELECT *
                FROM signals
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

        return [
            dict(row)
            for row in data
        ]

    # --------------------------------------------------
    # GET OPEN SIGNALS
    # --------------------------------------------------

    def open_signals(self):

        with self._connect() as con:

            data = con.execute("""
                SELECT *
                FROM signals
                WHERE status = 'OPEN'
                ORDER BY created_at DESC
            """).fetchall()

        return [
            dict(row)
            for row in data
        ]

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    def stats(self):

        with self._connect() as con:

            total = con.execute("""
                SELECT COUNT(*)
                FROM signals
            """).fetchone()[0]

            wins = con.execute("""
                SELECT COUNT(*)
                FROM signals
                WHERE status = 'TP'
            """).fetchone()[0]

            losses = con.execute("""
                SELECT COUNT(*)
                FROM signals
                WHERE status = 'SL'
            """).fetchone()[0]

            opened = con.execute("""
                SELECT COUNT(*)
                FROM signals
                WHERE status = 'OPEN'
            """).fetchone()[0]

            expired = con.execute("""
                SELECT COUNT(*)
                FROM signals
                WHERE status = 'EXPIRED'
            """).fetchone()[0]

            cancelled = con.execute("""
                SELECT COUNT(*)
                FROM signals
                WHERE status = 'CANCELLED'
            """).fetchone()[0]

        closed = wins + losses

        win_rate = (
            round((wins / closed) * 100, 2)
            if closed else 0.0
        )

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "open": opened,
            "expired": expired,
            "cancelled": cancelled,
            "win_rate": win_rate
        }

    # --------------------------------------------------
    # DATABASE HEALTH CHECK
    # --------------------------------------------------

    def health_check(self):

        try:

            with self._connect() as con:

                con.execute("SELECT 1").fetchone()

            return {
                "status": "OK",
                "database": str(self.db_path)
            }

        except sqlite3.Error as exc:

            return {
                "status": "ERROR",
                "message": str(exc)
            }
