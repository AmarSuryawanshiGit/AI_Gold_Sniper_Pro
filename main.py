
from __future__ import annotations

from range_engine import RangeEngine
from candlestick_strategy import CandlestickStrategy

import sys
import json
import traceback

from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
    Qt
)

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QHeaderView
)

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel

from config import (
    APP_NAME,
    BARS_TO_LOAD,
    SCAN_INTERVAL_SECONDS,
    MIN_SIGNAL_SCORE
)

from market import Feed
from sniper_engine import analyse_market
from chart_bridge import build_chart_payload
from web_bridge import WebBridge
from store import Store
from alerts import send_signal


# =========================================================
# BACKGROUND MT5 WORKER
# =========================================================

class MarketWorker(QObject):

    connected = Signal(bool, str, object)
    marketDataReady = Signal(object)
    disconnected = Signal(str)
    workerError = Signal(str)

    def __init__(self):
        super().__init__()

        self.feed = None
        self.is_connected = False
        self.busy = False

        self.range_engine = RangeEngine()
        self.candlestick_strategy = CandlestickStrategy()

    @Slot()
    def connect_mt5(self):

        if self.is_connected:
            return

        try:
            self.feed = Feed()

            if not self.feed.connect():

                error = (
                    self.feed.last_error
                    or "MT5 connection failed"
                )

                self.feed.close()
                self.feed = None

                self.connected.emit(False, error, None)
                return

            self.is_connected = True
            self.candlestick_strategy.reset()

            info = self.feed.symbol_info()

            if info is None:

                self.feed.close()
                self.feed = None
                self.is_connected = False

                self.connected.emit(
                    False,
                    "Unable to read MT5 symbol information",
                    None
                )
                return

            self.connected.emit(
                True,
                self.feed.symbol,
                info
            )

        except Exception as exc:

            self.is_connected = False

            if self.feed:
                try:
                    self.feed.close()
                except Exception:
                    pass

            self.feed = None

            self.connected.emit(False, str(exc), None)
            traceback.print_exc()

    @Slot(str)
    def refresh(self, timeframe):

        if not self.is_connected or self.feed is None:
            return

        # Prevent overlapping MT5 requests.
        if self.busy:
            return

        self.busy = True

        try:

            frames = self.feed.all_frames(BARS_TO_LOAD)

            if not frames:

                self.marketDataReady.emit({
                    "error": (
                        self.feed.last_error
                        or "No market data received"
                    )
                })
                return

            tick = self.feed.tick()
            info = self.feed.symbol_info()

            if tick is None or info is None:

                self.marketDataReady.emit({
                    "error": (
                        self.feed.last_error
                        or "Waiting for MT5 tick..."
                    )
                })
                return

            required = ["M1", "M5", "M15", "M30", "H1"]

            missing = [
                tf for tf in required
                if tf not in frames
                or frames[tf] is None
                or frames[tf].empty
            ]

            if missing:

                self.marketDataReady.emit({
                    "error": (
                        "Waiting for timeframe data: "
                        + ", ".join(missing)
                    ),
                    "tick": tick
                })
                return

            if len(frames["M1"]) < 3:

                self.marketDataReady.emit({
                    "error": "Not enough M1 candles",
                    "tick": tick
                })
                return

            # Run strategy analysis in worker thread.
            result = analyse_market(
                frames,
                bid=tick["bid"],
                ask=tick["ask"],
                point=info["point"]
            )

            candle_result = self.candlestick_strategy.evaluate(frames)
            ranges = self.range_engine.update(frames)

            self.marketDataReady.emit({
                "frames": frames,
                "tick": tick,
                "info": info,
                "result": result,
                "symbol": self.feed.symbol,
                "candle_result": candle_result,
                "ranges": ranges,
                "timeframe": timeframe
            })

        except Exception as exc:

            self.marketDataReady.emit({
                "error": str(exc)
            })

            traceback.print_exc()

        finally:
            self.busy = False

    @Slot()
    def disconnect_mt5(self):

        try:

            if self.feed is not None:
                self.feed.close()

            self.feed = None
            self.is_connected = False

            self.candlestick_strategy.reset()

            self.disconnected.emit("MT5 disconnected")

        except Exception as exc:
            self.workerError.emit(str(exc))

    @Slot()
    def shutdown(self):

        try:

            if self.feed is not None:
                self.feed.close()

            self.feed = None
            self.is_connected = False

        except Exception:
            traceback.print_exc()


# =========================================================
# MAIN WINDOW
# =========================================================

class MainWindow(QMainWindow):

    refreshRequested = Signal(str)
    connectRequested = Signal()
    disconnectRequested = Signal()
    shutdownRequested = Signal()

    def __init__(self):

        super().__init__()

        self.setWindowTitle(APP_NAME + " 2.0")
        self.resize(1550, 950)

        # =============================================
        # APPLICATION STATE
        # =============================================

        self.store = Store()

        self.connected = False
        self.chart_loaded = False
        self.chart_fullscreen = False
        self.closing = False

        self.chart_timeframe = "M1"
        self.last_processed_candle = None

        self.last_frames = None
        self.last_tick = None
        self.last_info = None

        # Latest verified signal is retained so its chart
        # marker does not disappear on the next refresh.
        self.latest_verified_signal = None

        # =============================================
        # RANGE ENGINE
        # =============================================

        self.range_engine = RangeEngine()
        self.last_detected_ranges = []

        # =============================================
        # BACKGROUND THREAD
        # =============================================

        self.worker_thread = QThread(self)

        self.worker = MarketWorker()
        self.worker.moveToThread(self.worker_thread)

        self.connectRequested.connect(
            self.worker.connect_mt5,
            Qt.ConnectionType.QueuedConnection
        )

        self.disconnectRequested.connect(
            self.worker.disconnect_mt5,
            Qt.ConnectionType.QueuedConnection
        )

        self.refreshRequested.connect(
            self.worker.refresh,
            Qt.ConnectionType.QueuedConnection
        )

        self.shutdownRequested.connect(
            self.worker.shutdown,
            Qt.ConnectionType.QueuedConnection
        )

        self.worker.connected.connect(
            self.on_mt5_connected
        )

        self.worker.marketDataReady.connect(
            self.on_market_data
        )

        self.worker.disconnected.connect(
            self.on_mt5_disconnected
        )

        self.worker.workerError.connect(
            self.on_worker_error
        )

        # =============================================
        # HEADER LABELS
        # =============================================

        self.status_label = QLabel("Disconnected")

        self.price_label = QLabel(
            "Bid: -- | Ask: --"
        )

        self.signal_label = QLabel(
            "Waiting for signal..."
        )

        self.candle_strategy_label = QLabel(
            "Verified Candlestick Setups: Waiting..."
        )

        self.trend_label = QLabel(
            "Market trend: --"
        )

        # =============================================
        # CONNECTION BUTTONS
        # =============================================

        self.connect_button = QPushButton(
            "Connect XM MT5"
        )

        self.connect_button.clicked.connect(
            self.connect_mt5
        )

        self.disconnect_button = QPushButton(
            "Disconnect"
        )

        self.disconnect_button.clicked.connect(
            self.disconnect_mt5
        )

        self.disconnect_button.setEnabled(False)

        # =============================================
        # HEADER LAYOUT
        # =============================================

        header = QHBoxLayout()

        header.addWidget(self.connect_button)
        header.addWidget(self.disconnect_button)
        header.addWidget(self.status_label)

        header.addStretch()
        header.addWidget(self.price_label)

        self.header_widget = QWidget()
        self.header_widget.setLayout(header)

        # =============================================
        # WEB CHART
        # =============================================

        self.web = QWebEngineView()

        self.web.settings().setAttribute(
            self.web.settings().WebAttribute.JavascriptEnabled,
            True
        )

        self.channel = QWebChannel(self.web.page())
        self.bridge = WebBridge()

        self.channel.registerObject(
            "pyBridge",
            self.bridge
        )

        self.web.page().setWebChannel(self.channel)

        self.bridge.timeframeChanged.connect(
            self.change_timeframe
        )

        self.bridge.fullscreenRequested.connect(
            self.set_chart_fullscreen
        )

        self.web.loadFinished.connect(
            self.on_chart_loaded
        )

        self.chart_path = (
            Path(__file__).resolve().parent
            / "chart.html"
        )

        if not self.chart_path.exists():

            QMessageBox.critical(
                self,
                "Chart Error",
                f"chart.html not found:\n{self.chart_path}"
            )

        else:

            self.web.load(
                QUrl.fromLocalFile(
                    str(self.chart_path)
                )
            )

        # =============================================
        # SIGNAL JOURNAL
        # =============================================

        self.journal = QTableWidget(0, 9)

        self.journal.setHorizontalHeaderLabels([
            "Time",
            "Symbol",
            "Setup",
            "Side",
            "Score",
            "Entry",
            "SL",
            "TP",
            "Status"
        ])

        self.journal.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.stats_label = QLabel(
            "Signals: 0 | TP: 0 | SL: 0 | Open: 0"
        )

        # =============================================
        # TABS
        # =============================================

        self.tabs = QTabWidget()

        # LIVE CHART TAB
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)

        live_layout.setContentsMargins(4, 4, 4, 4)
        live_layout.setSpacing(4)

        live_layout.addWidget(self.trend_label)
        live_layout.addWidget(self.signal_label)
        live_layout.addWidget(self.candle_strategy_label)

        live_layout.addWidget(
            self.web,
            1
        )

        self.tabs.addTab(
            live_tab,
            "Professional Live Chart"
        )

        # JOURNAL TAB
        journal_tab = QWidget()
        journal_layout = QVBoxLayout(journal_tab)

        journal_layout.addWidget(self.journal)
        journal_layout.addWidget(self.stats_label)

        self.tabs.addTab(
            journal_tab,
            "Signal Journal"
        )

        # =============================================
        # MAIN WINDOW LAYOUT
        # =============================================

        root = QWidget()
        self.main_layout = QVBoxLayout(root)

        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(5)

        self.main_layout.addWidget(self.header_widget)
        self.main_layout.addWidget(self.tabs)

        self.setCentralWidget(root)

        # =============================================
        # REFRESH TIMER
        # =============================================

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.request_market_refresh
        )

        self.refresh_journal()

        # Start worker only after UI initialization.
        self.worker_thread.start()

    # =================================================
    # CHART INITIALIZATION
    # =================================================

    def on_chart_loaded(self, ok):

        if not ok:

            self.chart_loaded = False

            self.status_label.setText(
                "Chart HTML loading failed"
            )

            print("ERROR: chart.html could not load.")
            return

        print("Chart HTML loaded successfully.")

        js = """
        JSON.stringify({
            libraryLoaded:
                typeof LightweightCharts !== 'undefined',

            chartReady:
                typeof window.updateChart === 'function',

            bridgeReady:
                typeof window.pyBridge !== 'undefined'
        })
        """

        self.web.page().runJavaScript(
            js,
            self.check_chart_status
        )

    def check_chart_status(self, result):

        try:

            if isinstance(result, str):
                result = json.loads(result)

            if not isinstance(result, dict):

                self.chart_loaded = False
                self.status_label.setText(
                    "Unable to verify chart"
                )
                return

            if not result.get("libraryLoaded"):

                self.chart_loaded = False
                self.status_label.setText(
                    "Chart library missing"
                )
                return

            if not result.get("chartReady"):

                self.chart_loaded = False
                self.status_label.setText(
                    "Chart initialization failed"
                )
                return

            self.chart_loaded = True

            self.status_label.setText(
                "Chart ready · Connect MT5"
            )

            print(
                "Chart JavaScript initialized successfully."
            )

            if self.connected:
                self.request_market_refresh()

        except Exception as exc:

            self.chart_loaded = False

            self.status_label.setText(
                "Chart initialization error"
            )

            print("Chart status error:", exc)
            traceback.print_exc()

    # =================================================
    # FULLSCREEN
    # =================================================

    def set_chart_fullscreen(self, enabled):

        enabled = bool(enabled)

        if enabled == self.chart_fullscreen:
            return

        self.chart_fullscreen = enabled

        if enabled:

            self.header_widget.hide()
            self.tabs.tabBar().hide()

            self.trend_label.hide()
            self.signal_label.hide()

            self.showFullScreen()

        else:

            self.showNormal()

            self.header_widget.show()
            self.tabs.tabBar().show()

            self.trend_label.show()
            self.signal_label.show()

        QTimer.singleShot(
            180,
            self.notify_chart_resize
        )

    def notify_chart_resize(self):

        if self.chart_loaded:

            self.web.page().runJavaScript(
                "if (window.resizeChart) window.resizeChart();"
            )

    # =================================================
    # TIMEFRAME
    # =================================================

    def change_timeframe(self, timeframe):

        allowed = {
            "M1", "M5", "M15", "M30", "H1"
        }

        if timeframe not in allowed:
            return

        self.chart_timeframe = timeframe

        print(
            f"Chart timeframe changed to {timeframe}"
        )

        if self.connected and self.chart_loaded:
            self.request_market_refresh()

    # =================================================
    # MT5 CONNECTION
    # =================================================

    def connect_mt5(self):

        if self.connected:
            return

        self.connect_button.setEnabled(False)

        self.status_label.setText(
            "Connecting to XM MT5..."
        )

        self.connectRequested.emit()

    def on_mt5_connected(
        self,
        success,
        message,
        info
    ):

        if not success:

            self.connected = False

            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)

            QMessageBox.warning(
                self,
                "MT5 Error",
                str(message)
            )

            self.status_label.setText(
                "Connection failed"
            )
            return

        self.connected = True

        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)

        self.last_processed_candle = None
        self.latest_verified_signal = None

        self.status_label.setText(
            f"Connected · {message} · "
            f"{info['digits']} digits · "
            f"point {info['point']}"
        )

        print(
            f"Connected to MT5 symbol: {message}"
        )

        self.timer.start(
            max(
                1000,
                SCAN_INTERVAL_SECONDS * 1000
            )
        )

        self.request_market_refresh()

    def disconnect_mt5(self):

        if not self.connected:
            return

        self.timer.stop()
        self.connected = False

        self.last_frames = None
        self.last_tick = None
        self.last_info = None
        self.latest_verified_signal = None

        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

        self.status_label.setText(
            "Disconnecting MT5..."
        )

        self.disconnectRequested.emit()

    def on_mt5_disconnected(self, message):

        self.status_label.setText(message)

        self.price_label.setText(
            "Bid: -- | Ask: --"
        )

        print(message)

    def on_worker_error(self, error):

        print("Worker error:", error)

        self.status_label.setText(
            f"Worker error: {error}"
        )

    # =================================================
    # MARKET REFRESH REQUEST
    # =================================================

    def request_market_refresh(self):

        if not self.connected:
            return

        self.refreshRequested.emit(
            self.chart_timeframe
        )


    # =================================================
    # MARKET DATA RECEIVED FROM WORKER
    # =================================================

    def on_market_data(self, data):

        if not self.connected:
            return

        if not isinstance(data, dict):
            return

        # =============================================
        # MARKET DATA ERROR HANDLING
        # =============================================

        if data.get("error"):

            self.status_label.setText(
                str(data["error"])
            )

            print(
                "Market data warning:",
                data["error"]
            )

            tick = data.get("tick")

            if isinstance(tick, dict):
                try:
                    self.price_label.setText(
                        f"Bid: {float(tick['bid']):.2f} | "
                        f"Ask: {float(tick['ask']):.2f}"
                    )
                except (KeyError, TypeError, ValueError):
                    pass

            return

        # =============================================
        # EXTRACT MARKET DATA
        # =============================================

        frames = data.get("frames")
        tick = data.get("tick")
        info = data.get("info")
        result = data.get("result")

        candle_result = data.get("candle_result", {})

        if not isinstance(candle_result, dict):
            candle_result = {}

        # =============================================
        # CANDLESTICK STRATEGY STATUS
        # =============================================

        pattern_scan = candle_result.get(
            "pattern_scan", {}
        )

        if not isinstance(pattern_scan, dict):
            pattern_scan = {}

        verified_labels = []

        # Only verified S/R patterns are displayed.
        # M1 is used for entry confirmation, not pattern scanning.

        for tf in ["H1", "M15", "M5"]:

            scan = pattern_scan.get(tf, {})

            if not isinstance(scan, dict):
                continue

            if scan.get("verified"):

                labels = scan.get("labels", [])

                if isinstance(labels, (list, tuple)):
                    verified_labels.extend(
                        str(label) for label in labels
                    )

        if verified_labels:

            pattern_text = " | ".join(
                dict.fromkeys(verified_labels)
            )

        else:

            pattern_text = "No verified S/R pattern"

        # =============================================
        # H1 STRATEGY STATUS
        # =============================================

        h1_strategy = candle_result.get(
            "h1_strategy", {}
        )

        if not isinstance(h1_strategy, dict):
            h1_strategy = {}

        h1_setup = h1_strategy.get(
            "setup", {}
        )

        h1_entry = h1_strategy.get(
            "entry", {}
        )

        if not isinstance(h1_setup, dict):
            h1_setup = {}

        if not isinstance(h1_entry, dict):
            h1_entry = {}

        h1_setup_status = h1_setup.get(
            "status", "WAITING"
        )

        h1_entry_status = h1_entry.get(
            "status", "WAITING"
        )

        # =============================================
        # EMA9 STRATEGY STATUS
        # =============================================

        ema_strategy = candle_result.get(
            "ema9_strategy",
            candle_result.get("ema_entry", {})
        )

        if not isinstance(ema_strategy, dict):
            ema_strategy = {}

        ema_status = ema_strategy.get(
            "status", "WAITING"
        )

        # =============================================
        # UPDATE CANDLE STRATEGY LABEL
        # =============================================

        self.candle_strategy_label.setText(
            f"Verified Patterns: {pattern_text} | "
            f"H1 Setup: {h1_setup_status} | "
            f"H1 Entry: {h1_entry_status} | "
            f"EMA9: {ema_status}"
        )

        # =============================================
        # VALIDATE MARKET FRAMES
        # =============================================

        if not isinstance(frames, dict):
            return

        if tick is None or info is None:
            return

        if not isinstance(tick, dict):
            return

        try:

            bid = float(tick["bid"])
            ask = float(tick["ask"])

        except (KeyError, TypeError, ValueError):

            print("Invalid tick data")
            return

        self.last_frames = frames
        self.last_tick = tick
        self.last_info = info

        # =============================================
        # UPDATE PRICE
        # =============================================

        self.price_label.setText(
            f"Bid: {bid:.2f} | Ask: {ask:.2f}"
        )

        self.status_label.setText(
            f"Connected · {data.get('symbol', '')}"
        )

        # =============================================
        # PROCESS ONLY NEW CLOSED M1 CANDLE
        # =============================================

        m1_df = frames.get("M1")

        if m1_df is None or len(m1_df) < 3:
            return

        closed_time = m1_df.index[-2]

        is_new_closed_candle = (
            closed_time != self.last_processed_candle
        )

        if is_new_closed_candle:

            self.last_processed_candle = closed_time

            if isinstance(result, dict):
                self.process_signal_result(
                    result,
                    closed_time
                )

        # =============================================
        # RANGE DETECTION + CHART UPDATE
        # =============================================

        chart_df = frames.get(
            self.chart_timeframe
        )

        if (
            chart_df is not None
            and not chart_df.empty
            and self.chart_loaded
        ):

            try:

                all_ranges = data.get(
                    "ranges", []
                )

                if not isinstance(all_ranges, list):
                    all_ranges = []

                visibility = {
                    "H1": {"H1"},
                    "M30": {"H1"},
                    "M15": {"H1", "M15"},
                    "M5": {"H1", "M15", "M5"},
                    "M1": {"H1", "M15", "M5"},
                }

                allowed = visibility.get(
                    self.chart_timeframe,
                    set()
                )

                visible_ranges = [
                    item
                    for item in all_ranges
                    if (
                        isinstance(item, dict)
                        and item.get("timeframe") in allowed
                    )
                ]

                self.last_detected_ranges = visible_ranges

                payload = build_chart_payload(
                    chart_df,
                    symbol=data.get("symbol", "GOLD"),
                    timeframe=self.chart_timeframe,
                    bid=bid,
                    ask=ask,
                    ranges=visible_ranges,
                    signal=self.latest_verified_signal,
                    pattern_scan=pattern_scan
                )

                self.send_chart_payload(payload)

            except Exception as exc:

                print(
                    "Range / Chart update error:",
                    exc
                )

                traceback.print_exc()

        # =============================================
        # OPEN TRADE OUTCOMES
        # =============================================

        self.update_open_outcomes(tick)

        # =============================================
        # REFRESH JOURNAL
        # =============================================

        self.refresh_journal()

    # =================================================
    # CHART PAYLOAD
    # =================================================

    def send_chart_payload(self, payload):

        if not self.chart_loaded:
            return

        try:

            encoded = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":")
            )

            js = f"""
            if (typeof window.updateChart === 'function') {{
                window.updateChart({encoded});
            }} else {{
                console.warn('Chart is not initialized yet.');
            }}
            """

            self.web.page().runJavaScript(js)

        except (TypeError, ValueError) as exc:

            print(
                "Chart payload serialization error:",
                exc
            )

        except Exception as exc:

            print(
                "Chart update error:",
                exc
            )

            traceback.print_exc()

    # =================================================
    # SIGNAL PROCESSING
    # =================================================

    def process_signal_result(
        self,
        result,
        closed_time=None
    ):

        market_trend = result.get(
            "market_trend", "--"
        )

        trends = result.get(
            "trends", {}
        )

        self.trend_label.setText(
            "Trend: "
            + str(market_trend)
            + " | MTF: "
            + str(trends)
        )

        if result.get("status") == "SIGNAL":

            try:
                score = float(result.get("score", 0))
            except (TypeError, ValueError):
                score = 0

            if score >= MIN_SIGNAL_SCORE:

                direction = str(
                    result.get("direction", "")
                ).upper()

                if direction not in ("BUY", "SELL"):

                    self.signal_label.setText(
                        "Signal rejected: Invalid direction"
                    )
                    return

                # Keep only a verified signal for chart marker.
                self.latest_verified_signal = dict(result)

                self.store.add_signal(result)

                telegram_sent = send_signal(result)

                self.signal_label.setText(
                    f"{direction} | "
                    f"{result.get('setup')} | "
                    f"Score {score:g} | "
                    f"Entry {result.get('entry')} | "
                    f"SL {result.get('sl')} | "
                    f"TP {result.get('tp')} | "
                    f"Telegram: {telegram_sent}"
                )

                print(
                    f"NEW SIGNAL: "
                    f"{direction} {result.get('setup')} "
                    f"Score: {score:g}"
                )

            else:

                self.signal_label.setText(
                    "Signal rejected: Score below minimum"
                )

        else:

            self.signal_label.setText(
                result.get(
                    "reason",
                    "No qualifying closed-candle setup"
                )
            )

    # =================================================
    # OPEN SIGNAL OUTCOME TRACKER
    # =================================================

    def update_open_outcomes(self, tick):

        open_rows = [
            row
            for row in self.store.rows(500)
            if row["status"] == "OPEN"
        ]

        for row in open_rows:

            direction = row["direction"]

            sl = float(row["sl"])
            tp = float(row["tp"])

            if direction == "BUY":

                price = tick["bid"]

                if price <= sl:

                    self.store.close_signal(
                        row["id"],
                        "SL",
                        price
                    )

                elif price >= tp:

                    self.store.close_signal(
                        row["id"],
                        "TP",
                        price
                    )

            elif direction == "SELL":

                price = tick["ask"]

                if price >= sl:

                    self.store.close_signal(
                        row["id"],
                        "SL",
                        price
                    )

                elif price <= tp:

                    self.store.close_signal(
                        row["id"],
                        "TP",
                        price
                    )

    # =================================================
    # JOURNAL
    # =================================================

    def refresh_journal(self):

        rows = self.store.rows(200)

        self.journal.setRowCount(
            len(rows)
        )

        for i, row in enumerate(rows):

            values = [
                row["created_at"][:19].replace(
                    "T", " "
                ),
                row["symbol"],
                row["setup"],
                row["direction"],
                str(row["score"]),
                f"{row['entry']:.2f}",
                f"{row['sl']:.2f}",
                f"{row['tp']:.2f}",
                row["status"]
            ]

            for j, value in enumerate(values):

                self.journal.setItem(
                    i,
                    j,
                    QTableWidgetItem(str(value))
                )

        stats = self.store.stats()

        self.stats_label.setText(
            f"Signals: {stats['total']} | "
            f"TP: {stats['wins']} | "
            f"SL: {stats['losses']} | "
            f"Open: {stats['open']} | "
            f"Win rate: {stats['win_rate']}%"
        )

    # =================================================
    # CLOSE EVENT
    # =================================================

    def closeEvent(self, event):

        if self.closing:
            event.accept()
            return

        self.closing = True
        self.timer.stop()
        self.connected = False

        # Stop worker safely.
        if self.worker_thread.isRunning():

            self.shutdownRequested.emit()
            self.worker_thread.quit()

        if not self.worker_thread.wait(5000):

            print(
                "Warning: MT5 worker did not stop "
                "within 5 seconds."
            )

        event.accept()


# =========================================================
# APPLICATION START
# =========================================================

def main():

    app = QApplication(sys.argv)

    window = MainWindow()
    window.showMaximized()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
