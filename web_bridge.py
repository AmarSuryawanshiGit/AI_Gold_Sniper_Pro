
from PySide6.QtCore import QObject, Signal, Slot


class WebBridge(QObject):

    timeframeChanged = Signal(str)
    fullscreenRequested = Signal(bool)

    @Slot(str)
    def changeTimeframe(self, timeframe):

        allowed = {"M1", "M5", "M15", "M30", "H1"}

        if timeframe not in allowed:
            return

        self.timeframeChanged.emit(timeframe)

    @Slot(bool)
    def setFullscreen(self, enabled):

        self.fullscreenRequested.emit(bool(enabled))
