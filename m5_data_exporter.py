
import MetaTrader5 as mt5
import pandas as pd

from datetime import datetime, timezone, timedelta
from pathlib import Path


class M5DataExporter:

    def __init__(self, symbol="GOLD", terminal_path=None):

        self.symbol = symbol
        self.terminal_path = terminal_path

    def connect(self):

        if self.terminal_path:
            connected = mt5.initialize(path=self.terminal_path)
        else:
            connected = mt5.initialize()

        if not connected:
            raise RuntimeError(
                f"MT5 connection failed: {mt5.last_error()}"
            )

        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(
                f"Symbol not available: {self.symbol}"
            )

        print("MT5 connected successfully")
        print("Symbol:", self.symbol)

    def download_data(self, start_date, end_date):

        all_batches = []
        current_start = start_date

        while current_start < end_date:

            current_end = min(
                current_start + timedelta(days=30),
                end_date
            )

            print(
                f"Downloading {current_start} to {current_end}"
            )

            rates = mt5.copy_rates_range(
                self.symbol,
                mt5.TIMEFRAME_M5,
                current_start,
                current_end
            )

            if rates is None:
                print("MT5 error:", mt5.last_error())
            elif len(rates) > 0:
                batch = pd.DataFrame(rates)
                all_batches.append(batch)
                print("Candles received:", len(batch))
            else:
                print("No candles received in this batch")

            current_start = current_end

        if not all_batches:
            raise RuntimeError(
                "No M5 history received. Check MT5 chart history."
            )

        df = pd.concat(all_batches, ignore_index=True)

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

        df.drop_duplicates(
            subset=["time"],
            keep="last",
            inplace=True
        )

        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        columns = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume"
        ]

        return df[columns]

    def export(
        self,
        days=365,
        output_file="data/GOLD_M5_history.csv"
    ):

        try:

            self.connect()

            # Last N days, using UTC
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            print("\nStart:", start_date)
            print("End:", end_date)

            df = self.download_data(
                start_date,
                end_date
            )

            output = Path(output_file)
            output.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            df.to_csv(output, index=False)

            print("\nM5 DATA EXPORT SUCCESSFUL")
            print("Saved:", output.resolve())
            print("Total candles:", len(df))
            print("First candle:", df.iloc[0]["time"])
            print("Last candle:", df.iloc[-1]["time"])

            return df

        finally:
            mt5.shutdown()
            print("\nMT5 disconnected")


if __name__ == "__main__":

    exporter = M5DataExporter(
        symbol="GOLD"
    )

    exporter.export(
        days=365,
        output_file="data/GOLD_M5_history.csv"
    )
