
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

M1_FILE = BASE_DIR / "M1.csv"
TRADES_FILE = BASE_DIR / "backtest_trades_v2.csv"

OUTPUT_FILE = BASE_DIR / "gold_v2_oos_results.csv"


def load_data():
    if not M1_FILE.exists():
        raise FileNotFoundError(f"Missing file: {M1_FILE}")

    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRADES_FILE}")

    m1 = pd.read_csv(M1_FILE)
    trades = pd.read_csv(TRADES_FILE)

    m1["time"] = pd.to_datetime(m1["time"], utc=True)

    for col in ["entry_time", "exit_time"]:
        trades[col] = pd.to_datetime(trades[col], utc=True)

    required = [
        "entry_time", "exit_time",
        "net_profit_usd", "exit_reason"
    ]

    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    trades["net_profit_usd"] = pd.to_numeric(
        trades["net_profit_usd"], errors="raise"
    )

    trades = trades.sort_values("entry_time").reset_index(drop=True)
    m1 = m1.sort_values("time")

    return m1, trades


def calculate_metrics(df, name):
    print(f"\n========== {name} ==========")

    if df.empty:
        print("No trades in this period.")
        return

    pnl = df["net_profit_usd"]

    total = len(df)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    breakeven = int((pnl == 0).sum())

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak

    print("Trades       :", total)
    print("Wins         :", wins)
    print("Losses       :", losses)
    print("Breakeven    :", breakeven)
    print(f"Win rate     : {wins / total * 100:.2f}%")
    print(f"Net P/L      : ${pnl.sum():.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Max Drawdown : ${drawdown.min():.2f}")

    print("\nExit reasons:")
    print(df["exit_reason"].value_counts().to_string())


def main():
    print("Loading M1 data and V2 trades...")

    m1, trades = load_data()

    start = m1["time"].min()
    end = m1["time"].max()

    cutoff = start + (end - start) * 0.70

    print("\n========== DATA SPLIT ==========")
    print("Data start :", start)
    print("Data end   :", end)
    print("70% cutoff :", cutoff)

    development = trades[
        (trades["entry_time"] < cutoff) &
        (trades["exit_time"] < cutoff)
    ].copy()

    test = trades[
        (trades["entry_time"] >= cutoff) &
        (trades["exit_time"] <= end)
    ].copy()

    crossing = trades[
        (trades["entry_time"] < cutoff) &
        (trades["exit_time"] >= cutoff)
    ].copy()

    print("\nDevelopment trades:", len(development))
    print("Unseen test trades:", len(test))
    print("Boundary excluded :", len(crossing))

    calculate_metrics(
        development,
        "DEVELOPMENT - FIRST 70%"
    )

    calculate_metrics(
        test,
        "OUT OF SAMPLE - LAST 30%"
    )

    if not test.empty:
        test.to_csv(OUTPUT_FILE, index=False)
        print("\nTest trades saved:", OUTPUT_FILE.name)

    print("\n========== VALIDATION NOTES ==========")
    print("P/L column used: net_profit_usd")
    print("Commission column in source CSV is retained.")
    print("No strategy parameters were changed.")
    print("This splits the existing V2 trade export.")
    print("It does not independently rerun the strategy.")
    print("======================================")


if __name__ == "__main__":
    main()
