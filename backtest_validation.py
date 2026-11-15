
from pathlib import Path
import pandas as pd
import math

# =============== SETTINGS ===============
TRADE_FILE = "backtest_trades.csv"

TRAIN_FRACTION = 0.70
MIN_TRAIN_TRADES = 20
MIN_TRAIN_PF = 1.10
# ========================================


def load_trades():
    path = Path(TRADE_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"{TRADE_FILE} not found. Keep this script "
            "in the same folder as your trade file."
        )

    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"entry_time", "direction", "pnl_usd"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["entry_time"] = pd.to_datetime(
        df["entry_time"], utc=True, errors="coerce"
    )
    df["direction"] = df["direction"].astype(str).str.upper().str.strip()
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")

    df = df.dropna(
        subset=["entry_time", "direction", "pnl_usd"]
    )

    df = df[df["direction"].isin(["BUY", "SELL"])]
    df = df.sort_values("entry_time").reset_index(drop=True)

    df["utc_hour"] = df["entry_time"].dt.hour

    if len(df) < 50:
        raise ValueError("Need at least 50 valid trades.")

    return df


def profit_factor(pnl):
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0

    return gross_profit / gross_loss


def max_drawdown(pnl):
    equity = pnl.cumsum()
    peak = equity.cummax()
    return (equity - peak).min()


def get_metrics(df, label):
    if df.empty:
        return {
            "Period": label,
            "Trades": 0,
            "Win Rate": "0.00%",
            "Net P/L": "$0.00",
            "Profit Factor": "0.00",
            "Max DD": "$0.00"
        }

    pnl = df["pnl_usd"]
    pf = profit_factor(pnl)

    return {
        "Period": label,
        "Trades": len(df),
        "Win Rate": f"{(pnl > 0).mean() * 100:.2f}%",
        "Net P/L": f"${pnl.sum():.2f}",
        "Profit Factor": "inf" if math.isinf(pf) else f"{pf:.2f}",
        "Max DD": f"${max_drawdown(pnl):.2f}"
    }


def select_training_filters(train):
    selected = []
    ranking = []

    for (direction, hour), group in train.groupby(
        ["direction", "utc_hour"]
    ):
        pnl = group["pnl_usd"]
        pf = profit_factor(pnl)
        net = pnl.sum()

        ranking.append({
            "Direction": direction,
            "UTC Hour": int(hour),
            "Trades": len(group),
            "Win Rate": round((pnl > 0).mean() * 100, 2),
            "Net P/L": round(net, 2),
            "PF": round(pf, 2) if not math.isinf(pf) else math.inf
        })

        # Selection uses TRAINING data only.
        if (
            len(group) >= MIN_TRAIN_TRADES
            and pf >= MIN_TRAIN_PF
            and net > 0
        ):
            selected.append((direction, int(hour)))

    ranking_df = pd.DataFrame(ranking)

    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            ["Net P/L", "PF"], ascending=False
        )

    return selected, ranking_df


def main():
    df = load_trades()

    # Chronological split. No random shuffle.
    split_index = int(len(df) * TRAIN_FRACTION)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    print("\n========== DATASET ==========")
    print("Total trades :", len(df))
    print("First trade  :", df.iloc[0]["entry_time"])
    print("Last trade   :", df.iloc[-1]["entry_time"])
    print("Training     :", len(train))
    print("Holdout      :", len(test))
    print("Holdout from :", test.iloc[0]["entry_time"])

    # Learn candidate filters only from first 70%.
    selected, ranking = select_training_filters(train)

    print("\n====== TRAINING FILTER SELECTION ======")
    print("Minimum trades per pair:", MIN_TRAIN_TRADES)
    print("Minimum PF:", MIN_TRAIN_PF)

    if selected:
        for direction, hour in selected:
            ist_hour = (hour + 5) % 24
            print(
                f"{direction} | UTC {hour:02d}:00"
                f" | IST {ist_hour:02d}:30"
            )
    else:
        print("No filters passed training criteria.")

    print("\nTop training combinations:")
    if not ranking.empty:
        print(ranking.head(15).to_string(index=False))

    # Freeze filters and test them on the final 30%.
    selected_set = set(selected)

    mask = [
        (row.direction, int(row.utc_hour)) in selected_set
        for row in test[
            ["direction", "utc_hour"]
        ].itertuples(index=False)
    ]

    filtered_test = test.loc[mask].copy()

    print("\n========== TRAINING BASELINE ==========")
    print(pd.DataFrame([
        get_metrics(train, "All training trades")
    ]).to_string(index=False))

    print("\n========== HOLDOUT TEST ==========")
    results = [
        get_metrics(test, "All holdout trades"),
        get_metrics(filtered_test, "Filtered holdout")
    ]
    print(pd.DataFrame(results).to_string(index=False))

    # Save results.
    pd.DataFrame(
        selected,
        columns=["direction", "utc_hour"]
    ).to_csv("selected_filters_from_training.csv", index=False)

    filtered_test.to_csv(
        "filtered_holdout_trades.csv", index=False
    )

    print("\nFiles saved:")
    print("1. selected_filters_from_training.csv")
    print("2. filtered_holdout_trades.csv")

    print("\n========== IMPORTANT ==========")
    print("Evaluate filters using HOLDOUT results.")
    print("PF <= 1.0 or negative net P/L means reject the filters.")
    print("Very few holdout trades means weak evidence.")
    print("Do not repeatedly tune filters against the same holdout.")
    print("This tests time filters, not the underlying entry strategy.")
    print("Verify XM spread, contract size, commission and slippage.")


if __name__ == "__main__":
    main()
