
import pandas as pd

# Load backtest results
d = pd.read_csv("backtest_trades.csv")

# Extract UTC hour
d["hour"] = pd.to_datetime(
    d["entry_time"], utc=True
).dt.hour

rows = []

# Analyze each direction and hour
for (direction, hour), g in d.groupby(["direction", "hour"]):

    wins = g.loc[g["pnl_usd"] > 0, "pnl_usd"].sum()
    losses = abs(g.loc[g["pnl_usd"] < 0, "pnl_usd"].sum())

    profit_factor = (
        round(wins / losses, 2) if losses > 0 else float("inf")
    )

    rows.append({
        "direction": direction,
        "UTC": hour,
        "trades": len(g),
        "win_rate": round((g["pnl_usd"] > 0).mean() * 100, 1),
        "net": round(g["pnl_usd"].sum(), 2),
        "avg": round(g["pnl_usd"].mean(), 2),
        "PF": profit_factor
    })

result = pd.DataFrame(rows).sort_values(
    "net", ascending=False
)

print("\n===== DIRECTION + HOURLY ANALYSIS =====")
print(result.to_string(index=False))

# Save detailed report
result.to_csv("hourly_analysis.csv", index=False)

print("\nSaved: hourly_analysis.csv")
