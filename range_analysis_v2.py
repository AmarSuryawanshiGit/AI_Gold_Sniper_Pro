
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# GOLD RANGE MARKET ANALYSIS V1
# Uses existing V2 trades and M5 candles.
# Does NOT change the V2 strategy.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

M5_FILE = BASE_DIR / "M5.csv"
TRADES_FILE = BASE_DIR / "backtest_trades_v2.csv"

OUTPUT_TRADES = BASE_DIR / "range_analysis_trades.csv"
OUTPUT_SUMMARY = BASE_DIR / "range_analysis_summary.csv"

ADX_PERIOD = 14
ATR_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
BB_PERIOD = 20
RANGE_LOOKBACK = 12

# Initial experimental thresholds — not optimized values.
ADX_LEVELS = [15, 18, 20, 22]
EMA_DISTANCE_ATR = 0.15
EMA_SLOPE_ATR = 0.10


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df["time"] = pd.to_datetime(
        df["time"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    df = df.sort_values("time")
    df = df.drop_duplicates("time")
    df = df.reset_index(drop=True)

    return df


def calculate_indicators(df):
    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]
    previous_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    # ATR using Wilder-style smoothing
    atr = tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD
    ).mean()

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    plus_dm_smooth = plus_dm.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    minus_dm_smooth = minus_dm.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["atr"] = atr

    df["adx"] = dx.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    # EMA
    df["ema20"] = close.ewm(
        span=EMA_FAST, adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=EMA_SLOW, adjust=False
    ).mean()

    # EMA distance relative to ATR
    df["ema_distance_atr"] = (
        (df["ema20"] - df["ema50"]).abs()
        / df["atr"].replace(0, np.nan)
    )

    # EMA20 slope over 3 completed M5 bars
    df["ema20_slope_atr"] = (
        (df["ema20"] - df["ema20"].shift(3)).abs()
        / df["atr"].replace(0, np.nan)
    )

    # Bollinger Band Width
    bb_mid = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std()

    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)

    df["bb_width_pct"] = (
        (bb_upper - bb_lower)
        / bb_mid.replace(0, np.nan)
        * 100
    )

    # Last 12 M5 candles' high-low range relative to ATR
    rolling_high = high.rolling(RANGE_LOOKBACK).max()
    rolling_low = low.rolling(RANGE_LOOKBACK).min()

    df["rolling_range_atr"] = (
        (rolling_high - rolling_low)
        / df["atr"].replace(0, np.nan)
    )

    # A candle timestamp in MT5 CSV is assumed to be candle OPEN.
    # Features become available only after the M5 candle closes.
    df["available_time"] = (
        df["time"] + pd.Timedelta(minutes=5)
    )

    return df


def add_regime_labels(df, adx_threshold=18):
    df = df.copy()

    df["weak_adx"] = df["adx"] < adx_threshold

    df["compressed_ema"] = (
        df["ema_distance_atr"] < EMA_DISTANCE_ATR
    )

    df["flat_ema"] = (
        df["ema20_slope_atr"] < EMA_SLOPE_ATR
    )

    conditions = (
        df["weak_adx"].fillna(False).astype(int)
        + df["compressed_ema"].fillna(False).astype(int)
        + df["flat_ema"].fillna(False).astype(int)
    )

    df["range_score"] = conditions

    # At least 2 out of 3 conditions = RANGING
    df["regime"] = np.where(
        conditions >= 2,
        "RANGING",
        "TRENDING"
    )

    # Unknown indicator history should not be classified.
    invalid = (
        df["adx"].isna()
        | df["ema_distance_atr"].isna()
        | df["ema20_slope_atr"].isna()
    )

    df.loc[invalid, "regime"] = "UNKNOWN"

    return df


def calculate_metrics(df, label):
    if df.empty:
        return {
            "scenario": label,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0,
            "net_pnl_usd": 0,
            "profit_factor": 0,
            "max_drawdown_usd": 0
        }

    pnl = pd.to_numeric(
        df["net_profit_usd"], errors="coerce"
    ).fillna(0)

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())

    if gross_loss == 0:
        pf = float("inf") if gross_profit > 0 else 0
    else:
        pf = gross_profit / gross_loss

    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    max_dd = drawdown.min()

    return {
        "scenario": label,
        "trades": len(df),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / len(df) * 100, 2),
        "net_pnl_usd": round(pnl.sum(), 2),
        "profit_factor": round(pf, 2),
        "max_drawdown_usd": round(max_dd, 2)
    }


def main():
    print("\nLoading M5 candles and V2 trades...")

    m5 = load_csv(M5_FILE)
    trades = pd.read_csv(TRADES_FILE)

    trades.columns = [
        str(c).strip().lower() for c in trades.columns
    ]

    required_trade_cols = {
        "entry_time",
        "exit_time",
        "net_profit_usd",
        "exit_reason"
    }

    missing = required_trade_cols - set(trades.columns)

    if missing:
        raise ValueError(
            f"Trade CSV missing columns: {sorted(missing)}"
        )

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    )

    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"], utc=True, errors="coerce"
    )

    trades["net_profit_usd"] = pd.to_numeric(
        trades["net_profit_usd"], errors="coerce"
    )

    trades = trades.dropna(
        subset=["entry_time", "exit_time", "net_profit_usd"]
    )

    trades = trades.sort_values("entry_time").reset_index(drop=True)

    print("Calculating M5 indicators...")
    m5 = calculate_indicators(m5)

    # Main regime: 18 ADX + 2-of-3 rule
    m5 = add_regime_labels(m5, adx_threshold=18)

    # Attach latest fully closed M5 candle to each trade.
    features = [
        "available_time",
        "adx",
        "atr",
        "ema20",
        "ema50",
        "ema_distance_atr",
        "ema20_slope_atr",
        "bb_width_pct",
        "rolling_range_atr",
        "weak_adx",
        "compressed_ema",
        "flat_ema",
        "range_score",
        "regime"
    ]

    regime_data = m5[features].copy()
    regime_data = regime_data.sort_values("available_time")

    trades = pd.merge_asof(
        trades.sort_values("entry_time"),
        regime_data,
        left_on="entry_time",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True
    )

    trades["regime"] = trades["regime"].fillna("UNKNOWN")

    # --------------------------------------------------------
    # 1. Main regime comparison
    # --------------------------------------------------------

    print("\n========== MAIN REGIME ANALYSIS ==========")

    summary = []

    for regime in ["RANGING", "TRENDING", "UNKNOWN"]:
        subset = trades[trades["regime"] == regime]

        metrics = calculate_metrics(subset, regime)
        summary.append(metrics)

        print(f"\n{regime}")
        print(f"Trades       : {metrics['trades']}")
        print(f"Wins         : {metrics['wins']}")
        print(f"Losses       : {metrics['losses']}")
        print(f"Win rate     : {metrics['win_rate_pct']}%")
        print(f"Net P/L      : ${metrics['net_pnl_usd']}")
        print(f"Profit Factor: {metrics['profit_factor']}")
        print(f"Max Drawdown : ${metrics['max_drawdown_usd']}")

    # --------------------------------------------------------
    # 2. Compare: all trades vs skip range
    # --------------------------------------------------------

    print("\n========== RANGE FILTER IMPACT ==========")

    all_metrics = calculate_metrics(trades, "ALL V2 TRADES")

    non_range = trades[
        trades["regime"].isin(["TRENDING"])
    ]

    filtered_metrics = calculate_metrics(
        non_range, "SKIP RANGING + UNKNOWN"
    )

    summary.append(all_metrics)
    summary.append(filtered_metrics)

    for metrics in [all_metrics, filtered_metrics]:
        print(
            f"{metrics['scenario']}: "
            f"Trades={metrics['trades']}, "
            f"WinRate={metrics['win_rate_pct']}%, "
            f"Net=${metrics['net_pnl_usd']}, "
            f"PF={metrics['profit_factor']}, "
            f"MaxDD=${metrics['max_drawdown_usd']}"
        )

    # --------------------------------------------------------
    # 3. Test different ADX thresholds
    # Other two conditions remain unchanged.
    # --------------------------------------------------------

    print("\n========== ADX THRESHOLD EXPERIMENT ==========")

    for threshold in ADX_LEVELS:
        temp = m5.copy()

        temp["weak_adx_test"] = temp["adx"] < threshold

        score = (
            temp["weak_adx_test"].fillna(False).astype(int)
            + temp["compressed_ema"].fillna(False).astype(int)
            + temp["flat_ema"].fillna(False).astype(int)
        )

        temp["regime_test"] = np.where(
            score >= 2,
            "RANGING",
            "TRENDING"
        )

        invalid = (
            temp["adx"].isna()
            | temp["ema_distance_atr"].isna()
            | temp["ema20_slope_atr"].isna()
        )

        temp.loc[invalid, "regime_test"] = "UNKNOWN"

        temp_features = temp[
            ["available_time", "regime_test"]
        ].sort_values("available_time")

        test_trades = pd.merge_asof(
            trades.drop(
                columns=[
                    "available_time",
                    "adx",
                    "atr",
                    "ema20",
                    "ema50",
                    "ema_distance_atr",
                    "ema20_slope_atr",
                    "bb_width_pct",
                    "rolling_range_atr",
                    "weak_adx",
                    "compressed_ema",
                    "flat_ema",
                    "range_score",
                    "regime"
                ],
                errors="ignore"
            ).sort_values("entry_time"),
            temp_features,
            left_on="entry_time",
            right_on="available_time",
            direction="backward",
            allow_exact_matches=True
        )

        selected = test_trades[
            test_trades["regime_test"] == "TRENDING"
        ]

        metrics = calculate_metrics(
            selected, f"ADX_{threshold}_SKIP_RANGE"
        )

        summary.append(metrics)

        print(
            f"ADX<{threshold}: "
            f"Trades={metrics['trades']}, "
            f"WinRate={metrics['win_rate_pct']}%, "
            f"Net=${metrics['net_pnl_usd']}, "
            f"PF={metrics['profit_factor']}, "
            f"MaxDD=${metrics['max_drawdown_usd']}"
        )

    # --------------------------------------------------------
    # 4. Save reports
    # --------------------------------------------------------

    trades.to_csv(OUTPUT_TRADES, index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n========== FILES SAVED ==========")
    print(OUTPUT_TRADES.name)
    print(OUTPUT_SUMMARY.name)

    print("\nImportant:")
    print("- V2 entries and exits have not been changed.")
    print("- This is a historical regime classification study.")
    print("- ADX thresholds are experimental, not optimized.")
    print("- M5 features use only candles closed by entry time.")
    print("- Spread, commission and slippage assumptions remain")
    print("  those already present in the V2 trade export.")
    print("- This does not independently rerun the strategy.")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
