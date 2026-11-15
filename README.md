# AI Gold Sniper Pro (Windows 11)
Dark PySide6 desktop terminal for XM MT5 XAUUSD monitoring. Independent pattern alerts, separate trend score, trade-candidate filter, Telegram, chart annotations, SQLite journal and CSV event-study backtest. No orders are placed.

## Install
1. Install Python 3.11 x64 and MetaTrader 5 desktop; log into XM DEMO.
2. Extract ZIP, open PowerShell in this folder:
   `py -3.11 -m venv .venv`
   `.venv\Scripts\Activate.ps1`
   `pip install -r requirements.txt`
   `copy .env.example .env`
3. Set exact XM Market Watch gold symbol in `.env` (XAUUSD / GOLD / suffix).
4. Run `python main.py`.
5. Build EXE with `build_exe.bat`; output is `dist\AI_Gold_Sniper_Pro\AI_Gold_Sniper_Pro.exe`.

MT5 desktop must remain open and logged in. Never put your MT5 password in the app. Telegram token/chat ID are optional and should be kept private.

## Included patterns
Liquidity sweeps, BOS/CHoCH approximations, FVG, engulfing, pin bars, inside bars, equal highs/lows, trendline regression breaks, double top/bottom heuristics. Pattern alerts are independent of trend score. Trade candidates are a separate filter.

## Safety / limitations
No tool can guarantee 90% accuracy. Trend score is an alignment score, not win probability. Detectors are heuristic and require testing. Backtest is a simplified event study, not a broker-exact simulator; costs, spread, slippage, commissions, swaps, news and execution constraints are not fully modelled. No automatic trading. Use demo first and verify XM contract size, tick value, lot minimum/step, margin and stop-level rules.
