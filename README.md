# regime-risk-bot (stage 1: backtest only)

Rule-based regime filter + RSI pullback signal + leverage-3x risk layer for Freqtrade (futures, isolated).
No ML yet. Untested template: expect to fix small things on the first run.

## Run from a phone
GitHub -> Actions -> `backtest` -> Run workflow. Read the Summary tab (and the `results` artifact).
If the exchange refuses the runner's IP (Binance/Bybit often do), try `okx`, `gate`, or `hyperliquid`.

## Move to dry-run ONLY if all hold (suggested thresholds, adjust yourself)
- Net profit > 0 after fees in BOTH in-sample and out-of-sample ranges
- Max drawdown acceptable to you (e.g. < 20%), profit factor > ~1.3
- >= 100 trades per range, profit not concentrated in 1-2 trades/months/pairs
- 0 liquidations; lookahead-analysis and recursive-analysis show no bias
- Result survives small param changes (adx_min, rsi levels, stop mult)

If not positive: change ONE thing, re-run, compare. Do not tune on the out-of-sample range.

## Later (dry-run / live)
Needs a VPS. Use API keys WITHOUT withdrawal rights, keep keys out of git (use a separate config_live.json).
Not financial advice; futures with leverage can lose the whole margin.
