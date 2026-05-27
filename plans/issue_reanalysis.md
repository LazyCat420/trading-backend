# Issue: Verify and Enable REANALYSIS_ENABLED Feature Flag in Production

## Description
The `REANALYSIS_ENABLED` feature flag is currently gated and defaults to `False` in `Settings`. It controls whether the trading bot uses a percentage of its analysis slots for re-analyzing existing watchlisted tickers to get multi-angle perspectives, vs purely analyzing fresh scraped tickers.

To safely enable this flag in production, we need to verify the performance, database scalability, and accuracy of the re-analysis logic.

## Tasks
- [ ] Verify that `app/pipeline/analysis/decision_engine.py` handles slot division correctly under `REANALYSIS_ENABLED = True`.
- [ ] Measure the DB performance impact of executing multiple analyses per ticker record.
- [ ] Perform a benchmark test comparing historical P&L with and without re-analysis enabled.
- [ ] Once verified, update the default value of `REANALYSIS_ENABLED` to `True` in `app/config/config.py`.
