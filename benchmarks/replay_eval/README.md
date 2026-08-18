# Replay evaluation

Tools for retuning `Config.search_ranking.calibration_threshold` against real
query traffic, without touching production while the sweep runs.

## Retuning the calibration threshold

`Config.search_ranking.calibration_threshold` defaults to 0.035, above the
maximum fused score the pipeline actually produces, so the sigmoid only ever
operates in its lower tail. `sweep.py` replays real labeled queries under a
baseline threshold and a set of challenger thresholds, and reports each
challenger's measured gain over the baseline with a bootstrap confidence
interval.

1. Extract fresh labels from telemetry (skip if `artifacts/labels.jsonl` is
   already current enough):

   ```
   uv run python -m benchmarks.replay_eval.cli
   ```

2. Pin a corpus snapshot so every threshold is compared against identical
   data. This step is required before running the sweep:

   ```
   ./scripts/replay-snapshot.sh
   ```

3. Run the sweep:

   ```
   uv run python -m benchmarks.replay_eval.sweep \
     --baseline-threshold 0.035 \
     --challenger-threshold 0.02 \
     --challenger-threshold 0.01 \
     --dsn postgresql://replay:replay@127.0.0.1:5455/mcp_memory \
     --seed 0
   ```

   A full sweep replays every label in `artifacts/labels.jsonl` (16k+ rows) once
   per threshold, each against the real production-composed search path. Budget
   roughly 10-20 minutes per variant, so a baseline plus a handful of
   challengers is an hour-plus run. Pass `--output` to change where the JSON
   report lands (default `artifacts/threshold-sweep.json`).

4. Read the report. Each entry under `"variants"` is a `VariantMetrics`
   mapping (mean reciprocal rank, hit@k, unretrieved rate) for one threshold;
   `"delta_vs_baseline"` on a challenger is a `PairedDelta` mapping — check
   `significant` before trusting `delta`, since an interval that still spans
   zero is not evidence either way.

Only adopt a new `calibration_threshold` once a challenger shows a
significant, positive delta; a threshold that only matches the baseline is not
worth the config change.
