# Top-of-Book Microstructure Signals for Short-Horizon Midprice Direction

This project tests whether a small, interpretable set of top-of-book state and recent
trade-flow variables contains out-of-sample predictive information about short-horizon
BTCUSDT midprice direction.

The study is deliberately narrow:

- one instrument: Binance USDT-M perpetual `BTCUSDT`
- top-of-book quotes and aggregate trades only
- event-time labels at 10, 20, and 50 distinct quote-state updates
- chronological train / validation / test splits
- simple baselines and multinomial logistic regression
- explicit leakage, split-boundary, and timestamp-ordering audits

The purpose is not to claim a tradable strategy. The purpose is to test whether weak
statistical signal survives an honest out-of-sample protocol, and to document where it
breaks.

## Current Status

The current canonical experiment is the `v2_float64_features` 7-day checkpoint:

- symbol: `BTCUSDT`
- sample: `2024-03-01` to `2024-03-07` UTC
- artifact tag: `v2_float64_features`
- quote-event rows after feature/label construction: about 223 million
- model-eligible final test rows: 28,715,045
- `mid_return_5` and `realized_vol_20` are computed and stored as `float64`
- the log-return-derived feature precision audit now passes
- compact prediction/probability summary diagnostics are included under `outputs/reports`
- plots and a current result manifest are available under `outputs/reports`
- selected plots are discussed in `docs/current_results.md`
- frozen final-test evaluation is complete for this checkpoint

The research protocol targets 30 clean calendar days, or at least 14 if 30 are not
available. This 7-day v2 run should therefore be treated as a checkpoint, not the
final generalization claim. A longer 14-day replication remains pending.

## Research Question

At each distinct top-of-book quote-state event, can current book state and recent trade
flow predict whether the midprice moves down, stays unchanged, or moves up after a
short event-time horizon?

For horizon `h`, the label is:

```text
y_h = sign(midprice[t + h] - midprice[t])
```

The primary task is ternary classification over:

```text
-1 = down
 0 = unchanged
 1 = up
```

The non-zero subset is also reported as a diagnostic, but it is not the headline task.

## Data and Event Construction

Raw data comes from Binance public USDT-M futures daily files:

- `bookTicker`: top-of-book quotes
- `aggTrades`: aggregate trades

The unit of analysis is a distinct quote-state event. Consecutive duplicate quote states
with identical bid, ask, bid size, and ask size are collapsed. Event time is therefore
the index of distinct top-of-book states, not raw message count.

Quote rows are rejected if:

- bid price exceeds ask price
- bid or ask price is non-positive
- displayed bid or ask size is non-positive

The data quality report also checks quote/trade timestamp gaps and invalid records by
UTC day before event construction.

## Features

The main feature set is intentionally small and interpretable.

Book-state features:

- relative spread
- queue imbalance
- log bid size
- log ask size

Short-memory quote-state features:

- change in bid size
- change in ask size
- 5-event log midprice return
- 20-event realized volatility proxy

Recent trade-flow features over the previous one second:

- trade intensity
- signed trade count imbalance
- signed trade volume imbalance

Trade-flow features use trades in the interval:

```text
(quote_timestamp - 1000 ms, quote_timestamp)
```

Trades exactly equal to the quote-event timestamp are excluded.

## Splits and Leakage Controls

The split is chronological and timestamp-based:

- train: first 60% of the sample window
- validation: next 20%
- test: final 20%

Important controls:

- no random split
- no shuffling
- scalers fit on train only through an sklearn pipeline
- features use only current and past quote/trade information
- labels use future midprices only for the target
- the last 50 labeled observations of train and validation are marked as boundary drops
- validation is used for threshold selection and diagnostics
- final test is evaluated once for final reporting

The audit checks:

- label alignment against manually recomputed event-time labels
- split-boundary target spillover
- quote-feature timing
- `float64` recomputation of log-return-derived features
- equal-timestamp trade exclusion
- target drift across train / validation / test
- daily test stability
- coefficient sign interpretation
- confusion matrix behavior under class imbalance
- spread-relative label magnitude
- protocol-defined regime performance by spread, realized volatility, and trade intensity
- For realized volatility, the train median is zero, so the high-volatility bucket should be interpreted as a nonzero-volatility / more active regime rather than a balanced half-sample split.
Audit summary:

```text
critical_audit_passed: true
leakage_evidence: no_evidence_found
main_caveat: pass
log_feature_precision_status: pass
```

## Models

The main comparison uses:

- `majority_baseline`: predicts the train majority class
- `queue_imbalance_logistic`: multinomial logistic regression using queue imbalance only
- `full_logistic`: multinomial logistic regression using all 11 features
- `full_logistic_thresholded`: secondary validation-selected probability threshold rule

The primary model is `full_logistic`. The thresholded model is a secondary operating
point selected on validation macro F1 before final test evaluation.

## Frozen Final Test Results

Final test metrics for the current 7-day v2 frozen experiment:

| Horizon | Model | Accuracy | Macro F1 | Balanced Accuracy |
| ---: | --- | ---: | ---: | ---: |
| 10 | majority baseline | 0.798 | 0.296 | 0.333 |
| 10 | queue-imbalance logistic | 0.798 | 0.296 | 0.333 |
| 10 | full logistic | 0.821 | 0.454 | 0.423 |
| 10 | full logistic, thresholded | 0.770 | 0.584 | 0.600 |
| 20 | majority baseline | 0.686 | 0.271 | 0.333 |
| 20 | queue-imbalance logistic | 0.686 | 0.271 | 0.333 |
| 20 | full logistic | 0.724 | 0.435 | 0.423 |
| 20 | full logistic, thresholded | 0.705 | 0.628 | 0.647 |
| 50 | majority baseline | 0.468 | 0.213 | 0.333 |
| 50 | queue-imbalance logistic | 0.468 | 0.213 | 0.333 |
| 50 | full logistic | 0.556 | 0.443 | 0.456 |
| 50 | full logistic, thresholded | 0.657 | 0.660 | 0.664 |

The full-feature logistic model beats both baselines on final-test macro F1 and
balanced accuracy at all three horizons. The default argmax decision rule remains
conservative and predicts `unchanged` very often. The validation-selected thresholded
rule sacrifices some unchanged-class accuracy but recovers substantially more down/up
recall.

Non-zero subset performance for the thresholded model:

| Horizon | Non-zero Accuracy | Non-zero Macro F1 | Non-zero Balanced Accuracy |
| ---: | ---: | ---: | ---: |
| 10 | 0.476 | 0.418 | 0.477 |
| 20 | 0.591 | 0.485 | 0.591 |
| 50 | 0.681 | 0.530 | 0.681 |

## Interpretation

The result supports a narrow statistical claim:

> In this frozen 7-day BTCUSDT sample, simple top-of-book state and recent trade-flow
> features contain out-of-sample information about short-horizon event-time midprice
> direction, relative to majority and queue-imbalance-only baselines.

The result does not support a trading claim. The project does not model:

- fees
- latency
- queue position
- fill probability
- market impact
- adverse selection
- executable PnL
- cross-venue or cross-asset generalization

The audit also found strong target drift: validation and test contain materially more
directional labels than train, especially at longer horizons. This makes random-split
style evaluation inappropriate and reinforces the need for replication on a fresh,
longer sample.

## Limitations

1. The current frozen experiment is a 7-day v2 checkpoint. The final protocol target
   is 14 or 30 clean days, so the result should not be read as a full-sample
   generalization claim.
2. The study covers one instrument and one venue: Binance USDT-M `BTCUSDT`.
3. The model is deliberately simple and linear. This is useful for auditability and
   interpretation, but it does not test more flexible nonlinear learners.
4. Validation and test contain materially more directional labels than train. The
   target drift is reported explicitly and motivates replication on a longer sample.
5. Regime-performance tables are included, but the current regime conclusions are
   still based on the 7-day checkpoint sample.

## Repository Structure

```text
src/
  artifact_naming.py             # tagged artifact filename helpers
  data_loader.py                 # Binance download/load/parsing helpers
  quality_checks.py              # raw quote/trade quality checks
  event_builder_opt.py           # memory-lean distinct quote-event construction
  feature_builder_opt_float_64.py # v2 float64 log-return-derived feature builder
  split_config.py                # chronological split and boundary drops
  modeling/
    majority_baseline.py
    logistic_models.py
  evaluation/
    metrics.py
    model_comparison.py
    probability_diagnostics.py
    target_diagnostics.py

scripts/
  load_data.py
  quality_report.py
  build_events.py
  build_features_labels_opt.py
  build_splits.py
  run_majority_baseline.py
  run_queue_imbalance_logistic.py
  run_full_logistic.py
  run_probability_diagnostics.py
  run_final_test_evaluation.py
  run_regime_analysis.py
  run_research_audit.py
  save_compact_test_predictions.py
  make_result_plots.py
  write_current_result_manifest.py
  smoke_test_metrics.py

docs/
  microstructure_signal_memo.md
  current_results.md

outputs/
  reports/
  results/
  logs/
```

## Setup

The pinned conda environment is named `quant_env`:

```bash
conda activate quant_env
```

For a fresh environment, install the Python dependencies from `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate quant_env
```

Alternatively, create a virtual environment and install pinned pip dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the lightweight metric smoke test:

```bash
python scripts/smoke_test_metrics.py
```

### 1-Day Pipeline Smoke Run

This run exercises the full pipeline shape on a smaller date range. It is for
installation and pipeline validation only; do not compare its metrics to the frozen
7-day research result.

If raw zip files are already present locally, omit `--download`.

```bash
START=2024-03-01
END=2024-03-01
TAG=v2_float64_features

python scripts/load_data.py --start "$START" --end "$END" --symbol BTCUSDT --download
python scripts/quality_report.py --start "$START" --end "$END" --symbol BTCUSDT
python scripts/build_events.py --start "$START" --end "$END" --symbol BTCUSDT
python scripts/build_features_labels_opt.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/build_splits.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/target_diagnostics.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/run_majority_baseline.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/run_queue_imbalance_logistic.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --max-iter 50
python scripts/run_full_logistic.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --solver saga --max-iter 50 --n-jobs -1
python scripts/run_probability_diagnostics.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --solver saga --max-iter 50 --n-jobs -1
python scripts/run_final_test_evaluation.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --solver saga --max-iter 50 --n-jobs -1
python scripts/run_regime_analysis.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --splits test
```

## Reproducing the Current Pipeline

The current frozen run uses:

```text
START=2024-03-01
END=2024-03-07
SYMBOL=BTCUSDT
TAG=v2_float64_features
```

The full 7-day run is compute-heavy. It processes hundreds of millions of
quote-event rows and can produce tens of GB of local data/output artifacts. Before
running the full pipeline, use the 1-day smoke run above to check that the local
environment, data paths, and scripts work.

Download and parse local Binance files:

```bash
python scripts/load_data.py --start "$START" --end "$END" --symbol "$SYMBOL" --download
```

If the raw zip files are already present, omit `--download`:

```bash
python scripts/load_data.py --start "$START" --end "$END" --symbol "$SYMBOL"
```

Build the data quality report:

```bash
python scripts/quality_report.py --start "$START" --end "$END" --symbol "$SYMBOL"
```

Build distinct quote events:

```bash
python scripts/build_events.py --start "$START" --end "$END" --symbol "$SYMBOL"
```

Build labels and `float64` log-return-derived features:

```bash
python scripts/build_features_labels_opt.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
```

Build chronological train / validation / test splits:

```bash
python scripts/build_splits.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
```

Freeze dataset metadata and target diagnostics:

```bash
python scripts/target_diagnostics.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
```

Run validation models and diagnostics:

```bash
python scripts/run_majority_baseline.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
python scripts/run_queue_imbalance_logistic.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --max-iter 300
python scripts/run_full_logistic.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --solver saga --max-iter 300 --n-jobs -1
python scripts/compare_validation_results.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
python scripts/run_probability_diagnostics.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --solver saga --max-iter 300 --n-jobs -1
```

Run final test evaluation only after validation diagnostics are frozen:

```bash
python scripts/run_final_test_evaluation.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --solver saga --max-iter 300 --n-jobs -1
```

Run protocol regime analysis using train-median cutoffs applied unchanged to validation/test:

```bash
python scripts/run_regime_analysis.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
```

Run the research audit:

```bash
python scripts/run_research_audit.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --experiment-tag "$TAG"
```

Save compact test-only prediction/probability diagnostics:

```bash
python scripts/save_compact_test_predictions.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
```

Create compact plots and the current result manifest:

```bash
python scripts/make_result_plots.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
python scripts/write_current_result_manifest.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
```

Important: after `run_final_test_evaluation.py`, do not change features, thresholds,
labels, or model selection based on this test result. Any further changes should be a
new experiment on a fresh validation/test protocol.

## Key Output Files

Current frozen result files:

```text
data/processed/feature_table_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet
data/processed/model_dataset_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet
outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_aggregate.csv
outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_nonzero_subset.csv
outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_daily_blocks.csv
outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_aggregate.csv
outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_nonzero_subset.csv
outputs/results/compact_test_predictions_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet
outputs/reports/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_full_vs_baselines_deltas.csv
outputs/reports/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_frozen_thresholds.csv
outputs/reports/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_thresholds.csv
outputs/reports/compact_test_predictions_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_summary.csv
outputs/reports/current_result_manifest.md
outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_target_drift.png
outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_horizon_performance.png
outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_thresholded_vs_argmax_recall.png
outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_performance.png
outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_nonzero_performance.png
outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v2_float64_features/audit_summary.json
outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v2_float64_features/final_audit_conclusion.csv
```

## Future Work

Planned extensions:

1. Add prediction-conditioned diagnostics using the compact probability summaries.
2. Add an incremental-learning model as a secondary extension using the same features, labels, splits, and validation-only model selection discipline.
3. Run a fresh 14-day clean-sample replication after the static and incremental pipelines are fixed.
4. Update the README, memo, and result review after the replication run.
