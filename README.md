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

The current canonical experiment is the `v3_fixed_window_features` 7-day checkpoint:

- symbol: `BTCUSDT`
- sample: `2024-03-01` to `2024-03-07` UTC
- artifact tag: `v3_fixed_window_features`
- distinct quote-state events: 223,033,235
- feature/label rows after the 50-event terminal drop: 223,033,185
- model-eligible final test rows: 28,715,045
- `mid_return_5` and `realized_vol_20` are computed and stored as `float64`
- the log-return-derived feature precision audit passes
- compact prediction/probability summary diagnostics are included under `outputs/reports`
- plots and a current result manifest are available under `outputs/reports`
- selected plots are discussed in `docs/current_results.md`
- frozen final-test evaluation is complete for this checkpoint
- pooled, regime-conditioned, and compact-prediction results reconcile exactly
- integration verification passes with maximum metric reconstruction error of
  `1.11e-16`

The research protocol targets 30 clean calendar days, or at least 14 if 30 are not
available. This 7-day run should therefore be treated as a checkpoint, not the
final generalization claim. A longer 14-day replication remains pending.

## Engineering Safeguards

The pipeline now fails closed around the main numerical and leakage boundaries:

- canonical horizons, features, labels, and split rules live in `src/protocol.py`
- fitted logistic artifacts contain ordered features/classes, scaler state,
  coefficients, intercepts, convergence metadata, and dataset/protocol hashes
- model metadata includes a SHA-256 checksum of its array payload, so incomplete
  or mixed artifact replacements fail closed
- validation thresholds are bound to an exact fingerprint of the scaler,
  coefficients, intercepts, feature order, and class order
- serialized probabilities must match the fitted sklearn estimator, after applying
  the stored train-only scaler, within `1e-12`
- probability diagnostics, final test, regime analysis, and compact predictions
  consume frozen artifacts rather than refitting or reconstructing scaler state
- raw quality checks, event construction, feature construction, split construction,
  target diagnostics, and high-volume audit checks use bounded-memory parquet
  processing
- model data is stored by chronological split, and readers request only required
  columns and partitions
- CI runs compilation, Ruff, the invariant test suite, and at least 75% coverage across
  the critical optimized modules

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
- scaler statistics fit on train only with ordered `StandardScaler.partial_fit`
- features use only current and past quote/trade information
- labels use future midprices only for the target
- the last 50 labeled observations of train and validation are marked as boundary drops
- validation is used for threshold selection and diagnostics
- test labels are not used for feature, model, or threshold selection; the canonical
  v3 result is frozen after evaluation

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
- protocol-defined regime cutoffs for spread, realized volatility, and trade intensity

Regime analysis uses medians estimated on the training split and applies those
cutoffs unchanged to validation and test. Predictions come from the same complete
frozen scaler/model artifacts used by the pooled evaluation. The integration
verifier independently reconstructs pooled metrics, recombines regime slices, and
checks compact-prediction confusion counts.

Audit summary:

```text
critical_audit_passed: true
leakage_evidence: no_evidence_found
main_result_status: frozen_test_result_supported_by_audit
log_feature_precision_status: pass
```

The audit combines deterministic full-table checks with sampled manual
recomputations. Passing it supports the result within that scope; it is not a
proof that every possible implementation error has been excluded.

## Models

The main comparison uses:

- `majority_baseline`: predicts the train majority class
- `queue_imbalance_logistic`: multinomial logistic regression using queue imbalance only
- `full_logistic`: multinomial logistic regression using all 11 features
- `full_logistic_thresholded`: secondary validation-selected probability threshold rule

The primary model is `full_logistic`. The thresholded model is a secondary operating
point selected on validation macro F1 before final test evaluation. Its down/up
thresholds are chosen from a fixed 15-by-15 grid on one validation segment, so the
larger thresholded gains carry more model-selection uncertainty than the primary
argmax result and require fresh-sample replication.

## Frozen Final Test Results

Final test metrics for the current 7-day v3 frozen experiment:

| Horizon | Model | Accuracy | Macro F1 | Balanced Accuracy |
| ---: | --- | ---: | ---: | ---: |
| 10 | majority baseline | 0.798 | 0.296 | 0.333 |
| 10 | queue-imbalance logistic | 0.798 | 0.296 | 0.333 |
| 10 | full logistic | 0.821 | 0.454 | 0.423 |
| 10 | full logistic, thresholded | 0.770 | 0.584 | 0.600 |
| 20 | majority baseline | 0.686 | 0.271 | 0.333 |
| 20 | queue-imbalance logistic | 0.686 | 0.271 | 0.333 |
| 20 | full logistic | 0.724 | 0.434 | 0.423 |
| 20 | full logistic, thresholded | 0.705 | 0.628 | 0.648 |
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
| 20 | 0.592 | 0.486 | 0.593 |
| 50 | 0.681 | 0.530 | 0.681 |

Regime-conditioned balanced accuracy for the thresholded 50-event model:

| Regime variable | Lower regime | Balanced Accuracy | Upper regime | Balanced Accuracy |
| --- | --- | ---: | --- | ---: |
| Relative spread | tight | 0.651 | wide | 0.568 |
| Realized volatility | zero | 0.597 | positive | 0.664 |
| Trade intensity | low | 0.641 | high | 0.663 |

The realized-volatility training median is exactly zero, so that diagnostic is a
zero-versus-positive split rather than two similarly sized volatility buckets.
Regime results are descriptive diagnostics from the same final test, not separate
confirmatory tests.

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

1. The current frozen experiment is a 7-day checkpoint. The final protocol target
   is 14 or 30 clean days, so the result should not be read as a full-sample
   generalization claim.
2. The study covers one instrument and one venue: Binance USDT-M `BTCUSDT`.
3. The model is deliberately simple and linear. This is useful for auditability and
   interpretation, but it does not test more flexible nonlinear learners.
4. Validation and test contain materially more directional labels than train. The
   target drift is reported explicitly and motivates replication on a longer sample.
5. The final test covers about 33.6 hours over two UTC dates. Regime slices are
   therefore useful for diagnosis but too temporally narrow for broad stability claims.
6. The realized-volatility median is zero, making its low/high regime split
   structurally imbalanced and best interpreted as zero versus positive volatility.
7. Adjacent event-time labels overlap and observations are serially dependent. The
   row count is therefore not an independent sample size; this checkpoint reports
   predictive metrics, not iid standard errors, p-values, or confidence intervals.
8. The thresholded operating point was selected on a single validation segment.
   Its test performance is secondary evidence and may include validation-selection
   optimism even though the test labels were not used to choose the thresholds.

## Repository Structure

```text
src/
  artifact_naming.py             # tagged artifact filename helpers
  data_loader.py                 # Binance download/load/parsing helpers
  quality_checks.py              # raw quote/trade quality checks
  event_builder_opt.py           # memory-lean distinct quote-event construction
  feature_builder_opt_float_64.py # float64 log-return-derived feature builder
  streaming_features.py          # bounded-memory exact feature construction
  partitioned_splits.py          # two-pass split-specific parquet writer
  model_dataset_io.py            # split/column-pruned model-data reads
  protocol.py                    # canonical experiment definition and fingerprint
  split_config.py                # chronological split and boundary drops
  modeling/
    majority_baseline.py
    logistic_models.py
    model_artifacts.py
  evaluation/
    metrics.py
    model_comparison.py
    probability_diagnostics.py
    target_diagnostics.py

scripts/
  load_data.py
  quality_report.py
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
  engineering_design.md

outputs/
  reports/
  results/
  logs/

tests/                         # numerical, leakage, and boundary invariants
benchmarks/                    # reproducible Pandas/Polars benchmark
.github/workflows/ci.yml       # compile, lint, test, and coverage checks
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

Install development checks and run the full local quality gate:

```bash
python -m pip install -r requirements-dev.txt
ruff check src scripts tests benchmarks
python -m pytest
```

### 1-Day Pipeline Smoke Run

This run exercises the full pipeline shape on a smaller date range. It is for
installation and pipeline validation only; do not compare its metrics to the frozen
7-day research result.

If raw zip files are already present locally, omit `--download`.

```bash
START=2024-03-01
END=2024-03-01
TAG=v3_fixed_window_features

python scripts/load_data.py --start "$START" --end "$END" --symbol BTCUSDT --download
python scripts/quality_report.py --start "$START" --end "$END" --symbol BTCUSDT
python scripts/build_events.py --start "$START" --end "$END" --symbol BTCUSDT
python scripts/build_features_labels_opt.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/build_splits.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/target_diagnostics.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/run_majority_baseline.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/run_queue_imbalance_logistic.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/run_full_logistic.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/compare_validation_results.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/run_probability_diagnostics.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG"
python scripts/run_final_test_evaluation.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --threshold-selection-metric macro_f1
python scripts/run_regime_analysis.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --threshold-selection-metric macro_f1 --splits validation,test
python scripts/save_compact_test_predictions.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --threshold-selection-metric macro_f1
python scripts/run_research_audit.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --experiment-tag "$TAG"
python scripts/verify_integration_run.py --start "$START" --end "$END" --symbol BTCUSDT --artifact-tag "$TAG" --threshold-selection-metric macro_f1
```

Model fitting and probability diagnostics commit a small checkpoint after each
completed horizon. If a run is interrupted, repeat the same command with
`--resume`. Resume fails closed if the dataset, protocol, features, optimizer
configuration, source-tree fingerprint, or frozen model parameters differ.

## Running the Canonical 7-Day Pipeline

The canonical regeneration uses:

```text
START=2024-03-01
END=2024-03-07
SYMBOL=BTCUSDT
TAG=v3_fixed_window_features
```

The full 7-day run is compute-heavy. It processes hundreds of millions of
quote-event rows and can produce tens of GB of local data/output artifacts. Before
running the full pipeline, use the 1-day smoke run above to check that the local
environment, data paths, and scripts work.

Create a reviewed local commit before this run. Canonical high-volume commands
below use `--require-clean-git`; development smoke runs may omit it. Each stage
writes JSON metadata under `outputs/reports/run_metadata/`, including the Git
commit, dirty/clean status, source-tree SHA-256, command arguments, elapsed time,
peak resident memory, and available dataset/protocol fingerprints.

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
python scripts/build_events.py --start "$START" --end "$END" --symbol "$SYMBOL" --require-clean-git
```

Build labels and `float64` log-return-derived features:

```bash
python scripts/build_features_labels_opt.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --require-clean-git
```

Build chronological train / validation / test splits:

```bash
python scripts/build_splits.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --require-clean-git
```

Freeze dataset metadata and target diagnostics:

```bash
python scripts/target_diagnostics.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
```

Run validation models and diagnostics:

```bash
python scripts/run_majority_baseline.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --require-clean-git
python scripts/run_queue_imbalance_logistic.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --require-clean-git
python scripts/run_full_logistic.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --require-clean-git
python scripts/compare_validation_results.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG"
python scripts/run_probability_diagnostics.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --require-clean-git
```

After an interruption, repeat only the affected checkpointed command with
`--resume --require-clean-git`. Do not combine `--resume` and `--overwrite`.

Run final test evaluation only after validation diagnostics are frozen:

```bash
python scripts/run_final_test_evaluation.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --threshold-selection-metric macro_f1 --require-clean-git
```

Run protocol regime analysis using train-median cutoffs applied unchanged to validation/test:

```bash
python scripts/run_regime_analysis.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --threshold-selection-metric macro_f1 --splits validation,test --require-clean-git
```

Run the research audit:

```bash
python scripts/run_research_audit.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --experiment-tag "$TAG"
```

Save compact test-only prediction/probability diagnostics:

```bash
python scripts/save_compact_test_predictions.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --threshold-selection-metric macro_f1 --require-clean-git
```

Verify row counts, dataset identity, metric reconstruction, regime recombination,
compact-prediction parity, and audit status:

```bash
python scripts/verify_integration_run.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --threshold-selection-metric macro_f1
```

Create compact plots and the current result manifest:

```bash
python scripts/make_result_plots.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --include-regime-plot
python scripts/write_current_result_manifest.py --start "$START" --end "$END" --symbol "$SYMBOL" --artifact-tag "$TAG" --include-regime-artifacts
```

Important: after `run_final_test_evaluation.py`, do not change features, thresholds,
labels, or model selection based on this test result. Any further changes should be a
new experiment on a fresh validation/test protocol.

## Key Output Files

Current frozen result files:

```text
data/processed/feature_table_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet
data/processed/model_dataset_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07/
outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_aggregate.csv
outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_nonzero_subset.csv
outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_aggregate.csv
outputs/results/compact_test_predictions_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet
outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_frozen_thresholds.csv
outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_thresholds.csv
outputs/reports/compact_test_predictions_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_summary.csv
outputs/reports/current_result_manifest.md
outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_horizon_performance.png
outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_performance.png
outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_target_drift.png
outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_thresholded_vs_argmax_recall.png
outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v3_fixed_window_features/audit_summary.json
outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v3_fixed_window_features/final_audit_conclusion.csv
```

## Future Work

Planned extensions:

1. Add a delayed-feedback incremental-learning extension using the same
   features, labels, splits, and validation-only model selection discipline.
2. Add calibration, abstention, and transaction-cost sensitivity diagnostics
   without presenting them as an executable PnL backtest.
3. Freeze the extension protocol before running a fresh 14-day comparison of
   static, rolling-refit, and online models.
4. Test cross-period and, where data permits, cross-instrument stability.
