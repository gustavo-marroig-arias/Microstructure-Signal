# Current Result Manifest

- symbol: `BTCUSDT`
- sample: `2024-03-01` to `2024-03-07` UTC
- artifact tag: `v3_fixed_window_features`
- status: frozen 7-day checkpoint

## Audit Status

- `critical_audit_passed`: `True`
- `log_feature_precision_status`: `pass`
- `mid_return_5_all_match_float64`: `True`
- `realized_vol_20_all_match_float64`: `True`
- `main_result_status`: `frozen_test_result_supported_by_audit`

## File Manifest

`Exists` records the generation machine at manifest creation; only rows
marked `yes` under `Included?` belong to the public snapshot.

| Category | Path | Exists | Size MB | Included? | Note |
| --- | --- | ---: | ---: | --- | --- |
| data | `data/processed/feature_table_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet` | True | 2743.89 | no | excluded large artifact |
| data | `data/processed/model_dataset_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07` | True | 2479.96 | no | excluded large artifact |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_aggregate.csv` | True | 0.00 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_class_proportions.csv` | True | 0.00 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_confusion_counts.csv` | True | 0.01 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_confusion_true_normalized.csv` | True | 0.01 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_daily_blocks.csv` | True | 0.01 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_logistic_coefficients.csv` | True | 0.01 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_majority_fit_summary.csv` | True | 0.00 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_nonzero_subset.csv` | True | 0.00 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_per_class.csv` | True | 0.00 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_aggregate.csv` | True | 0.07 | yes | small CSV |
| small_result | `outputs/results/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_nonzero_subset.csv` | True | 0.05 | yes | small CSV |
| small_report | `outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_frozen_thresholds.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_full_vs_baselines_deltas.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_ranking.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_signal_decay.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_thresholded_deltas.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_thresholds.csv` | True | 0.00 | yes | small CSV |
| diagnostic_data | `outputs/results/compact_test_predictions_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet` | True | 1160.89 | no | excluded row-level parquet |
| small_report | `outputs/reports/compact_test_predictions_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_summary.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/feature_summary_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/label_distribution_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/split_summary_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/target_diagnostics_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07.csv` | True | 0.00 | yes | small CSV |
| audit | `outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v3_fixed_window_features/audit_summary.json` | True | 0.01 | yes | small JSON |
| audit | `outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v3_fixed_window_features/final_audit_conclusion.csv` | True | 0.00 | yes | small CSV |
| plot | `outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_target_drift.png` | True | 0.06 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_horizon_performance.png` | True | 0.08 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_thresholded_vs_argmax_recall.png` | True | 0.06 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_nonzero_performance.png` | True | 0.05 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_performance.png` | True | 0.10 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v3_fixed_window_features_BTCUSDT_2024-03-01_to_2024-03-07_manifest.csv` | True | 0.00 | yes | small CSV |

## Artifact Policy

- Raw, interim, processed, and row-level parquet files are excluded.
- Included artifacts are code, documentation, small CSV/JSON summaries, and selected plots.
- API keys, environment-specific files, caches, and logs are excluded.
- A 14-day replication should use a new artifact tag.
