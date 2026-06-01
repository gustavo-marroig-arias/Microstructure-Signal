# Current Result Manifest

- symbol: `BTCUSDT`
- sample: `2024-03-01` to `2024-03-07` UTC
- artifact tag: `v2_float64_features`
- status: frozen 7-day checkpoint

## Audit Status

- `critical_audit_passed`: `True`
- `log_feature_precision_status`: `pass`
- `mid_return_5_all_match_float64`: `True`
- `realized_vol_20_all_match_float64`: `True`
- `main_result_status`: `frozen_test_result_supported_by_audit`

## File Manifest

| Category | Path | Exists | Size MB | Included? | Note |
| --- | --- | ---: | ---: | --- | --- |
| data | `data/processed/feature_table_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet` | True | 3852.42 | no | excluded large artifact |
| data | `data/processed/model_dataset_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet` | True | 3856.71 | no | excluded large artifact |
| small_result | `outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_aggregate.csv` | True | 0.00 | yes | small CSV |
| small_result | `outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_nonzero_subset.csv` | True | 0.00 | yes | small CSV |
| small_result | `outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_daily_blocks.csv` | True | 0.01 | yes | small CSV |
| small_result | `outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_aggregate.csv` | True | 0.04 | yes | small CSV |
| small_result | `outputs/results/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_nonzero_subset.csv` | True | 0.03 | yes | small CSV |
| small_report | `outputs/reports/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_frozen_thresholds.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_full_vs_baselines_deltas.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_thresholded_deltas.csv` | True | 0.00 | yes | small CSV |
| small_report | `outputs/reports/final_test_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_thresholds.csv` | True | 0.00 | yes | small CSV |
| diagnostic_data | `outputs/results/compact_test_predictions_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07.parquet` | True | 1154.71 | no | excluded row-level parquet |
| small_report | `outputs/reports/compact_test_predictions_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_summary.csv` | True | 0.00 | yes | small CSV |
| audit | `outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v2_float64_features/audit_summary.json` | True | 0.01 | yes | small JSON |
| audit | `outputs/reports/research_audit/BTCUSDT_2024-03-01_to_2024-03-07_v2_float64_features/final_audit_conclusion.csv` | True | 0.00 | yes | small CSV |
| plot | `outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_target_drift.png` | True | 0.06 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_horizon_performance.png` | True | 0.07 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_thresholded_vs_argmax_recall.png` | True | 0.06 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_regime_performance.png` | True | 0.09 | yes | small PNG |
| plot | `outputs/reports/plots/plot_v2_float64_features_BTCUSDT_2024-03-01_to_2024-03-07_nonzero_performance.png` | True | 0.05 | yes | small PNG |

## Artifact Policy

- Raw, interim, processed, and row-level parquet files are excluded.
- Included artifacts are code, documentation, small CSV/JSON summaries, and selected plots.
- API keys, environment-specific files, caches, and logs are excluded.
- A 14-day replication should use a new artifact tag.
