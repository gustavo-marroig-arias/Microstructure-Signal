# One-Page Research Memo: Top-of-Book Microstructure Signals

Project: BTCUSDT short-horizon top-of-book signal study  
Status: frozen 7-day `v3_fixed_window_features` checkpoint

## Question

Can a small, interpretable set of top-of-book state and recent trade-flow variables predict short-horizon event-time midprice direction for BTCUSDT under strict chronological evaluation?

Target:

```text
y_h = sign(midprice[t + h] - midprice[t]), h in {10, 20, 50}
```

The task is ternary classification: down, unchanged, up. The project is an empirical signal test, not a tradable strategy or PnL backtest.

## Design

Data uses Binance USDT-M perpetual `BTCUSDT` top-of-book quotes and aggregate trades. The observation is a distinct top-of-book state change; consecutive duplicate quote states are collapsed. The current run covers `2024-03-01` to `2024-03-07` UTC, with 223,033,235 distinct quote-state events, 223,033,185 feature/label rows after the terminal horizon drop, and 28.7 million model-eligible final-test rows.

Features are deliberately narrow: relative spread, queue imbalance, log bid/ask size, bid/ask size changes, 5-event log midprice return, 20-event realized volatility proxy, 1-second trade intensity, signed trade count imbalance, and signed trade volume imbalance. The two log-return-derived features, `mid_return_5` and `realized_vol_20`, are computed and stored as `float64`.

Evaluation uses a timestamp-based chronological split: 60% train, 20% validation, 20% test. Scalers are fit on train only. The final 50 labeled observations of train and validation are excluded to prevent target spillover. Trade-flow features use only trades in `(quote_timestamp - 1000 ms, quote_timestamp)`, excluding trades exactly equal to the quote timestamp.

Models: majority-class baseline, queue-imbalance-only multinomial logistic regression, full-feature multinomial logistic regression with L2 regularization, and a validation-selected thresholded full-logistic operating point.

## Main Result

The full-feature logistic model beats both baselines on final-test macro F1 and balanced accuracy at all three horizons.

| Horizon | Majority Macro F1 | Full Macro F1 | Majority Bal. Acc. | Full Bal. Acc. |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.296 | 0.454 | 0.333 | 0.423 |
| 20 | 0.271 | 0.434 | 0.333 | 0.423 |
| 50 | 0.213 | 0.443 | 0.333 | 0.456 |

The validation-selected thresholded rule improves directional recall and balanced accuracy further:

| Horizon | Thresholded Macro F1 | Thresholded Bal. Acc. | Non-zero Bal. Acc. |
| ---: | ---: | ---: | ---: |
| 10 | 0.584 | 0.600 | 0.477 |
| 20 | 0.628 | 0.648 | 0.593 |
| 50 | 0.660 | 0.664 | 0.681 |

Interpretation: the full feature set contains directional probability-ranking information. The default argmax rule is conservative under class imbalance and often predicts `unchanged`; thresholding shifts the operating point toward recovering more down/up moves. Thresholds were selected from a fixed 15-by-15 grid on one validation segment, so this is secondary evidence with more selection uncertainty than the primary argmax result.

## Audit

The audit found no evidence of label alignment leakage, split-boundary leakage, quote-feature lookahead, or equal-timestamp trade leakage. The precision audit passes: `mid_return_5_all_match_float64=True`, `realized_vol_20_all_match_float64=True`, and `log_feature_precision_status=pass`.

Other findings: validation and test are materially more directional than train; selected standardized up-versus-down coefficient contrasts have microstructure-consistent signs, but they are associational rather than causal; many non-zero labels correspond to moves larger than the current spread. Regime cutoffs are fixed from training medians and applied unchanged. At horizon 50, thresholded balanced accuracy is higher in tight-spread than wide-spread events (`0.651` versus `0.568`) and in positive- versus zero-realized-volatility events (`0.664` versus `0.597`). The realized-volatility cutoff is zero, so this is a zero-versus-positive diagnostic rather than a balanced volatility split.

Supporting artifacts include compact probability/prediction summary diagnostics, selected summary plots, a pinned runtime environment, automated invariant tests, and a current result manifest. Independent integration verification reconstructs final metrics and regime aggregation to `1.11e-16` and matches compact-prediction confusion counts exactly.

## Claim and Limits

Allowed claim: in this frozen 7-day BTCUSDT sample, simple top-of-book state and recent trade-flow features show out-of-sample statistical signal for short-horizon event-time midprice direction relative to majority and queue-imbalance-only baselines.

Not allowed claim: tradable alpha. This project does not model fees, latency, queue position, fill probability, slippage, market impact, adverse selection, or executable PnL.

Main limitations: the sample is 7 days rather than the target 14 or 30 clean days; final-test evidence spans about 33.6 hours over two UTC dates; adjacent event-time labels overlap and observations are serially dependent, so 28.7 million rows are not 28.7 million independent observations; no iid standard errors or significance claims are reported; the study covers one instrument and one venue; there is no execution-aware backtest; the validation-selected threshold rule and all regime findings need confirmation on a longer fresh sample.

## Next Step

Future work is to test delayed-feedback online adaptation, calibration, abstention, and cost sensitivity under the same chronological protocol, then run a fresh 14-day clean-sample comparison after the extension is frozen.
