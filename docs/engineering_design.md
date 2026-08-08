# Engineering Design and Numerical Safeguards

This note describes the production path for the static microstructure study. It
focuses on reproducibility, numerical equivalence, bounded memory, and failure
behavior.

## Canonical Protocol

`src/protocol.py` defines the symbol, ternary class order, horizons, feature
order, split fractions, boundary drop, and trade lookback. The canonical JSON
representation is hashed and stored with each fitted model. Downstream scripts
reject an artifact if its protocol fingerprint, dataset hash, feature order,
class order, model name, horizon, or convergence status is incompatible.

## Frozen Models

Logistic models are serialized without pickle. Each artifact contains:

- ordered features and classes;
- train-fitted `StandardScaler` mean and scale;
- standardized coefficients and intercepts;
- optimizer configuration, iteration counts, and convergence status;
- sklearn version, creation timestamp, dataset SHA-256, and protocol hash.
- Git commit and a content hash of the exact source tree used for fitting.

An exact parameter fingerprint hashes the ordered scaler state, class order,
coefficients, intercepts, feature order, and experiment identity. Validation
threshold files carry this fingerprint; final-test and regime scripts reject
thresholds selected from any other fitted model.

The metadata also stores the SHA-256 checksum of the NumPy payload. This makes an
interrupted or mixed two-file replacement detectable at load time.

Before writing an artifact, predictions from the serialized representation must
match the fitted sklearn estimator after applying the stored train-only scaler on
a fixed sample, with maximum absolute probability error no greater than `1e-12`.
Probability diagnostics, final-test evaluation, regime analysis, and compact
prediction export all consume this artifact. They do not refit a model or
reconstruct scaler statistics.

## Bounded-Memory Data Path

The high-volume path operates on parquet row groups or exact event ranges:

- quality checks carry prior timestamps across batches so boundary gaps count;
- event construction carries the previous valid quote state across row groups;
- feature construction reads 20 historical and 50 future context events around
  each emitted chunk;
- fixed 20-return sums make realized volatility independent of chunk boundaries;
- split construction uses one validation pass and one write pass;
- model readers prune columns and load only requested chronological partitions;
- static logistic training fits the train-only scaler in ordered batches,
  writes one temporary standardized float64 matrix and aligned label matrix to
  disk, and reuses them across horizons;
- validation and test metrics accumulate exact confusion matrices by batch;
- exact AUC/AP, lift, and quantiles use disk-backed per-horizon probabilities;
- exact regime medians use train-only disk-backed values and deterministic ties;
- compact test predictions are written as bounded Parquet row groups;
- audit sampling draws from an integer range without allocating the population.

The temporary training matrix removes the full pandas table and repeated
train/validation concatenations from peak memory. The scikit-learn batch
optimizer still uses work arrays that grow with the number of training rows, so
model fitting is lower-memory rather than constant-memory. Run metadata records
measured peak RSS for capacity planning before longer replications.

High-volume parquet transformations and compact prediction exports use temporary
files and atomic replacement after all checks pass. Those transformations require
an explicit `--overwrite` before replacing existing outputs.

## Checkpointing and Execution Records

The logistic training stages and probability diagnostics write one checksummed
checkpoint per completed horizon. A checkpoint contains result
tables, the frozen model parameter fingerprint, and a strict run identity.
`--resume` accepts it only when model configuration, feature order, protocol,
dataset, source-tree content, and model parameters all match.

Canonical commands may add `--require-clean-git`. Each high-volume stage records
the Git commit, dirty/clean status, a content hash covering tracked and untracked
source files, complete command arguments, UTC timestamps, elapsed seconds, peak
RSS, and available dataset/protocol fingerprints under
`outputs/reports/run_metadata/`.

## Exactness Tests

Synthetic multi-row-group tests compare streaming and in-memory implementations:

- event rows and duplicate-collapse summaries match exactly;
- feature rows, labels, dtypes, and float64 return features match exactly;
- partitioned splits match the original timestamp-based split logic exactly;
- train and validation each contain exactly 50 boundary-drop events;
- trade windows are open on the left and exclude equal-timestamp trades;
- optimized threshold metrics match brute-force sklearn metrics to `1e-14`;
- frozen model probabilities match sklearn before and after serialization.
- streaming pooled, nonzero, and daily metrics match batch sklearn metrics;
- checkpoint resume rejects changed source, configuration, data, or checksums;
- disk-backed regime medians match exact in-memory medians, including ties.

CI compiles the project, runs Ruff, executes the invariant suite, and enforces
at least 75% coverage across the critical optimized and operational modules.

## Performance Decisions

The threshold search combines precomputed down/up confusion contributions. For
the 15-by-15 grid, row-level passes fall from 225 to 30 without approximating
metrics.

A controlled local 500,000-row benchmark compared Pandas and Polars for split
assignment plus grouped regime aggregation. In one three-repeat run, median times
were 0.0963 seconds for Pandas and 0.0116 seconds for Polars, an 8.33x speedup on
that synthetic workload. The dominant feature path is already NumPy/PyArrow and
has stricter event-window semantics, so Polars has not replaced it. The benchmark
is available at `benchmarks/benchmark_dataframe_engines.py`; a production
migration requires a material gain on the actual bottleneck, not this proxy alone.

## Artifact Status

The canonical seven-day checkpoint is `v3_fixed_window_features`. Generation
provenance records the clean source revision and source-tree SHA-256 in local run
metadata and frozen model artifacts. Pooled final-test tables,
regime-conditioned tables, and compact predictions all consume the same complete
frozen model artifacts.

The integration verifier independently reconstructs pooled metrics from
confusion counts, recombines regime slices to pooled accuracy, checks compact
prediction confusions, and requires the critical research audit to pass. For the
canonical run, both metric reconstruction and regime recombination have maximum
absolute error `1.11e-16`; compact prediction confusions match exactly.

Large processed datasets, frozen model files, checkpoints, run metadata, logs,
and row-level predictions remain local. The public evidence set contains only
code, documentation, selected plots, and compact CSV/JSON summaries. The main
remaining empirical limitation is temporal: the canonical result covers seven
days and requires confirmation on a fresh, longer sample.
