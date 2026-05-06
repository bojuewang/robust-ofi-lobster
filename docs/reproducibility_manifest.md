# Reproducibility Manifest

## Data

The project expects five LOBSTER sample zip files under `data/raw/`:

- `LOBSTER_SampleFile_AMZN_2012-06-21_10.zip`
- `LOBSTER_SampleFile_AAPL_2012-06-21_10.zip`
- `LOBSTER_SampleFile_GOOG_2012-06-21_10.zip`
- `LOBSTER_SampleFile_INTC_2012-06-21_10.zip`
- `LOBSTER_SampleFile_MSFT_2012-06-21_10.zip`

These files are not committed.

## Determinism

The scripts use fixed seeds where stochastic training is used. Numerical results may vary slightly across BLAS/LAPACK implementations, but benchmark gates are designed to tolerate harmless floating-point differences.

## Benchmark gates

Every module writes a markdown report under `reports/benchmarks/` and a PASS marker under `.pipeline_state/`.
