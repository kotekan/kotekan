# Masked visibility accumulation

`N2Accumulate` normalizes correlator sums by the number of accepted voltage
samples. With `variance_mode: EvenOddPosDef`, it estimates weights from
adjacent even/odd pairs.

For each product, let `N` be the accepted sample count and `k` the number of
accepted pairs with positive counts in both frames:

```text
Q = sum_pairs [n0*n1/(n0+n1) * abs(corr1/n1 - corr0/n0)^2]
V = sum_accepted corr / N
weight = N*k/Q
```

Rejecting either frame with the second-stage mask drops the whole pair. If
only one frame has samples, it contributes to `V` and `N`, but not `Q` or `k`.
The weight is zero if `N`, `k` or `Q` is zero, or if `Q` or the weight is
nonfinite. A valid mean can therefore have zero weight. The output conjugates
the lower-triangular input into upper-triangular order.

`Q/(k*N)` estimates the variance of the mean when sample errors are independent,
have a common variance, and both frames have the same expected normalized
visibility after fringestopping. Masking based on the data can break these
assumptions. Taking its reciprocal does not give an unbiased estimate of inverse
variance.

## Tests

Run the CPU stage tests against the selected build:

```sh
KOTEKAN_BUILD_DIRNAME=build OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m pytest -q tests/test_n2_accumulate.py \
  tests/test_n2_accumulate_mask_normalization.py tests/test_rfi_masksum.py
```

The tests compare saved output with independently calculated counts, means and
weights. Cases cover masked pairs, empty frames, unequal counts between frames,
large counts and pairs that span input frames.
