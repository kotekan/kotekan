# T5: Gains

T5 checks that the configured gains act as configured: `upchan_gain` per fine channel,
`bb_gain`, and `frb1_input_scale`. Two runs see the same input, one with the production gains
and one with scaled gains, and the check compares their outputs.

## Errors this test catches

- **Upchannelizer gain on the wrong fine channel**: gains permuted, mirrored, shifted by one,
  or taken in bit-reversed order; one gain for all fine channels; the gain of another U.
- **Gain applied wrongly**: ignored, applied twice, or as a power (G² or √G).
- **BB gain split wrongly** into the shift `s` and the phase amplitude, e.g. `s` off by one.
- **FRB1 weights**: `frb1_input_scale` not reaching W, or applied after squaring, so that the
  intensity scales by 2 instead of 4. FRB1 reading another fine channel than the upchannelizer
  wrote, since its ratio must follow that channel's gain.
- **Config wiring**: a gain variable that does not reach its stage.

## Runs

Both runs start at FPGA sample `2^33 + 7·T` and use the same noise seed. The input is complex
Gaussian noise with σ = 1.2 LSB per component plus a weak tone (amplitude 0.1) from
θ = (0.5°, 0.8°). One frame of each output is written.

- **`reference`**: the production gains, `upchan_gain` = 0.8·√U, `bb_gain` = 1/√D = 1/8,
  `frb1_input_scale` = 1.
- **`scaled`**: fine channel u gets `upchan_gain` × f_u with f_u = 1 + (k·u mod U)/U and
  k = max(U/2 − 1, 1). The factors are distinct, lie in [1, 2), and differ by at least 14%
  between neighbouring channels. `bb_gain` and `frb1_input_scale` are doubled.

σ = 1.2 is half the production level so that the doubled BB gain does not saturate. The
rounded input has rms 1.23. The upchannelizer noise is then 0.88 before rounding in the
reference run (int4 rms 0.93) and up to 1.75 in the scaled run. The BB noise is 1.23 and 2.46,
with 0.83% at ±7 in the scaled run.

## Predictions

**Rounding.** An int4 output is `q = clamp(round(x), −7, 7)`. For a smooth distribution of
width σ ≳ 0.5, rounding adds 1/12 to the variance (Sheppard's correction), so
rms² ≈ σ² + 1/12. When x scales by r, the raw rms ratio is therefore

    sqrt(r² σ² + 1/12) / sqrt(σ² + 1/12)  <  r,

which is 3.6% low for σ = 0.88 and r = 2. The check inverts the rounded-Gaussian model of T3
(rounding and clamping) for each run: σ̂ is the σ whose model rms equals the measured rms. Then
σ̂_scaled / σ̂_reference must equal r. The tone does not matter here: the noise makes the
distribution smooth, and the correction does not depend on its shape. Clamping is negligible
except in the scaled BB run, where the model includes it.

**Upchannelizer.** It computes

    Ē_u = (G_u/U) Σ_s W(s) e^{−2πi (u − (U−1)/2) s/U} E(w + s),

which is linear in G_u. G_u is stored as float16, so the expected ratio is
`r_u = fp16(f_u G_u) / fp16(G_u)`. The check measures σ̂ per U and fine channel u, over all
coarse channels, times, polarizations and dishes. Fine channels are identified by
`freq_upchan_index`.

**Baseband beams.** `calcBBPhase` realizes `bb_gain = 2^7 a / 2^s` with a in (1/2, 1] and phase
amplitude |A| = 127.5 a. The beamformer computes

    J = clamp(shift(shift(Σ_d A_d E_d, σ), s − σ), −7, 7),   shift(x, n) = (x + 2^(n−1)) >> n.

Doubling `bb_gain` keeps a, and hence A, and lowers s by one. The value before the last rounding
doubles exactly, so r = 2. The check measures σ̂ over all beams.

**FRB1.** `setFRB1Phase` sets W = `frb1_input_scale` for every dish. The kernel multiplies W by
`input_gain` = 1/(2√(MN)), forms beams on the 2M × 2N grid by FFT, squares and averages over
time. Summed over all beams, Parseval gives

    Σ_beams I = c · W² · Σ_{p, dishes, t} |Ē|²

with a constant c. Per fine channel, the intensity ratio is therefore 4 × P_s/P_r, where P is
the power Σ|Ē|² of the written int4 upchannelized voltages. Using the measured P makes
rounding exact. For U = 1, FRB1 reads the coarse voltages, which are equal in both runs, so the
ratio is 4. Scaling float16 values by 2 is exact.

The FRB1 frame covers 49152 samples, but the upchannelized frame only 8192. The power ratio
depends on the time window only through the rounding noise; we estimate this at below
10^−4.

## Tolerances

- **int4 outputs, 3%.** The model, statistics and float16 arithmetic in the upchannelizer
  contribute about 0.2%. The errors above change a ratio by at least 14% (a neighbouring
  channel) or 41% (a factor √2). The raw rms ratio, without the rounding correction, deviates
  by up to 3.7% and would fail.
- **FRB1, 1%.** The float16 accumulation in the kernel contributes about 0.2%. A linear W
  would be off by 100%.

## Mutation check

The check code alone was changed and the same output re-checked with `--check-only`:

- **Gains expected on the neighbouring fine channel** (`np.roll` by one): the ratios deviate by
  −43% to +50%; all six upchannelizer outputs fail.
- **FRB1 linear in W** (expected ratio 2 instead of 4): +100%; FRB1 fails for every U.
- **No rounding correction** (raw rms ratio): the upchannelizer fails for U ≥ 4 (−3.3% to
  −3.7%). U = 2 (−2.5%) and BB (−2.35%) stay within 3%.

## Result (pathfinder, cx67, Release)

**PASS**, with the production gains of #1784 as the reference.

| output     | expected ratio      | measured / expected − 1 |
|------------|---------------------|-------------------------|
| U=2        | 1.000 to 1.500      | 0 to +0.0023            |
| U=4        | 1.000 to 1.751      | −0.0003 to 0            |
| U=8        | 1.000 to 1.874      | −0.0003 to 0            |
| U=16       | 1.000 to 1.938      | −0.0006 to +0.0003      |
| U=32       | 1.000 to 1.967      | −0.0007 to +0.0004      |
| U=64       | 1.000 to 1.985      | −0.0016 to +0.0012      |
| BB         | 2                   | −6·10^−6 (σ̂ 1.232 → 2.464) |
| FRB1 U=1   | 4                   | 0                       |
| FRB1 U=2…64 | 4 × P_s/P_r, up to 14.6 | −0.0014 to +0.0016  |

- The scaled run has at most 0.02% of the upchannelized values at ±7 and no -8.
- Without the rounding correction, the upchannelizer ratios would be 2.5% (U = 2) to 3.7%
  (U = 64) low.

Kotekan runs for 22 s per run and the check for about 10 s. The output is 1.7 GB on disk
(0.85 GB per run, of which FRB1 is 0.5 GB).
