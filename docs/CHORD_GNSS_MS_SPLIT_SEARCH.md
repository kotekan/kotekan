# The ms-split search — design plan (2026-08-02)

## 1. Why

The acquire integrates coherently over **one replica period**, `Mp = code_samples /
gcd(fft_len, code_samples)` hops. Nobody chose that as a window length; it is what the cyclic
correlation needs in order to be valid, and on airspy it happened to equal exactly one code
period:

| | airspy | CHORD |
|---|---|---|
| code period | 5000 samples | 3,200,000 samples |
| `fft_len` | 20 | 16384 |
| hops per code period | **250 (integer)** | **195.3125** |
| `Mp` | 250 hops = **1 ms** | 3125 hops = **16 ms** |

`fft_len` divides `code_samples` on airspy and does not on CHORD, so the same line of code
gives a window 16× longer than the lag search actually needs — and that one number then drags
two more axes with it:

* **coarse lag 3125 instead of ~196.** The lag axis is that long *because* the NH20 overlay is
  applied across the 16 ms window, which makes the replica non-periodic over the window so all
  3125 lags are genuinely distinct. For the bare code, lag is ambiguous modulo one code period
  — 196 hops covers the whole space. (This is also exactly why the coarse-lag fold in
  CHORD_GNSS_STATE.md §6.3 measured wrong: it assumed a periodicity the overlay destroys.)
* **Doppler 13 bins instead of 1.** A window of length T resolves Doppler to 1/T. 16 ms → 62.5
  Hz, so the grid is 31.25 Hz and the hinted ±200 Hz needs 13 bins. 1 ms → 1000 Hz, and ±200 Hz
  is a single bin.
* **20 NH alignments instead of 1.** 16 ms straddles 16 of the 20 overlay chips at an unknown
  starting position, so all 20 are searched. 1 ms spans exactly one overlay chip: a constant
  ±1 on the whole window, invisible to |D|².

Measured cost today, per PRN per pass: **2.81 G cross-channel ops** (13 × 3125 × 128 × 27 × 20).

## 2. The geometry, stated exactly

```
code period      3,200,000 samples  =  195.3125 hops  =  195 hops + 5120 samples
slip per period  0.3125 hop = 5120 samples = 16.36 chips
16 code periods  51,200,000 samples =  3125 hops exactly      <- why Mp = 3125
```

The **sample** grid is already aligned (3.2 GHz × 1 ms is an integer). Only the **hop** grid is
not, and it cannot be re-referenced: the F-engine applied the PFB in hardware, so by the time we
see the data the hop boundaries are fixed. There is no "pad by a fractional sample" available —
the slip is 5120 whole samples per ms, and it is a *sub-hop* offset, which is precisely the
quantity the fine-lag axis already represents.

## 3. Algorithm

### Phase A — incoherent (do this first)

Split the snapshot into `K` sub-windows of `N ≈ 196` hops (one code period of lag ambiguity).

1. **Per sub-window k**, build the replica at that sub-window's own start phase and correlate.
   Use a **zero-padded (linear) correlation**, not cyclic: over 195 hops the replica advances
   0.9984 code periods, so a cyclic wrap is 16.4 chips wrong. FFT length 512 (≥ 2N) removes the
   artefact for ~2× the transform cost, which is not where the money is.
2. **Cross-channel combine** to recover fine lag, exactly as now:
   `D(s) = Σ_c P_c e^{i2πf_c s/sph}`.
3. **Roll each sub-surface** by the known slip before summing. Sub-window k starts at hop `kN`,
   whose code phase differs from sub-window 0's by a deterministic amount (nominal advance +
   code Doppler). In the stored lag axis this is an integer column shift — cheap.
4. **Sum |D|²** across the K sub-windows.

### Phase B — coherent recombination (only if Phase A's sensitivity is not enough)

Keep the sub-surfaces **complex** instead of squaring, then combine afterwards:

```
for each NH hypothesis a in 0..19:
    for each fine-Doppler bin δf:
        S_a,δf(cell) = Σ_k  nh_sign[(k + a) mod 20] · e^{i2π δf k T_sub} · D_k(cell)
```

This recovers full 20 ms coherent gain. It is **exact** — the sub-correlations are genuinely
independent complex quantities and are combined by addition. It is NOT the NH restructure that
failed in §6.3: that one tried to synthesise 20 shifted copies from a *single* correlation via
the cross-channel reconstruction, which is only approximate for a windowed PFB.

The cost is that coherent recombination needs Doppler good enough not to smear across 20 ms
(δf ≲ 8 Hz), so the fine Doppler axis returns — in the combine stage, where it is a phase
rotation on precomputed values, not a re-correlation.

## 4. Cost and sensitivity

Per PRN per pass, cross-channel ops (the dominant term):

| | Doppler | lags | fine | chan | align | sub-win | ops | vs now |
|---|---|---|---|---|---|---|---|---|
| now | 13 | 3125 | 128 | 27 | 20 | 1 | 2.81 G | — |
| Phase A | 1 | 196 | 128 | 27 | 1 | 100 | 67.7 M | **41×** cheaper |
| Phase B | 1 | 196 | 128 | 27 | 1 | 100 | 67.7 M + combine | ~4× cheaper |

Sensitivity, SNR ∝ T_coh·√N_incoh over a 100 ms snapshot:

| | coherent | incoherent | SNR figure |
|---|---|---|---|
| now | 16 ms | ×√6 | 39.2 |
| Phase A | 1 ms | ×√100 | 10.0 — **3.9× worse (≈6 dB)** |
| Phase B | 20 ms | ×√5 | 44.7 — slightly *better* than now |

Phase A's 6 dB matters: the current noise ceiling is 19.1 with `acquire_snr` 30, and ~51% of
live detections already sit in that margin. Phase A alone would make that worse. **It should be
deployed together with, or after, the window/overlay bookkeeping fix** (route window w into bin
`(a + 4w) mod 20`), which restores multi-window integration and brings the ceiling to 4.45.

## 5. Validation — bench before sky, no exceptions

`scripts/gnss/e2e` already injects a known code phase and reports the recovered phase and
period against ground truth. The ms-split path must be validated the same way, and the specific
things to test are the ones that have gone wrong before:

1. **Recovered cp and period vs injection**, across a cp sweep and a hop0 sweep — same
   acceptance bar as the current path (period 16/16, residual ≲ 0.2 chips).
2. **The sub-window roll (step 3)** — this is the exact bookkeeping that `acquire_windows` got
   wrong, where each window landed in a different NH bin and a second window *halved* SNR. Test
   it by summing K sub-windows on a noiseless injection and confirming the peak grows as K, not
   as √K and not at all.
3. **The zero-padding** — confirm a cyclic (unpadded) implementation reproduces the predicted
   16.4-chip wrap error, so we know the padding is doing what it claims.
4. **Measured SNR vs the analytic table above**, on identical injected data, both paths.

## 6. Risks

* **The 6 dB.** Phase A is a straight sensitivity loss. If the overlay bookkeeping fix does not
  land first, this trades a cost problem for a detection problem.
* **Phase B's Doppler coupling.** 20 ms coherent needs δf ≲ 8 Hz; the acquire's own
  interpolation error is a −2.0…+2.1 Hz sawtooth, so this is feasible but not with margin to
  spare. Measure before relying on it.
* **The lag ambiguity is only mod one code period.** The overlay period must then come from the
  NH combine (Phase B) or from continuity/GPS-time assist. Phase A on its own reports a phase
  mod 10230, which is *less* than the search reports today (mod 204600) — a regression in what
  the seed can carry, unless Phase B lands with it.
* It does not address the ±60 chip per-detection seed scatter, which is a separate open item.

## 7. Not yet answered

Whether the 100 sub-correlations can share work. Sub-window k's replica differs from k−1's only
by a known phase advance, so the per-channel filter (the slow half of the hop-rate generator)
should hoist across all K — the same trick the refine already uses. If it does, Phase A's
correlation cost drops further and Phase B becomes clearly the better deal.
