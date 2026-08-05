# Self-reference traps in per-record phase estimators

*Written 2026-08-05, after three of these in one day on the CHORD GNSS deep fold. Every one was
unbiased in the MEAN and fatal to the VARIANCE, and none was visible from the resulting number.*

## The setting

A deep fold coherently sums a per-record complex amplitude over N records. If the records carry
a common unknown per-record phase (on CHORD: ~0.75 rad of propagation phase, common across
elements and nodes, near-white in time), the sum decoheres and the fold saturates. The fix is to
estimate that phase per record and derotate — and *how you estimate it* is where the traps are.

The significance is computed by `gnss::coherent_sum`: rotate the total sum onto the real axis and
take the **component orthogonal to it** as the noise estimate.

> **That is the vulnerability.** Any derotation that systematically pushes records toward a common
> direction shrinks the orthogonal component, and the SNR inflates — even though nothing about
> the mean is wrong.

## THE RULE

> **Whatever you integrate must never appear in the phase reference you derotate it by** — not
> partially, and not through a symmetrization.

## Trap 1 — leave-one-out over the things being summed

For elements `c_e`, `tot = Σ c_e`, derotate each by the phase of the rest:

```
v = Σ_e c_e · e^{−i·arg(tot − c_e)}
```

`c_e` and `rest_e` are independent, so `E[v] = 0` under noise: **unbiased**. But expand for
`|c_e| ≪ |tot|`, using `arg(tot − c_e) ≈ arg(tot) − Im(c_e/tot)`:

```
v ≈ |tot|  +  i·(second-order term)
```

Every element is rotated by essentially the *same* angle `arg(tot)`, and rotating `tot` by
`−arg(tot)` just takes its magnitude. The correction **reconstitutes the self-derotation it was
meant to prevent**. Then:

- numerator → `Σ_k |tot_k|`, a sum of positive numbers growing ∝ N even for pure noise;
- denominator → only the second-order remainder, suppressed by ~1/N.

**Excluding one of N removes only 1/N of the self-reference.** More elements makes it *worse*
(`rest_e → tot`). Measured on synthetic data: **47.3 against a genie of 17.7** — it beat perfect
knowledge, which is the tell.

## Trap 2 — symmetrizing a split

Two disjoint halves S and R, each derotated by the other:

```
S·e^{−i arg R} + R·e^{−i arg S} = (|S|+|R|)·cos Δ + i·(|S|−|R|)·sin Δ,   Δ = arg S − arg R
```

Balanced halves (`|S| ≈ |R|`) **annihilate the imaginary part by construction**. Measured live:
`deep_snr` 38526 at `coh_frac` exactly 1.000.

## Trap 3 — forgetting constant offsets

Instances/elements carry arbitrary constant phase offsets (different combs, different NCO
history). Summing them unaligned produces noise, not a reference — an unaligned "fleet average"
read as a 13 rad phase excursion of pure garbage, and briefly supported a wrong conclusion.
Align first: `rot_i = e^{−i·arg(⟨A_i · conj(A_ref)⟩)}`.

## What survives

**One-way split.** Partition into S (integrated) and R (reference only):

```
v = S · e^{−i·arg(R)}   →   |S|·e^{i(arg S − arg R)}
```

R is independent of S, so `arg S − arg R` stays random under noise, the orthogonal component
survives, and the statistic keeps its meaning. Price: half the aperture measures while half
integrates — a factor √2, which is the honest cost of not knowing the phase.

Full-aperture alternative with *no* phase estimate at all (hence no self-reference): the
noise-debiased cross-power `|tot|² − Σ_e|c_e|² = Σ_{e≠f} c_e c_f*`. It cancels the common phase
identically, but integrates incoherently in time (√N instead of N), so for deep folds the
one-way split usually still wins.

## Why the same estimator is SAFE in time

`gnss::phase_track_loo` derotates each record by its temporal neighbours and passes its noise
control cleanly. The difference: each record's reference is a **different local window**, so
there is no single direction for everything to collapse onto. The collapse needs *both*
conditions — all terms sharing one reference, **and** that reference being the sum being rotated.

Same estimator name, opposite behaviour. Mis-transferring this intuition from the temporal case
to the element case is what produced Trap 1.

## Validation protocol — run BOTH

1. **Genie**: derotate by the *true* injected phase. Your estimator must never beat it.
2. **Shuffled null**: shuffle records independently per element/instance — real amplitudes,
   common phase destroyed. Your estimator must collapse to it.

Shuffle, **not roll**: a roll shifts a shared *linear* phase ramp by a constant and leaves it
fully intact, so a rolled null measures the rate-search floor (18–25 on CHORD) rather than the
estimator's own (~4). The first null I built was a roll and would have shipped a floor 5× too
high.

Both tests are cheap. Neither failure mode above is detectable without them.
