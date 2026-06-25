# GNSS Satellites as Calibrators on CHORD — Framework Overview

Open-code GNSS satellites are bright, precisely-known, moving point sources. By
despreading them we can measure each antenna's complex response toward a known
sky direction with the signal's full code processing gain, then use those gains
to **calibrate** the array and to **remove** the satellites from the data. This
note sketches how that maps onto CHORD's F-X system. It builds on the
channelized GNSS pipeline in `lib/stages/Gnss*` / `gnss*` and
`python/scripts/gps_broker.py`.

## In-band signals

CHORD covers ~300–1500 MHz, so the usable GNSS carriers are:

| Band | Carrier | Signals |
|------|---------|---------|
| L5/E5a | 1176.45 MHz | GPS L5, Galileo E5a, BeiDou B2a, NavIC L5 (all CDMA, **same channels**) |
| L2 | 1227.6 MHz | GPS L2C |

GPS L1 (1575 MHz) is **above** the band and not available. The 1176 cluster
stacks several constellations into the *same* PFB channels.

## The corner-turn reality

After the F-engine, CHORD transposes so each GPU node holds **all antennas for a
comb of ~48 channels** spread across the band. A GNSS carrier is wide (a 2 MHz
code ≈ 10 channels; an L5 code ≈ 100), so its channels scatter across ~that many
nodes — at most one per node.

This is fine: the channelized despread is a **coherent sum of per-channel
correlations**, `G = Σ_c ⟨X_c, R_c⟩`, which doesn't care where the channels
live. Two options:

- **Remap** the ~5 MHz GNSS sub-band onto one node → run the pipeline locally,
  unchanged. Simplest.
- **Leave it combed** → each node correlates its own channels and a combiner
  network-reduces the (tiny) partial correlations.

Either way you ship *correlations*, never wideband voltage, and never reassemble
the timestream. The F-engine is one coherent system, so per-channel phases are
mutually consistent regardless of routing.

## Acquisition is single-antenna (≈ free)

Finding a satellite's code phase + Doppler is a 2-D search, but it runs on **one
antenna** — a single feed already has ample SNR thanks to the code gain (it's
what a phone does). Cost is ~10 Gflop/s per satellite during the brief lock
window, 1–2 satellites acquiring at once → a sliver of a GPU or a couple of CPU
cores. Only the *measurement* scales to all antennas; the *search* never does.

## Measurement folds into the N² correlator

CHORD's input path takes 64 per-frequency-custom synthetic streams alongside the
1024 sky inputs. Once locked, inject each satellite's channelized replica `R_c`
as a synthetic input. The N² kernel's cross-products then yield, for free:

- `V_{i,k} = Σ_m X_{c,i}[m] · conj(R_c[m])` = antenna *i*'s **complex gain** toward
  satellite *k* — for every antenna.
- `V_{k,k} = Σ |R_c|²` = the matched-filter **normalization**.

So the per-antenna gains are just the replica's column of the visibility matrix.
The FLOPs are identical to a standalone despread, but ride the optimized kernel.
Augmentation cost ≈ `2K/N` (a few % for K~30, N~1024), and lanes are reusable
across disjoint bands (watch the 1176 stack, which is not).

## Removing the satellites: peel + null

**Voltage peel (pre-vis, deep).** Subtract the satellite's waveform from each
antenna before correlation:

```
X_i'[m] = X_i[m] − â_i · exp(j·φ_geo,i(t)) · R[m]
```

The apparent causality problem — the peel needs gains that only exist *after*
correlation — dissolves once you split the gain:

- `φ_geo,i(t)` (geometric fringe, the fast ~0.5 rad/s part) is **deterministic**:
  predict it feed-forward from ephemeris + array geometry (~−40 to −50 dB), no
  measurement needed.
- `â_i` (per-antenna electronic gain) is **quasi-static**: a one-frame-old
  estimate is essentially exact.

After peeling, the synthetic-lane column reads out the **residual**,
`V'_{i,k} = (a_i − â_i)·⟨R,R⟩` — the gain *error* — which closes a cheap,
stable first-order tracker (`â_i ← â_i + μ·error`). The "loop back" is carrying a
tiny `N×K` table forward one frame, not re-running anything. Seed `â` from the
array's existing per-antenna calibration at rise.

**Spatial null (post-vis, robust backstop).** From the measured `{g_k}` build a
rank-K projector `V' = P_⊥ V P_⊥` (or null-constrained beam weights). It strips
all satellites at once, costs `O(N²K)` (not `N³`), and because the gains are
*measured* the null sits at the true response — deep, calibrated pointing. Since
the peel already removed the bulk, the residual subspace is tiny, so the null
costs near-zero spatial DOF.

**Peel for depth and DOF-efficiency; null for robustness against peel error.**

**Encrypted signals come (mostly) for free.** Restricted signals are *co-hosted*
on the same satellites as open ones, in the same band — GPS L2 carries L2C
(open) alongside P(Y) and M-code (encrypted); Galileo E6 carries an open
component alongside PRS. Because the spatial null is per-*direction*, nulling the
satellite from the co-hosted open signal's measured gains removes its **entire
in-band emission at once** — open and encrypted together — and it is a *measured*
(deep) null, not a nominal one: same direction ⇒ same geometric fringe, and same
band ⇒ same per-antenna electronic gain and same (dispersive) ionospheric delay,
so the open-signal gains transfer exactly. You only lose the **waveform peel** on
the encrypted part (no code to build a replica), so the recipe is: peel the open
component for depth, and let the direction-null sweep up the encrypted remainder.
(P(Y)/M are wider than L2C; the channels L2C doesn't illuminate are covered by
the array's standard bandpass × the predicted fringe. Inter-signal antenna
phase-centre offsets are a fixed cm-level bias, calibratable if deep nulls
demand it.)

## Timing: everything at the 30 ms native cadence

A satellite's fringe rotates ~0.5 rad/s on the longest baseline, so it smears
across directions within a 1 s integration and is no longer rank-1. **Remove it
at 30 ms, before downintegration**, then integrate the cleaned `V` down to 1 s /
10–30 s on the CPU. This lines up exactly with the gains, which emerge from the
30 ms correlator dumps. One measurement, three consumers at the native cadence:
the peel/null on `V`, the tracking loop, and the array gain calibration.

## Orchestration (control plane)

A slow external process (the `gps_broker.py` pattern) reads the almanac and:

- queues **rising** satellites into the single-antenna acquisition stage;
- hands each resulting (code phase, Doppler) to the inject/track stage;
- updates **which codes go into the synthetic lanes**;
- prunes **setting** satellites.

Restricted-code signals (e.g. GPS P(Y)/M) can't be waveform-peeled, but they
ride the *same* satellites as open ones, so the per-direction null built from the
co-hosted open signal removes them along with everything else that satellite
radiates in-band (see "Encrypted signals come (mostly) for free" above). A
satellite with *no* in-band open signal at all would fall back to a
nominal-direction null, but in practice every GNSS satellite broadcasts an open
in-band signal.

## Data flow

```
 F-engine channels ─┬─────────────► (sky voltages)
                    │
        ┌───────────┴───────────┐
        │  voltage peel:        │   â (tracked, slow)  +  φ_geo (predicted)
        │  X' = X − â·e^{jφ}·R  │◄──────────────┐
        └───────────┬───────────┘               │
                    │ + replica synthetic lanes  │
                    ▼                            │
            N² correlator (30 ms) ──► V'         │
                    │                            │
        ┌───────────┼─────────────┐              │
        │           │             │              │
   V'_{i,k}     rank-K null    cleaned V' ──► 1 s ─► 10–30 s (CPU)
  (residual →     P⊥V'P⊥
   tracker) ──────────────────────────────────────┘
```

---

*Status:* the channelized acquire → refine → measure pipeline and the broker
exist and are validated locally on a single airspy (`config/airspy_gps_channelized.yaml`).
The CHORD-specific pieces — synthetic-lane replica injection, the predictive
voltage peel, and the rank-K vis-matrix null — are the integration work this note
describes.
