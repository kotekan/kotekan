# GNSS Replica: Generate-and-Distribute vs Generate-Local

How the channelized satellite replicas `R_c` (and their nav-wiped form `d̂·R_c`)
get to the X-engine nodes that inject/peel them. Companion to
`gnss_chord_framework.md` (peel/null) and `gnss_pipeline_reference.md` (the stages).

## The object and two ways to deliver it

Each node holds ~1 L5 + ~1 L2 channel of the comb; the L5 channel carries **all
~M visible PRNs** (CDMA), so a node needs `R_c` for *every* PRN on its channel(s):
~M ≈ 30 streams, each one complex value per hop (195.3 kHz). Two architectures:

- **(A) Central + distribute.** One aggregator generates every PRN's covering
  channels (one `fft_len`-FFT per PRN gives all its covering bins at once) and ships
  each node its `(channel, PRN)` streams.
- **(B) Local.** Each node generates only its own channel(s)'s replicas. No
  distribution, but the per-node FLOPs depend on how cheaply `R_c` can be made — see
  `gnss_pipeline_reference.md` pathway #5 (hop-rate generation), under evaluation.

Two wins below are **shared by both** and are the important structural results.

## Win 1 — `R_c` is feed-forward: generate *ahead* of the sky

`R_c` is deterministic given the `(code-phase, Doppler)` trajectory, and that
trajectory is **predictable** from the lock + orbit + clock (the FLL gives the
carrier, the orbit the Doppler rate, the cp-slope the code rate; all smooth for
seconds–minutes, re-anchored from the live track). So the replica for time *t* is
generated at *t − Δ* and is in place **before the node processes that sky** — the
node buffers only the wire latency (~ms), never the generation. The only genuine
lag is **first acquisition at satellite rise** (~seconds to lock; that sat, low and
weak, is briefly un-peeled). This needs the aggregator to hold a **trajectory/clock
predictor** (the broker's state promoted from "re-seed every 0.2 s" to "predict the
next N s").

## Win 2 — the nav bits factor out: `d̂` is a last-moment post-multiply

`d̂` is constant over a 20 ms bit while the PFB's memory is only `P·fft_len` ≈ 20 µs,
so it passes straight through the linear channelizer:

```
d̂·R_c[m] = R_c[m] · d̂[bit(m)]     (error only in the ~20 µs straddling a bit edge, 0.1%)
```

So generation ships/uses the **bare `R_c`** (predictable, feed-forward, Win 1), and
`d̂` — the one unpredictable, data-dependent piece — is applied **last-moment at the
node** as a sign flip. `d̂` is decoded centrally from a strong reference and
**broadcast**: 50 bps × M ≈ **1.5 kbps**, negligible (or ephemeris-predicted, zero
latency). This decouples the unpredictable data from the predictable waveform — the
offline "wipe last" pattern, lifted into the distributed system.

## Architecture (A): central generation + distribute

```
 acquire ─► lock (cp,Doppler)/sat ─► AGGREGATOR (~1 node)
   trajectory predictor → bare R_c, all PRN×covering channels, FEED-FORWARD ─► nodes
   nav decode from strong ref → broadcast d̂ (~kbps)
 node:  d̂·R_c  ─►  inject into N² kernel / voltage peel   (cheap)
```

- **Aggregator compute:** ~3–5 Tflop/s (one node; the FFT amortizes a PRN's covering
  channels) — vs ~190 Tflop/s if 128 nodes each regenerate (the duplication this avoids).
- **Per-node reception:** ~M × 1–2 channels ≈ 30–40 streams → **~60 Mbps (4-bit) to
  ~0.5 Gbps (cfloat)**. Trivial.
- **Aggregator egress (the design constraint):** ~(covering channels) × M PRNs ≈
  105 × 30 for L5 → **~5 Gbps at 4-bit, ~40–50 Gbps at cfloat.** 4-bit is plenty for a
  replica (quantization ~25 dB down → peel residual ~45 dB under the noise); split
  across 2–3 aggregator nodes for headroom. Manageable on 100 GbE, but not free.

## Architecture (B): local generation — VALIDATED, the recommended path

A node makes its own `R_c` cheaply, so **the 5–50 Gbps distribution disappears** and
the central path shrinks to a **~kbps broadcast** of the per-PRN trajectory params +
nav bits. Full-rate `code×carrier×PFB` would be ~1.5 Tflop/s/node (too much ×128),
but the **hop-rate reformulation** wins: the code is constant over a chip, so the
polyphase filter collapses to a per-chip sum — precompute the channel filter
integrated over each chip vs sub-chip phase (a φ-bank, rebuilt slowly as the Doppler
drifts), then each hop is `~P·f_chip/channel_BW` MACs (a ~`Fs/f_chip` ≈ 313× cut →
**~0.01 Tflop/s/node**). Both carrier images kept; the carrier is a per-hop output
phasor; the nav wipe is the free per-hop sign post-multiply.

**Validated, and exact** (`python/scripts/gps_hoprate_validate.py`; C++
`ChannelizedReplicaBank::channels_hoprate()` + boost test `hoprate_matches_exact_pfb`):
the per-chip form reproduces the exact full-PFB to **machine precision** (−144 dB at
float32). The right form is the **prefix-sum**, not a φ-bank: a chip's filter
contribution is `Φ[k_hi]−Φ[k_lo−1]` over its *integer* tap range (chip edges fall
*between* samples, so there's nothing to interpolate). Exact, and memory-light — just
the cumulative filter `Φ` (length `P·fft_len`), not an `n_phi×n_chips` table. The φ-bank
with interpolation was a stepping stone (it plateaued, since `R_c` is a step function of
sub-chip phase); the prefix-sum is exact with less memory, so no Farrow is needed.

**Nav-bit edge handled exactly.** A 20 ms data-bit edge falls on a code-period (= chip)
boundary, so the wipe is applied **per chip** in the same sum (the `nav_bit` arg), giving
the exact `d̂·R_c`. The per-hop post-multiply is *not* good enough here — it smears the
~`P` hops straddling each edge by ~100% (worst-hop +13 dB), a ~−30 dB power leak that a
+60 dB GNSS source would splash all over the celestial signal.

```
 acquire ─► lock (cp,Doppler)/sat ─► trajectory predictor + nav decode
   broadcast per-PRN {cp, Doppler, rates, clock} + d̂   (~kbps total)
 node:  build R_c locally (hop-rate φ-bank, ~0.01 Tflop/s) · d̂ · ─► inject/peel
```

## Status

Wins 1 & 2 are settled. Hop-rate local generation (pathway #5) is validated, so
**(B) is the recommendation** — generate-local, only the kbps trajectory+bit
broadcast, no bulk replica distribution. (A) is the fallback if a node ever can't
afford even ~0.01 Tflop/s. The streaming generator wrapper is **built**
(`HopRateReplicaStream`: prefix-sum `Φ` once, stream hops, rebuild slowly as the
Doppler drifts; boost test `hoprate_stream_amortizes_and_matches`). Remaining: the GPU
kernel form (a bank of per-PRN chip-rate filters).
