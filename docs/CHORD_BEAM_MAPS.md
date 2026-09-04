# CHORD GNSS beam maps — the production pipeline

How to roll a beam map from the element archive, end to end, without re-deriving anything.
Written 2026-08-26 after the first maps off `elem_*.jsonl`; supersedes the ad-hoc
`fixtures/beam/mkobs.py` route (which reconstructed rows from `FLEET-COH` log lines because
the obs logger was not running).

**TL;DR**

```bash
PY=/home/kvand/gnss/venv/bin/python          # ⚠️ venv, NOT venv-ft -- healpy lives here
S=/home/kvand/gnss/kotekan/python/scripts/gnss
OBS=/home/kvand/gnss/fixtures/obs
mkdir -p /tmp/beam/maps && cd /tmp/beam

# 1. archive -> obs rows (one chain at a time; run the five in parallel, ~70 s each)
$PY $S/gnss_beam_elem2obs.py --elem $OBS/elem_gps_l5_20260825.jsonl \
        $OBS/elem_gps_l5_20260826.jsonl --sys G --out p2_G.jsonl

# 2. keep real detections, then veto the epochs that rail the quantiser (CROSS-CHAIN)
$PY -c 'import json,sys
[sys.stdout.write(l) for l in open("p2_G.jsonl") if json.loads(l)["sig"]>=2.0]' > p2s_G.jsonl
$PY $S/gnss_beam_veto.py --obs p2s_*.jsonl --el0 81.41 --az0 180.0 --deg 5 --suffix v

# 3. transits -> coadd -> combine -> render   (the pre-existing pipeline, unchanged)
$PY $S/gnss_beam_map.py transits --obs p2s_G_v.jsonl --tag G --night n20260825 \
        --outdir data --tmin 1786900000 --tmax 1787772000
$PY $S/gnss_beam_map.py coadd --transits 'data/transits/n20260825/G_*.npz' \
        --out maps/G.npz --quantity inc
$PY $S/gnss_beam_map.py combine --inputs maps/G.npz maps/gal_e5a.npz maps/bds_b2a.npz \
        --out maps/BAND1176.npz
$PY $S/gnss_beam_map.py render --map maps/BAND1176.npz --png beam_1176.png --title '...'

# 4. read it as a profile, against a prediction
$PY $S/gnss_beam_radial.py --map maps/BAND1176.npz:'all epochs' maps/V1176.npz:'vetoed' \
        --png radial_1176.png --sectors --title '...'
```

---

## 1. The data source

`/home/kvand/gnss/fixtures/obs/elem_<chain>_<YYYYMMDD>.jsonl`, written by the broker's
`--element-archive-dir` (`gnss_broker/instruments.py`). One append-per-day file per chain,
present satellites plus the noise probes, throttled to `--element-archive-every-s`.

⚡ **It is the most durable record the instrument has.** It survived every broker restart and
every log truncation of 2026-08-24…26 — continuous 00:00–23:59 for all five chains on days
whose broker logs are gone entirely. When a question needs overnight history, look here first.
Cost: ~200 MB per chain per day, ~110k rows, ~70 s to scan a day with stock `json`.

A row is `{t, chain, prn, inst, probe, keff, hop, u[32], p2[32], q[32]}`.

## 2. ⚠️⚠️ WHICH FIELD IS THE BEAM

`GnssCoherentCombiner::get_elements_callback` defines all four parts, and only one is a beam:

| field | meaning |
|---|---|
| `u`  | `<A_e · conj(LOO ref)>` — per-element complex **gain**, phase in the array-mean convention |
| `p2` | `<\|A_e\|²>` — *"the incoherent beam-map power, still biased — its debias is the broker's job, from the probes"* ⇐ **THE BEAM** |
| `q`  | `<\|LOO ref\|²>` — the **normaliser**; `amp_e ~ \|u\|/q` *"in array-mean units"* |

**`|u|/q` is a ratio to the array mean and is therefore flat across the sky by construction** —
every element rises and falls with its own reference. Mapping it produces a smooth, plausible,
completely meaningless ~6 dB all-sky "beam". That happened on the first attempt here, and
**the tell was physical, not statistical**: a 6 m dish at 1176 MHz cannot have a 6 dB full-sky
pattern. Mapping probe-debiased `p2` instead gives 14–16 dB horizon-to-zenith on all four
chains independently and ~40 dB of range, with a compact core in the right place.

The general rule this is an instance of: **ask what the denominator is before mapping
anything.** A normalised quantity looks exactly like a measurement.

`gnss_beam_elem2obs.py` writes both quantities on every row, into the two fields the existing
pipeline already transports:

* `cn0_inc_dbhz` = `10 log10( median_e ( p2_e − floor_e ) )` — **the beam**. Debias in the
  POWER domain; floors are per (instance, **element**, 5-min bin) from the probe PRNs, never
  medianed across elements (that over-subtracts the quiet ones into negative power). A sample
  below its own pedestal is dropped, not clamped — clamping piles non-detections at an
  artificial floor and fakes a beam edge.
* `cn0_coh_dbhz` = `20 log10( |Σ_e u_e| / Σ_e q_e )` — **not a beam**: the raw-parts phase
  alignment. Kept because it is free and diagnostic (see §6).

Range is divided out by default (`--no-range-norm` to keep it): a satellite is ~2.5 dB
stronger overhead than at the horizon from spreading loss alone, which would masquerade as
beam. BRDC supplies the range at the row epoch.

Instances are **medianed, not summed** — the sky does not care which GPU served a record, and
a wedged instance serving a ten-minute-old sky would otherwise pull a mean.

## 3. ⚠️ THE BORESIGHT RAILING VETO

A satellite close to boresight rails the 4+4b quantiser **for every chain at once** — L5, E5a,
E5b, B2a and B2b all ride the same nibbles. Measured over 2026-08-25/26 (51,607 samples,
7,480 epochs), with the beam removed first by subtracting each sample's own 2° radius-bin
median:

| closest sat to boresight | residual of far (>40°) sats | tracked sats/epoch |
|---|---|---|
| 0–3° | **+2.70 dB** | **4.0** |
| 3–5° | +2.04 dB | 6.0 |
| 5–8° | +0.68 dB | 6.0 |
| 8–12° | +0.11 dB | 6.0 |
| 12–20° | −0.16 dB | 7.0 |
| 20–90° | −0.30 dB | 5.0 |

Two effects at once, pushing opposite ways:

* the **survivors read 2–3 dB HIGH**, not low — railing redistributes power rather than
  cleanly attenuating, and the 5-minute probe pedestal cannot follow a transit that lasts
  minutes, so the debias under-subtracts exactly when it matters;
* the **population collapses**, 7 satellites per epoch down to 4 — losses the archive can
  never show directly, because it only records satellites that were present. Judging the bias
  on the survivors alone is the survivor-bias trap in map form.

`gnss_beam_veto.py` drops every row in an epoch whose closest satellite (pooled over **all**
chains) is inside `--deg`. **The pooling is the point**: a per-chain veto would clean the chain
carrying the bright satellite and leave the other four contaminated *while they look like
clean controls*. At 5° it removes ~4.7% of epochs.

⚠️ Vetoing removes the main lobe itself. That is the honest outcome, not a bug: **the main
lobe cannot be measured this way because the main lobe is what breaks the measurement.** Quote
core numbers from the unvetoed map as a LOWER BOUND, with the bias named.

## 4. Coadd, combine, render — the pre-existing pipeline

`gnss_beam_map.py` is unchanged and its design still holds: transits stay as raw
(t, az, el, value) tracks (a pixelisation would only blur a 1-D track); healpix (nside 64,
~0.92°) enters at the **coadd**; coadds are stored as raw accumulators (`n`, `s1`, `s2`) so
nights and constellations **combine by addition forever after**. That is the build-up-across-
days workflow — new nights never require reprocessing old ones.

Combine only within a band (`--quantity` must match, and coh/inc must never be mixed):
1176.45 MHz = gps_l5 + gal_e5a + bds_b2a; 1207.14 MHz = gal_e5b + bds_b2b.

## 5. Reading it: `gnss_beam_radial.py`

A healpix map is the right thing to store and the wrong thing to read — a dish beam is
circularly symmetric to first order, so the informative cut is 1-D. The tool gives a radial
profile (median with a 16–84% band, log-spaced bins), an azimuth-sector panel, and an **Airy
overlay** so the measurement is read against a prediction rather than against itself.

The prediction, from the pointing memo: boresight **az 180.0, el 81.41** (dishes 8.59° south
of zenith, dec +40.73 — ⚠️ `telescope.dish_coelev_deg` reads −27.3 and does **not** give
this); 6 m dish at L5 → FWHM 2.48°, first null 2.97°, first sidelobe 3.97° at −17.6 dB.

⚠️ nside 64 is ~0.92°/pixel against a 2.48° FWHM, so the main lobe is barely three pixels
across. Never quote a fitted FWHM from this without saying so.

## 6. What the first run found (2026-08-25/26)

* **Core centroid el 82.7° az 176.4°** (1176) and **el 81.5° az 170.4°** (1207) — two bands,
  three constellations, landing within ~1.4° of the independently-known boresight. The
  pipeline validates against the pointing memo without being told it.
* **The measured skirt sits 20–30 dB above the ideal Airy envelope**: ~−15 dB at 10° where a
  clean 6 m aperture predicts ~−35 to −40, and ~−30 dB at 80°. The wide-angle response is
  therefore *not* diffraction sidelobes — feed spillover and structure scattering dominate.
  This is also precisely why GNSS satellites are trackable all over the sky.
* **No well-sampled pixel inside 2.7°** of boresight, i.e. the main lobe is unmeasured — the
  railing and the rarity of near-boresight passes conspire (§3).
* **Azimuthal asymmetry**: the southern sector runs ~5 dB above east/west beyond 20°, with
  north intermediate. Note this is *despite* the southern sector sampling LOWER elevations at
  the same boresight radius (boresight is 8.59° south of zenith, so radius θ south → el
  81.4−θ, radius θ north → el 180−81.4−θ). Unexplained; re-check with more nights before
  treating it as real.
* Free sanity check: the raw-parts coherent sum sits at −12.3 dB against a −13.4 dB
  random-phase prediction for 22 live elements, with no spatial structure — exactly right for
  an uncalibrated, unsteered sum (`u`'s phases still carry cable and geometry).

## 7. Next steps

* **More nights**: pure addition, no reprocessing. The obvious first extension.
* **Phased-array map**: apply the elemcal gains before summing `u`, turning the §6 sanity
  check into a real coherent-beam map instead of a random-phase floor.
* **Per-element maps**: `p2` is per element, so each antenna's own pattern is already in the
  archive — a per-element beam map is a loop, not new physics, and would separate feed
  problems from array problems.
* **The main lobe**: needs either an attenuator/AGC change so a near-boresight pass does not
  rail, or a deliberate observation of one of the known near-boresight passes with the gain
  backed off. PRN 19 passes 0.40° off boresight at ~08:04 UTC (recurring, drifting ~4 min
  earlier daily).
* **Run the obs logger** (`gnss_observables.py`) if the full RINEX-in-spirit record is wanted;
  the element archive is richer per element but carries no code/phase observables.
