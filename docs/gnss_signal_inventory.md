# GNSS Signal Inventory — peel-framework reference

Planning reference for the GNSS-calibrator peel framework. Bandwidths are the approximate
null-to-null main-lobe span (MHz); "center" is the carrier (MHz).

## How to read this for peel

A satellite is removed via **one civil "handle"** per band: a known-code, reconstructable
component (e.g. GPS C/A) locks the source's **direction `a`** and timing. From `a` you then
clean the *whole* source across the band — encrypted components included — **without despreading
them**, because the per-channel contribution is rank-1 `a(f)·a(f)ᴴ·P(f)`:

- **Reconstructable** (open, known code) → can **voltage-PEEL** (subtract the waveform `aᵢ·g(t)`;
  preserves the spatial DOF behind the sat).
- **Encrypted / classified** (P(Y), M, PRS, military) → **no waveform** → **voltage-NULL** only
  (project the direction out; loses a DOF that heals as the sat sweeps) or **vis-PEEL** (rank-1
  power subtraction, no DOF loss, for imaging). Direction-only — encryption is irrelevant.

So the peel needs ONE handle per occupied center, not every component. Overlapping centers (below)
mean one channelized correlator serves several constellations at once — later constellations are
just **more handles into the same channels**.

Status key: ✅ implemented (gnssSignal.hpp) · ⚪ civil, addable handle · 🔒 encrypted → null-only.

## GPS

| Component | center | modulation | ~BW | pilot | status |
|---|---|---|---|---|---|
| C/A      | 1575.42 | BPSK(1)         | 2  | – | ✅ L1 handle |
| L1C      | 1575.42 | BOC(1,1)+TMBOC  | 4  | ✓ | ⚪ Block III civil |
| P(Y)     | 1575.42 | BPSK(10)        | 20 | – | 🔒 W-encrypted (semi-codeless L1×L2 only) |
| M        | 1575.42 | BOC(10,5)       | 30 (lobes ±10) | – | 🔒 classified code, no replica |
| L2C CM/CL| 1227.6  | BPSK(1.023 comb)| 2  | ✓ (CL) | ✅ L2 handle |
| P(Y)     | 1227.6  | BPSK(10)        | 20 | – | 🔒 |
| M        | 1227.6  | BOC(10,5)       | 30 | – | 🔒 |
| L5 I5/Q5 | 1176.45 | BPSK(10)        | 20 | ✓ (Q5) | ✅ L5 handle (deep wipe done) |

## Galileo

| Component | center | modulation | ~BW | pilot | status |
|---|---|---|---|---|---|
| E1-B/C   | 1575.42  | CBOC(6,1,1/11) | 4  | ✓ (E1-C) | ⚪ overlaps GPS L1 |
| E1-A PRS | 1575.42  | BOC(15,2.5)    | 30 | – | 🔒 |
| E6-B/C   | 1278.75  | BPSK(5)        | 10 | ✓ | ⚪ HAS / commercial |
| E6-A PRS | 1278.75  | BOC(10,5)      | 30 | – | 🔒 |
| E5a      | 1176.45  | BPSK(10)       | 20 | ✓ | ⚪ overlaps GPS L5 |
| E5b      | 1207.14  | BPSK(10)       | 20 | ✓ | ⚪ |
| E5 (full)| 1191.795 | AltBOC(15,10)  | ~51 | ✓ | (E5a+E5b as one wideband signal) |

## GLONASS

| Component | center | modulation | ~BW | pilot | status |
|---|---|---|---|---|---|
| L1OF | ~1602 + k·0.5625 (k=−7..6) | BPSK(0.5) | 1 | – | ⚪ **FDMA** — per-satellite frequency |
| L2OF | ~1246 + k·0.4375            | BPSK(0.5) | 1 | – | ⚪ **FDMA** |
| L3OC | 1202.025 | BPSK(10)    | 20 | ✓ | ⚪ modernized **CDMA** (GLONASS-K) |
| L1OC/L2OC | 1600.995 / 1248.06 | BOC(1,1)/BPSK | 4 | ✓ | ⚪ modernized CDMA, rolling out |
| L1SF/L2SF | (FDMA) | – | 1 | – | 🔒 military |

**FDMA is the wrinkle:** legacy GLONASS puts each *satellite* on a different frequency (one code,
N carriers) — the opposite of CDMA's one-carrier-many-codes. The replica generator + channel plan
must retune per satellite. The modernized L3OC/L1OC are CDMA (same machinery as the rest), so favor
those once enough GLONASS-K are up.

## BeiDou (BDS-3)

| Component | center | modulation | ~BW | pilot | status |
|---|---|---|---|---|---|
| B1I  | 1561.098 | BPSK(2)        | 4  | – | ⚪ legacy |
| B1C  | 1575.42  | BOC(1,1)+QMBOC | 4  | ✓ | ⚪ overlaps GPS L1 / Gal E1 |
| B2a  | 1176.45  | BPSK(10)       | 20 | ✓ | ⚪ overlaps GPS L5 / Gal E5a |
| B2b  | 1207.14  | BPSK(10)       | 20 | ✓ | ⚪ overlaps Gal E5b |
| B3I  | 1268.52  | BPSK(10)       | 20 | – | ⚪ |
| B1A/B3A | 1575.42 / 1268.52 | BOC(14,2) | 30 | – | 🔒 military |

## Frequency-overlap map (what stacks where)

The peel-valuable insight: a few centers carry three constellations at once, so one channelized
correlator + one set of covering channels serves all of them.

| center MHz | who's there |
|---|---|
| **1575.42** | GPS (C/A·L1C·P(Y)·M) · Galileo E1 · BeiDou B1C — the busiest band |
| **1176.45** | GPS L5 · Galileo E5a · BeiDou B2a |
| **1207.14** | Galileo E5b · BeiDou B2b |
| 1227.6  | GPS L2 (L2C·P(Y)·M) |
| 1268.52 | BeiDou B3 |
| 1278.75 | Galileo E6 |
| 1202.025| GLONASS L3OC |
| 1561.098| BeiDou B1I |
| ~1602 / ~1246 | GLONASS L1OF / L2OF (FDMA, off on their own) |

Three "stacks" reward a shared correlator first: **1575.42** (GPS+Gal+BDS), **1176.45**
(GPS+Gal+BDS), **1207.14** (Gal+BDS).

## Peel-relevance summary

- **Handles you have:** GPS C/A (1575.42), L2C (1227.6), L5 Q5 (1176.45). Enough to peel/null the
  *entire* GPS source on all three bands, P(Y)/M included.
- **Encrypted → null-only (no waveform):** GPS P(Y) & M, Galileo PRS (E1-A/E6-A), BeiDou & GLONASS
  military. Remove via vis-peel or voltage-null using the civil handle's direction — never peeled.
- **Voltage-peel targets (reconstructable):** the civil pilots — GPS C/A & L1C & L5 Q5, Galileo
  E1-C/E5/E6 pilots, BeiDou B1C/B2a/B2b pilots, GLONASS L3OC. Pilots are ideal: dataless, fully
  known modulation.
- **Bandwidth to capture for a full peel:** the C/A handle is ±1 MHz, but the P(Y)/M power lives out
  to ±15 MHz on L1/L2 — so a *complete* source removal needs the wideband channels (CHORD's full-band
  correlator), even though the *handle* that finds the direction is narrowband.
- **Next constellations are additive, not new machinery:** each civil pilot is another handle into
  channels you're already correlating (esp. the three stacked centers). FDMA GLONASS is the one that
  needs real new plumbing (per-satellite retune).
