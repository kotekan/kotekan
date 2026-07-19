# ADR trim subtraction — design (2026-07-19)

## Problem
The exported ADR is the COMMANDED carrier phase: `fcar·t_abs ± phi`, where phi integrates
`f_nco = ctrim + ff` (GnssGpuRecordAssemble). The broker's shared-carrier-loop trim
(`ctrim`) therefore integrates directly into `adr_cycles`. The 2026-07-18/19 robustness
machinery (REACQ demotions, BOOTSTRAP re-pulls, watchdog re-seeds) moves trims by ±2–10 Hz
routinely; each move is a phase-ramp change the ADR faithfully records:
1 Hz held for 100 s = 100 cycles ≈ 25 m ≈ 200 TECU at L-band — the measured gf arc drifts
(±50–800 TECU, 07-19) match this in magnitude and shape. The 07-17 "instrument-grade"
ADR validation (0.105 Hz rms vs BRDC) was performed on SETTLED trims and remains valid;
the contamination is specifically the trim TRANSIENTS.

Falsified alternatives (07-19, measured): inter-band time-base offset (per-arc δ fit
inconsistent by orders of magnitude); emit-epoch jitter (combiner utc is exact — emits are
period-locked to integer stream seconds and the obs logger already stamps from it).

## Key insight
Unlike iono/multipath, this contamination is EXACTLY KNOWN — the broker commands it.
Subtracting the known command history removes it losslessly.

What must NOT be blindly subtracted: the trim also absorbs real slow signal (any carrier
drift the f_ref/feed-forward model misses, including some iono rate). The subtraction must
remove the transients (steps, Hz-scale) while preserving slow content (≲0.05 Hz scale).
In practice the per-record age re-pins fold slow dynamics into f_ref, so the trim should
hold only (a) the frac-N LO constant (harmless: drops in rates) and (b) transients — but
v1 verifies this empirically by comparing full-subtraction vs high-pass-subtraction arcs.

## v1 (python-only, works on existing data): broker-log trim journal
The broker's CAR lines log every trim update, timestamped since 2ec49e04:
`[broker HH:MM:SS.mmm] CAR: PRN 20 resid +0.06 Hz trim +5.22; ...`
- New tool `gnss_trim_journal.py`: parse per-band broker logs → per-sat step series
  trim(t) (piecewise-constant), write jsonl.
- `gnss_tec.py --trim-a/--trim-b <journal>`: build trim_cycles(t) = ∫trim dt per sat,
  subtract λ·trim_cycles from each band's ADR before differencing. Two modes:
  `--trim-subtract full|highpass` (highpass = subtract trim minus its per-arc linear fit).
- Timing accuracy: the log records the command at broker-POST time; the tracker applies
  it within ~0.2–1 s → residual error ≈ step × 1 s ≈ 0.25 m per event (vs 25 m
  uncorrected): ~100× improvement, enough to validate the concept on today's data.
- Limitation: CAR lines are NOT emitted while gates hold the trim (no update = no line,
  value unchanged — piecewise-constant interpolation is exact there). Trim clears on
  re-seed (car_trim deleted) — journal must reset the level at FIRST SEED lines.

## v2 (durable): per-record export through the pipeline
The tracker already ships everything needed — no tracker change:
`ctrim = (f_nco + fcar_report − fcar) / 2` (from PrnCtl fields, exact identity).
- GnssGpuRecordAssemble: accumulate `_trim_cyc[p] += ctrim · dt_record`; write into the
  record (schema bump record_floats 20 → 21, same procedure as the 16→20 segmented-wipe
  bump: gnssRecord.hpp + assembler write + combiner read + configs).
- GnssCoherentCombiner: carry `trim_cycles` per sat; export in get_status next to
  adr_cycles (drift-free, emit-exact — no ±1 s journal error).
- gnss_observables: pass through; gnss_tec: prefer the exported field over the journal.
- Deploy: rebuild + relaunch; validate v2-vs-v1 agreement on one soak, then the
  gf arc scatter target: ≤ 1 TECU (from 20–200).

## Order of work
1. v1 journal + gnss_tec subtraction → validate on today's post-11:19 L2C/L1 logs
   (trims churn there = maximal signal).
2. If arc scatter collapses as predicted → v2 schema bump (C++), bench replay smoke,
   fleet at a relaunch.
3. Then revisit dTEC quality ladder: next floors are L5-GPS duty and arc lengths.
