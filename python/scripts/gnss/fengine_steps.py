#!/usr/bin/env python3
"""Whole-sample steps of the F-engine's inputs at each F-engine start.

The F-engine does not start with its inputs aligned: against the start before, an input can
come up a whole number of ADC samples earlier or later, by board or part of a board. The table
(TABLE, JSON) holds one row per start, so a product made under one start (a calibration,
beamforming weights, an element reference) can be used on data from another:

  adc_sample_ns     the ADC sample period
  n_inputs          the number of inputs
  rows              one per start, oldest first:
    frame0_int        the start: the integer second of its frame0, as in
                      builds/gw1202/frame0.json
    start_utc         the same as a UTC time
    prev_frame0_int   the frame0_int of the row before; null in the first row
    steps             per input, the ADC samples the input arrives later (+) or earlier (-)
                      than under the start before, within its pol half (inputs 0-63, 64-127)
                      against the largest group of inputs that moved together. That zero is a
                      convention (two groups can be near equal: 2026-11-26 pol 0, 23 and 22-24
                      inputs); pol_step is measured against the row as written. null for an
                      input no measurement read; null in place of the list for a start that
                      was not measured.
    pol_step          the same for inputs 64-127 as a whole against inputs 0-63; null if not
                      measured. A product that combines both halves needs it. A row that
                      names it half_step is read as well.
    measured_utc, source   when and from what the steps were measured

A step of n samples turns an input's phase at sky frequency f by 2 pi f n Ts. Measured on the
2026-11-25 start: the input's Tau A calibration weight turns by exp(+2 pi i f n Ts) and its
entry in the tracker's element model (get_elem_cal g_shared) by exp(-2 pi i f n Ts).

The watchdog of the element reference (elem_shared_ref.py watch) adds a row when the frame0
file shows a new start and fills its steps from the trackers' element models (fit_steps) once
the models have settled after it.

  show [TABLE]                           the rows, steps as input ranges
  set-steps FRAME0 FILE SOURCE [TABLE]   set a row's steps from FILE, a JSON list with one
                                         entry per input (null = not read), measured otherwise
  pol-step FRAME0 N [TABLE]              set a row's pol_step
"""
import argparse
import collections
import json
import os
import time

import numpy as np

TABLE = "/home/kvand/gnss/fixtures/obs/fengine_steps/sample_steps.json"
MAX_STEP = 16  # fit_steps tries steps up to this many samples either way
# fit_steps: inputs this close to one of them moved together, in at least GROUP_FRAC of the
# bands that read both (one band whose models are off does not split the group)
GROUP_DEG = 30.0
GROUP_FRAC = 0.75
MIN_BANDS = 3  # fit_steps: bands an input's step is fitted from, at least
# fit_steps: a band whose rms residual is above both is left out (a band off the others; on
# 2026-11-26 every pol-0 band read 12-34 deg, and leaving out all above 20 left the L5 and E5b
# bands alone, which cannot tell the steps apart)
MAX_BAND_RMS_DEG = 20.0
BAND_RMS_FACTOR = 1.5  # x the median band's
MIN_SCORE = 0.8  # fit_steps: mean cosine of an input's residuals at its step, at least
# fit_steps: by how much that beats the next best step, at least. In the GNSS bands a step and
# the step 8 samples off differ by 0.1 to 0.15, and across the L1 bands alone steps 2 samples
# apart differ by 0.005.
MIN_MARGIN = 0.05
# fit_steps: share of the inputs read by enough bands that must get a step
MIN_FIT = 0.75


def load(path=TABLE):
    with open(path) as f:
        table = json.load(f)
    for row in table["rows"]:
        if "half_step" in row:
            row.setdefault("pol_step", row.pop("half_step"))
    return table


def save(table, path=TABLE):
    """Write the table through a rename, so a reader sees the old file or the new one."""
    with open(path + ".tmp", "w") as f:
        json.dump(table, f, indent=1)
    os.replace(path + ".tmp", path)


def add_start(table, frame0_int, start_utc):
    """Append the row of a start that is not measured yet."""
    rows = table["rows"]
    rows.append(
        {
            "frame0_int": frame0_int,
            "start_utc": start_utc,
            "prev_frame0_int": rows[-1]["frame0_int"] if rows else None,
            "steps": None,
            "pol_step": None,
        }
    )


def steps_between(table, made, data, pol_step=False):
    """Per input, the samples it arrives later under the start `data` than under the start
    `made` (both frame0_int): what a product made under `made` is turned by to be used on data
    taken under `data`. None for an input one of the rows in between did not read; zeros when
    the two are the same start. With pol_step the rows' pol_step is added to the second half
    of the inputs, for a product that combines both pol halves.

    Raises ValueError when the rows do not lead from one start to the other without a gap, or
    one of them was not measured."""
    lo, hi = min(made, data), max(made, data)
    total, at = [0] * table["n_inputs"], lo
    for row in table["rows"]:
        if not lo < row["frame0_int"] <= hi:
            continue
        if row["prev_frame0_int"] != at:
            raise ValueError(
                "the row of start %d follows start %s, not %d: a start in between has no row"
                % (row["frame0_int"], row["prev_frame0_int"], at)
            )
        if row["steps"] is None or (pol_step and row["pol_step"] is None):
            raise ValueError(
                "the %s of start %d (%s) were not measured"
                % (
                    "steps" if row["steps"] is None else "pol_step",
                    row["frame0_int"],
                    row["start_utc"],
                )
            )
        half = len(row["steps"]) // 2
        for i, s in enumerate(row["steps"]):
            if s is None or total[i] is None:
                total[i] = None
            else:
                total[i] += s + (row["pol_step"] if pol_step and i >= half else 0)
        at = row["frame0_int"]
    if at != hi:
        raise ValueError("start %d has no row after start %d" % (hi, at))
    sign = 1 if data >= made else -1
    return [None if s is None else sign * s for s in total]


def phase(steps, freq_hz, sample_ns):
    """2 pi f n Ts in radians per input; NaN where the step is None."""
    n = np.array([np.nan if s is None else s for s in steps], dtype=float)
    return 2.0 * np.pi * freq_hz * sample_ns * 1e-9 * n


def _group(phases):
    """The largest group of inputs that moved together: around each input read by the most
    bands, the inputs read by MIN_BANDS or more that sit within GROUP_DEG of it in at least
    GROUP_FRAC of the bands that read both. An input no band reads is never a member, so dark
    inputs cannot make up the group."""
    read = np.isfinite(phases)
    n_read = read.sum(axis=0)
    best = np.zeros(phases.shape[1], dtype=bool)
    for q in np.flatnonzero(n_read == n_read.max()):
        both = read & read[:, [q]]
        with np.errstate(invalid="ignore"):
            d = np.abs(np.angle(np.exp(1j * (phases - phases[:, [q]]))))
        n = both.sum(axis=0)
        close = (both & (d <= np.radians(GROUP_DEG))).sum(axis=0)
        near = (n >= MIN_BANDS) & (close >= GROUP_FRAC * n)
        if near.sum() > best.sum():
            best = near
    return best


def _fit(phases, turn):
    """One pass of fit_steps over the bands given: (steps, residuals). steps holds None where
    too few bands read the input or no step fits. residuals[b][i] is input i's phase in band b
    less the band's constant and the input's best step, in radians, NaN where the input was not
    read."""
    n_in = phases.shape[1]
    # The inputs that moved together give each band its constant.
    group = _group(phases)
    const = np.array([np.angle(np.nansum(np.exp(1j * p[group]))) for p in phases])
    lags = np.arange(-MAX_STEP, MAX_STEP + 1)
    steps, resid = [None] * n_in, np.full(phases.shape, np.nan)
    for i in range(n_in):
        ok = np.isfinite(phases[:, i])
        if ok.sum() < MIN_BANDS:
            continue
        d = phases[ok, i] - const[ok]
        score = np.cos(d[None, :] - lags[:, None] * turn[None, ok]).mean(axis=1)
        first, second = np.argsort(score)[::-1][:2]
        resid[ok, i] = np.angle(np.exp(1j * (d - lags[first] * turn[ok])))
        if score[first] >= MIN_SCORE and score[first] - score[second] >= MIN_MARGIN:
            steps[i] = int(lags[first])
    return steps, resid


def fit_steps(phases, freqs_hz, sample_ns):
    """The steps of one pol half from the phase its inputs gained across a start, in several
    bands.

    phases[b][i] is the gain of input i in band b: radians, in the sense of 2 pi f n Ts, NaN
    where the band does not read the input. Each band carries a constant of its own.
    Returns (steps, rms): per input the step against the largest group of inputs that moved
    together, None where fewer than MIN_BANDS bands read the input or no step fits them clear
    of the next best; and the rms residual of the fitted inputs in degrees. A band whose
    residuals are well above the other bands' (MAX_BAND_RMS_DEG, BAND_RMS_FACTOR) is left out
    and the rest fitted again.

    Raises ValueError when fewer than MIN_FIT of the inputs read by enough bands get a step."""
    phases = np.asarray(phases, dtype=float)
    turn = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) * sample_ns * 1e-9
    steps, resid = _fit(phases, turn)
    with np.errstate(invalid="ignore"):
        band_rms = np.sqrt(np.nanmean(resid ** 2, axis=1))
        limit = max(
            np.radians(MAX_BAND_RMS_DEG), BAND_RMS_FACTOR * np.nanmedian(band_rms)
        )
        good = ~(band_rms > limit)
    if MIN_BANDS <= good.sum() < len(good):
        steps, resid = _fit(phases[good], turn[good])
    fitted = [s for s in steps if s is not None]
    n_read = int(np.isfinite(resid).any(axis=0).sum())
    if not fitted or len(fitted) < MIN_FIT * n_read:
        raise ValueError(
            "a step fits %d of the %d inputs read by %d bands or more"
            % (len(fitted), n_read, MIN_BANDS)
        )
    zero = collections.Counter(fitted).most_common(1)[0][0]
    ok = [s is not None for s in steps]
    return (
        [None if s is None else s - zero for s in steps],
        float(np.degrees(np.sqrt(np.nanmean(resid[:, ok] ** 2)))),
    )


def _run(a, b):
    return str(a) if a == b else "%d-%d" % (a, b)


def ranges(steps):
    """The steps as text, runs of inputs with one value together: "16-23 -4, 32-47 +4"; inputs
    at 0 are left out and unread ones listed last."""
    runs, start = [], 0
    for i in range(1, len(steps) + 1):
        if i == len(steps) or steps[i] != steps[start]:
            runs.append((start, i - 1, steps[start]))
            start = i
    moved = ["%s %+d" % (_run(a, b), s) for a, b, s in runs if s]
    unread = [_run(a, b) for a, b, s in runs if s is None]
    return (", ".join(moved) or "none moved") + (
        "; unread " + ", ".join(unread) if unread else ""
    )


def show(a):
    table = load(a.table)
    print("ADC sample %s ns; + = the input arrives later" % table["adc_sample_ns"])
    for row in table["rows"]:
        print("%s  frame0 %d" % (row["start_utc"], row["frame0_int"]))
        if row["steps"] is None:
            print("  not measured")
            continue
        print("  steps: %s" % ranges(row["steps"]))
        print(
            "  pol_step (inputs 64-127 against 0-63): %s"
            % ("not measured" if row["pol_step"] is None else "%+d" % row["pol_step"])
        )
        print("  measured %s, %s" % (row.get("measured_utc"), row.get("source")))


def row_of(table, frame0_int):
    rows = [r for r in table["rows"] if r["frame0_int"] == frame0_int]
    if not rows:
        raise SystemExit("the table has no row for start %d" % frame0_int)
    return rows[0]


def set_steps(a):
    table = load(a.table)
    with open(a.file) as f:
        steps = json.load(f)
    if len(steps) != table["n_inputs"]:
        raise SystemExit(
            "%s holds %d steps, not %d" % (a.file, len(steps), table["n_inputs"])
        )
    row = row_of(table, a.frame0)
    row["steps"] = steps
    row["measured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row["source"] = a.source
    save(table, a.table)


def set_pol_step(a):
    table = load(a.table)
    row_of(table, a.frame0)["pol_step"] = a.n
    save(table, a.table)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("table", nargs="?", default=TABLE)
    t = sub.add_parser("set-steps")
    t.add_argument("frame0", type=int)
    t.add_argument("file")
    t.add_argument("source")
    t.add_argument("table", nargs="?", default=TABLE)
    h = sub.add_parser("pol-step")
    h.add_argument("frame0", type=int)
    h.add_argument("n", type=int)
    h.add_argument("table", nargs="?", default=TABLE)
    a = ap.parse_args()
    {"show": show, "set-steps": set_steps, "pol-step": set_pol_step}[a.cmd](a)


if __name__ == "__main__":
    main()
