#!/usr/bin/env python3
"""Fold the F-engine's per-input sample steps into cudaDtvSubtract state files.

    dtv_fold.py check STATE.bin                      # what a fold to the current F-engine start would do
    dtv_fold.py fold STATE.bin OUT.bin [--to FRAME0_INT]

An F-engine start can move inputs by whole ADC samples (11-25 18:23Z: 51 inputs by -4..+4). Input i's response then
turns by exp(+i 2 pi f n_i Ts) (n_i > 0 = later), so a state made before the start subtracts with stale weights. The
fold gives each source the weights it would have had, w e^{-i 2 pi f n Ts}, and g e^{+i 2 pi f n Ts} when the state
was saved before the start. A state saved after it has a g the rolling fit refitted for the new inputs against the
stale weights: that g is zeroed, and the rolling fit refits it for the folded weights within seconds (as after an
eigen start), where turning it would turn the moved inputs twice and trip the channel.

Steps: STEPS (fixtures/obs/fengine_steps/sample_steps.json, one row per F-engine start, keyed on frame0_int; positive =
later). A state's .json beside it (dtv_roll.py save) gives the start each source's weights were made under ("epochs",
"freq_id:source" -> frame0_int). A source folds the sum of the rows between its start and the target, and is left as
it is (and reported) when the chain has a gap, a row was not measured, or inputs not read carry SHARE or more of
its response.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnss")
)
import dtv_state  # noqa: E402

STEPS = os.environ.get(
    "DTV_FENGINE_STEPS", "/home/kvand/gnss/fixtures/obs/fengine_steps/sample_steps.json"
)
FRAME0 = os.environ.get("DTV_FRAME0", "/home/kvand/gnss/builds/gw1202/frame0.json")
NI = 128
# inputs not read by a row (dark or flagged) take step 0 for a source while they carry less than this share of its
# response power |g|^2, or when one of them carries all but this share (a source on one input: its step cancels
# between w and g)
SHARE = 0.05


def fid_hz(fid):
    return (300.0 + (fid - 1536) * 0.1953125) * 1e6


def frame0_now(path=FRAME0):
    return int(json.load(open(path))[-1]["frame0_int"])


def load_steps(path=STEPS):
    return json.load(open(path)) if os.path.exists(path) else {"rows": []}


def steps_between(rows, e, d, halves=True):
    """Sample step of each input from the F-engine start e to d (both frame0_int; NaN = not read), or ValueError.
    halves: the product combines inputs 0-63 and 64-127, so it needs the rows' step between them ("half_step")."""
    if e == d:
        return np.zeros(NI)
    lo, hi, sign = (e, d, 1) if e < d else (d, e, -1)
    span = [r for r in rows if lo < r["frame0_int"] <= hi]
    if not span or span[-1]["frame0_int"] != hi:
        raise ValueError("no row for the start %d" % hi)
    prev = lo
    for r in span:
        if r.get("prev_frame0_int") != prev:
            raise ValueError("gap before the start %d" % r["frame0_int"])
        prev = r["frame0_int"]
    n = np.zeros(NI)
    for r in span:
        if r.get("steps") is None:
            raise ValueError("the start %d was not measured" % r["frame0_int"])
        h = r.get("half_step")
        if halves and h is None:
            raise ValueError("the start %d: no half_step" % r["frame0_int"])
        n += np.array([np.nan if v is None else v for v in r["steps"]], float)
        n[64:] += h or 0
    return sign * n


def fold(st, epochs, d, doc, g_refit=False):
    """Fold the sources of st (dtv_state.read_state) made before the start d, in place; g_refit: st was saved
    under d, so its g is zeroed rather than turned. (epochs after, [(source, why not folded)])."""
    ts = doc.get("adc_sample_ns", 0.3125) * 1e-9
    out, skipped = dict(epochs), []
    for fid, r in st["recs"].items():
        for k in range(st["n_src"]):
            kk = "%d:%d" % (fid, k)
            e = epochs.get(kk)
            if e is None or e == d:
                continue
            p = (np.abs(r["g"][k]) ** 2).sum(axis=0)
            try:
                n = steps_between(doc["rows"], e, d, p[:64].any() and p[64:].any())
            except ValueError as x:
                skipped.append((kk, str(x)))
                continue
            u = np.isnan(n)
            share = p[u].sum() / p.sum() if p.sum() > 0 else 0.0
            if share > SHARE and not (
                u[p.argmax()] and p.max() >= (1 - SHARE) * p.sum()
            ):
                skipped.append(
                    (
                        kk,
                        "inputs %s not read carry %.0f%%"
                        % (list(np.where(u & (p > 0))[0]), 100 * share),
                    )
                )
                continue
            n[u] = 0
            ph = np.exp(2j * np.pi * fid_hz(fid) * n * ts).astype(np.complex64)
            r["w"][k] *= np.conj(ph)
            if g_refit:
                r["g"][k] = 0
            else:
                r["g"][k] *= ph[None, :]
            out[kk] = d
    return out, skipped


def fold_file(p, q, d, doc=None):
    """Fold the state p (and its .json) into q for the start d; (number folded, skipped)."""
    side = json.load(open(p[:-4] + ".json"))
    if side.get("frame0_int") is None:
        return 0, [("*", "saved while the frame0 file did not match the frames")]
    st = dtv_state.read_state(p)
    epochs, skipped = fold(
        st, side.get("epochs", {}), d, doc or load_steps(), side.get("frame0_int") == d
    )
    n = sum(1 for kk, e in epochs.items() if e != side.get("epochs", {}).get(kk))
    dtv_state.write_state(q, st)
    with open(q[:-4] + ".json", "w") as fh:
        json.dump(
            dict(side, epochs=epochs, frame0_int=d, folded_from=p),
            fh,
            indent=1,
            sort_keys=True,
        )
    return n, skipped


def stale(side, d):
    return sorted({e for e in side.get("epochs", {}).values() if e not in (None, d)})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("state")
    p.add_argument("--to", type=int)
    p = sub.add_parser("fold")
    p.add_argument("state")
    p.add_argument("out")
    p.add_argument("--to", type=int)
    a = ap.parse_args()
    d = a.to or frame0_now()
    if a.cmd == "check":
        side = json.load(open(a.state[:-4] + ".json"))
        st = dtv_state.read_state(a.state)
        epochs, skipped = fold(st, side.get("epochs", {}), d, load_steps())
        n = sum(1 for kk, e in epochs.items() if e != side["epochs"].get(kk))
        print("to %d: sources from starts %s; %d would fold" % (d, stale(side, d), n))
        for kk, why in skipped:
            print("  %s not folded: %s" % (kk, why))
    else:
        n, skipped = fold_file(a.state, a.out, d)
        print("%s: %d sources folded to %d, %d left" % (a.out, n, d, len(skipped)))
        for kk, why in skipped:
            print("  %s: %s" % (kk, why))


if __name__ == "__main__":
    main()
