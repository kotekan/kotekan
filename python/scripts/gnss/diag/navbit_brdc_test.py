#!/usr/bin/env python3
"""Fast test of BrdcLnavSource: stub the decoder instead of running it.

LnavPredictor.ingest is far too slow to drive a test (2500 emits > 15 min -- its _try_sync
repair search is ~O(hist^2) per call), and it is not what is under test here anyway. The
thing under test is the ALIGNMENT solve: can a satellite that never decoded be handed bits
that land on the right 20 ms slots, using only BRDC geometry plus a clock offset calibrated
from a DIFFERENT satellite?

So: build the sync/anchor state directly from the captured bits, then hold out each satellite
in turn -- calibrate the offset from the OTHERS only, and score the held-out one. That is
exactly the weak-satellite case, since the held-out satellite contributes nothing to its own
alignment.
"""
import datetime as _dt
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/lwlab/airspy_gps/kotekan/python/scripts/gnss")
sys.path.insert(0, "/home/lwlab/airspy_gps/kotekan/python/scripts/gnss/diag")
import gnss_ephemeris as ge                     # noqa: E402
from gps_nav_decode import frame_sync, decode_subframe  # noqa: E402
from navbit_reuse import stitch                 # noqa: E402
from navbit_brdc import BrdcLnavSource          # noqa: E402

LAT, LON, ALT = 43.9687, -79.2521, 260.0
BIT_S = 0.02


class St:
    """Minimal stand-in for navbit_predictor._PrnState."""
    def __init__(self, sync, hist, sf_data, utc_anchor, bit_s):
        self.sync, self.hist, self.sf_data = sync, hist, sf_data
        self.utc_anchor = utc_anchor      # EXACT (slot, utc) -- sub-bit phase preserved
        self.bit_s = bit_s


class Pred:
    def __init__(self, p):
        self._p = p


path = sys.argv[1]
wall = None
for line in open(path):
    wall = json.loads(line)["wall"]
    break

# EXACT bit-edge time per PRN. stitch() rounds onto a shared 20 ms slot grid, which discards
# each satellite's sub-bit phase -- and that phase IS the propagation delay we are trying to
# solve for. Recover an unrounded (slot, utc) pair per PRN straight from the emits.
anchor = {}
for line in open(path):
    d = json.loads(line)
    o, prn = d["nav_obs"], d["prn"]
    if not o.get("bits"):
        continue
    rec_dt, phase, br = float(o["rec_dt"]), int(o["phase"]), int(o["br"])
    bit_s = br * rec_dt
    bi = int(o["bits"][0][0])
    utc_bit = float(o["utc_ref"]) + (bi * br - phase) * rec_dt
    anchor.setdefault(prn, []).append((int(round(utc_bit / bit_s)), utc_bit, bit_s))

merged = stitch(path)
states = {}
for prn, m in merged.items():
    slots = sorted(m)
    runs, cur = [], [slots[0]]
    for a, b in zip(slots, slots[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    run = max(runs, key=len)
    if len(run) < 620:
        continue
    ba = np.array([(1 - m[s]) // 2 for s in run], dtype=np.int8)
    sync = None; sf1 = None
    for i, tow, sfid in frame_sync(ba):
        if i + 300 > len(ba):
            continue
        d, ok, _, _, _, _ = decode_subframe(
            ba, i, int(ba[i - 2]) if i >= 2 else 0, int(ba[i - 1]) if i >= 1 else 0)
        if not ok:
            continue
        if sync is None:
            sync = (run[0] + i, tow, sfid)
        if sfid == 1 and sf1 is None:
            sf1 = d
    if sync is not None:
        cands = anchor.get(prn)
        if not cands:
            continue
        a = min(cands, key=lambda c: abs(c[0] - sync[0]))   # nearest in time to the sync
        states[prn] = St(sync, m, {1: sf1} if sf1 is not None else {},
                         (a[0], a[1]), a[2])

print(f"stub-synced {sorted(states)} from {len(merged)} PRNs\n")

_when = _dt.datetime.fromtimestamp(wall, tz=_dt.timezone.utc)
eph = ge.parse_rinex_nav(ge.fetch_brdc(_when))
pa = ge.predict_all(eph, LAT, LON, ALT, _when, mask_deg=-90.0)
geom = {prn: (0.0, 0.0, v["el"], v["range_m"], v["sat_clk_s"])
        for (sysc, prn), v in pa.items() if sysc == "G"}

# offsets from every satellite, to see the spread
src = BrdcLnavSource(log=print)
src.update(eph, geom, Pred(states))
print(f"all-sat calibration: offset {src.offset:.6f} s, spread {(src.spread or 0)*1e3:.3f} ms, "
      f"{src.n_cal} sats, ready={src.ready()} {src.why_not()}\n")

print("HOLD-ONE-OUT (calibrated from the OTHER satellites only -- the weak-sat case):")
print(f"{'PRN':>4} {'el':>6} {'cal sats':>9} {'spread':>9} {'compared':>9} {'agree':>7} {'rate':>8}")
for held in sorted(states):
    others = {p: s for p, s in states.items() if p != held}
    if len(others) < 1:
        continue
    s2 = BrdcLnavSource(log=lambda m: None)
    s2.update(eph, geom, Pred(others))
    if not s2.ready():
        print(f"G{held:02d}  not ready: {s2.why_not()}")
        continue
    r = s2.verify(held, Pred(states))
    if r is None:
        print(f"G{held:02d}  no overlap")
        continue
    n, ag = r
    print(f"G{held:02d} {geom[held][2]:6.1f} {s2.n_cal:9d} {(s2.spread or 0)*1e3:8.2f}ms "
          f"{n:9d} {ag:7d} {100*ag/max(n,1):7.2f}%")
