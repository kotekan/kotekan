#!/usr/bin/env python3
"""How much of the LNAV stream can be supplied WITHOUT decoding the satellite itself?

That is the question behind "predict weak-sat bits": subframes 1-3 are the ephemeris (which
we already hold from BRDC), and subframes 4-5 are almanac pages that MAY be common across the
constellation. Rather than write the IS-GPS-200 bit packing on faith, measure both directly on
live bits:

  A. WORD STABILITY -- for one satellite, which of the 10 words of each subframe change from
     frame to frame? Anything stable is supplyable from a single old decode; only the changing
     words need a live source. (Expected: HOW's TOW, and nothing else, for ~2 h.)

  B. CROSS-SATELLITE IDENTITY -- at the SAME TOW, which words are bit-identical between
     different satellites? Those are borrowable from any strong sat to feed a weak one. This
     is the empirical version of the sf4/5 page-sharing hypothesis: if it holds, sf4/5 come
     free for a satellite that has never been decoded.

Both together bound what a BRDC-driven predictor would still have to invent.
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/lwlab/airspy_gps/kotekan/python/scripts/gnss")
from gps_nav_decode import frame_sync, decode_subframe  # noqa: E402


def stitch(path):
    """nav_obs emits -> per-PRN (slot -> +-1), chained across emits by overlap vote.

    Emits carry their own polarity (the combiner's per-bit decision is sign-ambiguous per
    emit), so consecutive emits are aligned on their overlapping slots before merging.
    """
    emits = defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        o, prn = d["nav_obs"], d["prn"]
        rec_dt, phase, br = float(o["rec_dt"]), int(o["phase"]), int(o["br"])
        bit_s = br * rec_dt
        slots, sgns = [], []
        for bi, sg in o["bits"]:
            if int(sg) not in (1, -1):
                continue
            utc_bit = float(o["utc_ref"]) + (int(bi) * br - phase) * rec_dt
            slots.append(int(round(utc_bit / bit_s)))
            sgns.append(int(sg))
        if slots:
            emits[prn].append((slots, sgns))
    out = {}
    for prn, ems in emits.items():
        ems.sort(key=lambda e: e[0][0])
        merged = {}
        for slots, sgns in ems:
            ov = [(s, g) for s, g in zip(slots, sgns) if s in merged]
            flip = 1
            if ov:
                agree = sum(1 for s, g in ov if merged[s] == g)
                flip = 1 if agree * 2 >= len(ov) else -1
            for s, g in zip(slots, sgns):
                merged.setdefault(s, g * flip)
        out[prn] = merged
    return out


def subframes(merged):
    """Longest contiguous slot run -> frame_sync -> [(tow, sfid, data[10][24])]."""
    slots = sorted(merged)
    if not slots:
        return []
    runs, cur = [], [slots[0]]
    for a, b in zip(slots, slots[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    run = max(runs, key=len)
    if len(run) < 620:
        return []
    # GPS parity is polarity-independent -- the RECOVERED DATA is identical whether or not the
    # stream is inverted -- so the global sign never has to be resolved for this comparison.
    bits = np.array([(1 - merged[s]) // 2 for s in run], dtype=np.int8)  # +1->0, -1->1
    out = []
    for i, _tow, _sfid in frame_sync(bits):
        if i + 300 > len(bits):
            continue
        D29s = int(bits[i - 2]) if i >= 2 else 0
        D30s = int(bits[i - 1]) if i >= 1 else 0
        data, ok, tow, sfid, _, _ = decode_subframe(bits, i, D29s, D30s)
        if ok:
            out.append((tow, sfid, np.array(data, dtype=np.int8)))
    return out


def main():
    merged = stitch(sys.argv[1])
    per = {p: subframes(m) for p, m in merged.items()}
    per = {p: v for p, v in per.items() if v}
    print(f"synced {len(per)} PRNs: " + ", ".join(
        f"G{p:02d}({len(v)} sf)" for p, v in sorted(per.items())))
    if not per:
        print("no PRN synced -- nothing to measure")
        return

    print("\nA. WORD STABILITY frame-to-frame (per sat, per subframe id)")
    print("   word:  0=TLM 1=HOW 2..9=data.  '.'=identical across all frames, digit=#bits changed")
    for prn, sfs in sorted(per.items()):
        by = defaultdict(list)
        for tow, sfid, data in sfs:
            by[sfid].append(data)
        for sfid in sorted(by):
            v = by[sfid]
            if len(v) < 2:
                continue
            marks = []
            for w in range(10):
                ch = max(int((v[0][w] ^ x[w]).sum()) for x in v[1:])
                marks.append("." if ch == 0 else (str(ch) if ch < 10 else "+"))
            print(f"   G{prn:02d} sf{sfid} x{len(v):<2d}  " + " ".join(f"{m:>2s}" for m in marks))

    print("\nB. CROSS-SATELLITE word identity at MATCHED TOW")
    print("   (a word identical across sats is borrowable from ANY strong sat)")
    bytow = defaultdict(dict)
    for prn, sfs in per.items():
        for tow, sfid, data in sfs:
            bytow[(tow, sfid)][prn] = data
    agg = defaultdict(lambda: [0, 0])  # (sfid, word) -> [identical_pairs, total_pairs]
    for (tow, sfid), d in bytow.items():
        prns = sorted(d)
        if len(prns) < 2:
            continue
        for i in range(len(prns)):
            for j in range(i + 1, len(prns)):
                for w in range(10):
                    same = int((d[prns[i]][w] ^ d[prns[j]][w]).sum()) == 0
                    a = agg[(sfid, w)]
                    a[0] += int(same); a[1] += 1
    if not agg:
        print("   no TOW with >=2 satellites -- need a longer collection")
    else:
        for sfid in sorted({s for s, _ in agg}):
            row = []
            for w in range(10):
                ident, tot = agg.get((sfid, w), [0, 0])
                row.append("--" if not tot else f"{100 * ident // tot:>3d}")
            npair = max(agg[(sfid, w)][1] for w in range(10) if (sfid, w) in agg)
            print(f"   sf{sfid}  % identical by word: " + " ".join(f"{x:>3s}" for x in row)
                  + f"   ({npair} sat-pairs)")


if __name__ == "__main__":
    main()
