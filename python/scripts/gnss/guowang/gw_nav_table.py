#!/usr/bin/env python3
"""The broker's GuoWang table with each sender's own ephemeris and clock where the receiver
decoded them.

gw_ephem_table.py tables the satellites from their TLEs (hundreds of metres) and the library's
theta, which drifts by microseconds between measurements. Each satellite also broadcasts its
orbit (good to ~1 cm over toe - 100 s to toe + 600 s) and clock bias on its low-rate channel,
re-issued every 5 minutes (guowang1202_nav). This job polls the receiver's symbols
(GnssNavSymbols, gw_symbols), decodes every satellite's issues, and writes the TLE table with
each decoded issue's arc in place of its TLE: ITRS position and velocity on a 10 s grid from
the broadcast orbit, and clk_s = af0 + offset, the broadcast bias being the sender's clock
offset as GPS's af0 is, and offset (--clock-offset-us) one constant for the receiver. Satellites
share codes, so each issue takes its shell, name and code fields from the TLE record of its PRN
that flies where the issue puts it. The satellites outside the almanac (sender IDs 109 and up)
send only fill pages, which are skipped. A TLE segment is cut around the decoded arcs, so the TLE
still covers the rest of the pass. The output is rewritten (atomically) whenever an issue or the
TLE table changes; the broker reloads it by mtime (gw-table).

With --decode-health-file it also exports each code's decode state for the viewer's
decode-health panel (decode_health.py): subframes decoded, whether its symbols decode now, and
whether a broadcast arc is in the table. With --progress-file it writes nav_decode's progress
file: each code's subframe count and when each subframe ended, over the last 2 minutes.

    gw_nav_table.py --tle-table gw_ephem_tle.json --out gw_ephem.json [--clock-offset-us 0]
                    [--symbols http://localhost:12048/gw_symbols] [--every 5] [--log PATH]
                    [--decode-health-file PATH] [--progress-file PATH]
"""

import argparse
import copy
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import decode_health  # noqa: E402
import guowang1202_nav as gn  # noqa: E402
from gnss_ephemeris import BDT_GPST  # noqa: E402

ARC = (-100.0, 600.0)  # the fit arc about toe (PAYLOAD.md)
# An issue's TLE record is one that puts the satellite this close, m: TLEs are good to a few km,
# and the wrong shell or satellite puts it hundreds of km off.
MATCH_M = 20e3
# The clock's uncertainty (clk_sigma_s), which the broker's clock solve reads: the library's
# theta scatters by microseconds; the broadcast bias agrees across satellites to tens of
# nanoseconds (PAYLOAD.md).
CLK_SIGMA_TLE_S = 5e-6
CLK_SIGMA_NAV_S = 50e-9
# An issue needs all four pages, and the symbols hold only 6 s of a 12 s page cycle, so a fresh
# start takes a few cycles to rebuild its issues. Until then the table on disk keeps its arcs
# (each good to toe + 600 s).
FIRST_WRITE_S = 60.0
PROGRESS_KEEP_S = 120.0


def gpst(bdt_week, sow):
    """GPS seconds of BDT week and second of week."""
    return (bdt_week + 1356) * 604800.0 + sow - BDT_GPST


def shell_of(rec):
    """50 or 86 from a TLE record: the 50 deg shell flies lower."""
    rad = float(np.mean([np.linalg.norm(p) for p in rec["pos"]]))
    return 50 if rad < 7.54e6 else 86


def sender(w, recs):
    """The TLE record among `recs` (one PRN's) of the satellite that sent the issue `w`: the one
    that puts the satellite nearest the issue's position on the record's own shell, at the
    record's grid point nearest toe within the issue's arc; None if none is within MATCH_M.
    """
    week, toe = int(w["week"]), w["toe"]
    best, best_d = None, MATCH_M
    for r in recs:
        k = int(round((gpst(week, toe) - r["t0_gpst"]) / r["dt"]))
        k = min(max(k, 0), len(r["pos"]) - 1)
        sow = toe + r["t0_gpst"] + k * r["dt"] - gpst(week, toe)
        if not ARC[0] <= sow - toe <= ARC[1]:
            continue
        p = gn.Ephemeris(shell=shell_of(r)).position(w, sow)
        d = float(np.linalg.norm(p - np.array(r["pos"][k])))
        if d < best_d:
            best, best_d = r, d
    return best


def is_fill(sf):
    """True for a fill page, sent by the satellites outside the almanac in place of an
    ephemeris: its last 24 bytes all read 0x2d + 8 (page - 1)."""
    b = np.packbits(np.asarray(sf.payload, dtype=np.uint8)[: len(sf.payload) // 8 * 8])
    return bool(np.all(b[-24:] == 0x2D + 8 * (sf.page - 1)))


def nav_record(prn, eph, w, offset_s, step_s, like):
    """A table record over the issue's fit arc from its words `w`; `like` (its sender's TLE
    record) supplies the shell, the satellite's name and code fields."""
    toe = w["toe"]
    week = int(w["week"])
    sows = np.arange(toe + ARC[0], toe + ARC[1] + 0.5 * step_s, step_s)
    orbit = gn.Ephemeris(shell=shell_of(like))
    pos = [orbit.position(w, s) for s in sows]
    vel = [(orbit.position(w, s + 0.5) - orbit.position(w, s - 0.5)) for s in sows]
    bias = eph.clock(w, toe + 0.5 * (ARC[0] + ARC[1]))
    r = {k: v for k, v in like.items() if k not in ("pos", "vel")}
    r.update(
        prn=prn,
        clk_s=bias + offset_s,
        t0_gpst=gpst(week, sows[0]),
        t1_gpst=gpst(week, sows[-1]),
        dt=step_s,
        toe_gpst=gpst(week, toe),
        pos=[[float(x) for x in p] for p in pos],
        vel=[[float(x) for x in v] for v in vel],
        source="nav",
        clk_sigma_s=CLK_SIGMA_NAV_S,
        nav_bias_s=bias,
        nav_drift=w["af1"],
    )
    return r


def unix_of(sow, t_now):
    """The unix time of BDT second of week `sow`, in the week up to t_now."""
    now_sow = (t_now - 315964800 + 18 + BDT_GPST) % 604800  # GPST = UTC + 18 s
    return t_now - (now_sow - sow) % 604800


def cut(rec, t0, t1):
    """The parts of a gridded record outside [t0, t1], as records on its own grid (at least two
    points each)."""
    out = []
    g0, dt, n = rec["t0_gpst"], rec["dt"], len(rec["pos"])
    tt = [g0 + k * dt for k in range(n)]
    for keep in (
        [k for k in range(n) if tt[k] <= t0],
        [k for k in range(n) if tt[k] >= t1],
    ):
        if len(keep) < 2:
            continue
        a, b = keep[0], keep[-1]
        r = dict(rec, pos=rec["pos"][a : b + 1], vel=rec["vel"][a : b + 1])
        r["t0_gpst"] = tt[a]
        r["t1_gpst"] = min(rec["t1_gpst"], tt[b])
        out.append(r)
    return out


def build(tle, nav):
    """The TLE table with the nav records in and the TLE records cut around them."""
    doc = copy.deepcopy(tle)
    spans = {}
    for r in nav:
        spans.setdefault(r["prn"], []).append((r["t0_gpst"], r["t1_gpst"]))
    recs = []
    for r in doc["records"]:
        r["clk_sigma_s"] = CLK_SIGMA_TLE_S
        parts = [r]
        for t0, t1 in spans.get(r["prn"], []):
            parts = [q for p in parts for q in cut(p, t0, t1)]
        recs += parts
    doc["records"] = recs + nav
    doc["nav_records"] = len(nav)
    return doc


def write(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tle-table", required=True, help="gw_ephem_table.py's output")
    ap.add_argument("--out", required=True, help="the broker's gw-table")
    ap.add_argument("--symbols", default="http://localhost:12048/gw_symbols")
    ap.add_argument(
        "--clock-offset-us",
        type=float,
        default=0.0,
        help="the receiver's GuoWang code against its GPS clock: +0.47 on the bladeRF",
    )
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument(
        "--min-snr", type=float, default=8.0, help="symbol SNR to decode, dB"
    )
    ap.add_argument("--step-s", type=float, default=10.0)
    ap.add_argument("--log", help="append each decoded subframe here (jsonl)")
    ap.add_argument(
        "--decode-health-file", help="the viewer's decode-health JSON for this decoder"
    )
    ap.add_argument(
        "--progress-file", help="subframes decoded per code, as nav_decode's"
    )
    a = ap.parse_args()
    dhw = decode_health.DecodeHealthWriter(
        a.decode_health_file, chain="gw_1202", sys="W", log=print
    )
    n_sf = {}  # prn -> subframes decoded
    ev = {}  # prn -> [(unix time the subframe ended, 1)]
    t_start = time.time()
    offset_s = a.clock_offset_us * 1e-6
    ephs, seen = {}, set()
    tle, tle_mtime, written = None, None, None
    logf = open(a.log, "a") if a.log else None
    while True:
        try:
            m = os.stat(a.tle_table).st_mtime_ns
            if m != tle_mtime:
                with open(a.tle_table) as fh:
                    tle = json.load(fh)
                tle_mtime = m
        except (OSError, ValueError) as e:
            print("TLE table unreadable: %s" % e, flush=True)
        try:
            rows = json.load(
                urllib.request.urlopen(a.symbols + "/get_status", timeout=5)
            )
        except Exception as e:
            print("symbols unreadable: %s" % e, flush=True)
            rows = []
        synced = set()
        for row in rows:
            prn = row["prn"]
            if row["snr_db"] < a.min_snr:
                continue
            synced.add(prn)
            sym = np.array([b[1] for b in row["nav_obs"]["bits"]], dtype=float)
            for sf in gn.decode_subframes(sym):
                if (prn, sf.sow) in seen:
                    continue
                seen.add((prn, sf.sow))
                n_sf[prn] = n_sf.get(prn, 0) + 1
                ev.setdefault(prn, []).append(
                    (round(unix_of(sf.sow + 3, time.time()), 2), 1)
                )
                # The shell is the sender's, found once an issue is complete (nav_record).
                if not is_fill(sf):
                    ephs.setdefault(prn, gn.Ephemeris(shell=50)).add(sf)
                if logf:
                    logf.write(
                        json.dumps(
                            dict(
                                prn=prn,
                                page=sf.page,
                                sow=sf.sow,
                                payload="".join(str(int(b)) for b in sf.payload),
                                t_unix=time.time(),
                            )
                        )
                        + "\n"
                    )
                    logf.flush()
        if a.progress_file:
            t = time.time()
            for prn in ev:
                ev[prn] = [e for e in ev[prn] if t - e[0] < PROGRESS_KEEP_S]
            prns = {str(p): {"n": n_sf[p], "ev": ev[p]} for p in sorted(n_sf)}
            write(
                {
                    "schema": 1,
                    "t": round(t, 2),
                    "signals": {"GW_1202": {"sys": "W", "prns": prns}},
                },
                a.progress_file,
            )
        if tle is None:
            time.sleep(a.every)
            continue
        # Every complete issue whose arc reaches from 5 minutes ago into the future.
        now_sow = (
            time.time() - 315964800 + 18 + BDT_GPST
        ) % 604800  # GPST = UTC + 18 s
        nav, keys = [], []
        recs = {}
        for r in tle["records"]:
            recs.setdefault(r["prn"], []).append(r)
        for prn, eph in sorted(ephs.items()):
            for toe in sorted(eph.pages):
                if not all(k in eph.pages[toe] for k in (1, 2, 3, 4)):
                    continue
                if not (toe + ARC[0] - 300 <= now_sow <= toe + ARC[1]):
                    continue
                w = eph.words(toe)
                like = sender(w, recs.get(prn, []))
                if like is None:
                    continue
                nav.append(nav_record(prn, eph, w, offset_s, a.step_s, like))
                keys.append((prn, toe))
        with_arc = {k[0] for k in keys}
        for prn in n_sf:
            dhw.observe(
                "GW_1202",
                prn,
                time.time(),
                count=n_sf[prn],
                synced=prn in synced,
                eph=prn in with_arc,
            )
        dhw.flush(time.time())
        state = (tle_mtime, tuple(keys))
        if state != written and (
            written is not None or time.time() - t_start >= FIRST_WRITE_S
        ):
            write(build(tle, nav), a.out)
            written = state
            print(
                "%s wrote %s: %d nav arc(s) %s"
                % (
                    time.strftime("%H:%M:%S"),
                    a.out,
                    len(nav),
                    " ".join("%d@%d" % k for k in keys),
                ),
                flush=True,
            )
        # Forget subframes and issues more than 20 minutes old.
        for prn in ephs:
            for toe in [t for t in ephs[prn].pages if t < now_sow - 1200]:
                del ephs[prn].pages[toe]
        seen = {k for k in seen if k[1] >= now_sow - 1200}
        time.sleep(a.every)


if __name__ == "__main__":
    main()
