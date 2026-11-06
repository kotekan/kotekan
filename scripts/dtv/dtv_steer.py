#!/usr/bin/env python3
"""Steer cudaDtvSubtract's sources onto emitters that move across the sky: satellites from TLEs.

    dtv_steer.py --nodes cx51 --freq 1845-1867,1871-1892,1897-1918,1922-1943 anchor [--sources 0,1,2]
    dtv_steer.py --nodes cx51 --freq ... identify [--minutes 15] [--every 120]
    dtv_steer.py --nodes cx19,cx27,... --freq ... run [--every 10]
    dtv_steer.py --nodes cx19,cx27,... --freq ... status

anchor:   eigen starts of the sources on the target's channels only, in order, each subtracting with a
          zero g until the rolling fit refits it (the rolling fit must be on); waits --settle s between
          sources, so each start sees the residual after the sources before it.
identify: rolling saves every --every s for --minutes; per (GPU, channel, source) the per-input phase
          drift of g (its strongest lag) between the first and last save, against each satellite's
          predicted turn: assigned when the best coherence is >= --min-coh and leaves at most --ratio
          of the incoherence (1 - coherence) of the next satellite and of no motion, and the drift is
          large enough to judge (no motion's incoherence >= 0.01).
run:      every --every s, POST /steer for each assigned source: its turn since the frame its weights
          were set (/status anchor_seq), at the latest frame, and its rate over the next 30 s. A source
          whose weights were set again since it was identified loses its assignment.

Geometry: input i (pol * 64 + dish) sits at its dish's position (--positions, ENU m). An emitter at
topocentric unit vector s(t) turns input i's response by psi_i = SIGN 2 pi f / c r_i . (s(t) - s(t_a)),
f the channel's centre; SIGN = +1 was checked on recv1's N2 on 11-05 (MUOS-5's mode over 6 h, coherence
0.99). Times: seq is hops of 5.12 us since the F-engine's frame0 (--frame0, the latest entry), checked
against the wall clock. Assignments are kept in WORK/steer_assign.json.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnss")
)
import dtv_state  # noqa: E402

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
WORK = os.environ.get("DTV_STEER_WORK", "/var/tmp/kvand-dtvsteer")
NODE_DIR = "/tmp/gnss/dtvroll"
HOP_S = 5.12e-6
SIGN = +1.0
C = 299792458.0
SITE = wgs84.latlon(49.3199845, -119.62262691, 528.4)
TS = load.timescale()


def stamp():
    return time.strftime("%H:%M:%SZ", time.gmtime())


def fid_hz(fid):
    return (300.0 + (fid - 1536) * 0.1953125) * 1e6


def ranges(spec):
    out = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def url(node, g, ep):
    return "http://%s:12048/run_send_voltage/gpu_%d/commands/0/%s" % (node, g, ep)


def get(node, g, ep="status"):
    try:
        with urllib.request.urlopen(url(node, g, ep), timeout=8) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": str(e)}


def post(node, g, ep, body):
    req = urllib.request.Request(
        url(node, g, ep),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return {"error": e.code, "message": e.read().decode()[:300]}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": str(e)}


class Clock:
    """seq <-> UTC from the F-engine's frame0, checked against a node's latest frame and the wall clock."""

    def __init__(self, path):
        e = json.load(open(path))[-1]
        self.f0 = e["frame0_int"] + e["frame0_frac"]
        self.start = e.get("start_utc")

    def utc(self, seq):
        return self.f0 + seq * HOP_S

    def seq(self, unix):
        return int(round((unix - self.f0) / HOP_S))

    def check(self, seq_now):
        d = self.utc(seq_now) - time.time()
        if abs(d) > 30.0:
            raise SystemExit(
                "frame0 %.6f (from %s) puts the latest frame %.0f s from the wall clock: an F-engine "
                "re-base not yet in the frame0 file?" % (self.f0, self.start, d)
            )
        return d


class Sky:
    def __init__(self, tle):
        lines = [l.rstrip() for l in open(tle) if l.strip()]
        self.sats = {
            lines[i].strip(): EarthSatellite(
                lines[i + 1], lines[i + 2], lines[i].strip(), TS
            )
            for i in range(0, len(lines) - 2, 3)
        }

    def enu(self, name, unix):
        t = TS.from_datetime(
            datetime.datetime.fromtimestamp(unix, datetime.timezone.utc)
        )
        alt, az, _ = (self.sats[name] - SITE).at(t).altaz()
        a, z = np.radians(alt.degrees), np.radians(az.degrees)
        return np.array([np.cos(a) * np.sin(z), np.cos(a) * np.cos(z), np.sin(a)])

    def up(self, unix):
        return [n for n in self.sats if self.enu(n, unix)[2] > 0]


def positions(path):
    pos = json.load(open(path))["positions_by_dish"]
    return np.array([pos[str(i % 64)] for i in range(128)], float)


def turn(r, fid, s0, s1):
    return SIGN * 2 * np.pi * fid_hz(fid) / C * (r @ (s1 - s0))


def load_assign():
    p = os.path.join(WORK, "steer_assign.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_assign(a):
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, "steer_assign.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(a, fh, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def key(node, g, fid, k):
    return "%s:%d:%d:%d" % (node, g, fid, k)


def targets(st, fids):
    """{freq_id: channel} of the GPU's channels in the target."""
    return {c["freq_id"]: c for c in st.get("channels", []) if c["freq_id"] in fids}


def cmd_anchor(a, fids):
    srcs = [int(v) for v in a.sources.split(",")]
    for node in a.nodes:
        for g in (0, 1):
            st = get(node, g)
            if "error" in st:
                print("%s %s g%d: %s" % (stamp(), node, g, st["error"]))
                continue
            if not st.get("rolling", {}).get("enable"):
                print(
                    "%s %s g%d: rolling fit off; a re-pointed source would keep a zero g: skipped"
                    % (stamp(), node, g)
                )
                continue
            ch = targets(st, fids)
            for k in srcs:
                ids = sorted(f for f, c in ch.items() if k < c["sources"])
                if not ids:
                    continue
                r = post(
                    node,
                    g,
                    "start",
                    {
                        "source": k,
                        "frames": a.frames,
                        "freq_ids": ids,
                        "subtract": True,
                    },
                )
                print(
                    "%s %s g%d source %d on %d channels: %s"
                    % (stamp(), node, g, k, len(ids), r),
                    flush=True,
                )
                if "error" in r:
                    break
                while get(node, g).get("start_frames_left", 0) > 0:
                    time.sleep(0.5)
                time.sleep(a.settle)


def pull_save(node, g, tag, clock):
    """A rolling save of GPU g, copied to WORK/identify/<node>/; (path, seq at the save)."""
    name = "steer_g%d_%s.bin" % (g, tag)
    d = os.path.join(WORK, "identify", node)
    os.makedirs(d, exist_ok=True)
    subprocess.run(["ssh", node, "mkdir -p " + NODE_DIR], check=False)
    r = post(node, g, "rolling", {"save": NODE_DIR + "/" + name})
    # a stage before the steering one reports no seq: the wall clock, to ~1 s
    seq = get(node, g).get("seq", clock.seq(time.time()))
    p = os.path.join(d, name)
    if "error" in r:
        return None, seq
    subprocess.run(["rsync", "-a", "%s:%s/%s" % (node, NODE_DIR, name), p], check=False)
    subprocess.run(["ssh", node, "rm -f %s/%s" % (NODE_DIR, name)], check=False)
    return (p if os.path.exists(p) else None), seq


def cmd_identify(a, fids, clock, sky, r):
    n = max(2, int(a.minutes * 60 / a.every) + 1)
    snaps = {}  # (node, g) -> [(path, seq)]
    status, first = {}, {}  # the latest and the first /status of each GPU
    for q in range(n):
        tag = "%s_%02d" % (time.strftime("%H%M%S", time.gmtime()), q)
        for node in a.nodes:
            for g in (0, 1):
                st = get(node, g)
                if "error" in st:
                    continue
                status[(node, g)] = st
                first.setdefault((node, g), st)
                p, seq = pull_save(node, g, tag, clock)
                if p:
                    snaps.setdefault((node, g), []).append((p, seq))
        print("%s save %d of %d" % (stamp(), q + 1, n), flush=True)
        if q + 1 < n:
            time.sleep(a.every)
    asg = load_assign()
    for (node, g), sv in sorted(snaps.items()):
        if len(sv) < 2:
            continue
        (p0, q0), (p1, q1) = sv[0], sv[-1]
        clock.check(q1)
        t0, t1 = clock.utc(q0), clock.utc(q1)
        s0 = {n_: sky.enu(n_, t0) for n_ in sky.up(t0)}
        s1 = {n_: sky.enu(n_, t1) for n_ in s0}
        st0, st1 = dtv_state.read_state(p0), dtv_state.read_state(p1)
        st = status[(node, g)]
        if st["frames"] < first[(node, g)]["frames"]:
            print(
                "%s %s g%d restarted during the window: not identified"
                % (stamp(), node, g)
            )
            continue
        ch_first = targets(first[(node, g)], fids)
        for fid, ch in sorted(targets(st, fids).items()):
            steered = ch.get(
                "steered", [0] * ch["sources"]
            )  # absent before the steering stage
            anchor = ch.get("anchor_seq", [-1] * ch["sources"])
            anchor0 = ch_first.get(fid, ch).get("anchor_seq", anchor)
            for k in range(ch["sources"]):
                if (
                    ch["act"][k] < 2
                    or steered[k]
                    or anchor0[k] != anchor[k]
                    or fid not in st0["recs"]
                    or fid not in st1["recs"]
                ):
                    continue
                g0, g1 = st0["recs"][fid]["g"][k], st1["recs"][fid]["g"][k]
                j = int(np.argmax((np.abs(g1) ** 2).sum(axis=1)))
                x = g1[j] * np.conj(g0[j])
                w = np.abs(x)
                if w.sum() == 0:
                    continue
                ph = x / np.maximum(w, 1e-30)
                none = abs(np.sum(w * ph)) / w.sum()
                coh = sorted(
                    (
                        (
                            abs(
                                np.sum(
                                    w * ph * np.exp(-1j * turn(r, fid, s0[n_], s1[n_]))
                                )
                            )
                            / w.sum(),
                            n_,
                        )
                        for n_ in s0
                    ),
                    reverse=True,
                )
                best, name = coh[0]
                nxt = coh[1][0] if len(coh) > 1 else 0.0
                ok = (
                    best >= a.min_coh
                    and 1 - none >= 0.01
                    and 1 - best <= a.ratio * (1 - nxt)
                    and 1 - best <= a.ratio * (1 - none)
                )
                kk = key(node, g, fid, k)
                print(
                    "%s %s g%d %d src %d: %s (no motion %.3f)%s"
                    % (
                        stamp(),
                        node,
                        g,
                        fid,
                        k,
                        " ".join("%s %.3f" % (n_, c_) for c_, n_ in coh),
                        none,
                        " -> " + name if ok else "",
                    ),
                    flush=True,
                )
                if ok:
                    asg[kk] = dict(
                        sat=name,
                        anchor_seq=anchor[k],
                        coherence=round(best, 3),
                        when=stamp(),
                    )
                else:
                    asg.pop(kk, None)
    save_assign(asg)


def cmd_run(a, fids, clock, sky, r):
    while True:
        asg = load_assign()
        changed = False
        for node in a.nodes:
            for g in (0, 1):
                st = get(node, g)
                if "error" in st:
                    print("%s %s g%d: %s" % (stamp(), node, g, st["error"]), flush=True)
                    continue
                if "seq" not in st:
                    raise SystemExit(
                        "%s g%d: the stage has no /steer (a binary before the steering stage)"
                        % (node, g)
                    )
                seq = st["seq"]
                clock.check(seq)
                t = clock.utc(seq)
                by_src = {}
                for fid, ch in targets(st, fids).items():
                    for k in range(ch["sources"]):
                        e = asg.get(key(node, g, fid, k))
                        if not e:
                            continue
                        if ch["anchor_seq"][k] != e["anchor_seq"]:
                            print(
                                "%s %s g%d %d src %d: weights set again (seq %d, identified on %d): "
                                "assignment dropped"
                                % (
                                    stamp(),
                                    node,
                                    g,
                                    fid,
                                    k,
                                    ch["anchor_seq"][k],
                                    e["anchor_seq"],
                                ),
                                flush=True,
                            )
                            asg.pop(key(node, g, fid, k))
                            changed = True
                            continue
                        ta = clock.utc(e["anchor_seq"])
                        sa = sky.enu(e["sat"], ta)
                        p0 = turn(r, fid, sa, sky.enu(e["sat"], t))
                        p1 = turn(r, fid, sa, sky.enu(e["sat"], t + 30.0))
                        by_src.setdefault(k, []).append(
                            {
                                "freq_id": fid,
                                "phase": [float(v) for v in p0],
                                "rate": [float(v) for v in (p1 - p0) / 30.0],
                            }
                        )
                for k, chans in sorted(by_src.items()):
                    resp = post(
                        node,
                        g,
                        "steer",
                        {"source": k, "ref_seq": seq, "channels": chans},
                    )
                    rms = np.sqrt(
                        np.mean([np.mean(np.square(c_["phase"])) for c_ in chans])
                    )
                    print(
                        "%s %s g%d src %d: %d channels, turn rms %.2f rad%s"
                        % (
                            stamp(),
                            node,
                            g,
                            k,
                            len(chans),
                            rms,
                            "" if "error" not in resp else " ERROR %s" % resp,
                        ),
                        flush=True,
                    )
        if changed:
            save_assign(asg)
        if a.once:
            return
        time.sleep(a.every)


def cmd_status(a, fids):
    asg = load_assign()
    for node in a.nodes:
        for g in (0, 1):
            st = get(node, g)
            if "error" in st:
                print("%s g%d: %s" % (node, g, st["error"]))
                continue
            for fid, ch in sorted(targets(st, fids).items()):
                parts = []
                for k in range(ch["sources"]):
                    e = asg.get(key(node, g, fid, k), {})
                    parts.append(
                        "src%d act %d %s%s"
                        % (
                            k,
                            ch["act"][k],
                            "steered " if ch["steered"][k] else "",
                            e.get("sat", "-"),
                        )
                    )
                print("%s g%d %d: %s" % (node, g, fid, "; ".join(parts)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--nodes", required=True, type=lambda s: s.split(","))
    ap.add_argument(
        "--freq",
        default="1845-1867,1871-1892,1897-1918,1922-1943",
        help="the target's freq_ids",
    )
    ap.add_argument("--tle", default=os.path.join(WORK, "muos.tle"))
    ap.add_argument(
        "--positions",
        default=os.path.join(
            REPO, "config", "elem_positions_taua_20261024_by_dish.json"
        ),
    )
    ap.add_argument("--frame0", default="/home/kvand/gnss/builds/gw1202/frame0.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("anchor")
    p.add_argument("--sources", default="0,1,2")
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--settle", type=float, default=15.0)
    p = sub.add_parser("identify")
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--every", type=float, default=120.0)
    p.add_argument("--min-coh", type=float, default=0.9)
    p.add_argument("--ratio", type=float, default=0.3)
    p = sub.add_parser("run")
    p.add_argument("--every", type=float, default=10.0)
    p.add_argument("--once", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args()
    fids = ranges(a.freq)
    if a.cmd == "anchor":
        return cmd_anchor(a, fids)
    if a.cmd == "status":
        return cmd_status(a, fids)
    clock, sky, r = Clock(a.frame0), Sky(a.tle), positions(a.positions)
    return (cmd_identify if a.cmd == "identify" else cmd_run)(a, fids, clock, sky, r)


if __name__ == "__main__":
    main()
