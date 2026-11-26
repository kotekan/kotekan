#!/usr/bin/env python3
"""Drive cudaDtvSubtract's rolling fit (kv/dtv-rolling 539d93615) on live nodes, both GPUs.

    dtv_roll.py --nodes cx51 arm [--gpus 0] [--tau 1.5 4 8]     # enable (the sums restart)
    dtv_roll.py --nodes cx51 disarm [--gpus 0]
    dtv_roll.py --nodes cx51 freeze 0 1 1                        # per source: 1 = hold g
    dtv_roll.py --nodes cx19,cx51 status
    dtv_roll.py --nodes cx51 save            # the state in use -> STATES/<node>/st_g<g>_roll<HHMMSS>.bin, current
    dtv_roll.py --nodes cx19,cx27,... watch [--every 60] [--save-every 600] [--protect-min 25]
        [--protect NAME:RA:DEC ...] [--rearm] [--restore] [--min-explained F]
        [--pass-tle FILE [--pass-freq 1886-1893] [--pass-pad 60]]

watch: one status line per GPU each --every s; saves each --save-every s (keeps the last 3 per node, so a
restarted node's restore loads a g at most minutes old); freezes sources 1.. while a protected sky source
(--protect, J2000 degrees; default Tau A) is within --protect-min minutes of drift of the boresight (on the
meridian at --boresight-dec), so the blind sources do not fit it as it crosses the beam; with --pass-tle,
freezes every source on the GPUs holding a --pass-freq channel while a satellite of that file is above the
horizon, from --pass-pad s before its rise to as long after its set, so the fit does not follow the
satellite's burst and trip when it ends (CHUANGXIN 1-03/1-04 at 369 MHz); with --rearm, re-enables a GPU that subtracts with the rolling fit off (a node restart loses it); with
--restore, first loads the node's current state on a GPU in pass-through 600..30000 frames after a restart,
and records which satellite each loaded source follows and when its weights are valid for (from the save's
.json beside it, written at each save from dtv_steer's assignments) in dtv_steer's WORK/restored/, so
dtv_steer run steers them again without an identify; a save folded to the current F-engine start first when its
weights were made under an earlier one (dtv_fold.py: the start's sample steps), and a running GPU is saved, folded
and loaded again once the steps of its start are measured;
with --min-explained, sets the rolling fit's hold on a GPU that reports another value (a node restart brings
back its config's). Saves go through the node's local /tmp/gnss/dtvroll (kotekan cannot write /home).
REST calls only: safe on the VM.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnss")
)
import dtv_state  # noqa: E402

STATES = os.environ.get("DTV_STATES", "/home/kvand/gnss/builds/dtv-live/states-rev2")
NODE_DIR = "/tmp/gnss/dtvroll"
FRAME0 = os.environ.get("DTV_FRAME0", "/home/kvand/gnss/builds/gw1202/frame0.json")
# a node that dies mid-save must not hang the watcher
SSH = [
    "ssh",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=3",
]


def url(node, g, ep):
    return "http://%s:12048%s/%s" % (node, dtv_state.stage_path(node, g), ep)


def get(node, g, ep="status"):
    try:
        with urllib.request.urlopen(url(node, g, ep), timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001 - a node that does not answer is reported, not fatal
        dtv_state.forget_stage(node, g)
        return {"error": str(e)[:120]}


def post(node, g, ep, body):
    req = urllib.request.Request(
        url(node, g, ep),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        dtv_state.forget_stage(node, g)
        return {"error": e.code, "message": e.read().decode()[:200]}
    except Exception as e:  # noqa: BLE001
        dtv_state.forget_stage(node, g)
        return {"error": str(e)[:120]}


def stamp():
    return time.strftime("%H:%M:%SZ", time.gmtime())


def line(node, g, s, prev=None):
    if "error" in s:
        return "%s g%d: no answer (%s)" % (node, g, s["error"])
    r = s.get("rolling")
    if r is None:
        return "%s g%d: no rolling fit in this binary" % (node, g)
    sub = sum(1 for c in s["channels"] if 2 in c["act"])
    d = (
        ""
        if prev is None or "rolling" not in prev
        else " (+%d)" % (r["refits"] - prev["rolling"]["refits"])
    )
    ch = []
    for c in r["channels"]:
        flag = "T" if c.get("tripped") else ("W" if c.get("weak") else "")
        pr, ex = c.get("power_ratio"), c.get("explained0")
        ch.append(
            "%d:%s/%s%s"
            % (
                c["freq_id"],
                "%.3f" % pr if pr is not None and pr >= 0 else "-",
                "%.3f" % ex if ex is not None else "-",
                flag,
            )
        )
    return (
        "%s g%d %s refits %d%s holds %d trips %d failed %d %.1f ms tau %s freeze %s subtracting %d/%d | res/raw / expl0: %s"
        % (
            node,
            g,
            "ON " if r["enable"] else "off",
            r["refits"],
            d,
            r["holds"],
            r["trips"],
            r["failed_fits"],
            r["refit_ms"],
            r["tau_s"],
            r["freeze"],
            sub,
            len(s["channels"]),
            " ".join(ch),
        )
    )


def save_steer(node, g, p, prev):
    """Beside the save p: the satellite of each steered source and the time its weights are valid for, and the
    F-engine start (frame0_int) each source's weights were made under: the restore's or the previous save's (prev)
    while its anchor is the same, else the current one."""
    import dtv_fold
    import dtv_steer  # skyfield: only where the steering runs

    st = get(node, g)
    src = (
        dtv_steer.valid_sources(
            node, g, st, dtv_steer.load_assign(), dtv_steer.Clock(FRAME0)
        )
        if "error" not in st
        else {}
    )
    d = dtv_fold.frame0_now(FRAME0)
    rec = dtv_steer.restored(node, g)
    epochs, anchor = {}, {}
    for c in st.get("channels", []):
        for k in range(c["sources"]):
            kk = "%d:%d" % (c["freq_id"], k)
            a = anchor[kk] = c["anchor_seq"][k]
            if rec.get("anchor", {}).get(kk) == a and kk in rec.get("epochs", {}):
                epochs[kk] = rec["epochs"][kk]
            elif prev.get("anchor", {}).get(kk) == a and kk in prev.get("epochs", {}):
                epochs[kk] = prev["epochs"][kk]
            else:
                epochs[kk] = d
    with open(p[:-4] + ".json", "w") as fh:
        json.dump(
            dict(
                seq=st.get("seq"),
                frame0_int=d,
                sources=src,
                epochs=epochs,
                anchor=anchor,
            ),
            fh,
            indent=1,
            sort_keys=True,
        )


def side(node, g, tag):
    q = os.path.join(STATES, node, "st_g%d_%s.json" % (g, tag))
    return json.load(open(q)) if os.path.exists(q) else {}


def fold_for_load(p):
    """(p, or a copy of it folded to the current F-engine start, what was done)."""
    import dtv_fold

    q = p[:-4] + ".json"
    d = dtv_fold.frame0_now(FRAME0)
    if not os.path.exists(q) or not dtv_fold.stale(json.load(open(q)), d):
        return p, ""
    out = "%s_f%d.bin" % (p[:-4], d)
    n, skipped = dtv_fold.fold_file(p, out, d)
    why = "; %d left (%s)" % (len(skipped), skipped[0][1]) if skipped else ""
    if not n:
        for f in (out, out[:-4] + ".json"):
            os.remove(f)
        return p, "nothing folded to %d%s" % (d, why)
    return out, "folded %d sources to %d%s" % (n, d, why)


def refold(node, g, tried):
    """A subtracting GPU whose weights were made under an earlier F-engine start: once that start's steps are
    measured, save what runs, fold it and load it (at most once per start and steps file)."""
    import dtv_fold

    try:
        tag = open(os.path.join(STATES, node, "current")).read().strip()
    except OSError:
        return
    d = dtv_fold.frame0_now(FRAME0)
    if not dtv_fold.stale(side(node, g, tag), d):
        return
    m = os.path.getmtime(dtv_fold.STEPS) if os.path.exists(dtv_fold.STEPS) else 0
    if (node, g, d, m) in tried:
        return
    tried.add((node, g, d, m))
    t = save(node, [g])
    if not t:
        return
    p, why = fold_for_load(os.path.join(STATES, node, "st_g%d_%s.bin" % (g, t)))
    print(
        stamp(),
        node,
        "g%d" % g,
        "weights from an earlier F-engine start:",
        why,
        flush=True,
    )
    if p.endswith("_f%d.bin" % d):
        print(
            stamp(),
            node,
            "g%d" % g,
            "load ->",
            post(node, g, "load", {"file": p}),
            flush=True,
        )
        time.sleep(2)
        print(
            stamp(),
            node,
            "g%d" % g,
            "steering kept for %d sources" % note_restore(node, g, p, get(node, g)),
            flush=True,
        )


def note_restore(node, g, p, st):
    """WORK/restored/<node>_g<g>.json: the saved sources of p, with the anchor_seq the load gave them."""
    import dtv_steer

    q = p[:-4] + ".json"
    sd = json.load(open(q)) if os.path.exists(q) else {}
    src = sd.get("sources", {})
    anchors = {
        "%d:%d" % (c["freq_id"], k): c["anchor_seq"][k]
        for c in st.get("channels", [])
        for k in range(c["sources"])
    }
    out = {
        kk: dict(v, anchor_seq=anchors[kk]) for kk, v in src.items() if kk in anchors
    }
    d = os.path.join(dtv_steer.WORK, "restored")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "%s_g%d.json" % (node, g))
    with open(f + ".tmp", "w") as fh:
        json.dump(
            dict(
                file=p,
                when=stamp(),
                sources=out,
                anchor=anchors,
                epochs={
                    kk: e for kk, e in sd.get("epochs", {}).items() if kk in anchors
                },
            ),
            fh,
            indent=1,
            sort_keys=True,
        )
    os.replace(f + ".tmp", f)
    return len(out)


def save(node, gpus):
    # kotekan cannot write /home (root-squashed NFS, or another account): the node writes to its local NODE_DIR,
    # world-writable so either account works, and we pull the file.
    tag = "roll" + time.strftime("%H%M%S", time.gmtime())
    ok = True
    try:
        last = open(os.path.join(STATES, node, "current")).read().strip()
    except OSError:
        last = None
    subprocess.run(
        SSH + [node, "mkdir -p %s; chmod 0777 %s" % (NODE_DIR, NODE_DIR)], check=False
    )
    for g in gpus:
        name = "st_g%d_%s.bin" % (g, tag)
        p = os.path.join(STATES, node, name)
        r = post(node, g, "rolling", {"save": NODE_DIR + "/" + name})
        if "error" not in r:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "-e",
                    " ".join(SSH),
                    "%s:%s/%s" % (node, NODE_DIR, name),
                    p,
                ],
                check=False,
            )
            subprocess.run(SSH + [node, "rm -f %s/%s" % (NODE_DIR, name)], check=False)
            if os.path.exists(p):
                try:
                    save_steer(node, g, p, side(node, g, last) if last else {})
                except Exception as e:  # noqa: BLE001 - the save stands without it; a restore then needs an identify
                    print(
                        "%s %s g%d steering note failed: %s" % (stamp(), node, g, e),
                        flush=True,
                    )
        if "error" in r or not os.path.exists(p):
            print("%s %s g%d save failed: %s" % (stamp(), node, g, r), flush=True)
            ok = False
    if ok and set(gpus) == {0, 1}:
        with open(os.path.join(STATES, node, "current"), "w") as fh:
            fh.write(tag + "\n")
        old = sorted(
            glob.glob(os.path.join(STATES, node, "st_g0_roll*.bin")),
            key=os.path.getmtime,
        )[:-3]
        d = json.load(open(FRAME0))[-1]["frame0_int"]
        for p0 in old:
            # the last saves of an earlier F-engine start stay: the next start's steps are measured against them
            if side(node, 0, os.path.basename(p0)[6:-4]).get("frame0_int", d) != d:
                continue
            for g in (0, 1):
                q = p0.replace("st_g0_", "st_g%d_" % g)
                for f in (q, q[:-4] + ".json"):
                    if os.path.exists(f):
                        os.remove(f)
    return tag if ok else None


PROTECT = ["TauA:83.6331:22.0145"]  # NAME:RA:DEC, J2000 degrees
BORESIGHT_DEC = (
    22.02  # the dishes' declination on the meridian (Tau A at el 62.70, since 10-24)
)


def nearest_protected(specs, dec_b):
    """(name, degrees from the boresight, hour angle in minutes) of the protected source nearest the
    boresight now."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vis_dtv_plot import hour_angle_min, radec_of_date, sky_time  # skyfield

    best = None
    for spec in specs:
        name, ra, dec = spec.split(":")
        ha = float(hour_angle_min(time.time() * 1e9, float(ra), float(dec)))
        _, d = radec_of_date(float(ra), float(dec), sky_time(time.time()))
        cos_sep = math.sin(math.radians(d)) * math.sin(math.radians(dec_b)) + math.cos(
            math.radians(d)
        ) * math.cos(math.radians(dec_b)) * math.cos(math.radians(ha / 4.0))
        sep = math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))
        if best is None or sep < best[1]:
            best = (name, sep, ha)
    return best


def satellites_up(sky, t0, t1):
    """'NAME up (TLE d old)' for each satellite of sky (dtv_steer.Sky) above the horizon at any time
    in [t0, t1]."""
    up = sorted({n for t in range(int(t0), int(t1) + 30, 30) for n in sky.up(t)})
    return [
        "%s up (TLE %.1f d old)"
        % (n, (t0 - sky.sats[n].epoch.utc_datetime().timestamp()) / 86400)
        for n in up
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--nodes", required=True, help="comma list, e.g. cx19,cx51")
    ap.add_argument("--gpus", default="0,1", help="comma list")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arm")
    p.add_argument("--tau", nargs="+", type=float)
    sub.add_parser("disarm")
    p = sub.add_parser("freeze")
    p.add_argument("flags", nargs="+", type=int)
    sub.add_parser("status")
    sub.add_parser("save")
    p = sub.add_parser("watch")
    p.add_argument("--every", type=float, default=60.0)
    p.add_argument("--save-every", type=float, default=600.0)
    p.add_argument(
        "--protect-min",
        "--taua",
        dest="protect_min",
        type=float,
        default=25.0,
        help="freeze sources 1.. while a protected source is within this many minutes of drift of "
        "the boresight (25: Tau A's |HA| < 25 min); 0: never",
    )
    p.add_argument(
        "--protect",
        action="append",
        help="NAME:RA:DEC (J2000 degrees), repeatable; default %s" % PROTECT[0],
    )
    p.add_argument("--boresight-dec", type=float, default=BORESIGHT_DEC)
    p.add_argument("--rearm", action="store_true")
    p.add_argument(
        "--restore",
        action="store_true",
        help="load STATES/<node>/current on a restarted pass-through GPU",
    )
    p.add_argument(
        "--min-explained",
        type=float,
        help="keep the rolling fit's min_explained at this value",
    )
    p.add_argument(
        "--pass-tle",
        help="TLE file (name + 2 lines each) of satellites whose passes freeze every source",
    )
    p.add_argument(
        "--pass-freq", default="1886-1893", help="freq_ids of the GPUs a pass freezes"
    )
    p.add_argument(
        "--pass-pad", type=float, default=60.0, help="s before a rise and after a set"
    )
    a = ap.parse_args()
    a.nodes = a.nodes.split(",")
    a.gpus = [int(g) for g in a.gpus.split(",")]

    if a.cmd in ("arm", "disarm", "freeze"):
        body = {"arm": {"enable": True}, "disarm": {"enable": False}}.get(a.cmd, {})
        if a.cmd == "arm" and a.tau:
            body["tau_s"] = a.tau
        if a.cmd == "freeze":
            body["freeze"] = a.flags
        for n in a.nodes:
            for g in a.gpus:
                r = post(n, g, "rolling", body)
                print(
                    "%s %s g%d %s -> %s"
                    % (
                        stamp(),
                        n,
                        g,
                        json.dumps(body),
                        r
                        if "error" in r
                        else "enable %s tau %s freeze %s"
                        % (r["enable"], r["tau_s"], r["freeze"]),
                    )
                )
    elif a.cmd == "status":
        for n in a.nodes:
            for g in a.gpus:
                print(stamp(), line(n, g, get(n, g)))
    elif a.cmd == "save":
        for n in a.nodes:
            print(stamp(), n, "saved", save(n, a.gpus))
    elif a.cmd == "watch":
        prev, last_save, tried = {}, time.time(), set()
        sky = None
        if a.pass_tle:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from dtv_steer import Sky, ranges  # skyfield

            sky, pass_fids = Sky(a.pass_tle), ranges(a.pass_freq)
            sky_mtime = os.path.getmtime(a.pass_tle)
        while True:
            # the TLE file is refreshed in place by cron (tle_refresh.py, #235): re-read it when it
            # changes, and keep the old sets if the new file does not load
            if sky is not None:
                try:
                    m = os.path.getmtime(a.pass_tle)
                    if m != sky_mtime:
                        sky_mtime = m
                        sky = Sky(a.pass_tle)
                        print(
                            stamp(),
                            "pass TLE reloaded:",
                            ", ".join(sky.sats),
                            flush=True,
                        )
                except Exception as e:
                    print(
                        stamp(),
                        "pass TLE reload failed, kept the old sets:",
                        e,
                        flush=True,
                    )
            want = None
            if a.protect_min > 0:
                name, sep, ha = nearest_protected(a.protect or PROTECT, a.boresight_dec)
                radius = a.protect_min / 4.0 * math.cos(math.radians(a.boresight_dec))
                want = 1 if sep < radius else 0
            up = []
            if sky is not None:
                now = time.time()
                up = satellites_up(sky, now - a.pass_pad, now + a.every + a.pass_pad)
            for n in a.nodes:
                for g in a.gpus:
                    s = get(n, g)
                    print(stamp(), line(n, g, s, prev.get((n, g))), flush=True)
                    r = s.get("rolling")
                    if r is None:
                        continue
                    passthru = bool(s["channels"]) and all(
                        int(c["act"][0]) == 0 for c in s["channels"]
                    )
                    if a.restore and passthru and 600 <= s.get("frames", 0) <= 30000:
                        # a node that restarted minutes ago (a deliberate /clear on a long-running node is left alone)
                        tag = open(os.path.join(STATES, n, "current")).read().strip()
                        p = os.path.join(STATES, n, "st_g%d_%s.bin" % (g, tag))
                        try:
                            p, why = fold_for_load(p)
                            if why:
                                print(stamp(), n, "g%d" % g, why, flush=True)
                        except Exception as e:  # noqa: BLE001 - the unfolded save still beats pass-through
                            print(stamp(), n, "g%d" % g, "fold failed:", e, flush=True)
                        print(
                            stamp(),
                            n,
                            "g%d" % g,
                            "restarted (%d frames, pass-through): load %s ->"
                            % (s["frames"], tag),
                            post(n, g, "load", {"file": p}),
                            flush=True,
                        )
                        time.sleep(2)
                        s = get(n, g)
                        if "error" in s:
                            continue
                        try:
                            print(
                                stamp(),
                                n,
                                "g%d" % g,
                                "steering kept for %d sources"
                                % note_restore(n, g, p, s),
                                flush=True,
                            )
                        except Exception as e:  # noqa: BLE001
                            print(
                                stamp(),
                                n,
                                "g%d" % g,
                                "restore note failed:",
                                e,
                                flush=True,
                            )
                    subtracting = any(2 in c["act"] for c in s["channels"])
                    if a.restore and subtracting:
                        try:
                            refold(n, g, tried)
                        except Exception as e:  # noqa: BLE001 - the weights in use stay
                            print(
                                stamp(), n, "g%d" % g, "refold failed:", e, flush=True
                            )
                    if a.rearm and subtracting and not r["enable"]:
                        print(
                            stamp(),
                            n,
                            "g%d" % g,
                            "re-arm:",
                            post(n, g, "rolling", {"enable": True}).get("enable"),
                            flush=True,
                        )
                    me = a.min_explained
                    if me is not None and r.get("min_explained") != me:
                        print(
                            stamp(),
                            n,
                            "g%d" % g,
                            "min_explained %s ->" % r.get("min_explained"),
                            post(n, g, "rolling", {"min_explained": me}).get(
                                "min_explained"
                            ),
                            flush=True,
                        )
                    n_src, fz, why = len(r["freeze"]), None, []
                    if want is not None:
                        fz = [0] + [want] * (n_src - 1)
                        why.append(
                            "%s %.1f deg from the boresight (HA %+.1f min)"
                            % (name, sep, ha)
                        )
                    if sky is not None and any(
                        c["freq_id"] in pass_fids for c in r["channels"]
                    ):
                        if up:
                            fz, why = [1] * n_src, up
                        else:
                            fz, why = fz or [0] * n_src, ["no satellite up"] + why
                    if fz is not None and r["freeze"] != fz:
                        print(
                            stamp(),
                            n,
                            "g%d" % g,
                            "%s: freeze %s ->" % (", ".join(why), fz),
                            post(n, g, "rolling", {"freeze": fz}).get("freeze"),
                            flush=True,
                        )
                    prev[(n, g)] = s
            if time.time() - last_save >= a.save_every:
                last_save = time.time()
                for n in a.nodes:
                    if all(get(n, g).get("rolling", {}).get("enable") for g in (0, 1)):
                        print(stamp(), n, "saved", save(n, [0, 1]), flush=True)
            time.sleep(a.every)


if __name__ == "__main__":
    main()
