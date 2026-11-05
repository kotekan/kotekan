#!/usr/bin/env python3
"""Drive cudaDtvSubtract's rolling fit (kv/dtv-rolling 539d93615) on live nodes, both GPUs.

    dtv_roll.py --nodes cx51 arm [--gpus 0] [--tau 1.5 4 8]     # enable (the sums restart)
    dtv_roll.py --nodes cx51 disarm [--gpus 0]
    dtv_roll.py --nodes cx51 freeze 0 1 1                        # per source: 1 = hold g
    dtv_roll.py --nodes cx19,cx51 status
    dtv_roll.py --nodes cx51 save            # the state in use -> STATES/<node>/st_g<g>_roll<HHMMSS>.bin, current
    dtv_roll.py --nodes cx19,cx27,... watch [--every 60] [--save-every 600] [--taua 25] [--rearm] [--restore]

watch: one status line per GPU each --every s; saves each --save-every s (keeps the last 3 per node, so a
restarted node's restore loads a g at most minutes old); freezes sources 1.. while |Tau A hour angle| < --taua
min; with --rearm, re-enables a GPU that subtracts with the rolling fit off (a node restart loses it); with
--restore, first loads the node's current state on a GPU in pass-through 600..30000 frames after a restart.
Saves go through the node's local /tmp/gnss/dtvroll (kotekan is root; /home is root-squashed NFS).
REST calls only: safe on the VM.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

STATES = os.environ.get("DTV_STATES", "/home/kvand/gnss/builds/dtv-live/states-rev2")
NODE_DIR = "/tmp/gnss/dtvroll"


def url(node, g, ep):
    return "http://%s:12048/run_send_voltage/gpu_%d/commands/0/%s" % (node, g, ep)


def get(node, g, ep="status"):
    try:
        with urllib.request.urlopen(url(node, g, ep), timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001 - a node that does not answer is reported, not fatal
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
        return {"error": e.code, "message": e.read().decode()[:200]}
    except Exception as e:  # noqa: BLE001
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


def save(node, gpus):
    # kotekan runs as root and /home is root-squashed NFS: the node writes to its local NODE_DIR, we pull the file.
    tag = "roll" + time.strftime("%H%M%S", time.gmtime())
    ok = True
    subprocess.run(["ssh", node, "mkdir -p " + NODE_DIR], check=False)
    for g in gpus:
        name = "st_g%d_%s.bin" % (g, tag)
        p = os.path.join(STATES, node, name)
        r = post(node, g, "rolling", {"save": NODE_DIR + "/" + name})
        if "error" not in r:
            subprocess.run(
                ["rsync", "-a", "%s:%s/%s" % (node, NODE_DIR, name), p], check=False
            )
            subprocess.run(["ssh", node, "rm -f %s/%s" % (NODE_DIR, name)], check=False)
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
        for p0 in old:
            for g in (0, 1):
                q = p0.replace("st_g0_", "st_g%d_" % g)
                if os.path.exists(q):
                    os.remove(q)
    return tag if ok else None


def taua_ha_min():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vis_dtv_plot import hour_angle_min  # skyfield; imported only when needed

    return float(hour_angle_min(time.time() * 1e9))


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
        "--taua",
        type=float,
        default=25.0,
        help="freeze sources 1.. while |HA| < this (min); 0: never",
    )
    p.add_argument("--rearm", action="store_true")
    p.add_argument(
        "--restore",
        action="store_true",
        help="load STATES/<node>/current on a restarted pass-through GPU",
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
        prev, last_save = {}, time.time()
        while True:
            want = None
            if a.taua > 0:
                ha = taua_ha_min()
                want = 1 if abs(ha) < a.taua else 0
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
                    subtracting = any(2 in c["act"] for c in s["channels"])
                    if a.rearm and subtracting and not r["enable"]:
                        print(
                            stamp(),
                            n,
                            "g%d" % g,
                            "re-arm:",
                            post(n, g, "rolling", {"enable": True}).get("enable"),
                            flush=True,
                        )
                    if want is not None:
                        fz = [0] + [want] * (len(r["freeze"]) - 1)
                        if r["freeze"] != fz:
                            print(
                                stamp(),
                                n,
                                "g%d" % g,
                                "Tau A HA %.1f min: freeze %s ->" % (ha, fz),
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
