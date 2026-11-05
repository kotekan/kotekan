#!/usr/bin/env python3
"""Drive cudaDtvSubtract on a live node: eigen starts, output windows, fits and loads, per GPU.

    dtv_live.py --node cx51 status
    dtv_live.py --node cx51 window --seconds 20 --tag base [--beam]   # arm the dtv (and beam-dump) gates
    dtv_live.py --node cx51 start --source 0 [--frames 24]            # eigen start on both GPUs
    dtv_live.py --node cx51 fit --source 0 --tag s0 [--state-tag prev] [--freq g0:2903 g1:2909]
    dtv_live.py --node cx51 load --tag s0                             # load each GPU's state
    dtv_live.py --node cx51 clear [--source 0]
    dtv_live.py --node cx51 report --tag after                        # power removed + beam-dump lambda

Run on cf06. Output files: the node's /tmp/gnss/dtv (and beamdump), copied to WORK/<node>/<tag>/;
states WORK/<node>/states/st_g<g>_<tag>.bin (NFS, so the node loads them by path).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

GNSS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnss")
sys.path.insert(0, GNSS)
import numpy as np  # noqa: E402

import dtv_state  # noqa: E402

WORK = os.environ.get(
    "DTV_WORK", "/var/tmp/kvand-dtvlive"
)  # window copies (cf06 local)
STATES = os.environ.get(
    "DTV_STATES", "/home/kvand/gnss/builds/dtv-live/states"
)  # NFS: the nodes read them
PY = os.path.expanduser("~/gnss/venv/bin/python3")
ST = os.path.join(GNSS, "dtv_state.py")
BD = os.path.join(GNSS, "beam_dump_read.py")
HOPS = 195312.5


def url(a, path):
    return "http://%s:12048%s" % (a.node, path)


def get(a, path):
    with urllib.request.urlopen(url(a, path), timeout=10) as r:
        return json.loads(r.read())


def post(a, path, body):
    req = urllib.request.Request(
        url(a, path),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "message": e.read().decode()[:300]}


def cmd_path(g):
    return "/run_send_voltage/gpu_%d/commands/0" % g


def cmd_status(a):
    for g in (0, 1):
        s = get(a, cmd_path(g) + "/status")
        print(
            "gpu%d: frames %d, state %s, start source %d (%d left), last error %r; channels %s"
            % (
                g,
                s["frames"],
                s["state_file"] or "-",
                s["start_source"],
                s["start_frames_left"],
                s["last_error"],
                " ".join(
                    "%d:%+d..%+d:%s"
                    % (
                        c["freq_id"],
                        c["lag_min"],
                        c["lag_min"] + c["n_lag"] - 1,
                        c["act"],
                    )
                    for c in s["channels"]
                ),
            )
        )


def node_files(a, sub):
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", a.node, "ls /tmp/gnss/%s 2>/dev/null" % sub],
        capture_output=True,
        text=True,
    ).stdout
    return set(out.split())


def arm_gate(a, gate, seconds):
    st = get(a, "/%s/status" % gate)
    s = st["last_seq"] + int(0.5 * HOPS)
    e = s + int(seconds * HOPS)
    r = post(a, "/%s/arm" % gate, {"start_seq": s, "end_seq": e})
    return s, e, r


def cmd_window(a):
    sub = ["dtv"] + (["beamdump"] if a.beam else [])
    before = {d: node_files(a, d) for d in sub}
    gates = ["gnss%d_dtvdump_gate" % g for g in (0, 1)] + (
        ["gnss%d_beamdump_gate" % g for g in (0, 1)] if a.beam else []
    )
    if a.beam:
        for g in (0, 1):
            st = get(a, "/gnss%d_beamdump_gate/status" % g)
            if st["armed"] and st["end_seq"] > st["last_seq"]:
                print(
                    "gnss%d_beamdump_gate already armed [%d, %d) by the scheduler: not touching it"
                    % (g, st["start_seq"], st["end_seq"])
                )
                gates.remove("gnss%d_beamdump_gate" % g)
    for gate in gates:
        s, e, r = arm_gate(a, gate, a.seconds)
        print("%s armed [%d, %d): %s" % (gate, s, e, r.get("armed", r)))
    time.sleep(a.seconds + 6.0)
    d = os.path.join(WORK, a.node, a.tag)
    os.makedirs(d, exist_ok=True)
    for s_ in sub:
        new = sorted(node_files(a, s_) - before[s_])
        if not new:
            print("no new %s files" % s_)
            continue
        subprocess.run(
            ["rsync", "-a"]
            + ["%s:/tmp/gnss/%s/%s" % (a.node, s_, f) for f in new]
            + [d + "/"],
            check=True,
        )
        print("copied %d %s files to %s" % (len(new), s_, d))


def tag_files(a, tag, g, kind="dtv"):
    return sorted(
        glob.glob(
            os.path.join(
                WORK,
                a.node,
                tag,
                "%s_gnss%d_%s_*.raw" % (a.node, g, "dtv" if kind == "dtv" else "beam"),
            )
        )
    )


def gpus(a):
    """The GPUs a command acts on (setup and refit narrow it to those with the source slot)."""
    return getattr(a, "gpus", None) or (0, 1)


def cmd_start(a):
    for g in gpus(a):
        print(
            "gpu%d start: %s"
            % (
                g,
                post(
                    a, cmd_path(g) + "/start", {"source": a.source, "frames": a.frames}
                ),
            )
        )


def cmd_fit(a):
    os.makedirs(os.path.join(STATES, a.node), exist_ok=True)
    only = dict(x.split(":") for x in a.freq) if a.freq else {}
    fr = {}
    for g in gpus(a):
        files = tag_files(a, a.tag, g)
        if not files:
            print("gpu%d: no files under tag %s" % (g, a.tag))
            continue
        out = os.path.join(STATES, a.node, "st_g%d_%s.bin" % (g, a.tag))
        cmd = [PY, ST, "fit", "--source", str(a.source), "--out", out]
        if a.state_tag:
            cmd += [
                "--state",
                os.path.join(STATES, a.node, "st_g%d_%s.bin" % (g, a.state_tag)),
            ]
        if "g%d" % g in only:
            cmd += ["--freq", only["g%d" % g]]
        r = subprocess.run(cmd + files, capture_output=True, text=True)
        print("gpu%d:\n%s%s" % (g, r.stdout, r.stderr[-500:] if r.returncode else ""))
        fr[g] = [
            float(l.rsplit(" ", 1)[1])
            for l in r.stdout.splitlines()
            if "the model explains" in l
        ]
    return fr


def cmd_load(a):
    for g in gpus(a):
        p = os.path.join(STATES, a.node, "st_g%d_%s.bin" % (g, a.tag))
        print(
            "gpu%d load %s: %s"
            % (g, os.path.basename(p), post(a, cmd_path(g) + "/load", {"file": p}))
        )


def cmd_clear(a):
    for g in (0, 1):
        print(
            "gpu%d clear: %s"
            % (g, post(a, cmd_path(g) + "/clear", {"source": a.source}))
        )


def cmd_report(a):
    for g in (0, 1):
        files = tag_files(a, a.tag, g)
        if files:
            r = subprocess.run([PY, ST, "info"] + files, capture_output=True, text=True)
            print("gpu%d dtv output:\n%s" % (g, r.stdout))
        bfiles = tag_files(a, a.tag, g, "beam")
        if bfiles:
            sys.path.insert(0, os.path.dirname(BD))
            import beam_dump_read as bdr

            lam = {}
            for p in bfiles:
                for fr in bdr.frames(p):
                    for k, fid in enumerate(fr["freq_ids"]):
                        lam.setdefault(int(fid), []).append(fr["lam"][k][:2])
            dtv = [f for f in sorted(lam) if f < 3000]
            print(
                "gpu%d beam dump (ring after the stage), whitened lambda1/lambda2 median over %d frames: %s"
                % (
                    g,
                    len(lam[dtv[0]]) if dtv else 0,
                    "; ".join(
                        "%d: %.2f/%.2f" % (f, *np.median(np.array(lam[f]), axis=0))
                        for f in dtv
                    ),
                )
            )


def sub_args(a, **kw):
    b = argparse.Namespace(**vars(a))
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def current_tag(a, tag=None):
    p = os.path.join(STATES, a.node, "current")
    if tag is not None:
        with open(p, "w") as fh:
            fh.write(tag + "\n")
        return tag
    return open(p).read().strip() if os.path.exists(p) else None


def drop_window(a, tag):
    shutil.rmtree(os.path.join(WORK, a.node, tag), ignore_errors=True)


def gpu_slots(a):
    """{gpu: the stage's source slots}: the most any of its channels has (dtv_sources)."""
    out = {}
    for g in (0, 1):
        ch = get(a, cmd_path(g) + "/status")["channels"]
        out[g] = len(ch[0]["act"]) if ch else 1
    return out


def carry(a, g, old, new):
    """GPU g has no slot for this step: its state under the new tag is its state under the old one,
    so every tag (and `current`) names a file for both GPUs."""
    src = os.path.join(STATES, a.node, "st_g%d_%s.bin" % (g, old))
    if old and os.path.exists(src):
        shutil.copyfile(src, os.path.join(STATES, a.node, "st_g%d_%s.bin" % (g, new)))
        print("gpu%d: no source slot here; %s carried to %s" % (g, old, new))


def cmd_setup(a):
    """Eigen starts, fits and loads of every source slot in turn (or --sources N) on both GPUs, then a judging window.
    Source k's eigen start runs on the residual after sources 0..k-1, so each one is fitted before the next starts.
    A GPU without slot k (its channels' dtv_sources) skips it; a channel without it is left alone by the stage's
    eigen start and keeps its record in the fit."""
    t = a.prefix
    slots = gpu_slots(a)
    n = a.sources or max(slots.values())
    prev = None
    for k in range(n):
        if k:
            time.sleep(2)
        on = tuple(g for g in (0, 1) if k < slots[g])
        cmd_start(sub_args(a, source=k, frames=24, gpus=on))
        time.sleep(3)
        cmd_window(sub_args(a, seconds=a.seconds, tag=t + "s%d" % k, beam=False))
        cmd_fit(
            sub_args(a, source=k, tag=t + "s%d" % k, state_tag=prev, freq=None, gpus=on)
        )
        cmd_load(sub_args(a, tag=t + "s%d" % k, gpus=on))
        for g in set((0, 1)) - set(on):
            carry(a, g, prev, t + "s%d" % k)
        prev = t + "s%d" % k
    current_tag(a, prev)
    time.sleep(2)
    cmd_window(sub_args(a, seconds=a.seconds, tag=t + "judge", beam=False))
    cmd_report(sub_args(a, tag=t + "judge"))
    for k in range(n):
        drop_window(a, t + "s%d" % k)


def cmd_refit(a):
    """Refit source 0 (and source 1 with --both) from a fresh window, on the current state."""
    cur = current_tag(a)
    if cur is None:
        raise SystemExit("%s: no current state (run setup first)" % a.node)
    t = a.prefix
    cmd_window(sub_args(a, seconds=a.seconds, tag=t + "r0", beam=False))
    ps = power_summary(a, t + "r0")
    print(
        "%s %s before the refit of %s: power left %s"
        % (
            a.node,
            time.strftime("%H:%M:%SZ", time.gmtime()),
            cur,
            " ".join("%d:%.3f" % kv for kv in sorted(ps.items())),
        ),
        flush=True,
    )
    # Guards (11-03 22:10Z: after a node restart the refit skipped every channel, rewrote the previous state
    # and loaded it; its g had been fitted while the analog chains were off, and the stage multiplied the
    # DTV channels' power by 1e2..1e5 for 6 min).
    # (a) a stage in pass-through (restart, clear) is not refitted: it needs a setup
    for g in (0, 1):
        fr = list(dtv_state.frames(tag_files(a, t + "r0", g)))
        if not fr or not any(
            (f["act"][c][0] >= 1) for f in fr for c in range(len(f["freq"]))
        ):
            print(
                "%s gpu%d: source 0 inactive in the window (node restart or clear?): NOT refitting; run a setup"
                % (a.node, g),
                flush=True,
            )
            drop_window(a, t + "r0")
            return
    # (b) a stage that ADDS power is cleared to pass-through
    if ps and max(ps.values()) > 2.0:
        print(
            "%s ALARM: the stage multiplies the power by up to %.3g: clearing every source (pass-through)"
            % (a.node, max(ps.values())),
            flush=True,
        )
        for g in (0, 1):
            print(
                "gpu%d clear: %s" % (g, post(a, cmd_path(g) + "/clear", {"source": 0}))
            )
        drop_window(a, t + "r0")
        return
    fx = cmd_fit(sub_args(a, source=0, tag=t + "r0", state_tag=cur, freq=None))
    # (c) a source-0 fit that explains almost nothing is a fit to noise (sky/analog off): keep the loaded state
    allf = [v for g in fx for v in fx[g]]
    if not allf or float(np.median(allf)) < 0.1:
        print(
            "%s: source 0 explains a median %s of the power (analog off?): NOT loading"
            % (a.node, "%.3f" % np.median(allf) if allf else "none"),
            flush=True,
        )
        drop_window(a, t + "r0")
        return
    cmd_load(sub_args(a, tag=t + "r0"))
    cur = current_tag(a, t + "r0")
    drop_window(a, t + "r0")
    slots = gpu_slots(a)
    for k in ([1] if a.both else []) + [
        k for k in (a.extra or []) if not (a.both and k == 1)
    ]:
        on = tuple(g for g in (0, 1) if k < slots[g])
        if not on:
            continue
        time.sleep(2)
        cmd_window(sub_args(a, seconds=a.seconds, tag=t + "r%d" % k, beam=False))
        cmd_fit(
            sub_args(a, source=k, tag=t + "r%d" % k, state_tag=cur, freq=None, gpus=on)
        )
        cmd_load(sub_args(a, tag=t + "r%d" % k, gpus=on))
        for g in set((0, 1)) - set(on):
            carry(a, g, cur, t + "r%d" % k)
        cur = current_tag(a, t + "r%d" % k)
        drop_window(a, t + "r%d" % k)


def power_summary(a, tag):
    """freq_id -> total residual / raw power over a window's subtracting frames, both GPUs."""
    out = {}
    for g in (0, 1):
        fr = list(dtv_state.frames(tag_files(a, tag, g)))
        if not fr:
            continue
        for c, fid in enumerate(fr[0]["freq"]):
            use = [f for f in fr if (f["act"][c] >= 2).any()]
            if use:
                out[int(fid)] = sum(f["pw_res"][c].sum() for f in use) / max(
                    sum(f["pw_raw"][c].sum() for f in use), 1e-9
                )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--node", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("window")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--tag", required=True)
    p.add_argument("--beam", action="store_true")
    p.set_defaults(fn=cmd_window)
    p = sub.add_parser("start")
    p.add_argument("--source", type=int, required=True)
    p.add_argument("--frames", type=int, default=24)
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("fit")
    p.add_argument("--source", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--state-tag")
    p.add_argument("--freq", nargs="*", help="g0:F,F g1:F,F")
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("load")
    p.add_argument("--tag", required=True)
    p.set_defaults(fn=cmd_load)
    p = sub.add_parser("clear")
    p.add_argument("--source", type=int, default=0)
    p.set_defaults(fn=cmd_clear)
    p = sub.add_parser("report")
    p.add_argument("--tag", required=True)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("setup")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--prefix", default=time.strftime("%H%M", time.gmtime()) + "_")
    p.add_argument(
        "--sources",
        type=int,
        default=0,
        help="source slots to set up (default: all the stage has)",
    )
    p.set_defaults(fn=cmd_setup)
    p = sub.add_parser("refit")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--both", action="store_true", help="also refit source 1")
    p.add_argument(
        "--extra",
        type=int,
        nargs="*",
        help="also refit these sources, in order, after source 0",
    )
    p.add_argument("--prefix", default=time.strftime("%H%M%S", time.gmtime()) + "_")
    p.set_defaults(fn=cmd_refit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
