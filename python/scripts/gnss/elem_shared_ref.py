#!/usr/bin/env python3
"""The fleet reference of the shared element model's phase pin (#154).

  snapshot OUT.json   Poll every n2assemble's /get_elem_cal and form, per band and pol, the
                      consensus of the warm, unfrozen models (instances more than --exclude-deg
                      off it, or whose shape it does not describe, are left out and listed).
                      The vector is phased so the included instances' offsets average zero, and
                      written with the array epoch the configs were generated for. The generator
                      copies each band's entry into that band's assemblers (--elem-shared-ref).
  post FILE           POST every band's reference to its instances (--mode, --slew-deg-s,
                      --only cx27/0,cx42).
  status [FILE]       Per instance and pol: offset and shape similarity against FILE (works on any
                      binary), plus the node's own fleet_ref block where the binary has one.
  watch FILE DIR      The safety net (scripts/gnss/site/elem_ref_cron.sh, every 5 min): per band, R and
                      each instance's offset against FILE into DIR/current.json; DIR/ALERT exists
                      while any instance is > --alert-deg off for > --min-run s outside a freeze.
                      --act re-posts FILE in live mode to such an instance (at most every
                      --act-every s); without it nothing is changed.

Instances are read from config/generated/chord_gnss_<node>_multi.yaml (port 12048). Band names
are snap_xinst.py's: l5 for gnss<g>_n2assemble, else the chain tag (e5a, b2a, ...).
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

K = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
GEN = os.path.join(K, "config", "generated", "chord_gnss_%s_multi.yaml")
PORT = 12048


def nodes():
    return sorted(
        re.sub(r".*chord_gnss_(cx\d+)_multi\.yaml$", r"\1", p)
        for p in glob.glob(GEN % "cx*")
    )


def endpoints():
    out = []
    for n in nodes():
        for st in re.findall(
            r"^(gnss[01][a-z0-9_]*n2assemble):", open(GEN % n).read(), re.M
        ):
            out.append((n, st))
    return out


def band_of(stage):
    parts = stage.split("_")
    return "l5" if len(parts) == 2 else parts[1]


def inst_name(node, stage):
    return "%s/%s" % (node, stage[4])


def epoch_of_configs():
    keys = set()
    for n in nodes():
        keys.update(
            re.findall(r"^\s+elem_positions_epoch:\s*(\S+)", open(GEN % n).read(), re.M)
        )
    if len(keys) != 1:
        sys.exit(
            "the generated configs carry %d different elem_positions_epoch values: %s"
            % (len(keys), sorted(keys))
        )
    return keys.pop()


def get(ep):
    node, stage = ep
    try:
        with urllib.request.urlopen(
            "http://%s:%d/%s/get_elem_cal" % (node, PORT, stage), timeout=5
        ) as r:
            return ep, json.loads(r.read())
    except Exception as e:  # noqa: BLE001 -- a dead node is a row in the report, not a crash
        return ep, {"error": repr(e)[:80]}


def poll():
    eps = endpoints()
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        return dict(pool.map(get, eps))


def halves(d):
    """(pol-0 half, pol-1 half) of an instance's shared model, or None if it has none warm."""
    if not d.get("shared_warm") or not d.get("g_shared"):
        return None
    g = np.array([complex(a, b) for a, b in d["g_shared"]])
    h = len(g) // 2
    return g[:h], g[h:]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def consensus(vs):
    """Phase-aligned mean of unit vectors (snap_xinst.py's construction)."""
    keys = sorted(vs)
    ref = vs[keys[0]].copy()
    for _ in range(5):
        acc = sum(vs[k] * np.exp(-1j * np.angle(np.vdot(ref, vs[k]))) for k in keys)
        ref = acc / np.linalg.norm(acc)
    return ref


def offsets(ref, vs):
    """Per instance: (offset deg, similarity) of each unit vector against ref."""
    return {
        k: (float(np.degrees(np.angle(np.vdot(ref, v)))), float(abs(np.vdot(ref, v))))
        for k, v in vs.items()
    }


def snapshot(a):
    res = poll()
    epoch = epoch_of_configs()
    bands, n_elem = {}, None
    for (node, stage), d in sorted(res.items()):
        if "error" in d:
            print(
                "  %-22s %s"
                % (inst_name(node, stage) + " " + band_of(stage), d["error"])
            )
    for band in sorted({band_of(st) for _, st in res}):
        per_pol, entry = [], {"R": [], "n": [], "excluded": [], "sim_median": []}
        for pol in (0, 1):
            vs = {}
            for (node, stage), d in res.items():
                if band_of(stage) != band or "error" in d or d.get("shared_frozen"):
                    continue
                hv = halves(d)
                u = unit(hv[pol]) if hv else None
                if u is not None:
                    vs[inst_name(node, stage)] = u
                    n_elem = 2 * len(u)
            if len(vs) < 4:
                sys.exit(
                    "band %s pol %d: only %d warm, unfrozen instances -- not a consensus"
                    % (band, pol, len(vs))
                )
            keep = dict(vs)
            for _ in range(3):
                ref = consensus(keep)
                off = offsets(ref, vs)
                z = np.mean([np.exp(1j * np.radians(off[k][0])) for k in keep])
                rel = {
                    k: (float(np.degrees(np.angle(np.exp(1j * np.radians(o)) / z))), s)
                    for k, (o, s) in off.items()
                }
                keep = {
                    k: vs[k]
                    for k, (o, s) in rel.items()
                    if abs(o) <= a.exclude_deg and s >= a.min_sim
                }
                if len(keep) < 4:
                    sys.exit(
                        "band %s pol %d: %d instances within %.0f deg of the consensus"
                        % (band, pol, len(keep), a.exclude_deg)
                    )
            ref = consensus(keep)
            off = offsets(ref, keep)
            z = np.mean([np.exp(1j * np.radians(o)) for o, _ in off.values()])
            ref = ref * z / abs(z)  # included instances' offsets now average zero
            fin = offsets(ref, vs)
            entry["R"].append(
                round(
                    float(
                        abs(np.mean([np.exp(1j * np.radians(fin[k][0])) for k in keep]))
                    ),
                    4,
                )
            )
            entry["n"].append(len(keep))
            entry["sim_median"].append(
                round(float(np.median([fin[k][1] for k in keep])), 3)
            )
            entry["excluded"].append(
                {
                    k: [round(o, 1), round(s, 3)]
                    for k, (o, s) in fin.items()
                    if k not in keep
                }
            )
            per_pol.append(ref)
            print(
                "%-4s pol%d  R %.3f  n %2d  sim med %.3f  excluded %s"
                % (
                    band,
                    pol,
                    entry["R"][-1],
                    len(keep),
                    entry["sim_median"][-1],
                    entry["excluded"][-1] or "-",
                )
            )
        entry["ref"] = [
            [round(float(x.real), 6), round(float(x.imag), 6)]
            for x in np.concatenate(per_pol)
        ]
        bands[band] = entry
    out = {
        "what": "fleet reference of the shared element model's phase pin (#154): per band, "
        "pol-0 half then pol-1 half, each half unit-norm",
        "epoch": epoch,
        "n_elements": n_elem,
        "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exclude_deg": a.exclude_deg,
        "min_sim": a.min_sim,
        "bands": bands,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote %s (epoch %s, %d bands)" % (a.out, epoch, len(bands)))


def load_ref(path):
    snap = json.load(open(path))
    epoch = epoch_of_configs()
    if snap["epoch"] != epoch:
        sys.exit(
            "%s was taken against epoch %s, the configs are %s"
            % (path, snap["epoch"], epoch)
        )
    return snap


def selected(node, stage, only):
    if not only:
        return True
    name = inst_name(node, stage)
    return any(o in (node, name, band_of(stage)) for o in only)


def post(a):
    snap = load_ref(a.file)
    only = [o for o in (a.only or "").split(",") if o]
    body_extra = {}
    if a.mode:
        body_extra["mode"] = a.mode
    if a.slew_deg_s is not None:
        body_extra["slew_deg_s"] = a.slew_deg_s
    for node, stage in endpoints():
        band = band_of(stage)
        if band not in snap["bands"] or not selected(node, stage, only):
            continue
        body = dict(body_extra, ref=snap["bands"][band]["ref"])
        req = urllib.request.Request(
            "http://%s:%d/%s/set_elem_sum_shared_ref" % (node, PORT, stage),
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                rep = json.loads(r.read())
                print(
                    "  %-8s %-4s was %-4s staged %s"
                    % (
                        inst_name(node, stage),
                        band,
                        rep.get("mode"),
                        ",".join(k for k, v in rep.get("staged", {}).items() if v),
                    )
                )
        except Exception as e:  # noqa: BLE001
            print(
                "  %-8s %-4s FAILED %s" % (inst_name(node, stage), band, repr(e)[:100])
            )


def status(a):
    snap = load_ref(a.file) if a.file else None
    res = poll()
    for band in sorted({band_of(st) for _, st in res}):
        rows = []
        for (node, stage), d in sorted(res.items()):
            if band_of(stage) != band:
                continue
            name = inst_name(node, stage)
            if "error" in d:
                rows.append("  %-8s %s" % (name, d["error"]))
                continue
            line = "  %-8s warm %-5s frozen %-5s" % (
                name,
                d.get("shared_warm"),
                d.get("shared_frozen"),
            )
            hv = halves(d)
            if snap and hv and band in snap["bands"]:
                F = np.array([complex(x, y) for x, y in snap["bands"][band]["ref"]])
                h = len(F) // 2
                for pol in (0, 1):
                    f, g = unit(F[pol * h : (pol + 1) * h]), unit(hv[pol])
                    if f is None or g is None:
                        line += "  pol%d    -" % pol
                        continue
                    y = np.vdot(f, g)
                    line += "  pol%d %+5.0f deg sim %.2f" % (
                        pol,
                        np.degrees(np.angle(y)),
                        abs(y),
                    )
            fr = d.get("fleet_ref")
            if fr:
                line += "  | node %s %s err %s sim %s applied %s" % (
                    fr["mode"],
                    "F" if fr["present"] else "noF",
                    "/".join("%+.0f" % x for x in fr["err_deg"]),
                    "/".join("%.2f" % x for x in fr["sim"]),
                    "/".join("y" if x else "n" for x in fr["applied"]),
                )
            rows.append(line)
        print("== %s" % band)
        print("\n".join(rows))


def instance_offsets(d, F):
    """(offset deg, sim) per pol of one instance's model against reference F, or None."""
    hv = halves(d)
    if not hv:
        return None
    h = len(F) // 2
    out = []
    for pol in (0, 1):
        f, g = unit(F[pol * h : (pol + 1) * h]), unit(hv[pol])
        if f is None or g is None:
            out.append(None)
            continue
        y = np.vdot(f, g)
        out.append((float(np.degrees(np.angle(y))), float(abs(y))))
    return out


def watch(a):
    snap = load_ref(a.file)
    os.makedirs(a.dir, exist_ok=True)
    state_path = os.path.join(a.dir, "state.json")
    try:
        state = json.load(open(state_path))
    except (OSError, ValueError):
        state = {"bad": {}, "acted": {}}
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    res = poll()
    cur = {
        "t": stamp,
        "file": a.file,
        "taken_utc": snap["taken_utc"],
        "bands": {},
        "errors": {},
    }
    alerts, seen = [], set()
    for band, entry in sorted(snap["bands"].items()):
        F = np.array([complex(x, y) for x, y in entry["ref"]])
        rows, ph = {}, {0: [], 1: []}
        for (node, stage), d in sorted(res.items()):
            if band_of(stage) != band:
                continue
            name = inst_name(node, stage)
            if "error" in d:
                cur["errors"][name + " " + band] = d["error"]
                continue
            off = instance_offsets(d, F)
            frozen = bool(d.get("shared_frozen"))
            fr = d.get("fleet_ref") or {}
            rows[name] = {
                "frozen": frozen,
                "mode": fr.get("mode"),
                "pol": [
                    None if o is None else [round(o[0], 1), round(o[1], 3)]
                    for o in (off or [None, None])
                ],
            }
            for pol in (0, 1):
                o = off[pol] if off else None
                key = "%s %s pol%d" % (name, band, pol)
                if o is None:
                    continue
                ph[pol].append(np.exp(1j * np.radians(o[0])))
                if frozen:  # a transit freeze neither starts nor clears an episode
                    if key in state["bad"]:
                        seen.add(key)
                    continue
                if abs(o[0]) > a.alert_deg:
                    seen.add(key)
                    t0 = state["bad"].setdefault(key, now)
                    if now - t0 >= a.min_run:
                        alerts.append((key, name, node, stage, o, now - t0, fr))
        cur["bands"][band] = {
            "R": [
                round(float(abs(np.mean(ph[p]))), 3) if ph[p] else None for p in (0, 1)
            ],
            "instances": rows,
        }
    for key in list(state["bad"]):
        if key not in seen:
            del state["bad"][key]
    lines = []
    for key, name, node, stage, o, dur, fr in alerts:
        why = (
            "shape no longer described by the reference (sim %.2f): take a new snapshot"
            % o[1]
            if o[1] < a.min_sim
            else ""
        )
        line = "%s %+.0f deg (sim %.2f) for %.0f min, node mode %s%s" % (
            key,
            o[0],
            o[1],
            dur / 60.0,
            fr.get("mode", "n/a (binary without #154)"),
            ("; " + why) if why else "",
        )
        act_key = "%s %s" % (name, band_of(stage))
        if (
            a.act
            and not why
            and fr
            and now - state["acted"].get(act_key, 0) >= a.act_every
        ):
            body = {"ref": snap["bands"][band_of(stage)]["ref"], "mode": "live"}
            req = urllib.request.Request(
                "http://%s:%d/%s/set_elem_sum_shared_ref" % (node, PORT, stage),
                data=json.dumps(body).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    r.read()
                state["acted"][act_key] = now
                line += "; ACTED: reference re-posted, mode live"
            except Exception as e:  # noqa: BLE001
                line += "; act FAILED: %s" % repr(e)[:80]
        lines.append(line)
    cur["alerts"] = lines
    alert_path = os.path.join(a.dir, "ALERT")
    if lines:
        with open(alert_path, "w") as f:
            f.write("%s\n%s\n" % (stamp, "\n".join(lines)))
    elif os.path.exists(alert_path):
        os.remove(alert_path)
    for path, obj in ((state_path, state), (os.path.join(a.dir, "current.json"), cur)):
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(path + ".tmp", path)
    print(
        "%s R %s; %d instance-pols off > %.0f deg, %d alerting%s"
        % (
            stamp,
            " ".join(
                "%s %s/%s"
                % (b, *(("%.2f" % r) if r is not None else "-" for r in v["R"]))
                for b, v in sorted(cur["bands"].items())
            ),
            len(state["bad"]),
            a.alert_deg,
            len(lines),
            "".join("\n  " + ln for ln in lines),
        )
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("out")
    s.add_argument("--exclude-deg", type=float, default=30.0)
    s.add_argument("--min-sim", type=float, default=0.5)
    p = sub.add_parser("post")
    p.add_argument("file")
    p.add_argument("--mode", choices=("off", "log", "live"))
    p.add_argument("--slew-deg-s", type=float)
    p.add_argument(
        "--only", help="comma list of nodes (cx27), instances (cx27/0) or bands (e6)"
    )
    t = sub.add_parser("status")
    t.add_argument("file", nargs="?")
    w = sub.add_parser("watch")
    w.add_argument("file")
    w.add_argument("dir")
    w.add_argument("--alert-deg", type=float, default=45.0)
    w.add_argument("--min-run", type=float, default=900.0)
    w.add_argument("--min-sim", type=float, default=0.5)
    w.add_argument("--act", action="store_true")
    w.add_argument("--act-every", type=float, default=1800.0)
    a = ap.parse_args()
    {"snapshot": snapshot, "post": post, "status": status, "watch": watch}[a.cmd](a)


if __name__ == "__main__":
    main()
