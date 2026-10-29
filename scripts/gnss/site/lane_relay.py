#!/usr/bin/env python3
"""Lane relay: the chains' lane directions carried between the projecting nodes
(docs/CHORD_FLEET_LANES.md).

  run  [--period S] [--inst-period S] [--to NODE,..] [--from NODE,..] [--status DIR] [--once]
       Every period: GET gnss_n2_project/proj_<g>/get_lanes from every projecting GPU, keep
       the copy of best coherence of each (band, PRN), take the target channels from the
       fleet file's fleet-lane-transfer table, and POST .../set_lanes to every receiving GPU
       the lanes whose targets meet its live channels, less the ones it measured itself.
       Every inst-period the gps_l5 assemblers' instrumental delays (get_lane_inst) are
       pooled and sent along. --to limits the receivers (a canary), --from the sources.
  show One cycle, printed: per GPU what it serves and what it would receive. Nothing posted.

Reads config/gnss_fleet_chord.yaml (fleet-lane-transfer), config/chord_gnss_node.yaml
(gpu_offsets_mod16), the generated node configs (live_freq_ids, source, fleet_lanes) and the
nodes' REST, nothing else: it never touches the control path. A node that stops answering
drops out of the set; a receiver that stops hearing from the relay ages the set out on its
own (fleet_lane_max_age_s). DIR/current.json is rewritten every cycle.
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import yaml

K = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
FLEET = "config/gnss_fleet_chord.yaml"
NODES = "config/chord_gnss_node.yaml"
GEN = "config/generated/chord_gnss_%s_multi.yaml"
PORT = 12048
CH_MHZ = 0.1953125
OWNER = re.compile(r"^gnss(\d)(?:_([a-z0-9]+))?_n2assemble/lane/(\d+)$")
LOG_EVERY_S = 60.0
ERR_EVERY_S = 300.0


def log(msg):
    sys.stdout.write(
        "%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), msg)
    )
    sys.stdout.flush()


class RateLimited:
    def __init__(self):
        self.last = {}

    def __call__(self, key, msg, every_s):
        now = time.monotonic()
        if now - self.last.get(key, -1e9) >= every_s:
            self.last[key] = now
            log(msg)


def http_json(url, body=None, timeout=1.0):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}


def transfer_table(fleet_path):
    """fleet-lane-transfer TAG:LO-HI (MHz) -> {tag: [[lo_fid, hi_fid], ...]}, the channels
    centred inside [LO, HI], as the generator reads lane-transfer."""
    common = (yaml.safe_load(open(fleet_path)) or {}).get("common", {}) or {}
    out = {}
    for spec in common.get("fleet-lane-transfer", []) or []:
        tag, _, band = str(spec).partition(":")
        lo, _, hi = band.partition("-")
        lo_f = int(-(-float(lo) // CH_MHZ))  # ceil
        hi_f = int(float(hi) // CH_MHZ)
        if not tag or hi_f < lo_f:
            raise SystemExit("fleet-lane-transfer: bad entry %r" % spec)
        out.setdefault(tag.strip("_"), []).append([lo_f, hi_f])
    return out


def gpu_offsets(nodes_path):
    nodes = (yaml.safe_load(open(nodes_path)) or {}).get("nodes", {}) or {}
    return {
        n: list(v["gpu_offsets_mod16"])
        for n, v in nodes.items()
        if isinstance(v, dict) and "gpu_offsets_mod16" in v
    }


class Gpu:
    """One projecting GPU: a source when its stage publishes lanes, a receiver with
    fleet_lanes."""

    def __init__(self, node, gpu, stage, offset):
        self.node, self.gpu = node, gpu
        self.name = "%s/%d" % (node, gpu)
        self.source = stage.get("source", "eigen") != "eigen"
        self.receiver = bool(stage.get("fleet_lanes", False))
        live = stage.get("live_freq_ids", []) or []
        self.live = sorted(int(f) for f in live if int(f) % 16 == offset)
        self.base = "http://%s:%d/gnss_n2_project/proj_%d" % (node, PORT, gpu)
        self.inst_url = "http://%s:%d/gnss%d_n2assemble/get_lane_inst" % (
            node,
            PORT,
            gpu,
        )

    def wants(self, ranges):
        for lo, hi in ranges:
            for f in self.live:
                if lo <= f <= hi:
                    return True
        return False


def gpus(root, offsets, only_from=None, only_to=None):
    out = []
    for path in sorted(glob.glob(os.path.join(root, GEN % "cx*"))):
        node = re.sub(r".*chord_gnss_(cx\d+)_multi\.yaml$", r"\1", path)
        if node not in offsets:
            continue
        cfg = yaml.safe_load(open(path)) or {}
        proj = cfg.get("gnss_n2_project", {}) or {}
        for g in (0, 1):
            stage = proj.get("proj_%d" % g)
            if not isinstance(stage, dict):
                continue
            u = Gpu(node, g, stage, offsets[node][g])
            if only_from is not None and node not in only_from:
                u.source = False
            if only_to is not None and node not in only_to:
                u.receiver = False
            out.append(u)
    return out


def parse_owner(owner):
    m = OWNER.match(owner)
    if not m:
        return None, None
    return (m.group(2) or "l5"), int(m.group(3))


def pool_inst(replies):
    """Element-wise median over the warm instances of their delays, each referenced to its
    own median (a delay common to every element is not measured)."""
    rows = []
    for r in replies:
        if not isinstance(r, dict) or not r.get("warm") or not r.get("delay_ns"):
            continue
        d = [float(x) for x in r["delay_ns"]]
        med = sorted(d)[len(d) // 2]
        rows.append([x - med for x in d])
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        return [], len(rows)
    n = len(rows[0])
    out = []
    for i in range(n):
        col = sorted(r[i] for r in rows)
        out.append(round(col[len(col) // 2], 3))
    return out, len(rows)


class Relay:
    def __init__(self, a):
        self.a = a
        self.rl = RateLimited()
        self.table = transfer_table(os.path.join(a.root, FLEET))
        self.offsets = gpu_offsets(os.path.join(a.root, NODES))
        self.only_from = set(a.only_from.split(",")) if a.only_from else None
        self.only_to = set(a.only_to.split(",")) if a.only_to else None
        self.units = gpus(a.root, self.offsets, self.only_from, self.only_to)
        self.inst = (
            None  # the pooled delays (ns), sent on inst cycles; None = never read
        )
        self.inst_t = -1e9
        self.inst_n = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max(4, 2 * len(self.units)))
        log(
            "lane relay: %d GPU(s), %d source(s), %d receiver(s); transfers %s"
            % (
                len(self.units),
                sum(u.source for u in self.units),
                sum(u.receiver for u in self.units),
                ", ".join(
                    "%s:%s" % (t, ",".join("%d-%d" % (lo, hi) for lo, hi in r))
                    for t, r in sorted(self.table.items())
                )
                or "(none: nothing will be posted)",
            )
        )

    def get(self, u):
        try:
            return u, http_json(u.base + "/get_lanes", timeout=self.a.timeout)
        except Exception as e:  # noqa: BLE001 -- a dead node is a row in the status, not a crash
            return u, {"error": repr(e)[:100]}

    def get_inst(self, u):
        try:
            return u, http_json(u.inst_url, timeout=self.a.timeout)
        except Exception as e:  # noqa: BLE001
            return u, {"error": repr(e)[:100]}

    def post(self, u, body):
        try:
            return u, http_json(u.base + "/set_lanes", body, timeout=self.a.timeout)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:120]
            e.close()
            return u, {"error": "HTTP %d %s" % (e.code, msg)}
        except Exception as e:  # noqa: BLE001
            return u, {"error": repr(e)[:100]}

    def collect(self):
        """Every source's lanes: {(tag, prn): (lane, source Gpu)}, the best coherence."""
        srcs = [u for u in self.units if u.source]
        status = {}
        best = {}
        for u, r in self.pool.map(self.get, srcs):
            if "error" in r:
                status[u.name] = {"error": r["error"]}
                self.rl(
                    ("get", u.name),
                    "get_lanes %s: %s" % (u.name, r["error"]),
                    ERR_EVERY_S,
                )
                continue
            lanes = r.get("lanes", []) or []
            status[u.name] = {
                "ready": bool(r.get("ready")),
                "lanes": len(lanes),
                "n": r.get("n"),
            }
            for L in lanes:
                tag, prn = parse_owner(L.get("owner", ""))
                if tag is None or tag not in self.table:
                    continue
                key = (tag, prn)
                if key not in best or L["coh"] > best[key][0]["coh"]:
                    best[key] = (L, u)
        return best, status

    def body_for(self, u, best, inst):
        lanes = []
        for (tag, prn), (L, src) in sorted(best.items()):
            if src is u or not u.wants(self.table[tag]):
                continue
            lanes.append(
                {
                    "id": "%s/%s" % (src.node, L["owner"]),
                    "f_src_hz": float(L["freq_id"]) * CH_MHZ * 1e6,
                    "t_utc": float(L.get("t_utc", 0.0)),
                    "los": [float(x) for x in L["los"]],
                    "los_rate": [float(x) for x in L["los_rate"]],
                    "coh": float(L["coh"]),
                    "power": float(L.get("power", 0.0)),
                    "fids": self.table[tag],
                    "q": [
                        [round(float(re_), 5), round(float(im), 5)]
                        for re_, im in L["q"]
                    ],
                }
            )
        body = {"t_utc": time.time(), "lanes": lanes}
        if inst is not None:
            body["inst_delay_ns"] = inst
        return body

    def cycle(self, post=True):
        t0 = time.monotonic()
        best, src_status = self.collect()
        inst = None
        if time.monotonic() - self.inst_t >= self.a.inst_period:
            self.inst_t = time.monotonic()
            replies = [
                r
                for _, r in self.pool.map(
                    self.get_inst, [u for u in self.units if u.source]
                )
            ]
            self.inst, self.inst_n = pool_inst(replies)
            inst = self.inst
        rx_status = {}
        if post:
            jobs = [(u, self.body_for(u, best, inst)) for u in self.units if u.receiver]
            for u, r in self.pool.map(lambda j: self.post(*j), jobs):
                n_sent = len(next(b for v, b in jobs if v is u)["lanes"])
                if "error" in r:
                    rx_status[u.name] = {"error": r["error"], "sent": n_sent}
                    self.rl(
                        ("post", u.name),
                        "set_lanes %s: %s" % (u.name, r["error"]),
                        ERR_EVERY_S,
                    )
                else:
                    rx_status[u.name] = {"sent": n_sent, "ready": bool(r.get("ready"))}
        status = {
            "t": time.time(),
            "cycle_s": round(time.monotonic() - t0, 3),
            "sources": src_status,
            "lanes": [
                {
                    "tag": t,
                    "prn": p,
                    "from": src.name,
                    "coh": round(L["coh"], 3),
                    "freq_id": L["freq_id"],
                }
                for (t, p), (L, src) in sorted(best.items())
            ],
            "receivers": rx_status,
            "inst": {"instances": self.inst_n, "sent": inst is not None},
        }
        return status

    def write_status(self, status):
        if not self.a.status:
            return
        os.makedirs(self.a.status, exist_ok=True)
        tmp = os.path.join(self.a.status, ".current.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(status, fh, indent=1)
        os.replace(tmp, os.path.join(self.a.status, "current.json"))

    def run(self):
        while True:
            t0 = time.monotonic()
            status = self.cycle(post=True)
            self.write_status(status)
            sent = [
                v.get("sent", 0)
                for v in status["receivers"].values()
                if "error" not in v
            ]
            self.rl(
                "cycle",
                "sources %d ok / %d, %d lane(s) in %d band(s); posted to %d/%d receiver(s)"
                " (%s lanes each), inst delays from %d instance(s); %.2f s"
                % (
                    sum("error" not in v for v in status["sources"].values()),
                    len(status["sources"]),
                    len(status["lanes"]),
                    len({l["tag"] for l in status["lanes"]}),
                    len(sent),
                    len(status["receivers"]),
                    ("%d-%d" % (min(sent), max(sent))) if sent else "-",
                    self.inst_n,
                    status["cycle_s"],
                ),
                LOG_EVERY_S,
            )
            if self.a.once:
                return
            time.sleep(max(0.05, self.a.period - (time.monotonic() - t0)))

    def show(self):
        status = self.cycle(post=False)
        best, _ = self.collect()
        print("sources:")
        for name, v in sorted(status["sources"].items()):
            print("  %-8s %s" % (name, v))
        print("lanes (%d):" % len(status["lanes"]))
        for l in status["lanes"]:
            print(
                "  %-6s %3d  from %-8s coh %.3f  freq_id %d"
                % (l["tag"], l["prn"], l["from"], l["coh"], l["freq_id"])
            )
        print("receivers:")
        for u in self.units:
            if not u.receiver:
                continue
            body = self.body_for(u, best, None)
            print(
                "  %-8s %2d lane(s): %s"
                % (
                    u.name,
                    len(body["lanes"]),
                    ", ".join(l["id"].split("/")[-1] for l in body["lanes"]),
                )
            )
        print(
            "inst delays: %d warm instance(s), %s"
            % (self.inst_n, "pooled" if self.inst else "none")
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["run", "show"])
    ap.add_argument("--period", type=float, default=1.0, help="s between cycles (1.0)")
    ap.add_argument(
        "--inst-period", type=float, default=10.0, help="s between delay polls (10)"
    )
    ap.add_argument("--timeout", type=float, default=0.8, help="s per request (0.8)")
    ap.add_argument(
        "--to",
        dest="only_to",
        help="receivers: NODE,NODE (default every fleet_lanes GPU)",
    )
    ap.add_argument(
        "--from",
        dest="only_from",
        help="sources: NODE,NODE (default every projecting GPU)",
    )
    ap.add_argument("--status", help="directory for current.json")
    ap.add_argument("--once", action="store_true", help="run: one cycle")
    ap.add_argument(
        "--root",
        default=K,
        help="the checkout whose config/ and config/generated/ to read",
    )
    a = ap.parse_args()
    r = Relay(a)
    if a.mode == "show":
        r.show()
    else:
        r.run()


if __name__ == "__main__":
    main()
