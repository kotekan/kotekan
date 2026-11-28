#!/usr/bin/env python3
"""Lane relay: the chains' lane directions carried between the projecting nodes
(docs/CHORD_FLEET_LANES.md).

  run  [--period S] [--inst-period S] [--to NODE,..] [--from NODE,..] [--status DIR] [--once]
       Every period: GET gnss_n2_project/proj_<g>/get_lanes from every projecting GPU, keep
       the copy of best coherence of each (chain, PRN), take the target channels from the
       fleet file's fleet-lane-transfer table (by broker chain: gal_e6, gps_l2c, ...), and
       POST .../set_lanes to every receiving GPU the lanes whose targets meet its live
       channels, less the ones it measured itself on a channel it projects (a lane it only
       serves, L1 C/A's, goes back to it too). Of two chains of one satellite whose targets
       overlap, only the lane measured nearer them goes out (L2C's, not L1 C/A's, at L2).
       A lane whose power is fleet-lane-transfer-bright-db (15) above its chain's median also
       goes to the fleet-lane-transfer-bright targets (a close pass's sidelobes), on every
       GPU including its own.
       Every inst-period the instrumental delays (get_lane_inst) of one transferring
       assembler per GPU (inst_stage) are pooled and sent along. With --inst-phase, the
       instrumental phase table (inst_phase_table.py) turned to the current F-engine start by
       the step table goes along too, when it changes and every PHASE_REPOST_S; while the
       steps since the table's start are not measured an empty one clears it on the nodes,
       which fall back to the pooled delays. --to limits the receivers (a canary), --from the
       sources.
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
sys.path.insert(0, os.path.join(K, "config"))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))
import gnss_fleet  # noqa: E402
import fengine_steps  # noqa: E402
import inst_phase_table  # noqa: E402
from gnss_broker import node_api  # noqa: E402

FLEET = "config/gnss_fleet_chord.yaml"
NODES = "config/chord_gnss_node.yaml"
GEN = "config/generated/chord_gnss_%s_multi.yaml"
PORT = 12048
CH_MHZ = 0.1953125
OWNER = re.compile(r"^(/?gnss\d+_\w+)/lane/(\d+)$")
LOG_EVERY_S = 60.0
ERR_EVERY_S = 300.0
PHASE_REPOST_S = 300.0  # an unchanged phase table is posted again this often


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


def transfer_table(fleet_path, chains, key="fleet-lane-transfer"):
    """KEY CHAIN:LO-HI (MHz) -> {chain: [[lo_fid, hi_fid], ...]}, the channels centred inside
    [LO, HI], as the generator reads lane-transfer. A CHAIN not in `chains` is refused."""
    common = (yaml.safe_load(open(fleet_path)) or {}).get("common", {}) or {}
    out = {}
    for spec in common.get(key, []) or []:
        chain, _, band = str(spec).partition(":")
        chain = chain.strip()
        lo, _, hi = band.partition("-")
        lo_f = int(-(-float(lo) // CH_MHZ))  # ceil
        hi_f = int(float(hi) // CH_MHZ)
        if not chain or hi_f < lo_f:
            raise SystemExit("%s: bad entry %r" % (key, spec))
        if chain not in chains:
            raise SystemExit(
                "%s: %r names %r, not a chain the fleet builds; name one by its broker "
                "key: %s" % (key, spec, chain, ", ".join(sorted(chains)))
            )
        out.setdefault(chain, []).append([lo_f, hi_f])
    return out


def bright_lanes(best, powers, gate_db):
    """The (chain, prn) keys of best whose power is gate_db above the median of every lane
    of their chain this cycle; none for a chain with fewer than 3 lanes."""
    out = set()
    for key, (L, _src) in best.items():
        p = sorted(powers.get(key[0], []))
        if len(p) >= 3 and L.get("power", 0.0) >= p[len(p) // 2] * 10.0 ** (
            gate_db / 10.0
        ):
            out.add(key)
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

    def __init__(self, node, gpu, stage, offset, inst, hosts=()):
        self.node, self.gpu = node, gpu
        self.name = "%s/%d" % (node, gpu)
        self.source = stage.get("source", "eigen") != "eigen"
        self.receiver = bool(stage.get("fleet_lanes", False))
        live = stage.get("live_freq_ids", []) or []
        self.live = sorted(int(f) for f in live if int(f) % 16 == offset)
        self.base = "http://%s:%d/gnss_n2_project/proj_%d" % (node, PORT, gpu)
        # By chain where the node's config hosts its assemblers (gnss_broker/node_api.py).
        self.inst_url = inst and node_api.url(
            "http://%s:%d/%s" % (node, PORT, inst), "get_lane_inst", hosts=hosts
        )

    def wants(self, ranges):
        for lo, hi in ranges:
            for f in self.live:
                if lo <= f <= hi:
                    return True
        return False


def inst_stage(cfg, gpu):
    """The assembler on this GPU that serves get_lane_inst, or None: one with lane transfers
    and at least two lane channels (GnssGpuRecordAssemble's _inst_on), a stage of its own or a
    row of gnss<g>_records (gnss_fleet.assembler_rows). The GPS L5 one first, then SBAS L5,
    then the one with the most channels."""
    names = []
    for k, v in gnss_fleet.assembler_rows(cfg):
        if not re.match(r"^gnss%d_" % gpu, k):
            continue
        chain = gnss_fleet.stage_chain(k, "n2assemble")
        if not chain or not v.get("lane_transfer_freq_ids"):
            continue
        if (
            len(v.get("channel_ids") or []) < 2
            or v.get("lane_transfer_inst_slope") is False
        ):
            continue
        names.append(
            ({"gps_l5": 0, "sbas_l5": 1}.get(chain, 2), -len(v["channel_ids"]), k)
        )
    return min(names)[2] if names else None


def gpus(root, offsets, only_from=None, only_to=None):
    out = []
    for path in sorted(glob.glob(os.path.join(root, GEN % "cx*"))):
        node = re.sub(r".*chord_gnss_(cx\d+)_multi\.yaml$", r"\1", path)
        if node not in offsets:
            continue
        cfg = yaml.safe_load(open(path)) or {}
        proj = cfg.get("gnss_n2_project", {}) or {}
        hosts = gnss_fleet.config_host_bases(cfg, "http://%s:%d" % (node, PORT))
        for g in (0, 1):
            stage = proj.get("proj_%d" % g)
            if not isinstance(stage, dict):
                continue
            u = Gpu(node, g, stage, offsets[node][g], inst_stage(cfg, g), hosts)
            if only_from is not None and node not in only_from:
                u.source = False
            if only_to is not None and node not in only_to:
                u.receiver = False
            out.append(u)
    return out


def parse_owner(owner):
    """A lane owner ("<assembler unique_name>/lane/<prn>") -> (chain, prn), or (None, None)."""
    m = OWNER.match(owner)
    chain = m and gnss_fleet.stage_chain(m.group(1), "n2assemble")
    if not chain:
        return None, None
    return chain, int(m.group(2))


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
        fleet = gnss_fleet.Fleet(
            os.path.join(a.root, FLEET), os.path.join(a.root, NODES)
        )
        chains = fleet.tracker_signals()
        self.sys = {r.key: r.proj_sys for r in fleet.chains}
        self.table = transfer_table(os.path.join(a.root, FLEET), chains)
        self.bright = transfer_table(
            os.path.join(a.root, FLEET), chains, "fleet-lane-transfer-bright"
        )
        common = (yaml.safe_load(open(os.path.join(a.root, FLEET))) or {}).get(
            "common", {}
        )
        self.bright_db = float(
            (common or {}).get("fleet-lane-transfer-bright-db", 15.0)
        )
        self.offsets = gpu_offsets(os.path.join(a.root, NODES))
        self.only_from = set(a.only_from.split(",")) if a.only_from else None
        self.only_to = set(a.only_to.split(",")) if a.only_to else None
        self.units = gpus(a.root, self.offsets, self.only_from, self.only_to)
        self.inst = (
            None  # the pooled delays (ns), sent on inst cycles; None = never read
        )
        self.inst_t = -1e9
        self.inst_n = 0
        self.phase = (
            None  # the turned table's set_lanes field; {} clears; None = not read
        )
        self.phase_why = ""
        self.phase_sent = {}  # receiver name -> (table key, monotonic time of the post)
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
        if self.bright:
            log(
                "lane relay: %.0f dB above the band median also to %s"
                % (
                    self.bright_db,
                    ", ".join(
                        "%s:%s" % (t, ",".join("%d-%d" % (lo, hi) for lo, hi in r))
                        for t, r in sorted(self.bright.items())
                    ),
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
        """Every source's lanes: {(chain, prn): (lane, source Gpu)}, the best coherence, and
        {chain: [power of every lane]}."""
        srcs = [u for u in self.units if u.source]
        status = {}
        best = {}
        powers = {}
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
                if tag is None or (tag not in self.table and tag not in self.bright):
                    continue
                powers.setdefault(tag, []).append(float(L.get("power", 0.0)))
                key = (tag, prn)
                if key not in best or L["coh"] > best[key][0]["coh"]:
                    best[key] = (L, u)
        return best, status, powers

    def read_phase(self):
        """The inst_phase field for set_lanes: the table turned to the newest F-engine start,
        or {} (clear) with the reason in phase_why."""
        try:
            with open(self.a.inst_phase) as fh:
                tab = json.load(fh)
            with open(self.a.frame0) as fh:
                start = json.load(fh)[-1]["frame0_int"]
            tab = inst_phase_table.turned(tab, fengine_steps.load(self.a.steps), start)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            self.phase_why = repr(e)[:160]
            return {}
        self.phase_why = ""
        return {
            "made_frame0": tab["made_frame0"],
            "fids": tab["fids"],
            "phase_deg": tab["phase_deg"],
        }

    def phase_due(self, u):
        """The phase field to send u this cycle, or None."""
        if self.phase is None:
            return None
        key = json.dumps(self.phase, sort_keys=True)
        last = self.phase_sent.get(u.name)
        if last and last[0] == key and time.monotonic() - last[1] < PHASE_REPOST_S:
            return None
        return self.phase

    def nearest(self, best):
        """The keys of best whose lanes go to the table's targets: of two chains of one
        satellite whose targets overlap, the one measured nearer the overlap."""
        keep = set(best)
        for (tag, prn), (L, _src) in best.items():
            for (tag2, prn2), (L2, _src2) in best.items():
                if (
                    prn2 != prn
                    or tag2 == tag
                    or self.sys.get(tag2) != self.sys.get(tag)
                ):
                    continue
                for lo, hi in self.table.get(tag, []):
                    for lo2, hi2 in self.table.get(tag2, []):
                        if max(lo, lo2) > min(hi, hi2):
                            continue
                        mid = 0.5 * (max(lo, lo2) + min(hi, hi2))
                        if abs(L2["freq_id"] - mid) < abs(L["freq_id"] - mid):
                            keep.discard((tag, prn))
        return keep

    def body_for(self, u, best, inst, bright=(), phase=None, keep=None):
        lanes = []
        for (tag, prn), (L, src) in sorted(best.items()):
            fids = []
            # A GPU's own lane reaches its targets through its own lane-transfer, unless it
            # was measured on a channel the GPU only serves.
            own = src is u and int(L["freq_id"]) in u.live
            if tag in self.table and not own and (keep is None or (tag, prn) in keep):
                fids += self.table[tag]
            if (tag, prn) in bright:
                fids += self.bright[tag]
            if not fids or not u.wants(fids):
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
                    "fids": fids,
                    "q": [
                        [round(float(re_), 5), round(float(im), 5)]
                        for re_, im in L["q"]
                    ],
                }
            )
        body = {"t_utc": time.time(), "lanes": lanes}
        if inst is not None:
            body["inst_delay_ns"] = inst
        if phase is not None:
            body["inst_phase"] = phase
        return body

    def cycle(self, post=True):
        t0 = time.monotonic()
        best, src_status, powers = self.collect()
        bright = bright_lanes(best, powers, self.bright_db) if self.bright else set()
        inst = None
        if time.monotonic() - self.inst_t >= self.a.inst_period:
            self.inst_t = time.monotonic()
            replies = []
            for u, r in self.pool.map(
                self.get_inst, [u for u in self.units if u.source and u.inst_url]
            ):
                if isinstance(r, dict) and "error" in r:
                    self.rl(
                        ("inst", u.name),
                        "get_lane_inst %s: %s" % (u.inst_url, r["error"]),
                        ERR_EVERY_S,
                    )
                replies.append(r)
            self.inst, self.inst_n = pool_inst(replies)
            inst = self.inst
            if self.a.inst_phase:
                self.phase = self.read_phase()
                if self.phase_why:
                    self.rl(
                        "phase",
                        "no instrumental phase table: %s" % self.phase_why,
                        ERR_EVERY_S,
                    )
        rx_status = {}
        if post:
            keep = self.nearest(best)
            jobs = [
                (u, self.body_for(u, best, inst, bright, self.phase_due(u), keep))
                for u in self.units
                if u.receiver
            ]
            for u, r in self.pool.map(lambda j: self.post(*j), jobs):
                body = next(b for v, b in jobs if v is u)
                n_sent = len(body["lanes"])
                if "error" not in r and "inst_phase" in body:
                    self.phase_sent[u.name] = (
                        json.dumps(body["inst_phase"], sort_keys=True),
                        time.monotonic(),
                    )
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
                    "bright": (t, p) in bright,
                }
                for (t, p), (L, src) in sorted(best.items())
            ],
            "receivers": rx_status,
            "inst": {"instances": self.inst_n, "sent": inst is not None},
            "inst_phase": None
            if self.phase is None
            else {
                "made_frame0": self.phase.get("made_frame0"),
                "knots": len(self.phase.get("fids", [])),
                "why_none": self.phase_why,
            },
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
        best, _, powers = self.collect()
        bright = bright_lanes(best, powers, self.bright_db) if self.bright else set()
        print("sources:")
        for name, v in sorted(status["sources"].items()):
            print("  %-8s %s" % (name, v))
        print("lanes (%d):" % len(status["lanes"]))
        for l in status["lanes"]:
            print(
                "  %-6s %3d  from %-8s coh %.3f  freq_id %d%s"
                % (
                    l["tag"],
                    l["prn"],
                    l["from"],
                    l["coh"],
                    l["freq_id"],
                    "  BRIGHT" if l["bright"] else "",
                )
            )
        print("receivers:")
        for u in self.units:
            if not u.receiver:
                continue
            body = self.body_for(u, best, None, bright, keep=self.nearest(best))
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
    ap.add_argument(
        "--inst-phase",
        default="",
        help="instrumental phase table to send along (inst_phase_table.py; default none)",
    )
    ap.add_argument("--steps", default=fengine_steps.TABLE, help="the step table")
    ap.add_argument(
        "--frame0", default=inst_phase_table.FRAME0, help="the F-engine starts"
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
