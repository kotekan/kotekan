#!/usr/bin/env python3
"""gen_chord_gnss_config.py's own checks: the off-node instances' defaults and headers, the
aggregator's feed ports, the record-epoch check and which ports the running fleet is asked on.

    python3 config/test_gen_chord_gnss_config.py

Hermetic: the instance branches run before any EOP or bad-input probe, an unpinned epoch probes
nothing, and every probe that does run here goes to a stubbed urllib.request.urlopen.
"""
import argparse
import ast
import contextlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEN = os.path.join(HERE, "gen_chord_gnss_config.py")
BASE = "config/base/live_config_20260730.json"

sys.path.insert(0, HERE)
import gen_chord_gnss_config as g  # noqa: E402

sys.path.append(os.path.join(ROOT, "python", "scripts", "gnss"))
import kotekan_src  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def node_cfg():
    with open(g.NODE_FILE) as fh:
        return yaml.safe_load(fh)


def run_gen(*argv):
    """The generator as a subprocess from the repo root: (returncode, stdout, stderr)."""
    p = subprocess.run(
        [sys.executable, GEN, *argv], capture_output=True, text=True, cwd=ROOT
    )
    return p.returncode, p.stdout, p.stderr


def recipe_tokens(text):
    """The REGENERATE WITH recipe in a generated header, split back into argv."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# REGENERATE WITH"))
    body = []
    for ln in lines[i + 1 :]:
        if not ln.startswith("#   "):
            break
        body.append(ln[1:].strip())
    return shlex.split(" ".join(body))


class FakeFleet:
    """urllib.request.urlopen for a fleet that serves time0_ns = `ns` everywhere (None: nobody
    answers). Every URL asked is recorded."""

    def __init__(self, ns=None):
        self.ns, self.urls = ns, []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        if self.ns is None:
            raise OSError("no answer")
        body = json.dumps({"time0_ns": self.ns}).encode()

        class _R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _R(body)


@contextlib.contextmanager
def fleet(fake):
    real = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        yield fake
    finally:
        urllib.request.urlopen = real


def run_main(argv):
    """main() in process: (SystemExit message or None, stdout, stderr)."""
    out, err, exc = io.StringIO(), io.StringIO(), None
    saved = sys.argv
    sys.argv = [GEN] + argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            g.main()
    except SystemExit as e:
        exc = str(e.code)
    finally:
        sys.argv = saved
    return exc, out.getvalue(), err.getvalue()


def instance_args(**kw):
    a = dict(
        rest_port=None,
        aggregator_instance=None,
        gather_instance=False,
        cube_archive_instance=False,
        frame0_nano=None,
    )
    a.update(kw)
    return argparse.Namespace(**a)


def test_gather_defaults():
    rc, out, err = run_gen("--base", BASE, "--node", "cx19", "--gather-instance")
    check(rc == 0, "gather instance generates (rc %d)" % rc)
    cfg = yaml.safe_load(out)
    n_win = cfg["fleet_trim"]["n_win"]
    check(n_win == 2, "gather fleet_trim.n_win defaults to 2 (got %s)" % n_win)
    check("rest     12051" in err, "gather REST port defaults to 12051")
    check(
        "WARNING" not in err and "unverified" not in err,
        "an unpinned epoch prints no frame0 warning",
    )


def test_cube_default_port():
    rc, _, err = run_gen(
        "--base",
        "config/base/live_config_20260831.json",
        "--node",
        "cx19",
        "--cube-archive-instance",
    )
    check(rc == 0, "cube archiver generates (rc %d)" % rc)
    check("rest     12052" in err, "cube archiver REST port defaults to 12052")


def test_instance_refuses_aggregator_port():
    for flag in ("--gather-instance", "--cube-archive-instance"):
        rc, _, err = run_gen(
            "--base", BASE, "--node", "cx19", flag, "--rest-port", "12050"
        )
        check(
            rc != 0 and "refusing port 12050" in err,
            "%s refuses the aggregator's port 12050" % flag,
        )


def test_gather_recipe():
    cores = ",".join(str(c) for c in range(64))
    with tempfile.TemporaryDirectory() as d:
        dest = os.path.join(d, "gather.yaml")
        argv = [
            "--base",
            BASE,
            "--node",
            "cx19",
            "--gather-instance",
            "--rest-port",
            "12051",
            "--gather-cores",
            cores,
        ]
        rc, _, _ = run_gen(*argv[:5], "--out", dest, *argv[5:])
        check(rc == 0, "gather instance writes --out (rc %d)" % rc)
        with open(dest) as fh:
            got = recipe_tokens(fh.read())
    check(
        got == ["config/gen_chord_gnss_config.py"] + argv,
        "gather header recipe splits back to argv, whole tokens, no --out",
    )


def test_recipe_lines():
    prns = ",".join(str(p) for p in range(1, 37))
    argv = [
        "--base",
        "config/chord/pathfinder.j2",
        "--node",
        "cx19",
        "--extra-signal",
        "GAL_E5A_Q:" + prns,
        "--out",
        "x.yaml",
        "--out=y.yaml",
        "--telem-host",
        "h",
    ]
    lines = g.recipe_lines(argv)
    toks = shlex.split(" ".join(ln[1:].strip() for ln in lines))
    want = [
        "config/gen_chord_gnss_config.py",
        "--base",
        "config/chord/pathfinder.j2",
        "--node",
        "cx19",
        "--extra-signal",
        "GAL_E5A_Q:" + prns,
        "--telem-host",
        "h",
    ]
    check(toks == want, "recipe_lines drops --out and keeps the PRN list whole")
    check(
        all(ln.startswith("#   ") for ln in lines), "recipe_lines are '#   ' comments"
    )


def test_aggregator_feed_ports():
    cfg = node_cfg()
    # cx19 (combs 4, 14) holds 6020 on GPU 0 only; cx27 (combs 0, 8) holds 6016 and 6024.
    want = {6016, 6020, 6024}
    real = g.covering_channels
    g.covering_channels = lambda chans, *a, **k: [c[0] for c in chans if c[0] in want]
    try:
        args = argparse.Namespace(
            aggregator_cores="0,1,2,3,4,5",
            search_port_base=11040,
            aggregator_extra_search=None,
            cuda_acquire=False,
        )
        out, feeds, _ = g.build_aggregator_instance(cfg, ["cx19", "cx27"], args, 12050)
    finally:
        g.covering_channels = real
    ports = [out["agg%d_recv" % i]["listen_port"] for i in range(len(feeds))]
    check(
        ports == [11040, 11042, 11043],
        "a node missing a GPU leaves its port unused, later nodes keep base+2k+g (got %s)"
        % ports,
    )


def test_aggregator_spds():
    cfg = node_cfg()
    args = argparse.Namespace(
        aggregator_cores="0,1,2,3,4,5",
        search_port_base=11040,
        aggregator_extra_search=None,
        cuda_acquire=False,
    )
    out, _, _ = g.build_aggregator_instance(cfg, ["cx19"], args, 12050, spds=4096)
    check(
        out["samples_per_data_set"] == 4096,
        "aggregator takes samples_per_data_set from the base",
    )
    with open(os.path.join(ROOT, BASE)) as fh:
        base = json.load(fh)
    base["samples_per_data_set"] = 4096
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "base.json")
        with open(path, "w") as fh:
            json.dump(base, fh)
        rc, out, _ = run_gen(
            "--base",
            path,
            "--node",
            "cx19",
            "--aggregator-instance",
            "cx19",
            "cx27",
            "--aggregator-cores",
            "0,1,2,3,4,5",
        )
    spds = yaml.safe_load(out)["samples_per_data_set"] if rc == 0 else None
    check(spds == 4096, "--aggregator-instance takes --base's samples_per_data_set")


def test_frame0_nano_matches_fleet():
    ns = 1_700_000_000_123_456_789
    argv = [
        "--base",
        BASE,
        "--node",
        "cx19",
        "--gather-instance",
        "--rest-port",
        "12051",
        "--gather-cores",
        "0",
        "--frame0-nano",
        str(ns),
    ]
    with fleet(FakeFleet(ns)) as f:
        exc, out, _ = run_main(argv)
    check(exc is None, "--frame0-nano that the fleet serves is accepted (%s)" % exc)
    check(
        f.urls and all(":12048/" in u for u in f.urls),
        "an instance run asks the nodes on runtime.rest_port, not its own (asked %s)"
        % sorted({u.split("/")[2] for u in f.urls}),
    )
    with fleet(FakeFleet(ns + 10 ** 9)):
        exc, _, _ = run_main(argv)
    check(
        exc is not None and "--frame0-nano" in exc,
        "--frame0-nano 1 s off the fleet's is refused, naming the flag",
    )


def test_check_frame0():
    cfg = node_cfg()
    cfg["fengine"].pop("frame0_utc", None)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        g.check_frame0(cfg, instance_args(), None)
    check("WARNING" not in err.getvalue(), "check_frame0: no pin, no warning")
    cfg["fengine"]["frame0_utc"] = 1.7e9
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        g.check_frame0(cfg, instance_args(), None)
    check(
        "unverified" in err.getvalue(),
        "check_frame0: an unconfirmed node-file pin warns",
    )
    try:
        g.check_frame0(cfg, instance_args(), 1.7e9 + 5.0)
        check(False, "check_frame0: a node-file pin 5 s off the fleet's is refused")
    except SystemExit as e:
        check(
            "sed -i" in str(e.code),
            "check_frame0: a node-file pin 5 s off the fleet's is refused",
        )


def test_probe_ports():
    cfg = node_cfg()
    with fleet(FakeFleet()) as f:
        g.live_eop_table(cfg, instance_args(rest_port=12049), timeout=0.01)
    hosts = {u.split("/")[2] for u in f.urls}
    check(
        "cx19:12049" in hosts, "node config: our nodes asked for EOP on its --rest-port"
    )
    check(
        "cx47:12048" in hosts and "cx52:12048" in hosts,
        "node config: the stock reference nodes asked for EOP on 12048 (asked %s)"
        % sorted(hosts),
    )
    check(
        not any(h.startswith("bringup_order") for h in hosts),
        "bringup_order is not probed as a host",
    )
    with fleet(FakeFleet()) as f:
        g.live_bad_inputs(cfg, timeout=0.01)
    hosts = sorted({u.split("/")[2] for u in f.urls})
    check(
        hosts == ["cx47:12048", "cx52:12048"],
        "bad inputs asked of the stock nodes on 12048 (asked %s)" % hosts,
    )


def test_node_names():
    rc, _, err = run_gen("--base", BASE, "--node", "bringup_order", "--gather-instance")
    check(
        rc != 0 and "unknown node bringup_order" in err,
        "--node bringup_order is refused as an unknown node",
    )
    names = g.node_names(node_cfg())
    check("bringup_order" not in names and "cx19" in names, "node_names: hosts only")


def test_live_ranges_pol_mirror():
    def refused(ranges):
        try:
            g.live_element_ranges({"live_element_ranges": ranges})
        except SystemExit:
            return True
        return False

    check(not refused([[0, 15], [64, 79]]), "32 elements, one range per pol: accepted")
    check(not refused([[0, 47], [64, 111]]), "96 elements, one range per pol: accepted")
    check(refused([[0, 31], [64, 79]]), "48 elements (unequal pol halves): refused")
    check(refused([[0, 15], [16, 31]]), "two pol-0 ranges: refused")
    check(refused([[0, 63]]), "a single range: refused")
    check(refused([[0, 79], [64, 143]]), "a range past the pol boundary: refused")


def test_load_summary():
    cfg = {
        "runtime": {
            "cpu_affinity": [1, 2, 3],
            "cpu_affinity_gpu0": list(range(10, 20)),
            "cpu_affinity_gpu1": list(range(20, 30)),
        }
    }
    out = {
        "gnss0_a": {"kotekan_stage": "A", "cpu_affinity": [10]},
        "gnss0_b": {"kotekan_stage": "B", "cpu_affinity": [10]},
        "gnss1_c": {"kotekan_stage": "C", "cpu_affinity": [20]},
    }
    s = g.load_summary(out, cfg, [0, 1])
    check(
        s.endswith("(pool of 20)"), "load summary sizes the per-GPU pools (got %r)" % s
    )
    check(s.startswith("core 10: 2 stages"), "load summary leads with the busiest core")


def production_cores():
    """Every logical cpu production's template pins (pathfinder.j2): the numa_halves entries,
    the DPDK lcores and the instance's root cpu_affinity, and the DPDK lcores' HT siblings."""
    src = open(kotekan_src.src("config", "chord", "pathfinder.j2")).read()
    halves = ast.literal_eval(
        re.search(r"\{% set numa_halves = (\[.*?\]) %\}", src, re.S).group(1)
    )
    cores = set()
    for h in halves:
        for k, v in h.items():
            if k.endswith("_cpu"):
                cores.add(v)
            elif k.endswith("_cpus"):
                cores.update(v)
    lcores = [
        int(c)
        for c in re.search(r"lcore_cpu_map: \[([0-9, ]+)\]", src).group(1).split(",")
    ]
    cores.update(lcores)
    cores.add(int(re.search(r"main_lcore_cpu: (\d+)", src).group(1)))
    root = re.search(r"^cpu_affinity: \[([0-9, ]+)\]", src, re.M).group(1)
    cores.update(int(c) for c in root.split(","))
    return cores, {c + 32 for c in lcores}


def test_core_pools_clear_production():
    """The GNSS pools and the node's REST cores never share a logical cpu with production, and
    each GPU pool sits on its GPU's NUMA node with a length coprime to dual_core's stride."""
    prod, dpdk_sib = production_cores()
    rt = node_cfg()["runtime"]
    numa = {
        0: set(range(0, 16)) | set(range(32, 48)),
        1: set(range(16, 32)) | set(range(48, 64)),
    }
    for gpu in (0, 1):
        pool = rt["cpu_affinity_gpu%d" % gpu]
        check(not set(pool) - numa[gpu], "GPU%d pool is on NUMA node %d" % (gpu, gpu))
        check(
            len(pool) % 3 != 0,
            "GPU%d pool length %d is coprime to 3" % (gpu, len(pool)),
        )
    for key in ("cpu_affinity_gpu0", "cpu_affinity_gpu1", "cpu_affinity"):
        clash = sorted(set(rt[key]) & (prod | dpdk_sib))
        check(
            not clash,
            "%s clears production's cores (shared: %s)" % (key, clash or "none"),
        )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
