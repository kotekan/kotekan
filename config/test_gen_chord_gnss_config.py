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
import copy
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
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
from gnss_broker import iface  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def node_cfg():
    with open(g.NODE_FILE) as fh:
        return yaml.safe_load(fh)


def run_gen(*argv, env=None):
    """The generator as a subprocess from the repo root, with env added to this environment:
    (returncode, stdout, stderr)."""
    p = subprocess.run(
        [sys.executable, GEN, *argv],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=None if env is None else dict(os.environ, **env),
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


def test_headers_carry_the_iface_stamp():
    """Every config written names the interface golden its frames were sized from, on the
    header's second line; the deploy preflight compares a binary's sizing sections with it."""
    line = iface.stamp_line(iface.golden())
    for flags in (
        ["--gather-instance"],
        ["--cube-archive-instance"],
        ["--aggregator-instance", "cx19", "--aggregator-cores", "0,1,2,3,4,5"],
    ):
        rc, out, _ = run_gen("--base", BASE, "--node", "cx19", *flags)
        check(
            rc == 0 and out.splitlines()[1] == line and iface.parse_stamp(out),
            "%s: the header carries the gnss_iface stamp (rc %d)" % (flags[0], rc),
        )
    if kotekan_src.SRC is None:
        print("  [SKIP] node config: no kotekan tree to render the base from")
        return
    # A node config, its live reads (EOP table, bad inputs, frame 0) stubbed out.
    names = (
        "choco_eop_table",
        "live_eop_table",
        "choco_bad_inputs",
        "live_bad_inputs",
        "live_frame0_utc",
    )
    real = {n: getattr(g, n) for n in names}
    for n in names:
        setattr(g, n, lambda *a, **k: None)
    try:
        base = kotekan_src.src("config", "chord/pathfinder.j2")
        argv = ["--base", base, "--node", "cx19", "--telem-host", "gnss"]
        exc, out, _ = run_main(argv + ["--cube-host", "cf06"])
    finally:
        for n, f in real.items():
            setattr(g, n, f)
    check(
        exc is None and out.splitlines()[1] == line,
        "a node config carries the gnss_iface stamp (%s)" % exc,
    )


def test_unknown_descriptor_is_refused():
    for flags, where in (
        (["--gather-instance", "--extra-signal", "GPS_L9_X:1,2"], "--extra-signal"),
        (
            ["--aggregator-instance", "cx19", "--aggregator-extra-search", "GPS_L9_X"],
            "--aggregator-extra-search",
        ),
    ):
        rc, _, err = run_gen("--base", BASE, "--node", "cx19", *flags)
        check(
            rc != 0
            and "%s GPS_L9_X: not a descriptor in the interface golden" % where in err,
            "%s GPS_L9_X is refused, naming it (rc %d)" % (where, rc),
        )
    cfg = node_cfg()
    cfg["signals"]["tracker"] = "GPS_L5_Q_NHX"
    args = argparse.Namespace(extra_signal=None, aggregator_extra_search=None)
    try:
        g.check_descriptors(cfg, args)
        msg = None
    except SystemExit as e:
        msg = str(e)
    check(
        msg is not None and msg.startswith("signals.tracker GPS_L5_Q_NHX:"),
        "a tracker descriptor the golden lacks is refused (%s)" % msg,
    )
    cfg = node_cfg()
    args = argparse.Namespace(
        extra_signal=["SBAS_L5_I_MC:131"], aggregator_extra_search=["GAL_E5A_Q"]
    )
    g.check_descriptors(cfg, args)  # every name known: no exit


def test_a_kotekan_golden_that_moved_is_warned():
    """The frames are sized from the vendored golden. A kotekan tree whose own golden differs
    from it in signals or a sizing section is warned, naming the sync; the config is the same.
    Nothing is said when the two agree, when they differ elsewhere, or with no golden there."""
    d = tempfile.mkdtemp(prefix="gen_tree_golden_")
    path = os.path.join(d, iface.KOTEKAN_GOLDEN)
    real = kotekan_src.SRC

    def said(doc):
        """What the check says of a tree holding doc as its golden (None: no golden)."""
        if doc is not None:
            with open(path, "w") as fh:
                fh.write(iface.format_golden(doc))
        elif os.path.exists(path):
            os.unlink(path)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            g.check_tree_golden()
        return err.getvalue()

    try:
        os.makedirs(os.path.dirname(path))
        kotekan_src.SRC = d
        doc = copy.deepcopy(iface.golden().doc)
        check(said(doc) == "", "the vendored golden itself: nothing said")
        doc["sections"]["record.cmb"]["CMB_ARC"]["meaning"] += ";x"
        check(said(doc) == "", "a change outside signals and sizing: nothing said")
        for section, key in (
            ("record.elem", "ELEM_FLOATS"),
            ("record.row", "RECORD_FLOATS"),
        ):
            moved = copy.deepcopy(doc)
            moved["sections"][section][key] += 1
            err = said(moved)
            check(
                err.startswith("WARNING: ")
                and " in %s. " % section in err
                and "python3 -m gnss_broker.iface sync --from %s" % d in err,
                "%s moved: warned, naming the sync (%r)" % (section, err[:200]),
            )
        doc["sections"]["record.row"]["RECORD_FLOATS"] += 1
        check(said(None) == "", "no golden in the tree: nothing said")
        kotekan_src.SRC = None
        check(said(None) == "", "no kotekan tree: nothing said")
        with open(path, "w") as fh:
            fh.write(iface.format_golden(doc))
        flags = ("--base", BASE, "--node", "cx19", "--gather-instance")
        rc0, out0, err0 = run_gen(*flags)
        rc1, out1, err1 = run_gen(*flags, env={"KOTEKAN_SRC": d})
        check(
            rc0 == rc1 == 0
            and out1 == out0
            and "sync --from" not in err0
            and "sync --from %s" % d in err1,
            "the generator warns and writes the same config (rc %d, %d)" % (rc0, rc1),
        )
    finally:
        kotekan_src.SRC = real
        shutil.rmtree(d, ignore_errors=True)


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


def test_psi_synth_needs_epl():
    """--psi-synth is refused without --epl-from-prompt, before anything is probed."""
    with fleet(FakeFleet(None)) as fake:
        exc, _, _ = run_main(["--base", BASE, "--node", "cx19", "--psi-synth"])
    check(
        exc is not None and "--epl-from-prompt" in exc,
        "--psi-synth alone is refused naming --epl-from-prompt (%r)" % (exc,),
    )
    check(not fake.urls, "the refusal probes nothing")


def test_paths_in_the_kotekan_tree():
    """A recipe records a file in the kotekan tree relative to that tree, wherever the tree
    is, and a relative input not there from the cwd is looked for in it."""
    saved = g.kotekan_src.SRC
    with tempfile.TemporaryDirectory() as d:
        kot = os.path.realpath(d)
        os.makedirs(os.path.join(kot, "config", "chord"))
        open(os.path.join(kot, "config", "chord", "only_here.j2"), "w").close()
        try:
            g.kotekan_src.SRC = kot
            j2 = os.path.join(kot, "config", "chord", "pathfinder.j2")
            check(
                g._rel(j2) == "config/chord/pathfinder.j2",
                "_rel: a file in a separate kotekan tree is relative to it",
            )
            check(
                g._rel(GEN) == "config/gen_chord_gnss_config.py",
                "_rel: a file in this repo is still relative to the repo",
            )
            outside = os.path.join(os.path.dirname(kot), "elsewhere.j2")
            check(g._rel(outside) == outside, "_rel: a file in neither stays absolute")
            g.kotekan_src.SRC = os.path.join(ROOT, "kotekan")
            check(
                g._rel(os.path.join(ROOT, "kotekan", "config", "x.j2"))
                == "config/x.j2",
                "_rel: a kotekan submodule inside the repo wins over the repo",
            )
            g.kotekan_src.SRC = kot
            rel = "config/chord/only_here.j2"
            check(
                g._kotekan_input(rel) == os.path.join(kot, rel),
                "--base: relative and not here, found in the kotekan tree",
            )
            check(
                g._kotekan_input("config/chord/no_such.j2")
                == "config/chord/no_such.j2",
                "--base: in neither place, left as given",
            )
            check(g._kotekan_input(GEN) == GEN, "--base: absolute, left as given")
        finally:
            g.kotekan_src.SRC = saved


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
        out2, feeds2, _ = g.build_aggregator_instance(
            cfg, ["cx19:11040", "cx27:11046"], args, 12050
        )
    finally:
        g.covering_channels = real
    ports = [out["agg%d_recv" % i]["listen_port"] for i in range(len(feeds))]
    check(
        ports == [11040, 11042, 11043],
        "a node missing a GPU leaves its port unused, later nodes keep base+2k+g (got %s)"
        % ports,
    )
    ports2 = [out2["agg%d_recv" % i]["listen_port"] for i in range(len(feeds2))]
    check(
        ports2 == [11040, 11046, 11047],
        "NODE:BASE puts that node's GPU g on BASE + g, whatever its position (got %s)"
        % ports2,
    )
    try:
        g.aggregator_node("cx19:x", 11040)
        check(False, "a malformed NODE:BASE is refused")
    except SystemExit as e:
        check("NODE:PORT_BASE" in str(e), "a malformed NODE:BASE is refused")


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
    # Two stock reference hosts whatever the fleet list holds today.
    cfg = node_cfg()
    real_refs = g.stock_reference_hosts
    g.stock_reference_hosts = lambda cfg: ["cx47", "cx52"]
    try:
        _probe_ports(cfg)
    finally:
        g.stock_reference_hosts = real_refs


def _probe_ports(cfg):
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
    check(
        not refused([[0, 63], [64, 127]]), "128 elements, one range per pol: accepted"
    )
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
    if kotekan_src.SRC is None:
        print("  [SKIP] core pools: no kotekan tree holding production's pathfinder.j2")
        return
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


def test_bad_inputs_reach_the_gnss_branch():
    """bffs's flagged inputs leave the GNSS branch's static lists: the projection stations, the
    channel-noise inputs and the fleet reference's entries (zeroed, length kept); one fetch per
    run, kept on args."""
    block = {"bad_inputs": [5, 69, 120], "start_time": 1.0, "update_id": "t"}
    real = (g.choco_bad_inputs, g.live_bad_inputs, g.gnss_arraymap.manual_bad_inputs)
    calls = []
    g.choco_bad_inputs = lambda *a, **k: (calls.append(1), (dict(block), "stub"))[1]
    g.live_bad_inputs = lambda *a, **k: None
    g.gnss_arraymap.manual_bad_inputs = lambda *a, **k: set()
    try:
        args = argparse.Namespace()
        bad = g.bad_input_set(_ARR128, args)
        g.bad_input_set(_ARR128, args)
        check(
            bad == {5, 69, 120} and len(calls) == 1, "one fetch per run, kept on args"
        )
        st = g.project_stations("0-6,8-127", bad)
        check(
            st == sorted((set(range(0, 7)) | set(range(8, 128))) - bad),
            "flagged inputs leave the projection stations (%d left)" % len(st),
        )
        d = tempfile.mkdtemp()
        try:
            snap = os.path.join(d, "ref.json")
            with open(snap, "w") as fh:
                json.dump(
                    {"epoch": "E", "bands": {"l5": {"ref": [[0.1, 0.2]] * 128}}}, fh
                )
            a = argparse.Namespace(
                elem_shared_ref_mode="live", elem_sum_shared=True, elem_shared_ref=snap
            )
            keys = g.elem_shared_ref_keys(
                a, "l5", {"elem_positions_epoch": "E"}, 128, bad
            )
            ref = keys["elem_sum_shared_ref"]
            zeros = {
                e for e in range(128) if ref[2 * e] == 0.0 and ref[2 * e + 1] == 0.0
            }
            check(
                len(ref) == 256 and zeros == bad and ref[0] == 0.1,
                "flagged entries of the fleet reference are zero, the rest kept (%s)"
                % sorted(zeros),
            )
        finally:
            shutil.rmtree(d)
        a2 = argparse.Namespace()
        g.choco_bad_inputs = lambda *a, **k: None
        check(g.bad_input_set(_ARR128, a2) == set(), "no source: nothing excluded")
    finally:
        g.choco_bad_inputs, g.live_bad_inputs, g.gnss_arraymap.manual_bad_inputs = real


_ARR128 = {"array": {"live_element_ranges": [[0, 63], [64, 127]]}}


def test_bad_inputs_from_file_and_outage_guard():
    """--bad-inputs-from takes the block from a file (gen_fleet --check passes the one the
    config on disk carries) without asking bffs; the recipe header leaves it out; and a list
    flagging more than half the inputs (no data, not bad feeds) refuses the generation rather
    than empty the GNSS branch's element lists."""
    real = (g.choco_bad_inputs, g.live_bad_inputs, g.gnss_arraymap.manual_bad_inputs)
    g.choco_bad_inputs = lambda *a, **k: check(
        False, "bffs asked despite --bad-inputs-from"
    )
    g.live_bad_inputs = lambda *a, **k: None
    g.gnss_arraymap.manual_bad_inputs = lambda *a, **k: set()
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "bi.json")
        with open(path, "w") as fh:
            json.dump(
                {"bad_inputs": [3, 70], "start_time": 1.0, "update_id": "disk"}, fh
            )
        a = argparse.Namespace(bad_inputs_from=path)
        check(g.bad_input_set(_ARR128, a) == {3, 70}, "the block comes from the file")
        check(g.bad_input_block(_ARR128, a)[1] == path, "its source is the file")
        rec = "\n".join(
            g.recipe_lines(["--node", "cx19", "--bad-inputs-from", path, "--out", "x"])
        )
        check(
            "--bad-inputs-from" not in rec and path not in rec and "--node cx19" in rec,
            "the recipe header leaves --bad-inputs-from out (%s)" % rec,
        )
        with open(path, "w") as fh:
            json.dump({"bad_inputs": list(range(65)), "update_id": "outage"}, fh)
        try:
            g.bad_input_set(_ARR128, argparse.Namespace(bad_inputs_from=path))
            check(False, "65 of 128 flagged was applied")
        except SystemExit as e:
            check(
                "outage" in str(e) and "65 of 128" in str(e),
                "an outage list refuses: %s" % e,
            )
        with open(path, "w") as fh:
            json.dump({"bad_inputs": list(range(64)), "update_id": "half"}, fh)
        check(
            len(g.bad_input_set(_ARR128, argparse.Namespace(bad_inputs_from=path)))
            == 64,
            "exactly half is still applied",
        )
        with open(path, "w") as fh:
            json.dump({"not": "a block"}, fh)
        try:
            g.bad_input_set(_ARR128, argparse.Namespace(bad_inputs_from=path))
            check(False, "a file without bad_inputs was accepted")
        except SystemExit as e:
            check("not a bad-input block" in str(e), "a malformed file is refused")
    finally:
        shutil.rmtree(d)
        g.choco_bad_inputs, g.live_bad_inputs, g.gnss_arraymap.manual_bad_inputs = real


_LIVE_NAMES = (
    "choco_eop_table",
    "live_eop_table",
    "choco_bad_inputs",
    "live_bad_inputs",
    "live_frame0_utc",
)


@contextlib.contextmanager
def stubbed(mask, **stubs):
    """The live reads stubbed (None unless given in stubs) and config/gnss_bad_inputs.json
    read as `mask`."""
    real = {n: getattr(g, n) for n in _LIVE_NAMES}
    real_mask = g.gnss_arraymap.manual_bad_inputs
    for n in _LIVE_NAMES:
        setattr(g, n, stubs.get(n, lambda *a, **k: None))
    g.gnss_arraymap.manual_bad_inputs = lambda *a, **k: set(mask)
    try:
        yield
    finally:
        for n, f in real.items():
            setattr(g, n, f)
        g.gnss_arraymap.manual_bad_inputs = real_mask


_HY = set(range(120, 128))


def test_manual_mask_joins_bffs():
    """config/gnss_bad_inputs.json's inputs leave the GNSS branch's lists with bffs's: it
    applies when no bffs source answers, and the outage guard counts bffs's list only."""
    block = {"bad_inputs": [5, 69], "start_time": 1.0, "update_id": "t"}
    with stubbed(_HY, choco_bad_inputs=lambda *a, **k: (dict(block), "stub")):
        check(
            g.bad_input_set(_ARR128, argparse.Namespace()) == {5, 69} | _HY,
            "the mask joins bffs's list",
        )
    with stubbed(_HY):
        check(
            g.bad_input_set(_ARR128, argparse.Namespace()) == _HY,
            "no bffs source: the mask still applies",
        )
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "bi.json")
        for n, ok in ((64, True), (65, False)):
            with open(path, "w") as fh:
                json.dump({"bad_inputs": list(range(n)), "update_id": "u"}, fh)
            try:
                with stubbed(_HY):
                    got = g.bad_input_set(
                        _ARR128, argparse.Namespace(bad_inputs_from=path)
                    )
                check(
                    ok and len(got) == n + 8, "%d from bffs + 8 masked is applied" % n
                )
            except SystemExit as e:
                check(not ok, "%d from bffs + 8 masked refused: %s" % (n, e))
    finally:
        shutil.rmtree(d)


def test_manual_mask_reaches_the_node_config():
    """One node config with the mask and a bffs block: the stations, the channel-noise inputs
    and the shared ref's zeros leave out both; updatable_config/bad_inputs, the science mask
    bffs owns, carries bffs's list only."""
    if kotekan_src.SRC is None:
        print("  [SKIP] node config: no kotekan tree to render the base from")
        return
    sys.path.insert(0, os.path.join(ROOT, "python", "scripts", "gnss"))
    import gnss_arraymap

    key = gnss_arraymap.at(time.time()).key()
    d = tempfile.mkdtemp()
    try:
        bi = os.path.join(d, "bi.json")
        with open(bi, "w") as fh:
            json.dump({"bad_inputs": [5, 69], "start_time": 1.0, "update_id": "t"}, fh)
        ref = os.path.join(d, "ref.json")
        with open(ref, "w") as fh:
            json.dump({"epoch": key, "bands": {"l5": {"ref": [[0.1, 0.2]] * 128}}}, fh)
        argv = [
            "--base",
            kotekan_src.src("config", "chord/pathfinder.j2"),
            "--node",
            "cx51",
            "--telem-host",
            "gnss",
            "--cube-host",
            "cf06",
            "--bad-inputs-from",
            bi,
            "--n2-project",
            "live",
            "--n2-project-stations",
            "0-6,8-127",
            "--rfi-excision-exempt-freq-ids",
            "5971-6076",
            "--n2-project-live-freq-ids",
            "exempt",
            "--n2-project-source",
            "lanes+residual",
            "--elem-dir-publish",
            "all",
            "--elem-sum-shared",
            "--elem-shared-ref",
            ref,
            "--elem-shared-ref-mode",
            "live",
        ]
        with stubbed(_HY):
            exc, out, err = run_main(argv)
    finally:
        shutil.rmtree(d)
    check(exc is None, "the node config generates (%s)" % exc)
    if exc is not None:
        return
    cfg = yaml.safe_load(out)
    out_set = {5, 69} | _HY
    st = cfg["gnss_n2_project"]["proj_0"]["stations"]
    check(
        st == sorted(set(range(128)) - {7} - out_set),
        "the stations leave out bffs's and the mask's inputs (%d)" % len(st),
    )
    asm = cfg["gnss0_n2assemble"]
    check(
        asm["chan_noise_norm_elems"] == sorted(set(range(128)) - out_set),
        "the channel-noise inputs leave them out",
    )
    r = asm["elem_sum_shared_ref"]
    check(
        {e for e in range(128) if r[2 * e] == r[2 * e + 1] == 0.0} == out_set,
        "the shared ref's entries for them are zero",
    )
    check(
        cfg["updatable_config"]["bad_inputs"]["bad_inputs"] == [5, 69],
        "updatable_config/bad_inputs is bffs's list, without the mask",
    )
    check(
        "1 tracker elements are outside" in err, "the tracker warning counts only B04X",
    )


def main():
    # The generator reads relative paths from the cwd and _rel() resolves every argv token
    # against it, so the in-process calls run from the repo root, as gen_fleet and run_gen do.
    os.chdir(ROOT)
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
