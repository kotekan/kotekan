#!/usr/bin/env python3
"""Run every GNSS Python test, each under its interpreter and from its working directory.

    /usr/bin/python3 scripts/gnss/run_tests.py [--root DIR] [--out DIR] [-j N] [--only REGEX]
                                               [--timeout S] [--strict] [--list]

Tests are found by glob, so a new one runs without being named here: test_*.py under
python/scripts/gnss, its gnss_broker/ (as `python -m gnss_broker.<name>` from
python/scripts/gnss), beam_cube_viewer/, python/scripts/js_viewer, scripts/gnss,
scripts/gnss/site and config/, tests/test_gps_*.py and tests/test_beam_cube.py, and every
*_code_check.py. Added to those: gnss_broker/selftest.py, the checks in CHECKS, check_js.sh,
ft_stress.py and its --selftest, and the broker's --list smoke run. Left out: the gates that
build or run kotekan or bind fixed ports (fleetdll_gate, fleettrim_multichain_gate,
rf_stats_gate) and those that read the live fleet (site/port_axis_gate,
site/stack_contract_gate).

The tables below pick the interpreter; a file they do not name runs under the venv:

    GNSS_VENV     /home/kvand/gnss/venv     CPython 3.12
    GNSS_VENV_FT  /home/kvand/gnss/venv-ft  CPython 3.14t: the broker package, the tests whose
                                            docstrings name it, the dukpy tests (PYTHON_GIL=0)
    GNSS_SYS_PY   /usr/bin/python3          the tests of the scripts cron runs with it

Each job runs with PYTHONDONTWRITEBYTECODE=1, the broker's BLAS thread caps, and
GNSS_REQUIRE_CXX, GNSS_REQUIRE_NODE and GNSS_REQUIRE_PLOT set to 1 (GNSS_REQUIRE_FT too under
venv-ft), so a missing compiler, node, plotting library or free-threaded build fails instead
of skipping. A value already in the environment wins, and an empty one lifts the requirement.
GNSS_REQUIRE_GPU and KOTEKAN_SRC pass through: set GNSS_REQUIRE_GPU=1 on a GPU host with an
e2e build.

A job PASSes on exit 0, SKIPs on exit 77 (its environment is absent) or on exit 0 with
unittest cases skipped, TIMEOUTs past --timeout (its process group is killed), and FAILs
otherwise. A table entry naming a file that is not there FAILs as stale, unless the file is one
of the C++ verifiers that stay in kotekan. Exit status: 1 on any FAIL or TIMEOUT, and on any
SKIP under --strict; 2 on a usage error or a missing interpreter.
"""
import argparse
import collections
import concurrent.futures
import glob
import importlib.util
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.dont_write_bytecode = True

PY = {
    "venv": os.path.join(
        os.environ.get("GNSS_VENV", "/home/kvand/gnss/venv"), "bin", "python"
    ),
    "ft": os.path.join(
        os.environ.get("GNSS_VENV_FT", "/home/kvand/gnss/venv-ft"), "bin", "python"
    ),
    "sys": os.environ.get("GNSS_SYS_PY", "/usr/bin/python3"),
    "sh": "/bin/sh",
}

# dukpy is installed only in venv-ft, and importing it re-enables the GIL unless PYTHON_GIL=0.
NO_GIL = {
    "python/scripts/gnss/beam_cube_viewer/test_beamcube_currency.py",
    "python/scripts/js_viewer/test_amp_history.py",
    "python/scripts/js_viewer/test_lock_metrics.py",
}
FT = NO_GIL | {
    "python/scripts/gnss/test_beam_cube_build.py",
    "python/scripts/gnss/test_chain_health.py",
    "python/scripts/gnss/test_code_check_cpp.py",
    "python/scripts/gnss/test_cube_read.py",
    "python/scripts/gnss/test_elem_shared_ref.py",
    "python/scripts/gnss/test_hoprate_validate.py",
    "python/scripts/gnss/test_tec_chord.py",
}
# The cron scripts' tests, under /usr/bin/python3 as cron runs them (with apt's yaml).
SYS = {
    "scripts/gnss/site/test_bad_inputs_cron.py",
    "scripts/gnss/site/test_tle_refresh.py",
}
# Checks no test runs: module self-checks (a __main__ that exits non-zero on a failure), and
# trim_sign_gate, which needs an e2e build and a GPU and exits 77 without them.
CHECKS = [
    "python/scripts/gnss/beidou_bcnav1.py",
    "python/scripts/gnss/beidou_bcnav2.py",
    "python/scripts/gnss/beidou_bcnav3.py",
    "python/scripts/gnss/cnav2_predictor.py",
    "python/scripts/gnss/cnav_predictor.py",
    "python/scripts/gnss/galileo_fnav.py",
    "python/scripts/gnss/gps_cnav2.py",
    "python/scripts/gnss/navbit_predictor.py",
    "scripts/gnss/rf_panel_gate.py",
    "scripts/gnss/trim_sign_gate.py",
    "scripts/gnss/viscap_selftest.py",
]
# The Python that only verifies kotekan's C++, and stays in kotekan when the rest moves.
KOTEKAN_ONLY = {
    "python/scripts/gnss/%s_code_check.py" % n
    for n in "b1i_b3i b2b e5b e6 glonass_ca l1cd l2oc l3oc".split()
} | {
    "python/scripts/gnss/test_code_check_cpp.py",
    "scripts/gnss/trim_sign_gate.py",
    "scripts/gnss/test_trim_sign_gate.py",
    "scripts/gnss/test_e2e_exact_seed.py",
}
TEST_DIRS = [
    "python/scripts/gnss",
    "python/scripts/gnss/beam_cube_viewer",
    "python/scripts/js_viewer",
    "scripts/gnss",
    "scripts/gnss/site",
    "config",
]

VERDICTS = ("PASS", "FAIL", "TIMEOUT", "SKIP")
Job = collections.namedtuple("Job", "name py args cwd env")


def find_jobs(root):
    """Every job, in run order. A Job with py None is a stale table entry."""

    def found(pattern):
        return sorted(
            os.path.relpath(p, root) for p in glob.glob(os.path.join(root, pattern))
        )

    gnss = os.path.join(root, "python", "scripts", "gnss")
    jobs = [
        Job(r, "ft", ["-m", "gnss_broker." + os.path.basename(r)[:-3]], gnss, {})
        for r in found("python/scripts/gnss/gnss_broker/test_*.py")
    ]
    jobs.append(
        Job(
            "python/scripts/gnss/gnss_broker/selftest.py",
            "ft",
            ["gnss_broker/selftest.py"],
            gnss,
            {},
        )
    )
    files = [r for d in TEST_DIRS for r in found(d + "/test_*.py")]
    files += found("tests/test_gps_*.py") + found("tests/test_beam_cube.py")
    files += found("python/scripts/gnss/*_code_check.py")
    files += [
        r
        for r in CHECKS
        if r not in KOTEKAN_ONLY or os.path.exists(os.path.join(root, r))
    ]
    for r in files:
        py = "ft" if r in FT else "sys" if r in SYS else "venv"
        jobs.append(Job(r, py, [r], root, {"PYTHON_GIL": "0"} if r in NO_GIL else {}))
    jobs += [
        Job(
            "python/scripts/js_viewer/check_js.sh",
            "sh",
            ["python/scripts/js_viewer/check_js.sh"],
            root,
            {},
        ),
        Job("scripts/gnss/ft_stress.py", "ft", ["scripts/gnss/ft_stress.py"], root, {}),
        Job(
            "scripts/gnss/ft_stress.py --selftest",
            "ft",
            ["scripts/gnss/ft_stress.py", "--selftest"],
            root,
            {},
        ),
        Job(
            "scripts/gnss/broker_multi.py --list",
            "ft",
            ["scripts/gnss/broker_multi.py", "config/gnss_chains_chord.yaml", "--list"],
            root,
            {},
        ),
    ]
    for r in sorted((FT | SYS) - KOTEKAN_ONLY):
        if not os.path.exists(os.path.join(root, r)):
            jobs.append(Job(r, None, [], root, {}))
    return jobs


RUNNING = set()
LOCK = threading.Lock()


def run(job, base_env, log, timeout):
    """(verdict, seconds, note) for one job, its output written to log."""
    if job.py is None:
        with open(log, "w") as f:
            f.write(
                "%s is named in run_tests.py's tables but is not there\n" % job.name
            )
        return "FAIL", 0.0, "stale table entry"
    env = dict(base_env, **job.env)
    if job.py == "ft":
        env.setdefault("GNSS_REQUIRE_FT", "1")
    argv = [PY[job.py]] + job.args
    t0 = time.time()
    with open(log, "w") as f:
        f.write("# cd %s && %s\n" % (job.cwd, " ".join(argv)))
        f.flush()
        p = subprocess.Popen(
            argv,
            cwd=job.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with LOCK:
            RUNNING.add(p)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = None
        finally:
            # The whole group: a test's servers and children must not outlive it.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            with LOCK:
                RUNNING.discard(p)
    dt = time.time() - t0
    if rc is None:
        return "TIMEOUT", dt, "killed after %d s" % timeout
    if rc == 77:
        return "SKIP", dt, "rc 77"
    if rc != 0:
        return "FAIL", dt, "rc %d" % rc
    with open(log, errors="replace") as f:
        m = re.search(r"^(?:OK|FAILED) \(.*\bskipped=(\d+)", f.read(), re.M)
    if m:
        return "SKIP", dt, "%s unittest case(s) skipped" % m.group(1)
    return "PASS", dt, ""


def kotekan_tree(root):
    path = os.path.join(root, "python", "scripts", "gnss", "kotekan_src.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("kotekan_src", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SRC


def git_head(root):
    try:
        out = subprocess.run(
            ["git", "-C", root, "describe", "--always", "--dirty"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).stdout.strip()
    except OSError:
        out = ""
    return out or "not a git checkout"


def row(job, verdict, dt, note):
    return "%-7s %6.1fs %-4s %s%s" % (
        verdict,
        dt,
        job.py or "-",
        job.name,
        "  (%s)" % note if note else "",
    )


def tail(log, n=3):
    with open(log, errors="replace") as f:
        lines = [l.rstrip() for l in f if l.strip()]
    return lines[-n:]


def main():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=here, help="the checkout (default: %(default)s)")
    ap.add_argument(
        "--out", help="directory for the logs (default: a new temporary one)"
    )
    ap.add_argument("-j", type=int, default=1, help="jobs at once (default: 1)")
    ap.add_argument("--only", help="run only the jobs whose name matches this regex")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per job")
    ap.add_argument("--strict", action="store_true", help="a SKIP fails the run")
    ap.add_argument("--list", action="store_true", help="list the jobs, run nothing")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    if not os.path.isdir(os.path.join(root, "python", "scripts", "gnss")):
        ap.error("%s holds no python/scripts/gnss" % root)
    jobs = find_jobs(root)
    if a.only:
        jobs = [j for j in jobs if re.search(a.only, j.name)]
    if not jobs:
        ap.error("no job matches")
    if a.list:
        for j in jobs:
            print("%-4s %-20s %s" % (j.py or "-", os.path.relpath(j.cwd, root), j.name))
        return 0
    missing = sorted(
        {PY[j.py] for j in jobs if j.py and not os.access(PY[j.py], os.X_OK)}
    )
    if missing:
        print("missing interpreter(s): %s" % ", ".join(missing), file=sys.stderr)
        return 2

    # Relative to here, not to each job's cwd.
    if os.environ.get("KOTEKAN_SRC"):
        os.environ["KOTEKAN_SRC"] = os.path.abspath(os.environ["KOTEKAN_SRC"])
    base_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for v in ("GNSS_REQUIRE_CXX", "GNSS_REQUIRE_NODE", "GNSS_REQUIRE_PLOT"):
        base_env.setdefault(v, "1")
    # As the broker runs, and so that N jobs at once do not each start a thread per core.
    for v in ("OPENBLAS", "OMP", "MKL", "NUMEXPR"):
        base_env.setdefault(v + "_NUM_THREADS", "1")

    out = a.out or tempfile.mkdtemp(prefix="gnss-tests-")
    os.makedirs(out, exist_ok=True)
    head = [
        "root     %s (%s)" % (root, git_head(root)),
        "kotekan  %s" % (kotekan_tree(root) or "none found"),
        "python   venv %s, venv-ft %s, sys %s" % (PY["venv"], PY["ft"], PY["sys"]),
        "env      %s"
        % " ".join(
            "%s=%s" % (k, base_env[k])
            for k in sorted(base_env)
            if k.startswith("GNSS_REQUIRE_") or k == "KOTEKAN_SRC"
        ),
        "logs     %s" % out,
    ]
    print("\n".join(head))
    sys.stdout.flush()
    os.nice(10)

    results = [None] * len(jobs)

    def one(i):
        log = os.path.join(out, re.sub(r"[^A-Za-z0-9_.-]+", "_", jobs[i].name) + ".log")
        results[i] = run(jobs[i], base_env, log, a.timeout) + (log,)
        with LOCK:
            done = sum(r is not None for r in results)
            v, dt, note, _ = results[i]
            print("[%3d/%d] %s" % (done, len(jobs), row(jobs[i], v, dt, note)))
            sys.stdout.flush()

    t0 = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.j))
    try:
        for f in [pool.submit(one, i) for i in range(len(jobs))]:
            f.result()
    except KeyboardInterrupt:
        with LOCK:
            for p in RUNNING:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
        print("interrupted", file=sys.stderr)
        os._exit(130)
    pool.shutdown()

    counts = collections.Counter(r[0] for r in results)
    lines = [row(j, v, dt, note) for j, (v, dt, note, _) in zip(jobs, results)]
    lines.append("")
    lines.append(
        "%d jobs in %.0f s: %s"
        % (
            len(jobs),
            time.time() - t0,
            ", ".join("%d %s" % (counts[v], v) for v in VERDICTS),
        )
    )
    for j, (v, dt, note, log) in zip(jobs, results):
        if v != "PASS":
            lines += ["", "%s %s (%s): %s" % (v, j.name, note, log)]
            lines += ["    " + l for l in tail(log)]
    print("")
    print("\n".join(lines))
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(head + [""] + lines) + "\n")
    bad = counts["FAIL"] + counts["TIMEOUT"] + (counts["SKIP"] if a.strict else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
