#!/usr/bin/env python3
"""broker_equiv.py's own checks: what a replay runs under, what selftest replays, what a
bless calls dirty, and that the canonical check fails a request sent to another chain's row.

    python3 scripts/gnss/test_broker_equiv.py

replay() is driven with subprocess.run captured, selftest with replay() replaced, and
_tree_state() against a throwaway git repository laid out like this one. The canonical check's
cases replay broker_fake_rows.jsonl, changed copies of it, through the broker.
"""
import contextlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


be = _load(os.path.join(HERE, "broker_equiv.py"), "broker_equiv_under_test")


def _gather_of(cmd):
    """The --telem-gather the broker would parse from `cmd` (argparse keeps the last)."""
    val = None
    for i, a in enumerate(cmd):
        if a == "--telem-gather" and i + 1 < len(cmd):
            val = cmd[i + 1]
        elif a.startswith("--telem-gather="):
            val = a.split("=", 1)[1]
    return val


def _refused(endpoint):
    host, _, port = endpoint.rpartition(":")
    try:
        socket.create_connection((host or "127.0.0.1", int(port)), timeout=1.0).close()
    except OSError:
        return True
    return False


def _write_transcript(tmp, header):
    """A transcript holding only its argv header; returns its path."""
    path = os.path.join(tmp, "t.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"k": "argv", "v": header}) + "\n")
    return path


def _replay_run(path, extra=()):
    """(command, environment) replay() runs the broker under for the transcript at `path`."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="digest\n", stderr="")

    saved = be.subprocess.run
    be.subprocess.run = fake_run
    try:
        be.replay(path, extra=extra)
    finally:
        be.subprocess.run = saved
    return seen["cmd"], seen["env"]


def _replay_cmd(tmp, header, extra=()):
    """The broker command replay() runs for a transcript whose argv header is `header`."""
    return _replay_run(_write_transcript(tmp, header), extra)[0]


def test_replay_never_reaches_a_live_gather():
    """A transcript does not hold the telemetry stream, so a replay must not connect to the
    gather its recording named. A listener stands in for the live one."""
    lis = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lis.bind(("127.0.0.1", 0))
    lis.listen(4)
    live = "127.0.0.1:%d" % lis.getsockname()[1]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for label, header, extra in (
                ("--telem-gather <ep>", ["--telem-gather", live], ()),
                ("--telem-gather=<ep>", ["--telem-gather=" + live], ()),
                ("a variant's --telem-gather", [], ["--telem-gather", live]),
            ):
                cmd = _replay_cmd(tmp, ["--signal", "gps_l5"] + header, extra)
                g = _gather_of(cmd)
                check(
                    g is not None and _refused(g),
                    "%s: the replay keeps a gather and every connect to it is refused "
                    "(replays under %s; a gather listens at %s)" % (label, g, live),
                )
                check(
                    live not in " ".join(cmd),
                    "%s: the recorded endpoint is not in the replay's argv" % label,
                )
            cmd = _replay_cmd(tmp, ["--signal", "gps_l5"])
            check(
                _gather_of(cmd) is None,
                "a recording without --telem-gather replays without one",
            )
    finally:
        lis.close()


def test_replay_pins_the_igs_registry_snapshot():
    """A capability gate seeds only the satellites the IGS registry calls capable, so a
    transcript's `<transcript>.igs.snx` snapshot must reach the broker as GNSS_IGS_SNX."""
    saved = os.environ.pop("GNSS_IGS_SNX", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_transcript(tmp, ["--signal", "bds_b2a"])
            env = _replay_run(path)[1]
            check(
                "GNSS_IGS_SNX" not in env,
                "a transcript without a registry snapshot leaves GNSS_IGS_SNX unset",
            )
            with open(path + ".igs.snx", "w") as f:
                f.write("+SATELLITE/IDENTIFIER\n-SATELLITE/IDENTIFIER\n")
            env = _replay_run(path)[1]
            check(
                env.get("GNSS_IGS_SNX") == path + ".igs.snx",
                "a transcript with <transcript>.igs.snx replays under GNSS_IGS_SNX=<it> "
                "(got %s)" % env.get("GNSS_IGS_SNX"),
            )
    finally:
        if saved is not None:
            os.environ["GNSS_IGS_SNX"] = saved


def test_replay_pins_the_boresight_a_recording_ran_under():
    """A recording that neither set --boresight nor read the pointing from a node as its first
    GET ran under the default az 180 el 81.41 and replays under it; one that did, or set the
    flag, replays as recorded, and a golden's flags still come last."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.jsonl")

        def cmd_of(header, records, extra=()):
            with open(path, "w") as f:
                for r in [{"k": "argv", "v": header}] + records:
                    f.write(json.dumps(r) + "\n")
            cmd = _replay_run(path, extra)[0]
            return [cmd[i + 1] for i, x in enumerate(cmd) if x == "--boresight"]

        base = ["--signal", "gps_l5"]
        now = {"k": "now", "v": 1.0}
        cfg = {"k": "get", "u": "http://n:12048/config", "r": {}}
        det = {"k": "get", "u": "http://n:12050/gps_search/get_detections", "r": []}
        check(
            cmd_of(base, [now, det]) == [be.NOMINAL_BORESIGHT],
            "no --boresight and no /config read: the nominal boresight",
        )
        check(
            cmd_of(base, [now, cfg, now, det]) == [],
            "a /config read first: replayed as recorded",
        )
        check(
            cmd_of(base, [now, det, cfg]) == [be.NOMINAL_BORESIGHT],
            "a /config read after another GET is not the start-up read",
        )
        check(
            cmd_of(base + ["--boresight", "none"], [now]) == ["none"],
            "--boresight in the recording: replayed as recorded",
        )
        check(
            cmd_of(base, [now], ["--boresight", "1,2"])
            == [be.NOMINAL_BORESIGHT, "1,2"],
            "a golden's --boresight comes after the pin, so it wins",
        )


ROWS = os.path.join(HERE, "fixtures", "broker_fake_rows.jsonl")


def _swap_chain(tmp, edit):
    """A copy of the rows fixture with edit(records) applied; its path."""
    with open(ROWS) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    edit(recs)
    path = os.path.join(tmp, os.path.basename(ROWS))
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    return path


def test_a_request_to_another_chains_row_fails():
    """The rows fixture replays to its golden with no divergence, and does not when one
    recorded request names another chain's row, or when the broker sends one verb to another
    chain's row (a tracker of its argv swapped)."""
    want = be._read_gold(be._gold_path(ROWS))[0]
    got, err = be.replay(ROWS)
    check(
        got == want and be._replay_tail(err)[1] == 0,
        "the fixture replays to its golden with no divergence (%s)" % got[:16],
    )

    def recorded(recs):
        r = next(
            r for r in recs if r["k"] == "post" and "gnss1_gps_l5_inject" in r["u"]
        )
        r["u"] = r["u"].replace("gps_l5", "gal_e5a")
        r["c"] = r["c"].replace("gps_l5", "gal_e5a")

    def sent(recs):
        v = recs[0]["v"]
        i = v.index("--trackers") + 1
        v[i] = v[i].replace(
            "gnss{0..1}_gps_l5_inject",
            "gnss0_gps_l5_inject,%s/gnss1_gal_e5a_inject" % v[i].split("/gnss", 1)[0],
        )

    with tempfile.TemporaryDirectory() as tmp:
        for what, edit in (
            ("one recorded set_seeds names gal_e5a's row", recorded),
            ("the broker sends set_seeds to gal_e5a's row", sent),
        ):
            got, err = be.replay(_swap_chain(tmp, edit))
            check(
                got != want and be._replay_tail(err)[1] > 0,
                "%s: the digest moves (%s) and the replay diverges (%s)"
                % (what, got[:16], be._replay_tail(err)[1]),
            )


def test_each_form_replays_as_the_other():
    """The fake rows recorded at the rows' own paths, replayed asking every request by chain
    (--node-hosts), and the ones recorded by chain, replayed at the rows' own paths, keep their
    goldens: each variant golden is its transcript's plain one, and check verifies both."""
    for name, variant, hosts in (
        ("broker_fake_rows.jsonl", "hosted", True),
        ("broker_fake_rows_hosted.jsonl", "legacy", False),
    ):
        path = os.path.join(HERE, "fixtures", name)
        plain = be._read_gold(be._gold_path(path))[0]
        want, prov = be._read_gold(be._gold_path(path, variant))
        argv = be._gold_argv(prov)
        check(
            want == plain
            and argv[0] == "--node-hosts"
            and bool(argv[1]) == hosts
            and bool(be._last_value(be._header_argv(path), "--node-hosts")) != hosts,
            "%s: the %s variant (%s) is the plain golden"
            % (name, variant, " ".join(argv)[:40]),
        )


def test_selftest_replays_the_flags_after_the_separator():
    """selftest under flags after -- must replay under them every time, as digest does."""
    calls = []

    def fake_replay(path, env=None, extra=()):
        calls.append(list(extra))
        return ("perturbed" if env else "plain"), ""

    saved = (be.replay, be._census, sys.argv)
    be.replay = fake_replay
    be._census = lambda path: {"now": 3, "get": 3, "post": 3}
    sys.argv = ["broker_equiv.py", "selftest", "t.jsonl", "--", "--dll-gain", "0.3"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            be.main()
    except SystemExit as e:
        check(False, "selftest exited: %s" % e)
    finally:
        be.replay, be._census, sys.argv = saved
    check(
        len(calls) == 7 and all(c == ["--dll-gain", "0.3"] for c in calls),
        "all 7 selftest replays carry --dll-gain 0.3 (got %s)" % calls,
    )
    det = [ln for ln in out.getvalue().splitlines() if ln.startswith("determinism OK")]
    check(
        len(det) == 1 and "--dll-gain 0.3" in det[0],
        "the determinism line names the flags (%s)" % det,
    )


def test_tree_state_covers_the_broker_imports():
    """An uncommitted edit to a module the broker imports from python/scripts/gnss/ makes a
    bless DIRTY, as an edit to the broker or gnss_broker/ does."""
    with tempfile.TemporaryDirectory() as k:

        def git(*a):
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
                + ["-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"]
                + list(a),
                cwd=k,
                check=True,
                capture_output=True,
            )

        with open(be.__file__) as f:
            harness = f.read()
        files = {
            "python/scripts/gnss/gps_distributed_broker.py": "",
            "python/scripts/gnss/gnss_brdc_supply.py": "",
            "python/scripts/gnss/gnss_broker/__init__.py": "",
            "scripts/gnss/broker_equiv.py": harness,
            "docs/notes.md": "",
        }
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(k, rel)), exist_ok=True)
            with open(os.path.join(k, rel), "w") as f:
                f.write(text)
        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "base")
        m = _load(os.path.join(k, "scripts", "gnss", "broker_equiv.py"), "be_tree")
        check(m._tree_state()[1] is False, "a clean tree is not dirty")
        for rel, dirty in (
            ("python/scripts/gnss/gnss_brdc_supply.py", True),
            ("python/scripts/gnss/gps_distributed_broker.py", True),
            ("python/scripts/gnss/gnss_broker/__init__.py", True),
            ("scripts/gnss/broker_equiv.py", True),
            ("docs/notes.md", False),
        ):
            p = os.path.join(k, rel)
            with open(p, "a") as f:
                f.write("# edit\n")
            check(
                m._tree_state()[1] is dirty,
                "an edit to %s is %s" % (rel, "dirty" if dirty else "not dirty"),
            )
            with open(p, "w") as f:
                f.write(files[rel])


def main():
    test_replay_never_reaches_a_live_gather()
    test_replay_pins_the_igs_registry_snapshot()
    test_replay_pins_the_boresight_a_recording_ran_under()
    test_a_request_to_another_chains_row_fails()
    test_each_form_replays_as_the_other()
    test_selftest_replays_the_flags_after_the_separator()
    test_tree_state_covers_the_broker_imports()
    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print(
        "GATE GOOD: replay argv and registry pin, selftest flags, bless provenance, "
        "canonical check"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
