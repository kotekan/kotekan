#!/usr/bin/env python3
"""gen_fleet.py's own checks: the generator's per-node notes reach the operator, and --check
exits 1 only for a config that differs (2 when nothing could be compared).

    python3 scripts/gnss/test_gen_fleet.py

No generator runs here: main() is driven with subprocess.run replaced, against a throwaway
manifest, a one-line template as the base and a preview directory.
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_fleet  # noqa: E402

NOTE = (
    "  note: GPS_L2C_CM has no covering channel on cx44 (carrier 1227.60 MHz, mainlobe "
    "1.023 MHz) -- chain SKIPPED on this node\n"
)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def manifest(d, base):
    path = os.path.join(d, "m.yaml")
    with open(path, "w") as fh:
        fh.write("base: %s\nnodes:\n  cx44: {}\n" % base)
    return path


def test_relay_notes():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        gen_fleet._relay_notes("cx44", "  node          cx44\n" + NOTE + NOTE)
    check(
        err.getvalue() == "cx44: " + NOTE.strip() + "\n",
        "a SKIPPED note is relayed once with the node's name, the summary is not",
    )
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        gen_fleet._relay_notes("cx44", "  WARNING   the EOP table above already ends\n")
    check(
        err.getvalue() == "cx44: WARNING   the EOP table above already ends\n",
        "a WARNING is relayed",
    )
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        gen_fleet._relay_notes("cx44", "  node          cx44\n  rest port     12048\n")
    check(err.getvalue() == "", "a run with no SKIPPED or WARNING relays nothing")


def test_check_relays_notes():
    """--check of one node whose generator succeeds with a note on stderr."""
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.j2")
        with open(base, "w") as fh:
            fh.write("a: 1\n")
        with open(os.path.join(d, "chord_gnss_cx44.yaml"), "w") as fh:
            fh.write("a: 1\n")
        argv = ["gen_fleet.py", manifest(d, base), "--check", "--preview-dir", d]

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="a: 1\n", stderr=NOTE)

        real_run, real_argv = subprocess.run, sys.argv
        subprocess.run, sys.argv = fake_run, argv
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = gen_fleet.main()
        finally:
            subprocess.run, sys.argv = real_run, real_argv
    check(rc is None and "ok " in out.getvalue(), "the node checks ok")
    check(
        "cx44: note: GPS_L2C_CM" in err.getvalue(),
        "--check relays the generator's SKIPPED note",
    )


def test_check_names_a_role_off_config():
    """A node set role off keeps its slot; its config left in the output directory is named by
    --check (a write removes it)."""
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.j2")
        with open(base, "w") as fh:
            fh.write("a: 1\n")
        for n in ("cx44", "cx19"):
            with open(os.path.join(d, "chord_gnss_%s.yaml" % n), "w") as fh:
                fh.write("a: 1\n")
        man = os.path.join(d, "m.yaml")
        with open(man, "w") as fh:
            fh.write("base: %s\nnodes:\n  cx19: {role: 'off'}\n  cx44: {}\n" % base)
        argv = ["gen_fleet.py", man, "--check", "--preview-dir", d]

        def fake_run(cmd, **kw):
            check("cx19" not in cmd, "a role-off node is not generated")
            check(
                cmd[cmd.index("--search-port-base") + 1] == "11042",
                "cx44 keeps slot 1's port with cx19 off",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="a: 1\n", stderr="")

        real_run, real_argv = subprocess.run, sys.argv
        subprocess.run, sys.argv = fake_run, argv
        out, exc = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                io.StringIO()
            ):
                gen_fleet.main()
        except SystemExit as e:
            exc = str(e.code)
        finally:
            subprocess.run, sys.argv = real_run, real_argv
    check(
        exc is not None and "cx19 is role off but" in out.getvalue(),
        "--check names the role-off node's leftover config (%r)" % out.getvalue(),
    )


STAMP = "# gnss_iface: fnv1a64:0123456789abcdef pin c0ffee sizing telem=fnv1a64:0000000000000001"


def check_against(disk, regenerated):
    """--check of one node whose config on disk is `disk` and whose generator prints
    `regenerated`: (exit message or None, stdout)."""
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.j2")
        with open(base, "w") as fh:
            fh.write("a: 1\n")
        with open(os.path.join(d, "chord_gnss_cx44.yaml"), "w") as fh:
            fh.write(disk)
        argv = ["gen_fleet.py", manifest(d, base), "--check", "--preview-dir", d]

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout=regenerated, stderr="")

        real_run, real_argv = subprocess.run, sys.argv
        subprocess.run, sys.argv = fake_run, argv
        out, exc = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(out):
                gen_fleet.main()
        except SystemExit as e:
            exc = str(e.code)
        finally:
            subprocess.run, sys.argv = real_run, real_argv
    return exc, out.getvalue()


def test_check_leaves_the_stamp_out():
    """A config written before the interface stamp has none; the regenerated one has it. That
    must not read as a difference, or every node would read DIFFERS and eop_cron would stop
    regenerating them."""
    old = "# GENERATED\na: 1\n"
    new = "# GENERATED\n%s\na: 1\n" % STAMP
    exc, out = check_against(old, new)
    check(
        exc is None and "ok*" in out and "the gnss_iface stamp (none on disk)" in out,
        "an unstamped config on disk checks ok* (%s, %r)" % (exc, out.strip()),
    )
    exc, out = check_against(new, new)
    check(exc is None and "ok      " in out, "the stamped config checks ok (%r)" % out)
    moved = new.replace("0123456789abcdef", "fedcba9876543210")
    exc, out = check_against(moved, new)
    check(
        exc is None and "differs only in the gnss_iface stamp)" in out,
        "a stamp from another golden checks ok* (%r)" % out.strip(),
    )
    exc, out = check_against("# GENERATED\na: 1\nb: 2\n", new)
    check(exc is not None and "DIFFERS" in out, "a changed body still differs")
    note = "the gnss_iface stamp differs too"
    diff = out.split(note, 1)[-1]
    check(
        note in out and "-b: 2" in diff and "gnss_iface" not in diff,
        "its diff shows the body, not the stamp, which it notes once (%r)" % out,
    )
    exc, out = check_against(new.replace("a: 1", "a: 2"), new)
    check(
        "-a: 2" in out and "+a: 1" in out and "gnss_iface" not in out,
        "a stamped config that differs in its body notes no stamp (%r)" % out,
    )


def run_check(base):
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, "gen_fleet.py"), manifest(d, base)]
            + ["--check", "--preview-dir", d],
            capture_output=True,
            text=True,
        )
    return p.returncode, p.stderr


def test_missing_base_exits_2():
    rc, err = run_check("/nonexistent/base.j2")
    check(rc == 2, "--check with a missing base exits 2 (got %d)" % rc)
    check("does not exist" in err, "and says the base is missing")


def test_crash_exits_2():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "broken.j2")
        with open(base, "w") as fh:
            fh.write("{% not a tag %}\n")
        rc, err = run_check(base)
    check(rc == 2, "--check whose stock render raises exits 2 (got %d)" % rc)
    check("Traceback" in err, "and prints the traceback")
    # Any crash passes the two checks above, a missing jinja2 among them: the render must
    # have reached the template.
    check(
        "TemplateSyntaxError" in err,
        "and the traceback is the template's error (%s)"
        % (err.strip().splitlines() or ["no stderr"])[-1],
    )


def test_no_kotekan_tree():
    """In a checkout with no kotekan tree beside it and KOTEKAN_SRC unset, --print-path still
    answers, and --check, which renders the base from kotekan, exits 2 with the reason."""
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(os.path.realpath(d), "c")
        for rel in (
            "scripts/gnss/gen_fleet.py",
            "scripts/gnss/stock_parity.py",
            "python/scripts/gnss/kotekan_src.py",
            "config/gnss_fleet.py",
            "config/chord_gnss_node.yaml",
        ):
            os.makedirs(os.path.dirname(os.path.join(c, rel)), exist_ok=True)
            shutil.copy(os.path.join(HERE, "..", "..", rel), os.path.join(c, rel))
        man = manifest(c, "chord/pathfinder.j2")
        env = {k: v for k, v in os.environ.items() if k != "KOTEKAN_SRC"}
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        def gen_fleet(*args):
            p = subprocess.run(
                [
                    sys.executable,
                    os.path.join(c, "scripts", "gnss", "gen_fleet.py"),
                    man,
                ]
                + list(args),
                capture_output=True,
                text=True,
                env=env,
            )
            return p.returncode, p.stdout, p.stderr

        rc, out, err = gen_fleet("--print-path")
        want = os.path.join(c, "config", "generated", "chord_gnss_cx44.yaml")
        check(
            rc == 0 and out.splitlines() == [want],
            "--print-path needs no kotekan (rc %d, %r, %r)"
            % (rc, out.splitlines(), err.strip().splitlines()[-1:]),
        )
        rc, out, err = gen_fleet("--check")
        check(
            rc == 2 and "set KOTEKAN_SRC" in err and "Traceback" not in err,
            "--check with no kotekan tree exits 2 with the reason, no traceback (rc %d, %r)"
            % (rc, err.strip().splitlines()[-1:]),
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
