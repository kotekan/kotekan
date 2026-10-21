#!/usr/bin/env python3
"""The site shell scripts' own checks, each run against stubs in a throwaway directory.

    python3 scripts/gnss/site/test_site_scripts.py

Nothing here reaches a node, a unit or a process outside the test: ssh, sudo, systemctl, curl,
scp and the like are stubs on PATH, and pgrep/pkill are wrapped to match only processes in the
test's own session (pgrep -s 0), so a pattern can never touch the live stack on this host.
"""
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def stub(d, name, body):
    """An executable shell script `name` in directory d."""
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as fh:
        fh.write("#!/bin/bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def session_pgrep(bindir):
    """pgrep and pkill restricted to the caller's session: the test runs every script in a
    session of its own, so a pattern matches only the test's processes."""
    for tool in ("pgrep", "pkill"):
        stub(bindir, tool, 'exec /usr/bin/%s -s 0 "$@"\n' % tool)


def run(cmd, env, timeout=120, cwd=None, bare=False):
    """Run cmd in a new session with env added to this environment, or with bare=True to
    cron's (PATH and HOME alone); (rc, stdout+stderr). On a timeout the session's process
    group is killed, so nothing it started outlives it."""
    if bare:
        e = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/")}
    else:
        e = dict(os.environ)
    e.update(env)
    p = subprocess.Popen(
        cmd,
        env=e,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        out += "\n(timed out after %d s)" % timeout
    return p.returncode, out


def test_contract_gate_env_is_whole_record():
    d = tempfile.mkdtemp(prefix="site_gate_")
    try:
        shutil.copy(os.path.join(HERE, "stack_contract_gate.sh"), d)
        stub(
            d,
            "run_component.sh",
            r"printf 'argv\tsleep\nargv\t30\nenv\tOPENBLAS_NUM_THREADS=1\n'" + "\n",
        )
        bindir = os.path.join(d, "bin")
        stub(bindir, "systemctl", 'echo "$GATE_TEST_PID"\n')
        path = bindir + ":" + os.environ["PATH"]
        for have, want_ok in (("16", False), ("1", True)):
            env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
            env["OPENBLAS_NUM_THREADS"] = have
            p = subprocess.Popen(["sleep", "30"], env=env)
            try:
                rc, out = run(
                    ["bash", os.path.join(d, "stack_contract_gate.sh"), "broker"],
                    {"PATH": path, "GATE_TEST_PID": str(p.pid)},
                )
            finally:
                p.kill()
                p.wait()
            if want_ok:
                check(
                    rc == 0 and " ok " in out,
                    "OPENBLAS_NUM_THREADS=1 satisfies the contract (rc %d)" % rc,
                )
            else:
                check(
                    rc != 0 and "ENV MISSING: OPENBLAS_NUM_THREADS=1" in out,
                    "OPENBLAS_NUM_THREADS=16 does not satisfy OPENBLAS_NUM_THREADS=1 (rc %d: %s)"
                    % (rc, out.strip().splitlines()[0] if out.strip() else ""),
                )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_contract_gate_fails_when_nothing_checked():
    """With no component named and none running the gate has checked nothing, which is not a
    pass. A named component that is not running is still skipped."""
    d = tempfile.mkdtemp(prefix="site_gate_")
    try:
        k, _ = site_tree(d, "stack_contract_gate.sh", "stack_components.sh")
        gate = os.path.join(k, "scripts", "gnss", "site", "stack_contract_gate.sh")
        bindir = os.path.join(d, "bin")
        stub(bindir, "systemctl", "echo 0\n")
        env = {"PATH": bindir + ":" + os.environ["PATH"]}
        rc, out = run(["bash", gate], env)
        check(
            rc != 0 and "checked 0 component(s)" in out,
            "nothing running and nothing named fails (rc %d)" % rc,
        )
        rc, out = run(["bash", gate, "broker"], env)
        check(
            rc == 0 and "not running -- skipped" in out,
            "a named component that is not running is skipped (rc %d)" % rc,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def gate_tree(d):
    """A checkout holding the real definition, launcher and gate, whose broker and rail are
    stubs that sleep, and a systemctl that gives the pid in GATE_<COMPONENT>_PID as a unit's
    MainPID; (gate, launcher, stack_components.sh, the kotekan tree)."""
    k, kt = site_tree(
        d, "stack_contract_gate.sh", "stack_components.sh", "run_component.sh"
    )
    site = os.path.join(k, "scripts", "gnss", "site")
    for path in (
        os.path.join(k, "scripts", "gnss", "broker_multi.py"),
        os.path.join(site, "rail_watch.py"),
    ):
        with open(path, "w") as fh:
            fh.write("import time\ntime.sleep(60)\n")
    stub(
        os.path.join(d, "bin"),
        "systemctl",
        'case "$3" in\n'
        '  gnss-broker.service) echo "${GATE_BROKER_PID:-0}" ;;\n'
        '  gnss-rail.service) echo "${GATE_RAIL_PID:-0}" ;;\n'
        "  *) echo 0 ;;\n"
        "esac\n",
    )
    return (
        os.path.join(site, "stack_contract_gate.sh"),
        os.path.join(site, "run_component.sh"),
        os.path.join(site, "stack_components.sh"),
        kt,
    )


def start_component(launcher, name):
    """Start a component through run_component.sh in cron's environment, as its unit does, in
    a session of its own; the Popen, once the stub has replaced the launcher."""
    p = subprocess.Popen(
        ["bash", launcher, name],
        env={"PATH": "/usr/bin:/bin", "HOME": "/", "GNSS_PY": "/usr/bin/python3"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    script = {"broker": b"broker_multi.py", "rail": b"rail_watch.py"}[name]
    for _ in range(100):
        try:
            if script in open("/proc/%d/cmdline" % p.pid, "rb").read():
                break
        except OSError:
            break
        time.sleep(0.05)
    return p


def stop_component(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass
    p.wait()


def test_contract_gate_sees_a_dropped_export():
    """A definition that sets the broker's BLAS caps without exporting them starts a broker
    without them, and the gate, reading the same definition, says so (#134)."""
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_gate_"))
    try:
        gate, launcher, comps, _ = gate_tree(d)
        line = "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1"
        text = open(comps).read()
        check(text.count(line) == 1, "the broker's BLAS line is in the definition")
        env = {
            "PATH": os.path.join(d, "bin") + ":/usr/bin:/bin",
            "GNSS_PY": "/usr/bin/python3",
        }
        for dropped in (False, True):
            if dropped:
                with open(comps, "w") as fh:
                    fh.write(text.replace(line, line[len("export ") :]))
            p = start_component(launcher, "broker")
            try:
                env["GATE_BROKER_PID"] = str(p.pid)
                rc, out = run(["bash", gate, "broker"], env, bare=True)
            finally:
                stop_component(p)
            if dropped:
                check(
                    rc == 1
                    and "broker" in out
                    and "ENV MISSING: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1" in out,
                    "with the export dropped the gate reports the caps missing (rc %d: %r)"
                    % (rc, out.strip().splitlines()[:1]),
                )
            else:
                check(
                    rc == 0 and "broker" in out and " ok " in out,
                    "with the export in place the same broker passes (rc %d: %r)"
                    % (rc, out.strip().splitlines()[:1]),
                )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_contract_gate_ignores_the_callers_kotekan_src():
    """The gate expects what a unit gets, which never includes the operator's KOTEKAN_SRC: with
    one exported, naming the tree the broker uses or another, a matching stack still passes."""
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_gate_"))
    try:
        gate, launcher, _, kt = gate_tree(d)
        procs = {name: start_component(launcher, name) for name in ("broker", "rail")}
        try:
            env = {
                "PATH": os.path.join(d, "bin") + ":/usr/bin:/bin",
                "GNSS_PY": "/usr/bin/python3",
                "GATE_BROKER_PID": str(procs["broker"].pid),
                "GATE_RAIL_PID": str(procs["rail"].pid),
            }
            for src in (None, kt, os.path.join(d, "elsewhere")):
                e = dict(env) if src is None else dict(env, KOTEKAN_SRC=src)
                rc, out = run(["bash", gate], e, bare=True)
                check(
                    rc == 0 and "checked 2 component(s), 0 mismatch(es)" in out,
                    "KOTEKAN_SRC %s in the caller: broker and rail pass (rc %d: %r)"
                    % (
                        "unset" if src is None else "=" + src,
                        rc,
                        [x for x in out.splitlines() if "MISS" in x][:2],
                    ),
                )
        finally:
            for p in procs.values():
                stop_component(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)


SSH_STUB = r"""# ssh [opts] host 'remote command' -> the remote command, run here
while [ $# -gt 1 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
shift
exec bash -c "$*"
"""


def node_up_stubs(d):
    """ssh runs the remote string here; sudo runs only what node_up needs, and systemd-run
    fails; systemctl and curl fail; mkdir and sleep do nothing."""
    bindir = os.path.join(d, "bin")
    stub(bindir, "ssh", SSH_STUB)
    stub(
        bindir,
        "sudo",
        'case "$1" in\n'
        '  systemd-run) echo "$*" >> "$SUDO_LOG"; exit 1 ;;\n'
        "  tee) cat > /dev/null ;;\n"
        '  systemctl|mv) exec "$@" ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n",
    )
    stub(bindir, "systemctl", "exit 1\n")
    stub(bindir, "curl", "exit 7\n")
    stub(bindir, "mkdir", "exit 0\n")
    stub(bindir, "sleep", "exit 0\n")
    return bindir + ":" + os.environ["PATH"]


def test_node_up_systemd_run_failure_retries():
    d = tempfile.mkdtemp(prefix="site_node_up_")
    try:
        path = node_up_stubs(d)
        cfg = os.path.join(d, "node.yaml")
        open(cfg, "w").write("x: 1\n")
        sudo_log = os.path.join(d, "sudo.log")
        env = {
            "PATH": path,
            "GNSS_SKIP_PREFLIGHT": "1",
            "GNSS_CFG": cfg,
            "GNSS_BIN": "/bin/true",
            "GNSS_LOG": os.path.join(d, "node.log"),
            "SUDO_LOG": sudo_log,
        }
        rc, out = run(["bash", os.path.join(HERE, "node_up.sh"), "cxtest", "up"], env)
        starts = (
            open(sudo_log).read().count("systemd-run")
            if os.path.exists(sudo_log)
            else 0
        )
        check(
            "config unreadable" not in out,
            "a systemd-run failure on a readable config is not reported as 'config unreadable'",
        )
        check(
            starts == 3, "it is retried: three systemd-run attempts (saw %d)" % starts
        )
        check(
            rc == 1 and "FAILED after 3 attempts" in out,
            "and ends as a failed start after three (rc %d)" % rc,
        )
        # The node's log path and the config are one file here, so the remote side moves the
        # config away (to $LOG.1) before its readability test: that test must stay fatal.
        open(cfg, "w").write("x: 1\n")
        open(sudo_log, "w").close()
        env["GNSS_LOG"] = cfg
        rc, out = run(["bash", os.path.join(HERE, "node_up.sh"), "cxtest", "up"], env)
        check(
            rc == 2 and "config unreadable" in out and open(sudo_log).read() == "",
            "an unreadable config on the node still exits 2 before any systemd-run (rc %d)"
            % rc,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_stack_down_summary_is_one_count():
    d = tempfile.mkdtemp(prefix="site_stack_down_")
    try:
        bindir = os.path.join(d, "bin")
        session_pgrep(bindir)
        rc, out = run(
            ["bash", os.path.join(HERE, "stack_down.sh"), "--no-archive"],
            {"PATH": bindir + ":" + os.environ["PATH"]},
        )
        tail = out.rstrip("\n").split("\n")[-2:]
        check(
            rc == 0 and tail[-1] == "remaining cube-leg processes: 0" and tail[0] == "",
            "with nothing running the summary is one line ending in a single 0 (got %r)"
            % tail,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_wedge_probe_finds_kotekan_by_name():
    d = tempfile.mkdtemp(prefix="site_wedge_")
    try:
        bindir = os.path.join(d, "bin")
        session_pgrep(bindir)
        stub(bindir, "ssh", SSH_STUB)
        stub(bindir, "curl", "exit 7\n")
        stub(bindir, "nvidia-smi", "exit 1\n")
        env = {"PATH": bindir + ":" + os.environ["PATH"]}
        probe = os.path.join(HERE, "node_wedge_probe.sh")
        rc, out = run(["bash", probe, "cxtest"], env)
        check(
            "no kotekan process" in out,
            "with no kotekan running the probe says so, rather than probing its own shell",
        )
        # Binaries staged outside build/ (GNSS_BIN=builds/<name>/kotekan/kotekan) and beside
        # the default one (kotekan.next_<tag>), each run in the probe's session so the wrapped
        # pgrep can see it. The driver waits for the fake's exec (its process name) before the
        # probe looks.
        for rel in ("builds/x/kotekan/kotekan", "build/kotekan/kotekan.next_test"):
            fake = os.path.join(d, rel)
            os.makedirs(os.path.dirname(fake))
            shutil.copy("/bin/sleep", fake)
            pidf = os.path.join(d, "fake.pid")
            rc, out = run(
                [
                    "bash",
                    "-c",
                    '"$1" 30 & echo $! > "$2"; for _ in $(seq 50); do '
                    'grep -q "^kotekan" /proc/$!/comm 2>/dev/null && break; sleep 0.1; done; '
                    'bash "$3" cxtest; kill $!',
                    "driver",
                    fake,
                    pidf,
                    probe,
                ],
                env,
            )
            pid = open(pidf).read().strip()
            check(
                ("kotekan pid %s\n" % pid) in out and "no kotekan process" not in out,
                "a kotekan staged as %s is the process probed (pid %s; got %r)"
                % (rel, pid, out.strip().splitlines()[1:2]),
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cubecompact_up_waits_for_the_old_loop():
    d = tempfile.mkdtemp(prefix="site_cubecompact_")
    try:
        bindir = os.path.join(d, "bin")
        session_pgrep(bindir)
        # setsid keeps the new loop in the test's session, and starts nothing outside the test
        # tree (a launcher that ignored GNSS_K would start the installed loop).
        stub(
            bindir,
            "setsid",
            'case "$1" in "$CC_TEST_DIR"/*) exec "$@" ;; esac\necho "setsid stub: refusing $1"\nexit 1\n',
        )
        k = os.path.join(d, "k")
        # The new loop: records when it started, then idles.
        stub(
            os.path.join(k, "scripts", "gnss", "site"),
            "cubecompact_loop.sh",
            'date +%s.%N > "$CC_TEST_DIR/new_start"\necho $$ > "$CC_TEST_DIR/new_pid"\n'
            "while :; do sleep 1; done\n",
        )
        # The old loop: on TERM it finishes a pass (lingers 6 s) before it exits, as the real
        # loop does, with no compactor running meanwhile. It says when its trap is set.
        old = stub(
            os.path.join(d, "old"),
            "cubecompact_loop.sh",
            "trap 'sleep 6; date +%s.%N > \"$CC_TEST_DIR/old_exit\"; exit 0' TERM\n"
            'touch "$CC_TEST_DIR/old_ready"\n'
            "while :; do sleep 0.2; done\n",
        )
        py = stub(bindir, "fakepy", "exit 0\n")
        os.makedirs(os.path.join(d, "root", "raw"))
        env = {
            "PATH": bindir + ":" + os.environ["PATH"],
            "GNSS_K": k,
            "GNSS_PY": py,
            "CUBE_ROOT": os.path.join(d, "root"),
            "CUBE_HOST_NAME": os.uname().nodename.split(".")[0],
            "CUBE_COMPACT_LOG": os.path.join(d, "compact.log"),
            "CC_TEST_DIR": d,
            "CC_OLD": old,
            "CC_UP": os.path.join(HERE, "cubecompact_up.sh"),
        }
        # Paths travel in the environment: a driver whose command line named the old loop
        # would itself match the launcher's pkill pattern.
        rc, out = run(
            [
                "bash",
                "-c",
                '"$CC_OLD" & echo $! > "$CC_TEST_DIR/old_pid"; for _ in $(seq 100); do '
                '[ -e "$CC_TEST_DIR/old_ready" ] && break; sleep 0.1; done; exec bash "$CC_UP"',
            ],
            env,
        )
        t_old = os.path.join(d, "old_exit")
        t_new = os.path.join(d, "new_start")
        if not (os.path.exists(t_old) and os.path.exists(t_new)):
            check(False, "both loops ran (rc %d): %s" % (rc, out.strip()[-300:]))
            return
        old_exit = float(open(t_old).read())
        new_start = float(open(t_new).read())
        check(
            new_start > old_exit,
            "the new loop starts after the old one has exited (new - old = %+.1f s)"
            % (new_start - old_exit),
        )
        check(rc == 0, "and the launcher reports it up (rc %d)" % rc)
    finally:
        # Only a process still running a script under d: a pid that has exited may be reused.
        for name in ("old_pid", "new_pid"):
            try:
                pid = int(open(os.path.join(d, name)).read())
                if d in open("/proc/%d/cmdline" % pid).read():
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ValueError):
                pass
        shutil.rmtree(d, ignore_errors=True)


GEN_FLEET_STUB = r"""import os, sys
K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(K, "gen_fleet.env"), "a") as fh:
    fh.write(os.environ.get("KOTEKAN_SRC", "(unset)") + "\n")
if "--print-path" in sys.argv:
    print(os.path.join(K, "config", "generated", "chord_gnss_cx19_other.yaml"))
    sys.exit(0)
with open(os.path.join(K, "gen_fleet.calls"), "a") as fh:
    fh.write(" ".join(sys.argv[2:]) + "\n")
"""


def site_tree(d, *scripts):
    """A GNSS checkout d/k holding the named site scripts and kotekan_src.sh, with a kotekan
    tree beside it at d/kotekan; (checkout, kotekan tree)."""
    k = os.path.join(d, "k")
    site = os.path.join(k, "scripts", "gnss", "site")
    os.makedirs(site)
    for name in scripts + ("kotekan_src.sh",):
        shutil.copy(os.path.join(HERE, name), site)
    kt = os.path.join(d, "kotekan")
    os.makedirs(os.path.join(kt, "lib", "stages", "gnss"))
    return k, kt


def test_kotekan_src_helper_follows_the_python_rule():
    """kotekan_src.sh and python/scripts/gnss/kotekan_src.py, copied into the same checkout,
    find the same tree in every layout the rule names, in cron's environment."""
    py = os.path.join(
        HERE, "..", "..", "..", "python", "scripts", "gnss", "kotekan_src.py"
    )
    mark = os.path.join("lib", "stages", "gnss")
    # (what, trees holding lib/stages/gnss, relative to d, with c the checkout; KOTEKAN_SRC;
    #  the tree expected, relative to d unless absolute, or None for none)
    layouts = (
        ("one tree", ("c",), None, "c"),
        ("a submodule", ("c/kotekan",), None, "c/kotekan"),
        ("a checkout beside", ("kotekan",), None, "kotekan"),
        ("the checkout before a submodule", ("c", "c/kotekan", "kotekan"), None, "c"),
        ("a submodule before one beside", ("c/kotekan", "kotekan"), None, "c/kotekan"),
        ("none", (), None, None),
        ("KOTEKAN_SRC before the checkout", ("c",), "/no/such", "/no/such"),
        ("a relative KOTEKAN_SRC is made absolute", ("c",), "x/../kt", "kt"),
        ("an empty KOTEKAN_SRC is unset", ("kotekan",), "", "kotekan"),
    )
    for what, trees, env_src, want in layouts:
        d = tempfile.mkdtemp(prefix="site_ksrc_")
        try:
            c = os.path.join(d, "c")
            site = os.path.join(c, "scripts", "gnss", "site")
            pydir = os.path.join(c, "python", "scripts", "gnss")
            os.makedirs(site)
            os.makedirs(pydir)
            shutil.copy(os.path.join(HERE, "kotekan_src.sh"), site)
            shutil.copy(py, pydir)
            for t in trees:
                os.makedirs(os.path.join(d, t, mark))
            env = {} if env_src is None else {"KOTEKAN_SRC": env_src}
            # Sourced by a relative path from d, as `scripts/gnss/site/x.sh` is from a checkout.
            rc, sh_out = run(
                [
                    "bash",
                    "-c",
                    'set -u; . "$1"; echo "$GNSS_CHECKOUT"; gnss_kotekan_src "$GNSS_CHECKOUT"',
                    "sh",
                    os.path.join("c", "scripts", "gnss", "site", "kotekan_src.sh"),
                ],
                env,
                cwd=d,
                bare=True,
            )
            lines = sh_out.splitlines()
            _, py_out = run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); import kotekan_src; "
                    "print(kotekan_src.SRC)",
                    pydir,
                ],
                env,
                cwd=d,
                bare=True,
            )
            want_abs = None if want is None else os.path.join(d, want)
            check(
                lines[:1] == [c],
                "%s: GNSS_CHECKOUT is the checkout the helper is in (%r)"
                % (what, lines[:1]),
            )
            if want_abs is None:
                check(
                    rc == 1
                    and len(lines) == 2
                    and "set KOTEKAN_SRC" in lines[1]
                    and py_out.strip() == "None",
                    "%s: no tree, rc 1 and a message naming KOTEKAN_SRC; Python finds none "
                    "too (rc %d, %r, %r)" % (what, rc, lines[1:], py_out.strip()),
                )
            else:
                check(
                    rc == 0 and lines[1:] == [want_abs] and py_out.strip() == want_abs,
                    "%s: both find %s (shell rc %d %r, Python %r)"
                    % (what, want, rc, lines[1:], py_out.strip()),
                )
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_run_component_exports_kotekan_src_to_the_broker_alone():
    d = tempfile.mkdtemp(prefix="site_print_")
    try:
        k, kt = site_tree(d, "run_component.sh", "stack_components.sh")
        # The viewer's cwd: --print changes into it.
        os.makedirs(os.path.join(k, "python", "scripts", "js_viewer"))
        rc_sh = os.path.join(k, "scripts", "gnss", "site", "run_component.sh")
        out = {}
        names = "broker gather aggregator viewer obs rail cubearch cubecompact"
        for name in names.split():
            rc, out[name] = run(
                ["bash", rc_sh, "--print", name],
                {"GNSS_CHAIN": "gps_l5"} if name == "obs" else {},
                bare=True,
            )
            check(
                rc == 0,
                "--print %s exits 0 (rc %d%s)"
                % (name, rc, "" if rc == 0 else ": " + out[name][-200:]),
            )
        check(
            "argv\t%s/scripts/gnss/broker_multi.py\n" % k in out["broker"],
            "the broker runs from the checkout the scripts are in",
        )
        check(
            "env\tKOTEKAN_SRC=%s\n" % kt in out["broker"],
            "the broker's environment carries KOTEKAN_SRC, the kotekan tree beside it",
        )
        check(
            "argv\t%s/build_nodpdk/kotekan/kotekan\n" % kt in out["gather"],
            "the gather runs that kotekan tree's binary",
        )
        others = [n for n in out if n != "broker" and "env\tKOTEKAN_SRC" in out[n]]
        check(not others, "no other component exports KOTEKAN_SRC (%s)" % others)
        check(
            "env\tGNSS_CHAIN=gps_l5\n" in out["obs"],
            "an obs writer's GNSS_CHAIN, set by its unit, is still listed",
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_node_up_exports_kotekan_src_to_gen_fleet():
    d = tempfile.mkdtemp(prefix="site_node_up_src_")
    try:
        path = node_up_stubs(d)
        k, kt = site_tree(d, "node_up.sh")
        os.makedirs(os.path.join(k, "config"))
        with open(os.path.join(k, "scripts", "gnss", "gen_fleet.py"), "w") as fh:
            fh.write(GEN_FLEET_STUB)
        open(os.path.join(k, "config", "gnss_fleet_chord.yaml"), "w").write("x: 1\n")
        rc, out = run(
            [
                "bash",
                os.path.join(k, "scripts", "gnss", "site", "node_up.sh"),
                "cxtest",
                "status",
            ],
            {"PATH": path},
            bare=True,
        )
        seen = os.path.join(k, "gen_fleet.env")
        seen = open(seen).read().splitlines() if os.path.exists(seen) else []
        check(
            seen == [kt],
            "node_up runs gen_fleet with KOTEKAN_SRC, the kotekan tree beside its checkout "
            "(saw %r; rc %d)" % (seen, rc),
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_eop_cron_reads_the_configs_the_manifest_owns():
    d = tempfile.mkdtemp(prefix="site_eop_")
    try:
        k, kt = site_tree(d, "eop_cron.sh")
        os.makedirs(os.path.join(k, "config", "generated"))
        with open(os.path.join(k, "scripts", "gnss", "gen_fleet.py"), "w") as fh:
            fh.write(GEN_FLEET_STUB)
        open(os.path.join(k, "config", "gnss_fleet_chord.yaml"), "w").write(
            "suffix: _other\n"
        )
        now_ns = int(time.time()) * 10 ** 9
        disk_end, file_end = now_ns + 86400 * 10 ** 9, now_ns + 3 * 86400 * 10 ** 9
        # A manifest suffix other than _multi: the config's name carries it.
        with open(
            os.path.join(k, "config", "generated", "chord_gnss_cx19_other.yaml"), "w"
        ) as fh:
            fh.write(
                "earth_rotation_data:\n  earth_orientation_parameter_table:\n"
                "  - t_inst_ns: %d\n" % disk_end
            )
        state = os.path.join(d, "eop-state.json")
        with open(state, "w") as fh:
            fh.write(
                '{"earth_orientation_parameter_table": [{"t_inst_ns": %d}]}\n'
                % file_end
            )
        bindir = os.path.join(d, "bin")
        stub(bindir, "scp", "exit 1\n")
        stub(bindir, "curl", "exit 7\n")
        log = os.path.join(d, "eop.log")
        rc, out = run(
            ["bash", os.path.join(k, "scripts", "gnss", "site", "eop_cron.sh")],
            {
                "PATH": bindir + ":" + os.environ["PATH"],
                "GNSS_NODES": "cxtest",
                "EOP_LOG": log,
                "EOP_STATE": state,
            },
            bare=True,
        )
        text = open(log).read() if os.path.exists(log) else out
        calls = os.path.join(k, "gen_fleet.calls")
        calls = open(calls).read().splitlines() if os.path.exists(calls) else []
        check(
            ("on-disk configs regenerated: table end %d -> %d" % (disk_end, file_end))
            in text,
            "a config whose table ends before the staged one is found under the manifest's suffix "
            "and regenerated (%r)" % text.strip().splitlines()[-1:],
        )
        check(
            calls == ["--check", ""],
            "after a --check, then a write (gen_fleet calls %r)" % calls,
        )
        seen = os.path.join(k, "gen_fleet.env")
        seen = open(seen).read().splitlines() if os.path.exists(seen) else []
        check(
            seen == [kt] * 3,
            "each gen_fleet run gets KOTEKAN_SRC, the kotekan tree beside the checkout (%r)"
            % seen,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
