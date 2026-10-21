#!/usr/bin/env python3
"""The site shell scripts' own checks, each run against stubs in a throwaway directory.

    python3 scripts/gnss/site/test_site_scripts.py

Nothing here reaches a node, a unit or a process outside the test: ssh, sudo, systemctl, curl,
scp and the like are stubs on PATH, and pgrep/pkill are wrapped to match only processes in the
test's own session (pgrep -s 0), so a pattern can never touch the live stack on this host.
"""
import copy
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "scripts", "gnss"))
from gnss_broker import iface  # noqa: E402

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


def run(cmd, env, timeout=120, cwd=None, bare=False, stdin=None):
    """Run cmd in a new session with env added to this environment, or with bare=True to
    cron's (PATH and HOME alone), and stdin as its input when given; (rc, stdout+stderr). On a
    timeout the session's process group is killed, so nothing it started outlives it."""
    if bare:
        e = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/")}
    else:
        e = dict(os.environ)
    e.update(env)
    p = subprocess.Popen(
        cmd,
        env=e,
        cwd=cwd,
        stdin=None if stdin is None else subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(input=stdin, timeout=timeout)
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


# ssh as ssh is: it reads its stdin for the remote side unless -n. Each call logs what it read,
# tagged preflight (the --interfaces-json read) or start.
SSH_STDIN_STUB = r"""n=0
while [ $# -gt 1 ]; do
    case "$1" in -n) n=1; shift ;; -o) shift 2 ;; -*) shift ;; *) break ;; esac
done
shift
case "$*" in *--interfaces-json*) tag=preflight ;; *) tag=start ;; esac
got=; [ $n = 1 ] || got=$(cat)
echo "$tag:$got" >> "$SSH_LOG"
exec bash -c "$*" < /dev/null
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


def iface_binary(d, kind):
    """A stand-in kotekan binary in d: `--interfaces-json` prints what a binary of `kind` prints
    (legacy: kotekan's "Invalid option" and exit 255; silent: nothing and exit 1; match: the
    golden; mismatch: a record slot moved; cube: the cube header grown; search: the search's
    REST version moved); anything else prints RUN and its argv, as the started program."""
    doc = copy.deepcopy(iface.golden().doc)
    if kind == "mismatch":
        doc["sections"]["record.row"]["REC_CP"]["slot"] += 1
    elif kind == "cube":
        doc["sections"]["cube"]["CUBE_HEADER_BYTES"] += 8
    elif kind == "search":
        doc["sections"]["rest"]["search"] += 1
    entry = {
        "schema": iface.SCHEMA,
        "digest": iface.doc_digest(doc),
        "sections": iface.section_digests(doc),
        "doc": doc,
    }
    out = os.path.join(d, "ij_%s.json" % kind)
    with open(out, "w") as fh:
        json.dump({"gnss": entry}, fh)
    ij = {
        "legacy": 'printf "Invalid option, run with -h to see options"; exit 255',
        "silent": "exit 1",
    }.get(kind, "cat '%s'; exit 0" % out)
    return stub(
        os.path.join(d, "bin_" + kind),
        "kotekan",
        'if [ "$1" = --interfaces-json ]; then %s; fi\n' 'echo "RUN $*"\n' % ij,
    )


def iface_config(d, stamped, extra=""):
    """A config whose baked EOP table ends two days ahead, with or without the stamp."""
    path = os.path.join(d, "cfg_%s.yaml" % ("stamped" if stamped else "plain"))
    end_ns = int((time.time() + 2 * 86400) * 1e9)
    with open(path, "w") as fh:
        fh.write("# GENERATED\n")
        if stamped:
            fh.write(iface.stamp_line(iface.golden()) + "\n")
        fh.write("t_inst_ns: %d\nstage:\n    signal: GPS_L5_Q_NH\n%s" % (end_ns, extra))
    return path


def test_node_up_iface_preflight():
    """The binary's interface document is read on the node before the start: a LEGACY binary,
    an unstamped config or a binary that cannot say warn and start; a disagreement with the
    golden or with the config's stamp refuses; GNSS_SKIP_PREFLIGHT=1 and GNSS_SKIP_IFACE=1 skip
    it, and only GNSS_SKIP_PREFLIGHT=1 the EOP check too."""
    d = tempfile.mkdtemp(prefix="site_node_iface_")
    try:
        path = node_up_stubs(d)
        # The F-engine controller answers; the node's REST port never does.
        stub(
            os.path.join(d, "bin"),
            "curl",
            'case "$*" in *frame0*) echo \'{"frame0_nano": 1}\' ;; *) exit 7 ;; esac\n',
        )
        sudo_log = os.path.join(d, "sudo.log")
        plain, stamped = iface_config(d, False), iface_config(d, True)

        def node_up(kind, cfg, **env):
            open(sudo_log, "w").close()
            e = {
                "PATH": path,
                "GNSS_CFG": cfg,
                "GNSS_BIN": iface_binary(d, kind),
                "GNSS_LOG": os.path.join(d, "node.log"),
                "SUDO_LOG": sudo_log,
            }
            e.update(env)
            rc, out = run(["bash", os.path.join(HERE, "node_up.sh"), "cxtest", "up"], e)
            return rc, out, "systemd-run" in open(sudo_log).read()

        for kind, cfg, want in (
            ("legacy", stamped, "WARN LEGACY binary"),
            ("silent", stamped, "the binary's interface is not checked"),
            ("match", plain, "carries no gnss_iface stamp"),
            ("match", stamped, "config stamp agrees"),
        ):
            rc, out, started = node_up(kind, cfg)
            check(
                started and want in out and "REFUSED" not in out,
                "a %s binary, %s config: '%s', and it starts (rc %d%s)"
                % (
                    kind,
                    os.path.basename(cfg),
                    want,
                    rc,
                    "" if started else ": " + out[-400:],
                ),
            )
        for kind, cfg, want in (
            ("mismatch", plain, "ERROR record.row/REC_CP/slot"),
            ("cube", stamped, "ERROR cube/CUBE_HEADER_BYTES: config sized for"),
        ):
            rc, out, started = node_up(kind, cfg)
            check(
                rc == 1 and not started and want in out and "REFUSED" in out,
                "a %s binary, %s config: refused before any start (rc %d)"
                % (kind, os.path.basename(cfg), rc),
            )
            for skip in ("GNSS_SKIP_PREFLIGHT", "GNSS_SKIP_IFACE"):
                rc, out, started = node_up(kind, cfg, **{skip: "1"})
                check(
                    started and "iface preflight" not in out,
                    "and %s=1 skips the check and starts it (rc %d)" % (skip, rc),
                )
        # The EOP table ends in the past: GNSS_SKIP_IFACE=1 still refuses it.
        stale = os.path.join(d, "cfg_stale.yaml")
        with open(stale, "w") as fh:
            fh.write("t_inst_ns: %d\n" % int((time.time() - 86400) * 1e9))
        rc, out, started = node_up("match", stale, GNSS_SKIP_IFACE="1")
        check(
            rc == 1 and not started and "baked EOP table" in out,
            "GNSS_SKIP_IFACE=1 leaves the EOP check in place (rc %d)" % rc,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_node_up_iface_preflight_leaves_stdin_alone():
    """The preflight's ssh reads none of node_up's stdin: what is typed ahead for the start's
    sudo prompt reaches the start."""
    d = tempfile.mkdtemp(prefix="site_node_stdin_")
    try:
        path = node_up_stubs(d)
        bindir = os.path.join(d, "bin")
        stub(
            bindir,
            "curl",
            'case "$*" in *frame0*) echo \'{"frame0_nano": 1}\' ;; *) exit 7 ;; esac\n',
        )
        stub(bindir, "ssh", SSH_STDIN_STUB)
        log = os.path.join(d, "ssh.log")
        env = {
            "PATH": path,
            "GNSS_CFG": iface_config(d, True),
            "GNSS_BIN": iface_binary(d, "legacy"),
            "GNSS_LOG": os.path.join(d, "node.log"),
            "SUDO_LOG": os.path.join(d, "sudo.log"),
            "SSH_LOG": log,
        }
        rc, out = run(
            ["bash", os.path.join(HERE, "node_up.sh"), "cxtest", "up"],
            env,
            stdin="typed-ahead\n",
        )
        got = open(log).read().splitlines() if os.path.exists(log) else []
        check(
            got == ["preflight:", "start:typed-ahead"],
            "the preflight's ssh reads nothing, the start's gets the input (%r)" % got,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_run_component_iface_preflight():
    """The gather and the aggregator run the check before their exec; --print never does, and
    a refusal stops the start. Each is checked on what it exchanges and is sized from: the
    cube is neither's, a record slot only the gather's."""
    d = tempfile.mkdtemp(prefix="site_rc_iface_")
    try:
        cfg = iface_config(d, True)
        rc_sh = os.path.join(HERE, "run_component.sh")
        for name, cfg_var, theirs, not_theirs in (
            ("gather", "GNSS_GATHER_CFG", "mismatch", "cube"),
            ("aggregator", "GNSS_AGG_CFG", "search", "mismatch"),
        ):
            for kind, ok in (
                ("legacy", True),
                ("match", True),
                (not_theirs, True),
                (theirs, False),
            ):
                env = {"GNSS_BIN": iface_binary(d, kind), cfg_var: cfg}
                rc, out = run(["bash", rc_sh, name], env)
                ran = "RUN --config %s" % cfg in out
                check(
                    (rc == 0 and ran)
                    if ok
                    else (rc == 1 and not ran and "refused" in out),
                    "%s on a %s binary %s (rc %d)"
                    % (name, kind, "starts" if ok else "is refused", rc),
                )
            rc, out = run(["bash", rc_sh, "--print", name], env)
            check(
                rc == 0 and "iface preflight" not in out,
                "%s --print runs no check" % name,
            )
            for skip in ("GNSS_SKIP_PREFLIGHT", "GNSS_SKIP_IFACE"):
                rc, out = run(["bash", rc_sh, name], dict(env, **{skip: "1"}))
                check(rc == 0 and "RUN --config" in out, "%s=1 skips it" % skip)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def cubearch_site(d):
    """cubearch_up.sh in a checkout of its own under d, with its own copy of the interface
    module and golden, so no node config there can disagree on frame_size; (script, config,
    the gnss_broker copy)."""
    k, _ = site_tree(d, "cubearch_up.sh")
    pkg = os.path.join(k, "python", "scripts", "gnss", "gnss_broker")
    os.makedirs(pkg)
    for name in ("__init__.py", "iface.py", "iface_golden.json", "iface_pin.json"):
        shutil.copy(os.path.join(os.path.dirname(iface.__file__), name), pkg)
    arch = os.path.join(d, "archive")
    extra = (
        "cube_recv: {listen_port: 11070}\ncube_write: {base_dir: %s}\n"
        "cube_buf: {frame_size: 1024}\n" % arch
    )
    cfg = iface_config(d, True, extra)
    return os.path.join(k, "scripts", "gnss", "site", "cubearch_up.sh"), cfg, pkg


def test_cubearch_up_iface_preflight():
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_cube_iface_"))
    try:
        sh, cfg, _ = cubearch_site(d)
        for kind, ok in (("legacy", True), ("match", True), ("cube", False)):
            rc, out = run(
                ["bash", sh, "--check", cfg], {"GNSS_BIN": iface_binary(d, kind)}
            )
            check(
                (rc == 0 and "pre-flight OK" in out)
                if ok
                else (rc == 1 and "ERROR cube/CUBE_HEADER_BYTES" in out),
                "cubearch_up --check on a %s binary %s (rc %d)"
                % (kind, "passes" if ok else "fails", rc),
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_preflight_that_cannot_run_never_refuses():
    """Only the check's refusal status (3) refuses. A gnss_broker that does not import (the
    interpreter exits 1) or a golden that does not load (exit 2) is a check that did not run:
    it warns and the start goes on, even for a binary the check would refuse."""
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_iface_broken_"))
    try:
        sh, cfg, pkg = cubearch_site(d)
        env = {"GNSS_BIN": iface_binary(d, "cube")}
        rc, out = run(["bash", sh, "--check", cfg], env, cwd=d)
        check(
            rc == 1 and "REFUSED" in out,
            "with the check running, the binary is refused (rc %d)" % rc,
        )
        for name, was, now, code in (
            ("iface.py", "import argparse", "<<<<<<< HEAD\nimport argparse", 1),
            ("iface_golden.json", '"CUBE_VERSION": 3', '"CUBE_VERSION": 4', 2),
        ):
            path = os.path.join(pkg, name)
            with open(path) as fh:
                text = fh.read()
            with open(path, "w") as fh:
                fh.write(text.replace(was, now, 1))
            rc, out = run(["bash", sh, "--check", cfg], env, cwd=d)
            with open(path, "w") as fh:
                fh.write(text)
            check(
                rc == 0
                and "WARN the check could not run (exit %d)" % code in out
                and "pre-flight OK" in out,
                "a broken %s (exit %d) warns and starts (rc %d%s)"
                % (name, code, rc, "" if rc == 0 else ": " + out[-400:]),
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
if "--check" in sys.argv:
    sys.exit(int(os.environ.get("GEN_FLEET_CHECK_RC", "0")))
"""


def site_tree(d, *scripts):
    """A GNSS checkout d/k holding the named site scripts and kotekan_src.sh, with a kotekan
    tree beside it at d/kotekan; (checkout, kotekan tree)."""
    k = os.path.join(d, "k")
    site = os.path.join(k, "scripts", "gnss", "site")
    os.makedirs(site)
    for name in scripts + ("kotekan_src.sh", "iface_preflight.sh"):
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
        d = os.path.realpath(tempfile.mkdtemp(prefix="site_ksrc_"))
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


def test_kotekan_src_helper_is_physical_and_takes_a_relative_checkout():
    """Through a symlinked checkout, with a kotekan tree beside the link and another beside the
    real directory, the helper and kotekan_src.py, each reached through the link, both take the
    one beside the real directory; and a relative checkout argument still reaches the tree
    beside it."""
    py = os.path.join(
        HERE, "..", "..", "..", "python", "scripts", "gnss", "kotekan_src.py"
    )
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_ksrc_"))
    try:
        real = os.path.join(d, "P", "c")
        os.makedirs(os.path.join(real, "scripts", "gnss", "site"))
        os.makedirs(os.path.join(real, "python", "scripts", "gnss"))
        shutil.copy(
            os.path.join(HERE, "kotekan_src.sh"),
            os.path.join(real, "scripts", "gnss", "site"),
        )
        shutil.copy(py, os.path.join(real, "python", "scripts", "gnss"))
        for t in ("P", "L"):
            os.makedirs(os.path.join(d, t, "kotekan", "lib", "stages", "gnss"))
        link = os.path.join(d, "L", "c")
        os.symlink(real, link)
        want = os.path.join(d, "P", "kotekan")
        # kotekan_src.py imported through the link.
        _, py_out = run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import kotekan_src; "
                "print(kotekan_src.SRC)",
                os.path.join(link, "python", "scripts", "gnss"),
            ],
            {},
            bare=True,
        )
        check(
            py_out.strip() == want,
            "through the link kotekan_src.py finds the tree beside the real checkout (%r)"
            % py_out.strip(),
        )
        helper = os.path.join("scripts", "gnss", "site", "kotekan_src.sh")
        # Sourced through the link by its absolute path; then asked for the link by name.
        rc, out = run(
            [
                "bash",
                "-c",
                'set -u; . "$1"; echo "$GNSS_CHECKOUT"; gnss_kotekan_src "$GNSS_CHECKOUT"; '
                'gnss_kotekan_src "$2"',
                "sh",
                os.path.join(link, helper),
                link,
            ],
            {},
            bare=True,
        )
        check(
            rc == 0 and out.splitlines() == [real, want, want],
            "through the link the checkout is the real directory, and both calls find %s "
            "(rc %d, %r)" % (want, rc, out.splitlines()),
        )
        # A relative checkout, as GNSS_K=. passes it: its ../kotekan is the tree beside it.
        for cwd, arg in ((real, "."), (os.path.join(d, "P"), "c")):
            rc, out = run(
                [
                    "bash",
                    "-c",
                    'set -u; . "$1"; gnss_kotekan_src "$2"',
                    "sh",
                    os.path.join(real, helper),
                    arg,
                ],
                {},
                cwd=cwd,
                bare=True,
            )
            check(
                rc == 0 and out.splitlines() == [want],
                "gnss_kotekan_src %s from %s finds %s (rc %d, %r)"
                % (arg, os.path.relpath(cwd, d), want, rc, out.splitlines()),
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_install_user_units_names_its_own_checkout():
    """install_user_units.sh <dir> writes every unit with its own checkout, as a physical path,
    in place of /home/kvand/gnss/kotekan, and changes nothing else: from the main checkout the
    units are installed byte for byte. It reloads nothing."""
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_units_"))
    try:
        # A checkout path holding the characters a sed replacement treats specially.
        k = os.path.join(d, "a&b|c", "k")
        units = os.path.join(k, "scripts", "gnss", "site", "systemd")
        shutil.copytree(os.path.join(HERE, "systemd"), units)
        link = os.path.join(d, "link")
        os.symlink(k, link)
        bindir = os.path.join(d, "bin")
        log = os.path.join(d, "calls.log")
        for tool in ("systemctl", "loginctl"):
            stub(bindir, tool, 'echo "$0 $*" >> "$UNITS_TEST_LOG"\n')
        out_dir = os.path.join(d, "out")
        rc, out = run(
            [
                "sh",
                os.path.join(
                    link, "scripts", "gnss", "site", "systemd", "install_user_units.sh"
                ),
                out_dir,
            ],
            {"PATH": bindir + ":/usr/bin:/bin", "UNITS_TEST_LOG": log},
            bare=True,
        )
        names = sorted(
            f
            for f in os.listdir(units)
            if f.startswith("gnss-") and f.endswith((".service", ".target"))
        )
        check(
            rc == 0 and os.path.isdir(out_dir) and sorted(os.listdir(out_dir)) == names,
            "it writes the %d units and nothing else (rc %d: %s)"
            % (len(names), rc, out.strip()[-200:]),
        )
        main = "/home/kvand/gnss/kotekan/"
        texts = {f: open(os.path.join(units, f)).read() for f in names}
        check(
            any(main in t for t in texts.values()),
            "the units name the main checkout, so there is something to replace",
        )
        wrong = [
            f
            for f in names
            if not os.path.exists(os.path.join(out_dir, f))
            or open(os.path.join(out_dir, f)).read() != texts[f].replace(main, k + "/")
        ]
        check(
            not wrong,
            "each unit names the installer's checkout by its real path, and is otherwise "
            "the file as it is (differ: %s)" % wrong,
        )
        check(not os.path.exists(log), "and nothing is reloaded")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_install_user_units_will_not_move_the_stack_unasked():
    """A live install from a checkout other than the one the installed broker unit runs is
    refused unless GNSS_UNITS_CHECKOUT names the checkout; an empty <dir> is refused."""
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_units_live_"))
    try:
        k = os.path.join(d, "k")
        shutil.copytree(
            os.path.join(HERE, "systemd"),
            os.path.join(k, "scripts", "gnss", "site", "systemd"),
        )
        inst = os.path.join(
            k, "scripts", "gnss", "site", "systemd", "install_user_units.sh"
        )
        bindir = os.path.join(d, "bin")
        log = os.path.join(d, "calls.log")
        for tool in ("systemctl", "loginctl"):
            stub(bindir, tool, 'echo "$0 $*" >> "$UNITS_TEST_LOG"\n')
        live = os.path.join(d, "config", "systemd", "user")
        os.makedirs(live)
        broker = os.path.join(live, "gnss-broker.service")
        with open(broker, "w") as f:
            f.write("ExecStart=/elsewhere/scripts/gnss/site/run_component.sh broker\n")
        env = {
            "PATH": bindir + ":/usr/bin:/bin",
            "UNITS_TEST_LOG": log,
            "XDG_CONFIG_HOME": os.path.join(d, "config"),
            "GNSS_LOGS": os.path.join(d, "logs"),
        }

        rc, out = run(["sh", inst, ""], env, bare=True)
        check(rc == 2 and "empty" in out, "an empty <dir> is refused (rc %d)" % rc)

        rc, out = run(["sh", inst], env, bare=True)
        check(
            rc == 1
            and "REFUSING" in out
            and "/elsewhere" in out
            and open(broker).read().startswith("ExecStart=/elsewhere/")
            and not os.path.exists(log),
            "a checkout other than the installed one is refused, and nothing is written "
            "(rc %d: %s)" % (rc, out.strip()[-200:]),
        )

        env["GNSS_UNITS_CHECKOUT"] = k
        rc, out = run(["sh", inst], env, bare=True)
        check(
            rc == 0
            and ("ExecStart=%s/scripts/gnss/site/run_component.sh broker" % k)
            in open(broker).read()
            and os.path.exists(log),
            "named by GNSS_UNITS_CHECKOUT, it installs and reloads (rc %d: %s)"
            % (rc, out.strip()[-200:]),
        )

        del env["GNSS_UNITS_CHECKOUT"]
        rc, out = run(["sh", inst], env, bare=True)
        check(
            rc == 0,
            "reinstalling from the checkout the units run needs no name (rc %d)" % rc,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_run_component_exports_no_kotekan_src():
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_print_"))
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
            "argv\t%s/build_nodpdk/kotekan/kotekan\n" % kt in out["gather"],
            "the gather runs the binary of the kotekan tree beside the checkout",
        )
        # The broker speaks the interface golden it carries and reads nothing from kotekan.
        exporting = [n for n in out if "env\tKOTEKAN_SRC" in out[n]]
        check(not exporting, "no component exports KOTEKAN_SRC (%s)" % exporting)
        check(
            "env\tGNSS_CHAIN=gps_l5\n" in out["obs"],
            "an obs writer's GNSS_CHAIN, set by its unit, is still listed",
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_node_up_exports_kotekan_src_to_gen_fleet():
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_node_up_src_"))
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
    d = os.path.realpath(tempfile.mkdtemp(prefix="site_eop_"))
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
        for check_rc, want in (
            (
                1,
                "on-disk configs differ from the manifest beyond live data; NOT regenerated",
            ),
            (2, "gen_fleet --check could not compare (rc 2); NOT regenerated"),
        ):
            rc, out = run(
                ["bash", os.path.join(k, "scripts", "gnss", "site", "eop_cron.sh")],
                {
                    "PATH": bindir + ":" + os.environ["PATH"],
                    "GNSS_NODES": "cxtest",
                    "EOP_LOG": log,
                    "EOP_STATE": state,
                    "GEN_FLEET_CHECK_RC": str(check_rc),
                },
                bare=True,
            )
            last = open(log).read().strip().splitlines()[-1:]
            check(
                bool(last) and want in last[0],
                "a --check that exits %d is logged as such (%r)" % (check_rc, last),
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
