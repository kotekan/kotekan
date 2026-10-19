#!/usr/bin/env python3
"""beamcube_daily.sh's own check: the chain-health step and the mask the build is given.

    python3 scripts/gnss/site/test_beamcube_daily.py

GNSS_PY is a stub, so neither gnss_chain_health.py nor gnss_beam_cube.py runs: the stub fails
the chain-health run (with or without writing a mask), records the build's argv and writes a
master, and writes an index for the export. Everything lives in a throwaway directory.
"""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DAY = "20260907"

PY_STUB = r"""#!/bin/bash
# $1 is -u, $2 the tool, $3 the subcommand
prev=""; out=""; outdir=""
for a in "$@"; do
    [ "$prev" = --out ] && out=$a
    [ "$prev" = --outdir ] && outdir=$a
    prev=$a
done
case "$2" in
*gnss_chain_health.py)
    if [ "$BC_HEALTH" = episodes ]; then mkdir -p "$(dirname "$out")"; echo '{"chains": {}}' > "$out"; fi
    exit 1 ;;
*gnss_beam_cube.py)
    case "$3" in
    build) printf '%s\0' "$@" > "$BC_TEST_DIR/build.argv"; echo master > "$outdir/cube_@DAY@_nside64.npz" ;;
    export) mkdir -p "$outdir"; echo '{"days": [{"day": "@DAY@"}]}' > "$outdir/index.json" ;;
    esac ;;
esac
exit 0
""".replace(
    "@DAY@", DAY
)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def run_daily(d, health, name=None, old_mask=False, extra=None):
    """One beamcube_daily.sh run in a fresh tree under d; (rc, output, build argv, its OBS).
    old_mask leaves a mask from an earlier run in place; extra is more environment."""
    root = os.path.join(d, name or health)
    os.makedirs(os.path.join(root, "cube", "rung12", "p1", "s1"))
    open(os.path.join(root, "cube", "rung12", "p1", "s1", DAY + ".h5"), "w").close()
    os.makedirs(os.path.join(root, "obs"))
    open(os.path.join(root, "obs", "gps_l5_%s.jsonl" % DAY), "w").write("{}\n")
    if old_mask:
        os.makedirs(os.path.join(root, "obs", "health"))
        mask = os.path.join(root, "obs", "health", "mask_%s.json" % DAY)
        with open(mask, "w") as fh:
            fh.write('{"chains": {"old": []}}\n')
        os.utime(mask, (time.time() - 3600, time.time() - 3600))
    py = os.path.join(root, "py")
    with open(py, "w") as fh:
        fh.write(PY_STUB)
    os.chmod(py, os.stat(py).st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env.update(
        {
            "GNSS_PY": py,
            "CUBE_ROOT": os.path.join(root, "cube"),
            "BEAMCUBE_DIR": os.path.join(root, "beam"),
            "GNSS_OBS_OUT": os.path.join(root, "obs"),
            "BEAMCUBE_LOCK": os.path.join(root, "lock"),
            "BC_TEST_DIR": root,
            "BC_HEALTH": health,
        }
    )
    env.update(extra or {})
    os.makedirs(env["BEAMCUBE_DIR"])
    p = subprocess.run(
        ["bash", os.path.join(HERE, "beamcube_daily.sh"), DAY],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    argv_path = os.path.join(root, "build.argv")
    argv = open(argv_path).read().split("\0")[:-1] if os.path.exists(argv_path) else []
    return p.returncode, p.stdout, argv, env["GNSS_OBS_OUT"]


def mask_arg(argv):
    return argv[argv.index("--health-mask") + 1] if "--health-mask" in argv else None


def test_chain_health_crash_is_not_episodes():
    d = tempfile.mkdtemp(prefix="beamcube_daily_")
    try:
        rc, out, argv, obs = run_daily(d, "crash")
        check(
            "chain-health FAILED rc=1" in out and "EPISODES" not in out,
            "a chain-health run that wrote no mask is logged as FAILED, not as masked episodes",
        )
        check(
            rc == 0 and mask_arg(argv) == "",
            "and the build is told explicitly to use no mask (rc %d, --health-mask %r)"
            % (rc, mask_arg(argv)),
        )
        rc, out, argv, obs = run_daily(d, "episodes")
        want = os.path.join(obs, "health", "mask_%s.json" % DAY)
        check(
            "chain-health: EPISODES on %s (masked)" % DAY in out
            and "FAILED" not in out,
            "a chain-health run that wrote its mask and exited 1 is logged as masked episodes",
        )
        check(
            rc == 0 and mask_arg(argv) == want,
            "and the build reads that mask under GNSS_OBS_OUT (rc %d, --health-mask %r)"
            % (rc, mask_arg(argv)),
        )
        # A regeneration that crashes leaves the earlier mask: not a mask this run wrote.
        rc, out, argv, obs = run_daily(
            d, "crash", "remask", old_mask=True, extra={"BEAMCUBE_REMASK": "1"}
        )
        want = os.path.join(obs, "health", "mask_%s.json" % DAY)
        check(
            "chain-health FAILED rc=1 on %s; building with the mask already there" % DAY
            in out
            and "EPISODES" not in out,
            "a crashed regeneration over an earlier mask is FAILED, not masked episodes",
        )
        check(
            rc == 0 and mask_arg(argv) == want,
            "and the build keeps the earlier mask (rc %d, --health-mask %r)"
            % (rc, mask_arg(argv)),
        )
        rc, out, argv, obs = run_daily(d, "crash", "kept", old_mask=True)
        want = os.path.join(obs, "health", "mask_%s.json" % DAY)
        check(
            "chain-health: keeping existing %s" % want in out
            and rc == 0
            and mask_arg(argv) == want,
            "an existing mask is kept and given to the build (rc %d, --health-mask %r)"
            % (rc, mask_arg(argv)),
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
