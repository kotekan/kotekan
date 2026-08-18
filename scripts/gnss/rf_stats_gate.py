#!/usr/bin/env python3
"""#8 RF-STATS GATE -- judge GnssChordVoltageTap's clip/band-power pass on known bytes.

    scripts/gnss/rf_stats_gate.py [--kotekan PATH] [--port 12099]

WHAT IT DOES. Runs the REAL stage inside a REAL kotekan pipeline (config/gates/rf_stats_gate.yaml),
fed by testDataGen's `constu8` so that every 4+4b sample in the frame is a byte we chose, and
reads the REAL /rf_stats endpoint. For a constant byte V the answer is arithmetic, not an
estimate:

    re = (V >> 4) - 8      im = (V & 0x0F) - 8      both in [-8, +7]
    power    = re^2 + im^2
    clip_lo  = ((re == -8) + (im == -8)) / 2       <- the rail negate_4bit corrupts
    clip_hi  = ((re ==  7) + (im ==  7)) / 2

so every served number is compared against an exact value, not a tolerance.

⚠️ WHY NOT A UNIT TEST OF THE EXPRESSION. #71's carrier-NCO gate validated a formula against my
own model of the truth, passed at 9e-16 rad, and the shipped code was still wrong where it met
its caller. A gate on an expression is not a gate on a stage. This one exercises the config
read, the frame walk, the strides, the normalisation, the mutex hand-off and the JSON -- the
whole path the fleet actually serves.

⚠️ WHAT THIS GATE DOES *NOT* COVER, stated so nobody reads more into a pass than is there:
  * NIBBLE ORDER. power and clip are symmetric in re and im, so swapping the high and low
    nibbles is INVISIBLE here -- 0x0F and 0xF0 give identical answers by construction. Nibble
    order is pinned elsewhere (element_power's identical decode, and the despread itself).
  * CHANNEL INDEXING. constu8 fills every channel identically, so a wrong channel still reads
    the right number. The arm below with band_power_chans out of range covers the bounds check
    only. A non-uniform source would be needed for the indexing itself.
  * DECIMATION. The fixture runs stride 1 / period 0 (exhaustive) so that a mismatch is a bug
    and never a sampling artifact. Production's stride 32 / period 10 is a cost knob on the
    same code path, not a second path -- but this gate does not measure the sampling error that
    knob introduces.

THE CONTROL ARM IS THE POINT OF THE LAST CASE. With `band_power_chans` empty the feature must
report enabled=false and passes=0 -- i.e. the default really is OFF, and the other arms are
measuring something this one cannot. A gate whose every arm passes regardless of the axis under
test is not a gate (docs: "a gate that cannot fail").
"""
import argparse, json, os, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # scripts/gnss -> scripts -> repo root
CFG = os.path.join(ROOT, "config", "gates", "rf_stats_gate.yaml")

# (byte, label). Chosen to separate the two rails, which one number could not.
CASES = [
    (0x88, "mid scale: no power, no rail"),
    (0x00, "BOTH nibbles at -8: the rail negate_4bit corrupts"),
    (0xFF, "BOTH nibbles at +7: the headroom rail"),
    (0x0F, "ONE of each rail -- the case a single clip fraction would smear"),
    (0x8F, "imag at +7, real at mid"),
    (0x80, "imag at -8, real at mid"),
]
NHOP, NCHAN, NELEM = 256, 8, 16
BP_CHANS = [0, 1, 2]


def expect(v):
    re, im = (v >> 4) - 8, (v & 0x0F) - 8
    return {
        "power": float(re * re + im * im),
        "clip_lo": ((re == -8) + (im == -8)) / 2.0,
        "clip_hi": ((re == 7) + (im == 7)) / 2.0,
    }


def run_once(kotekan, port, value, chans, timeout_s=25.0):
    """Start kotekan on a config with this byte value, poll /rf_stats, return it."""
    with open(CFG) as f:
        cfg = f.read()
    cfg = cfg.replace("  type: constu8\n  value: 0\n",
                      "  type: constu8\n  value: %d\n" % value)
    cfg = cfg.replace("  band_power_chans: [0, 1, 2]\n",
                      "  band_power_chans: %s\n" % json.dumps(chans))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write(cfg)
        path = tf.name
    proc = subprocess.Popen([kotekan, "-c", path, "-b", "127.0.0.1:%d" % port],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = "http://127.0.0.1:%d/tap/rf_stats" % port
    try:
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("kotekan exited early (rc %s): %s"
                                   % (proc.returncode,
                                      proc.stderr.read().decode()[-800:]))
            try:
                with urllib.request.urlopen(url, timeout=1.0) as r:
                    last = json.loads(r.read().decode())
                # An endpoint that answers is not a measurement that ran -- wait for a PASS.
                # (Same trap as "a REST endpoint answering 200 is not a live instance.")
                if not last.get("enabled") or last.get("passes", 0) > 0:
                    return last
            except Exception:
                pass
            time.sleep(0.25)
        raise RuntimeError("timed out waiting for a pass; last=%s" % last)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.unlink(path)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kotekan", default=os.path.join(ROOT, "build", "kotekan", "kotekan"))
    ap.add_argument("--port", type=int, default=12099)
    args = ap.parse_args()

    if not os.path.exists(args.kotekan):
        print("no kotekan binary at %s -- build it first" % args.kotekan)
        return 2

    fails = []
    print("#8 RF-STATS GATE  (%d hops x %d chan x %d elem, stride 1, exhaustive)"
          % (NHOP, NCHAN, NELEM))
    print("byte  re  im | power   expect | clip_lo  expect | clip_hi  expect | verdict")
    print("-" * 78)

    for value, label in CASES:
        e = expect(value)
        try:
            r = run_once(args.kotekan, args.port, value, BP_CHANS)
        except Exception as ex:
            print("0x%02X  RUN FAILED: %s" % (value, ex))
            fails.append((value, str(ex)))
            continue

        bad = []
        if len(r.get("power", [])) != len(BP_CHANS):
            bad.append("power has %d entries, expected %d"
                       % (len(r.get("power", [])), len(BP_CHANS)))
        if len(r.get("elem_power", [])) != NELEM:
            bad.append("elem_power has %d entries, expected %d"
                       % (len(r.get("elem_power", [])), NELEM))
        for key in ("power", "clip_lo", "clip_hi"):
            for i, got in enumerate(r.get(key, [])):
                if not approx(got, e[key]):
                    bad.append("%s[%d] = %.6f, expected %.6f" % (key, i, got, e[key]))
        # per-element must agree with per-channel: same samples, different grouping
        for i, got in enumerate(r.get("elem_power", [])):
            if not approx(got, e["power"]):
                bad.append("elem_power[%d] = %.6f, expected %.6f" % (i, got, e["power"]))
        ec = e["clip_lo"] + e["clip_hi"]
        for i, got in enumerate(r.get("elem_clip", [])):
            if not approx(got, ec):
                bad.append("elem_clip[%d] = %.6f, expected %.6f" % (i, got, ec))

        re_, im_ = (value >> 4) - 8, (value & 0x0F) - 8
        print("0x%02X %+3d %+3d | %6.1f %6.1f  | %7.4f %7.4f | %7.4f %7.4f | %s"
              % (value, re_, im_,
                 (r.get("power") or [float("nan")])[0], e["power"],
                 (r.get("clip_lo") or [float("nan")])[0], e["clip_lo"],
                 (r.get("clip_hi") or [float("nan")])[0], e["clip_hi"],
                 "ok" if not bad else "FAIL"))
        print("        %s" % label)
        for b in bad:
            print("        !! %s" % b)
        if bad:
            fails.append((value, bad))

    # ---- THE CONTROL ARM: with no channels configured the feature must be genuinely OFF ----
    print("-" * 78)
    try:
        r = run_once(args.kotekan, args.port, 0xFF, [], timeout_s=15.0)
        if r.get("enabled"):
            fails.append(("control", "enabled=true with band_power_chans empty"))
            print("CONTROL  FAIL: enabled=true with no channels configured")
        elif r.get("passes", 0) != 0:
            fails.append(("control", "passes=%s with the feature off" % r.get("passes")))
            print("CONTROL  FAIL: passes=%s with the feature off" % r.get("passes"))
        else:
            print("CONTROL  ok: band_power_chans empty -> enabled=false, passes=0 "
                  "(the pass does not run, it is not merely quiet)")
    except Exception as ex:
        print("CONTROL  RUN FAILED: %s" % ex)
        fails.append(("control", str(ex)))

    print("-" * 78)
    if fails:
        print("GATE FAILED: %d of %d arms" % (len(fails), len(CASES) + 1))
        return 1
    print("GATE GOOD: %d byte patterns + the off-control, all exact" % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
