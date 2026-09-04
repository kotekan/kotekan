#!/usr/bin/env python3
"""THE ACTUATOR'S SIGN, measured through the shipped chain (task #51 F3).

A code loop with the sign inverted does not track badly -- it DIVERGES, and no update rate,
gain or clamp saves it. The sign lives in a convention shared by four components (the
discriminator's E/L labelling, tau's definition, the trim's addition to the model phase in
propagate_seed, and the despread's spacing), and this repository has been bitten repeatedly by
exactly that kind of cross-component convention: gnssSeedTransport.hpp exists because every
seeding bug in the week to 2026-08-02 lived between two stages and was invisible to a test of
either alone.

So this measures it end to end rather than reasoning about it. `e2e --skip-search --trim X`
commands the actuator exactly as cudaGnssChordTrack does, on a synthetic sky with known truth:
a trim is then a KNOWN code error of known sign, and the question is whether the loop's
response points back.

    tau = -clamp(disc)/4 * (spacing/0.5)          (gnss::dll_tau / combdll.dll_tau)
    trim <- trim + gain*tau                       (dll_integrate)

Negative feedback requires sign(tau) == -sign(err) for every point inside the pull-in region.

    ./trim_sign_gate.py            # sweep, verdict
    ./trim_sign_gate.py -v         # and print the table

⚠️ THE NULL IS NOT AT trim 0. e2e's default seed carries a real 27 s age and a real Doppler
rate, so the open-loop error is ~-1.17 chips. The first version of this sweep ran +-0.4 chips
about ZERO, which put every sample outside the +-0.5-chip pull-in region where the E/L
discriminator has no gradient at all -- disc sat railed near -0.94 and the sweep "showed"
nothing. The null is located first, and the sweep is about the null.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s+0\s+\S+\s+\S+\s+(\S+)\s+\S+\s+\|\s+(\S+)\s+(\S+)\s+(\S+)")


def run(exe, trim, extra):
    """(err_chips, q, disc) for one commanded trim."""
    r = subprocess.run([exe, "--skip-search", "--nrec", "1", "--trim", "%.6f" % trim] + extra,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("e2e failed (%d): %s" % (r.returncode, r.stderr.strip()[:400]))
    for line in r.stdout.splitlines():
        m = ROW.match(line)
        if m:
            return float(m.group(1)), float(m.group(3)), float(m.group(4))
    raise SystemExit("could not find the record row in e2e output:\n" + r.stdout[-800:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=os.path.join(HERE, "e2e"))
    ap.add_argument("--span", type=float, default=0.40, help="sweep half-width about the null")
    ap.add_argument("--step", type=float, default=0.10)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra e2e flags (e.g. --signal GAL_E5A_Q_CS)")
    a = ap.parse_args()
    if not os.path.exists(a.exe):
        raise SystemExit("no %s -- build it with ./build_tool.sh e2e" % a.exe)

    # LOCATE THE NULL, don't assume it. err is linear in the commanded trim with unit slope
    # (the trim is added to the model phase), so one probe finds it exactly.
    err0, _q0, _d0 = run(a.exe, 0.0, a.extra)
    null = -err0
    print("open-loop error %+.4f chips -> null at trim %+.4f" % (err0, null))

    rows = []
    n = int(round(a.span / a.step))
    for i in range(-n, n + 1):
        t = null + i * a.step
        rows.append((t,) + run(a.exe, t, a.extra))
    if a.verbose:
        print("\n   trim      err       q      disc      tau=-disc/4")
        for t, e, q, d in rows:
            print("  %+6.3f  %+7.4f  %6.3f  %+8.4f  %+8.4f" % (t, e, q, d, -d / 4.0))

    # ---- THE VERDICT ------------------------------------------------------------------
    bad = []
    inside = [r for r in rows if abs(r[1]) < 0.45]   # the pull-in region; outside it the
    if len(inside) < 5:                              # discriminator has no gradient at all
        bad.append("only %d samples inside the pull-in region -- the sweep is not testing the "
                   "discriminator" % len(inside))
    for t, e, q, d in inside:
        if abs(e) < 0.02:
            continue                                  # at the null both signs are ~0
        tau = -d / 4.0
        if tau * e >= 0.0:
            bad.append("trim %+.3f: err %+.4f and tau %+.4f have the SAME sign -- POSITIVE "
                       "feedback, the loop diverges here" % (t, e, tau))
    # disc must actually cross zero, and q must peak at the null: a discriminator that is
    # merely one-signed would pass a sign test while carrying no information.
    if not (min(d for _t, _e, _q, d in inside) < 0.0 < max(d for _t, _e, _q, d in inside)):
        bad.append("disc never changes sign across the null -- it is not a discriminator here")
    qn = max(rows, key=lambda r: r[2])
    if abs(qn[1]) > a.step:
        bad.append("q peaks at err %+.4f, not at the null -- the prompt tap and the "
                   "discriminator disagree about where the peak is" % qn[1])

    if bad:
        print("\nFAIL -- the actuator's sign is not verified:")
        for b in bad:
            print("  " + b)
        return 1
    slope = [( -d / 4.0) / e for _t, e, _q, d in inside if abs(e) > 0.05]
    print("\nPASS -- negative feedback confirmed over %d samples in the pull-in region."
          % len(inside))
    print("  disc crosses zero at the null; q peaks there (%.3f)." % qn[2])
    print("  tau/err = %.2f .. %.2f (must be negative; ~-1 means tau estimates the error)"
          % (min(slope), max(slope)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
