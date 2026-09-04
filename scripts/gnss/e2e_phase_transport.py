#!/usr/bin/env python3
"""Does a broker-computed PHASE command the same code as its own cp0 ARGUMENT? (#45 step 6)

The dead-reckon/slew seed paths ship code_phase_chips -- an argument back-referenced to
sample 0, ~2.3 days of lever -- while the search-fed path already ships
code_phase_at_ref_chips, which propagate_seed prefers and which carries no lever at all.
Closing that gap means the broker computing the phase itself, and the only question that
matters is whether the broker's notion of "physical phase at ref_hop" is the one the C++
side uses. It is NOT obvious: ChannelizedReplicaBank references a hop at its LAST sample
(window_start + fft_len - 1) while the broker's dr_cp0/dr_seed_phys pair references the
first, so they differ by ~52.377 chips -- a constant the clock solve currently absorbs,
and which would become a 52-chip STEP the moment the transport switches.

This measures it against the shipped code instead of asserting it: build one seed for a
known injected truth, ship it three ways through the REAL propagate_seed + GPU despread,
and compare the commanded-phase errors.

  arg      cp0 only                        (production, model-primary chains today)
  phase    cp0 + phase = dr_seed_phys      (the naive port -- first-sample convention)
  phase+   cp0 + phase = dr_seed_phys+hop  (the correction under test)

PASS = `phase+` matches `arg`; the size of `phase`'s disagreement is the offset itself.

usage:  e2e_phase_transport.py [--prn N] [--e2e-arg ...]
"""
import argparse
import json
import os
import re
import subprocess
import sys

K = "/home/kvand/gnss/kotekan"
sys.path.insert(0, os.path.join(K, "python/scripts/gnss"))
from gnss_broker.fits import dr_cp0, dr_seed_phys   # noqa: E402

HPS = 195312.5
CHIP = 10.23e6
CARR = 1176.45e6
SGN = 1.0
L = 10230.0
MOD = 20 * L
FFT_LEN = 8192

ap = argparse.ArgumentParser()
ap.add_argument("--prn", type=int, default=3)
ap.add_argument("--outdir", default="/tmp/e2e_phase_transport")
ap.add_argument("--e2e-arg", action="append", default=["--s-stride", "1",
                                                       "--s-nchan", "106"])
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
DET = os.path.join(a.outdir, "det.json")

# 1. one detection for a known truth, through the shipped search
subprocess.run([os.path.join(K, "scripts/gnss/e2e"), "--prn", str(a.prn),
                "--emit-detection", DET] + a.e2e_arg,
               check=True, stdout=subprocess.DEVNULL)
det = json.load(open(DET))[0]
dop, ref_hop = det["doppler_hz"], det["ref_hop"]
cp0 = det["code_phase_chips"]

# 2. TRUTH: the phase the tracker must command at ref_hop, in ITS convention. Taken from
# e2e's own --skip-search seed line (sd.phase_ref_chips = the injected truth), so this
# script never has to reimplement the C++ side to judge it.
_probe = subprocess.run([os.path.join(K, "scripts/gnss/e2e"), "--prn", str(a.prn),
                         "--skip-search"] + a.e2e_arg,
                        check=True, capture_output=True, text=True).stdout
phys_tracker = float(re.search(r"ph@ref\s+([\d.]+)", _probe).group(1))

# What the BROKER's dead-reckon path holds: the same phase in ITS convention (a hop earlier
# -- dr_cp0/dr_seed_phys reference a hop's FIRST sample), and the argument it ships today.
cps_dop = CHIP / HPS / FFT_LEN * (1.0 + SGN * dop / CARR)
hop_off = (FFT_LEN - 1) * cps_dop
phase_first = (phys_tracker - hop_off) % MOD
cp0_broker = dr_cp0(phase_first, ref_hop / HPS, dop, CHIP, CARR, SGN, MOD)

print("detection: PRN %d dop %+.4f ref_hop %d" % (a.prn, dop, ref_hop))
print("  truth phase at ref_hop (tracker convention) %12.3f" % phys_tracker)
print("  broker-held phase       (first-sample)      %12.3f" % phase_first)
print("  one-hop offset                              %12.4f chips" % hop_off)
print("  cp0 the broker ships today (dr_cp0)         %12.3f" % cp0_broker)

# 3. three seed files, three runs of the shipped chain
def run(tag, extra):
    p = os.path.join(a.outdir, "seed_%s.jsonl" % tag)
    row = {"prn": a.prn, "doppler_hz": dop, "code_phase_chips": cp0_broker,
           "code_phase_rate": 0.0, "ref_hop": ref_hop, "doppler_rate_hz_s": 0.0,
           # -1 = "no phase supplied", which is what makes the `arg` arm test the
           # sample-0 argument path instead of inheriting e2e's own truth phase.
           "code_phase_at_ref_chips": -1.0}
    row.update(extra)
    open(p, "w").write(json.dumps([row]) + "\n")
    out = subprocess.run([os.path.join(K, "scripts/gnss/e2e"), "--prn", str(a.prn),
                          "--skip-search", "--seed-file", p] + a.e2e_arg,
                         check=True, capture_output=True, text=True).stdout
    m = re.search(r"VERDICT: worst \|error\| over \d+ records = ([\d.]+) chips", out)
    errs = [float(x) for x in re.findall(r"^\s+\d+\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)",
                                         out, re.M)]
    return (float(m.group(1)) if m else float("nan")), errs

worst = {}
for tag, extra in (("arg", {}),
                   ("phase", {"code_phase_at_ref_chips": phase_first}),
                   ("phase+", {"code_phase_at_ref_chips":
                               (phase_first + hop_off) % MOD})):
    w, errs = run(tag, extra)
    worst[tag] = w
    print("  %-7s worst |err| %9.3f chips   per-record %s"
          % (tag, w, " ".join("%+.3f" % e for e in errs)))

d_naive = abs(worst["phase"] - worst["arg"])
d_fixed = abs(worst["phase+"] - worst["arg"])
print()
print("  naive port disagrees with production by %.3f chips" % d_naive)
print("  corrected  disagrees with production by %.3f chips" % d_fixed)
print("  VERDICT: %s" % ("PASS -- the hop correction is the whole difference"
                         if d_fixed < 0.01 and d_naive > 1.0 else
                         "INVESTIGATE -- the offset is not a clean hop"))
