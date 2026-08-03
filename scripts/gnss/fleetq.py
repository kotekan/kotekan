#!/usr/bin/env python3
"""Print the fleet-combined q = 2P/(E+L) per PRN, using the broker's OWN fleet_dll().

The acceptance test for the shared DLL (docs/CHORD_GNSS_SHARED_DLL.md section 9) is q before
and after a change, so before/after have to be the SAME measurement -- hence importing the
shipped function rather than re-deriving the sum here. A probe that restates the arithmetic
tests the author's understanding of it, which is the thing already known to be unreliable.

q is a peak-SHARPNESS metric: exactly 1.0 when there is no peak (all three taps see equal noise
power), 4.0 for a clean lock at 0.5-chip spacing. Summing instances does NOT raise it -- it
shrinks its spread -- so read the floor, not the absolute value (section 8).

usage:  fleetq.py [label]
"""
import importlib.util
import sys
import time

K = "/home/kvand/gnss/kotekan"
spec = importlib.util.spec_from_file_location(
    "b", K + "/python/scripts/gnss/gps_distributed_broker.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b._log_rl = lambda k, m: None  # the probe is not the broker; do not narrate its polling

# cx19/cx51 were regenerated with --combine-gpus (ONE merged 14-channel combiner); the other
# six still expose two 7-channel ones. See STATE 8.16.4.
EPS = (["http://cx19:12049/gnss0_combine", "http://cx51:12049/gnss0_combine"]
       + ["http://%s:12049/gnss%d_combine" % (n, g)
          for n in ("cx27", "cx42", "cx43", "cx44", "cx47", "cx52") for g in (0, 1)])

label = sys.argv[1] if len(sys.argv) > 1 else ""
out = b.fleet_dll(EPS, 97656, 2, 3.0, 2.2)  # 0.5 s hop window, >=2 instances, 3 sigma
if not out:
    print("%s %s  no fleet rows" % (time.strftime("%H:%M:%S"), label))
    raise SystemExit
fl = next(iter(out.values()))
print("%s %s  floor %.3f (noise median %.3f, sigma %.4f)  %d PRNs, %d instances, %d chan"
      % (time.strftime("%H:%M:%S"), label, fl["q_floor"], fl["q_med"], fl["q_sigma"],
         len(out), fl["n_src"], fl["n_chan"]))
for prn, v in sorted(out.items(), key=lambda kv: -kv[1]["q"]):
    print("   P%-3d q %.3f  disc %+.4f%s"
          % (prn, v["q"], v["disc"], "   <== ABOVE FLOOR" if v["q"] >= v["q_floor"] else ""))
