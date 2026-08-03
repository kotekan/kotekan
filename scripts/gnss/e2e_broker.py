#!/usr/bin/env python3
"""Put the REAL broker in the e2e loop -- the last stage still tested by proxy (STATE 7.4/2).

`scripts/gnss/e2e` closes search -> seed -> propagate_seed -> despread to 0.373 chips offline,
but it builds the seed ITSELF, the way the broker would if it had nothing to add. The broker in
production has plenty to add (nh offset, cp-rate seeding, continuity resolution, DLL trim at
POST time), and none of that arithmetic is covered by any test. On sky the trackers sit ~1-2
chips off the true peak -- inside the tracker's own grating-lobe null -- so the gap between
"detection" and "commanded phase" is exactly where to look.

WHAT THIS DOES
  1. `e2e --emit-detection` writes one detection in /get_detections wire format, for a KNOWN
     injected truth.
  2. This script serves that as a fake search stage, and stands in for the trackers and the
     combiner, capturing whatever the broker POSTs to /set_seeds.
  3. `e2e --seed-file <captured>` runs the REAL seed through the REAL propagate_seed and the
     REAL GPU despread, and prints the error in chips against the same truth.

Any difference between run 1's error and run 3's error is the broker's contribution, isolated.

usage:  e2e_broker.py [--prn N] [--port P] [--broker-arg ...]
"""
import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
import time

K = "/home/kvand/gnss/kotekan"
PY = "/home/kvand/gnss/venv/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--prn", type=int, default=3)
ap.add_argument("--port", type=int, default=12777)
ap.add_argument("--settle-s", type=float, default=25.0,
                help="how long to let the broker run before reading the captured seed")
ap.add_argument("--outdir", default="/tmp/e2e_broker")
ap.add_argument("--e2e-arg", action="append", default=[],
                help="extra argument passed to BOTH e2e legs (e.g. --e2e-arg=--s-stride "
                     "--e2e-arg=1). Use the 8-node comb for a realistic search leg.")
ap.add_argument("--broker-arg", action="append", default=[],
                help="extra argument passed to the broker (repeatable)")
a = ap.parse_args()

os.makedirs(a.outdir, exist_ok=True)
DET = os.path.join(a.outdir, "detections.json")
SEEDS = os.path.join(a.outdir, "seeds.jsonl")
for f in (DET, SEEDS):
    if os.path.exists(f):
        os.remove(f)

# ---------------------------------------------------------------------------- leg 1: detection
print("=" * 92)
print("LEG 1 -- e2e alone (the control): search -> its OWN seed -> despread")
print("=" * 92)
leg1 = subprocess.run([K + "/scripts/gnss/e2e", "--prn", str(a.prn),
                       "--emit-detection", DET] + a.e2e_arg,
                      capture_output=True, text=True, cwd=K)
sys.stdout.write(leg1.stdout)
if not os.path.exists(DET):
    sys.exit("e2e did not write a detection (%s)" % leg1.stderr[-500:])
det = json.load(open(DET))
print("detection served: %s\n" % json.dumps(det[0]))

# --------------------------------------------------------------------- the fake pipeline server
captured = []


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # the broker polls several times a second; its own log is the interesting one

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/get_detections"):
            self._send(json.load(open(DET)))
        elif self.path.endswith("/get_status"):
            self._send([])          # no combiner: no DLL trim, no drop decisions
        elif self.path.endswith("/get_trim"):
            self._send([])
        else:
            self._send([])

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode() if n else "[]"
        if self.path.endswith("/set_seeds"):
            captured.append(raw)
            with open(SEEDS, "a") as f:
                f.write(raw + "\n")
        self._send({})


srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = "http://127.0.0.1:%d" % a.port
print("fake pipeline on %s (get_detections / set_seeds / get_status)" % base)

# ------------------------------------------------------------------------------- run the broker
# Same signal geometry as production. Deliberately NOT almanac/dead-reckon/cl-assist: those need
# BRDC and a live clock, and the question here is the SEED ARITHMETIC, not the sky model. Add
# them back with --broker-arg when testing those paths specifically.
cmd = [PY, "-u", "python/scripts/gnss/gps_distributed_broker.py",
       "--rest-url", base,
       "--detectors", base + "/gps_search",
       "--trackers", base + "/gps_track",
       "--combiner", base + "/gps_combine",
       "--constellation", "G", "--carrier-hz", "1176.45e6",
       "--chip-rate-hz", "10.23e6", "--code-length", "10230",
       "--hops-per-sec", "195312.5",
       "--seed-doppler", "det", "--acquire-snr", "30",
       "--dll-gain", "0.0", "--carrier-gain", "0.0",
       "--interval", "1"] + a.broker_arg
print("broker: %s\n" % " ".join(cmd[2:]))
blog = open(os.path.join(a.outdir, "broker.log"), "w")
proc = subprocess.Popen(cmd, cwd=K, stdout=blog, stderr=subprocess.STDOUT)
t0 = time.time()
while time.time() - t0 < a.settle_s and not captured:
    time.sleep(0.5)
time.sleep(2.0)
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
blog.close()
srv.shutdown()

if not captured:
    print(open(os.path.join(a.outdir, "broker.log")).read()[-3000:])
    sys.exit("broker POSTed no seeds in %.0f s -- see %s/broker.log" % (a.settle_s, a.outdir))

seed = None
for row in json.loads(captured[-1]):
    if int(row.get("prn", -1)) == a.prn:
        seed = row
print("=" * 92)
print("THE BROKER'S SEED (%d POST(s) captured)" % len(captured))
print("=" * 92)
print(json.dumps(seed, indent=2, sort_keys=True))
d = det[0]
if seed:
    for k, dk in (("code_phase_at_ref_chips", "code_phase_at_ref_chips"),
                  ("code_phase_chips", "code_phase_chips"),
                  ("doppler_hz", "doppler_hz"), ("ref_hop", "ref_hop")):
        if k in seed and dk in d:
            try:
                delta = float(seed[k]) - float(d[dk])
            except (TypeError, ValueError):
                continue
            print("  %-26s detection %18.6f  seed %18.6f  DELTA %+14.6f%s"
                  % (k, float(d[dk]), float(seed[k]), delta,
                     "   <== broker changed it" if abs(delta) > 1e-6 else ""))

# ------------------------------------------------------------- leg 3: the real seed, real chain
print()
print("=" * 92)
print("LEG 3 -- the SAME truth, but seeded from the BROKER's payload")
print("=" * 92)
leg3 = subprocess.run([K + "/scripts/gnss/e2e", "--prn", str(a.prn),
                       "--seed-file", SEEDS] + a.e2e_arg,
                      capture_output=True, text=True, cwd=K)
sys.stdout.write(leg3.stdout)


def verdict(txt):
    for line in txt.splitlines():
        if "VERDICT" in line or "SEARCH LEG ERROR" in line:
            print("   " + line.strip())


print("=" * 92)
print("SIDE BY SIDE")
print("=" * 92)
print(" e2e's own seed:")
verdict(leg1.stdout)
print(" the broker's seed:")
verdict(leg3.stdout)
