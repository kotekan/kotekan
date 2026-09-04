#!/usr/bin/env python3
"""A synthetic fleet for the broker's equivalence gate (task #27 M0).

WHY THIS EXISTS. The gate in `broker_equiv.py` needs a transcript, and the best transcripts
come from real chains -- on-sky GPS L5, model-primary E5a, the L2C replay bench. None of
them is available with the F-engine down, which is exactly when a refactor gets done. This
serves a deterministic, self-contained fleet so a transcript can always be captured:

    broker_fakefleet.py --port 12777 &
    broker_equiv.py record /tmp/t.jsonl -- \
        --detectors http://localhost:12777/gps_search \
        --trackers http://localhost:12777/track \
        --combiner http://localhost:12777/combiner \
        --time0-endpoint telescope/time0_ns --rest-url http://localhost:12777 \
        --carrier-hz 1176.45e6 --chip-rate-hz 10.23e6 --code-length 10230 \
        --hops-per-sec 195312.5 --nh-overlay-len 20 --interval 0.05 --acquire-snr 12

It is NOT a simulator and makes no claim to be physical. It exists to drive code paths:
detections that ADVANCE (so cp_hist fills, the rate fit runs, freshness stamps move), a
status table that reports locks (so the hold/watchdog/DLL/carrier paths execute), and a
time anchor (so the CL time-assist arithmetic runs). What it must never be is STATIC -- a
frozen fleet replays perfectly and tests nothing, which is the trap
docs/CHORD_BROKER_REFACTOR.md 3.2 warns about and which cost two scans on 2026-08-08.

Everything it emits is a deterministic function of the poll index, so two recordings of the
same length are identical apart from wall-clock timing (which the transcript pins anyway).
"""
import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=12777)
ap.add_argument("--prns", default="3,19,26,32")
ap.add_argument("--hop0", type=int, default=114436200145)
ap.add_argument("--hops-per-sec", type=float, default=195312.5)
ap.add_argument("--revisit-s", type=float, default=1.0,
                help="snapshot-hop advance between successive detection sets")
ap.add_argument("--code-length", type=float, default=10230.0)
ap.add_argument("--time0", type=float, default=1786167610.000002870,
                help="the anchor served at telescope/time0_ns (CHORD's real one by default, "
                     "so the CL/dead-reckon arithmetic sees a plausible epoch)")
a = ap.parse_args()

PRNS = [int(x) for x in a.prns.split(",") if x.strip()]
_n = [0]           # detection-poll counter: the ONLY state, so output is a pure function
_lock = threading.Lock()
_seeds = [[]]      # last seed payload received, echoed into the status table


def _detections():
    """One set per poll, at an advancing ref_hop. Doppler drifts, code phase drifts with
    it (roughly the real code-Doppler ratio), so the rate fit has something to fit."""
    with _lock:
        k = _n[0]
        _n[0] += 1
    out = []
    dh = int(a.revisit_s * a.hops_per_sec)
    for i, prn in enumerate(PRNS):
        t = k * a.revisit_s
        dop = -1500.0 + 120.0 * i + 0.42 * t          # Hz, drifting like a real pass
        cp = (137.5 * (i + 1) + 0.0361 * dop * t) % a.code_length
        out.append({"prn": prn,
                    "snr": 34.0 + 3.0 * math.sin(0.7 * k + i),
                    "doppler_hz": dop,
                    "code_phase_chips": cp,
                    "ref_hop": a.hop0 + k * dh,
                    "nh": (k + i) % 20,
                    "code_phase_long_chips": cp + a.code_length * ((k + i) % 20),
                    "code_phase_at_ref_chips": cp})
    return out


def _status():
    """Combiner rows for whatever is currently seeded: locked, with a live discriminator."""
    with _lock:
        seeded = list(_seeds[0])
        k = _n[0]
    rows = []
    for i, s in enumerate(seeded):
        prn = int(s.get("prn", 0))
        # A small, changing E-L imbalance so the DLL has a real discriminator to act on,
        # and a deep that clears the floor so coherence_s > 0 certifies the lock.
        d = 0.04 * math.sin(0.31 * k + 0.9 * i)
        rows.append({"prn": prn,
                     "amplitude": 8.0 + i, "amp_snr": 21.0 + 2.0 * i,
                     "deep_snr": 60.0 + 5.0 * i, "deep_amplitude": 40.0 + i,
                     "coherence_s": 1.048576, "deep_records": 100, "deep_floor": 7.0,
                     "unbiased_amplitude": 7.5 + i, "coh_amplitude": 30.0 + i,
                     "doppler_hz": float(s.get("doppler_hz", 0.0)),
                     "deep_rate_hz": 0.31 * math.cos(0.17 * k + i), "deep_rate_q": 3.4,
                     "carrier_hz_resid": 0.5 * math.sin(0.11 * k + i),
                     "e_pow": 100.0 * (1.0 - d), "p_pow": 140.0, "l_pow": 100.0 * (1.0 + d),
                     "dll_disc": d, "records": 100, "n_chan": 7,
                     "hop": a.hop0 + k * int(a.revisit_s * a.hops_per_sec)})
    return rows


class H(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p.endswith("get_detections"):
            return self._send(_detections())
        if p.endswith("get_status"):
            return self._send(_status())
        if "time0_ns" in p:
            return self._send({"time0_ns": a.time0 * 1e9})
        return self._send([])

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n).decode()) if n else {}
        except Exception:
            body = {}
        if self.path.rstrip("/").endswith("set_seeds"):
            # The broker POSTs a bare LIST of seed dicts; tolerate a {"seeds": [...]}
            # wrapper too so this fixture survives a schema change without going silently
            # empty (an empty seed table here means an empty status table, which reads as
            # "the broker stopped seeding" -- a fixture must not be able to fake that).
            if isinstance(body, list):
                rows = body
            elif isinstance(body, dict):
                rows = body.get("seeds") or []
            else:
                rows = []
            with _lock:
                _seeds[0] = [r for r in rows if isinstance(r, dict)]
        return self._send({"ok": True})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print("fake fleet on :%d  prns=%s" % (a.port, PRNS), flush=True)
    srv.serve_forever()
