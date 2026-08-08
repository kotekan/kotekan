"""FleetPublisher: the broker's own REST surface (GET /get_status).

Extracted verbatim from gps_distributed_broker.py (task #27 M1).

⚠️ THIS IS THE ONE PIECE M6 MUST CHANGE SHAPE. Today one publisher serves one chain on one
port, which is why CHORD needs a viewer instance per constellation (12060 GPS, 12061 E5a).
The unified broker publishes every chain on one port, keyed by chain id -- see
docs/CHORD_BROKER_REFACTOR.md M7. Moved unchanged here so that change is isolated.
"""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .transport import _now


class FleetPublisher:
    """Serve the broker's FLEET-MERGED per-PRN state over REST, in a combiner's schema.

    WHY THE BROKER. The broker is already the shared-knowledge node -- it fuses the pooled (l-a)
    code rate, the clock-frequency bias (with cross-band sibling sharing), the fused LO, the
    cross-band Doppler assist. Merging a track across frequency subbands is the same kind of
    object, and fleet_dll() already computes it every cycle. The only thing missing was
    publication.

    WHY NOT LET THE VIEWER DO IT. The viewer's polling is browser-side: livebeam_server hands
    the page a rest_port and the JS fetches kotekan directly (there is even a comment in it
    about cross-origin failures from doing exactly that). On the airspy prototype that is one
    origin. On CHORD it would be FOURTEEN origins across eight hosts, and each would show only
    that instance's 6.7% of the L5 lobe. The merge has to happen upstream of the browser, and
    upstream of the browser is here.

    SCHEMA. Rows carry the field names GnssCoherentCombiner::get_status uses, so the viewer's
    signal_metrics() consumes them unchanged -- amplitude, coh_amplitude, deep_amplitude,
    unbiased_amplitude, doppler_hz, coherence_s, deep_snr, deep_records, amp_snr, deep_floor,
    peel_*. What each MEANS is chosen honestly rather than uniformly:

      * MERGED across the fleet (this is the added value): dll_disc and the E/P/L powers, and
        amp_snr / amplitude derived from the summed prompt power against the live noise
        population -- full 20.46 MHz rather than one node's 1.37 MHz.
      * BEST-OF a single instance: every COHERENT statistic (deep_amplitude, deep_snr,
        coherence_s, peel_*). These need cross-node phase alignment to merge, which is the very
        thing the power combine avoids; claiming a fleet number for them would be a lie. The
        source node ships as `coh_src` so a reader can see whose view it is.
      * BROKER-OWNED: doppler_hz, code_phase_chips, code_phase_rate, dll_trim -- the shared
        model, which no single combiner knows.

    Read-only, no side effects, and entirely optional: without --publish-port nothing starts.
    """

    def __init__(self, port, log):
        self._rows, self._meta, self._dets, self._lock = [], {}, [], threading.Lock()
        # RUNTIME CONTROL, deliberately narrow. Everything else here is read-only; this one
        # value is writable because the experiment that needs it CANNOT be run any other way.
        # Measuring the carrier loop's open-loop transfer function means holding a fixed
        # carrier_trim_hz and watching deep_rate_hz -- but deep_rate_hz is measured against the
        # tracker's f_ref, and changing the trim via --carrier-trim-const requires a broker
        # restart, whose first seed list drops PRNs (the tracker's `active` fill is
        # authoritative, so a dropped PRN sets f_ref = NaN and re-acquires). The step and the
        # reference change are then inseparable, which is exactly how the 2026-08-04 attempt
        # came out uninterpretable. Setting it in a LIVE broker holds f_ref still.
        self._ctl = {"carrier_trim_const": None}
        pub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # a browser polls this at 1 Hz; the broker's own log stays readable

            def _cors(self):
                # ON EVERY RESPONSE, INCLUDING ERRORS. A 404 or a 501 without these headers
                # reaches the browser as a CORS failure, not as the status it actually is --
                # so the console says "blocked by CORS policy" when the truth is "no such
                # endpoint here". That misdirection cost real time on 2026-08-08: the viewer
                # was asking this port for prototype stage names (gal_search, gal_combiner,
                # airspy_in) and the reported symptom was CORS rather than 404.
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self):
                # WITHOUT THIS, BaseHTTPRequestHandler answers 501 with no CORS headers, and
                # every preflighted request dies. A POST of application/json IS preflighted,
                # so /set_carrier_trim was unreachable from a browser entirely.
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                # The viewer builds every URL as <base>/<stage>/<endpoint> from ONE host:port,
                # so it cannot straddle the search (12050) and this publisher. Serving the raw
                # detections here as well makes the broker a single origin for both -- which is
                # also the right shape: it already merges across all 14 combiners, and a browser
                # cannot poll 14 origins itself.
                with pub._lock:
                    p = self.path.rstrip("/")
                    if p.endswith("get_detections"):
                        body = json.dumps(pub._dets).encode()
                    elif p.endswith("get_status"):
                        body = json.dumps(pub._rows).encode()
                    else:
                        body = json.dumps(pub._meta).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                # The viewer is served from a different origin than this port, and its whole
                # job is to fetch from here -- so say so explicitly rather than leaving the
                # browser to fail a preflight with nothing in the log.
                self._cors()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                # ONLY /set_carrier_trim. Body {"hz": <float>} holds that trim on every seeded
                # PRN; {"hz": null} releases it back to --carrier-trim-const. Diagnostic: pair
                # it with --carrier-gain 0 so the loop does not immediately correct the step away.
                p = self.path.rstrip("/")
                if not p.endswith("set_carrier_trim"):
                    self.send_response(404)
                    self._cors()
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(n) or b"{}")
                    hz = req.get("hz")
                    hz = None if hz is None else float(hz)
                except Exception as e:
                    body = json.dumps({"error": str(e)}).encode()
                    self.send_response(400)
                    self._cors()
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                with pub._lock:
                    pub._ctl["carrier_trim_const"] = hz
                pub._log("carrier trim const set to %s by REST (diagnostic)"
                         % ("released" if hz is None else "%+.3f Hz" % hz))
                body = json.dumps({"carrier_trim_const": hz}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._log = log
        self._srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        log("fleet publisher on :%d (GET /get_status -- fleet-merged per-PRN state; "
            "POST /set_carrier_trim {\"hz\": x} -- diagnostic open-loop trim)" % port)

    def carrier_trim_const(self, fallback):
        """The REST override if one has been posted, else the command-line value."""
        with self._lock:
            v = self._ctl["carrier_trim_const"]
        return fallback if v is None else v

    def update(self, fleet, seeds, dll_trim, n_endpoints, dets=None, fcoh=None):
        rows = []
        fcoh = fcoh or {}
        for prn, v in sorted(fleet.items()):
            c = v.get("coh_row") or {}
            sd = seeds.get(prn, {})
            # Fleet incoherent amplitude/significance from the SUMMED prompt power, referenced
            # to the live noise median -- the same population the gate is built on, so the
            # number in the viewer and the number the loop gates on cannot drift apart.
            p_med = v.get("p_med") or 0.0
            ratio = (v["p_pow"] / p_med) if p_med > 0 else 0.0
            row = dict(c)                      # start from the best instance's row...
            row.update({                       # ...then override what the fleet knows better
                "prn": prn,
                "amp_snr": math.sqrt(max(0.0, ratio - 1.0)) if ratio > 0 else 0.0,
                "amplitude": math.sqrt(max(0.0, v["p_pow"])),
                "unbiased_amplitude": math.sqrt(max(0.0, v["p_pow"] - p_med)),
                "dll_disc": v["disc"],
                "doppler_hz": sd.get("doppler_hz", c.get("doppler_hz", 0.0)),
                "code_phase_chips": sd.get("code_phase_chips", c.get("code_phase_chips", 0.0)),
                # fleet-only extras: not in the combiner schema, ignored by older consumers
                "fleet_q": v["q"], "fleet_q_floor": v["q_floor"],
                "fleet_p_over_noise": ratio, "fleet_present": bool(v["present"]),
                "fleet_instances": v["n_src"], "fleet_channels": v["n_chan"],
                "fleet_hop": v["hop"], "coh_src": v.get("coh_src"),
                "code_phase_rate": sd.get("code_phase_rate", 0.0),
                # The SECOND-ORDER carrier term. propagate_seed turns this into the quadratic
                # CODE term (quad = 0.5*(chip/f_c)*dop_rate*dt^2), which is what holds the phase
                # while the Doppler accelerates -- maximal near zenith, i.e. exactly where the
                # signal is strongest. Published so its ABSENCE is visible: a seed that omits it
                # walks the code several chips per seed interval and no loop can hold that.
                "doppler_rate_hz_s": sd.get("doppler_rate_hz_s"),
                "dll_trim": dll_trim.get(prn, 0.0),
            })
            # FLEET-COHERENT OVERRIDE. This is what overturns the "BEST-OF a single instance"
            # rule in the class docstring above: the coherent statistics ARE mergeable now,
            # because fleet_coherent solves the cross-node phase alignment that the power
            # combine deliberately avoids. Only for PRNs that cleared the MEASURED null floor;
            # everything else keeps the best single instance's numbers, which stay honest.
            fc = fcoh.get(prn)
            if fc and fc.get("present"):
                row.update({
                    "deep_snr": fc["deep_snr"],
                    "deep_amplitude": fc["deep_amplitude"],
                    "coh_frac": fc["coh_frac"],
                    "coh_src": "fleet:%d" % fc["n_src"],
                    "fleet_coh_floor": fc["floor"],
                    "fleet_coh_align": fc["align"],
                    "fleet_coh_records": fc["n_rec"],
                    # The best single instance kept alongside, so the GAIN this buys is
                    # visible in the same row rather than inferred across restarts.
                    "fleet_coh_best_inst": fc["best_inst_snr"],
                })
            elif fc:
                # Measured and rejected: publish the floor it failed against, so "no fleet
                # number" is distinguishable from "fleet never looked at this PRN".
                row.update({"fleet_coh_floor": fc["floor"],
                            "fleet_coh_align": fc["align"]})
            rows.append(row)
        with self._lock:
            self._rows = rows
            if dets is not None:
                self._dets = dets
            self._meta = {"n_prn": len(rows), "n_endpoints": n_endpoints,
                          "present": sum(1 for r in rows if r["fleet_present"]),
                          "utc": _now()}
