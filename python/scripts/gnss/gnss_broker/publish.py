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
import urllib.parse
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
        # ONE PORT, MANY CHAINS (task #27 M6). Each registered chain gets its own row/det
        # store; a request selects one chain or gets them all, tagged. Before this, one
        # publisher served one chain on one port, which is why CHORD needed a viewer
        # instance per constellation (12060 GPS, 12061 E5a).
        self._chains = {}          # chain id -> {"rows", "dets", "meta", "ctl", "sig", "band"}
        self._order = []           # registration order, so output is deterministic
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
                #
                # CHAIN SELECTION, two ways, because the consumers differ:
                #   /get_status?chain=gps_l5   explicit, for anything we write
                #   /gps_l5/get_status         a PATH SEGMENT naming a chain -- which is what
                #                              the viewer already emits, since it builds
                #                              <base>/<stage>/<endpoint> and cannot add a
                #                              query string. Point its --gps-combiner-stage at
                #                              a chain id and it filters with no code change.
                # Neither given -> every chain, each row tagged. With ONE chain registered
                # that is byte-identical to the old single-chain output plus the tag, so
                # nothing that exists today has to change at once.
                raw = self.path.split("?", 1)
                p = raw[0].rstrip("/")
                q = urllib.parse.parse_qs(raw[1]) if len(raw) > 1 else {}
                want = (q.get("chain") or [None])[0]
                segs = [s for s in p.strip("/").split("/") if s]
                # ⚠️ AN UNKNOWN CHAIN MUST NOT FALL THROUGH TO THE MERGED TABLE. The old code
                # was `ids = [want] if want in _chains else list(_order)`, which conflated two
                # completely different requests: "no selector -- give me everything" (correct)
                # and "give me chain X" where X is not running (must not silently answer with
                # every OTHER chain's rows). The viewer keys one column per chain and fetches
                # <chain>/get_status trusting it to be filtered (livebeam_server's
                # discover_broker_chains sets combiner = the chain id, and gps_feed.js hangs
                # metrics off that key), so the fall-through populated a column with another
                # signal's satellites.
                #
                # MEASURED 2026-08-09, and it is why this is a 404 and not a tidy-up: with the
                # two 1207.14 MHz chains removed from the config, /gal_e5b/get_status,
                # /bds_b2b/get_status, /nonsense_chain/get_status and /zzz/get_status all
                # returned the SAME 28 rows -- byte-identical, the concatenation of gps_l5 (10)
                # + gal_e5a (9) + bds_b2a (9), PRNs 8 and 24 appearing twice. On the viewer that
                # read as "E5b and B2b are tracking nicely, with almost uniformly identical
                # significance to their E5a/B2a partners". They were their E5a/B2a partners.
                # Identical numbers from two independent measurements is never the good news it
                # looks like.
                #
                # 404 rather than an empty list, because gps_feed.js already documents "a 404 on
                # a missing gal_/bds_ stage just skips that chain" -- so the column goes away
                # instead of going quietly blank, and curl says which chains DO exist.
                _EPS = ("get_status", "get_detections", "get_chains")
                asked = want
                if asked is None and len(segs) >= 2 and segs[-1] in _EPS:
                    asked = segs[-2]
                unknown = None
                with pub._lock:
                    if (asked is not None and asked not in pub._chains
                            and not p.endswith("get_chains")):
                        unknown = (asked, list(pub._order))
                    ids = [asked] if asked in pub._chains else list(pub._order)
                    if p.endswith("get_chains"):
                        # DISCOVERY. The viewer's own discover_signals() reads kotekan's
                        # /config, which a broker publisher does not serve -- so it fell
                        # back to a static PROTOTYPE table and asked for stage names that
                        # have never existed on CHORD (gal_search, gal_combiner,
                        # airspy_in). The broker already knows every chain it runs, its
                        # constellation, band and record length; publishing that lets ONE
                        # viewer instance build its table from what is actually running.
                        body = json.dumps([pub._chains[c]["desc"]
                                           for c in pub._order]).encode()
                    elif p.endswith("get_detections"):
                        body = json.dumps(pub._collect(ids, "dets")).encode()
                    elif p.endswith("get_status"):
                        body = json.dumps(pub._collect(ids, "rows")).encode()
                    else:
                        body = json.dumps(pub._collect_meta(ids)).encode()
                if unknown is not None:
                    body = json.dumps({"error": "unknown chain",
                                       "asked": unknown[0],
                                       "chains": unknown[1]}).encode()
                self.send_response(404 if unknown is not None else 200)
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
                raw = self.path.split("?", 1)
                p = raw[0].rstrip("/")
                q = urllib.parse.parse_qs(raw[1]) if len(raw) > 1 else {}
                sel = (q.get("chain") or [None])[0]
                if sel is None:
                    for seg in p.strip("/").split("/"):
                        if seg in pub._chains:
                            sel = seg
                            break
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
                # A trim is a PER-CHAIN experiment. With one chain registered, name it
                # implicitly; with several, an unqualified POST would silently step every
                # signal at once, so require ?chain= and say so.
                with pub._lock:
                    targets = [sel] if sel in pub._chains else list(pub._order)
                    if len(targets) > 1:
                        body = json.dumps({"error": "several chains registered (%s); name "
                                                    "one with ?chain=<id>"
                                                    % ", ".join(targets)}).encode()
                        targets = []
                    else:
                        for c in targets:
                            pub._chains[c]["ctl"]["carrier_trim_const"] = hz
                        pub._ctl["carrier_trim_const"] = hz
                        body = json.dumps({"carrier_trim_const": hz,
                                           "chain": targets[0] if targets else None}).encode()
                if not targets:
                    self.send_response(400)
                    self._cors()
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                pub._log("carrier trim const set to %s by REST on %s (diagnostic)"
                         % ("released" if hz is None else "%+.3f Hz" % hz, targets[0]))
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

    # -- multi-chain plumbing (task #27 M6) ---------------------------------------------
    def register(self, chain, signal=None, band=None, meta=None):
        """Claim a slot on this port and get a handle that looks like the old publisher.

        Returns a per-chain VIEW rather than self, so every existing call site --
        `publisher.update(...)`, `publisher.carrier_trim_const(...)` -- keeps working
        unchanged whether it is the only chain or one of five."""
        with self._lock:
            if chain not in self._chains:
                self._chains[chain] = {"rows": [], "dets": [], "meta": {},
                                       "ctl": {"carrier_trim_const": None},
                                       "sig": signal, "band": band,
                                       "desc": dict(meta or {}, chain=chain,
                                                    signal=signal, band=band)}
                self._order.append(chain)
        return _ChainView(self, chain)

    def _collect(self, ids, key):
        """Rows/detections for `ids`, each tagged with which chain it came from.

        The tag is ADDED, never substituted: the schema stays GnssCoherentCombiner's, so
        the viewer's signal_metrics() consumes these unchanged and simply ignores three
        extra keys."""
        out = []
        for c in ids:
            st = self._chains.get(c)
            if not st:
                continue
            for r in st[key]:
                if isinstance(r, dict):
                    r = dict(r)
                    r["chain"], r["signal"], r["band"] = c, st["sig"], st["band"]
                out.append(r)
        return out

    def _collect_meta(self, ids):
        if len(ids) == 1 and ids[0] in self._chains:
            return self._chains[ids[0]]["meta"]
        return {"chains": {c: self._chains[c]["meta"] for c in ids if c in self._chains}}

    def carrier_trim_const(self, fallback, chain=None):
        """The REST override if one has been posted, else the command-line value."""
        with self._lock:
            st = self._chains.get(chain) if chain else None
            v = st["ctl"]["carrier_trim_const"] if st else self._ctl["carrier_trim_const"]
        return fallback if v is None else v

    def update(self, fleet, seeds, dll_trim, n_endpoints, dets=None, fcoh=None, chain=None):
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
                # ⚠️ `doppler_hz` IS THE COMMANDED SEED, NOT WHAT THE CORRELATOR RAN AT.
                # These two differ by the tracker's feed-forward, dop_rate x (seed age), and
                # the seed changes only when the broker reseeds -- so this field is a STAIRCASE
                # even while the tracker is following the sky perfectly smoothly.
                #
                # THAT COST A WHOLE MISDIAGNOSIS (2026-08-09). The viewer plots this as
                # "tracked Doppler"; it froze for 12 minutes at a time, and I read it as the
                # tracker failing to apply dop_rate -- reporting "75 Hz un-applied" that was
                # really just the seed being stale. Polling the combiner directly showed the
                # applied Doppler advancing smoothly at the model rate the whole time.
                #
                # So publish BOTH, and never make anyone infer the tracker's state from the
                # broker's own command again. `c` is the best instance's combiner row, whose
                # doppler_hz is record slot REC_DOPPLER = PrnCtl.fcar_report = the propagated
                # value the despread actually used.
                "doppler_hz": sd.get("doppler_hz", c.get("doppler_hz", 0.0)),
                "doppler_applied_hz": c.get("doppler_hz"),
                # Fleet phase-slope delay fit (task #32): tau of the correlation peak
                # RELATIVE to the replica placement, from the cross-channel phase ramp.
                # Measurement-only today -- published next to the disc it is intended to
                # replace, so the two can be judged against each other on sky.
                "spec_tau_chips": v.get("spec_tau"),
                "spec_peak_ratio": v.get("spec_ratio"),
                # The b_sat actually being APPLIED to this sat's seeds (0 = none/stale) --
                # published so the closed loop's health is checkable from outside: b_sat
                # should hold near the P1 open-loop means while spec_tau collapses to 0.
                "bsat_chips": v.get("bsat"),
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
            else:
                if fc:
                    # Measured and rejected: publish the floor it failed against, so "no
                    # fleet number" is distinguishable from "fleet never looked at this PRN".
                    row.update({"fleet_coh_floor": fc["floor"],
                                "fleet_coh_align": fc["align"]})
                # QUADRATURE FALLBACK (2026-08-10, docs 11.31). The argmax this replaces sat
                # a measured 4.9 dB below the fleet value, so every fleet-gate flicker
                # stepped the published series 5-8 dB -- and at the observed 20-70% duty
                # that mixture alone contributed ~3.5 dB of C/N0 scatter. The quadrature
                # lands at the fleet level (-0.7..+2.0 dB over 5 strong PRNs), making the
                # series CONTINUOUS across the gate. It is not quieter within-state than
                # the argmax (no scalar is; the engagement rate is the real fix, #10) --
                # continuity is what it buys. The argmax stays published alongside so the
                # A/B lives in the row, and coh_src says which population produced the
                # number ('quad:N' vs a URL).
                _q = v.get("coh_quad")
                row["coh_best_inst_snr"] = c.get("deep_snr")
                if _q is not None:
                    row.update({"deep_snr": _q[0],
                                "coh_src": "quad:%d" % _q[1]})
            # C/N0, PUBLISHED RATHER THAN DERIVED (task #35, 2026-08-11). The viewer used
            # to compute 20log10(deep_snr) - 10log10(coherence_s), which is only valid
            # when deep_snr is ONE instance's amplitude SNR. It is not: the fleet override
            # and the quadrature fallback serve two estimators with different
            # normalisations into that same field (quad inflates by exactly 10log10(N) =
            # 10.8 dB at N=12; measured +8.72 dB vs the best instance against the fleet's
            # +0.79), so every fleet<->quad flip stepped the displayed value ~8 dB and at
            # ~50% duty that mixture WAS the "C/N0 scatter" (docs 11.31, the 2026-08-10
            # G32-vs-E32 investigation).
            #
            # Definition: the BEST SINGLE INSTANCE's deep SNR over its own coherent span.
            # One estimator, one normalisation, continuous across serving states BY
            # CONSTRUCTION -- it never switches population. The fleet's coherent gain
            # stays visible in deep_snr + coh_src, which remain untouched: deep_snr is a
            # detection SIGNIFICANCE and gates consume it; this field is the radiometry.
            # ⚠️ THE NUMERATOR AND THE DENOMINATOR MUST DESCRIBE THE SAME INTEGRATION.
            # This is a C/N0, i.e. amplitude-SNR^2 per unit time, so pairing an SNR
            # measured over N records with a span measured over M != N records is not an
            # approximation -- it is a different quantity, and it JUMPS whenever either
            # count changes.
            #
            # MEASURED 2026-08-11 on PRN 21: `coherence_s` is the best INSTANCE's ladder
            # pick and hops 12 / 25 / 50 / 100 records poll to poll, while
            # `fleet_coh_best_inst` is integrated over `fleet_coh_records` (42, 52, ...).
            # The four rungs put -10log10(T) at -0.21 / +2.81 / +5.81 / +9.00 dB, so the
            # pairing alone injects up to 9.2 dB of DISCRETE steps -- observed 41.6 ->
            # 33.6 -> 22.8 -> 40.5 dB-Hz within 30 s on an unchanging satellite. That
            # bimodal "switch" is the last of the C/N0 scatter (KV spotted the shape:
            # discrete transitions, not a continuous multiplicative scaling, which is what
            # sent this hunt after gain-like mechanisms for a day).
            #
            # So take each estimator WITH ITS OWN span: the fleet value over the records
            # the fleet actually summed, the instance value over the instance's own ladder
            # pick. t_rec is derived from the instance row rather than hardcoded, so a
            # different record geometry cannot silently desynchronise this.
            _t_rec = None
            if c.get("coherence_s") and c.get("deep_records"):
                _t_rec = float(c["coherence_s"]) / float(c["deep_records"])
            if fc and fc.get("present") and fc.get("n_rec") and _t_rec:
                _bi, _T = fc["best_inst_snr"], fc["n_rec"] * _t_rec
            else:
                _bi, _T = c.get("deep_snr"), float(c.get("coherence_s") or 0.0)
            row["cn0_coh_db"] = (20.0 * math.log10(_bi) - 10.0 * math.log10(_T)
                                 if (_bi and _bi > 0.0 and _T > 0.0) else None)
            rows.append(row)
        meta = {"n_prn": len(rows), "n_endpoints": n_endpoints,
                "present": sum(1 for r in rows if r["fleet_present"]),
                "utc": _now()}
        with self._lock:
            st = self._chains.get(chain)
            if st is not None:
                st["rows"] = rows
                if dets is not None:
                    st["dets"] = dets
                st["meta"] = meta
            self._rows = rows          # legacy single-chain mirror
            if dets is not None:
                self._dets = dets
            self._meta = meta


class _ChainView(object):
    """One chain's handle on a shared publisher.

    Exists so every call site keeps the pre-M6 shape -- `publisher.update(...)`,
    `publisher.carrier_trim_const(...)` -- whether the process runs one chain or five. The
    alternative was threading a chain id through the cycle body, which is the 3252 lines
    this refactor has deliberately not touched."""

    __slots__ = ("_pub", "_chain")

    def __init__(self, pub, chain):
        self._pub, self._chain = pub, chain

    def update(self, *a, **kw):
        kw["chain"] = self._chain
        return self._pub.update(*a, **kw)

    def carrier_trim_const(self, fallback):
        return self._pub.carrier_trim_const(fallback, chain=self._chain)
