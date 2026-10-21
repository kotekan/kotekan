"""FleetPublisher: the broker's own REST surface (GET /get_status and the other endpoints).

One publisher serves every chain on one port, keyed by chain id. scripts/gnss/broker_multi.py
starts it on the first --publish-port and each chain claims a slot (register); a chain run
alone binds its own. A request names a chain as /<chain>/get_status or /get_status?chain=<id>.
"""
import collections
import json
import math
import time
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .transport import _now
from . import timebase

# Prompt-intensity Rayleigh bar for prompt_rayleigh / prompt_lock. A prompt tap on noise has a
# Rayleigh amplitude, so s4_raw ~= 1; a tap holding a coherent signal sits well below 1. 0.90 sits
# in the empty gap between the blind population (~1.0) and the tracking one (below ~0.8):
# re-derive it rather than nudge it if that split stops being bimodal.
PROMPT_RAYLEIGH_S4 = 0.90

# Trailing window (s) for det_duty_s and the windowed median behind `sig`: long enough to
# average over many cycles of the deep stage's flicker rather than alias against one. It is the
# averaging time of a measurement of how often the deep stage holds a satellite, not a
# smoothing constant to tune for looks.
DUTY_WIN_S = 120.0


class FleetPublisher:
    """Serve the broker's fleet-merged per-PRN state over REST, in a combiner's schema.

    The merge belongs in the broker: it already pools state across instances and bands (the
    (l-a) code-rate clock and the receiver clock, shared between sibling chains), and
    fleet_dll computes the merged track every cycle. The viewer polls from the browser, where
    every tracker instance would be its own origin holding only its share of the band (one
    instance correlates 6.7% of the L5 lobe), so the merge has to happen upstream of it.

    Rows carry the field names GnssCoherentCombiner::get_status uses, so the viewer's
    signal_metrics() (gps_feed.js) consumes them unchanged. What each field means depends on
    what can be merged:

      * Merged across the fleet: dll_disc; amplitude from the fleet's prompt power (the whole
        lobe rather than one instance's channels) and unbiased_amplitude / amp_snr from it
        against the probes' noise median; the fleet_* fields. The row's own e_pow / p_pow /
        l_pow stay the best instance's.
      * Coherent statistics (deep_amplitude, deep_snr, coherence_s) come from the best single
        instance, since merging them needs the cross-node phase alignment the power combine
        avoids. When fleet_coherent clears its measured null floor its deep_snr /
        deep_amplitude / coh_frac replace them; otherwise the quadrature of the
        floor-cleared instances' deep_snr may replace deep_snr. `coh_src` names the source:
        'fleet:N', 'quad:N', or one instance's URL.
      * Broker-owned: doppler_hz, code_phase_chips, code_phase_rate, dll_trim -- the shared
        model, which no single combiner knows.

    Read-only, no side effects, and optional: without --publish-port nothing starts.
    """

    def __init__(self, port, log):
        # The interface watcher (iface_watch.Watcher), set by broker_multi: GET /get_iface serves
        # its verdicts and /get_chains adds each chain's summary. None in a single-chain broker.
        self.iface = None
        self._rf = {"t": None, "instances": {}}  # receiver-wide RF health (set_rf)
        self._dark = {}  # chain -> {"t", "instances": {tap url: verdict}} (set_dark)
        # Receiver-wide sky, so the viewer draws what the broker believes instead of
        # re-deriving it. Keyed by constellation because each chain contributes its own and
        # they must not overwrite each other.
        self._sky = {}  # sysc -> {"t": float, "sats": {prn: (el, az)}}
        # One port, many chains: each registered chain has its own store ({"rows", "dets",
        # "meta", "elem", "sig", "band", "desc"}, see register), and a request selects one
        # chain or gets them all, tagged.
        self._chains = {}  # chain id -> its store
        self._order = []  # registration order, so output is deterministic
        self._lock = threading.Lock()
        # Trailing detection history, (chain, prn) -> deque[(t, inst_snr_med, cleared floor)].
        # Every instantaneous deep statistic in a row flickers, so one poll samples a
        # flickering quantity; the trailing duty (det_duty_s) measures the flicker instead.
        self._hist = {}
        pub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # a browser polls this continuously; keep the broker's own log readable

            def _cors(self):
                # On every response do_GET sends, the 404 included: an error without these
                # headers reaches the browser as a CORS failure rather than as its status, which
                # hides which endpoint is missing.
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_GET(self):
                # The viewer builds every URL as <base>/<stage>/<endpoint> from one host:port,
                # so it cannot straddle the search stage and this publisher; the search
                # detections are served here too (get_detections), making the broker a single
                # origin for both.
                #
                # Chain selection, two ways, because the consumers differ:
                #   /get_status?chain=gps_l5   explicit, for scripts
                #   /gps_l5/get_status         a path segment naming the chain, which is what
                #                              the viewer emits: it builds
                #                              <base>/<stage>/<endpoint> and cannot add a
                #                              query string.
                # Neither given -> every chain, each row tagged with its chain.
                raw = self.path.split("?", 1)
                p = raw[0].rstrip("/")
                q = urllib.parse.parse_qs(raw[1]) if len(raw) > 1 else {}
                want = (q.get("chain") or [None])[0]
                segs = [s for s in p.strip("/").split("/") if s]
                # An unknown chain must not fall through to the merged table. "No selector"
                # (every chain) and "chain X" where X is not running are different requests: the
                # viewer keys one column per chain and trusts <chain>/get_status to be filtered
                # (livebeam_server's discover_broker_chains names each stage by chain id), so a
                # fall-through would fill X's column with the other chains' satellites, numbers
                # indistinguishable from real tracking. It answers 404 with the running chains
                # rather than an empty list, so the request reads as wrong instead of as a chain
                # with nothing tracked, and curl shows which chains exist.
                _EPS = ("get_status", "get_detections", "get_chains", "get_elements")
                asked = want
                if asked is None and len(segs) >= 2 and segs[-1] in _EPS:
                    asked = segs[-2]
                unknown = None
                with pub._lock:
                    if (
                        asked is not None
                        and asked not in pub._chains
                        and not p.endswith("get_chains")
                    ):
                        unknown = (asked, list(pub._order))
                    ids = [asked] if asked in pub._chains else list(pub._order)
                    if p.endswith("get_chains"):
                        # Discovery: each registered chain's descriptor (constellation,
                        # band, record length, ...; see register), so one viewer instance
                        # builds its table from what is actually running.
                        _w = pub.iface
                        body = json.dumps(
                            [
                                pub._chains[c]["desc"]
                                if _w is None
                                else dict(
                                    pub._chains[c]["desc"], iface=_w.chain_summary(c)
                                )
                                for c in pub._order
                            ]
                        ).encode()
                    elif p.endswith("get_iface"):
                        # Receiver-wide: the interface watcher's verdict on every origin
                        # (iface_watch.py), whatever the chain selector.
                        _w = pub.iface
                        body = json.dumps(
                            _w.snapshot()
                            if _w is not None
                            else {"mode": None, "error": "no interface watcher here"}
                        ).encode()
                    elif p.endswith("get_detections"):
                        body = json.dumps(pub._collect(ids, "dets")).encode()
                    elif p.endswith("get_elements"):
                        # One chain asked -> its table directly (the viewer's shape); no
                        # selector -> {chain: table}, so the merged form can never be
                        # mistaken for a single chain's (see the unknown-chain note above).
                        _et = {c: pub._chains[c].get("elem") or {} for c in ids}
                        body = json.dumps(
                            _et[ids[0]]
                            if (asked in pub._chains and len(ids) == 1)
                            else _et
                        ).encode()
                    elif p.endswith("get_sky"):
                        # Receiver-wide like get_rf: the sky is not a per-chain quantity, so
                        # the body ignores the chain selector. Shape:
                        # [{"const": "G", "prn": 3, "el": .., "az": .., "age_s": ..}], flat
                        # because the consumer draws markers; age_s makes a frozen chain read
                        # as stale rather than as a satellite standing still.
                        _now_s = time.time()
                        _sky = []
                        for _sc, _d in sorted(pub._sky.items()):
                            _age = (_now_s - _d["t"]) if _d.get("t") else None
                            for _p, (_el, _az) in sorted(_d["sats"].items()):
                                _sky.append(
                                    {
                                        "const": _sc,
                                        "prn": _p,
                                        "el": _el,
                                        "az": _az,
                                        "age_s": (
                                            round(_age, 1) if _age is not None else None
                                        ),
                                    }
                                )
                        body = json.dumps(_sky).encode()
                    elif p.endswith("get_rf"):
                        # Receiver-wide, so the body ignores the chain selector. `dark` holds
                        # each chain's dark instances, keyed by the tap URLs of `instances`.
                        body = json.dumps(dict(pub._rf, dark=pub._dark)).encode()
                    elif p.endswith("get_status"):
                        body = json.dumps(pub._collect(ids, "rows")).encode()
                    else:
                        body = json.dumps(pub._collect_meta(ids)).encode()
                if unknown is not None:
                    body = json.dumps(
                        {
                            "error": "unknown chain",
                            "asked": unknown[0],
                            "chains": unknown[1],
                        }
                    ).encode()
                self.send_response(404 if unknown is not None else 200)
                self.send_header("Content-Type", "application/json")
                # The viewer is served from a different origin than this port, so every
                # response carries the CORS headers.
                self._cors()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._log = log
        self._srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        log(
            "fleet publisher on :%d (GET /get_status -- fleet-merged per-PRN state)"
            % port
        )

    # -- multi-chain plumbing -----------------------------------------------------------
    def register(self, chain, signal=None, band=None, meta=None):
        """Claim a slot on this port for `chain`; returns its _ChainView.

        The view carries the chain id, so call sites -- `publisher.update(...)`,
        `publisher.set_sky(...)` -- are the same whether the process runs one chain or
        several."""
        with self._lock:
            if chain not in self._chains:
                self._chains[chain] = {
                    "rows": [],
                    "dets": [],
                    "meta": {},
                    "elem": {},
                    "sig": signal,
                    "band": band,
                    "desc": dict(meta or {}, chain=chain, signal=signal, band=band),
                }
                self._order.append(chain)
        return _ChainView(self, chain)

    def _collect(self, ids, key):
        """Rows/detections for `ids`, each copied and tagged with its chain, signal and band.

        The tags are added, never substituted, so the rows keep GnssCoherentCombiner's schema
        and the viewer's signal_metrics() ignores the three extra keys."""
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
        return {
            "chains": {c: self._chains[c]["meta"] for c in ids if c in self._chains}
        }

    def set_rf(self, rf, t):
        """Publish the fleet's RF-path health, receiver-wide rather than per chain.

        The voltage tap is per GPU and serves every signal on it, so clip fraction and band
        power are properties of the receiver, not of a tracking chain. Filing them under a
        chain would invite reading an instrument-wide saturation as one chain's problem.
        """
        with self._lock:
            self._rf = {"t": t, "instances": rf}

    def rf_instances(self):
        """The newest set_rf instances (tap URL -> health), for each chain's dark check."""
        with self._lock:
            return dict(self._rf["instances"])

    def set_dark(self, dark, t, chain=None):
        """One chain's dark instances (fleet.dark_instances), served in get_rf under `dark`.

        Per chain, because a channel map can be wrong for one chain and right for the others on
        the same GPU; keyed by tap URL, so a reader joins each verdict to its tap's row.
        """
        with self._lock:
            self._dark[chain] = {"t": t, "instances": dark}

    def set_sky(self, sysc, sats, t):
        """One constellation's az/el, from the chain that predicts it. Receiver-wide output.

        Served so the viewer draws the broker's sky. Computing its own would mean fetching and
        merging a second copy of the broadcast ephemeris (a second writer of the shared nav
        cache), with a sky that could disagree with the broker's.
        """
        with self._lock:
            self._sky[sysc] = {
                "t": t,
                "sats": {
                    int(p): (round(float(v[0]), 2), round(float(v[1]), 2))
                    for p, v in (sats or {}).items()
                },
            }

    def set_elements(self, table, chain=None):
        """The per-element complex-gain table, served on /get_elements.

        {prn: {probe, inst: {tag: {keff, amp[], ph[], sig[]}}}} -- amplitude and phase per
        antenna per instance (elemgain.gain_table). Stored whole and replaced whole; a stamped
        `utc` rides along so a stalled producer shows as an aging table rather than a
        fresh-looking stale one."""
        with self._lock:
            st = self._chains.get(chain)
            if st is not None:
                st["elem"] = {"utc": _now(), "prns": table}

    def update(
        self,
        fleet,
        seeds,
        dll_trim,
        dets=None,
        fcoh=None,
        chain=None,
        pcn0=None,
        kcoh=None,
        innov=None,
        cpp_trim=None,
        integ=None,
        integ_now=None,
        trk=None,
        fadr=None,
    ):
        rows = []
        fcoh = fcoh or {}
        pcn0 = pcn0 or {}
        kcoh = kcoh or {}
        innov = innov or {}
        cpp_trim = cpp_trim or {}
        # dead-reckon integrity, keyed by PRN: (residual_chips, when). Absent until the
        # clock and its drift both exist, which is the honest state to publish.
        _integ = integ or {}
        _now_w = integ_now or time.time()
        # the tracker's code residual (trkresid), keyed by PRN; same absence semantics
        _trk = trk or {}
        _fadr = fadr or {}
        for prn, v in sorted(fleet.items()):
            c = v.get("coh_row") or {}
            sd = seeds.get(prn, {})
            # Fleet incoherent amplitude/significance from the summed prompt power, referenced
            # to the probes' median prompt power (p_med), the same reference the presence gate
            # uses, so the number in the viewer and the number the loop gates on agree.
            p_med = v.get("p_med") or 0.0
            ratio = (v["p_pow"] / p_med) if p_med > 0 else 0.0
            row = dict(c)  # start from the best instance's row...
            row.update(
                {  # ...then override what the fleet knows better
                    "prn": prn,
                    "amp_snr": math.sqrt(max(0.0, ratio - 1.0)) if ratio > 0 else 0.0,
                    "amplitude": math.sqrt(max(0.0, v["p_pow"])),
                    "unbiased_amplitude": math.sqrt(max(0.0, v["p_pow"] - p_med)),
                    "dll_disc": v["disc"],
                    # `doppler_hz` is the commanded seed, not what the correlator ran at. The
                    # two differ by the tracker's feed-forward, dop_rate x (seed age), and the
                    # seed changes only when the broker reseeds, so this field is a staircase
                    # even while the tracker follows the sky smoothly. Both are published so no
                    # one infers the tracker's state from the broker's command:
                    # doppler_applied_hz is the best instance's combiner doppler_hz, record slot
                    # 1 (REC_DOPPLER, PrnCtl::fcar_report), the propagated value the despread
                    # used.
                    "doppler_hz": sd.get("doppler_hz", c.get("doppler_hz", 0.0)),
                    "doppler_applied_hz": c.get("doppler_hz"),
                    "code_phase_chips": sd.get(
                        "code_phase_chips", c.get("code_phase_chips", 0.0)
                    ),
                    # fleet-only fields, outside the combiner schema
                    "fleet_q": v["q"],
                    "fleet_q_floor": v["q_floor"],
                    "fleet_present": bool(v["present"]),
                    # telescope-wide: the epoch the nodes serve is suspect (gnss_broker/timebase).
                    # Consumers withhold geometry while this is set; the row's own numbers stand.
                    "time_base_suspect": timebase.VERDICT.suspect,
                    # The dead-reckon integrity residual, in this chain's chips: search minus
                    # model with the solved receiver clock removed (deadreckon: r_i =
                    # wrap(d_i - clk), d_i = measured - predicted), the code observable the
                    # escape referee's veto reads. It cannot be rebuilt from code_phase_chips,
                    # an argument whose Doppler lever grows with uptime (~5095 chips/Hz at 6.8
                    # days; gnssSeedTransport.hpp).
                    "dr_integ_chips": _integ.get(prn, (None, None))[0],
                    "dr_integ_age_s": (
                        None
                        if _integ.get(prn) is None
                        else round(_now_w - _integ[prn][1], 2)
                    ),
                    # The tracker's code residual (trkresid): the replica placement the closed
                    # fleet DLL holds, lifted from the records' slot 1/2 pair, minus the model,
                    # minus the clock the seeds ride (clk_now). The quantity dr_integ_chips forms
                    # against the solved clock, but from the stage that resolves ~1e-2 chips,
                    # and present on chains with no detectors.
                    # `_s` is the currency-free form (seconds): a consumer converting chips to
                    # metres must know which chips (L2C's are CM chips), and seconds do not ask.
                    "trk_resid_s": (_trk.get(prn) or {}).get("s"),
                    # the same residual with the receiver clock left in: the code range residual,
                    # the quantity a carrier residual also carries (Hatch differences the two)
                    "trk_range_s": (_trk.get(prn) or {}).get("raw_s"),
                    "trk_resid_sd_chips": (_trk.get(prn) or {}).get("sd"),
                    "trk_resid_n": (_trk.get(prn) or {}).get("n"),
                    "trk_resid_age_s": (
                        None
                        if _trk.get(prn) is None
                        else round(_now_w - _trk[prn]["t"], 2)
                    ),
                    # The record's own (argument, Doppler, hop) triple, forwarded untouched. The
                    # argument is meaningful only against the Doppler it was expressed in, so the
                    # three travel together; `code_phase_chips` above is the seed's argument and
                    # pairs with `doppler_hz`, not with `doppler_applied_hz`.
                    "cp_rec_chips": c.get("code_phase_chips"),
                    "dop_rec_hz": c.get("doppler_hz"),
                    "rec_hop": c.get("pow_hop"),
                    # The fleet ADR (fleetadr): accumulated carrier phase folded from every
                    # instance's records at exact hops. The combiner's adr_cycles (in the best
                    # instance's row) is one instance's arc, located by its length in records
                    # rather than by a hop, so it cannot be paired across bands; this can.
                    # fadr_dop_cycles has the nominal f_c*dt stripped (exactly), so it is the
                    # Doppler-integrated phase a consumer can difference against a model at
                    # t(fadr_hop); fadr_hop0 is the arc's first hop.
                    "fadr_cycles": (_fadr.get(prn) or {}).get("cycles"),
                    "fadr_dop_cycles": (_fadr.get(prn) or {}).get("dop_cycles"),
                    "fadr_hop": (_fadr.get(prn) or {}).get("hop"),
                    "fadr_hop0": (_fadr.get(prn) or {}).get("hop0"),
                    "fadr_arc": (_fadr.get(prn) or {}).get("arc"),
                    "fadr_n_rec": (_fadr.get(prn) or {}).get("n_rec"),
                    "fadr_n_inst": (_fadr.get(prn) or {}).get("n_inst"),
                    "fadr_n_silent": (_fadr.get(prn) or {}).get("n_silent"),
                    # the bright-satellite projection on this satellite (fleetadr.publish)
                    "proj_b": (_fadr.get(prn) or {}).get("proj_b"),
                    "proj_k": (_fadr.get(prn) or {}).get("proj_k"),
                    "proj_src": (_fadr.get(prn) or {}).get("proj_src"),
                    "fadr_trim_cycles": (_fadr.get(prn) or {}).get("trim_cycles"),
                    "fadr_res_cycles": (_fadr.get(prn) or {}).get("res_cycles"),
                    # the ADR at the newest fleet-wide grid hop (fleetadr.GRID_HOPS): the sample
                    # at which every chain's rows carry the same hop, so bands pair exactly
                    "fadr_g_hop": (_fadr.get(prn) or {}).get("g_hop"),
                    "fadr_g_dop_cycles": (_fadr.get(prn) or {}).get("g_dop_cycles"),
                    "fadr_g_cycles": (_fadr.get(prn) or {}).get("g_cycles"),
                    # The last GRID_KEEP grid hops as [hop, dop_cycles, cycles, n_rec], newest
                    # last. The row refreshes once per broker cycle, slower than the ~1.0066 s
                    # grid, so the scalars alone skip grid hops and two chains polled out of
                    # phase pair on only some of them; the history makes the pairing
                    # independent of poll phase.
                    "fadr_g_hist": (_fadr.get(prn) or {}).get("g_hist"),
                    "fleet_instances": v["n_src"],
                    # Cross-sender coherence of the derotated prompts (combdll): ~1 when the
                    # senders share one phase reference, ~0 when they do not.
                    "fleet_xcoh": v.get("xcoh"),
                    "fleet_hop": v["hop"],
                    "coh_src": v.get("coh_src"),
                    "code_phase_rate": sd.get("code_phase_rate", 0.0),
                    # The second-order carrier term. propagate_seed turns it into the quadratic
                    # code term (quad = 0.5*(f_chip/f_c)*dop_rate*dt^2), which holds the phase
                    # while the Doppler accelerates, most near zenith where the signal is
                    # strongest. Published so its absence is visible: a seed without it lets the
                    # code walk off quadratically with seed age.
                    "doppler_rate_hz_s": sd.get("doppler_rate_hz_s"),
                    "dll_trim": dll_trim.get(prn, 0.0),
                    # The C++ fleet loop's standing trim, read back from <fleet-trim-url>/get_dll.
                    # dll_trim above is the Python arm only, which stands down for the PRNs the
                    # C++ loop is armed on; the applied correction is the sum of the two.
                    "dll_trim_cpp": cpp_trim.get(prn, 0.0),
                }
            )
            # The innovations ride the row when measured (fleetdll.py): innov_* from recent
            # detections, minnov_* from the model, each the freshest value, its 10-minute p95
            # and count. Absent keys mean "not measurable", not zero.
            iv = innov.get(prn)
            if iv:
                row.update(iv)
            # Fleet-coherent override: fleet_coherent solves the cross-node phase alignment the
            # power combine avoids, so for a PRN that cleared its measured null floor the
            # coherent statistics are the fleet's. Everything else keeps the best single
            # instance's numbers.
            fc = fcoh.get(prn)
            if fc and fc.get("present"):
                row.update(
                    {
                        "deep_snr": fc["deep_snr"],
                        "deep_amplitude": fc["deep_amplitude"],
                        "coh_frac": fc["coh_frac"],
                        "coh_src": "fleet:%d" % fc["n_src"],
                        # The record-stream carrier rate, Hz (fleet._fit_record_rate), which
                        # the next cycle's KCOH fold centres on. Served beside the fold's
                        # argmax deep_rate_full_hz, which remains the rrate state's coarse
                        # feed, so the two estimators can be compared on sky.
                        "rec_rate_hz": fc.get("rate_hz"),
                    }
                )
            else:
                # Quadrature fallback: the root-sum-square of the floor-cleared instances'
                # deep_snr (fleet_dll's coh_quad, two or more instances). It lands near the
                # fleet value, so the published series stays continuous across the fleet
                # gate; coh_src 'quad:N' names the population.
                _q = v.get("coh_quad")
                if _q is not None:
                    row.update({"deep_snr": _q[0], "coh_src": "quad:%d" % _q[1]})
            # Per-instance health from the fleet-coherent per-instance deep_snr. Every
            # instantaneous statistic in this row flickers (the deep stage collapses and
            # recovers fleet-wide), so these fields are built to measure the flicker:
            #  * The median over instances, not the max: the max is an order statistic that
            #    churns whenever the winning instance changes and is biased upward by the most
            #    favourable noise draw. Instance agreement depends on the regime (tight when
            #    strong, wide when noise-dominated), so the spread ships as inst_snr_lo/_hi
            #    beside the median rather than being reduced away.
            #  * A trailing duty, det_duty_s, rather than a snapshot (below).
            _pi = sorted((fc.get("per_inst") or {}).values()) if fc else []
            if _pi:
                _n = len(_pi)
                row["inst_snr_med"] = (
                    _pi[_n // 2] if _n % 2 else 0.5 * (_pi[_n // 2 - 1] + _pi[_n // 2])
                )
                row["inst_snr_lo"], row["inst_snr_hi"] = _pi[0], _pi[-1]
                row["inst_snr_n"] = _n
            # det_duty_s: the fraction of samples in the last DUTY_WIN_S in which the median
            # instance's coherent SNR beat the fleet's measured floor. A satellite that is
            # half dead reads 0.5, and that is the finding.
            #
            # It is a deep-stage duty, not "is the satellite being tracked": the deep fold can
            # fail to certify a satellite the despread holds continuously (prompt power flat
            # through the collapse), identically on every instance because it is a
            # deterministic algorithm over near-identical data. Read a low duty as "the deep
            # stage keeps losing this", never as "the array lost the sky"; prompt_hold answers
            # the second question.
            _t_now = _now()
            _key = (chain, prn)
            _h = self._hist.setdefault(_key, collections.deque())
            if _pi:
                _fl = float(fc.get("floor") or 0.0)
                _h.append(
                    (_t_now, row["inst_snr_med"], row["inst_snr_med"] > _fl > 0.0)
                )
            while _h and _t_now - _h[0][0] > DUTY_WIN_S:
                _h.popleft()
            _med_win = None
            if _h:
                row["det_duty_s"] = sum(1 for _, _, ok in _h if ok) / float(len(_h))
                _m = sorted(v for _, v, _ok in _h)
                _med_win = _m[len(_m) // 2]
            # `sig` is published with its source (sig_src) so no consumer re-derives it as
            # coherence_s > 0 ? max(deep_snr, amp_snr) : amp_snr, which pairs a deep_snr that
            # may be the fleet's with one instance's coherence_s and flickers as the winning
            # instance changes.
            #
            # prompt_hold: the fleet-summed prompt power over the probes' noise median. No deep
            # fold, rate search, nav-wipe ladder or floor certification, just "is there signal
            # under the tap we commanded", so it does not flicker with the deep stage. It
            # answers "are we on the satellite"; the deep columns answer "did the coherent
            # stage manage to use it".
            row["prompt_hold"] = ratio  # fleet prompt power / noise median
            # `sig`, from the first source that applies (sig_src names it):
            #   kcoh:N       the known-rate fold's significance over the probes' identically
            #                folded floor, when the fold cohered (eta >= n_rec/4);
            #   inc:N        else the incoherent t-statistic (combdll.prompt_cn0's sig_inc);
            #   kcoh_weak:N  else the fold that did not cohere;
            #   inst_med:N   else the trailing median of the per-instance deep SNR;
            #   inst_best    else the best instance's certified deep_snr;
            #   incoh        else amp_snr.
            # The deep-fold significances re-search rate per integration and read low on
            # satellites the search sees strongly; a fold that did not cohere reports a weak
            # detection, true of the fold and false of the satellite, hence the eta test. The
            # deep numbers stay published beside it.
            # _kc2/_pc2 are their own lookups: _pc/_kc are assigned further down this loop
            # body, so reading them here would raise UnboundLocalError on the first PRN and
            # read the previous PRN's values after it.
            _kc2 = kcoh.get(prn) or {}
            _pc2 = pcn0.get(prn) or {}
            _ks, _eta, _kn = _kc2.get("sig"), _kc2.get("eta"), _kc2.get("n_rec")
            _si = _pc2.get("sig_inc")
            if _ks is not None and _eta is not None and _kn and _eta >= 0.25 * _kn:
                row["sig"] = _ks
                row["sig_src"] = "kcoh:%d" % _kn
            elif _si is not None:
                row["sig"] = _si
                row["sig_src"] = "inc:%d" % (_pc2.get("n_rec") or 0)
            elif _ks is not None:
                row["sig"] = _ks
                row["sig_src"] = "kcoh_weak:%d" % (_kn or 0)
            elif _pi and row.get("det_duty_s") is not None:
                row["sig"] = _med_win or row["inst_snr_med"]
                row["sig_src"] = "inst_med:%d" % row["inst_snr_n"]
            elif (c.get("coherence_s") or 0.0) > 0.0 and c.get("deep_snr"):
                row["sig"] = c["deep_snr"]  # one instance, its own certified span
                row["sig_src"] = "inst_best"
            else:
                row["sig"] = row.get("amp_snr") or 0.0
                row["sig_src"] = "incoh"
            # cn0_coh_db = 20log10(deep_snr) - 10log10(coherence_s), both from the best
            # instance's own combiner row `c`, so the SNR and the span describe the same
            # integration by construction. They must: a C/N0 is amplitude-SNR^2 per unit time,
            # and an SNR over N records paired with a span over M != N records is a different
            # quantity that jumps whenever either count changes. The row's deep_snr (fleet or
            # quadrature) and coherence_s (one instance's ladder pick) are not such a pair.
            # It is one estimator unconditionally: switching between the fleet and the instance
            # value would mix two populations several dB apart.
            #
            # The fleet's coherent gain does not belong in a C/N0. C/N0 is a property of the
            # signal and the receiver's sensitivity, not of how many nodes were summed;
            # combining instances raises the detection significance, which is what deep_snr
            # is for and what the gates consume. The fleet's view stays published in
            # deep_snr / coh_src.
            _bi = c.get("deep_snr")
            _T = float(c.get("coherence_s") or 0.0)
            row["cn0_coh_db"] = (
                20.0 * math.log10(_bi) - 10.0 * math.log10(_T)
                if (_bi and _bi > 0.0 and _T > 0.0)
                else None
            )
            # The prompt C/N0 (combdll.prompt_cn0), the served radiometry: per-record prompt
            # power, kept per window where the prompt is the tallest tap, noise-debiased by
            # the probes' clipped-mean prompt power. No fit in it: the rate is the tracker's
            # and the tap is the loop's, where cn0_coh_db's deep fold re-searches a residual
            # rate per integration. cn0_prompt_duty is the fraction of records kept: near 0 the
            # number rests on a few upward fluctuations (decline it). cn0_prompt_split_db is
            # the even/odd-record self-consistency.
            _pc = pcn0.get(prn)
            if _pc:
                row["cn0_prompt_db"] = _pc.get("cn0_db")
                row["cn0_prompt_duty"] = _pc.get("duty")
                row["cn0_prompt_split_db"] = _pc.get("split_db")
                # The probes themselves ride the same rows (they are seeded PRNs); flag
                # them so no consumer plots a below-horizon noise reference as a satellite.
                row["noise_probe"] = bool(_pc.get("probe"))
            # The known-rate coherent C/N0 (combdll.coh_cn0): the ~1 s fold centred on the
            # previous cycle's record-stream rate (or zero), plus a one-parameter residual-rate
            # fit the probes ride identically so their floor absorbs its gain. It is a power
            # mean over instances, so on a strong satellite it reads below cn0_prompt_db by up
            # to 10log10(n_inst); their per-row difference, that loss plus any decoherence, is
            # a standing cross-check. kcoh_sig is the fold power over the probes' identically
            # folded floor (the detection number for deep-sidelobe satellites below the
            # per-record gate); kcoh_eta is the coherence efficiency (n_rec when fully
            # coherent, ~1 with no coherent gain, mostly None on noise, where a debiased power
            # is often not positive): a wrong rate or a phase discontinuity shows up there
            # rather than as a silently low C/N0. cn0_kcoh_sky_db is the same fold on the
            # sky-corrected prompt.
            _kc = kcoh.get(prn)
            if _kc:
                row["cn0_kcoh_db"] = _kc.get("cn0_db")
                row["cn0_kcoh_sky_db"] = _kc.get("cn0_sky_db")
                row["kcoh_sig"] = _kc.get("sig")
                row["kcoh_eta"] = _kc.get("eta")
                row["kcoh_n"] = _kc.get("n_rec")
                row["kcoh_rate_hz"] = _kc.get("rate_hz")
                row["kcoh_t_coh_s"] = _kc.get("t_coh_s")
            #
            # Prompt lock. The deep-fold numbers (deep_snr, cn0_coh_db) are blind to code error:
            # the fold integrates the prompt coherently (rate and phase searched, code not), and
            # its ~10x integration gain still reads a satellite whose prompt tap has slid off the
            # peak, so a row can serve a strong cn0_coh_db while E/P/L sit on noise. A zero
            # discriminator means on-peak or blind (E ~= L ~= noise gives zero), so it is no
            # evidence of health either. Two witnesses of different kinds:
            #   s4_raw -- the prompt intensity's fractional variation (best instance's row).
            #       Absolute and per satellite: it needs no population or floor estimate. ~1.0
            #       is Rayleigh, no coherent component. Fails alone under strong scintillation,
            #       which is real on these bands.
            #   fleet_present -- q and summed prompt power against the probe floors
            #       (fleet.apply_presence). Immune to scintillation, but a chain without probes
            #       reports every row not present (UNANCHORED), leaving s4_raw to decide alone.
            # Blind is asserted only when both agree, so the flag never calls a satellite blind
            # on one estimator's weakness.
            _s4 = row.get("s4_raw")
            _rayleigh = _s4 is not None and _s4 >= PROMPT_RAYLEIGH_S4
            row["prompt_rayleigh"] = _rayleigh
            row["prompt_lock"] = not (_rayleigh and not row["fleet_present"])
            # cn0_coh_db is published unconditionally (an estimator's output is never destroyed:
            # a row that has had a model applied cannot be un-applied), with prompt_lock beside
            # it so consumers can decline it; gnss_observables carries the flag into the obs
            # records.
            rows.append(row)
        meta = {
            "n_prn": len(rows),
            "present": sum(1 for r in rows if r["fleet_present"]),
            "time_base_suspect": timebase.VERDICT.suspect,
            "time_base_dt_s": timebase.VERDICT.dt_s,
            "utc": _now(),
        }
        with self._lock:
            st = self._chains.get(chain)
            if st is not None:
                st["rows"] = rows
                if dets is not None:
                    st["dets"] = dets
                st["meta"] = meta


class _ChainView(object):
    """One chain's handle on a shared publisher.

    Carries the chain id, so call sites -- `publisher.update(...)`, `publisher.set_sky(...)` --
    are the same whether the process runs one chain or several, without threading a chain id
    through the cycle body."""

    __slots__ = ("_pub", "_chain")

    def __init__(self, pub, chain):
        self._pub, self._chain = pub, chain

    def update(self, *a, **kw):
        kw["chain"] = self._chain
        return self._pub.update(*a, **kw)

    def set_elements(self, table):
        return self._pub.set_elements(table, chain=self._chain)

    def set_rf(self, rf, t):
        # Receiver-wide, so it forwards without a chain tag: the voltage tap is per GPU and
        # serves every signal on it. It must still be here: the broker only ever holds a
        # _ChainView, so a FleetPublisher method missing from this class is an AttributeError
        # at its first call, which kills that chain while the others run on.
        # python/scripts/gnss/test_chainview_surface.py checks this class's surface.
        return self._pub.set_rf(rf, t)

    def rf_instances(self):
        # Receiver-wide: every chain reads the taps the polling chain published.
        return self._pub.rf_instances()

    def set_dark(self, dark, t):
        return self._pub.set_dark(dark, t, chain=self._chain)

    def set_sky(self, sysc, sats, t):
        # Per-chain call site, receiver-wide effect: the chain names its own constellation and
        # the publisher unions them. Present here for the reason set_rf gives.
        return self._pub.set_sky(sysc, sats, t)
