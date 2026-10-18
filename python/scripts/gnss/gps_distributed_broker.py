#!/usr/bin/env python3
"""Distributed GNSS broker: search detections -> consensus seeds -> trackers.

The control plane of the distributed-band channelized pipeline. It is decoupled from
the sample dataflow -- REST glue plus assignment policy -- so the same broker drives a
single kotekan instance and the cross-node CHORD layout; only the endpoint list changes.

Each iteration (every --interval seconds):
  1. GET <detector>/get_detections from every detection source (GnssChannelizedSearch)
     and keep the best-SNR detection per PRN. --detectors may be empty: the chain then
     seeds every visible satellite from the BRDC model (--dead-reckon), with no search.
  2. (optional) gate by visibility: drop below-horizon PRNs.
  3. GET <combiner>/get_status for the full-band |A| per PRN; coast a visible PRN
     through a signal dropout (hold its seed, forecast its Doppler forward from the
     orbit) so a radar sweep or brief fade does not lose the lock. Drop it only when it
     sets or stays unlocked for the whole coast budget (coast_budget); dead-reckoned
     seeds are model-owned and never dropped this way.
  4. POST the consensus seed set [{prn, doppler_hz, code_phase_chips, ...}] to every
     tracker's /set_seeds, so all subbands despread the same (cp, Doppler) and the
     per-subband products recombine coherently.

Endpoints: --detectors and --trackers are comma-separated lists, --combiner is one
endpoint. Each entry may be:
  * a bare stage name        -> resolved against --rest-url (one kotekan instance)
  * an absolute URL          -> http://host:port/stage (per node, mix freely)
  * a brace range {a..b}     -> track_{00..49} expands to track_00..track_49,
                                bash-style (zero-padded iff an operand is padded);
                                http://nodeB:12048/track_{0..24} works too (lists only).

REST via stdlib urllib. Visibility and Doppler prediction (--almanac, needs --lat/--lon)
come from the BRDC broadcast ephemeris, or from TLEs via skyfield (--almanac-source tle).
Runs one chain; scripts/gnss/broker_multi.py runs every chain of
config/gnss_chains_chord.yaml as threads of one process.
"""

import math
import os
import statistics
import sys
import time

# -- frame0 guard -----------------------------------------------------------------------
# A changed F-engine frame 0 invalidates every seed this process computes. Confirm it over
# several reads (one odd answer from the endpoint is not an F-engine restart), then exit so
# the supervisor restarts the broker and the startup path re-latches the anchor.
_ANCHOR_STRIKES = 3  # consecutive 60 s re-reads that must disagree
_EXIT_ANCHOR_CHANGED = 3  # distinct exit code, so the supervisor can log why
from datetime import datetime, timezone

sys.path.insert(
    0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))
)


# ---------------------------------------------------------------------------------------
# The stateless parts of the broker (transport, fits, sky, the per-stage modules) live in
# gnss_broker/. This file keeps main(), one chain's loop; broker_multi.py runs one per chain
# as threads of one process.
# ---------------------------------------------------------------------------------------
from gnss_broker.transport import (  # noqa: E402
    _TranscriptDone,
    _TR,
    _now,
    _get,
    _post,
    _log,
    _log_rl,
    resolve_prefix,
    parse_endpoints,
    log_tag,
    install_dns_cache,
    record_cycle,
)
from gnss_broker import sky as _sky  # noqa: E402
from gnss_broker import telem as _telem  # noqa: E402
from gnss_broker.fits import (  # noqa: E402
    tracker_phase_at,
    rate_residuals,
    cp_rate_from_code_bias,
)
from gnss_broker.publish import FleetPublisher  # noqa: E402
from gnss_broker.seed import Seed  # noqa: E402
from gnss_broker.cli import (
    build_parser,
    _FROZEN,
    SIGNAL_IMPLIED,
)  # noqa: E402
from gnss_broker.context import ChainContext  # noqa: E402  (the stage interface)
from gnss_broker.clockbias import ClockBias  # noqa: E402  (the receiver LO bias)
from gnss_broker.loopstate import (  # noqa: E402
    NhOverlay,
    DllLoopState,
    HoldState,
    CpTracking,
    RateFeedState,
    NavDecoders,
)
from gnss_broker import deadreckon  # noqa: E402  (the clock pipeline)
from gnss_broker import almanac as almanac_stage  # noqa: E402  (orbit + visibility)
from gnss_broker import ratefeed  # noqa: E402  (the carrier-rate feeds)
from gnss_broker import trimarm  # noqa: E402  (C++ trim arming)
from gnss_broker import prnmap  # noqa: E402  (live PRN membership)
from gnss_broker import searchhint  # noqa: E402  (narrow the search)
from gnss_broker import seeding  # noqa: E402  (detections -> seeds)
from gnss_broker import navbits  # noqa: E402  (off in production)
from gnss_broker import fleetdll  # noqa: E402  (the fleet DLL shell)
from gnss_broker.detectors import (  # noqa: E402  (D0, D1, D3)
    QSeries,
    BrownoutDetector,
    SawtoothDetector,
)
from gnss_broker import signals  # noqa: E402
from gnss_broker import receiver  # noqa: E402
from gnss_broker.state_filter import SatBiasFilter  # noqa: E402
from gnss_broker.sky import _dh_dpos  # noqa: E402


def main(argv=None, rx=None, publisher=None):
    # Uncached name resolution stalls the cycle on resolver timeouts (see install_dns_cache
    # in gnss_broker/transport.py). Idempotent, so the chain threads under broker_multi share
    # one cache.
    install_dns_cache()
    # `--signal help` before the parser, because --trackers is required and listing the
    # known signals must not depend on being able to name a fleet first.
    if "help" in (argv if argv is not None else sys.argv[1:]):
        _a = list(argv if argv is not None else sys.argv[1:])
        if (
            "--signal" in _a
            and _a.index("--signal") + 1 < len(_a)
            and _a[_a.index("--signal") + 1] == "help"
        ):
            print(
                "known signals (derived from lib/stages/gnss/gnssSignal.hpp):\n"
                + signals.describe()
            )
            return
    ap = build_parser(__doc__)
    args = ap.parse_args(argv)
    # Frozen tuning (cli._FROZEN): constants, set as args attributes so use sites read them
    # like flags; no command line can change them.
    for _k, _v in _FROZEN.items():
        setattr(args, _k, _v)
    # Module state, since every consumer of the boresight (D3's veto, the dead-reckon transit
    # freeze, the "_bore" post) asks sky.nearest_boresight and the chains share the process.
    _sky.BORESIGHT_KNOWN = args.boresight != "none"

    # ── The loops' per-satellite state ────────────────────────────────────────────────
    # One owner object per loop (see gnss_broker/loopstate.py), constructed first because
    # everything else in this function may touch them.
    _nho = NhOverlay()  # NH overlay alignment
    _dls = DllLoopState()  # the code loop and the C++ arming handshake
    _hold = HoldState()  # why a sat is held rather than dropped
    _cpt = CpTracking()  # per-sat code-phase history
    _rf = RateFeedState()  # carrier-rate observables and the commanded reference
    _nav = NavDecoders()  # broadcast nav-message decoders (off in production)
    # Per-satellite q series that keeps the satellites that stopped reporting. Judge an arm on
    # it, never on the DLL line, which lists only the survivors (see gnss_broker/detectors.py).
    _qpop = QSeries()
    _brown = BrownoutDetector()  # chain-wide presence collapse, labelled as an episode
    _saw = SawtoothDetector()  # a standing trim that ramps, then gets wiped
    # Live slot->PRN membership against the sky; does nothing unless --prn-reconfig apply.
    _prnmap = prnmap.PrnMapState()

    _raw_argv = list(argv if argv is not None else sys.argv[1:])
    if args.signal:
        try:
            _sig = signals.get(args.signal)
        except (KeyError, ValueError, RuntimeError) as e:
            ap.error("--signal: %s" % e)
        # dest -> the value the named signal implies. A value the caller gave explicitly is
        # kept only if it agrees; a disagreement is an error naming both numbers, because
        # naming a signal exists so these constants are not retyped.
        _implied = {
            "carrier_hz": _sig.carrier_hz,
            "chip_rate_hz": _sig.chip_rate_hz,
            "code_length": float(_sig.code_length),
            "long_code_segments": _sig.long_code_segments,
            "long_code_epoch_s": _sig.long_code_epoch_s,
            "nh_overlay_len": _sig.nh_overlay_len,
            "constellation": _sig.constellation,
            "dr_constellation": _sig.constellation,
        }
        if _sig.min_prn is not None:
            # The floor is only the fallback: the IGS registry's block types decide which
            # satellites carry the signal (prnmap.signal_incapable_prns).
            _implied["dr_min_prn"] = _sig.min_prn
            _implied["signal_capability"] = _sig.key
        assert _implied.keys() <= SIGNAL_IMPLIED, _implied.keys() - SIGNAL_IMPLIED
        for _dest, _want in _implied.items():
            _flag = "--" + _dest.replace("_", "-")
            _given = any(a == _flag or a.startswith(_flag + "=") for a in _raw_argv)
            if not _given:
                setattr(args, _dest, _want)
                continue
            _have = getattr(args, _dest)
            _same = (
                abs(_have - _want) <= 1e-9 * max(1.0, abs(_want))
                if isinstance(_want, float)
                else _have == _want
            )
            if not _same:
                ap.error(
                    "%s %r contradicts --signal %s, which implies %r. One of them is "
                    "wrong and neither would have errored on its own -- fix the "
                    "command rather than letting a silent override pick."
                    % (_flag, _have, args.signal, _want)
                )
        _log("signal %s: %r" % (args.signal, _sig))
    if args.transcript_write and args.transcript_read:
        ap.error("--transcript-write and --transcript-read are mutually exclusive")
    if args.transcript_write:
        _raw = list(argv if argv is not None else sys.argv[1:])
        _keep, _skip = [], False
        for _a in _raw:
            if _skip:
                _skip = False
                continue
            if _a == "--transcript-write":
                _skip = True  # drop the flag and its separate value
                continue
            if _a.startswith("--transcript-write="):
                continue
            _keep.append(_a)
        _TR.open_write(args.transcript_write, _keep)
    elif args.transcript_read:
        _TR.open_read(args.transcript_read)
        # Test only, under replay: scripts/gnss/broker_equiv.py's selftest proves the gate
        # can fail by scaling one argument by (1 + eps), far below anything physical, and
        # requiring the digest to move. The value names the knob and eps ("dll_gain:1e-6");
        # the harness tries several, because each fixture exercises different code (the e2e
        # fixture runs --dll-gain 0).
        _pert = os.environ.get("GNSS_BROKER_EQUIV_PERTURB")
        if _pert:
            _knob, _, _eps = _pert.partition(":")
            _f = 1.0 + float(_eps or "1e-6")
            if not hasattr(args, _knob):
                sys.exit("perturb: no such argument %r" % _knob)
            setattr(args, _knob, getattr(args, _knob) * _f)

    # After the transcript is open, or the tick would be neither recorded nor replayed. Setup
    # reads the clock too (the BRDC fetch epoch, the primed clock's stamp, _cb.meas_t, the
    # estimator schedule), so it runs at one frozen instant like every cycle, and a
    # recording captures that instant explicitly.
    _TR.tick()

    def _alm_now():
        """The time the almanac evaluates at: the wall clock (the transcript's, in a replay)."""
        return datetime.fromtimestamp(_now(), tz=timezone.utc)

    # ---- Receiver scope -----------------------------------------------------------------
    # What every chain on this telescope shares: the F-engine time anchor, the BRDC store and
    # the receiver clock (see gnss_broker/receiver.py). broker_multi passes one shared `rx`;
    # a chain run alone owns a private one, and every contribute/consume pair below then
    # talks to itself, so a single chain behaves as if there were no sharing.
    rx = rx if rx is not None else receiver.Receiver(log=_log)
    # Who this chain is, and which band it measures the code clock in. The band key is the
    # carrier to 10 kHz, so GPS L5 and Galileo E5a (the same 1176.45 MHz hardware) share one
    # code-clock scope, while E5b at 1207.14 MHz does not.
    chain_id = args.signal or (
        "%s@%.2fMHz"
        % (args.constellation or args.dr_constellation, args.carrier_hz / 1e6)
    )
    band_id = "%.2fMHz" % (args.carrier_hz / 1e6)
    # The joint solve (dr_joint_shadow) runs whenever a --joint-consume consumer is named;
    # without it the consumer would read a state nothing feeds.
    args.joint_shadow = bool(args.joint_consume)
    if args.rrate_command and not args.rrate_state:
        args.rrate_state = True  # same rule: the command gates on the row's sigma, and
        # only the feed can bring that below infinity

    base = args.rest_url.rstrip("/")
    detectors = parse_endpoints(args.detectors, base)
    trackers = parse_endpoints(args.trackers, base)
    combiner = resolve_prefix(args.combiner, base)
    # Fleet DLL population: every combiner whose E/L powers join the sum. The same list feeds
    # fleet_coherent, the element polls and the sat-geometry posts; --combiner stays the one
    # status source (amplitudes, lock and drop decisions, nav bits).
    dll_combiners = (
        parse_endpoints(args.dll_combiners, base) if args.dll_combiners else []
    )
    # Path B's population: the same estimator (fleet_coherent) on a different record stream,
    # kept separate so the two fleet numbers can be compared rather than blended. It feeds no
    # loop, so an n2 chain that is down or absent costs a log line and nothing else.
    n2_combiners = parse_endpoints(args.n2_combiners, base) if args.n2_combiners else []
    spectrum_endpoints = (
        parse_endpoints(args.spectrum_endpoints, base)
        if args.spectrum_endpoints
        else []
    )
    # The frame-synced telemetry gather (--telem-gather). One reader thread per process
    # (telem.shared_client): the stream carries every chain on one connection and the store is
    # keyed by chain. `telem_chain` is this thread's chain key, the string the trackers stamp
    # on every frame, read from the log tag rather than re-derived from --signal (nothing
    # enforces that the two agree).
    _tg = _telem.parse_endpoint(args.telem_gather)
    telem_client = _telem.shared_client(*_tg) if _tg else None
    telem_chain = log_tag() or (args.signal or "")
    if telem_client is not None:
        _log(
            "telem: gather %s:%d, chain key %r%s"
            % (
                _tg[0],
                _tg[1],
                telem_chain,
                "; fleet_coherent will read it"
                if args.telem_coherent
                else " (store only)",
            )
        )
    # Integer hop tolerance, derived once from the record geometry. Kept as an int so every
    # comparison downstream is integer arithmetic on the F-engine's own counter.
    dll_hop_window = max(0, int(round(args.dll_hop_window_s * args.hops_per_sec)))
    # --dll-deep-gate: PRNs (or 'all') whose trim presence is gated on deep_snr instead of
    # prompt power. Parsed once at launch and logged, so the gated set is a fact in the log
    # rather than something to infer from behaviour.
    _dg = (args.dll_deep_gate or "").strip()
    if _dg.lower() == "all":
        _deep_gate = True
    elif _dg:
        _deep_gate = {int(x) for x in _dg.replace(",", " ").split()}
    else:
        _deep_gate = None
    # The search-generated half of the deep-gate set (--dll-deep-gate-from-search), unioned
    # with _deep_gate each cycle; the detection times live in DllLoopState.deep_gate_seen.
    _dg_auto_last = [set()]  # last logged auto set, so the line prints on change only
    if _deep_gate:
        _log(
            "DLL DEEP GATE active on %s at %.1fx deep_floor -- these PRNs are trimmed "
            "on detection (deep_snr) instead of on prompt power, which is on-peak-biased and "
            "latches an off-peak satellite out of its own correction"
            % (
                "all PRNs"
                if _deep_gate is True
                else "PRN " + ",".join(str(p) for p in sorted(_deep_gate)),
                args.dll_deep_gate_margin,
            )
        )
    # Optional REST publication of the fleet-merged state (FleetPublisher). broker_multi passes
    # one shared publisher and this chain claims a slot on it; a chain run alone binds
    # --publish-port here, at launch, so a bind failure is fatal rather than leaving the viewer
    # with no source. Either way `publisher` is a per-chain view. _pub_desc is what a viewer
    # needs to draw the chain without a static table (constellation, label, record length,
    # whether a search feeds it), from the --signal descriptor (gnssSignal.hpp) when there is one.
    _pub_desc = {
        "constellation": args.constellation or args.dr_constellation,
        "carrier_hz": args.carrier_hz,
        "code_length": args.code_length,
        "t_rec": args.code_length / args.chip_rate_hz,
        "has_search": bool(detectors),
        "n_trackers": len(trackers),
        "sigid": _sig.primary if args.signal else None,
        "label": _sig.label if args.signal else chain_id,
        "short": _sig.short if args.signal else chain_id,
        "rf_band": _sig.rf_band if args.signal else band_id,
    }
    if publisher is not None:
        publisher = publisher.register(chain_id, args.signal, band_id, _pub_desc)
    elif args.publish_port:
        publisher = FleetPublisher(args.publish_port, _log).register(
            chain_id, args.signal, band_id, _pub_desc
        )
    else:
        publisher = None
    _nav.cnav_combiner = (
        resolve_prefix(args.cnav_combiner, base) if args.cnav_combiner else None
    )
    _nav.inav_combiner = (
        resolve_prefix(args.inav_combiner, base) if args.inav_combiner else None
    )
    _nav.fnav_combiner = (
        resolve_prefix(args.fnav_combiner, base) if args.fnav_combiner else None
    )
    _nav.bcnav2_combiner = (
        resolve_prefix(args.bcnav2_combiner, base) if args.bcnav2_combiner else None
    )
    _nav.bcnav1_combiner = (
        resolve_prefix(args.bcnav1_combiner, base) if args.bcnav1_combiner else None
    )
    _nav.cnav2_combiner = (
        resolve_prefix(args.cnav2_combiner, base) if args.cnav2_combiner else None
    )
    gating = args.lat is not None and args.lon is not None

    # Almanac assist: BRDC (default; PRN-indexed, so immune to TLE name->PRN mislabelling) or
    # TLEs (--almanac-source tle, and the fallback when BRDC is unavailable at startup).
    almanac_sats = None  # TLE mode: {prn: EarthSatellite}
    brdc_alm = None  # BRDC mode: {"mod", "eph", "eph_t"} for brdc_predict
    alm_sys = args.constellation or args.dr_constellation
    alm_min_prn = 19 if alm_sys == "C" else 1  # BeiDou fallback floor (see below)
    if args.almanac:
        if not gating:
            _log("--almanac needs --lat/--lon; disabling")
            args.almanac = False
    if args.almanac and args.almanac_source == "brdc":
        try:
            import gnss_ephemeris as _alm_eph_mod

            when = _alm_now()
            # Through the receiver: the parse is multi-system (entries keyed (sys, prn), each
            # consumer filters at predict time), so one store serves every chain sharing `rx`.
            # Keyed by the day, so a replayed epoch and a live one do not share a store.
            brdc_alm = rx.brdc(
                ("brdc", when.strftime("%Y-%j")),
                lambda: {
                    "mod": _alm_eph_mod,
                    "eph": _alm_eph_mod.parse_rinex_nav(_alm_eph_mod.fetch_brdc(when)),
                    "eph_t": _now(),
                },
            )
            n = sum(
                1 for k in brdc_alm["eph"] if k[0] == alm_sys and k[1] >= alm_min_prn
            )
            if n == 0:
                # A parse with zero satellites for this constellation is a failure, not a
                # result: fall through to TLE as for an exception. Otherwise the chain hints
                # nothing, which from outside looks the same as codes that do not correlate.
                _log(
                    "almanac: BRDC parsed but has 0 %s sats%s -> treating as UNAVAILABLE and "
                    "falling back to TLE"
                    % (
                        alm_sys,
                        " with PRN >= %d" % alm_min_prn if alm_min_prn > 1 else "",
                    )
                )
                brdc_alm = None
            else:
                _log(
                    "almanac: BRDC %s (%d %s sats%s) @ (%.4f, %.4f)"
                    % (
                        args.almanac_source,
                        n,
                        alm_sys,
                        ", PRN >= %d" % alm_min_prn if alm_min_prn > 1 else "",
                        args.lat,
                        args.lon,
                    )
                )
        except Exception as e:
            _log("BRDC almanac unavailable (%s); falling back to TLE" % e)
    if args.almanac and brdc_alm is None:
        try:
            from gps_beamtrack import load_gps_satellites
            from gps_beamtrack import DEFAULT_TLE_URL

            almanac_sats = load_gps_satellites(args.tle or DEFAULT_TLE_URL)
            _log(
                "almanac: loaded %d TLEs; predicting Doppler @ (%.4f, %.4f)"
                % (len(almanac_sats), args.lat, args.lon)
            )
        except Exception as e:
            _log("almanac unavailable (%s); falling back to search Doppler" % e)
            args.almanac = False

    # Dead-reckoned cp seeding: the BRDC ephemeris and a receiver clock (solved, adopted or
    # primed) predict the code phase of every sat the search cannot (yet) see. State: eph =
    # parsed BRDC records; t0m = GPST of capture sample 0 pre-reduced mod _drp.t_code, the
    # period of the full code the tracker despreads (full-GPST doubles quantize at ~0.24 L1
    # chips; the reduction keeps the mod arithmetic exact, and its constant error is common
    # to solve and seeds, so it lands in the solved clock); clk = receiver clock (chips, mod
    # code period) at epoch clk_t; seeded/pin = which PRNs are model-owned and when each was
    # last re-anchored.
    # Bound unconditionally: the import below runs only with --dead-reckon, and any reference
    # to the name on a search-only chain would otherwise raise UnboundLocalError.
    dr_eph_mod = None
    # Signal-capability floor for the dead-reckon seeder (--dr-min-prn; 19 on BeiDou by
    # default). B1C, B2a and B2b are BDS-3 only: a BDS-2 satellite carries none of them yet
    # passes every visibility, ephemeris and health gate, so only capability excludes it. The
    # floor is the fallback; when --signal-capability reads the IGS registry, the registry
    # decides (below).
    dr_min_prn = (
        args.dr_min_prn
        if args.dr_min_prn is not None
        else (19 if args.dr_constellation == "C" else 1)
    )

    # Signal-capability PRN gate (--signal-capability), evaluated at startup and re-read from
    # the registry every prnmap.CAPABLE_REFRESH_S (almanac.py): slots change hands, GLONASS's
    # a few times a year. The IGS satellite-metadata SINEX (SVN -> block -> PRN with its
    # validity window, cached in ~/.cache/kotekan_gps/, refreshed past prnmap.REGISTRY_MAX_AGE_S
    # here, before the loop) decides GPS, BeiDou and GLONASS (prnmap.signal_incapable_prns);
    # the Celestrak TLE block names are the fallback for GPS and for GLONASS, where they see
    # only the K marker, not the GLO-M+ satellites that also carry the CDMA signals. Both fail
    # open: only proof of incapability excludes a satellite, and an empty or failed lookup
    # leaves the gate disabled (None) rather than darkening the chain.
    _capable = None
    _capable_t = None
    # --constellation is unset unless given or implied by --signal; resolve the system the
    # same way chain_id, _pub_desc and alm_sys do.
    _cap_sys = args.constellation or args.dr_constellation
    if args.signal_capability and _cap_sys in ("G", "R", "C"):
        _capable_t = _now()
        try:
            from gnss_broker import prnmap as _pm

            _pm.ensure_registry()
            _incap = _pm.signal_incapable_prns(args.signal_capability)
        except Exception:
            _incap = set()
        if _incap:
            _capable = set(range(1, 64)) - _incap
            _reg_age = _pm.registry_age_s()
            _log(
                "signal-capability %s: %d PRN(s) excluded by the IGS registry (%s; copy "
                "%s old) -- seeds + hints restricted to block-capable satellites (and with "
                "the search's require_hint, an unhinted PRN is never scanned)"
                % (
                    args.signal_capability,
                    len(_incap),
                    ", ".join(str(p) for p in sorted(_incap)),
                    "%.1f d" % (_reg_age / 86400.0) if _reg_age is not None else "?",
                )
            )
    if _capable is None and args.signal_capability and _cap_sys in ("G", "R"):
        try:
            import gps_beamtrack as _bt

            # Given the default (gps-ops) source, signal_capable_prns reads the glo-ops group
            # for a GLONASS signal itself; an empty result disables the gate (below).
            _cap = _bt.signal_capable_prns(
                args.signal_capability, args.tle or _bt.DEFAULT_TLE_URL
            )
            if _cap:
                _capable = _cap
                _log(
                    "signal-capability %s: seeds+hints restricted to %d block-capable PRNs %s"
                    % (args.signal_capability, len(_capable), sorted(_capable))
                )
            else:
                _log(
                    "signal-capability %s: block lookup returned EMPTY -> filter DISABLED"
                    % args.signal_capability
                )
        except Exception as _e:
            _log(
                "signal-capability %s: block lookup FAILED (%s) -> filter DISABLED"
                % (args.signal_capability, _e)
            )

    if _cap_sys == "C" and _capable is not None:
        # The registry decides for BeiDou (C11-C14 are BDS-3 MEOs on re-used PRNs below 19).
        # The PRN floor stays as the fallback for an unreadable registry, because a BDS-2
        # satellite seeded or probed from the model is noise (see --dr-min-prn).
        alm_min_prn = dr_min_prn = 1

    dr_state = None
    # Per-satellite slow code bias b_sat. Chain-local, unlike the clock, because PRN
    # namespaces differ per constellation. Fed from the spectrum fit's tau
    # (--spectrum-endpoints) for PRNs at or above lock_snr; read by the dead-reckon model.
    # Without --spectrum-endpoints it never updates and get() returns exactly 0.0.
    bsat = SatBiasFilter(gain=args.bsat_gain)
    joint_consume = {c.strip() for c in args.joint_consume.split(",") if c.strip()}

    # -- P2c rotation (--joint-p2c-rotate) -------------------------------------------------
    # Withhold the most-established satellite from the joint state, coast it for
    # joint_p2c_hold_s while logging its residual against the state's prediction, release it,
    # and rotate to one not tested in the last joint_p2c_skip rotations. On a transit
    # instrument one hand-picked PRN is one satellite at one geometry for under an hour;
    # rotating builds a distribution across satellites and elevations. The residual's growth
    # with coast age measures the b_sat random walk (the state's q_b).
    p2c = {"key": None, "t0": 0.0, "samples": [], "history": [], "n0": 0}

    # The joint state is in reference chips (Receiver.JOINT_REF_RATE_HZ); the P2C lines report
    # in this chain's, like the coast samples they sit beside.
    _jk = receiver.JOINT_REF_RATE_HZ / float(args.chip_rate_hz)

    def _p2c_pick(js):
        """The most-established satellite not tested in the last few rotations."""
        cand = [
            (js._n.get(k, 0), k)
            for k in js._idx
            if js._n.get(k, 0) >= args.joint_mask_after
        ]
        if not cand:
            return None
        recent = {h["key"] for h in p2c["history"][-args.joint_p2c_skip :]}
        fresh = [c for c in cand if c[1] not in recent]
        return max(fresh or cand)[1]

    def _p2c_summary(js, t_now, why):
        """Close out one coast and log its residual-vs-age curve."""
        s = p2c["samples"]
        key = p2c["key"]
        if s:
            band = [
                (lo, hi, [r for a, r in s if lo <= a < hi])
                for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 1200))
            ]
            txt = " ".join(
                "%d-%ds %+.2f(n%d)" % (lo, hi, statistics.fmean(v), len(v))
                for lo, hi, v in band
                if v
            )
            _log(
                "P2C %s END (%s): %d samples over %.0f s | %s | final b %+.3f sigma %.3f"
                % (
                    _p2c_name(key),
                    why,
                    len(s),
                    s[-1][0],
                    txt,
                    # The state is in reference chips; the samples above are in ours.
                    (js.bias(key) / _jk) if key in js._idx else float("nan"),
                    (js.sigma(key) / _jk) if key in js._idx else float("nan"),
                )
            )
            p2c["history"].append(
                {
                    "key": key,
                    "n": len(s),
                    "age": s[-1][0],
                    "resid": [r for _, r in s],
                    "why": why,
                }
            )
        else:
            _log("P2C %s END (%s): no samples" % (_p2c_name(key), why))
            p2c["history"].append(
                {"key": key, "n": 0, "age": 0.0, "resid": [], "why": why}
            )
        p2c["key"] = None
        p2c["samples"] = []

    def _p2c_name(key):
        try:
            return "%s%d" % (key[0][0].upper(), key[1])
        except Exception:
            return str(key)

    def _p2c_tick(js, t_now):
        """Advance the rotation: start a coast, or end one that is done or has lost its sat."""
        if not args.joint_p2c_rotate or js is None:
            return
        if p2c["key"] is None:
            k = _p2c_pick(js)
            if k is not None:
                p2c.update(key=k, t0=t_now, samples=[], n0=js._n.get(k, 0))
                _log(
                    "P2C %s START: withholding after %d accepted updates (b %+.3f, "
                    "sigma %.3f) -- coasting %.0f s"
                    % (
                        _p2c_name(k),
                        p2c["n0"],
                        js.bias(k) / _jk,  # reference chips -> ours
                        js.sigma(k) / _jk,
                        args.joint_p2c_hold_s,
                    )
                )
            return
        # A withheld satellite is not fed, so its _t_seen stops advancing and the state evicts
        # it after max_age_s (900 s); joint_p2c_hold_s (600 s) stays below that. An eviction
        # is logged as its own outcome, not as a completed coast.
        if p2c["key"] not in js._idx:
            _p2c_summary(js, t_now, "sat dropped from state")
        elif t_now - p2c["t0"] >= args.joint_p2c_hold_s:
            _p2c_summary(js, t_now, "hold complete")

    def _p2c_hold(js, key):
        """True while this sat is the one the P2C rotation is coasting."""
        return key == p2c["key"] and key in js._idx

    def _joint_state(rx_, band, a):
        """The receiver-wide joint state (Receiver.joint_receiver, the object the solve
        writes) if it is fit to be consumed, else None. Never raises: a consumer must fall
        back to its own estimator, not take the broker down."""
        try:
            js = rx_.joint_receiver(band, CODE_LEN, a.chip_rate_hz, gauge_mode="prior")
            if len(js._idx) < a.joint_min_sats:
                return None
            # A deaf state rejects every update, so it cannot correct itself while predict()
            # keeps integrating clk_rate. Refuse it here, where fitness to consume is decided,
            # rather than relying on each consumer's own bound.
            return None if getattr(js, "deaf", False) else js
        except Exception:
            return None

    if args.dead_reckon:
        if not args.almanac:
            _log("--dead-reckon needs --almanac; disabling")
        else:
            try:
                import gnss_ephemeris as dr_eph_mod

                dr_state = {
                    "eph": None,
                    "eph_t": 0.0,
                    "t0m": None,
                    "clk": None,
                    "clk_t": 0.0,
                    "next": 0.0,
                    "log_next": 0.0,
                    "pin": {},
                    "seeded": set(),
                }
                if args.dr_clock_chips is not None:
                    # Cold start without a search stage: the clock solve takes a median of
                    # measured code-phase residuals, so it needs satellites already tracking,
                    # and nothing tracks until something seeds it. With a GPS-disciplined
                    # F-engine the clock offset is known a priori, so --dr-clock-chips primes
                    # it. Only detections feed the solve (`best` comes from /get_detections),
                    # so with --detectors empty (E5a/B2a have no blind acquisition, see
                    # docs/CHORD_MULTIBAND.md) the prime stands until the chain adopts a
                    # sibling's clock (--dr-clock-adopt), or for the whole run without one.
                    dr_state["clk"] = args.dr_clock_chips % float(args.code_length)
                    dr_state["clk_t"] = _now()
                    # A prime is not a measurement: the first multi-sat solve, or an adopted
                    # clock, replaces it outright (dr_clock_quality, dr_clock_adopt_rx) rather
                    # than EMA-ing in from it, which would walk every satellite's integrity
                    # residual through the error and hand co-hosted chains a moving target.
                    # Seeding waits up to dr_clock_wait_s for that replacement.
                    dr_state["clk_primed"] = True
                    _log(
                        "dead-reckon: receiver clock PRIMED %.2f chips = %.3f us "
                        "(--dr-clock-chips; %s)"
                        % (
                            dr_state["clk"],
                            dr_state["clk"] / args.chip_rate_hz * 1e6,
                            "REPLACED outright by the first multi-sat solve; under "
                            "--detectors-monitor none runs, and it stands until "
                            "--dr-clock-adopt replaces it"
                            if detectors
                            else "no detectors: this value stands until another chain "
                            "contributes one (--dr-clock-adopt: same band first, else a "
                            "cross-band bootstrap), and is fixed for the run without that",
                        )
                    )
                _log(
                    "dead-reckon: BRDC cp seeding armed (%s, repin %.0f s)"
                    % (args.dr_constellation, args.dr_repin_s)
                )
            except Exception as e:
                _log("dead-reckon unavailable (%s); disabled" % e)

    seeds = {}  # prn -> {"doppler_hz", "code_phase_chips", ...} (consensus)
    _elem_arch_t = [0.0]  # last per-element archive append (element_archive_every_s)
    _geom_post_t = [0.0]  # last sat-geometry post (--post-sat-geometry)
    _elem_poll_t = [0.0]  # last /get_elements poll (element_poll_every_s)
    # prn -> residual carrier rate the known-rate coherent fold (KCOH) derotates with. Updated
    # after each cycle's fold from that cycle's record-stream fit, so the fold only uses a
    # rate estimated from earlier records: causal by construction.
    _kcoh_rates = {}
    # The telemetry-walk estimators' last results, served between runs (estimator_every_s;
    # the next-run time is _est_next below).
    _est_last = {"pcn0": None, "kcoh": None}

    # ── The DLL stage's per-cycle products ──────────────────────────────────────────────
    # What fleetdll.stage_fleet_dll computes and later stages read. None means not measured:
    # with --dll-gain 0 the stage does nothing and they keep these initial values, which
    # readers treat as empty. An owner object because attribute assignment needs no
    # `nonlocal`, so the stages that set these can live in other modules.
    class _DllProducts(object):
        """Per-cycle diagnostic products of the DLL stage. None means not measured."""

        __slots__ = (
            "fcoh",
            "kcoh",
            "fcoh_n2",
            "spec_fit",
            "innov_pub",
            "report",
            "deep_gate_eff",
            "run_est",
            "run_pcn0",
            "fleet",
            "pcn0",
            "inst_hops",
            "admit_disp",
            "trk",
            "fadr",
        )

        def __init__(self):
            self.fcoh = None
            self.kcoh = None
            self.fcoh_n2 = None
            self.spec_fit = None
            self.innov_pub = None
            self.report = None
            self.deep_gate_eff = None
            self.run_est = False
            self.run_pcn0 = False
            self.pcn0 = None
            # prn -> the tracker's code residual this cycle (trkresid); {} = not measured
            self.trk = {}
            # prn -> the fleet ADR (fleetadr); {} = no telemetry
            self.fadr = {}
            # instance -> newest telemetry hop seen this cycle; the instance-stall and
            # axis-freshness watches read it after the polls have filled it.
            self.inst_hops = None
            # --presence-admit-displaced thresholds, bundled once per cycle so every
            # presence arm judges with the same numbers; None = the flag is off.
            self.admit_disp = None
            # The cycle's central state: prn -> the fleet DLL's per-satellite row (present, q,
            # disc, hop, ...), read by nearly every stage. As an attribute of its producer, a
            # stage's local of the same name cannot overwrite it.
            self.fleet = None

    _dllp = _DllProducts()

    # ── The dead-reckon stage's per-cycle products ──────────────────────────────────────
    # The working set of deadreckon.stage_dead_reckon and its sub-stages, on an owner object
    # for the same reason as _dllp.
    class _DrProducts(object):
        """Per-cycle derived values of the dead-reckon stage. None means not yet solved."""

        __slots__ = (
            "clk_now",
            "raw_clk",
            "pd",
            "pd2",
            "transit_pd",  # the geometry-only sky the transit freezes read
            "offs",
            "la",
            "tag",
            "drift",
            "t_code",
            "t_fc_abs",
            "rx_sib",
            "hold",
            "mod",
            "slew_cap",
            "slew_k",
            "now_w",
            "t_eph_age",
            "t_now_abs",
        )

        def __init__(self):
            # The receiver clock (chips) propagated to now_w; dr_seed computes it, and
            # dr_joint_clk may move it onto the joint clock for this pass.
            self.clk_now = None
            self.transit_pd = None
            # The raw clock solve, before the quality gate accepts or rejects it.
            self.raw_clk = None
            # pd / pd2: the per-satellite model predictions at now_w and now_w + 2 s; offs: the
            # per-satellite code offsets the clock is solved from; la: the (l-a) receiver
            # code-rate bias; the rest are per-pass constants and flags the sub-stages share.
            self.pd = None
            self.pd2 = None
            self.offs = None
            self.la = None
            self.tag = None
            self.drift = None
            self.t_code = None
            self.t_fc_abs = None
            self.rx_sib = None
            self.hold = None
            self.mod = None
            self.slew_cap = None
            self.slew_k = None
            # The wall instant this dead-reckon pass propagates to, read by every sub-stage:
            # _now(), the cycle's frozen clock (the same instant as t0).
            self.now_w = None
            # Capture age (s since sample 0) at which the ephemeris was evaluated, on the wall
            # axis: now_w - utc0_sample0. cp_predicted propagates range from this, never from
            # t_now_abs: under --dr-fengine-axis those are different clocks, and mixing them
            # misplaces each satellite by range rate x telemetry lag.
            self.t_eph_age = None
            # Axis-derived absolute capture time (s since sample 0), produced by the dead-reckon
            # stage and read by the rate feeds, the joint shadow and the tracker residuals.
            # None, never 0.0, until the stage first has a model: a missing axis time is
            # unknown, a confident wrong timestamp is worse than a skipped measurement, and
            # every consumer must guard on `is not None`.
            self.t_now_abs = None

    _drp = _DrProducts()

    def cp_predicted(v, t_abs):
        """Physical code phase (chips) of the predicted signal at capture age
        t_abs, excluding the receiver clock. The range is carried linearly
        (range_rate) from the epoch the ephemeris was evaluated at
        (_drp.t_eph_age, the hop-axis now of that cycle: deadreckon.dr_eph_epoch),
        never from a wall-clock reading. The neglected range acceleration a (up to
        ~0.2 m/s^2) adds 0.5*a*(t_abs - t_eph_age)^2 of range: at the peak,
        about 1 m for a detection 3 s old and 360 m for one 60 s old. All mod
        arithmetic is on small numbers (t0m is the sample-0 GPST pre-reduced
        mod _drp.t_code)."""
        t_tx = (
            dr_state["t0m"]
            + t_abs
            - (v["range_m"] + v["range_rate_mps"] * (t_abs - _drp.t_eph_age))
            / dr_eph_mod.C_LIGHT
            + v["sat_clk_s"]
        )
        return (t_tx % _drp.t_code) * args.chip_rate_hz

    def _track_ok(_p):
        """Is the tracker seeing this satellite now?

        joint_min_snr gates on the search SNR carried in `best`, which asks
        whether there was a detection: a detection keeps its SNR until the
        search next scans that PRN, so a satellite that has left the beam can
        carry its last good SNR while its tracker output is noise. The joint
        measurement depends on the tracker's own view (joint_min_deep_snr,
        joint_min_coh_frac).

        deep_snr counts only when floor-cleared (coherence_s > 0), matching
        sig_of(): a floored deep sits near 7 on pure noise."""
        _row = _ctx.status.get(_p) or {}
        if float(_row.get("coherence_s", 0) or 0) <= 0.0:
            return False
        return (
            float(_row.get("deep_snr", 0) or 0) >= args.joint_min_deep_snr
            and float(_row.get("coh_frac", 0) or 0) >= args.joint_min_coh_frac
        )

    fe_axis = [
        None
    ]  # (newest pow_hop, wall time it last advanced): the raw F-engine axis sample
    # ── The filtered axis offset ──────────────────────────────────────────────────────
    # fe_off[0] is the (hop time - wall) offset that t_now_abs is built from under
    # --dr-fengine-axis. fe_axis re-samples the newest hop every poll, so the raw offset
    # carries the pipeline lag's jitter, which would land in every re-birth as lag x range
    # rate. The axis is drift-free, so the offset is a constant minus a strictly positive,
    # fluctuating lag, and a max filter recovers the constant: adopt any larger offset at
    # once (a fresher sample saw less lag) and decay slowly downward (0.5 ms/s, following
    # real drift and NTP slew). A disagreement over 2 s (an F-engine restart moves the axis)
    # snaps the filter only after it repeats at the same value for 3 polls: the newest hop is
    # a max over the chain's current status rows, so when the freshest satellite drops out
    # of one poll it falls back by seconds, and a dropout does not persist where a frame0
    # step does. An offset in the future of the wall clock is rejected outright.
    # [filtered offset, wall of last update, consecutive-disagree count, candidate offset]
    fe_off = [None, 0.0, 0, 0.0]
    _axis_utc0 = [0.0]  # cached frame0: the ingestion bound below must never go inert
    # while the anchor re-fetches (utc0_sample0 is a run constant)
    _rf_last = [0.0]  # cycle time of the last RF-health poll (rate-limits it)
    # Next run of the telemetry-walk estimators. The first run is offset per chain to spread
    # the chains' walks over different cycles.
    _est_next = [_now() + (hash(chain_id) % 5) * 8.0]
    _anchor_chk = [
        0.0,
        0,
    ]  # [wall time of the last anchor re-read, consecutive mismatches]

    cp_held = (
        set()
    )  # PRNs whose cp anchor is held (locked: the DLL owns the residual; see hold_snr)
    # None until the dead-reckon stage first has a model (see _DrProducts.t_now_abs).
    _drp.t_now_abs = None

    last_dets = []  # most recent raw /get_detections, re-served by the publisher so
    # the viewer has one origin for both search and combiner data
    dr_untrusted = {}  # prn -> reason: the model is wrong for this sat; use the search
    dr_bad = (
        {}
    )  # prn -> consecutive model-health failures (persistence, not a hair trigger)
    innov_hist = (
        {}
    )  # prn -> [(t, innov_chips)]: the detection innovation, served via innov_pub
    minnov_hist = {}  # prn -> [(t, minnov_chips)]: the model innovation (detection vs
    # the joint state's clk + b_sat + tau, prior-state); served, and the
    # model-primacy flip gate's input
    mp_flipped = (
        set()
    )  # PRNs currently model-primary (seeds from the dr-slew path; detections
    # feed only the filter, the innovations and the referee). Written only
    # by instruments.instr_model_primacy; read by the detection and
    # dead-reckon eligibility guards.
    mp_last_det = {}  # prn -> t of the last detection seen while flipped (starve exit)
    mp_cooldown = {}  # prn -> t of the last exit: no re-entry for 300 s, since a
    # re-entry within minutes is churn, not information
    rrate_prev_hop = {}  # the rrate feed's continuity state: last pow_hop per prn
    rrate_prev_val = {}  # ... and last rate residual (slew gate: catches f_ref re-pins)
    rr_kcoh_fed = {}  # {"last": <the kcoh dict object last fed>}: the estimator is
    # throttled (dllp.run_est), so _est_last serves the same dict for
    # several cycles; re-feeding it would count one measurement as
    # many and the filter would grow confident on repetition.
    rate_unit_hop = [
        0
    ]  # [emit spacing in hops], learned: see rate_residuals' continuity gate
    det_fresh = (
        {}
    )  # prn -> (ref_hop, epoch) of the last new detection; epoch = its capture time
    # (the cycle time while the F-engine anchor is unknown), so freshness is data age
    dop_rate_fitted = {}  # prn -> the fitted rate actually seeded (for the log)
    dop_rate_rejected = (
        {}
    )  # prn -> (fitted, model) when the fit disagreed with the model
    cp_rate_rejected = {}  # prn -> (fitted, pooled-clock) chips/s cross-check

    def cp_to_seed_currency(pts, dop_seed, dop_rate=0.0):
        """Re-express search cp0 points in the current seed's Doppler currency.

        The search anchors cp0 to absolute sample 0 through its own measured Doppler:
        cp0 = cp_local - t_abs*f_chip*(sign*dop_det/f_carrier). That projection multiplies
        per-detection Doppler noise (~10-25 Hz between scans) by the full run age (~0.65
        chips per Hz per 1000 s at L1), so the fit's input scatter, and its slope noise,
        would grow linearly with run time. The tracker's replica applies the same projection
        from the seed's Doppler, so the one consistent currency is dop_seed: adding back each
        point's own drift term (exact cancellation, the same numbers the search subtracted)
        and re-projecting with dop_seed leaves noise that never multiplies t_abs. The fitted
        slope keeps the residual convention (l-a = slope/f_chip).

        dop_rate (Hz/s) subtracts the known sky-Doppler curvature from each point in local
        baseline coordinates (dt = t_i - t_anchor, anchored at the latest point):
        -0.5*k*dop_rate*dt^2, k = f_chip*sign/f_carrier. Without it, the true Doppler's drift
        across a fit baseline T leaves a quadratic whose LS slope reads as a phantom
        (l-a) = |dop_rate|*T/(2*f_carrier). Do not instead re-project each point at a
        drifting Doppler model: (dd_i - dop_model(t_i)) rides the absolute t_abs multiplier,
        so a small model error becomes a large slope error. dop_seed is one scalar per call
        so that t_abs cancels in the slope; only the local dt^2 term (bounded by T^2) is
        safe to correct."""
        out = []
        h_anchor = pts[-1][0] if pts else 0
        for hh, cc, dd in pts:
            t_abs = hh / args.hops_per_sec
            corr = (
                t_abs
                * args.chip_rate_hz
                * (args.code_doppler_sign * (dd - dop_seed) / args.carrier_hz)
            )
            dt = (hh - h_anchor) / args.hops_per_sec
            corr -= (
                0.5
                * args.chip_rate_hz
                * args.code_doppler_sign
                * dop_rate
                / args.carrier_hz
                * dt
                * dt
            )
            out.append((hh, (cc + corr) % CODE_LEN))
        return out

    def sig_of_last(rec):
        """Hold-on-lock gate metric (against hold_snr) from the previous cycle's combiner
        status: the incoherent amp_snr, or the larger of amp_snr and deep_snr when the deep
        is floor-cleared. The combiner reports coherence_s > 0 only when a ladder rung beat
        its rectification floor by 2x, so that certifies the deep as a coherent detection;
        an uncertified deep_snr sits at the nav-wipe rectification floor (~7) off peak and
        would freeze bad anchors. Certified deep lets sharp-ACF (BOC) signals reach hold:
        seed jitter turns into per-record amplitude variance on their sharp peak, which
        suppresses the moment-debiased amp_snr below every gate while the coherent sum
        stays strong."""
        if not rec:
            return 0.0
        amp = float(rec.get("amp_snr", 0) or 0)
        if float(rec.get("coherence_s", 0) or 0) > 0.0:
            return max(amp, float(rec.get("deep_snr", 0) or 0))
        return amp

    # The receiver clock-frequency bias (gnss_broker/clockbias.py). Its invariant: a solved
    # bias unmeasured for bias_stale_s widens the search rather than narrowing it.
    _cb = ClockBias()
    _cb.cal = None  # calibration value, Hz (drift-alarm reference)
    _cb.meas_t = _now()  # last multi-sat bias measurement (stale-rescue clock;
    # every solve re-stamps it)
    _cb.stale = False  # solved but unmeasured for > bias_stale_s
    _cb.available = False  # is a solved bias in hand?
    # Nav-decode health export (--decode-health-file): one JSON per chain for the viewer.
    dhw = None
    if args.decode_health_file:
        try:
            import decode_health

            dhw = decode_health.DecodeHealthWriter(
                args.decode_health_file,
                chain=os.path.basename(args.decode_health_file).rsplit(".", 1)[0],
                sys=alm_sys,
                log=_log,
            )
            _log(
                "decode-health export -> %s (sys %s)"
                % (args.decode_health_file, alm_sys)
            )
        except Exception as e:
            _log("decode-health export DISABLED: %s" % e)
            dhw = None
    # cnav runs in a GPS band broker; label by carrier so L2C and L5 stay distinct in the file.
    _nav.cnav_sig = (
        "GPS_L2C_CNAV"
        if abs((args.carrier_hz or 0) - 1227.6e6) < 5e6
        else "GPS_L5_CNAV"
    )

    def _dh_obs(sig, prn, h, eph_obj, xc):
        """One decode-health observation, uniform across decoders. `eph_obj` is the extracted
        ephemeris object (None until this decoder has a full set); `xc` the BRDC-xcheck suffix
        (dpos parsed out). count is whatever monotonic decode counter the health dict carries.
        Never raises."""
        if dhw is None:
            return
        try:
            cnt = (
                h.get("words")
                or h.get("decoded")
                or h.get("decoded_sf")
                or h.get("pages")
                or 0
            )
            syn = h.get("synced")
            if syn is None:
                syn = (h.get("words") or 0) > 0 or eph_obj is not None
            dhw.observe(
                sig,
                prn,
                _now(),
                count=cnt,
                synced=bool(syn),
                eph=(eph_obj is not None),
                dpos_m=_dh_dpos(xc),
            )
        except Exception:
            pass

    # ---- Decoded-ephemeris BRDC fallback (decoded_eph.py, decoded_eph_fallback). Registry, built
    # once: (signal, sys, sv_position fn, toe field, decoder getter) per decoder this broker may
    # run. The getters read _nav at call time, since the nav-bit stage creates decoders lazily.
    _decfb = None
    if args.decoded_eph_fallback:
        try:
            import decoded_eph as _decfb
            from galileo_inav import sv_position_inav as _svp_inav
            from galileo_fnav import sv_position_fnav as _svp_fnav
            from gps_cnav import sv_position_cnav as _svp_cnav
            from gps_nav_decode import sv_position_lnav as _svp_lnav
            from beidou_bcnav1 import sv_position_bcnav1 as _svp_bc1
            from beidou_bcnav2 import sv_position_bcnav2 as _svp_bc2
        except Exception as e:
            _log("decoded-eph fallback DISABLED (import: %s)" % e)
            _decfb = None
    _dec_reg = (
        [
            ("GPS_L1_LNAV", "G", _svp_lnav, "toe", lambda: _nav.navbits)
            if _decfb
            else None,
            ("GAL_E1B_INAV", "E", _svp_inav, "t0e", lambda: _nav.inav)
            if _decfb
            else None,
            ("GAL_E5AI_FNAV", "E", _svp_fnav, "t0e", lambda: _nav.fnav)
            if _decfb
            else None,
            (_nav.cnav_sig, "G", _svp_cnav, "toe", lambda: _nav.cnav)
            if _decfb
            else None,
            ("BDS_B1C_BCNAV1", "C", _svp_bc1, "t_oe", lambda: _nav.bcnav1)
            if _decfb
            else None,
            ("BDS_B2A_BCNAV2", "C", _svp_bc2, "t_oe", lambda: _nav.bcnav2)
            if _decfb
            else None,
        ]
        if _decfb
        else []
    )
    _dec_reg = [r for r in _dec_reg if r]

    def _decoded_entries(now_w):
        """Build decoded_eph entries (sys, prn, signal, eph, sv_pos, toe_gpst) from whichever
        decoders exist. toe_gpst is the eph's time-of-week field placed within half a week of
        _drp.now_w. Never raises: a bad decoder or eph is skipped."""
        ents = []
        if not _decfb:
            return ents
        gpst_now = _decfb.gpst_of_utc(
            datetime.fromtimestamp(_drp.now_w, tz=timezone.utc)
        )
        wk = int(gpst_now // 604800)
        for signal, sys, svp, toef, getter in _dec_reg:
            try:
                dec = getter()
                if dec is None:
                    continue
                for prn in list(dec._p):
                    try:
                        e = dec.ephemeris(prn)
                        if e is None or toef not in e:
                            continue
                        toe_gpst = wk * 604800 + e[toef]  # nearest-week fold
                        if toe_gpst - gpst_now > 302400:
                            toe_gpst -= 604800
                        elif toe_gpst - gpst_now < -302400:
                            toe_gpst += 604800
                        ents.append((sys, prn, signal, e, svp, toe_gpst))
                    except Exception:
                        continue
            except Exception:
                continue
        return ents

    _decfb_log_t = [0.0]
    CODE_LEN = float(args.code_length)
    # The long/overlaid code the trackers despread, in primary periods and in seconds (L2C CL:
    # 75, 1.5 s; GPS L5 Q5 with NH20: 20, 0.02 s). The time-assist in seeding.py computes which
    # period a seed's cp sits in rather than searching it, so these must match the trackers'
    # signal, not the search's: a seed placed on the 10230-chip primary of a 204600-chip NH code
    # is right 1 time in 20.
    LC_SEG = int(args.long_code_segments)
    LC_EPOCH = float(args.long_code_epoch_s)
    # The hold's Doppler fence (seeding.py): while a held seed's fresh Doppler stays within this
    # of the frozen one the whole tuple stays frozen; beyond it the move is translated into the
    # new currency. 0.1 cycle per record of stale replica carrier is what the single-record
    # despread tolerates, so the fence scales with the record period: 100 Hz for 1 ms records,
    # 25 Hz for E1C's 4 ms, 10 Hz for B1C's 10 ms.
    args.hold_max_dop_hz = 0.1 * args.chip_rate_hz / CODE_LEN
    _log(
        "hold_max_dop_hz (derived): %.1f Hz (0.1 cycle per %.0f ms record)"
        % (args.hold_max_dop_hz, 1e3 * CODE_LEN / args.chip_rate_hz)
    )
    # Safety net, not a fence: clamp how far the seed Doppler may move in one cycle. A real MEO
    # Doppler moves under ~1 Hz/s, so this fires only on a bad model. It must sit well above the
    # fence: a tighter clamp would pull every step back inside the fence, where the freeze keeps
    # the old Doppler, and the seed Doppler would never update.
    args.dop_max_rate_hz = max(5.0, 3.0 * args.hold_max_dop_hz)
    _log(
        "dop_max_rate_hz (derived safety net, one cycle): %.1f Hz -- fires only on a "
        "bad model"
        % args.dop_max_rate_hz
    )
    # Reset the cp-fit history across a detection gap longer than fit_gap_s (a re-acquisition).
    # Set in time, not hops: a fixed hop count would scale with the band's hop rate.
    MAX_GAP_HOPS = args.fit_gap_s * args.hops_per_sec
    HIST_LEN = args.fit_hist_len  # snapshots kept for the slope fits (--fit-hist-len)
    _log(
        "detectors=%d trackers=%d combiner=%s interval=%.2fs gating=%s"
        % (len(detectors), len(trackers), combiner, args.interval, gating)
    )
    _log(
        "trackers: %s"
        % (
            trackers
            if len(trackers) <= 6
            else "%s ... %s (%d)" % (trackers[0], trackers[-1], len(trackers))
        )
    )

    # Seed currency audit state: the last shipped tuple per PRN (the payload fields
    # tracker_phase_at reads), so each POST can be compared with the previous one in physical
    # units.
    seed_audit_prev = {}
    # Broker start, for the joint feed warmup (joint_feed_warmup_s) and the sawtooth detector's
    # startup hold-off. Wall clock on purpose: the warmup guards against establishment-phase
    # measurements, and establishment runs on wall time whatever the transcript clock replays.
    broker_t0 = time.time()

    # ── The stage interface (gnss_broker/context.py) ──────────────────────────────────────
    # Every name a stage reads from main() that is not its own state. Stable names are bound
    # once and mutated in place, never rebound, so the context can hold them by reference;
    # per-cycle names are rebound on the context each pass.
    def sig_of(r):
        # deep_snr counts only when floor-cleared (coherence_s > 0): a floored deep (~7 on
        # noise) would keep phantom coasts alive.
        amp = float(r.get("amp_snr", 0) or 0)
        if float(r.get("coherence_s", 0) or 0) > 0.0:
            return max(amp, float(r.get("deep_snr", 0) or 0))
        return amp

    _ctx = ChainContext(
        args=args,
        band_id=band_id,
        chain_id=chain_id,
        code_len=CODE_LEN,
        telem_chain=telem_chain,
        base=base,
        alm_sys=alm_sys,
        alm_min_prn=alm_min_prn,
        lc_seg=LC_SEG,
        lc_epoch=LC_EPOCH,
        rx=rx,
        publisher=publisher,
        telem_client=telem_client,
        detectors=detectors,
        dll_combiners=dll_combiners,
        spectrum_endpoints=spectrum_endpoints,
        n2_combiners=n2_combiners,
        last_dets=last_dets,
        decfb=_decfb,
        decfb_log_t=_decfb_log_t,
        dr_bad=dr_bad,
        fe_axis=fe_axis,
        fe_off=fe_off,
        innov_hist=innov_hist,
        minnov_hist=minnov_hist,
        p2c=p2c,
        dop_rate_fitted=dop_rate_fitted,
        dop_rate_rejected=dop_rate_rejected,
        cp_rate_rejected=cp_rate_rejected,
        dll_hop_window=dll_hop_window,
        deep_gate=_deep_gate,
        dg_auto_last=_dg_auto_last,
        est_next=_est_next,
        sig_of=sig_of,
        capable=_capable,
        capable_t=_capable_t,
        alm_now=_alm_now,
        cb=_cb,
        almanac_sats=almanac_sats,
        brdc_alm=brdc_alm,
        det_fresh=det_fresh,
        nho=_nho,
        dls=_dls,
        hold=_hold,
        cpt=_cpt,
        rf=_rf,
        nav=_nav,
        qpop=_qpop,
        brown=_brown,
        saw=_saw,
        prnmap=_prnmap,
        trackers=trackers,
        joint_consume=joint_consume,
        broker_t0=broker_t0,
        dr_eph_mod=dr_eph_mod,
        dr_min_prn=dr_min_prn,
        hist_len=HIST_LEN,
        max_gap_hops=MAX_GAP_HOPS,
        cp_to_seed_currency=cp_to_seed_currency,
        dh_obs=_dh_obs,
        cp_predicted=cp_predicted,
        joint_state=_joint_state,
        track_ok=_track_ok,
        p2c_tick=_p2c_tick,
        p2c_hold=_p2c_hold,
        decoded_entries=_decoded_entries,
        sig_of_last=sig_of_last,
        dllp=_dllp,
        drp=_drp,
        seeds=seeds,
        dr_state=dr_state,
        bsat=bsat,
        cp_held=cp_held,
        dr_untrusted=dr_untrusted,
        est_last=_est_last,
        kcoh_rates=_kcoh_rates,
        rf_last=_rf_last,
        elem_arch_t=_elem_arch_t,
        elem_poll_t=_elem_poll_t,
        geom_post_t=_geom_post_t,
        mp_cooldown=mp_cooldown,
        mp_flipped=mp_flipped,
        mp_last_det=mp_last_det,
    )

    while True:
        # Sample the frozen cycle clock once (transport._Transcript): every _now() in this pass
        # returns t0, so every gate, age and EMA evaluates at one instant rather than across
        # the pass's own processing time.
        t0 = _TR.tick()
        _ctx.begin_cycle(t0=t0)
        t_wall = time.time()  # real clock, for the pass's busy time and the sleep
        # 1. collect best-SNR detection per PRN across all detection sources
        best = {}  # prn -> (snr, dop, cp, ref_hop, nh, cp_long, cp_at_ref)
        _ctx.begin_cycle(best=best)
        for d_ep in detectors:
            try:
                dets = _get("%s/get_detections" % d_ep)
                if dets:
                    last_dets[:] = dets  # published verbatim (see FleetPublisher)
            except Exception as e:
                _log("get_detections %s failed: %s" % (d_ep, e))
                continue
            for d in dets:
                prn, snr = int(d["prn"]), float(d["snr"])
                if snr < args.acquire_snr:
                    continue
                if prn not in best or snr > best[prn][0]:
                    best[prn] = (
                        snr,
                        float(d["doppler_hz"]),
                        float(d["code_phase_chips"]),
                        int(d.get("ref_hop", 0)),
                        int(d.get("nh", -1)),
                        float(d.get("code_phase_long_chips", -1.0)),
                        float(d.get("code_phase_at_ref_chips", -1.0)),
                    )
        # --detectors-monitor: the detections are kept for the record (stage_dead_reckon's
        # search_monitor) and withheld from every consumer that would act on them. `best` is
        # the dict begin_cycle handed the context, so it is emptied in place.
        _ctx.monitor_best = {}
        if args.detectors_monitor and best:
            _ctx.monitor_best = dict(best)
            best.clear()
        # Record each PRN's reported nh alignment and the hop it was measured at; the nh hint
        # (--nh-hint, searchhint.py) learns the receiver's alignment offset from them. Gated on
        # acquire_snr like the detection itself: a marginal detection's nh is a coin flip, and a
        # wrong offset narrows the scan away from the truth until the hint expires. A monitored
        # detection teaches the hint too: that narrows the search, it acts on nothing.
        for _p, _b in list(best.items()) + list(_ctx.monitor_best.items()):
            if _b[4] >= 0 and _b[3] > 0:
                _nho.seen[_p] = (_b[4], _b[3])
        for _p in list(_nho.seen):
            if _p not in seeds and _p not in best and _p not in _ctx.monitor_best:
                del _nho.seen[_p]  # neither seeded nor detected: forget it

        for _p, _b in best.items():
            # det_fresh: (ref_hop, stamp) per PRN, restamped when the ref_hop changes (the search
            # re-detected it; its REST table keeps serving old ones). The clock-bias solve and
            # the time-base check use only detections with t0 - stamp < bias_det_fresh_s: at
            # 0.4 Hz/s, 100 s of staleness fakes a 40 Hz mismatch. Stamp with the detection's
            # own epoch (ref_hop through the capture anchor), not the wall clock of first sight,
            # so (t0 - stamp) is the true data age even on a freshly restarted broker, which
            # would otherwise certify every entry in the REST table as new. Without the anchor
            # (utc0_sample0 == 0) use the cycle clock.
            if det_fresh.get(_p, (None,))[0] != _b[3]:
                t_det = (
                    _ctx.utc0_sample0 + _b[3] / args.hops_per_sec
                    if _ctx.utc0_sample0
                    else t0
                )
                det_fresh[_p] = (_b[3], t_det)

        _cb.value = _cb.ema if _cb.ema is not None else 0.0
        _cb.available = _cb.ema is not None

        # 2. Orbit-predicted Doppler and visibility (almanac.stage_almanac_predict)
        _ctx.pred = {}  # prn -> (doppler_hz, rate_hz_s, elev_deg, ...) [sign-applied]
        _ctx.pred_dark = (
            {}
        )  # the same rows for capability-excluded satellites (--probe-dark)
        # Stale-bias rescue (bias_stale_s): a solved bias unmeasured that long widens the search
        # margins until re-solved. If it latched away from truth, the narrow hints it centres
        # would prevent the measurements that fix it. The held value still centres the wide
        # hints (and the seeds under --seed-bias-source ema).
        _cb.stale = (
            args.bias_stale_s > 0.0
            and _cb.ema is not None
            and t0 - _cb.meas_t > args.bias_stale_s
        )
        if _cb.stale:
            _log_rl(
                "clkstale",
                "CLOCK BIAS STALE: no multi-sat measurement for %.0f s (holding %+.0f Hz "
                "%s) -- margins wide until re-solved"
                % (
                    t0 - _cb.meas_t,
                    _cb.value,
                    "for the hints; seeds ride 0 Hz"
                    if args.seed_bias_source == "zero"
                    else "for seeding",
                ),
                every_s=60.0,
            )
        _ctx.up = None
        almanac_stage.stage_almanac_predict(_ctx)

        # 2b. Narrow the search (--narrow-search): push each predicted, visible PRN's Doppler to
        # the detectors, so each scans doppler +- margin instead of its blind grid; that is what
        # makes acquiring an undetected sat affordable. The margin is wide until the common
        # clock-frequency bias is solved (the geometric Doppler is offset by the unknown
        # oscillator error) and while it is stale, narrow otherwise.
        searchhint.stage_narrow_search(_ctx)

        # Per-cycle outputs of stage_detections_to_seeds (below), which refreshes or adds seeds:
        # code phase from the search, Doppler from the orbit prediction when available, else the
        # search grid.
        _ctx.la_samples = (
            []
        )  # per-sat (l-a) estimates this cycle, from sats with a good code-rate fit
        _ctx.fitted = set()  # PRNs that got their own >=3-snapshot slope fit this cycle
        _ctx.cl_report = (
            []
        )  # per-PRN long-code (overlay period) lift log lines this cycle
        # Capture time anchor: UTC of sample 0, the F-engine's frame 0 (--time0-endpoint), 0.0
        # until it answers. Fetched once and latched (the `not _ctx.utc0_sample0` fetch below).
        # An F-engine restart re-establishes frame 0 silently: every seed is then wrong by the
        # epoch change, which looks like a tracking or geometry problem rather than a stale
        # anchor. So re-read it every 60 s and log a mismatch; after _ANCHOR_STRIKES consecutive
        # mismatches, exit. Never re-anchor in place: every node cached the old frame 0 at its own
        # startup. _now(), not now_w: _drp.now_w is set later in the cycle.
        _now_anchor = _now()
        if (
            args.time0_endpoint
            and _ctx.utc0_sample0
            and _now_anchor - _anchor_chk[0] > 60.0
        ):
            _anchor_chk[0] = _now_anchor
            try:
                _fresh = (
                    float(
                        _get("%s/%s" % (base, args.time0_endpoint.strip("/"))).get(
                            "time0_ns", 0.0
                        )
                    )
                    / 1e9
                )
                if _fresh and abs(_fresh - _ctx.utc0_sample0) > 1e-3:
                    _anchor_chk[1] += 1
                    _log(
                        "*** TIME ANCHOR CHANGED (%d/%d): frame0 was %.9f, endpoint now reports "
                        "%.9f (%+.3f days). The F-engine has been restarted. Every seed this "
                        "broker sends is wrong by that amount."
                        % (
                            _anchor_chk[1],
                            _ANCHOR_STRIKES,
                            _ctx.utc0_sample0,
                            _fresh,
                            (_fresh - _ctx.utc0_sample0) / 86400.0,
                        )
                    )
                    if _anchor_chk[1] >= _ANCHOR_STRIKES:
                        # Exit, do not re-anchor. frame0 is latched once per process here and
                        # in every node, and the seeds, clock solution, phase history and record
                        # windows are all built on it; only the startup path builds coherent
                        # state from an anchor.
                        # os._exit, not sys.exit: chains are threads (scripts/gnss/broker_multi.py
                        # catches SystemExit per chain), so sys.exit would stop only this chain,
                        # logged as CHAIN REFUSED TO START, and leave the others on the old epoch.
                        _log(
                            "*** STOPPING THE BROKER after %d confirmations, so frame0 is "
                            "re-latched by the startup path. Its supervisor (systemd "
                            "Restart=on-failure) relaunches it; the nodes are stopped by "
                            "their own fpga_monitor on this same event and come back on the "
                            "new epoch." % _anchor_chk[1]
                        )
                        try:
                            sys.stdout.flush()
                            sys.stderr.flush()
                        except Exception:
                            pass
                        os._exit(_EXIT_ANCHOR_CHANGED)
                elif _fresh:
                    _anchor_chk[
                        1
                    ] = 0  # one odd read must not accumulate into a restart
            except Exception:
                pass  # endpoint down is the normal outage case; the next re-read retries

        if dr_state is not None and not _ctx.utc0_sample0 and args.time0_endpoint:
            try:
                # CHORD's frame 0 is GPS-disciplined, so this is exact rather than an
                # estimate; time0_ns is the absolute time of fpga_seq_num 0. Fetched through
                # the Receiver: frame 0 belongs to the instrument, not a signal, so the
                # process fetches it once for all its chains, which could otherwise straddle
                # an F-engine restart and disagree.
                _ctx.utc0_sample0 = (
                    rx.time_anchor(
                        lambda: float(
                            _get("%s/%s" % (base, args.time0_endpoint.strip("/"))).get(
                                "time0_ns", 0.0
                            )
                        )
                        / 1e9,
                        chain_id,
                    )
                    or 0.0
                )
                if _ctx.utc0_sample0:
                    _log(
                        "time anchor: CHORD F-engine frame0 = %.9f s (GPS-disciplined)"
                        % _ctx.utc0_sample0
                    )
            except Exception as e:
                _log("time anchor unavailable (%s); retrying" % e)
        _ctx.dr_pd = (dr_state or {}).get("pd") or {}
        _ctx.dr_pd2 = (dr_state or {}).get("pd2") or {}
        _ctx.dr_pd0 = (dr_state or {}).get("pd0") or {}
        # The transit freezes' geometry-only sky; the strict sky stands in until the first predict.
        _ctx.transit_pd = (dr_state or {}).get("transit_pd") or _ctx.dr_pd
        seeding.stage_detections_to_seeds(_ctx)

        # 3. Coast / drop (seeding.stage_coast_drop). A visible sat that was not re-detected is
        # coasted through a dropout (radar sweep, brief fade): its seed is held and its Doppler
        # forecast forward from the orbit and clock, so the tracker keeps despreading on the
        # predicted trajectory and re-peaks when the signal returns. Drop only when the sat sets
        # or its lock metric stays below the bar for the whole coast budget; a lock reading
        # resets the coast.
        _ctx.coast_polls = max(
            1, int(round(args.coast_budget / max(args.interval, 1e-3)))
        )
        try:
            _ctx.status = {int(r["prn"]): r for r in _get("%s/get_status" % combiner)}
            # The newest F-engine hop, captured at fetch time. fe_axis pairs it with the wall
            # time it last advanced, so the dead-reckon stage builds its "now" on the F-engine
            # axis with wall time entering only as the elapsed time since.
            _fh = max(
                (float(r.get("pow_hop") or 0.0) for r in _ctx.status.values()),
                default=0.0,
            )
            # Bound at ingestion: a pow_hop in the future of the wall clock is never the axis
            # (staleness >= 0); birth rows echo the forecast-epoch seed hop. fe_axis,
            # fe_hop_now and fe_off all take this max, so one bound here covers every consumer.
            # On a violation fall back to the newest row that is not in the future.
            if _ctx.utc0_sample0:
                _axis_utc0[0] = float(_ctx.utc0_sample0)
            if _axis_utc0[0] > 0.0 and _fh > 0.0:
                _fh_bound = (_now() - _axis_utc0[0] + 0.05) * args.hops_per_sec
                if _fh > _fh_bound:
                    _fh_ok = max(
                        (
                            h
                            for h in (
                                float(r.get("pow_hop") or 0.0)
                                for r in _ctx.status.values()
                            )
                            if h <= _fh_bound
                        ),
                        default=0.0,
                    )
                    _log_rl(
                        "fe-ingest-future",
                        "fe-axis: newest pow_hop is %.2f s in the FUTURE of the wall "
                        "clock -- a forecast-epoch or corrupt row; dropped from the "
                        "axis (falling back to the newest sane row, %.2f s stale)."
                        % (
                            _fh / args.hops_per_sec - (_now() - _axis_utc0[0]),
                            (_now() - _axis_utc0[0]) - _fh_ok / args.hops_per_sec
                            if _fh_ok > 0.0
                            else float("nan"),
                        ),
                        every_s=60.0,
                    )
                    _fh = _fh_ok
            # Published for the PRN scheduler (prnmap._at_hop). None, not 0.0, when there is no
            # axis: a zero would look like a valid sample at the epoch and schedule every swap
            # in the past.
            _ctx.fe_hop_now = _fh if _fh > 0.0 else None
            # And when it was read, so the scheduler can age it. On the frozen cycle clock the
            # age within this cycle is 0; prn_reconfig_lead_s covers the delay to the POST.
            _ctx.fe_hop_t = _now() if _fh > 0.0 else None
            if _fh > 0.0:
                # Frozen time base (fe_axis_stale_s): this hop comes from the one --combiner, and
                # when its pow_hop stops advancing (a wedged capture window, an F-engine re-base)
                # the measurements stop while the orbit model moves on with the wall clock.
                # Log only: dr_clock_freeze_s holds the clock solve meanwhile.
                # The stamp is the time the hop last advanced, not the time of this poll, or the
                # age could never exceed one cycle. This guard covers the global case:
                # fits.instance_stall_verdict (same stamp rule) refuses to accuse an instance
                # while most of the fleet is also stalled (min_frac_advancing).
                _fh_prev = fe_axis[0]
                if (
                    _fh_prev is not None
                    and _fh <= _fh_prev[0]
                    and args.fe_axis_stale_s > 0.0
                    and _now() - _fh_prev[1] > args.fe_axis_stale_s
                ):
                    _log_rl(
                        "fe-stale",
                        "*** TIME BASE FROZEN: newest telemetry hop has not advanced in "
                        "%.0f s (hop %.0f, combiner %s). t_now_abs is built from this, so "
                        "the receiver clock, every det_age and every model-evaluated seed "
                        "on this chain are now standing still while the sky is not. This "
                        "is an instance stall upstream, not a tracking fault -- check that "
                        "combiner's pow_hop and its node's capture window."
                        % (_now() - _fh_prev[1], _fh, combiner),
                        every_s=60.0,
                    )
                # advanced -> restamp; frozen -> keep the stamp so the staleness accrues
                fe_axis[0] = (
                    (_fh, _now())
                    if (_fh_prev is None or _fh > _fh_prev[0])
                    else (_fh, _fh_prev[1])
                )
                # the filtered offset (see fe_off at its definition)
                _ow = _now()
                _oi = _fh / args.hops_per_sec - _ow
                # An offset in the future of the wall clock is not the axis. The max filter
                # rejects stale samples by construction, but it adopts any upward move under
                # 2.0 s at once and decays only 1.8 s/hour, so one future sample would hold the
                # axis ahead for hours and evaluate every model off-time. Staleness is >= 0, so
                # the offset can never exceed -utc0_sample0 (+0.05 s): reject such a sample from
                # the init, the ratchet and the snap counter. The ingestion bound above already
                # enforces the same limit.
                _utc0 = float(getattr(_ctx, "utc0_sample0", 0.0) or 0.0)
                if _utc0 > 0.0 and _oi > -_utc0 + 0.05:
                    _log_rl(
                        "fe-axis-future",
                        "fe-axis: REJECTED a pow_hop %.2f s in the FUTURE of the wall "
                        "clock (offset %+.3f, physical bound %+.3f) -- a corrupt or "
                        "forecast-epoch row surfaced by max(), never the axis."
                        % (_oi + _utc0, _oi, -_utc0),
                        every_s=60.0,
                    )
                elif fe_off[0] is None:
                    fe_off[0] = _oi
                elif abs(_oi - fe_off[0]) > 2.0:
                    # disagreement: believe it only if it repeats at the same value
                    if abs(_oi - fe_off[3]) < 0.5:
                        fe_off[2] += 1
                    else:
                        fe_off[2] = 1
                    fe_off[3] = _oi
                    if fe_off[2] >= 3:
                        _log(
                            "fe-axis offset SNAP: filtered %+.3f -> %+.3f s after %d "
                            "consecutive polls (a real axis move -- F-engine restart or "
                            "frame0 step; a dropout does not persist)"
                            % (fe_off[0], _oi, fe_off[2])
                        )
                        fe_off[0] = _oi
                        fe_off[2] = 0
                    else:
                        _log_rl(
                            "fe-axis-blip",
                            "fe-axis: instantaneous offset %+.3f s disagrees with the "
                            "filtered %+.3f by %.2f s -- HELD (%d/3). max(pow_hop) fell "
                            "back, i.e. the freshest row dropped out of this poll."
                            % (_oi, fe_off[0], _oi - fe_off[0], fe_off[2]),
                            every_s=60.0,
                        )
                else:
                    fe_off[2] = 0
                    _dec = 0.0005 * max(0.0, _ow - fe_off[1])
                    fe_off[0] = max(_oi, fe_off[0] - _dec)
                fe_off[1] = _ow
        except Exception as e:
            _ctx.status = {}
            _log("get_status failed: %s" % e)
        # Nav-message decode (--nav-bits): feed this cycle's nav_obs to the band's decoder.
        # Rows carry nav_obs only when the combiner exports them.
        navbits.stage_nav_bits(_ctx)
        # Lock metric: the detection significance (sig_of: the noise-debiased amp_snr, or the
        # floor-cleared deep_snr when larger), not the raw |A|. |A| is biased by the noise floor,
        # so judging lock by |A| > drop_amplitude would let phantoms coast forever. Falls back to
        # |A| only when no row reports a significance.
        _ctx.have_sig = any(sig_of(r) > 0 for r in _ctx.status.values())
        # Noise probes (--noise-probes N): keep the N deepest PRNs below -15 deg elevation seeded
        # so the combiner emits signal-free rows for them, the noise reference for the presence
        # gate's floors and the beam map's pedestal; an almanac-gated broker otherwise never
        # tracks one. Probes are exempt from coast/drop and never command a carrier; they fail
        # every lock gate on their own. Doppler and code phase are arbitrary for noise; the
        # predicted Doppler keeps the despread configuration representative.
        # --probe-dark puts the dark satellites (with ephemeris, but excluded by
        # --signal-capability: signal-free at any elevation) ahead of the deepest; prnmap's
        # desired_map orders its probe slots by the same prnmap.probe_order.
        _ctx.probe_set = set()
        if args.noise_probes > 0 and args.almanac and _ctx.pred:
            _dark = (_ctx.pred_dark or {}) if args.probe_dark else {}
            _cands = prnmap.probe_order(
                {p: v[2] for p, v in _ctx.pred.items()},
                _dark,
                held=_prnmap.consensus or (),
            )
            # --probe-require-slot: a probe the nodes hold no slot for is seeded but never
            # reports, which can leave presence short of the >= 3 probes its q floor needs.
            # Pick probes only from the PRNs the nodes hold (prnmap's /get_prns consensus).
            # Fails open: consensus is None until a full sweep lands, while the nodes disagree
            # and always under --prn-reconfig off (no sweep runs), and it is one cycle old
            # because the prnmap stage runs later in the cycle; without it the probes are
            # unfiltered.
            if args.probe_require_slot:
                _held = _prnmap.consensus
                if _held:
                    _drop = [
                        p for p in _cands[: args.noise_probes] if p not in set(_held)
                    ]
                    _cands = [p for p in _cands if p in set(_held)]
                    if _drop:
                        _log_rl(
                            "probe-slot",
                            "PROBE SLOT FILTER: %s have no slot on this chain and would "
                            "never report -- skipped; probes now %s"
                            % (
                                ",".join(str(p) for p in _drop),
                                ",".join(str(p) for p in _cands[: args.noise_probes]),
                            ),
                            every_s=300.0,
                        )
            deep_low = _cands[: args.noise_probes]
            _ctx.probe_set = set(deep_low)
            for p in deep_low:
                _pp = _ctx.pred[p] if p in _ctx.pred else _dark[p]
                if p not in seeds:
                    _log(
                        "noise probe PRN %d seeded (elev %.0f%s)"
                        % (
                            p,
                            _pp[2],
                            "" if p in _ctx.pred else ", dark: not on this signal",
                        )
                    )
                seeds[p] = Seed.born(
                    "probe",
                    epoch=0,
                    doppler_hz=_pp[0] + _cb.value,
                    code_phase_chips=0.0,
                    code_phase_rate=cp_rate_from_code_bias(
                        _cb.code_ema or 0.0, args.hops_per_sec, args.chip_rate_hz,
                    ),
                    ref_hop=0,
                    doppler_rate_hz_s=_pp[1],
                )
        seeding.stage_coast_drop(_ctx)

        # 3e. Dead-reckoned code-phase seeding (--dead-reckon, deadreckon.py). BRDC ephemeris
        # (~2 m orbits, ~5 ns satellite clocks) plus the receiver clock solved from the detected
        # sats predicts every other visible sat's code phase well inside the DLL capture range,
        # so seed them all: sub-threshold sats despread on peak with no detection. The search
        # becomes bootstrap (the clock solve), fallback (a detection re-anchors the seed through
        # the normal seed loop and hands the PRN back from the model-owned set, unless it is
        # model-primary) and integrity check.
        deadreckon.stage_dead_reckon(_ctx)

        # 3c. Delay-lock loop (fleetdll.py, codeloop.py): close the code loop from the
        # combiner's window-averaged E/L powers. disc = (<|E|^2>-<|L|^2>)/(<|E|^2>+<|L|^2>) ~
        # -4*tau at 0.5-chip spacing, with tau = (true - commanded) cp: the correlation triangle
        # gives |E|=R(d+tau), |L|=R(d-tau), so a late true peak strengthens L and drives disc
        # negative; tau_est = -disc/4 * spacing/0.5 (combdll.dll_tau). The per-PRN trim is a
        # leaky integral of gain*tau_est, clamped to +-3 chips and applied to the seed at POST
        # time only: the stored seed stays pure fit/coast state, so the trim converges to the
        # search fit's bias instead of double-counting on coasted seeds. Gated on lock (fleet
        # presence, or sig >= lock_snr), since an unlocked disc is noise. Reset every cycle, so
        # with dll_gain 0 everything downstream, the C++ trim arming included, sees an empty
        # fleet and arms nobody.
        _dllp.fleet = {}
        fleetdll.stage_fleet_dll(_ctx)

        # 3d'. rrate feed (--rrate-state): per-satellite orbital range-rate-error rows (m/s,
        # shared across bands) in the receiver-wide joint state, fed from this chain's carrier
        # rate residuals. Each measurement updates the sat's row and the shared f_carrier
        # through one H; within a band the two are degenerate and separated only by what is
        # shared across satellites. E5a and E5b put each Galileo sat on one row through two
        # carriers 1.0261 apart, so cross-band information adds instead of being fitted twice.
        # Measurement only unless --rrate-command is set. Fed from the uncapped fields:
        # deep_rate_hz is capped at the combiner's DEEP_RATE_MAX_HZ (25 Hz), and past the cap
        # it reports the best in-cap bin, which is noise. _rf.full_ok is False when no row
        # carries the uncapped fields this poll (the status poll failed or came back empty).
        _rf.full_ok = False
        if args.rrate_state:
            _rf.full_ok = any(
                isinstance(_r, dict) and _r.get("deep_rate_full_q") is not None
                for _r in (_ctx.status or {}).values()
            )
            try:
                _rf.resid2, _ = rate_residuals(
                    _ctx.status,
                    args.carrier_rate_min_q,
                    args.carrier_rate_clip_hz,
                    None,
                    prev_hop=rrate_prev_hop,
                    max_gap=args.carrier_rate_max_gap,
                    prev_val=rrate_prev_val,
                    max_step=args.carrier_rate_max_step,
                    unit_hop=rate_unit_hop,
                    rate_field="deep_rate_full_hz",
                    q_field="deep_rate_full_q",
                )
            except Exception as e:
                _rf.resid2 = {}
                _log_rl("jrr-err", "rrate residuals skipped: %s" % e, every_s=300.0)
        else:
            _rf.resid2 = {}
        if args.rrate_state and _rf.resid2:
            # Feed only cohering sats (coherence_s > 0 or coh_frac >= 0.3, the fine feed's
            # test). Strong but decohered sats carry real multi-Hz carrier residuals that swing
            # poll to poll from seed and f_ref churn, not orbit error, and the rrate model is a
            # slow per-sat drift. Never a dark probe (--probe-dark: a probe with no prediction on
            # this chain): it is a satellite another chain tracks, and its noise would feed that
            # satellite's receiver-wide row. Below-horizon probes still pass (docs/CHORD_BUGLIST.md
            # #192 (h)).
            _dark_probes = {p for p in _ctx.probe_set if p not in _ctx.pred}
            _rf.resid2 = {
                _p: _rv
                for _p, _rv in _rf.resid2.items()
                if _p not in _dark_probes
                and (
                    ((_ctx.status.get(_p) or {}).get("coherence_s") or 0.0) > 0.0
                    or ((_ctx.status.get(_p) or {}).get("coh_frac") or 0.0) >= 0.3
                )
            }
        # KCOH rate feed (--rrate-kcoh-feed): the KCOH fold's remaining rate into the rrate
        # rows. Runs before the coarse feed so a fresh acceptance here deweights this cycle's
        # coarse measurements of the same satellite (--rrate-coarse-deweight). The estimate is
        # read from _est_last, which serves one dict for several cycles; rr_kcoh_fed feeds each
        # dict once.
        _kco = _est_last.get("kcoh")
        if _kco is rr_kcoh_fed.get("last"):
            _kco = None  # same estimate object as last cycle: already fed once
        if (
            args.rrate_state
            and args.rrate_kcoh_feed
            and _kco
            and _drp.t_now_abs is not None
        ):
            rr_kcoh_fed["last"] = _kco
            try:
                _jrk = rx.joint_receiver(
                    band_id, CODE_LEN, args.chip_rate_hz, gauge_mode="prior"
                )
                _nk = 0
                _krows = []
                for _p, _kv in sorted(_kco.items()):
                    if (
                        _kv.get("probe")
                        or (_kv.get("sig") or 0.0) < args.rrate_kcoh_min_sig
                        or (_kv.get("rate_pairs") or 0) < 8
                    ):
                        continue
                    _rem = (_kv.get("rate_hz") or 0.0) + (
                        _kv.get("rate_resid_hz") or 0.0
                    )
                    # Fold safety, the fine feed's bound: the per-record fold is unambiguous
                    # to ~+-23 Hz, so beyond 20 the estimate is suspect.
                    if abs(_rem) >= 20.0:
                        continue
                    _yk = _rem + (
                        _rf.cmd_applied.get(_p, 0.0) if args.rrate_feed_applied else 0.0
                    )
                    _sigk = min(0.3, max(0.03, 2.0 / math.sqrt(_kv["sig"])))
                    _k2 = (args.dr_constellation, int(_p))
                    if (
                        _jrk.update_rrate(
                            _k2, _yk, _drp.t_now_abs, args.carrier_hz, sigma_hz=_sigk
                        )
                        is not None
                    ):
                        _rf.kcoh_t[_p] = t0
                        _nk += 1
                        _krows.append("%d:%+.2f+-%.2f" % (_p, _yk, _sigk))
                if _nk:
                    _jrk.gauge_rrate()
                    _log_rl(
                        "jrr-kcoh",
                        "JRR-KCOH %s: %d sat(s) fed from the fold's remaining rate "
                        "(y = remaining%s, Hz): %s"
                        % (
                            log_tag() or args.signal,
                            _nk,
                            " + applied"
                            if args.rrate_feed_applied
                            else " ALONE, blind plant",
                            " ".join(_krows),
                        ),
                        every_s=60.0,
                    )
            except Exception as e:
                _log_rl("jrr-kcoh-err", "JRR-KCOH: failed (%s) -- cycle continues" % e)
        ratefeed.stage_rate_feed_coarse(_ctx)
        # 3d''. Fine rate feed: the ADR's residual half (res_cycles) differenced over the span,
        # with a noise floor far below the rate spectrum's. Under --rrate-state the JRRP line
        # logs it beside the coarse value whether or not it feeds, which is how
        # --rrate-phase-sign is calibrated. It feeds only with --rrate-phase-feed and a nonzero
        # sign, and per sat only when adr_fine_rate's structural gates pass (same arc, counter
        # advanced), the sat is cohering and the command moved <= 0.6 Hz over the span.
        ratefeed.stage_rate_feed_fine(_ctx)

        # 3b. Receiver code-rate clock offset (l-a): the median of this cycle's per-fit samples,
        # EMA-smoothed (slow, so it tracks oscillator drift), then seeded below to every unheld
        # sat as its code-phase rate. The code-side twin of the carrier clock bias: strong
        # detections calibrate the clock so weak ones never drift off peak. Ephemeris-free (v/c
        # cancels). A median beyond code_bias_max (ppm) is refused.
        if len(_ctx.la_samples) >= args.code_bias_min_sats:
            raw_cb = statistics.median(_ctx.la_samples)
            if abs(raw_cb) < args.code_bias_max * 1e-6:
                _cb.code_ema = (
                    raw_cb
                    if _cb.code_ema is None
                    else _cb.code_ema + args.code_bias_alpha * (raw_cb - _cb.code_ema)
                )
                _log_rl(
                    "la-pool",
                    "code-rate clock offset (l-a) %+.3f ppm (raw %+.3f, %d fitted "
                    "sats, EMA a=%.2f)"
                    % (
                        _cb.code_ema * 1e6,
                        raw_cb * 1e6,
                        len(_ctx.la_samples),
                        args.code_bias_alpha,
                    ),
                )
                # Shared through the Receiver, keyed by band. The value is a fractional
                # frequency, so it holds for every band (Receiver.code_bias_any_band).
                rx.contribute_code_bias(
                    chain_id, band_id, _cb.code_ema, len(_ctx.la_samples), t0
                )
        cb_to_seed = _cb.code_ema
        # Joint consumer 'rate' (--joint-consume): seed the code-rate clock from the joint
        # state's clk_rate instead of the l-a EMA. They are the same quantity: clk_rate
        # (chips/s) divided by the chip rate is the fractional rate l-a. The l-a EMA remains the
        # fallback and is logged beside it.
        if "rate" in joint_consume:
            _jr = _joint_state(rx, band_id, args)
            # clk_rate is in the joint state's reference chips per second (Receiver.joint_unit),
            # not this chain's: dividing by our own chip rate would scale the fractional rate by
            # the chip-rate ratio (20x on L2C, 2x on E6).
            _ju = rx.joint_unit() if _jr is not None else None
            _ppm = (_jr.clk_rate / _ju[0] * 1e6) if _ju is not None else None
            # Plausibility bound: never hand the instrument a physically impossible rate because
            # an estimator produced it. CHORD's reference is GPS-disciplined, so a rate beyond
            # joint_max_rate_ppm (or a state under joint_min_sats) is refused and the l-a EMA
            # stays.
            if (
                _ju is not None
                and len(_jr._idx) >= args.joint_min_sats
                and abs(_ppm) <= args.joint_max_rate_ppm
            ):
                cb_to_seed = _jr.clk_rate / _ju[0]
                _log_rl(
                    "jointrate",
                    "code-rate clock from the JOINT state: %+.5f ppm "
                    "(l-a EMA says %+.5f)"
                    % (cb_to_seed * 1e6, (_cb.code_ema or 0.0) * 1e6),
                    every_s=60.0,
                )
        if cb_to_seed is not None:
            n_seeded = 0
            for prn, seed in seeds.items():
                # All sats, fitted included: l-a is common to the receiver, so the pooled value
                # is the right code rate for everyone and far quieter than each sat's own
                # short-baseline slope fit, which stays as the cp0 anchor only. Except held
                # (locked) sats: their frozen (anchor, rate) pair must stay consistent, and
                # changing the rate under a stale ref_hop jumps the extrapolated cp.
                if prn in cp_held:
                    continue
                seed.put(
                    "la_rate",
                    epoch=seed.get("ref_hop"),
                    code_phase_rate=cp_rate_from_code_bias(
                        cb_to_seed, args.hops_per_sec, args.chip_rate_hz,
                    ),
                )
                n_seeded += 1
            if n_seeded:
                _log_rl(
                    "la-seed",
                    "seeded code rate from (l-a) %+.3f ppm -> %d sat(s)"
                    % (cb_to_seed * 1e6, n_seeded),
                )

        # Carrier command (--rrate-command): one per (sat, band). carrier_correction_hz() is the
        # receiver-wide f_carrier plus this sat's orbital rrate, scaled to this band, posted by
        # stage_push_seeds as carrier_trim_hz (NCO derotation: a frequency step is
        # phase-continuous in the tracker). doppler_hz is never touched: seed continuity beats
        # freshness.
        _ctx.jrc = None
        if args.rrate_command:
            if not _rf.full_ok:
                # Closing the loop on a capped (in-cap noise) or absent measurement walks
                # every command, so no command is computed without the uncapped fields this
                # poll. jrc stays None, and stage_push_seeds slews standing commands toward 0.
                _log_rl(
                    "jrr-nofull",
                    "rrate-command HELD: no row carries deep_rate_full_* this poll "
                    "-- no new commands; standing ones return to 0",
                    every_s=120.0,
                )
            else:
                try:
                    _j = rx.joint_receiver(
                        band_id, CODE_LEN, args.chip_rate_hz, gauge_mode="prior"
                    )
                    # No receiver-wide term solved yet -> nothing to command. The sigma
                    # gate below then handles per-sat convergence one row at a time.
                    _ctx.jrc = _j if _j.f_carrier_sigma() != float("inf") else None
                except Exception:
                    _ctx.jrc = None
        _ctx.rr_cmd_new = {}
        # Reset per pass: these are `+=` counters for the JRR-CMD line, and nothing else
        # zeroes them.
        _rf.railed = 0
        _rf.released = 0

        # 4. Build the seed payload (the DLL trim is added to the posted copy only); it is
        # posted to every tracker below.
        _ctx.payload = []
        seeding.stage_push_seeds(_ctx)
        # The commands shipped this poll become the rrate feed's reference next poll. Rebuilt,
        # not updated: a sat that stopped being commanded (row widened, seed dropped) must fall
        # back to referencing 0, or a stale command would re-enter its measurements forever.
        _rf.cmd_applied.clear()
        _rf.cmd_applied.update(_ctx.rr_cmd_new)
        if _ctx.rr_cmd_new:
            _log_rl(
                "jrr-cmd",
                "JRR-CMD[%s]: %s Hz (rrate rows -> carrier_trim_hz, %d sat(s), "
                "%d slew-railed, %d releasing)"
                % (
                    args.dr_constellation,
                    " ".join("%d:%+.2f" % kv for kv in sorted(_ctx.rr_cmd_new.items())),
                    len(_ctx.rr_cmd_new),
                    _rf.railed,
                    _rf.released,
                ),
                every_s=60.0,
            )
        # Seed currency audit (SEEDAUDIT): does the tuple we ship imply the same physical code
        # phase as the last one shipped for that satellite? The tracker re-pins on every POST,
        # so the wrapped difference of the two tuples' phases at the new ref_hop is the step
        # its despread sees, across every mutation path (birth, slew, trim, nh/CM lifts,
        # coast). A currency leak (cp built with one Doppler, shipped beside another) shows as
        # t_abs*chip_rate/carrier chips per Hz of mismatch, large at a multi-day F-engine age;
        # hence the equivalent-Hz figure. Legitimate steps are slews, DLL trim increments,
        # escapes and re-anchors. The modulus is the seed's own long-code modulus, so whole
        # code-period flips show too.
        _aud_mod = (LC_SEG * CODE_LEN) if LC_SEG > 1 else CODE_LEN
        _aud_steps = []
        for d in _ctx.payload:
            _h_new = int(d.get("ref_hop", 0) or 0)
            _prevA = seed_audit_prev.get(d["prn"])
            if _h_new > 0 and all(k in d for k in ("code_phase_chips", "doppler_hz")):
                seed_audit_prev[d["prn"]] = {
                    k: d[k]
                    for k in (
                        "code_phase_chips",
                        "code_phase_at_ref_chips",
                        "doppler_hz",
                        "code_phase_rate",
                        "ref_hop",
                        "doppler_rate_hz_s",
                    )
                    if k in d
                }
            if (
                _prevA is None
                or _h_new <= 0
                or _h_new < int(_prevA["ref_hop"])
                or _h_new - int(_prevA["ref_hop"]) > 600 * args.hops_per_sec
            ):
                continue
            # Model what the tracker reads: propagate_seed prefers code_phase_at_ref_chips
            # when the payload carries it, and tracker_phase_at picks the same reference.
            _ph_prev = tracker_phase_at(
                _prevA,
                _h_new,
                args.hops_per_sec,
                args.chip_rate_hz,
                args.carrier_hz,
                args.code_doppler_sign,
                _aud_mod,
            )
            _ph_new = tracker_phase_at(
                d,
                _h_new,
                args.hops_per_sec,
                args.chip_rate_hz,
                args.carrier_hz,
                args.code_doppler_sign,
                _aud_mod,
            )
            _stp = ((_ph_new - _ph_prev + _aud_mod / 2.0) % _aud_mod) - _aud_mod / 2.0
            _ddopA = d["doppler_hz"] - _prevA["doppler_hz"]
            _dtA = (_h_new - int(_prevA["ref_hop"])) / args.hops_per_sec
            _aud_steps.append((abs(_stp), d["prn"], _stp, _ddopA, _dtA))
            if abs(_stp) > 5.0:
                _lev_hz = _stp / max(
                    1.0,
                    (_h_new / args.hops_per_sec) * args.chip_rate_hz / args.carrier_hz,
                )
                # name the writers of this tuple (Seed.owners)
                _sdA = seeds.get(d["prn"])
                _log(
                    "SEEDAUDIT STEP PRN %d: %+.2f chips (= %+.4f Hz x lever) "
                    "ddop %+.3f Hz dt %.1f s trim %+.3f%s"
                    % (
                        d["prn"],
                        _stp,
                        _lev_hz,
                        _ddopA,
                        _dtA,
                        _dls.trim.get(d["prn"], 0.0),
                        ("  [%s]" % _sdA.owners()) if isinstance(_sdA, Seed) else "",
                    )
                )
        for _pA in list(seed_audit_prev):
            if _pA not in seeds:
                del seed_audit_prev[_pA]
        if _aud_steps:
            _aud_steps.sort()
            _n = len(_aud_steps)
            _wA, _wp, _ws, _wd, _wt = _aud_steps[-1]
            _log_rl(
                "seedaudit",
                "SEEDAUDIT n=%d |step| med %.3f p90 %.3f max %.3f chips "
                "(PRN %d: %+.3f, ddop %+.3f Hz, dt %.1f s)"
                % (
                    _n,
                    _aud_steps[_n // 2][0],
                    _aud_steps[min(_n - 1, int(_n * 0.9))][0],
                    _wA,
                    _wp,
                    _ws,
                    _wd,
                    _wt,
                ),
                every_s=60.0,
            )
        # Epoch-skew census (log only): seeds shipping an at-epoch field (Doppler, rate,
        # at-ref phase) recorded against a different ref_hop than the one they ship with
        # (Seed.epoch_skew), counted per field with the owner that wrote it.
        _skewN, _skew_ex, _skew_f = 0, [], {}
        for _pS in sorted(seeds):
            _sS = seeds[_pS]
            if not isinstance(_sS, Seed):
                continue
            _sk = _sS.epoch_skew()
            if _sk:
                _skewN += 1
                for _kS in _sk:
                    _skew_f[_kS] = _skew_f.get(_kS, 0) + 1
                if len(_skew_ex) < 3:
                    _skew_ex.append(
                        "PRN %s: %s vs ref %s"
                        % (
                            _pS,
                            ",".join(
                                "%s=%s@%s" % (k, v[0], v[1])
                                for k, v in sorted(_sk.items())
                            ),
                            _sS.get("ref_hop"),
                        )
                    )
        if _skewN:
            # Per field, because the classes differ: a skewed at-ref phase is a chips-scale
            # error and should count 0 (the hold arms carry it with its own ref_hop); a skewed
            # doppler_rate enters only through the quadratic term.
            _log_rl(
                "epochskew",
                "EPOCH-SKEW %d/%d seed(s) ship an at-epoch field recorded against "
                "a different ref_hop (by field: %s): %s"
                % (
                    _skewN,
                    len(seeds),
                    " ".join("%s:%d" % _kv for _kv in sorted(_skew_f.items())),
                    "; ".join(_skew_ex),
                ),
                every_s=60.0,
            )
        # Hand the C++ fleet loop this cycle's decisions (trimarm.py). Policy stays here on the
        # broker cycle: ephemeris, sky, clock solve, joint state, presence, floors, deep gate,
        # who is armed. What crosses is the armed PRN list, the loop constants, the probe floor
        # and the targets; the C++ side forms the discriminator, steps the integrator and posts
        # the trim, and adds no gate and chooses no PRN.
        # Gain and leak cross per second, not per update: per-update constants would make the
        # loop's closed-loop and noise bandwidths scale with its update rate. The controller
        # divides by the rate it measures, so the loop keeps this bandwidth at whatever rate it
        # achieves. The targets ride with it because this process knows which instances serve
        # the chain (--trackers, brace-expanded): a node added to the fleet reaches the loop
        # without regenerating the gather's config.
        trimarm.stage_fleet_trim_arming(_ctx)

        # Live PRN membership (prnmap.py), after everything that seeds or arms this cycle, so
        # a swap lands on a settled chain and decides on this cycle's elevations. Does nothing
        # unless --prn-reconfig apply.
        prnmap.stage_prn_membership(_ctx)

        ok = 0
        for t_ep in trackers:
            try:
                _post("%s/set_seeds" % t_ep, _ctx.payload)
                ok += 1
            except Exception as e:
                _log("set_seeds %s failed: %s" % (t_ep, e))

        _log_rl(
            "active",
            "active=%s (%d); seeded %d/%d trackers"
            % (sorted(seeds), len(seeds), ok, len(trackers)),
        )
        if _ctx.cl_report:
            _log_rl("clreport", "CL: " + "; ".join(_ctx.cl_report))

        if dhw is not None:
            dhw.flush(t0)
        # Real elapsed time, not the frozen cycle clock: the sleep must know how long the pass
        # took, or the cadence would become interval + processing.
        _busy = time.time() - t_wall
        record_cycle(log_tag().strip() or "chain", _busy, args.interval)
        dt = args.interval - _busy
        if dt > 0 and _TR.mode != "read":
            time.sleep(dt)


if __name__ == "__main__":
    try:
        main()
    except _TranscriptDone as e:
        # Normal end of a replay: the recording ran out. Print the digest on stdout so a
        # bare --transcript-read is useful without the harness.
        _log(
            "transcript replay complete (%s); %d posts, digest %s"
            % (e, len(_TR.posts), _TR.digest())
        )
        print(_TR.digest())
    finally:
        _TR.close()
