#!/usr/bin/env python3
"""Distributed GNSS broker: search detections -> consensus seeds -> trackers.

The control plane of the distributed-band channelized pipeline. It is wholly
decoupled from the sample dataflow -- pure REST glue + assignment policy -- so the
same broker drives the local single-kotekan comb and the real cross-node CHORD
layout; only the endpoint list changes.

Each iteration:
  1. GET <detector>/get_detections from every detection source; keep the best-SNR
     detection per PRN. A "detector" is a GnssSearchAggregator (cross-channel
     consensus already done) or, in the older fan-out, a GnssChannelizedSearch per
     subband -- either way the broker takes the strongest hit per PRN across them.
  2. (optional) gate by visibility: drop below-horizon PRNs.
  3. GET <combiner>/get_status for the full-band |A| per PRN; COAST a visible PRN
     through a signal dropout (hold its seed + forecast its Doppler forward from the
     orbit) so a radar sweep / brief fade doesn't lose the lock -- drop only when it
     SETS or |A| stays down for the whole --coast-budget (the predictor promotion).
  4. POST the consensus seed set [{prn, doppler_hz, code_phase_chips}] to *every*
     tracker's /set_seeds, so all subbands despread the same (cp, Doppler) and the
     per-subband products recombine coherently.

Endpoints (--detectors / --trackers / --combiner) are comma-separated and support:
  * bare stage names         -> resolved against --rest-url (one kotekan instance)
  * absolute URLs            -> http://host:port/stage (per-node, mix freely)
  * brace ranges {a..b}      -> track_{00..49} expands to track_00..track_49,
                                bash-style (zero-padded iff an operand is padded);
                                http://nodeB:12048/track_{0..24} works too.

Seeds are refreshed from the latest detection every iteration, so the trackers'
code phase stays fresh against slow code-Doppler drift -- run --interval well below
the ~0.5 s decorrelation time. REST via stdlib urllib; visibility via skyfield
(optional, only if --lat/--lon given).
"""

import argparse
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request

C_LIGHT = 299792458.0  # m/s (audit rec E: was inlined at four sites)
from datetime import datetime, timezone, timedelta

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from gnss_stages import resolve_stage  # noqa: E402  (gps_* <-> bare stage-name aliasing)


# ---------------------------------------------------------------------------------------
# THE STATELESS HALF OF THIS FILE NOW LIVES IN gnss_broker/ (task #27 M1).
#
# Nothing below the import is different: the modules were sliced out line-for-line and the
# equivalence gate (scripts/gnss/broker_equiv.py) holds the POST stream byte-identical
# across the move. What changes is that the pieces the UNIFIED broker needs exactly one of
# -- the transport, the fits, the sky -- are now separable from the per-chain loop that
# `main()` still is. See docs/CHORD_BROKER_REFACTOR.md.
#
# Re-exported at module scope rather than referenced through the package, because these
# names are the file's public surface: test_code_bias.py imports this module for
# code_clock_bias_sample / cp_rate_from_code_bias, and every launch script drives main().
# ---------------------------------------------------------------------------------------
from gnss_broker.transport import (           # noqa: E402
    _TranscriptDone, _Transcript, _TR, _now, _get, _post, _log, _log_rl,
    expand_token, resolve_prefix, parse_endpoints, log_tag, install_dns_cache,
    record_cycle,
)
from gnss_broker import telem as _telem                    # noqa: E402  (task #59 gather)
from gnss_broker import combdll                            # noqa: E402  (task #63 comb DLL)
from gnss_broker import elemgain                           # noqa: E402  (task #57 step 2)
from gnss_broker.fits import (                # noqa: E402
    retag_seed_doppler, seed_phase_at_ref, track_vs_fit_chips, tracker_phase_at,
    fit_cp_rate, fit_dop_rate, code_clock_bias_sample, rate_residuals,
    cp_rate_from_code_bias, dr_cp0, dr_seed_phys, adr_fine_rate, q_stall_verdict,
    split_erratic_offsets,
    instance_stall_verdict,
    rf_lobes,
)
from gnss_broker.fleet import (               # noqa: E402
    fleet_dll, _coherent_sum, fleet_coherent, fleet_spectrum, fleet_spectrum_aligned,
    fit_spectrum_delay,
    poll_rf_stats,
)
from gnss_broker.publish import FleetPublisher            # noqa: E402
from gnss_broker.seed import Seed                          # noqa: E402  (task #83)
from gnss_broker.admission import AdmissionGate, reseed_step  # noqa: E402  (#90/#50)
from gnss_broker.handover import TrimHandover              # noqa: E402  (task #92)
from gnss_broker.rampfit import RampTracker                # noqa: E402  (task #93 shadow)
from gnss_broker.cli import build_parser, _FROZEN           # noqa: E402  (task #89 flag surface)
from gnss_broker.context import ChainContext                # noqa: E402  (the stage interface)
from gnss_broker.clockbias import ClockBias                 # noqa: E402  (the receiver LO bias)
from gnss_broker.loopstate import (                         # noqa: E402
    CarrierState, WatchdogState, NhOverlay, DllLoopState, HoldState, CpTracking,
    RateFeedState, NavDecoders, ClSibling,
)
from gnss_broker import instruments                         # noqa: E402  (the DLL's measurements)
from gnss_broker import deadreckon                          # noqa: E402  (the clock pipeline)
from gnss_broker import almanac as almanac_stage            # noqa: E402  (orbit + visibility)
from gnss_broker import codeloop                            # noqa: E402  (the DLL + watchdog)
from gnss_broker import statepub                            # noqa: E402  (the state record)
from gnss_broker import ratefeed                            # noqa: E402  (#33 rate feeds)
from gnss_broker import trimarm                             # noqa: E402  (C++ trim arming)
from gnss_broker import prnmap                              # noqa: E402  (live PRN membership)
from gnss_broker import carrierloop                         # noqa: E402  (off in production)
from gnss_broker import searchhint                          # noqa: E402  (narrow the search)
from gnss_broker import seeding                             # noqa: E402  (detections -> seeds)
from gnss_broker import navbits                             # noqa: E402  (off in production)
from gnss_broker import clsibling                           # noqa: E402  (the CM/CL sibling)
from gnss_broker import fleetdll                            # noqa: E402  (the fleet DLL shell)
from gnss_broker.detectors import (                         # noqa: E402  (D0-D3)
    QSeries, BrownoutDetector, LatchDetector, SawtoothDetector,
)
from gnss_broker import signals                            # noqa: E402
from gnss_broker import receiver                           # noqa: E402
from gnss_broker.state_filter import SatBiasFilter         # noqa: E402
from gnss_broker.sky import (                 # noqa: E402
    brdc_predict, visible_prns, _dh_dpos,
    _cnav_brdc_xcheck, _cnav2_brdc_xcheck, _inav_brdc_xcheck, _lnav_brdc_xcheck,
    _fnav_brdc_xcheck, _bcnav2_brdc_xcheck, _bcnav3_brdc_xcheck, _bcnav1_brdc_xcheck,
)


def make_spectrum_writer(path_tmpl, log=None):
    """Append-only JSONL writer for the raw per-channel spectrum (task #25).

    Returns f(t_utc, band, spec) -> n_points_written, or None when archiving is off.
    `spec` is fleet_spectrum's {prn: [(freq_id, A, energy, inst_key)]}.

    The file handle is reopened when the expanded path changes, so %Y/%m/%d roll a long
    run by day without a restart. Failures are the CALLER's to log and must never take
    the broker down -- an archive is a by-product, not a dependency.
    """
    if not path_tmpl:
        return None
    import time as _time
    state = {"path": None, "fh": None}

    def _write(t_utc, band, spec, t_rx=None):
        path = _time.strftime(path_tmpl, _time.gmtime(t_utc))
        if path != state["path"]:
            if state["fh"] is not None:
                state["fh"].close()
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            state["fh"] = open(path, "a")
            state["path"] = path
            if log:
                log("SPEC-ARCHIVE: writing raw per-channel spectra -> %s" % path)
        n = 0
        for prn, pts in sorted(spec.items()):
            for p in pts:
                # (freq_id, amplitude, energy, instance) -- tolerate a longer tuple so a
                # future field cannot silently truncate the archive.
                fid, amp, energy, inst = p[0], p[1], p[2], p[3]
                # AMPLITUDE IS COMPLEX. fit_spectrum_delay takes tau from the PHASE ramp
                # across frequency, so the per-channel value is a visibility, not a
                # magnitude -- storing abs() would throw away exactly the quantity the
                # delay fit exists to measure, and would make the archive unable to
                # reproduce its own tau. Store both parts; magnitude is a function of
                # them, they are not a function of magnitude.
                _re, _im = (float(amp.real), float(amp.imag)) if isinstance(amp, complex) \
                    else (float(amp), 0.0)
                state["fh"].write(json.dumps(
                    {"t": round(float(t_utc), 3), "band": band, "prn": int(prn),
                     "freq_id": int(fid), "inst": str(inst),
                     "re": _re, "im": _im, "amp": (_re * _re + _im * _im) ** 0.5,
                     "energy": float(energy),
                     "t_rx": (round(float(t_rx), 3) if t_rx is not None else None)}) + "\n")
                n += 1
        state["fh"].flush()
        return n

    return _write






def main(argv=None, rx=None, publisher=None):
    # Name resolution was the broker's cycle time -- see transport.py's DNS CACHE note.
    # Idempotent, so the per-chain threads under broker_multi all land on one cache.
    install_dns_cache()
    # `--signal help` before the parser, because --trackers is required and listing the
    # known signals must not depend on being able to name a fleet first.
    if "help" in (argv if argv is not None else sys.argv[1:]):
        _a = list(argv if argv is not None else sys.argv[1:])
        if "--signal" in _a and _a.index("--signal") + 1 < len(_a) \
                and _a[_a.index("--signal") + 1] == "help":
            print("known signals (derived from lib/stages/gnss/gnssSignal.hpp):\n"
                  + signals.describe())
            return
    ap = build_parser(__doc__)
    args = ap.parse_args(argv)
    # Frozen tuning: values that were flags and are now constants (task #89). Applied here so
    # every `args.<name>` use site downstream is untouched -- the POST stream is byte-identical
    # by construction, which is what the four broker_equiv fixtures verify.
    for _k, _v in _FROZEN.items():
        setattr(args, _k, _v)

    # ── THE LOOPS' OWN MEMORY ─────────────────────────────────────────────────────────
    # Six owner objects, constructed FIRST because everything else in this function may
    # touch them. Each is one loop's per-satellite state; see gnss_broker/loopstate.py for
    # which table belongs to which loop and why that grouping is the one that matters.
    _carrier = CarrierState()   # the shared carrier loop's memory; see gnss_broker/loopstate.py
    _watchdog = WatchdogState()   # the track watchdog's clocks
    _nho = NhOverlay()      # NH overlay alignment (#41: judge on the VERTEX)
    _dls = DllLoopState()   # the code loop + the C++ arming handshake
    _hold = HoldState()     # why a sat is held rather than dropped
    _cpt = CpTracking()     # per-sat code-phase history
    _rf = RateFeedState()   # carrier-rate observables + the commanded reference
    _nav = NavDecoders()    # broadcast nav-message decoders (off in production)
    _cls = ClSibling()      # the CM/CL long-code sibling's segment search
    # D0: the q series that KEEPS the satellites that stopped reporting. Every arm judges on
    # this, never on the DLL line -- see gnss_broker/detectors.py for what that cost.
    _qpop = QSeries()
    _brown = BrownoutDetector()   # D1: chain-wide presence collapse, as an episode
    _latch = LatchDetector()      # D2: healthy -> absent -> stays absent. UNARMED,
                                  #     measuring the base rate #90 never established
    _saw = SawtoothDetector()     # D3: a standing trim that ramps then gets WIPED
    # Live slot->PRN membership vs the sky. OFF unless --prn-reconfig; see prnmap.py.
    _prnmap = prnmap.PrnMapState()

    _raw_argv = list(argv if argv is not None else sys.argv[1:])
    if args.signal:
        try:
            _sig = signals.get(args.signal)
        except (KeyError, ValueError, RuntimeError) as e:
            ap.error("--signal: %s" % e)
        # dest -> the value the named signal implies. Anything the caller gave EXPLICITLY
        # wins only if it agrees; a disagreement is an error naming both numbers, because
        # the whole point of naming a signal is that these constants stop being retyped.
        _implied = {
            "carrier_hz": _sig.carrier_hz, "chip_rate_hz": _sig.chip_rate_hz,
            "code_length": float(_sig.code_length),
            "long_code_segments": _sig.long_code_segments,
            "long_code_epoch_s": _sig.long_code_epoch_s,
            "nh_overlay_len": _sig.nh_overlay_len,
            "constellation": _sig.constellation, "dr_constellation": _sig.constellation,
        }
        if _sig.min_prn is not None:
            _implied["dr_min_prn"] = _sig.min_prn
        for _dest, _want in _implied.items():
            _flag = "--" + _dest.replace("_", "-")
            _given = any(a == _flag or a.startswith(_flag + "=") for a in _raw_argv)
            if not _given:
                setattr(args, _dest, _want)
                continue
            _have = getattr(args, _dest)
            _same = (abs(_have - _want) <= 1e-9 * max(1.0, abs(_want))
                     if isinstance(_want, float) else _have == _want)
            if not _same:
                ap.error("%s %r contradicts --signal %s, which implies %r. One of them is "
                         "wrong and neither would have errored on its own -- fix the "
                         "command rather than letting a silent override pick."
                         % (_flag, _have, args.signal, _want))
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
                _skip = True          # drop the flag AND its separate value
                continue
            if _a.startswith("--transcript-write="):
                continue
            _keep.append(_a)
        _TR.open_write(args.transcript_write, _keep)
    elif args.transcript_read:
        _TR.open_read(args.transcript_read)
        # TEST-ONLY, and only under replay: scripts/gnss/broker_equiv.py's `selftest` needs
        # to prove the gate CAN fail, so it asks for a perturbation far below anything
        # physical and requires the digest to move. A gate that cannot fail is not a gate.
        # Which knob, and by how much: "dll_gain:1e-6". Different fixtures exercise
        # different code, so one fixed knob cannot serve them all -- the synthetic fleet
        # runs the DLL, while the e2e fixture (real GPU, known truth) runs --dll-gain 0 and
        # is entirely a seed-arithmetic test. The harness tries several and reports which
        # ones move the digest, which is a direct read on what the fixture COVERS.
        _pert = os.environ.get("GNSS_BROKER_EQUIV_PERTURB")
        if _pert:
            _knob, _, _eps = _pert.partition(":")
            _f = 1.0 + float(_eps or "1e-6")
            if not hasattr(args, _knob):
                sys.exit("perturb: no such argument %r" % _knob)
            setattr(args, _knob, getattr(args, _knob) * _f)

    # AFTER the transcript is open, or the tick would neither be recorded nor replayed.
    # The SETUP phase reads the clock too (warm-start file stamps, the almanac epoch
    # offset, _bias_meas_t). Tick once here so setup runs at one frozen instant like every
    # cycle does, and so a recording captures it explicitly rather than by whichever
    # _now() happened to land first.
    _TR.tick()
    if args.cl_assist and args.cl_tracker:
        # In-place lift + copied lift together would hand the CM tracker CL-lifted phases:
        # broken CM tracking with no error anywhere downstream. Refuse at the door.
        ap.error("--cl-assist (in-place, single-chain) and --cl-tracker (sibling-chain) are "
                 "mutually exclusive")

    # --almanac-epoch is a CLOCK OFFSET, not a frozen instant. The broker "lives in the
    # capture's time frame": every prediction site evaluates at now() + _alm_clock_offset, so
    # the sky ADVANCES as the replayed file plays, exactly as it did during the capture.
    # ⚠️ The first implementation evaluated the almanac at the fixed epoch forever. Measured
    # on the L5 replay bench (2026-07-27): PRN20 locked beautifully for the first ~15 s
    # (resid +1.6 Hz) while the prediction still matched the file, then the seeds -- refreshed
    # every cycle from a prediction that never moved while the data did (~1 Hz/s at L5) --
    # PULLED THE TRACKER OFF the satellite: amplitude collapsed to the noise floor, NH phase
    # flapped randomly, residuals swung +-95 Hz. Worse than no assist: a stale assist is an
    # active tug toward where the satellite used to be.
    # Assumes the replay paces ~realtime (rawFileRead frame_period_us); pacing error over a
    # few-minute bench is <<1 s, i.e. <1 Hz of Doppler.
    _alm_clock_offset = (args.almanac_epoch - _now()) if args.almanac_epoch else 0.0
    # FILE-POSITION clock (fills in once combiner status flows): wall-rate advance assumes the
    # replay paces at exactly 1.0x, and it does NOT -- measured 0.80x (frame_period_us plus
    # per-frame file-open overhead), so a wall-advancing epoch runs AHEAD of the data by 0.2 s
    # per second, i.e. dop_rate x 0.2t Hz of per-satellite seed error, growing without bound.
    # The combiner's rows carry the CAPTURE-CLOCK utc (capture_utc0 + samples/fs): utc minus
    # capture_utc0 is the exact file position at any pacing. One cycle stale (~interval) ->
    # sub-second epoch error -> <1 Hz. Falls back to wall-advance until the first status row.
    _alm_file_pos = [None]

    def _alm_now():
        """The time the ALMANAC thinks it is: wall clock in live runs; under --almanac-epoch,
        capture epoch + FILE POSITION (from the combiner's capture-clock utc) when available,
        else capture epoch + wall elapsed."""
        if args.almanac_epoch and _alm_file_pos[0] is not None:
            return datetime.fromtimestamp(args.almanac_epoch + _alm_file_pos[0],
                                          tz=timezone.utc)
        return datetime.fromtimestamp(_now() + _alm_clock_offset, tz=timezone.utc)

    # ---- RECEIVER SCOPE (task #27 M3) --------------------------------------------------
    # What every chain on this telescope shares: the F-engine time anchor, the BRDC store,
    # and the two halves of the receiver clock. `rx` is passed in by the multi-chain driver
    # (M5); standing alone, a chain owns a private one and every contribute/consume pair
    # below degenerates to the chain talking to itself -- which is exactly why M3 leaves
    # single-chain behaviour byte-identical.
    rx = rx if rx is not None else receiver.Receiver(log=_log)
    # WHO this chain is, and WHICH band it measures the code clock in. The band key is the
    # carrier to 10 kHz, so GPS L5 and Galileo E5a -- genuinely the same 1176.45 MHz
    # hardware -- share one code-clock scope, while a retune to E5b at 1207 MHz does not.
    chain_id = args.signal or ("%s@%.2fMHz" % (args.constellation or args.dr_constellation,
                                               args.carrier_hz / 1e6))
    band_id = "%.2fMHz" % (args.carrier_hz / 1e6)
    if args.joint_consume and not args.joint_shadow:
        args.joint_shadow = True   # a consumer without the solve running would read zeros
    if args.rrate_command and not args.rrate_state:
        args.rrate_state = True    # same rule: the command gates on the row's sigma, and
                                   # only the feed can ever bring that below infinity

    base = args.rest_url.rstrip("/")
    detectors = parse_endpoints(args.detectors, base)
    trackers = parse_endpoints(args.trackers, base)
    combiner = resolve_prefix(args.combiner, base)
    # FLEET DLL: every combiner whose E/L powers join the sum. --combiner stays the ONE status
    # source for everything else (amplitudes, drop decisions, nav bits) -- this list only feeds
    # the code loop, so a chain that does not set it is bit-for-bit unchanged.
    dll_combiners = parse_endpoints(args.dll_combiners, base) if args.dll_combiners else []
    # PATH B's own population, deliberately parallel and deliberately separate: same estimator,
    # different record stream, so the two fleet numbers can be compared rather than blended.
    # These endpoints feed NO loop -- fleet_coherent is an observable -- so an n2 chain that is
    # down, restarting, or absent costs a log line and nothing else.
    n2_combiners = parse_endpoints(args.n2_combiners, base) if args.n2_combiners else []
    spectrum_endpoints = (parse_endpoints(args.spectrum_endpoints, base)
                          if args.spectrum_endpoints else [])
    # TASK #59: the frame-synced telemetry gather. ONE reader thread per PROCESS -- broker_multi
    # runs all five chains here as threads, the stream carries every chain on one connection,
    # and the store is keyed by chain, so five clients would decode the same bytes five times
    # and throw four of them away. `telem_chain` is this thread's chain key (the same string
    # the trackers stamp on every frame), read from the log tag rather than re-derived from
    # --signal: they agree today and nothing enforces that they must.
    _tg = _telem.parse_endpoint(args.telem_gather)
    telem_client = _telem.shared_client(*_tg) if _tg else None
    telem_chain = log_tag() or (args.signal or "")
    if telem_client is not None:
        _log("telem: gather %s:%d, chain key %r%s"
             % (_tg[0], _tg[1], telem_chain,
                "; fleet_coherent will read it" if args.telem_coherent else " (store only)"))
    # Per-subband archive (task #25). Created once; None when --spectrum-archive is unset,
    # which is the only condition the call site checks.
    _spec_writer = make_spectrum_writer(args.spectrum_archive, log=_log)
    # Integer hop tolerance, derived once from the record geometry. Kept as an int so every
    # comparison downstream is integer arithmetic on the F-engine's own counter.
    dll_hop_window = max(0, int(round(args.dll_hop_window_s * args.hops_per_sec)))
    # Task #49 opt-in set. Parsed ONCE at launch and logged, so which satellites are on the
    # deep gate is a fact in the log rather than something to infer from behaviour -- the
    # whole point of the per-PRN rollout is that the A/B is readable afterwards.
    _dg = (args.dll_deep_gate or "").strip()
    if _dg.lower() == "all":
        _deep_gate = True
    elif _dg:
        _deep_gate = {int(x) for x in _dg.replace(",", " ").split()}
    else:
        _deep_gate = None
    # #79: PRN -> last time the SEARCH saw it at/above --dll-deep-gate-from-search. The
    # auto-generated half of the deep-gate set; unioned with _deep_gate each cycle.
    _dg_auto_last = [set()]   # last logged auto set, so the line prints on CHANGE only
    _rs = (args.reseed_spec_tau or "").strip()
    if _rs.lower() == "all":
        _dls.reseed_prns = True
    elif _rs:
        _dls.reseed_prns = {int(x) for x in _rs.replace(",", " ").split()}
    else:
        _dls.reseed_prns = None
    # #90 ABSENT-PRN ADMISSION. The strike/cooldown/was-present/population state and all four
    # flights' guards live in gnss_broker/admission.py, where they are unit-testable offline --
    # every one of those guards used to need a broker restart against the live fleet to
    # exercise, which is what made #90 cost four flights in one evening.
    _adm_gate = AdmissionGate(armed=bool(args.reseed_admit_absent))
    if _dls.reseed_prns:
        _log("SPEC-TAU RE-SEED (#50) active on %s: q<%.2f, spec_peak_ratio>=%.2f, gain %.2f, "
             "cap %.2f chips, span +-%.1f. Fires only where the discriminator has NO GRADIENT "
             "(far off-peak, E~P~L~noise); applied as a SEED step so the slew cap cannot "
             "swallow it"
             % ("ALL PRNs" if _dls.reseed_prns is True else
                "PRN " + ",".join(str(p) for p in sorted(_dls.reseed_prns)),
                args.reseed_q_max, args.reseed_min_ratio, args.reseed_gain,
                args.reseed_max_chips, args.spec_span_chips))
        if not _deep_gate:
            _log("SPEC-TAU RE-SEED (#50) WARNING: no --dll-deep-gate is set, so the PRNs this "
                 "targets will mostly fail `present` and never reach the re-seed test at all "
                 "(that is the #49 latch). Arm both, or this does nothing.")
    if _deep_gate:
        _log("DLL DEEP GATE (#49) active on %s at %.1fx deep_floor -- these PRNs are trimmed "
             "on DETECTION (deep_snr) instead of on prompt power, which is on-peak-biased and "
             "latches an off-peak satellite out of its own correction"
             % ("ALL PRNs" if _deep_gate is True else
                "PRN " + ",".join(str(p) for p in sorted(_deep_gate)),
                args.dll_deep_gate_margin))
    # Optional REST publication of the fleet-merged state (see FleetPublisher). Started here so
    # a bind failure is fatal at launch rather than silently leaving the viewer with no source.
    # ONE PORT FOR EVERY CHAIN (task #27 M6). The driver passes a shared publisher in and
    # this chain claims a slot on it; standing alone, the chain binds its own port exactly
    # as before. Either way `publisher` below is a per-chain view with the old interface.
    # What a VIEWER needs to draw this chain without a static table: constellation, a
    # human label, the record length, and whether a search feeds it. All of it is already
    # known here -- the descriptor came from gnssSignal.hpp -- so publish it rather than
    # making the browser guess from stage names.
    _pub_desc = {"constellation": args.constellation or args.dr_constellation,
                 "carrier_hz": args.carrier_hz, "code_length": args.code_length,
                 "t_rec": args.code_length / args.chip_rate_hz,
                 "has_search": bool(detectors), "n_trackers": len(trackers),
                 "sigid": _sig.primary if args.signal else None,
                 "label": _sig.label if args.signal else chain_id,
                 "short": _sig.short if args.signal else chain_id,
                 "rf_band": _sig.rf_band if args.signal else band_id}
    if publisher is not None:
        publisher = publisher.register(chain_id, args.signal, band_id, _pub_desc)
    elif args.publish_port:
        publisher = FleetPublisher(args.publish_port, _log).register(
            chain_id, args.signal, band_id, _pub_desc)
    else:
        publisher = None
    _cls.tracker = resolve_prefix(args.cl_tracker, base) if args.cl_tracker else None
    _cls.combiner = resolve_prefix(args.cl_combiner, base) if args.cl_combiner else None
    _nav.cnav_combiner = resolve_prefix(args.cnav_combiner, base) if args.cnav_combiner else None
    _nav.inav_combiner = resolve_prefix(args.inav_combiner, base) if args.inav_combiner else None
    _nav.fnav_combiner = resolve_prefix(args.fnav_combiner, base) if args.fnav_combiner else None
    _nav.bcnav2_combiner = resolve_prefix(args.bcnav2_combiner, base) if args.bcnav2_combiner else None
    _nav.bcnav1_combiner = resolve_prefix(args.bcnav1_combiner, base) if args.bcnav1_combiner else None
    _nav.cnav2_combiner = resolve_prefix(args.cnav2_combiner, base) if args.cnav2_combiner else None
    gating = args.lat is not None and args.lon is not None

    # Almanac assist: BRDC (default; PRN-indexed, label-rot-proof) or the legacy TLE path.
    almanac_sats = None       # TLE mode: {prn: EarthSatellite}
    brdc_alm = None           # BRDC mode: {"mod", "eph", "eph_t"} for brdc_predict
    alm_sys = args.constellation or args.dr_constellation
    alm_min_prn = 19 if alm_sys == "C" else 1  # BDS-3 only: C1-18 = B1I, out of band
    if args.almanac:
        if not gating:
            _log("--almanac needs --lat/--lon; disabling")
            args.almanac = False
    if args.almanac and args.almanac_source == "brdc":
        try:
            import gnss_ephemeris as _alm_eph_mod
            when = _alm_now()
            # THROUGH THE RECEIVER (task #27 M3). The parse is MULTI-SYSTEM -- entries are
            # keyed (sys, prn) and every consumer filters at predict time -- so one store
            # serves GPS, Galileo and BeiDou together. Today each broker parses its own copy
            # of the same file and throws the other constellations away. Keyed by the day so
            # a replayed epoch and a live one do not share a store.
            brdc_alm = rx.brdc(
                ("brdc", when.strftime("%Y-%j")),
                lambda: {"mod": _alm_eph_mod,
                         "eph": _alm_eph_mod.parse_rinex_nav(_alm_eph_mod.fetch_brdc(when)),
                         "eph_t": _now()})
            n = sum(1 for k in brdc_alm["eph"] if k[0] == alm_sys and k[1] >= alm_min_prn)
            if n == 0:
                # ★ A PARSE THAT SUCCEEDS WITH ZERO SATS IS A FAILURE, not a result. Only an
                # EXCEPTION used to reach the TLE fallback below, so an almanac that simply has
                # nothing for this constellation left the chain hinting NOTHING -- indistinguish-
                # able, from the outside, from codes that do not correlate. GLONASS makes this
                # reachable by construction: gnss_ephemeris' RINEX parser filters on sysc in
                # "GEC", so every R record is skipped and an R broker parses a full file into
                # zero satellites, every time. Falling through to TLE is right for ANY
                # constellation the almanac does not cover.
                _log("almanac: BRDC parsed but has 0 %s sats%s -> treating as UNAVAILABLE and "
                     "falling back to TLE" % (alm_sys,
                                              " with PRN >= %d" % alm_min_prn
                                              if alm_min_prn > 1 else ""))
                brdc_alm = None
            else:
                _log("almanac: BRDC %s (%d %s sats%s) @ (%.4f, %.4f)"
                     % (args.almanac_source, n, alm_sys,
                        ", PRN >= %d" % alm_min_prn if alm_min_prn > 1 else "",
                        args.lat, args.lon))
        except Exception as e:
            _log("BRDC almanac unavailable (%s); falling back to TLE" % e)
    if args.almanac and brdc_alm is None:
        try:
            from gps_beamtrack import load_gps_satellites, predict_dopplers
            from gps_beamtrack import DEFAULT_TLE_URL
            almanac_sats = load_gps_satellites(args.tle or DEFAULT_TLE_URL)
            if args.tle_name_filter:
                import re as _re
                n0 = len(almanac_sats)
                almanac_sats = {p: s for p, s in almanac_sats.items()
                                if _re.search(args.tle_name_filter, s.name or "")}
                _log("tle-name-filter %r: %d/%d sats kept"
                     % (args.tle_name_filter, len(almanac_sats), n0))
            _log("almanac: loaded %d TLEs; predicting Doppler @ (%.4f, %.4f)"
                 % (len(almanac_sats), args.lat, args.lon))
        except Exception as e:
            _log("almanac unavailable (%s); falling back to search Doppler" % e)
            args.almanac = False

    # Dead-reckoned cp seeding: BRDC ephemeris + a search-solved receiver clock predict the
    # code phase of every sat the search can't (yet) see. State: eph = parsed BRDC records;
    # t0m = GPST of capture sample 0 pre-reduced mod the code period (full-GPST doubles
    # quantize at ~0.24 chips; the reduction keeps the mod arithmetic exact, and its own
    # constant error is COMMON to solve and seeds so it lands in the solved clock); clk =
    # solved receiver clock (chips, mod code period) at epoch clk_t; seeded/pin = which
    # PRNs are model-owned and when each was last re-anchored.
    # Signal capability for the dead-reckon seeder. B1C/B2a are BDS-3 only: the BDS-2 birds
    # (C1-C18) transmit B1I at 1561 MHz, 14 MHz outside our band, so there is NOTHING to
    # despread -- yet they are real satellites, genuinely overhead (MEO, 55 deg), with valid
    # BRDC ephemerides. Every gate we own therefore waves them through: visible, predicted,
    # healthy. Only capability excludes them.
    # ⚠️ BOUND UNCONDITIONALLY: the import below is conditional on --dead-reckon, so
    # without this the name does not exist on a search-only chain and anything that
    # merely MENTIONS it raises UnboundLocalError. Same class as `receiver_state`; the
    # synthetic fixture is what caught it, because every on-sky fixture runs dead-reckon.
    dr_eph_mod = None
    dr_min_prn = (args.dr_min_prn if args.dr_min_prn is not None
                  else (19 if args.dr_constellation == "C" else 1))

    # Signal-capability PRN gate (--signal-capability): the general block filter, fetched once.
    # Empty/failed lookup -> None (disabled) so a network hiccup can't dark the chain.
    #
    # ⚠️ THE REGISTRY OUTRANKS THE TLE TITLES (2026-08-31, KV -- the second time this source
    # ranking has had to be stated; prnmap.signal_incapable_prns' docstring records the first).
    # Capability comes from the IGS satellite-metadata SINEX (SVN -> block -> PRN-with-validity
    # -window, cached at ~/.cache/kotekan_gps/) when it can prove anything; the Celestrak block
    # names below are only the fallback for what the registry path does not model (GLONASS K
    # markers), because Celestrak "carries a multitude of errors, BeiDou worst of all". Both
    # sources fail OPEN: no proof of incapability keeps the satellite.
    _capable = None
    # `args.constellation or args.dr_constellation`: a chain configured via `signal:` leaves
    # --constellation unset (None), which made this guard a SILENT NO-OP for every production
    # chain -- the same resolution every other consumer in this file already uses (chain_id,
    # _pub_desc, alm_sys).
    _cap_sys = args.constellation or args.dr_constellation
    if args.signal_capability and _cap_sys in ("G", "R"):
        try:
            from gnss_broker import prnmap as _pm
            _incap = _pm.signal_incapable_prns(args.signal_capability)
        except Exception:
            _incap = set()
        if _incap:
            _capable = set(range(1, 64)) - _incap
            _log("signal-capability %s: %d PRN(s) excluded by the IGS registry (%s) -- "
                 "seeds + hints restricted to block-capable satellites (and with the "
                 "search's require_hint, an unhinted PRN is never scanned)"
                 % (args.signal_capability, len(_incap),
                    ", ".join(str(p) for p in sorted(_incap))))
    if _capable is None and args.signal_capability and _cap_sys in ("G", "R"):
        try:
            import gps_beamtrack as _bt
            # Pass THIS broker's TLE source: the GLONASS block marker lives in the glo-ops
            # names, and signal_capable_prns' default is the gps-ops group. Omitting it would
            # read GPS names for GLONASS slots and mark every satellite not-K -- an EMPTY
            # capable set, which the branch below then quietly turns into "filter disabled".
            _cap = _bt.signal_capable_prns(args.signal_capability,
                                           args.tle or _bt.DEFAULT_TLE_URL)
            if _cap:
                _capable = _cap
                _log("signal-capability %s: seeds+hints restricted to %d block-capable PRNs %s"
                     % (args.signal_capability, len(_capable), sorted(_capable)))
            else:
                _log("signal-capability %s: block lookup returned EMPTY -> filter DISABLED"
                     % args.signal_capability)
        except Exception as _e:
            _log("signal-capability %s: block lookup FAILED (%s) -> filter DISABLED"
                 % (args.signal_capability, _e))

    dr_state = None
    # PER-SAT SLOW BIAS b_sat (task #33, P2 step 1): the first receiver-state loop closed on
    # the P1 slope-fit. Chain-LOCAL on purpose -- PRN namespaces differ per constellation, so
    # unlike the clock this is not receiver-scope. Fed only from presence-gated tau (below);
    # consumed by the dead-reckon model at both its sites. With no --spectrum-endpoints
    # (every recorded transcript) it never updates and get() returns exactly 0.0, so the
    # consumption sites add a float zero and the replay digests are untouched.
    bsat = SatBiasFilter(gain=args.bsat_gain)
    joint_consume = {c.strip() for c in args.joint_consume.split(",") if c.strip()}
    # ⚠️ CONSTELLATION-QUALIFIED, and this cost a void first run (2026-08-10). The joint state
    # is keyed (constellation, prn) because GPS 4, Galileo 4 and BeiDou 4 are three unrelated
    # spacecraft. A bare "--joint-mask-prn 4" matched the PRN NUMBER on every chain, so masking
    # one vetted GPS satellite silently also masked two Galileo/BeiDou ones nobody had looked
    # at -- and the residual that came back was theirs, not the one under test. A bare number
    # is now REFUSED rather than guessed at: the test is about a specific satellite.
    joint_mask = set()
    for _tok in args.joint_mask_prn.replace(" ", "").upper().split(","):
        if not _tok:
            continue
        if _tok[0].isalpha():
            joint_mask.add((_tok[0], int(_tok[1:])))
        else:
            raise SystemExit(
                "--joint-mask-prn %r: qualify the PRN with its constellation (G4, E12, C33). "
                "GPS 4, Galileo 4 and BeiDou 4 are different satellites and a bare number "
                "masks all three -- which is how the first P2c run measured a satellite that "
                "was never chosen." % _tok)

    # -- P2C, THE ROTATING FORM (2026-08-10) ------------------------------------------------
    # A hand-picked PRN is a sample of ONE satellite at ONE geometry, and on a transit
    # instrument the sky moves out from under it within the hour. That is not a hypothetical:
    # the first two P2c runs disagreed outright -- run 1 coasted flat (-0.19 +- 0.44 chips
    # over 848 s), run 2 drifted to -2.70 over 900 s on the SAME satellite hours later -- and
    # a third produced nothing at all because the named PRN had set and was no longer in the
    # state. Two anecdotes and a null are not an acceptance test for the architecture.
    #
    # So the test drives itself: withhold the best-established satellite, coast it for
    # --joint-p2c-hold-s, record the residual-vs-age curve, release, rotate to one not
    # recently tested. What accumulates is a DISTRIBUTION across satellites and elevations.
    #
    # AND IT MEASURES q_b, which is the point beyond pass/fail. The coast residual's growth
    # with age IS the b_sat random walk this filter assumes; q_b = 0.013 is currently a guess
    # and run 2's drift says it is probably wrong. Rotation turns the least-justified
    # parameter in the state into a measured one.
    p2c = {"key": None, "t0": 0.0, "samples": [], "history": [], "n0": 0}

    def _p2c_pick(js):
        """The most-established satellite not tested in the last few rotations."""
        cand = [(js._n.get(k, 0), k) for k in js._idx
                if js._n.get(k, 0) >= args.joint_mask_after]
        if not cand:
            return None
        recent = {h["key"] for h in p2c["history"][-args.joint_p2c_skip:]}
        fresh = [c for c in cand if c[1] not in recent]
        return max(fresh or cand)[1]

    def _p2c_summary(js, t_now, why):
        """Close out one coast and log its residual-vs-age curve."""
        s = p2c["samples"]
        key = p2c["key"]
        if s:
            band = [(lo, hi, [r for a, r in s if lo <= a < hi])
                    for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 1200))]
            txt = " ".join("%d-%ds %+.2f(n%d)" % (lo, hi, statistics.fmean(v), len(v))
                           for lo, hi, v in band if v)
            _log("P2C %s END (%s): %d samples over %.0f s | %s | final b %+.3f sigma %.3f"
                 % (_p2c_name(key), why, len(s), s[-1][0], txt,
                    js.bias(key) if key in js._idx else float("nan"),
                    js.sigma(key) if key in js._idx else float("nan")))
            p2c["history"].append({"key": key, "n": len(s), "age": s[-1][0],
                                   "resid": [r for _, r in s], "why": why})
        else:
            _log("P2C %s END (%s): no samples" % (_p2c_name(key), why))
            p2c["history"].append({"key": key, "n": 0, "age": 0.0, "resid": [], "why": why})
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
                _log("P2C %s START: withholding after %d accepted updates (b %+.3f, "
                     "sigma %.3f) -- coasting %.0f s"
                     % (_p2c_name(k), p2c["n0"], js.bias(k), js.sigma(k),
                        args.joint_p2c_hold_s))
            return
        # ⚠️ THE SAT CAN AGE OUT FROM UNDER THE TEST. A withheld satellite is not fed, so
        # its _t_seen stops advancing and _drop() evicts it after max_age_s (900 s). A hold
        # at or beyond that would end every coast by eviction and silently report nothing --
        # so the default hold is 600 s, and eviction is reported as its own outcome rather
        # than looking like a completed test.
        if p2c["key"] not in js._idx:
            _p2c_summary(js, t_now, "sat dropped from state")
        elif t_now - p2c["t0"] >= args.joint_p2c_hold_s:
            _p2c_summary(js, t_now, "hold complete")

    def _p2c_hold(js, key):
        """True once this sat is BOTH masked and established -- see --joint-mask-after."""
        if args.joint_p2c_rotate:
            return key == p2c["key"] and key in js._idx
        return (key in joint_mask and key in js._idx
                and js._n.get(key, 0) >= args.joint_mask_after)

    def _joint_state(rx_, band, a):
        """The joint state IF it is fit to be consumed, else None (never raises: a consumer
        must degrade to the old estimator, not take the broker down).

        ⚠️ joint_receiver(), NOT joint(). THE READER AND THE WRITER MUST NAME THE SAME
        OBJECT. There are two states on the Receiver: joint(band) keys one per BAND (P2),
        joint_receiver() keys the single receiver-wide one (P3, the one with tau_band). The
        solve writes to joint_receiver(); this read used joint(), which is created EMPTY on
        first access and populated by nothing. So every P2b consumer -- all of them go
        through here -- was wired to a dead object: len(_idx) is 0, it never clears
        joint_min_sats, this returns None, and the caller silently falls back to the legacy
        estimator. FOREVER, and without a single log line, because falling back is the
        designed behaviour when the state is not yet fit to consume.

        That is the failure mode to watch for when a refactor introduces a second object
        with a near-identical name: consumer 1 has been "live" in the sense of being written
        and reviewed since 2026-08-09 and could never once have fired. Found 2026-08-10 only
        because consumer 3's shadow log printed nothing and the reason had to be chased
        (docs 11.31)."""
        try:
            js = rx_.joint_receiver(band, CODE_LEN, rereference=a.joint_rereference, gauge_mode=a.joint_gauge)
            if len(js._idx) < a.joint_min_sats:
                return None
            # A DEAF STATE IS NOT FIT TO CONSUME (2026-08-21). It rejects everything, so it
            # cannot correct itself, while predict() keeps integrating clk_rate -- observed
            # walking to clk +684 against a legacy 151 with sigma still reporting 0.040.
            # Every consumer's own delta bound happened to refuse it, which is luck, not
            # design: refuse it HERE, once, where "fit to be consumed" is decided.
            return None if getattr(js, "deaf", False) else js
        except Exception:
            return None

    if args.dead_reckon:
        if not args.almanac:
            _log("--dead-reckon needs --almanac; disabling")
        else:
            try:
                import gnss_ephemeris as dr_eph_mod
                dr_state = {"eph": None, "eph_t": 0.0, "t0m": None, "clk": None,
                            "clk_t": 0.0, "next": 0.0, "log_next": 0.0,
                            "pin": {}, "seeded": set()}
                if args.dr_clock_chips is not None:
                    # COLD START WITHOUT A SEARCH STAGE. The bootstrap below takes a median of
                    # measured code-phase residuals, so it needs satellites already tracking --
                    # but nothing tracks until something seeds it. The airspy node breaks that
                    # circle with its search stage; a GPS-disciplined F-engine breaks it with
                    # arithmetic instead, because the receiver clock offset is known a priori.
                    # The live EMA below refines this from the moment the first satellite locks
                    # -- BUT ONLY IF THIS BROKER HAS DETECTORS. The solve takes its residuals
                    # from `best`, which is filled exclusively from /get_detections, so with
                    # --detectors empty (the model-primary configuration the multi-constellation
                    # chains run: E5a/B2a have no blind acquisition at all, see
                    # docs/CHORD_MULTIBAND.md) `offs` is always empty, the EMA never runs, and
                    # the primed constant stands for the whole run. Say which case we are in:
                    # "self-corrects" is a promise the detector-less configuration cannot keep,
                    # and believing it would mean shipping a wrong clock indefinitely.
                    dr_state["clk"] = args.dr_clock_chips % float(args.code_length)
                    dr_state["clk_t"] = _now()
                    # ⚠️ A PRIME IS NOT A MEASUREMENT, AND THE EMA MUST NOT TREAT IT AS ONE.
                    # The bootstrap below snaps `clk` to the first solved median (`raw`) only
                    # when clk is None -- so priming, which exists purely to let a
                    # detector-less chain seed at all, ALSO silenced the snap on the chain
                    # that can measure. gps_l5 then EMA'd in from 0.0 at alpha=0.2 per cycle:
                    # measured on sky 2026-08-08, 0 -> 54.6 -> 74.0 -> ... -> 151 chips over
                    # ~40 s (0.8^n, the alpha exactly), during which EVERY satellite's
                    # integrity residual reads BAD by the walk-in error itself (+117..+125 at
                    # clk=29.97) and every co-hosted chain adopts a moving target. This flag
                    # keeps the prime for seeding and hands the clock back to the first real
                    # measurement.
                    dr_state["clk_primed"] = True
                    if args.dr_clock_drift is not None:
                        dr_state["drift"] = float(args.dr_clock_drift)
                        _log("dead-reckon: clock DRIFT primed %+.4f chips/s"
                             % dr_state["drift"])
                    _log("dead-reckon: receiver clock PRIMED %.2f chips = %.3f us (no search "
                         "stage; %s)"
                         % (dr_state["clk"], dr_state["clk"] / args.chip_rate_hz * 1e6,
                            "REPLACED outright by the first multi-sat solve" if detectors else
                            "NO detectors: this value stands until a same-band chain "
                            "contributes one (--dr-clock-adopt), and is FIXED for the run "
                            "without that"))
                _log("dead-reckon: BRDC cp seeding armed (%s, repin %.0f s%s)"
                     % (args.dr_constellation, args.dr_repin_s,
                        ", DRY RUN" if args.dr_dry_run else ""))
            except Exception as e:
                _log("dead-reckon unavailable (%s); disabled" % e)

    seeds = {}       # prn -> {"doppler_hz", "code_phase_chips", ...} (consensus)
    # prn -> PROMPT HOLD, fleet prompt power / live noise median, from the previous cycle's
    # fleet dict. Carried exactly like `status` above and for the same reason: the lock gate
    # runs before this cycle's fleet_dll. See the --lock-prompt-hold note for why the gate
    # needs a fold-independent term at all.
    _elem_arch_t = [0.0]   # last per-element archive append (--element-archive-every-s)
    _geom_post_t = [0.0]   # last sat-geometry post (#102, --post-sat-geometry)
    _elem_poll_t = [0.0]   # last /get_elements poll (--element-poll-every-s)
    # #57 step 3: the residual carrier rates the known-rate coherent fold derotates with.
    # Updated AFTER each cycle's fold from that cycle's record-stream fit, so the fold only
    # ever uses a rate estimated from EARLIER records -- causal by construction, which is
    # the entire difference between this estimator and the deep fold it replaces.
    _kcoh_rates = {}
    # Telemetry-walk estimator throttle (--estimator-every-s): last results + next-run time,
    # staggered per chain so five chains never walk in the same beat.
    _est_last = {"pcn0": None, "kcoh": None}

    # ── THE DLL STAGE'S PER-CYCLE PRODUCTS ────────────────────────────────────────────────
    # Diagnostics the DLL stage computes and LATER stages read: the fleet-coherent rows and
    # the known-rate coherent estimate. They used to be bare locals assigned deep inside the
    # DLL block, which had two consequences worth naming:
    #
    #   * the block could not be extracted. A name assigned only inside it and read outside
    #     has no binding in main(), so `nonlocal` is a SyntaxError -- the extraction is
    #     blocked not by the logic but by where the value happens to live.
    #   * with --dll-gain 0 the block never runs and the readers hit a NameError, not a
    #     missing value. Every reader already spells `(fcoh or {})`, so `None` is exactly the
    #     "not measured this cycle" they were written for.
    #
    # AN OWNER OBJECT rather than more locals, because ATTRIBUTE ASSIGNMENT NEEDS NO nonlocal
    # DECLARATION. That is the whole trick that lets the instrument polls below become
    # routines: a poll that sets `_dllp.fcoh` can live anywhere, where one that set `fcoh`
    # could only live in the frame that owned the name.
    class _DllProducts(object):
        """Per-cycle diagnostic products of the DLL stage. None means NOT MEASURED."""
        __slots__ = ("fcoh", "kcoh", "fcoh_n2", "spec_fit", "innov_pub", "report",
                     "deep_gate_eff", "run_est", "run_pcn0", "fleet", "pcn0",
                     "inst_hops", "admit_disp")

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
            # instance -> newest telemetry hop seen this cycle; the instance-stall and
            # axis-freshness watches read it after the polls have filled it.
            self.inst_hops = None
            # --presence-admit-displaced thresholds, bundled once per cycle so every
            # presence arm judges with the same numbers; None = the flag is off.
            self.admit_disp = None
            # ⚠️ `fleet` IS THE CYCLE'S CENTRAL STATE: prn -> the fleet DLL's per-satellite
            # row (present, q, disc, hop, ...). Nearly every stage reads it, which is exactly
            # why it spent so long as a bare local and why the carrier loop was able to assign
            # a sorted LIST over it for the rest of a cycle without anything noticing (see the
            # buglist's A-FIXED entry). As an attribute of its producer, that collision stops
            # being a bug you have to catch and becomes one you cannot write.
            self.fleet = None

    _dllp = _DllProducts()

    # ── THE DEAD-RECKON STAGE'S PER-CYCLE PRODUCTS ────────────────────────────────────────
    # `clk_now` is the receiver clock propagated to this instant. The SEEDING sub-stage
    # computes it; the JOINT-SHADOW diagnostic, which runs EARLIER in the same pass, also
    # reads it.
    #
    # ⚠️⚠️ SO THAT DIAGNOSTIC READS THE PREVIOUS CYCLE'S CLOCK, and always has. It prints
    # `legacy clk` (this value, one cycle stale) beside `joint clk` (this cycle's `_js.clk`),
    # which means anyone comparing those two numbers to judge the joint filter has been
    # comparing them ACROSS A CYCLE BOUNDARY -- a ~2 s lag times the clock rate. Log-only: it
    # feeds no control path. Naming it here rather than quietly fixing it, because changing
    # the value would move a number people have already reasoned from, and that deserves its
    # own commit and its own falsifier.
    #
    # It lives on an owner object for the same reason `_dllp` does: an attribute needs no
    # `nonlocal`, so the sub-stages that read and write it can be routines. As a bare local it
    # had NO BINDING IN main() AT ALL, which is what blocked extracting this stage at all.
    class _DrProducts(object):
        """Per-cycle derived values of the dead-reckon stage. None means NOT YET SOLVED."""
        __slots__ = ("clk_now", "raw_clk", "pd", "pd2", "offs", "la", "tag", "drift",
                     "t_code", "t_fc_abs", "rx_sib", "hold", "mod", "slew_cap", "slew_k",
                     "now_w", "t_eph_age", "t_now_abs")

        def __init__(self):
            self.clk_now = None
            # The RAW clock solve, before the quality gate accepts or rejects it. It used to
            # be a bare local called `raw` -- a name the ALMANAC stage also uses, for an
            # entirely unrelated quantity, in the same flat namespace. The two never collided
            # because each writes before it reads, but nothing made that safety visible, and
            # extracting either block silently removed the other's binding. It did exactly
            # that on 2026-08-26, and the gate caught it as a SyntaxError.
            self.raw_clk = None
            # The stage's working set, shared between its shell and its five sub-stages.
            # These were bare locals, which is what made the sub-stages inseparable from the
            # shell: `pd` (the per-satellite model predictions), `offs` (the per-satellite
            # code offsets the clock is solved from), `la` (the (l-a) receiver code-rate
            # bias), and the propagation constants the seeding stage needs.
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
            # The wall instant this dead-reckon pass is propagating TO. Read by every
            # sub-stage; it is the pass's own clock, not the cycle's frozen t0.
            self.now_w = None
            # The WALL epoch the ephemeris was actually evaluated at. cp_predicted propagates
            # range from THIS, never from t_now_abs -- under --dr-fengine-axis those are two
            # different clocks, and mixing them is the walkoff this project spent a day on.
            self.t_eph_age = None
            # THE AXIS-DERIVED ABSOLUTE CAPTURE TIME, produced by the dead-reckon stage and
            # read by four others (both rate feeds, the joint shadow, the spectrum fit).
            #
            # ⚠️ None, NEVER 0.0, until the first ephemeris refresh assigns it. A missing axis
            # time is UNKNOWN; a confident wrong timestamp is worse than a skipped
            # measurement. As a bare local initialised to 0.0 it killed chain threads through
            # the #85 spectrum stash on 2026-08-26 -- every consumer is guarded on `is not
            # None` for that reason, and new consumers must be too.
            self.t_now_abs = None

    _drp = _DrProducts()


    def cp_predicted(v, t_abs):
        """Physical code phase (chips) of the predicted signal at capture age
        t_abs, EXCLUDING the receiver clock. One predict_all per cycle: the
        range is propagated to other epochs through range_rate (fine over the
        few-second detection staleness), FROM THE EPOCH THE EPHEMERIS WAS
        ACTUALLY EVALUATED AT (_t_eph_age, wall) -- never from t_now_abs,
        which under --dr-fengine-axis is a different clock (see above). All
        mod arithmetic on small numbers (t0m is the sample-0 GPST pre-reduced
        mod the code period)."""
        t_tx = (dr_state["t0m"] + t_abs
                - (v["range_m"] + v["range_rate_mps"] * (t_abs - _drp.t_eph_age))
                  / dr_eph_mod.C_LIGHT
                + v["sat_clk_s"])
        return (t_tx % _drp.t_code) * args.chip_rate_hz

    def _track_ok(_p):
        """Is the TRACKER seeing this satellite RIGHT NOW?

        --joint-min-snr gates on the SEARCH SNR carried in `best`, which answers
        a different question: 'was there a detection', and a latched detection
        keeps answering yes for up to the 1276 s per-PRN revisit after the sky
        has moved on. On 2026-08-10 PRN 25 transited out of the primary beam
        (elevation RISING 45->65 deg) and kept presenting its last good scan's
        SNR while its tracker output was already uniform noise over the code;
        that measurement destroyed the joint state. The tracker's own view is
        the one the measurement actually depends on.

        deep_snr counts only when floor-cleared (coherence_s > 0), matching
        sig_of(): a floored deep sits near 7 on pure noise and would otherwise
        clear a 10-chip bar on nothing at all."""
        _row = _ctx.status.get(_p) or {}
        if float(_row.get("coherence_s", 0) or 0) <= 0.0:
            return False
        return (float(_row.get("deep_snr", 0) or 0) >= args.joint_min_deep_snr
                and float(_row.get("coh_frac", 0) or 0)
                >= args.joint_min_coh_frac)

    fe_axis = [None]  # (newest telemetry pow_hop, wall at its fetch) -- #83 the axis fix
    # ── THE FILTERED AXIS OFFSET (2026-08-23, the birth-epoch jitter fix) ──────────────
    # fe_axis re-samples the newest hop every poll, so t_now_abs inherits the PIPELINE
    # LAG's jitter (measured IQR ~59 ms) -- and the ephemeris range is evaluated on WALL
    # time, so the frame mismatch lands in every rebirth as lag_jitter x range_rate:
    # measured fleet-wide, routine birth steps scale with |dop| at 4.0e-4 chips/Hz =
    # 46 ms of epoch jitter (2,476 births), with ~5 s excursions clustering multi-sat
    # multi-chain every few minutes (749 births at 10-100 chips). The axis is DRIFT-FREE,
    # so (hop_time - wall) is physically a constant minus a strictly-positive fluctuating
    # lag: a MAX filter recovers the constant (NTP's minimum-delay idea) -- adopt any
    # LARGER offset instantly (a fresher sample saw less lag), decay slowly downward
    # (0.5 ms/s: follows real drift and NTP slew, rejects poll-to-poll lag churn), and
    # SNAP on a >2 s disagreement (an F-engine restart genuinely moves the axis; a max
    # filter must not ride a dead frame0 for hours).
    fe_off = [None, 0.0, 0, 0.0]
    # [filtered offset, wall of last update, consecutive-disagree count, candidate offset]
    # ⚠️ THE SNAP NEEDS PERSISTENCE (2026-08-23, measured the same evening the filter went in).
    # `_fh` is max(pow_hop) over the chain's CURRENT status rows -- a MAX OVER A CHURNING SET,
    # the same disease as the clock median and the mean gauge. The rows span ~1.2 s live and
    # up to ~6 s across chains, so when the freshest satellite drops out of one poll the max
    # falls back several seconds and returns on the next. Measured: 14 SNAPs in 50 min
    # oscillating between exactly two offsets 4.96 s apart, which IS the multi-sat multi-chain
    # birth-glitch population (~100 chips at 2 kHz). A bare 2 s threshold reads every dropout
    # as a real axis move, so the max-filter -- which would otherwise reject a stale sample
    # by construction -- gets overridden precisely when it is right. Require the disagreement
    # to REPEAT before believing it: a frame0 step persists, a dropout does not.
    _rf_last = [0.0]  # #8: wall of the last RF-health poll (rate-limits it)
    _est_next = [_now() + (hash(chain_id) % 5) * 8.0]
    _anchor_seen = [0.0]   # frame0 as first latched (see the re-check in the cycle loop)
    _anchor_chk = [0.0]    # wall time of the last anchor re-read
    _cls.seg_s = float(args.long_code_epoch_s) / max(int(args.long_code_segments), 1)
    _cls.spiral = ([0] + [v for n in range(1, int(args.long_code_segments) // 2 + 1)
                            for v in (-n, n)])[:max(int(args.long_code_segments), 1)]
    xband = resolve_prefix(args.xband_combiner, base) if args.xband_combiner else None
    _xb_resid = []   # rolling cross-band prediction residuals (Hz), shadow accumulation
    _xb_dir = os.path.dirname(args.state_file) if args.state_file else None
    # WHERE SIBLING STATE IS READ FROM. Deliberately independent of --state-file: a chain that
    # ADOPTS a clock has no reason to publish one (it has no estimate of its own to contribute),
    # so deriving the read directory from the write path would make --dr-clock-adopt a silent
    # no-op on exactly the chains that need it. Defaults to the write directory when only that
    # is given, so the common case needs one flag rather than two.
    _xb_read_dir = args.state_read_dir or _xb_dir

    def _fused_lo_ppm(dongle):
        # this band's own LO comes from _fuse_cached; a sibling's is read fresh from its file
        if state_w is not None and dongle == args.state_dongle:
            f = _fuse_cached(_now())
        elif _xb_dir:
            try:
                f = receiver_state.fuse_dongle(
                    receiver_state.read_dongle(_xb_dir, dongle, max_age_s=30.0,
                                               t_now=_now()),
                    floor_ppm=0.001, reject_sigma=5.0)
            except Exception:
                f = None
        else:
            f = None
        return (f.get("lo_ppm") if f and not f.get("all_outliers") else None)
    # CL K-SCAN (diagnostic, --cl-kscan-prn; default 0 = OFF, zero effect). The recurring
    # "CL despreads noise on ~40% of launches while fine_ms looks perfect" is the signature
    # of a WHOLE-SEGMENT (N x 20 ms) anchor error: fine is the residual AFTER round(), so an
    # error that is an exact multiple of the segment folds entirely into k and reports a
    # perfect margin. A single restart cannot confirm this (60% base rate), and any absolute
    # cross-check needs the TOW convention + slot mapping exact. A k-scan is CONVENTION-FREE:
    # it steps the seeded segment for ONE probe PRN through {k-2..k+2}, dwelling long enough
    # for the CL combiner's deep integration to respond, and the verify names which offset
    # despreads. If k+-N wins, the whole-segment bug is PROVEN and its magnitude N is known.
    # Two scan modes share the machinery:
    #   SEGMENT mode (default): offsets are whole segments k+N -- the whole-segment test.
    #     RESULT 2026-07-29: falsified. On a broken launch NOTHING in k+-2 despreads
    #     (incl. k+0); positive control on a working launch shows k+0 winning 39x. The
    #     segment pin is EXONERATED.
    #   FRACTIONAL mode (--cl-kscan-chips "0,0.25,-0.25,..."): offsets are CHIPS added to
    #     the seeded cp -- the comb/sub-chip test. CM/CL are chip-interleaved at 1.023 Mcps
    #     (one comb slot = 0.5 chip at the 511.5 kcps code), so a TDM comb-phase fault
    #     lands somewhere on a sub-chip grid. A fine grid rather than a bet on +-0.5
    #     exactly: slot parity and code phase COUPLE when the replica timeline shifts, and
    #     a half-chip code offset degrades ~6 dB rather than nulling -- so any partial
    #     despread (~half of CM's deep) stands far above the noise floor of ~2 and names
    #     the true offset.
    if args.cl_kscan_chips:
        _cls.kscan_seq = [float(x) for x in args.cl_kscan_chips.split(",") if x.strip()]
        _cls.kscan_frac = True
    elif args.cl_kscan_segs:
        # explicit segment list -- built for the FULL-75 sweep after the +-2 scan was
        # over-read as exoneration (it exonerated |N|<=2 ONLY; the anchor's startup
        # latency jitter is tens of ms, i.e. potentially several 20 ms segments)
        _cls.kscan_seq = [int(x) for x in args.cl_kscan_segs.split(",") if x.strip()]
        _cls.kscan_frac = False
    else:
        _cls.kscan_seq = [0, -1, 1, -2, 2]   # true k first (baseline), then neighbours
        _cls.kscan_frac = False
    _cls.kfmt = (lambda o: "c%+.2f" % o) if _cls.kscan_frac else (lambda o: "k%+d" % o)
    bp_pushed = {}        # prn -> utc0 of the bit_pred table last ATTACHED to a seed row. The
                     # combiner regenerates bit_pred once per EMIT (~1 Hz) but seeds push every
                     # --interval (0.25 s), so re-attaching each cycle is 75% redundant payload
                     # -- and the seed POST has a known too-big failure mode (~14.5k numbers,
                     # see --nav-bits-brdc). The tracker KEEPS its stored table on rows without
                     # nav_bits (u.has_bits guard), so skipping unchanged tables is free. This
                     # matters most at L5: 1 ms records make each table 4x an L1 pilot's.
    cp_held = set()  # PRNs whose cp anchor is FROZEN this cycle (locked -> DLL owns the residual)
    # ⚠️ NOT ASSIGNED UNTIL THE FIRST DEAD-RECKON REFRESH THAT HAS AN EPHEMERIS (:6213), and
    # SIX consumers outside that block read it -- the spec-fit archive/stash and the four
    # joint feeds. Before the first refresh they raised UnboundLocalError; five sit inside a
    # try that logs and continues, and the sixth (#85's spec_y stash) does not, so it KILLED
    # THE CHAIN THREAD. Found 2026-08-26 by the first replay of a fresh on-sky transcript --
    # the equivalence gate earning its keep on a startup-ordering race that production hides
    # because the ephemeris normally loads before the first spec fit lands.
    # None, never 0.0: a missing axis time is UNKNOWN, and feeding 0.0 to the joint filter
    # would be a confident wrong timestamp instead of a skipped measurement.
    _drp.t_now_abs = None
    fast_lock = threading.Lock()
    fast_prns = set()   # published by the policy cycle: who may be trimmed right now
    fast_tmpl = {}      # prn -> (the EXACT dict the cycle last posted, base cp before trim)
    fast_stats = {"updates": 0, "posts": 0, "skipped": 0, "last_err": "", "rail": 0}
    # #51 F3: the C++ fleet loop's seam. `_fleet_trim_nominal_hz` converts --dll-leak-present
    # (per UPDATE, at this process's own cadence) into the per-SECOND leak the controller
    # wants. 23.84 is the frame rate the trackers ship at, which is the rate the C++ loop
    # steps at -- so at these defaults the C++ loop runs the SAME per-update leak the Python
    # arm did, and the only thing that changed is how often it steps. That is deliberate: one
    # variable at a time.
    _fleet_trim_nominal_hz = 23.84
    _g3_ramp = RampTracker()
    # #93 SHADOW v2: the direct per-sat ADR span rate (carrier frame, Hz), captured on
    # the span-shadow path each poll: prn -> (y_phi_hz, t, n_rec). The rrate ROW stays
    # logged beside it -- the 08-25 F1 said the row is ~97% carrier-only; v2 measures
    # whether the RAW ADR observable does better against the trim ramp.
    _handover = TrimHandover(enabled=bool(args.fleet_trim_rebase_adjust))
    # THE HANDOVER (#79/#49, the precondition eec1d2f12 set for re-arming the DR chains).
    # The set the C++ loop is ACTUATING RIGHT NOW, i.e. what we last POSTED -- not what this
    # cycle is about to compute. The Python integrator stands down per-PRN against this, so
    # authority is never held by both arms and never by neither.

    def _fast_trim_loop():
        """Run the DLL integrator and the seed POST at --fast-trim-hz. See the notes above.

        ⚠️ IT SUBSTITUTES ONE FIELD INTO THE CYCLE'S OWN PAYLOAD and builds nothing itself.
        Rebuilding the seed here would silently drop whatever the policy cycle had put in it
        (carrier_trim_hz, the joint-state terms, nav bits), and a POST that drops a field
        ZEROES it at the tracker -- an actuator quietly undoing another loop is the worst
        failure this could have. So the cycle publishes the exact dict it posted plus the
        untrimmed code phase, and this only ever replaces code_phase_chips.
        """
        period = 1.0 / max(args.fast_trim_hz, 1e-3)
        while True:
            t0 = time.time()
            try:
                with fast_lock:
                    prns = set(fast_prns)
                    tmpl = dict(fast_tmpl)
                # ⚠️ DECODE ONLY THE ARMED PRNs. Without this filter the loop decodes every
                # PRN's comb on every iteration -- ~15 satellites x 10 instances x 4 windows
                # x 4 records of Python -- and then throws all but the armed ones away. It
                # cost a factor ~10 in achieved rate: 5 Hz requested, 1.5 Hz delivered, which
                # is BELOW the 1.94 Hz break-even, i.e. the loop was still losing to the drift
                # while looking like it was running. The rate is the whole point of this
                # thread; anything that silently caps it defeats it.
                fl = combdll.fleet_dll_comb(
                    telem_client, telem_chain, n_win=max(1, args.fast_trim_windows),
                    min_instances=args.dll_min_instances, k_sigma=args.dll_quality_sigma,
                    q_fallback=args.dll_quality_min, per_channel=False,
                    prns=prns or None) if prns else {}
                if not fl or not prns or not tmpl:
                    fast_stats["skipped"] += 1
                else:
                    posted = []
                    for prn in sorted(prns & set(fl) & set(tmpl)):
                        disc = fl[prn]["disc"]
                        # THE SAME conversion the policy cycle uses -- and now literally the
                        # same function, which it was not before (2026-08-15). This comment
                        # claimed "one convention, one place it can be wrong" while the
                        # expression sat inline in TWO places; combdll.dll_integrate is that
                        # one place, and the C++ fleet loop's gnss::dll_integrate is its twin,
                        # compared byte-for-byte by scripts/gnss/fleetdll_gate.py.
                        with fast_lock:
                            t_new = combdll.dll_integrate(
                                _dls.trim.get(prn, 0.0), disc, args.dll_gain,
                                args.dll_leak_present, 3.0, args.dll_spacing)
                            if abs(t_new) >= 2.999:
                                fast_stats["rail"] += 1
                            _dls.trim[prn] = t_new
                            d0, base_cp, base_aref = tmpl[prn]
                            d = dict(d0)
                        d["code_phase_chips"] = (base_cp + t_new) % args.code_length
                        # §4.6: the phase moves with the trim or the tracker ignores it.
                        if base_aref is not None:
                            _tmod = ((LC_SEG * CODE_LEN) if LC_SEG > 1 else CODE_LEN)
                            d["code_phase_at_ref_chips"] = (base_aref + t_new) % _tmod
                        posted.append(d)
                        fast_stats["updates"] += 1
                    if posted:
                        for t_ep in trackers:
                            try:
                                _post("%s/set_seeds" % t_ep, posted)
                            except Exception as e:
                                fast_stats["last_err"] = "%s: %s" % (t_ep, e)
                        fast_stats["posts"] += 1
            except Exception as e:                 # a control thread must never take the
                fast_stats["last_err"] = str(e)    # broker down; the cycle still runs the loop
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    if args.fast_trim_hz > 0.0 and telem_client is not None:
        threading.Thread(target=_fast_trim_loop, daemon=True).start()
        _log("FAST-TRIM: code loop at %.1f Hz on %d windows (%.2f s), policy stays on the "
             "%.1f s cycle. Break-even vs the measured 0.121 chips/s drift is 1.94 Hz."
             % (args.fast_trim_hz, args.fast_trim_windows,
                args.fast_trim_windows * 4 * 0.0104857, args.interval))
    elif args.fast_trim_hz > 0.0:
        _log("FAST-TRIM requested at %.1f Hz but --telem-gather is not set -- staying on the "
             "policy cycle (it reads the gather store, never REST)." % args.fast_trim_hz)
    last_dets = []      # most recent raw /get_detections, re-served by the publisher so
                        # the viewer has ONE origin for both search and combiner data
    dr_untrusted = {}      # prn -> reason: the model is WRONG for this sat; use the search
    dr_bad = {}            # prn -> consecutive model-health failures (persistence, not a hair trigger)
    innov_hist = {}       # prn -> [(t, innov_chips)] -- #83 2(d): served, never consumed here
    minnov_hist = {}      # prn -> [(t, minnov_chips)] -- #83 P3-3a: the MODEL innovation
                          # (detection vs the joint state's clk + b_sat + tau, prior-state),
                          # the model-primacy flip gate's number; served, never consumed
    mp_flipped = set()    # #83 P3-3b: PRNs currently MODEL-PRIMARY (seeds from the dr-slew
                          # path; detections feed filter/innovations/referee only). Written
                          # ONLY by _mp_update below; consumed by the det loop and the dr
                          # loop's eligibility guards.
    mp_last_det = {}      # prn -> t of the last detection seen while flipped (starve exit)
    mp_cooldown = {}      # prn -> t of the last EXIT: no re-entry for 300 s (G23 measured
                          # enter->integrity-exit->enter twice in 5 min -- churn, not
                          # information; the sky does not change in 20 s)
    rate_prev_hop = {}  # prn -> last pow_hop used by the carrier loop (continuity gate)
    rate_prev_val = {}  # prn -> last rate residual (slew gate: catches f_ref re-pins)
    rrate_prev_hop = {}  # the rrate feed's OWN continuity state (#40): it may read different
    rrate_prev_val = {}  # fields (deep_rate_full_*) than the trim loop, so sharing the trim
                         # loop's prev dicts would corrupt both gates
    rr_kcoh_fed = {}     # {"last": <the kcoh dict object last fed>} -- the estimator is
                         # THROTTLED (_run_est), so _est_last serves the same dict for
                         # several cycles; re-feeding it would count one measurement as
                         # many and the filter would grow confident on repetition.
    rate_unit_hop = [0]  # [emit spacing in hops], LEARNED -- see rate_residuals' continuity gate
    # EXPLAIN-APPLY-VERIFY constants (2026-07-22, the robust replacement for gate tuning):
    # a residual can only be blamed for a sat's decoherence if it is big enough to null the
    # coherent window -- |resid| >= ~1/(2*T_emit) = 0.5 Hz at the 1 s emits every overlay
    # chain runs (L2C's 4 s window makes 0.5 conservative there: only delays acceptance).
    # Below this bar a stable residual does NOT explain a dark sat, and the carrier loop is
    # the wrong tool (that is the refade/watchdog's territory: nh misalignment, zombie
    # anchors). VERIFY_EMITS bounds the hypothesis: heal within 3 emits or be reverted.
    CARRIER_EXPLAIN_HZ = 0.5
    CARRIER_VERIFY_EMITS = 3
    # f_ref TRIM-BLEED shadow (2026-08-02): a converged, coherent, STANDING carrier trim means
    # f_ref was pinned off-true at acquisition and the sub-fence offset froze in (the AGE re-pin
    # keeps the slope but never re-adopts the seed) -- so the despread replica runs at f_ref while
    # the true carrier is f_ref+ctrim, costing sinc(ctrim*T_rec) (L2C's 20 ms records: 0.13-0.26
    # dB). The fix is an occasional gated re-adopt; this SHADOW logs where it WOULD fire, to
    # validate the trigger on live data BEFORE any tracker change (the fleet-collapse lesson: a
    # carrier correction that fires on the wrong sat is catastrophic -- see [[carrier-loop-
    # absorbing-state]] alias-escape v1/v2). Log-only; takes NO action.
    # ARMED action (--carrier-bleed): when a candidate fires, zero car_trim and flag the tracker to
    # re-adopt the seed (f_ref = dop, phase-continuous) -- folding the frozen offset into f_ref.
    det_fresh = {}      # prn -> (ref_hop, walltime) of the last NEW detection (alias escape)
    for _spec in os.environ.get("GNSS_TRIM_FORCE", "").split(","):
        # when the PRN is first SEEDED (a startup preload would be swept by the not-in-
        # seeds trim cleanup before the sat ever seeds). Reproduces the alias-capture
        # regime on the replay bench: BOOTSTRAP itself converges a -60 Hz NCO error to
        # the -50 Hz alias (the estimator reads the error mod 1/(2*T_rec)).
        if ":" in _spec:
            _p, _v = _spec.split(":")
            _carrier.trim_force[int(_p)] = float(_v)
            _log("TRIM FORCE (bench): PRN %s armed, car_trim %+.1f Hz at first seed"
                 % (_p, float(_v)))
    dop_rate_fitted = {} # prn -> the fitted rate actually seeded (for the log)
    dop_rate_rejected = {} # prn -> (fitted, model) when the fit disagreed with the model
    cp_rate_rejected = {}  # prn -> (fitted, pooled-clock) chips/s, #96 cross-check

    def cp_to_seed_currency(pts, dop_seed, dop_rate=0.0):
        """Re-express search cp0 points in the CURRENT seed's Doppler currency.

        dop_rate (Hz/s, 2026-07-18): subtract the KNOWN sky-doppler curvature from each
        point, in LOCAL baseline coordinates (t_i - t_anchor, anchored at the latest point):
        -0.5*k*dop_rate*dt^2, k = chip*sgn/fc. Without it, the drift of the true doppler
        across the fit baseline leaves a quadratic whose LS slope reads as a phantom (l-a)
        of chip*|dop_rate|*T/(2*fc) -- measured +0.0157 ppm per Hz/s on L2C (T=33 s), the
        per-sat spread of the +0.022 ppm phantom that railed the L2C DLL.
        ⚠️ Do NOT instead re-project per-point at a drifting dop model: (dd_i - dop_model(t_i))
        rides the ABSOLUTE t_abs multiplier, so a 4e-4 Hz/s model error becomes a 3 ppm slope
        poison at t_abs 10^4 s (measured, 2026-07-18 v2 shadow broker: fits exploded +0.5 to
        +3.1 ppm). The scalar dop_seed is constant per call precisely so t_abs cancels in the
        slope; only the LOCAL dt^2 term (bounded by T^2, ~0.1 chip) is safe to correct.

        The search anchors cp0 to absolute sample 0 through its own measured Doppler:
        cp0 = cp_local - t_abs*f_chip*(sign*dop_det/f_carrier). That projection multiplies
        per-detection Doppler noise (~10-25 Hz between scans) by the FULL RUN AGE --
        ~0.65 chips per Hz per 1000 s at L1 -- so the fit's input scatter (and thus its
        slope noise) grows linearly with run time: +-0.7 chips/s in the first minutes,
        +-10 chips/s by t~17 min (measured 2026-07-11; the root cause of every
        post-startup deep/incoherent collapse: the seeded rate sweeps the despread chips
        off-peak within one integration window). The tracker's replica applies the same
        projection FROM THE SEED's Doppler, so the one consistent currency is dop_seed:
        adding back each point's own drift term (exact cancellation -- same formula, same
        numbers the search subtracted) and re-projecting with dop_seed leaves noise that
        never multiplies t_abs. The fitted slope keeps the residual convention
        (l-a = slope/f_chip): dop_seed tracks the true Doppler to ~Hz, so the geometry
        stays fed-forward, minus the noise."""
        out = []
        h_anchor = pts[-1][0] if pts else 0
        for hh, cc, dd in pts:
            t_abs = hh / args.hops_per_sec
            corr = t_abs * args.chip_rate_hz * (args.code_doppler_sign
                                                * (dd - dop_seed) / args.carrier_hz)
            dt = (hh - h_anchor) / args.hops_per_sec
            corr -= (0.5 * args.chip_rate_hz * args.code_doppler_sign
                     * dop_rate / args.carrier_hz * dt * dt)
            out.append((hh, (cc + corr) % CODE_LEN))
        return out

    def sig_of_last(rec):
        """Hold-on-lock gate metric from the (previous-cycle) combiner status: the INCOHERENT
        amp_snr, OR the FLOOR-CLEARED deep_snr. deep used to be excluded outright -- its
        off-peak value IS the nav-wipe rectification floor (~7 sigma), far above lock_snr,
        so a deep gate would freeze every noisy first fit -- but since 2026-07-12 the
        combiner only reports coherence_s > 0 when a ladder rung beat its rectification
        floor by 2x, so coherence_s > 0 certifies the deep as a real coherent detection.
        Accepting certified deep un-traps sharp-ACF (BOC) signals: seed jitter modulates
        their per-record amplitude (the sharp peak turns code jitter into amplitude
        variance), which suppresses the moment-debiased amp_snr below every gate while
        the coherent sum stays strong -- without the deep path they can never reach hold,
        and the jitter that caused it never stops (observed: B1C amp ~1-8 / deep 90-150)."""
        if not rec:
            return 0.0
        amp = float(rec.get("amp_snr", 0) or 0)
        if float(rec.get("coherence_s", 0) or 0) > 0.0:
            return max(amp, float(rec.get("deep_snr", 0) or 0))
        return amp
    # THE RECEIVER CLOCK-FREQUENCY BIAS, as one object rather than five names travelling
    # together through this namespace. Its invariant -- a solved bias nobody has measured for
    # minutes must WIDEN the search rather than narrow it -- lives with it now; see
    # gnss_broker/clockbias.py for why that is counter-intuitive and what it cost.
    _cb = ClockBias()
    n_sib = 0              # sibling sat count folded into the last bias fusion (log only)
    if args.code_bias_init is not None:
        _cb.code_ema = args.code_bias_init * 1e-6
        _log("code-rate clock offset warm-started at %+.3f ppm (--code-bias-init)" % (_cb.code_ema * 1e6))
    elif args.code_bias_file:
        try:
            with open(args.code_bias_file) as f:
                _cb.code_ema = float(f.read().strip()) * 1e-6
            _log("code-rate clock offset loaded %+.3f ppm from %s" % (_cb.code_ema * 1e6, args.code_bias_file))
        except Exception:
            pass
    # CLOCK -> CALIBRATION + ALARMS (2026-07-19 audit rec D): the two receiver-clock loops
    # were built for a 2.6 ppm wandering TCXO; on the GPSDO they track flat constants
    # (measured all day: per-chain bias -152/-18/+30 Hz +-3, l-a 0.02-0.10 ppm). Warm-start
    # the carrier bias from its persisted file exactly like l-a -- this marks the bias
    # SOLVED from cycle 1, so the first-seed gate opens and the search margins start NARROW
    # (most of the remaining Tier-2 burn-in). The EMAs keep running purely as MONITORS: a
    # drift beyond the alarm bar means real hardware news (GPSDO unlock, thermal event),
    # not something to silently absorb.
    _cb.cal = None  # startup calibration value, Hz (drift-alarm reference)
    if args.clock_bias_file:
        try:
            with open(args.clock_bias_file) as f:
                # format: "<bias_hz> [n_sats] [unix_ts]" -- extended for --clock-bias-siblings;
                # the extra fields are ignored here (warm-start wants only the value).
                _cb.ema = float(f.read().split()[0])
            _cb.cal = _cb.ema
            _log("clock-freq bias warm-started %+.1f Hz from %s (margins narrow, seeding "
                 "enabled from cycle 1)" % (_cb.ema, args.clock_bias_file))
        except Exception:
            pass
    _clk_persist_t = [0.0]         # last clock-bias-file write (10 s rate limit)
    _cb.meas_t = _now()     # last multi-sat bias measurement (stale-rescue clock;
                                   # birth-stamped so warm-start gets a full grace window)
    _cb.stale = False             # solved-but-unmeasured for > --bias-stale-s
    _cb.available = False         # is ANY usable bias in hand (own or fused)? S2d gate
    # Fuse at most once a second and cache: the state files themselves only republish at
    # 1 Hz, so fusing at the broker's 5 Hz cycle would re-read the same bytes four times
    # for the same answer.
    _fus_cache = [0.0, None]
    _fus_seen = [False]           # have we EVER had a fused state? startup vs fault

    def _fuse_cached(t_now):
        if state_w is None or not _state_dir or not args.state_fuse:
            return None
        if t_now - _fus_cache[0] < 1.0:
            return _fus_cache[1]
        _fus_cache[0] = t_now
        try:
            _fus_cache[1] = receiver_state.fuse_dongle(
                receiver_state.read_dongle(_state_dir, args.state_dongle,
                                           max_age_s=30.0, t_now=t_now),
                floor_ppm=args.state_fuse_floor_ppm,
                reject_sigma=args.state_fuse_reject_sigma)
        except Exception:
            _fus_cache[1] = None
        return _fus_cache[1]

    # S2 observer (write-only). Import is local + tolerant so a missing/broken module can
    # never stop a broker from starting: a diagnostic riding in a live receiver's control
    # loop must fail to nothing, not fail loudly.
    state_w = None
    _state_dir = None
    # ⚠️ BOUND UNCONDITIONALLY, because the import below is conditional. Without this the
    # name simply does not exist when --state-file is unset, and anything that so much as
    # MENTIONS it -- the context construction does -- raises UnboundLocalError on a perfectly
    # ordinary configuration. The consumers are all guarded on `state_w`, so None is the
    # correct "this receiver publishes no state" value.
    receiver_state = None
    if args.state_file:
        try:
            import receiver_state
            _state_dir = os.path.dirname(args.state_file) or "."
            state_w = receiver_state.StateWriter(
                args.state_file,
                chain=os.path.basename(args.state_file).rsplit(".", 1)[0],
                dongle=args.state_dongle or "unknown",
                carrier_hz=args.carrier_hz, log=_log,
                flush_s=args.state_flush_s)
            _log("receiver-state export -> %s (dongle %s, %.1f s) [WRITE-ONLY: no estimate "
                 "or seed consumes this yet]"
                 % (args.state_file, args.state_dongle, args.state_flush_s))
        except Exception as e:
            _log("receiver-state export DISABLED: %s" % e)
            state_w = None
    # NAV-DECODE health export (viewer surface + the decoded-eph fallback's status hook).
    dhw = None
    if args.decode_health_file:
        try:
            import decode_health
            dhw = decode_health.DecodeHealthWriter(
                args.decode_health_file,
                chain=os.path.basename(args.decode_health_file).rsplit(".", 1)[0],
                sys=alm_sys, log=_log)
            _log("decode-health export -> %s (sys %s)" % (args.decode_health_file, alm_sys))
        except Exception as e:
            _log("decode-health export DISABLED: %s" % e)
            dhw = None
    # cnav runs in a GPS band broker; label by carrier so L2C and L5 stay distinct in the file.
    _nav.cnav_sig = ("GPS_L2C_CNAV" if abs((args.carrier_hz or 0) - 1227.6e6) < 5e6
                 else "GPS_L5_CNAV")

    def _dh_obs(sig, prn, h, eph_obj, xc):
        """One decode-health observation, uniform across decoders. `eph_obj` is the extracted
        ephemeris object (None until this decoder has a full set); `xc` the BRDC-xcheck suffix
        (dpos parsed out). count is whatever monotonic decode counter the health dict carries.
        Never raises."""
        if dhw is None:
            return
        try:
            cnt = (h.get("words") or h.get("decoded") or h.get("decoded_sf")
                   or h.get("pages") or 0)
            syn = h.get("synced")
            if syn is None:
                syn = (h.get("words") or 0) > 0 or eph_obj is not None
            dhw.observe(sig, prn, _now(), count=cnt, synced=bool(syn),
                        eph=(eph_obj is not None), dpos_m=_dh_dpos(xc))
        except Exception:
            pass

    # ---- Decoded-eph BRDC fallback (decoded_eph.py). Registry: for each decoder that may be
    # armed in THIS broker, (signal, sys, sv_position fn, toe-field, ephemeris getter). Built once;
    # the getters read the live decoder vars (inav/fnav/... reassigned in the loop) at call time.
    _decfb = None
    if args.decoded_eph_fallback or args.decoded_eph_fallback_force:
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
    # (signal, sys, sv_pos, toe_field, lambda -> live decoder or None)
    _dec_reg = [
        ("GPS_L1_LNAV",    "G", _svp_lnav, "toe", lambda: _nav.navbits) if _decfb else None,
        ("GAL_E1B_INAV",   "E", _svp_inav, "t0e", lambda: _nav.inav) if _decfb else None,
        ("GAL_E5AI_FNAV",  "E", _svp_fnav, "t0e", lambda: _nav.fnav) if _decfb else None,
        (_nav.cnav_sig,        "G", _svp_cnav, "toe", lambda: _nav.cnav) if _decfb else None,
        ("BDS_B1C_BCNAV1", "C", _svp_bc1,  "t_oe", lambda: _nav.bcnav1) if _decfb else None,
        ("BDS_B2A_BCNAV2", "C", _svp_bc2,  "t_oe", lambda: _nav.bcnav2) if _decfb else None,
    ] if _decfb else []
    _dec_reg = [r for r in _dec_reg if r]

    def _decoded_entries(now_w):
        """Build decoded_eph entries (sys, prn, signal, eph, sv_pos, toe_gpst) from whichever
        decoders are armed. toe_gpst is reconstructed in NOW's week from the eph's sow field.
        Never raises -- a bad decoder/eph is skipped, not fatal."""
        ents = []
        if not _decfb:
            return ents
        gpst_now = _decfb.gpst_of_utc(datetime.fromtimestamp(_drp.now_w, tz=timezone.utc))
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
    # The long/overlaid code the TRACKERS despread, in primary periods and in seconds. The
    # time-assist below computes WHICH period a seed's cp sits in rather than searching it;
    # both constants used to be L2C CL's (75, 1.5 s), which silently pinned every other signal
    # to period 0. GPS L5 Q5 with NH20 baked in is (20, 0.02 s) -- and without this the CHORD
    # trackers, which despread the 204600-chip NH code while the search acquires the 10230-chip
    # primary, get a seed that is right 1 time in 20.
    LC_SEG = int(args.long_code_segments)
    LC_EPOCH = float(args.long_code_epoch_s)
    # Search-Doppler record-alias quantum, 1/(2*t_rec): see the DETECTION ALIAS FOLD
    # in the seeding loop. 500 Hz on the 1 ms bands (never confused), 125 E1C, 50 B1C,
    # 25 L2C -- the alias-severity ranking that matches where zombie births appear.
    Q_ALIAS_HZ = 0.5 * args.chip_rate_hz / CODE_LEN
    if args.hold_max_dop_hz is None:
        args.hold_max_dop_hz = 0.1 * args.chip_rate_hz / CODE_LEN
        _log("hold-max-dop-hz auto: %.1f Hz (0.1 cycle per %.0f ms record)"
             % (args.hold_max_dop_hz, 1e3 * CODE_LEN / args.chip_rate_hz))
    if args.dop_max_rate_hz is None:
        # The one-cycle rate limit MUST sit well above the fence. A clamp TIGHTER than the
        # fence is a silent disaster: every legitimate fence step gets clamped back inside the
        # fence, the freeze branch then keeps the old Doppler, and the seed Doppler NEVER
        # UPDATES -- it goes permanently stale and the C/N0 falls all the way back to baseline.
        # (Measured 2026-07-14 with a fixed 5 Hz clamp against B1C's 10 Hz fence. A safety net
        # that quietly disables the mechanism it is protecting is worse than none.)
        args.dop_max_rate_hz = max(5.0, 3.0 * args.hold_max_dop_hz)
    _log("dop-max-rate-hz (safety net, one cycle): %.1f Hz -- fires only on a bad model"
         % args.dop_max_rate_hz)
    # Reset the cp-fit history across a snapshot gap larger than this (re-acquisition). TIME-
    # based, not hop-based: a fixed hop count silently scales with the band's hop rate (the L1-era
    # 2e6 hops = 16 s at 125 kHz but only 2 s at the L5 front end's 1 MHz -- shorter than the L5
    # search's snapshot cadence, so the history reset every cycle and the fit NEVER fired: no
    # empirical code rate, carrier-aiding quantization * seed staleness walked the prompt chips
    # off-peak. The 2026-07-04 L5 signature: strong search, trackers at the floor, 0 cp-fits).
    MAX_GAP_HOPS = args.fit_gap_s * args.hops_per_sec
    HIST_LEN = args.fit_hist_len  # snapshots kept for the slope fits (--fit-hist-len)
    _log("detectors=%d trackers=%d combiner=%s interval=%.2fs gating=%s"
         % (len(detectors), len(trackers), combiner, args.interval, gating))
    _log("trackers: %s" % (trackers if len(trackers) <= 6
                           else "%s ... %s (%d)" % (trackers[0], trackers[-1], len(trackers))))

    # SEED CURRENCY AUDIT state (#39 follow-up): last SHIPPED tuple per PRN, so each POST
    # can be compared against the previous one in physical units. Keys are the 5 fields
    # dr_seed_phys consumes -- nav_bits etc. are deliberately not retained.
    seed_audit_prev = {}
    # Broker start, for the joint feed warmup (--joint-feed-warmup-s). Wall clock on
    # purpose: the warmup guards against ESTABLISHMENT-PHASE garbage, and establishment
    # runs on wall time regardless of what the transcript clock replays.
    broker_t0 = time.time()






























    # ── THE STAGE INTERFACE ───────────────────────────────────────────────────────────────
    # What a stage needs that is not its own local state, named at last. The 29 stages read
    # 210 distinct free names out of this function between them and not one declared which;
    # a stage cannot move into its own module until its set is written down. See
    # gnss_broker/context.py for the stable/per-cycle split -- it is load-bearing, not
    # organisational.
    def sig_of(r):
        # deep counts only when floor-cleared (coherence_s > 0): a floored deep (~7)
        # otherwise keeps phantom coasts alive forever, exactly like raw |A| did.
        amp = float(r.get("amp_snr", 0) or 0)
        if float(r.get("coherence_s", 0) or 0) > 0.0:
            return max(amp, float(r.get("deep_snr", 0) or 0))
        return amp

    _ctx = ChainContext(
        args=args, band_id=band_id, chain_id=chain_id, code_len=CODE_LEN,
        telem_chain=telem_chain, base=base, alm_sys=alm_sys, alm_min_prn=alm_min_prn,
        lc_seg=LC_SEG, lc_epoch=LC_EPOCH,
        rx=rx, publisher=publisher, telem_client=telem_client, detectors=detectors,
        dll_combiners=dll_combiners, spectrum_endpoints=spectrum_endpoints,
        n2_combiners=n2_combiners, last_dets=last_dets,
        decfb=_decfb, decfb_log_t=_decfb_log_t, dr_bad=dr_bad, bp_pushed=bp_pushed,
        fe_axis=fe_axis, fe_off=fe_off,
        spec_writer=_spec_writer, state_dir=_state_dir, xb_read_dir=_xb_read_dir,
        innov_hist=innov_hist, minnov_hist=minnov_hist, p2c=p2c,
        dop_rate_fitted=dop_rate_fitted, dop_rate_rejected=dop_rate_rejected,
        cp_rate_rejected=cp_rate_rejected,
        dll_hop_window=dll_hop_window, deep_gate=_deep_gate, dg_auto_last=_dg_auto_last,
        est_next=_est_next,
        sig_of=sig_of, combiner=combiner, gating=gating, capable=_capable,
        receiver_state=receiver_state, alm_now=_alm_now, cb=_cb,
        almanac_sats=almanac_sats, brdc_alm=brdc_alm, det_fresh=det_fresh,
        state_w=state_w, clk_persist_t=_clk_persist_t,
        car=_carrier, wd=_watchdog, nho=_nho, dls=_dls, hold=_hold, cpt=_cpt, rf=_rf, nav=_nav, cls=_cls, qpop=_qpop, brown=_brown, latch=_latch, saw=_saw,
        prnmap=_prnmap,
        trackers=trackers, joint_consume=joint_consume, broker_t0=broker_t0,
        dr_eph_mod=dr_eph_mod, dr_min_prn=dr_min_prn,
        hist_len=HIST_LEN, max_gap_hops=MAX_GAP_HOPS, q_alias_hz=Q_ALIAS_HZ,
        carrier_explain_hz=CARRIER_EXPLAIN_HZ, carrier_verify_emits=CARRIER_VERIFY_EMITS,
        fuse_cached=_fuse_cached, cp_to_seed_currency=cp_to_seed_currency,
        dh_obs=_dh_obs, cp_predicted=cp_predicted, joint_state=_joint_state,
        track_ok=_track_ok, p2c_tick=_p2c_tick, p2c_hold=_p2c_hold,
        decoded_entries=_decoded_entries,
        sig_of_last=sig_of_last,
        dllp=_dllp, drp=_drp, handover=_handover, adm_gate=_adm_gate, g3_ramp=_g3_ramp,
        seeds=seeds, dr_state=dr_state, bsat=bsat, cp_held=cp_held,
        dr_untrusted=dr_untrusted,
        est_last=_est_last, kcoh_rates=_kcoh_rates, rf_last=_rf_last,
        elem_arch_t=_elem_arch_t, elem_poll_t=_elem_poll_t, geom_post_t=_geom_post_t,
        mp_cooldown=mp_cooldown, mp_flipped=mp_flipped, mp_last_det=mp_last_det,
    )

    while True:
        # Start of cycle: sample the frozen cycle clock ONCE (see the _Transcript note).
        # Every `_now()` below this returns exactly t0, so the whole pass -- every gate,
        # every age, every EMA -- evaluates at one instant instead of smearing over the
        # cycle's own processing time.
        t0 = _TR.tick()
        _ctx.begin_cycle(t0=t0)
        t_wall = time.time()   # REAL clock, kept solely for the sleep below
        # 1. collect best-SNR detection per PRN across all detection sources
        best = {}  # prn -> (snr, dop, cp, ref_hop, nh, cp_long, cp_at_ref)
        _ctx.begin_cycle(best=best)
        for d_ep in detectors:
            try:
                dets = _get("%s/get_detections" % d_ep)
                if dets:
                    last_dets[:] = dets   # published verbatim (see FleetPublisher)
            except Exception as e:
                _log("get_detections %s failed: %s" % (d_ep, e))
                continue
            for d in dets:
                prn, snr = int(d["prn"]), float(d["snr"])
                if snr < args.acquire_snr:
                    continue
                if prn not in best or snr > best[prn][0]:
                    best[prn] = (snr, float(d["doppler_hz"]), float(d["code_phase_chips"]),
                                 int(d.get("ref_hop", 0)), int(d.get("nh", -1)),
                                 float(d.get("code_phase_long_chips", -1.0)),
                                 float(d.get("code_phase_at_ref_chips", -1.0)))
        # Remember each PRN's REPORTED alignment + the hop it was measured at, to echo back as
        # that PRN's own nh hint. Gated on the same acquire_snr as the detection itself: a
        # marginal detection's nh is a coin flip, and a wrong hint narrows the scan AWAY from
        # the truth (recoverable -- the hint expires and the full scan returns -- but it costs a
        # revisit, so do not feed it noise).
        for _p, _b in best.items():
            if _b[4] >= 0 and _b[3] > 0:
                _nho.seen[_p] = (_b[4], _b[3])
        for _p in list(_nho.seen):
            if _p not in seeds and _p not in best:
                del _nho.seen[_p]          # sat set: stop hinting a PRN we no longer track

        for _p, _b in best.items():
            # A detection is FRESH when its ref_hop advanced (the stage re-detected it,
            # not a stale REST snapshot) -- the alias escape below must never act on a
            # stale Doppler: at 0.4 Hz/s slew, 100 s of staleness fakes a 40 Hz mismatch.
            #
            # Stamp with the DETECTION'S OWN EPOCH (ref_hop converted through the capture
            # anchor), NOT the wall clock of first sight. Wall-clock stamping defines
            # freshness relative to THIS BROKER's history: a freshly (re)started broker sees
            # every entry in the search's REST table as "new" and certifies minutes-stale
            # detections as fresh -- measured 2026-07-31 00:47 on CHORD: six 20-40-min-old
            # detections walked the bias EMA to +518 Hz and the dead-reckon clock off its
            # primed value within ten cycles. With the epoch stamp, every downstream
            # consumer's (t0 - stamp) measures TRUE data age, restart or not. When the
            # anchor is unavailable (utc0_sample0 == 0, pre-fetch), fall back to wall clock
            # -- the old behaviour, right for airspy where re-detection is seconds-fast.
            if det_fresh.get(_p, (None,))[0] != _b[3]:
                t_det = (_ctx.utc0_sample0 + _b[3] / args.hops_per_sec
                         if _ctx.utc0_sample0 else t0)
                det_fresh[_p] = (_b[3], t_det)

        # ---- S2d, REVISED SCOPE (2026-07-29): RESCUE-ONLY consumption ---------------
        # Always-on consumption was tried and REVERTED the same day: car_trim rose +30-36%
        # at matched node age, and rescored against the EMA the chains actually seed with,
        # fusion lost 7 of 8 -- the LO is flat within noise (a CONSTANT), and minutes of
        # time-averaging beat one cycle of cross-chain averaging. The original premise
        # ("consume always so the rescue path is never untested") was the wrong cure: the
        # durable one is publish + SCORE always (the SHADOW line below runs regardless) and
        # EXERCISE deliberately (diag/receiver_state_rescue_test.py offline; the
        # isolated-broker method live).
        #
        # So: the fused state is consumed EXACTLY when this chain has no estimate of its
        # own -- cold start, below min-sats, warm-start file lost. There it has no EMA to
        # lose to, and its unique value over --clock-bias-siblings is real: cross-FAMILY
        # rescue (code -> carrier), which the sibling files structurally cannot provide
        # (measured: all-carriers-dark recovers to 0.6-1.8 Hz on every dongle from code
        # fits alone, including the lone-chain L2C dongle where a sibling rescue cannot
        # exist). When the chain HAS its own estimate, this block is byte-identical to
        # pre-S2d -- proven exhaustively over the input combinations, not argued.
        _fus_now = _fuse_cached(t0)
        _fused_hz = None
        if _fus_now and _fus_now.get("lo_ppm") is not None and not _fus_now["all_outliers"]:
            _fused_hz = _fus_now["lo_ppm"] * 1e-6 * args.carrier_hz
        if _fus_now is not None:
            _fus_seen[0] = True
        if args.state_consume and _cb.ema is None and _fused_hz is not None:
            _cb.value = _fused_hz
            _cb.available = True
            _log_rl("fusrescue",
                    "FUSED-STATE RESCUE: this chain is UNSOLVED; consuming the dongle's "
                    "fused LO %+.1f Hz (%d src: %dc/%dd over %s) until it solves itself"
                    % (_fused_hz, _fus_now["n_src"], _fus_now["n_carrier"],
                       _fus_now["n_code"], ",".join(_fus_now["chains"])),
                    every_s=10.0)
        else:
            _cb.value = _cb.ema if _cb.ema is not None else 0.0
            _cb.available = _cb.ema is not None
            if args.state_consume and _cb.ema is None and _fus_now is None:
                # STARTUP is not a fault. On the first cycles after launch no broker has
                # published a fresh record yet and the previous run's files are correctly
                # refused as stale, so "unavailable" is the expected state for a few
                # seconds. Saying "infrastructure fault" there is a false alarm, and false
                # alarms are how real ones get ignored -- so only call it a fault once we
                # have actually HAD a fused state and then lost it.
                _log_rl("fusegone",
                        ("FUSED STATE not yet available (starting up) -- using this "
                         "chain's own bias %s meanwhile"
                         if not _fus_seen[0] else
                         "FUSED STATE LOST -- falling back to this chain's own bias %s. "
                         "We had one and it went away: infrastructure fault (state dir "
                         "unreadable, or every sibling gone stale), not a normal mode.")
                        % (("%+.1f Hz" % _cb.ema) if _cb.ema is not None
                           else "UNSOLVED"), every_s=30.0)

        # 2. orbit-predicted Doppler + visibility (almanac assist), else plain gate
        _ctx.pred = {}          # prn -> (doppler_hz, rate_hz_s, elev_deg) [sign-applied]
        # STALE-BIAS RESCUE (--bias-stale-s): a solved bias nobody has measured for minutes
        # is a LIABILITY, not a constant -- if it latched away from truth (mid-walk during
        # the 2026-07-20 GPSDO unlock) the narrow hints it centers are what PREVENT the
        # measurements that would fix it. Widen and re-solve; hold the value for seeding.
        _cb.stale = (args.bias_stale_s > 0.0 and _cb.ema is not None
                      and t0 - _cb.meas_t > args.bias_stale_s)
        if _cb.stale:
            _log_rl("clkstale",
                    "CLOCK BIAS STALE: no multi-sat measurement for %.0f s (holding %+.0f Hz "
                    "for seeding) -- margins WIDE until re-solved"
                    % (t0 - _cb.meas_t, _cb.value), every_s=60.0)
        _ctx.up = None
        almanac_stage.stage_almanac_predict(_ctx)

        # 2a-xband. S5 CROSS-BAND: read the sibling band's per-sat tracked Doppler ONCE and
        # predict THIS band's by the exact carrier ratio (satellite motion is geometry, common
        # to both bands, scaling as f_this/f_sib; the LO terms come from each band's own S2
        # fused state -- the dongle LOs are INDEPENDENT, measured, so neither is borrowed).
        # Feeds two things below: the SHADOW residual (validate + measure the inter-band bias)
        # and, when --xband-seed, RESCUE search-Doppler hints for sats BRDC does not predict.
        _ctx.xb_pred = {}   # prn -> cross-band predicted Doppler for THIS band (bias-removed)
        if xband and args.xband_carrier_hz and args.xband_lo_dongle:
            try:
                _sib = {int(r["prn"]): r for r in _get("%s/get_status" % xband)}
                _lo_sib = _fused_lo_ppm(args.xband_lo_dongle)
                _lo_own = _fused_lo_ppm(args.state_dongle)
                if _lo_sib is not None and _lo_own is not None:
                    _ratio = args.carrier_hz / args.xband_carrier_hz
                    _LOsib = _lo_sib * 1e-6 * args.xband_carrier_hz
                    _LOown = _lo_own * 1e-6 * args.carrier_hz
                    # inter-band bias = the rolling median residual (LO diff + iono divergence);
                    # it drifts ~20 Hz/day so it must be LIVE, not a constant. Removed from the
                    # prediction so the hint centers on the truth.
                    _bias = statistics.median(_xb_resid) if len(_xb_resid) >= 20 else 0.0
                    for _p, _sr in _sib.items():
                        _ds = _sr.get("doppler_hz")
                        if _ds is None or (_sr.get("amp_snr") or 0) < 30:
                            continue      # only ride a sat the sibling holds STRONGLY
                        _ctx.xb_pred[_p] = (_ds - _LOsib) * _ratio + _LOown - _bias
                        # SHADOW: accumulate the residual for every dual-tracked sat
                        _own = _ctx.status.get(_p) or {}
                        _do = _own.get("doppler_hz")
                        if _do is not None and (_own.get("amp_snr") or 0) >= 30:
                            _xb_resid.append(_do - ((_ds - _LOsib) * _ratio + _LOown))
                    if len(_xb_resid) > 4000:
                        del _xb_resid[:len(_xb_resid) - 4000]
                    if _xb_resid:
                        _med = statistics.median(_xb_resid)
                        _mad = receiver_state.mad(_xb_resid, _med) if len(_xb_resid) > 1 else None
                        _log_rl("xband",
                                "XBAND from %s: %d sibling-tracked; rolling n=%d bias %+.1f "
                                "mad %s Hz%s"
                                % (xband, len(_ctx.xb_pred), len(_xb_resid), _med,
                                   ("%.1f" % _mad) if _mad is not None else "-",
                                   " [seeding rescue hints]" if args.xband_seed else " [shadow]"),
                                every_s=30.0)
            except Exception as e:
                _log_rl("xband", "XBAND read failed: %s" % e)

        # 2b. Almanac-narrow the SEARCH: push per-PRN predicted Doppler to the detectors so each
        # scans only doppler +- margin instead of its blind grid -- far cheaper + more sensitive,
        # and it's what lets the not-yet-detected sats be acquired without a full sweep. The margin
        # is WIDE until the common clock-freq bias is solved (the geometric Doppler is then offset
        # by the unknown TCXO), NARROW once a few sats pin it. Sent for all predicted+visible sats.
        searchhint.stage_narrow_search(_ctx)

        # refresh / add consensus seeds: code phase from the search, Doppler from the
        # orbit prediction when available (precise enough for coherent integration),
        # else the coarse search grid.
        _ctx.la_samples = []   # per-sat (l-a) estimates this cycle, from sats with a good code-rate fit
        _ctx.fitted = set()    # PRNs that got their own >=3-snapshot slope fit this cycle
        _ctx.cl_report = []    # CL time-assist per-PRN (k, fine-time residual) log lines this cycle
        # Capture time anchor: wall-clock UTC of capture sample 0 (airspy stamps it at its
        # first USB callback; /adcstat serves it). 0.0 until the stream starts -- retry
        # lazily. Used by the CL time-assist and the dead-reckoned cp seeding.
        # THE ANCHOR IS LATCHED (`not utc0_sample0` above): fetched once and kept for the whole
        # run. That is right for a stable F-engine and WRONG ACROSS AN F-ENGINE RESTART, which
        # re-establishes frame 0 -- and the failure is silent. On 2026-08-07 the F-engine came
        # back with a frame 0 13.587 days later than before; cx19 restarted and took the new
        # epoch, the broker kept the old one, and every seed was 13.587 days wrong. What that
        # looks like from outside is "nothing locks, deep_snr ~1.5" -- i.e. a tracker or geometry
        # problem, not a stale anchor. So: re-read periodically and SAY SO, loudly, every cycle.
        #
        # Deliberately does NOT re-anchor itself. A changed frame 0 also invalidates the epoch
        # every NODE cached at ITS startup, so the correct response is to restart the fleet and
        # the broker together; silently re-anchoring here would fix the broker's arithmetic while
        # leaving the nodes wrong, and hide the condition that requires the operator.
        # _now() rather than the loop's now_w: that is assigned LATER in the cycle, so using
        # it here is a NameError on the first pass. With the cycle clock frozen they are now
        # the same number anyway -- which is what this always meant.
        _now_anchor = _now()
        if args.time0_endpoint and _ctx.utc0_sample0 and _now_anchor - _anchor_chk[0] > 60.0:
            _anchor_chk[0] = _now_anchor
            try:
                _fresh = float(_get("%s/%s" % (base, args.time0_endpoint.strip("/")))
                               .get("time0_ns", 0.0)) / 1e9
                if _fresh and abs(_fresh - _ctx.utc0_sample0) > 1e-3:
                    _log("*** TIME ANCHOR CHANGED: frame0 was %.9f, endpoint now reports %.9f "
                         "(%+.3f days). The F-engine has been restarted. EVERY SEED THIS BROKER "
                         "SENDS IS WRONG BY THAT AMOUNT, and every node still running cached the "
                         "old epoch too. Restart the nodes AND this broker."
                         % (_ctx.utc0_sample0, _fresh, (_fresh - _ctx.utc0_sample0) / 86400.0))
            except Exception:
                pass   # endpoint down is the normal outage case, already logged elsewhere

        if (args.cl_assist or args.cl_tracker or dr_state is not None) and not _ctx.utc0_sample0:
            try:
                if args.time0_endpoint:
                    # CHORD: frame 0 is GPS-disciplined, so this is exact rather than an
                    # estimate. time0_ns is the absolute time of fpga_seq_num 0.
                    # NB: NOT `t0` -- that name is the cycle-start timestamp in this loop, and
                    # shadowing it with a nanosecond epoch made the loop's
                    # `dt = interval - (_now() - t0)` about 1.8e18 seconds.
                    #
                    # THROUGH THE RECEIVER (task #27 M3): frame 0 is a property of the
                    # INSTRUMENT, not of a signal, so it is fetched at most once per process
                    # however many chains want it. Two brokers latching it independently can
                    # straddle an F-engine restart and disagree forever, each certain.
                    _ctx.utc0_sample0 = rx.time_anchor(
                        lambda: float(_get("%s/%s" % (base, args.time0_endpoint.strip("/")))
                                      .get("time0_ns", 0.0)) / 1e9,
                        chain_id) or 0.0
                    if _ctx.utc0_sample0:
                        _log("time anchor: CHORD F-engine frame0 = %.9f s (GPS-disciplined)"
                             % _ctx.utc0_sample0)
                        _anchor_seen[0] = _ctx.utc0_sample0
                else:
                    _ctx.utc0_sample0 = rx.time_anchor(
                        lambda: float(_get("%s/%s/adcstat" % (base, args.adc_stage))
                                      .get("utc0_sample0", 0.0)),
                        chain_id) or 0.0
                    if _ctx.utc0_sample0:
                        _log("CL time-assist: capture sample-0 UTC anchor %.3f" % _ctx.utc0_sample0)
            except Exception as e:
                _log("time anchor unavailable (%s); retrying" % e)
        _ctx.dr_pd = (dr_state or {}).get("pd") or {}
        _ctx.dr_pd2 = (dr_state or {}).get("pd2") or {}
        _ctx.dr_pd0 = (dr_state or {}).get("pd0") or {}
        seeding.stage_detections_to_seeds(_ctx)

        # 3. COAST / drop (the trajectory-predictor promotion). A visible sat is coasted through a
        # signal dropout (radar sweep, brief fade): its seed is held and its Doppler forecast
        # forward from the orbit + clock each poll, so the tracker keeps despreading at the
        # PREDICTED trajectory and re-peaks when the signal returns -- the lock survives the gap
        # instead of being pruned and re-acquired. The code prediction holds for ~the coast budget;
        # drop ONLY when the sat SETS (the unambiguous "gone") or |A| stays down for the whole
        # budget (genuine loss / prediction breakdown). |A| recovering resets the coast.
        _ctx.coast_polls = max(1, int(round(args.coast_budget / max(args.interval, 1e-3))))
        try:
            _ctx.status = {int(r["prn"]): r for r in _get("%s/get_status" % combiner)}
            if args.almanac_epoch:
                _u = [float(r["utc"]) for r in _ctx.status.values() if r.get("utc")]
                if _u:
                    _alm_file_pos[0] = max(_u) - args.almanac_epoch_utc0
            # #83 THE AXIS FIX: capture the newest F-engine hop AT FETCH TIME. The pair
            # (hop, wall-at-fetch) lets the dr block build its "now" on the F-engine axis
            # with wall entering only as the elapsed-since-fetch difference.
            _fh = max((float(r.get("pow_hop") or 0.0) for r in _ctx.status.values()),
                      default=0.0)
            # PUBLISHED FOR THE PRN SCHEDULER (prnmap._at_seq). None, not 0.0, when there is
            # no axis: a zero would look like a valid sample at the epoch and would schedule
            # every swap into the deep past, i.e. straight through the deadline test.
            _ctx.fe_hop_now = _fh if _fh > 0.0 else None
            # ⚠️ AND WHEN. The scheduler must advance this hop to the instant it POSTS, which
            # is later in this same cycle; without the stamp the deadline is built on a hop
            # that is already stale by the cycle's own elapsed time.
            _ctx.fe_hop_t = _now() if _fh > 0.0 else None
            if _fh > 0.0:
                # ⚠️ THE TIME BASE MUST NEVER FREEZE SILENTLY (2026-08-18, the cx19 collapse).
                # t_now_abs is built from this hop, and this hop comes from ONE combiner. When
                # cx19/gnss0 deadlocked its capture window, its pow_hop stopped advancing while
                # the broker kept polling it happily -- so t_now_abs froze, det_age went
                # NEGATIVE and grew at 1 s/s, every integrity residual read as enormous, the
                # clock solve declared "the CLOCK moved" and latched, and all four chains that
                # ADOPT that clock lost their seeds together (#75). Not one line said the time
                # base had stopped. This is that line.
                # ⚠️⚠️ THE STAMP IS THE TIME THE HOP LAST ADVANCED, NOT THE TIME OF THIS
                # POLL -- and getting that wrong made this guard STRUCTURALLY UNABLE TO FIRE
                # for as long as it existed. `fe_axis[0]` used to be rewritten every cycle
                # including its timestamp, so `_fh_prev[1]` was always the PREVIOUS CYCLE
                # (~2 s ago) and `now - _fh_prev[1] > 30` was false forever. Measured
                # 2026-08-26: the F-engine re-based at 21:43, every instance froze together,
                # the broker logged `AXIS INST: lag median -6975 s ... spread 0.00` 807 times
                # -- and this line, the one guard whose whole job is "has the whole time base
                # frozen?", never printed once.
                #
                # It matters more than one missing warning, because the two guards PARTITION
                # the space on purpose: instance_stall_verdict refuses to accuse anyone when
                # most of the fleet is also stalled (its min_frac_advancing control clause)
                # precisely because THIS one is supposed to cover the global case. One half of
                # a deliberate partition being inert leaves the global freeze -- the most
                # damaging case, and the one that needs a fleet restart -- with no detector at
                # all. instance_stall_verdict already keeps the right stamp (fits.py: refresh
                # `now` only when the hop CHANGED); this now does the same thing.
                _fh_prev = fe_axis[0]
                if (_fh_prev is not None and _fh <= _fh_prev[0]
                        and args.fe_axis_stale_s > 0.0
                        and _now() - _fh_prev[1] > args.fe_axis_stale_s):
                    _log_rl("fe-stale",
                            "*** TIME BASE FROZEN: newest telemetry hop has not advanced in "
                            "%.0f s (hop %.0f, combiner %s). t_now_abs is built from this, so "
                            "the receiver clock, every det_age and every model-evaluated seed "
                            "on this chain are now standing still while the sky is not. This "
                            "is an INSTANCE stall upstream, not a tracking fault -- check that "
                            "combiner's pow_hop and its node's capture window."
                            % (_now() - _fh_prev[1], _fh, combiner),
                            every_s=60.0)
                # ADVANCED -> restamp; FROZEN -> keep the stamp so the staleness accrues.
                fe_axis[0] = ((_fh, _now()) if (_fh_prev is None or _fh > _fh_prev[0])
                              else (_fh, _fh_prev[1]))
                # the filtered offset (see fe_off at its definition)
                _ow = _now()
                _oi = _fh / args.hops_per_sec - _ow
                if fe_off[0] is None:
                    fe_off[0] = _oi
                elif abs(_oi - fe_off[0]) > 2.0:
                    # disagreement: believe it only if it REPEATS at the same value
                    if abs(_oi - fe_off[3]) < 0.5:
                        fe_off[2] += 1
                    else:
                        fe_off[2] = 1
                    fe_off[3] = _oi
                    if fe_off[2] >= 3:
                        _log("fe-axis offset SNAP: filtered %+.3f -> %+.3f s after %d "
                             "consecutive polls (a real axis move -- F-engine restart or "
                             "frame0 step; a dropout does not persist)"
                             % (fe_off[0], _oi, fe_off[2]))
                        fe_off[0] = _oi
                        fe_off[2] = 0
                    else:
                        _log_rl("fe-axis-blip",
                                "fe-axis: instantaneous offset %+.3f s disagrees with the "
                                "filtered %+.3f by %.2f s -- HELD (%d/3). max(pow_hop) fell "
                                "back, i.e. the freshest row dropped out of this poll."
                                % (_oi, fe_off[0], _oi - fe_off[0], fe_off[2]), every_s=60.0)
                else:
                    fe_off[2] = 0
                    _dec = 0.0005 * max(0.0, _ow - fe_off[1])
                    fe_off[0] = max(_oi, fe_off[0] - _dec)
                fe_off[1] = _ow
        except Exception as e:
            _ctx.status = {}
            _log("get_status failed: %s" % e)
        # P7a nav-bit predictor: fold in this cycle's bit observations (rows carry nav_obs
        # only when the combiner runs bit_export, so non-GPS chains skip this for free).
        navbits.stage_nav_bits(_ctx)
        # Lock metric: the detection SIGNIFICANCE (sigma above noise) -- the deep nav-wiped SNR when
        # available, else the noise-debiased incoherent SNR -- not the raw |A|. The incoherent |A| is
        # biased by the noise floor (~the floor for weak sats), so judging "still locked" by |A| >
        # drop_amplitude let phantoms coast forever (|A| never falls below the floor). sig ~1 = noise,
        # >>1 = a real lock. Falls back to |A| only if the combiner reports no significance at all.
        _ctx.have_sig = any(sig_of(r) > 0 for r in _ctx.status.values())
        # NOISE PROBES (--noise-probes N): keep the N deepest-below-horizon PRNs seeded so
        # the combiner emits GENUINE noise records for them -- the beam map's pedestal
        # calibration (clip bias of the moment debias + the coherent estimator's selection
        # residue) needs signal-free samples, and an almanac-gated broker otherwise never
        # tracks one (2026-07-12: the GPS pedestal fell back to a signal percentile,
        # x=10.6 ~ 40 dB-Hz, blinding the map's low end). Probes are exempt from the
        # set-below-horizon drop and invisible to hints/hold/DLL/carrier (their sig ~ 0
        # fails every gate naturally); dop/cp are arbitrary for noise -- predicted values
        # keep the despread configuration representative.
        _ctx.probe_set = set()
        if args.noise_probes > 0 and args.almanac and _ctx.pred:
            _cands = sorted((p for p, v in _ctx.pred.items() if v[2] < -15.0),
                            key=lambda p: _ctx.pred[p][2])
            # ⚠️⚠️ A PROBE THE NODE HAS NO SLOT FOR IS NOT A PROBE (--probe-require-slot).
            # This picked the DEEPEST below-horizon PRNs straight out of the almanac, with no
            # check that the trackers can represent them -- so it kept choosing satellites the
            # node cannot despread. They are seeded, logged as seeded, and NEVER produce a
            # row. Measured 2026-08-27: BeiDou's probes were PRN 2, 26, 33 against a slot list
            # of 19-42, so PRN 2 could never report; two probes survived, which is below the
            # >= 3 the q+p gate needs, and presence fell back to the PEER COMPETITION -- q
            # floor 4.72 on bds_b2a, ABOVE the q ~ 4 ceiling any real satellite can reach, so
            # nothing could pass on q at all and roughly HALF the population passed on prompt
            # power by construction. That is the "exactly half of the up satellites lock"
            # symptom, and it is the same root as E36 ([[chord-prn-lists-diverge]]).
            #
            # ⚠️ THIS IS NOT THE DISCOVERY-BASED FILTER THAT WAS REJECTED. That one inferred
            # untrackability from "never reported", which cannot tell untrackable from
            # not-yet-chosen until a sidereal day has passed. This ASKS THE NODES what they
            # hold (prnmap's /get_prns sweep) -- authoritative, available immediately, no
            # bootstrap. It is the first consumer of the live-membership work.
            #
            # ⚠️ AND IT FAILS OPEN. `consensus` is None until a full unanimous sweep lands
            # (and on any split fleet), and one cycle behind besides -- the prnmap stage runs
            # later in the cycle than this. Absent map -> the old unfiltered behaviour, which
            # is a degraded probe set and not a dead chain.
            if args.probe_require_slot:
                _held = _prnmap.consensus
                if _held:
                    _drop = [p for p in _cands[:args.noise_probes] if p not in set(_held)]
                    _cands = [p for p in _cands if p in set(_held)]
                    if _drop:
                        _log_rl("probe-slot",
                                "PROBE SLOT FILTER: %s have no slot on this chain and would "
                                "never report -- skipped; probes now %s"
                                % (",".join(str(p) for p in _drop),
                                   ",".join(str(p) for p in _cands[:args.noise_probes])),
                                every_s=300.0)
            deep_low = _cands[:args.noise_probes]
            _ctx.probe_set = set(deep_low)
            for p in deep_low:
                if p not in seeds:
                    _log("noise probe PRN %d seeded (elev %.0f)" % (p, _ctx.pred[p][2]))
                seeds[p] = Seed.born(
                    "probe", epoch=0,
                    doppler_hz=_ctx.pred[p][0] + _cb.value,
                    code_phase_chips=0.0,
                    code_phase_rate=cp_rate_from_code_bias(
                        _ctx.pred[p][0], _cb.code_ema or 0.0, args.hops_per_sec,
                        args.chip_rate_hz, args.carrier_hz),
                    ref_hop=0,
                    doppler_rate_hz_s=_ctx.pred[p][1])
        # TRACK WATCHDOG (--watchdog-s, default off): a sat the SEARCH sees STRONGLY that
        # has not produced a single coherent emit for this long is broken, whatever the
        # cause -- aliased NCO (the resid estimator cannot see past +-1/(4*T_rec)), a
        # noise-walked BOOTSTRAP trim, a poisoned anchor. Every targeted correction tried
        # on 2026-07-18 (trim-step v1 8208dba6, v2 ce47509e) guessed the cause and lost;
        # the one rescue that always worked was the full re-seed lifecycle (C20, 19:54).
        # So: drop the sat from seeds entirely. Next cycle it re-enters as a FIRST SEED --
        # fresh det/dr dop blend, fleet-median trim, and the tracker's f_ref/phase state
        # resets via the one-cycle active[] gap. The det-snr bar keeps genuinely weak sats
        # (legitimately slow to cohere) out of reach; the seed-age bar gives a full window
        # before judging; firing re-stamps birth, so re-fires need a whole new window.
        codeloop.stage_watchdog(_ctx)
        seeding.stage_coast_drop(_ctx)

        # 3e. DEAD-RECKONED CODE-PHASE SEEDING (--dead-reckon): the search only exists to
        # measure what the model already knows. BRDC ephemeris (~2 m orbits + ~5 ns sat
        # clocks) plus the receiver clock solved from the sats we DO detect predict every
        # other visible sat's code phase to well inside the DLL capture range (0.10 chip
        # rms validated, gnss_deadreckon_check.py 2026-07-13) -- so seed them all:
        # sub-threshold sats despread on-peak with no detection ever required (the
        # sidelobe-mapping mode). The search demotes to bootstrap (clock solve), fallback
        # (a detection re-anchors via the normal seed loop, which also removes the PRN
        # from the model-owned set below) and integrity check (residuals logged here).
        deadreckon.stage_dead_reckon(_ctx)

        # 3c. Delay-lock loop (R1): close the CODE loop from the combiner's window-averaged E/L
        # powers. disc = (<|E|^2>-<|L|^2>)/(<|E|^2>+<|L|^2>) ~ -4*tau at 0.5-chip spacing, where
        # tau = (true - commanded) cp: the correlation triangle gives |E|=R(d+tau), |L|=R(d-tau),
        # so a LATE true peak strengthens L and drives disc negative -> tau_est = -disc/4,
        # scaled by (spacing/0.5). The per-PRN TRIM integrates gain*tau_est and is applied to
        # the seed at POST time only -- the stored seed stays pure fit/coast state, so the trim
        # converges to the search fit's quantization bias instead of double-counting on coasted
        # seeds. Gated on lock significance (an unlocked PRN's disc is noise). Bounded +-3 chips.
        # Per-cycle, so a dll_gain of 0 (the loop disabled) leaves a defined empty fleet for
        # everything downstream that reports on it -- including #51's fast-trim publication,
        # which must arm NOBODY rather than raise when the code loop is off.
        _dllp.fleet = {}
        fleetdll.stage_fleet_dll(_ctx)

        # 3d. SHARED CARRIER LOOP (the carrier twin of 3c): integrate the combiner's full-band
        # cross-record phase-walk residual into a commanded NCO frequency per PRN. The residual
        # is measured AFTER the current trim (the NCO derotates before records ship), so the
        # plain integrator converges: trim += gain * resid. No lock gate: the observable is
        # vector-averaged over the emit window at FULL-BAND SNR (the whole point -- per-channel
        # amplitude gates would exclude exactly the weak-band cases this loop exists for);
        # the clamp bounds any noise walk.
        # f_carrier FEED (task #33, 2026-08-11). rate_residuals -- the validated carrier
        # measurement (split-half ~0.2 Hz on strong sats, wrong-bin defence, q gate) --
        # used to run ONLY inside the carrier_gain > 0 branch below. With the gain at its
        # default 0.0 the measurement was never even computed: the joint state's f_carrier
        # had a characterised input that no code path produced. Harvest it whenever the
        # joint shadow is on, and feed the fleet CONSENSUS -- the quality-weighted common
        # mode across PRNs, which is exactly the receiver-wide offset f_carrier models.
        # Per-sat residuals stay with the (still-off) trim loop; feeding them here would
        # average per-sat noise into a common state (the 3d scope warning).
        # MEASUREMENT-ONLY: touches no seed, no trim, no loop. sigma 0.5 Hz -- looser than
        # the 0.2 Hz split-half bound because the consensus mixes sat qualities.
        # ⚠️ ONE CHAIN FEEDS, measured on first light (01:03): the five chains' consensuses
        # disagreed by tens of Hz (e5b +41, e5a -10, gps +44) because deep_rate_hz is the
        # residual AFTER each chain's own applied seed Doppler, and the chains apply
        # DIFFERENT clock biases (GPS solves one; the model-primary chains hold 0). Fed
        # jointly, the state was yanked between definitions and the gate rejected 23/min.
        # Scope to the detector chain (gps_l5): its carrier bias is actually solved, so
        # its consensus is the best-defined "LO minus solved bias" residual. A per-band
        # f_carrier (tau_band-style) is the eventual right shape if other chains need it.
        # ⚠️ ONE rate_residuals CALL PER POLL, shared by its three consumers (the f_carrier
        # consensus feed, the rrate per-sat feed, and the trim loop). The call MUTATES its
        # continuity state (prev_hop/prev_val/unit_hop): a second call in the same poll
        # reads "the same window again" for every PRN and returns {} -- so the consumers
        # must share one result, never each ask.
        _rf.resid, _rf.cons = {}, None
        if (args.carrier_source == "rate"
                and (args.carrier_gain > 0.0
                     or (args.joint_shadow and args.detectors
                         and not args.rrate_state))):
            try:
                _rf.resid, _rf.cons = rate_residuals(
                    _ctx.status, args.carrier_rate_min_q, args.carrier_rate_clip_hz,
                    _log if args.carrier_gain > 0.0 else None,
                    prev_hop=rate_prev_hop, max_gap=args.carrier_rate_max_gap,
                    prev_val=rate_prev_val, max_step=args.carrier_rate_max_step,
                    unit_hop=rate_unit_hop)
            except Exception as e:
                _log_rl("rate-resid-err", "rate_residuals skipped: %s" % e, every_s=300.0)
        if (args.joint_shadow and args.detectors and args.carrier_source == "rate"
                and args.carrier_gain <= 0.0):
            try:
                _fr_resid, _fr_cons = _rf.resid, _rf.cons
                # A consensus over 1-2 sats is a rotating PER-SAT sample, not a common
                # mode: measured 01:05-01:08, successive "consensuses" swung -29.6 ->
                # -14.3 -> +38.6 -> +18.3 Hz as different single sats passed the q gate
                # -- the standing ~24 Hz per-sat residual (STATE 8.20.3), not the LO.
                # Require a real cross-sat median and log the spread, which is the
                # per-sat-vs-common diagnostic this feed exists to provide.
                # Per-sat values logged EVERY poll (2026-08-11, KV): the breakdown that
                # resolved the "per-sat tens-of-Hz" mystery needed exactly this series --
                # instances agree to 0.05 Hz while the value hops 15-39 Hz between polls,
                # i.e. WRONG-BIN capture aliased into the +-47.7 Hz record-rate window
                # (NH20 sidebands at +-50 alias to -+45.4), coherent across the fleet
                # because every instance folds the same records. Not carrier physics.
                if _fr_resid:
                    _log_rl("jfcar-sat",
                            "JFCAR-SAT: %s (alias window +-%.1f Hz)"
                            % (" ".join("%d:%+.1f" % (p_, r_) for p_, r_
                                        in sorted(_fr_resid.items())),
                               0.5 * args.hops_per_sec / 2048.0), every_s=30.0)
                # SUPERSEDED BY THE PER-SAT FEED when --rrate-state is on: the consensus
                # is a weighted mean of the SAME residuals update_rrate consumes one by
                # one, so feeding both is the same data twice per poll (correlated
                # measurements the filter would treat as independent -> overconfident).
                # Under the rrate gauge the common mode lands on f_carrier anyway.
                if (_fr_cons is not None and len(_fr_resid) >= 3 and not args.rrate_state
                        and _drp.t_now_abs is not None):
                    _vals = sorted(_fr_resid.values())
                    _sprd = _vals[-1] - _vals[0]
                    _jfc = _joint_state(rx, band_id, args)
                    if _jfc is not None:
                        _jfc.update_carrier(_fr_cons, _drp.t_now_abs,
                                            sigma_hz=max(0.5, _sprd / 2.0))
                        _log_rl("jfcar",
                                "JOINT f_carrier %+.3f+-%.3f Hz (consensus %+.3f, %d sats, "
                                "per-sat spread %.1f Hz; n=%d rej=%d)"
                                % (_jfc.f_carrier(), _jfc.f_carrier_sigma(), _fr_cons,
                                   len(_fr_resid), _sprd, _jfc.n_fcar, _jfc.fcar_rejected),
                                every_s=60.0)
            except Exception as e:
                _log_rl("jfcar-err", "f_carrier feed skipped: %s" % e, every_s=300.0)

        # 3d'. rrate FEED (task #33 P3 step 2, --rrate-state). Per-satellite ORBITAL
        # range-rate-error rows (m/s, band-shared) in the RECEIVER-WIDE joint state, fed
        # from this chain's per-sat rate residuals. Each measurement updates BOTH the sat's
        # rrate row and the shared f_carrier through one H -- the two are degenerate within
        # a band and separated only by what is shared across satellites (the clk/b_sat
        # construction, on the carrier). E5a and E5b feeding the same Galileo sat land on
        # ONE row through two carriers 1.0261 apart: cross-band information ADDS instead of
        # being fitted twice, which is the entire point of the state.
        # MEASUREMENT-ONLY unless --rrate-command is also set: touches no seed, no POST.
        # ITS OWN rate_residuals CALL, on the UNCAPPED fields when the combiner serves them
        # (#40: deep_rate_hz is the fold's pick, clamped to deep_rate_max_hz -- past the cap
        # it reports the best in-cap bin, i.e. NOISE, which is what walked arm 1). Fallback
        # to the capped fields keeps the shadow alive against an old tracker binary, and
        # rr_full_ok is how the COMMAND refuses to close the loop on that degraded feed.
        _rf.full_ok = False
        if args.rrate_state and args.carrier_source == "rate":
            _rf.full_ok = any(isinstance(_r, dict) and _r.get("deep_rate_full_q") is not None
                             for _r in (_ctx.status or {}).values())
            try:
                _fd = ("deep_rate_full_hz", "deep_rate_full_q") if _rf.full_ok \
                    else ("deep_rate_hz", "deep_rate_q")
                _rf.resid2, _ = rate_residuals(
                    _ctx.status, args.carrier_rate_min_q, args.carrier_rate_clip_hz, None,
                    prev_hop=rrate_prev_hop, max_gap=args.carrier_rate_max_gap,
                    prev_val=rrate_prev_val, max_step=args.carrier_rate_max_step,
                    unit_hop=rate_unit_hop, rate_field=_fd[0], q_field=_fd[1])
            except Exception as e:
                _rf.resid2 = {}
                _log_rl("jrr-err", "rrate residuals skipped: %s" % e, every_s=300.0)
        else:
            _rf.resid2 = {}
        if args.rrate_state and _rf.resid2:
            # REGIME GATE (arm 4's lesson, 2026-08-13 15:4x). The full-band field revealed
            # a population the capped view called noise: STRONG but DECOHERED sats (PRN 27
            # at amp 56, coh_frac 0.02) carrying REAL multi-Hz carrier residuals that SWING
            # +-10 Hz poll-to-poll -- seed/f_ref churn, not orbit error. The rrate model is
            # a slow per-sat drift; fed those swings it rejects 2:1 and the escape
            # snap-moves rows. Feed only sats whose regime the model fits: COHERING ones
            # (same bar as the joint code feed's _track_ok). The decohered population is
            # its own open question -- the full field finally makes it VISIBLE (#48).
            _rf.resid2 = {
                _p: _rv for _p, _rv in _rf.resid2.items()
                if ((_ctx.status.get(_p) or {}).get("coherence_s") or 0.0) > 0.0
                or ((_ctx.status.get(_p) or {}).get("coh_frac") or 0.0) >= 0.3}
        # ── #83 PHASE 3 STEP 1: the fleet-coherent rate feed (see --rrate-kcoh-feed) ──
        # Runs BEFORE the coarse loop so a fresh acceptance here deweights this cycle's
        # coarse measurements for the same satellite (the FLL->PLL pattern, third feed).
        # _est_last, never the loop-local _kcoh: that name is only bound once the
        # telemetry block has run at least once this thread, and a NameError in the guard
        # would sit outside the try.
        _kco = _est_last.get("kcoh")
        if _kco is rr_kcoh_fed.get("last"):
            _kco = None   # same estimate object as last cycle: already fed once
        if args.rrate_state and args.rrate_kcoh_feed and _kco and _drp.t_now_abs is not None:
            rr_kcoh_fed["last"] = _kco
            try:
                _jrk = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference, gauge_mode=args.joint_gauge)
                _nk = 0
                _krows = []
                for _p, _kv in sorted(_kco.items()):
                    if (_kv.get("probe")
                            or (_kv.get("sig") or 0.0) < args.rrate_kcoh_min_sig
                            or (_kv.get("rate_pairs") or 0) < 8):
                        continue
                    _rem = ((_kv.get("rate_hz") or 0.0)
                            + (_kv.get("rate_resid_hz") or 0.0))
                    # Fold safety, same bound as the fine feed: the per-record fold is
                    # unambiguous to ~+-23 Hz; beyond 20 the estimate is suspect.
                    if abs(_rem) >= 20.0:
                        continue
                    _yk = _rem + (_rf.cmd_applied.get(_p, _carrier.trim.get(_p, 0.0))
                                  if args.rrate_feed_applied else 0.0)
                    _sigk = min(0.3, max(0.03, 2.0 / math.sqrt(_kv["sig"])))
                    _k2 = (args.dr_constellation, int(_p))
                    if _jrk.update_rrate(_k2, _yk, _drp.t_now_abs, args.carrier_hz,
                                         sigma_hz=_sigk) is not None:
                        _rf.kcoh_t[_p] = t0
                        _nk += 1
                        _krows.append("%d:%+.2f+-%.2f" % (_p, _yk, _sigk))
                if _nk:
                    _jrk.gauge_rrate()
                    _log_rl("jrr-kcoh",
                            "JRR-KCOH %s: %d sat(s) fed from the fold's remaining rate "
                            "(y = remaining%s, Hz): %s"
                            % (log_tag() or args.signal, _nk,
                               " + applied" if args.rrate_feed_applied else
                               " ALONE, blind plant",
                               " ".join(_krows)),
                            every_s=60.0)
            except Exception as e:
                _log_rl("jrr-kcoh-err", "JRR-KCOH: failed (%s) -- cycle continues" % e)
        ratefeed.stage_rate_feed_coarse(_ctx)
        # 3d''. PLL FINE STAGE (#33 phase-step feed). The ADR's residual half
        # (res_cycles), differenced over the poll span: ~5 mHz where the rate spectrum's
        # floor is ~60. SHADOW ALWAYS (the JRRP line calibrates the r2c sign against the
        # coarse observable on the uncommanded chains' standing residuals); FEED only when
        # --rrate-phase-feed is set AND the sign is calibrated AND, per sat: same arc,
        # counter advanced (adr_fine_rate's structural gates), the coarse loop converged,
        # and the command HELD over the span -- the fine value's reference is only exact
        # under a constant command, and gating beats averaging a moving one.
        ratefeed.stage_rate_feed_fine(_ctx)

        # S2 OBSERVER: the shadow surface for sky validation (present even when the feed
        # produced nothing this poll, so a dead feed shows n=0 rather than vanishing).
        if state_w is not None and args.rrate_state:
            try:
                _jro = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference, gauge_mode=args.joint_gauge)
                _ks = [k for k in _jro._rr_idx if k[0] == args.dr_constellation]
                _vs = sorted(_jro.rrate(k) for k in _ks)
                state_w.observe(
                    "rrate",
                    n=len(_ks),
                    median_mps=(statistics.median(_vs) if _vs else None),
                    spread_mps=((_vs[-1] - _vs[0]) if _vs else None),
                    f_carrier_hz=(_jro.f_carrier()
                                  if _jro.f_carrier_sigma() != float("inf") else None),
                    rejected=_jro.rrate_rejected)
            except Exception:
                pass

        carrierloop.stage_carrier_loop(_ctx)

        # S2 OBSERVER: the SECOND carrier-side estimator of the same LO. `car_trim`'s own
        # arg help calls the converged fleet trim "the chain's deterministic frac-N LO
        # offset ... stable across restarts (e.g. the L5 chain sits ~+30 Hz)" -- which is
        # the same physical number `clock_bias_ema` solves for from search Doppler, on a
        # different observable and a different timescale. The two have never been compared
        # and this one is never persisted, so it is rebuilt from zero every launch despite
        # being described as a restart-stable constant. Exported outside the carrier-loop
        # gate so a chain with the loop disabled reports null rather than vanishing.
        if state_w is not None:
            try:
                _ct = sorted(_carrier.trim.values())
                _ctm = statistics.median(_ct) if _ct else None
                state_w.observe(
                    "carrier_trim",
                    median_hz=_ctm,
                    mad_hz=receiver_state.mad(_ct, _ctm),
                    n=len(_ct),
                    railed=sum(1 for v in _ct if args.carrier_max_hz > 0.0
                               and abs(v) >= 0.95 * args.carrier_max_hz))
            except Exception:
                pass

        # 3b. Receiver code-rate clock offset (l-a): pool the strong sats' per-fit samples (robust
        # median), EMA-smooth it (slow -> tracks TCXO/OCXO drift), then SEED it to every sat that
        # could NOT fit its own slope (weak / just-acquired / coasting) as carrier-aiding + (l-a).
        # The code-side twin of the carrier clock_bias: strong detections calibrate the clock so weak
        # ones never drift off-peak. Ephemeris-free (v/c cancels). Bounded +-50 ppm against a rogue fit.
        # ── #91(c): DO NOT RE-FIT THE CLOCK FROM A COLLAPSING POPULATION ─────────────
        # --code-bias-min-sats is an ABSOLUTE floor (2). A chain that fell from 7 present
        # to 2 still passes it -- and that is precisely the population whose fit swung the
        # (l-a) clock +-96 chips on 2026-08-25 and had its garbage rate (+0.041 ppm, 40x
        # normal) adopted into every seed, a positive feedback that SUSTAINED the outage.
        # The relative collapse is what D1 measures, so hold the last good EMA through it.
        # Holding is the conservative direction: the oscillator does not care that our
        # satellites went away, and the EMA is slow by design.
        _cb_frozen = bool(args.code_bias_brownout_hold and _brown.established())
        if _cb_frozen:
            _log_rl("la-freeze",
                    "code-rate clock (l-a) HELD at %s ppm through the brownout (#91c): "
                    "%d fit sample(s) this cycle are a collapsed population, not a clock"
                    % ("%+.3f" % (_cb.code_ema * 1e6) if _cb.code_ema is not None else "unset",
                       len(_ctx.la_samples)),
                    every_s=60.0)
        if not _cb_frozen and len(_ctx.la_samples) >= args.code_bias_min_sats:
            raw_cb = statistics.median(_ctx.la_samples)
            if abs(raw_cb) < args.code_bias_max * 1e-6:
                _cb.code_ema = (raw_cb if _cb.code_ema is None
                                 else _cb.code_ema + args.code_bias_alpha * (raw_cb - _cb.code_ema))
                # SAT-SCALED bar, same rationale as the carrier-bias alarm above (few-fit
                # chains' l-a median is ~1/sqrt(n) noisy; the fixed bar cried wolf on the
                # weak chains 2026-07-20). A real dongle-clock event is large + sustained.
                _labar = args.code_bias_alarm_ppm * max(1.0, (5.0 / max(len(_ctx.la_samples), 1)) ** 0.5)
                if (_cb.code_cal is not None
                        and abs(_cb.code_ema - _cb.code_cal) > _labar * 1e-6):
                    _log_rl("laalarm",
                            "CLOCK DRIFT ALARM: l-a %+.3f ppm vs calibration %+.3f "
                            "(|d| > %.2f ppm, %d fits) -- dongle clock news, INVESTIGATE"
                            % (_cb.code_ema * 1e6, _cb.code_cal * 1e6,
                               _labar, len(_ctx.la_samples)), every_s=60.0)
                _log_rl("la-pool",
                        "code-rate clock offset (l-a) %+.3f ppm (raw %+.3f, %d fitted "
                        "sats, EMA a=%.2f)"
                        % (_cb.code_ema * 1e6, raw_cb * 1e6, len(_ctx.la_samples),
                           args.code_bias_alpha))
                # CONTRIBUTE (task #27 M3). PER BAND, not receiver-wide: cable and PFB group
                # delay are per carrier, so this number does not survive a retune. That is
                # exactly what --state-dongle asserts by hand today.
                rx.contribute_code_bias(chain_id, band_id, _cb.code_ema,
                                        len(_ctx.la_samples), t0)
                if args.code_bias_file:
                    try:
                        with open(args.code_bias_file, "w") as f:
                            f.write("%.4f\n" % (_cb.code_ema * 1e6))
                    except Exception:
                        pass
        # S2 OBSERVER: the code-side twin. Outside the min-sats gate, same reason as the
        # carrier export. This one is the honest cross-chain comparison of the two: l-a has
        # NO sibling fusion at all, so its spread across a band's chains is a real measure
        # of estimator scatter, where the carrier's is partly manufactured by the fusion.
        if state_w is not None:
            try:
                _raw_la = statistics.median(_ctx.la_samples) if _ctx.la_samples else None
                state_w.observe(
                    "code",
                    ppm=(_cb.code_ema * 1e6) if _cb.code_ema is not None else None,
                    raw_ppm=(_raw_la * 1e6) if _raw_la is not None else None,
                    mad_ppm=(lambda m: m * 1e6 if m is not None else None)(
                        receiver_state.mad(_ctx.la_samples, _raw_la)),
                    n=len(_ctx.la_samples),
                    cal_ppm=(_cb.code_cal * 1e6) if _cb.code_cal is not None else None,
                    forced=args.code_bias_force is not None)
            except Exception:
                pass
        cb_to_seed = (args.code_bias_force * 1e-6 if args.code_bias_force is not None
                      else _cb.code_ema)
        # -- P2b CONSUMER 1: the code-rate clock comes from the joint state --------------
        # l-a and the joint clk_rate are THE SAME QUANTITY (cp_rate_from_code_bias is
        # exactly f_chip*(l-a), i.e. the clock's drift in chips/s), which makes them
        # directly comparable -- and on 2026-08-09 they disagreed by ~80x. The l-a EMA read
        # +0.003 ppm, predicting 0.031 chips/s = +47 chips of clock drift across a 25-min
        # window in which the clock demonstrably moved +0.21 chips. The joint state read
        # +0.00038 chips/s, which matches the measured trend. So this switch is not the
        # noise reduction it was scoped as (70x quieter, real but secondary) -- it removes a
        # BIAS that has been seeding every unlocked satellite with ~1.8 chips/min of
        # fictitious code drift. Held sats were spared only because the l-a seeding skips
        # them (`if prn in cp_held: continue` below), which is why this survived so long.
        #
        # It also retires the --code-bias-force diagnostic's verdict: that test pinned l-a
        # to 0.001 ppm and read "a wash", but 0.001 ppm is still 25x the truth, so it never
        # tested a correct rate at all.
        if "rate" in joint_consume:
            _jr = _joint_state(rx, band_id, args)
            _ppm = (_jr.clk_rate / args.chip_rate_hz * 1e6) if _jr is not None else None
            # PLAUSIBILITY BOUND, and the incident's most transferable lesson: a consumer
            # must never hand a physically impossible number to the instrument just because
            # an estimator produced it. CHORD's reference is GPS-disciplined; the measured
            # code-rate offset is 4e-5 ppm. The runaway that froze the trackers published
            # -0.028 ppm -- 700x the truth and trivially refusable right here, no matter
            # what went wrong upstream.
            if (_jr is not None and len(_jr._idx) >= args.joint_min_sats
                    and abs(_ppm) <= args.joint_max_rate_ppm):
                cb_to_seed = _jr.clk_rate / args.chip_rate_hz
                _log_rl("jointrate", "code-rate clock from the JOINT state: %+.5f ppm "
                                     "(l-a EMA says %+.5f)"
                        % (cb_to_seed * 1e6,
                           (_cb.code_ema or 0.0) * 1e6), every_s=60.0)
        if cb_to_seed is not None:
            n_seeded = 0
            for prn, seed in seeds.items():
                # ALL sats (fitted included): l-a is common to the receiver, so the smooth pooled
                # value is the correct code rate for everyone -- far quieter than each sat's own
                # short-baseline slope fit (which stays as the cp0 anchor only). Weak/new/coasting
                # sats that never fit still get it here. EXCEPT held (locked) sats: their frozen
                # (anchor, rate) pair must stay consistent -- changing the rate under a stale
                # ref_hop jumps the extrapolated cp.
                if prn in cp_held:
                    continue
                seed.put("la_rate", epoch=seed.get("ref_hop"),
                         code_phase_rate=cp_rate_from_code_bias(
                             seed["doppler_hz"], cb_to_seed, args.hops_per_sec,
                             args.chip_rate_hz, args.carrier_hz))
                n_seeded += 1
            if n_seeded:
                _log_rl("la-seed", "seeded code rate from (l-a) %+.3f ppm%s -> %d sat(s)"
                     % (cb_to_seed * 1e6,
                        " [FORCED]" if args.code_bias_force is not None else "", n_seeded))

        # 3e. OPEN-LOOP TRIM (diagnostic). Command a FIXED carrier_trim_hz to every seeded PRN,
        # independent of --carrier-gain. This is the transfer-function probe the loop debugging
        # needs and could not otherwise get: GNSS_TRIM_FORCE fires once at first seed and lives
        # inside the gain>0 block, so the loop immediately corrects away from it and the step
        # response is unobservable.
        #
        # With the loop OFF and a held trim, deep_rate_hz answers three questions at once:
        #   response = -step  -> plumbed and correctly signed; any non-convergence is dynamics
        #   response = +step  -> SIGN ERROR (and a wrong-signed integrator runs to the rail,
        #                        which is what trims sitting at +28..31 against a +-40 clamp
        #                        look like)
        #   no response      -> the trim never reaches the despread; the loop is open, and no
        #                        amount of gain tuning would ever have helped.
        _ctc = (publisher.carrier_trim_const(args.carrier_trim_const)
                if publisher is not None else args.carrier_trim_const)
        if _ctc is not None:
            for prn in seeds:
                _carrier.trim[prn] = _ctc

        # -- P3 CONSUMER (task #33, --rrate-command): ONE carrier command per (sat, band) --
        # carrier_correction_hz() = receiver-wide f_carrier + this sat's orbital rrate,
        # scaled to THIS band. Posted as carrier_trim_hz (NCO derotation -- a frequency
        # step is phase-continuous in the tracker), REPLACING the trim-loop value: two
        # controllers on one state is the #52 disease this exists to end. doppler_hz is
        # never touched -- seed continuity beats freshness, measured.
        _ctx.jrc = None
        if args.rrate_command:
            if not _rf.full_ok:
                # ⚠️ THE ARM-1 GUARD. A capped measurement past +-deep_rate_max_hz is the
                # best in-cap NOISE bin, and closing the loop on it walked every command
                # (~1 Hz/min). The command therefore requires the combiner to be serving
                # the uncapped fields THIS POLL -- an old tracker binary degrades this
                # chain to shadow, loudly, instead of ratcheting.
                _log_rl("jrr-nofull",
                        "rrate-command HELD: combiner is not serving deep_rate_full_* "
                        "(old tracker binary?) -- shadow only", every_s=120.0)
            else:
                try:
                    _j = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference, gauge_mode=args.joint_gauge)
                    # No receiver-wide term solved yet -> nothing to command. The sigma
                    # gate below then handles per-sat convergence one row at a time.
                    _ctx.jrc = _j if _j.f_carrier_sigma() != float("inf") else None
                except Exception:
                    _ctx.jrc = None
        _ctx.rr_cmd_new = {}
        # RESET PER PASS. These are `+=` counters on an owner object now, so nothing
        # zeroes them implicitly the way a fresh loop local did -- forgetting this makes
        # them run-totals that only ever grow, and the log line reads plausibly for a
        # while before the numbers stop making sense.
        _rf.railed = 0
        _rf.released = 0

        # 4. push consensus seeds to every tracker (DLL trim applied at POST time only)
        _ctx.payload = []
        _ctx.bit_src, _ctx.bit_known = {}, {}
        seeding.stage_push_seeds(_ctx)
        # The commands actually shipped this poll become the rrate feed's reference next
        # poll. REBUILT, not updated: a sat that stopped being commanded (row widened, seed
        # dropped) must fall back to referencing car_trim/0, or a stale command would
        # silently re-enter its measurements forever.
        _rf.cmd_applied.clear()
        _rf.cmd_applied.update(_ctx.rr_cmd_new)
        if _ctx.rr_cmd_new:
            _log_rl("jrr-cmd",
                    "JRR-CMD[%s]: %s Hz (rrate rows -> carrier_trim_hz, %d sat(s), "
                    "%d slew-railed, %d releasing)"
                    % (args.dr_constellation,
                       " ".join("%d:%+.2f" % kv for kv in sorted(_ctx.rr_cmd_new.items())),
                       len(_ctx.rr_cmd_new), _rf.railed, _rf.released), every_s=60.0)
        # WHERE THE PEEL'S SIGNS ACTUALLY CAME FROM this cycle. Without this the only symptom
        # of a source that silently supplies nothing is `nobits` in a health line 10 s later on
        # a different process, which is what made the 30 s-horizon bug hard to see.
        _log_rl("bitsrc", "nav_bits by source: %s; known bits: %s"
                % (dict(sorted(_ctx.bit_src.items())), dict(sorted(_ctx.bit_known.items()))))
        if _nav.health is not None:
            _rep = _nav.health.report()
            if _rep:
                _log_rl("navhealth", _rep)
        if _nav.navbits is not None:
            _log_rl("fleet", _nav.navbits.fleet.stats())
        if os.environ.get("GNSS_SEED_DEBUG"):
            for d in _ctx.payload:
                if str(d["prn"]) in os.environ["GNSS_SEED_DEBUG"].split(","):
                    _log("SEEDDBG %s" % json.dumps(d, sort_keys=True))
        # SEED CURRENCY AUDIT (#39 follow-up, 2026-08-11 -- the per-seed dop log). The
        # question this answers: does the (cp0, dop, rate, ref_hop) tuple we SHIP imply
        # the same physical code phase as the previous tuple we shipped for the same
        # satellite? The tracker re-pins on every POST, so the wrapped difference of
        # dr_seed_phys(prev) and dr_seed_phys(new) at the new ref_hop IS the physical
        # step the tracker's despread experiences -- across EVERY mutation path (birth,
        # slew, trim, nh/CM lifts, coast), not just the slew path the selftest pins.
        # A currency leak -- cp built with one dop, shipped beside another -- shows up
        # here as steps of t_abs*chip_rate/carrier ~ 1696 chips per Hz of mismatch
        # (which is why the equivalent-Hz figure is printed: sub-chip consistency
        # needs the dop bookkeeping good to ~0.6 mHz at 2.24 days of F-engine age).
        # Legitimate steps are the slew caps (0.05/0.5), DLL trim increments, and
        # escapes/re-anchors; anything chips-scale outside those is the leak we have
        # been hunting all day. Audits mod the seed's own long-code modulus so whole
        # code-period assignment flips (#41 class) are visible too.
        _aud_mod = (LC_SEG * CODE_LEN) if LC_SEG > 1 else CODE_LEN
        _aud_steps = []
        for d in _ctx.payload:
            _h_new = int(d.get("ref_hop", 0) or 0)
            _prevA = seed_audit_prev.get(d["prn"])
            if _h_new > 0 and all(k in d for k in ("code_phase_chips", "doppler_hz")):
                seed_audit_prev[d["prn"]] = {
                    k: d[k] for k in ("code_phase_chips", "code_phase_at_ref_chips",
                                      "doppler_hz", "code_phase_rate", "ref_hop",
                                      "doppler_rate_hz_s") if k in d}
            if (_prevA is None or _h_new <= 0 or _h_new < int(_prevA["ref_hop"])
                    or _h_new - int(_prevA["ref_hop"]) > 600 * args.hops_per_sec):
                continue
            # #45 STEP 7 (#43): model WHAT THE TRACKER READS. propagate_seed prefers
            # code_phase_at_ref_chips when the payload carries it -- which the search-fed
            # path always has -- so auditing cp0 measured a stream no tracker consumes:
            # +-90,000-chip "steps" on gps_l5 while those satellites tracked at 40 dB-Hz.
            # tracker_phase_at picks the same reference propagate_seed would.
            _ph_prev = tracker_phase_at(_prevA, _h_new, args.hops_per_sec,
                                        args.chip_rate_hz, args.carrier_hz,
                                        args.code_doppler_sign, _aud_mod,
                                        args.search_fft_len or None)
            _ph_new = tracker_phase_at(d, _h_new, args.hops_per_sec,
                                       args.chip_rate_hz, args.carrier_hz,
                                       args.code_doppler_sign, _aud_mod,
                                       args.search_fft_len or None)
            _stp = ((_ph_new - _ph_prev + _aud_mod / 2.0) % _aud_mod) - _aud_mod / 2.0
            _ddopA = d["doppler_hz"] - _prevA["doppler_hz"]
            _dtA = (_h_new - int(_prevA["ref_hop"])) / args.hops_per_sec
            _aud_steps.append((abs(_stp), d["prn"], _stp, _ddopA, _dtA))
            if abs(_stp) > 5.0:
                _lev_hz = _stp / max(1.0, (_h_new / args.hops_per_sec)
                                     * args.chip_rate_hz / args.carrier_hz)
                # #83: name the writers that produced this tuple. A step's first
                # question was always "who wrote that" -- now the line answers it.
                _sdA = seeds.get(d["prn"])
                _log("SEEDAUDIT STEP PRN %d: %+.2f chips (= %+.4f Hz x lever) "
                     "ddop %+.3f Hz dt %.1f s trim %+.3f%s"
                     % (d["prn"], _stp, _lev_hz, _ddopA, _dtA,
                        _dls.trim.get(d["prn"], 0.0),
                        ("  [%s]" % _sdA.owners()) if isinstance(_sdA, Seed) else ""))
        for _pA in list(seed_audit_prev):
            if _pA not in seeds:
                del seed_audit_prev[_pA]
        if _aud_steps:
            _aud_steps.sort()
            _n = len(_aud_steps)
            _wA, _wp, _ws, _wd, _wt = _aud_steps[-1]
            _log_rl("seedaudit",
                    "SEEDAUDIT n=%d |step| med %.3f p90 %.3f max %.3f chips "
                    "(PRN %d: %+.3f, ddop %+.3f Hz, dt %.1f s)"
                    % (_n, _aud_steps[_n // 2][0],
                       _aud_steps[min(_n - 1, int(_n * 0.9))][0],
                       _wA, _wp, _ws, _wd, _wt), every_s=60.0)
        # EPOCH-SKEW CENSUS (#83 -> #80, measurement first). A seed whose at-epoch
        # field (doppler, rate, at-ref phase) was recorded against one ref_hop but
        # ships beside another is #80's disease -- the hold branches restore the tuple
        # from prev and never touch code_phase_at_ref_chips, and the cp-rate fit can
        # anchor at an older history hop than the phase the overlay lift wrote. Until
        # now that was a code-reading claim; this counts it per cycle, per PRN, with
        # the owner that wrote the skewed field. LOG ONLY -- the fix is Phase 2's,
        # and it starts from this number.
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
                    _skew_ex.append("PRN %s: %s vs ref %s" % (_pS, ",".join(
                        "%s=%s@%s" % (k, v[0], v[1]) for k, v in sorted(_sk.items())),
                        _sS.get("ref_hop")))
        if _skewN:
            # Per-field counts, because the classes are NOT equal: a skewed at-ref phase
            # is the chips-scale #80 disease (fixed in the hold arms above -- its count
            # here is the fix's regression gate, expected 0); a skewed doppler_rate is
            # second-order (enters via the quadratic term only) and stays measured, not
            # hidden, until its own fix.
            _log_rl("epochskew",
                    "EPOCH-SKEW %d/%d seed(s) ship an at-epoch field recorded against "
                    "a different ref_hop (#80 measured; by field: %s): %s"
                    % (_skewN, len(seeds),
                       " ".join("%s:%d" % _kv for _kv in sorted(_skew_f.items())),
                       "; ".join(_skew_ex)), every_s=60.0)
        # TASK #51: hand the fast control thread this cycle's decisions. It substitutes ONLY
        # code_phase_chips into these exact dicts, so nothing the policy put in a seed can be
        # dropped by the faster actuator. `base_cp` is the UNTRIMMED phase, because the fast
        # loop owns dll_trim from here and must not re-add a trim already folded in above.
        # TASK #51 F3: HAND THE C++ FLEET LOOP THIS CYCLE'S DECISIONS.
        #
        # THE SPLIT. Everything above this line is policy -- ephemeris, sky, the clock solve,
        # the joint state, presence, the floors, the deep gate, who is armed -- and it stays
        # here on the 12 s cycle. What crosses is a list of PRNs and four constants. The C++
        # side forms the discriminator, steps the integrator and posts the trim, and it
        # invents no gate and chooses no PRN.
        #
        # ⚠️ leak_per_s, NOT leak. The integrator's leak is PER UPDATE, so at unchanged
        # constants the loop's closed-loop AND noise bandwidths scale with the update rate --
        # 3.1 -> 23.8 Hz is ~8x. The controller divides by the rate it MEASURES (not the
        # nominal 23.84: 3-4% of frames arrive late and a chain with no senders closes
        # nothing), so the loop keeps the bandwidth this cycle asked for whatever rate it
        # actually achieves. --dll-leak-present is per update, so it is converted here.
        #
        # ⚠️ AND THE TARGETS RIDE WITH IT. The controller holds no deployment knowledge: this
        # is the process that knows which instances serve this chain (--trackers, already
        # brace-expanded), so a node added to the fleet reaches the loop without regenerating
        # the gather's config.
        trimarm.stage_fleet_trim_arming(_ctx)

        # LIVE PRN MEMBERSHIP. Placed AFTER everything that seeds or arms this cycle, so a
        # swap lands on a settled chain rather than in the middle of one being driven -- and
        # so the elevation it decides on is this cycle's, not last cycle's. Off unless
        # --prn-reconfig; `report` posts nothing.
        prnmap.stage_prn_membership(_ctx)

        if args.fast_trim_hz > 0.0:
            with fast_lock:
                fast_tmpl.clear()
                for _d in _ctx.payload:
                    _p = _d.get("prn")
                    if _p is None:
                        continue
                    _base = _d.get("code_phase_chips", 0.0) - _dls.trim.get(_p, 0.0)
                    # §4.6: the fast arm substitutes BOTH currencies (see the POST-time
                    # site above) -- the template records the untrimmed phase beside the
                    # untrimmed argument, or None when the payload carries no phase.
                    _ba = _d.get("code_phase_at_ref_chips")
                    if _ba is not None and _ba >= 0.0:
                        _tmod = (LC_SEG * CODE_LEN) if LC_SEG > 1 else CODE_LEN
                        _ba = (_ba - _dls.trim.get(_p, 0.0)) % _tmod
                    else:
                        _ba = None
                    fast_tmpl[_p] = (dict(_d), _base, _ba)
                fast_prns.clear()
                # Only PRNs this cycle judged present AND trimmable. Presence, floors and the
                # deep gate all stay here; the fast thread never re-decides who to touch.
                fast_prns.update(_p for _p in (_dllp.fleet or {})
                                 if (_dllp.fleet[_p].get("present") and _p in fast_tmpl))
            _log_rl("fast-trim",
                    "FAST-TRIM %s: %d PRNs armed, %d updates / %d posts since start "
                    "(%d skipped, %d railed)%s"
                    % (log_tag() or args.signal, len(fast_prns), fast_stats["updates"],
                       fast_stats["posts"], fast_stats["skipped"], fast_stats["rail"],
                       ("  last err %s" % fast_stats["last_err"])
                       if fast_stats["last_err"] else ""),
                    every_s=30.0)

        ok = 0
        for t_ep in trackers:
            try:
                _post("%s/set_seeds" % t_ep, _ctx.payload)
                ok += 1
            except Exception as e:
                _log("set_seeds %s failed: %s" % (t_ep, e))

        # L2C CL SIBLING CHAIN (--cl-tracker; Mechanism A of docs/gnss_shared_knowledge_framework
        # .md). DERIVATION, not acquisition: CM and CL are chip-interleaved on ONE 511.5 kcps
        # clock, so a CL row is the CM row with its code phase lifted into the 1.5 s CL period --
        # cp_CL = (cp_CM + k*10230) mod 767250. Everything else (doppler, dop-rate, carrier trim,
        # residual code rate, ref_hop) is copied VERBATIM: same carrier, same chip clock, so CM's
        # tracked solution IS CL's. nav_bits are deliberately NOT copied (CL is dataless -- the
        # whole point).
        #
        # The segment index k is CLASS-2 knowledge (integer): computed fresh each cycle from
        # coarse absolute time, with the CM code phase supplying the fine time --
        #   k_est = (SV-transmit-time-of-sample-0 mod 1.5)*chip_rate/10230 - cp_CM/10230
        # where t_sv = utc0_sample0 - range/c + sat_clk (the nh-assist convention, proven to
        # 0.01 chip). round(k_est)'s margin |fine_ms| < 10 ms is the pin budget; utc0 anchor
        # (~1-3 ms) + host NTP (~ms) dominate, and fine_ms MEASURES the actual total per sat.
        # k is derived from the row's FINAL cp (post-DLL-trim, post-hold), so the snap and the
        # lift always use the same cp value -- a cp near the 0/10230 wrap moves k by +-1 in
        # exact compensation. k STEPS by +-1 every ~2 h/sat as range advances (tau drifts
        # ~2.7 us/s): expected, logged at debug cadence; any LARGER step is a clock/anchor
        # fault and logs loudly. Never averaged, never held against fresh evidence.
        clsibling.stage_cl_sibling(_ctx)

        # (S5 cross-band read + shadow accumulation + rescue hints moved EARLY, block 2a-xband
        # above -- it must run before the search-hint POST it feeds.)

        _log_rl("active", "active=%s (%d); seeded %d/%d trackers" % (sorted(seeds), len(seeds), ok, len(trackers)))
        if _ctx.cl_report:
            _log_rl("clreport", "CL: " + "; ".join(_ctx.cl_report))

        # S2 OBSERVER: receiver clock, then publish the whole record. dr_state persists
        # across cycles so it is read here rather than at its solve site -- and it is the
        # one shared quantity with ZERO persistence today, so every restart re-bootstraps
        # it from nothing. `integ` is the only per-measurement residual the system keeps
        # anywhere; it is used to VETO satellites and has never been used as a weight.
        statepub.stage_publish_state(_ctx)
        if dhw is not None:
            dhw.flush(t0)
        if args.once:
            return
        # REAL elapsed time, not the frozen cycle clock: this is the one place in the loop
        # that must know how long the pass actually took, or the control cadence would
        # become interval + processing rather than interval.
        _busy = time.time() - t_wall
        record_cycle(log_tag().strip() or "chain", _busy, args.interval)
        dt = args.interval - _busy
        if dt > 0 and _TR.mode != "read":
            time.sleep(dt)


if __name__ == "__main__":
    try:
        main()
    except _TranscriptDone as e:
        # Normal, successful end of a replay: the recording ran out. Report the gate's
        # digest on stdout so a bare `--transcript-read` is useful without the harness.
        _log("transcript replay complete (%s); %d posts, digest %s"
             % (e, len(_TR.posts), _TR.digest()))
        # #83 COVERAGE CENSUS: which seed writers this fixture actually drove. The
        # gate vouches for exactly this set and nothing else -- an owner missing here
        # is a migration path the transcript never exercised, and any '?file:line'
        # entry is an unattributed writer that slipped past the migration.
        from gnss_broker import seed as _seed_mod
        _log("seed writers exercised: %s"
             % (", ".join(sorted(_seed_mod.SEEN_OWNERS)) or "NONE"))
        print(_TR.digest())
    finally:
        _TR.close()
