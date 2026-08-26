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
)
from gnss_broker import instruments                         # noqa: E402  (the DLL's measurements)
from gnss_broker import deadreckon                          # noqa: E402  (the clock pipeline)
from gnss_broker import almanac as almanac_stage            # noqa: E402  (orbit + visibility)
from gnss_broker import codeloop                            # noqa: E402  (the DLL + watchdog)
from gnss_broker import statepub                            # noqa: E402  (the state record)
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


def split_erratic_offsets(offs, hist, now_w, bound_chips, max_age_s, code_len):
    """Split clock-solve offsets into (keep, drop) on per-satellite CONTINUITY of raw d_i.

    `offs` is [(prn, d_i)], d_i = clk + b_i: both stable, so a real satellite's d_i moves
    a few chips between cycles at most.

    ⚠️ HISTORY, because this function has now been wrong in BOTH directions and the second
    time was worse (2026-08-11):

    * The 2026-08-11 afternoon "Doppler lever" version subtracted
      lever = t_i*chip_rate*sign*dop/carrier from d_i before differencing, on the theory
      that d_i carried the detection Doppler times the 2.24-day sample-0 age (1696
      chips/Hz). The arithmetic of the lever is real -- but it CANCELS EXACTLY inside
      cp_loc, because the search's published cp0 embeds -t*chip_rate*sign*dop/carrier
      with the SAME dop and the SAME ref_hop (gnssSeedTransport.cpp detection_phase);
      cp_loc adds it back. Raw d_i never carried the lever at all. Subtracting it
      RE-INTRODUCED the term, so (d - lev) stepped 1696*ddop for any Doppler re-estimate
      > 0.059 Hz -- i.e. every satellite, every cycle (detection Doppler jitters +-60 Hz
      pass to pass). The guard then flagged all sats, hit the min-sats floor, and kept
      everything: a no-op that also invalidated the A/B that "measured" the lever fix.
      MEASURED live 2026-08-11 21:1x: "6 PRN(s) jumped ... keeping all" every cycle;
      and 1423 consecutive live detections show cp_loc continuous to median 0.27 chips
      (p99 2.37, none > 5), so raw d_i is the right quantity to test.

    * The still-open question this guard exists for: the 2026-08-10 PRN 2 incident
      (a non-L5 PRN reading noise, +-3000-chip swings dragging the median) and the
      2026-08-11 19:24-19:38 burst of genuine raw-d_i jumps (era-dependent, absent from
      the live stream at 21:xx). When it fires again, the WHAT MOVED log at the call
      site now decomposes d(cp_loc) -- the cancelled, physical quantity -- so the next
      reading of it does not have to guess (raw dcp is uniform mod L by construction
      and means nothing without the embed removed).

    `hist` is {prn: (t, d_i)} from the previous cycle and IS MUTATED here (every
    satellite's current value is recorded, including dropped ones -- a track that
    settles down must be able to rejoin).

    A satellite with no fresh history is always kept: the test needs two cycles, and
    refusing first sightings would stall the bootstrap.
    """
    keep, drop = [], []
    for prn, d in offs:
        prev = hist.get(prn)
        hist[prn] = (now_w, d)
        if prev is not None and now_w - prev[0] <= max_age_s:
            delta = d - prev[1]
            jump = abs(((delta + code_len / 2.0) % code_len) - code_len / 2.0)
            if jump > bound_chips:
                drop.append((prn, jump))
                continue
        keep.append((prn, d))
    return keep, drop




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
    cl_tracker = resolve_prefix(args.cl_tracker, base) if args.cl_tracker else None
    cl_combiner = resolve_prefix(args.cl_combiner, base) if args.cl_combiner else None
    cnav_combiner = resolve_prefix(args.cnav_combiner, base) if args.cnav_combiner else None
    inav_combiner = resolve_prefix(args.inav_combiner, base) if args.inav_combiner else None
    fnav_combiner = resolve_prefix(args.fnav_combiner, base) if args.fnav_combiner else None
    bcnav2_combiner = resolve_prefix(args.bcnav2_combiner, base) if args.bcnav2_combiner else None
    bcnav1_combiner = resolve_prefix(args.bcnav1_combiner, base) if args.bcnav1_combiner else None
    cnav2_combiner = resolve_prefix(args.cnav2_combiner, base) if args.cnav2_combiner else None
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
    _capable = None
    if args.signal_capability and args.constellation in ("G", "R"):
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
            js = rx_.joint_receiver(band, CODE_LEN, rereference=a.joint_rereference)
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
    status = {}      # prn -> last combiner get_status record (previous cycle; lock gate below)
    # prn -> PROMPT HOLD, fleet prompt power / live noise median, from the previous cycle's
    # fleet dict. Carried exactly like `status` above and for the same reason: the lock gate
    # runs before this cycle's fleet_dll. See the --lock-prompt-hold note for why the gate
    # needs a fold-independent term at all.
    _elem_arch_t = [0.0]   # last per-element archive append (--element-archive-every-s)
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
                     "inst_hops")

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
        _row = status.get(_p) or {}
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
    cl_k = {}        # prn -> last CL segment index k (class-2 pin: log every step, never average)
    cl_pred0 = {}    # anchor-epoch geometry cache: {"key": (utc0, eph_t), "val": {prn: tuple}}
    cl_toff = [0.0]  # measured common clock offset (s): slow EMA of the across-sat median fine
    # CL SEGMENT AUTO-SEARCH (--cl-autoseg, the durable fix for the ~40%-of-launches CL
    # failure, root-caused 2026-07-29). utc0_sample0 is stamped from system_clock::now() on
    # the FIRST USB transfer (airspyInput.cpp:396-405) and carries tens of ms of per-launch
    # startup latency. The auto-center absorbs the FRACTIONAL part (its job), so fine_ms
    # always reads perfect -- but the INTEGER part (N x 20 ms) lands wholesale in the
    # segment index k: fleet-common, fixed at startup, invisible to every seed-level
    # diagnostic. Proven by the full-75 k-scan (k-1 despread 185 with all 74 others at
    # noise) and clinched by --cl-time-adjust -0.020 turning the whole fleet green on the
    # same anchor. N measured 0 (~60% of launches), 1, and >=3 -- so no fixed constant
    # fixes it. Instead: the broker already measures the truth every cycle (the CL-vs-CM
    # verify); when the fleet reads dead-CL-under-strong-CM, step the correction through
    # the spiral 0,-1,+1,-2,... (~25 s per step, negative first: a LATE anchor pushes k
    # HIGH, and USB latency only makes anchors late), and LATCH on green -- the class-2
    # discipline, verify + lockout, never averaged. A working launch latches 0 on the
    # first check and is untouched.
    _anchor_seen = [0.0]   # frame0 as first latched (see the re-check in the cycle loop)
    _anchor_chk = [0.0]    # wall time of the last anchor re-read
    cl_segsearch = {"corr": 0, "idx": 0, "latched": False, "t_step": 0.0}
    # ONE SEGMENT of the long code, in seconds, and a spiral that covers the whole segment
    # space. Both used to be L2C CL's (0.020 s; +-37 of 75 segments) even after LC_EPOCH/
    # LC_SEG were parameterised -- the epoch became generic while the STEP stayed L2C's, so
    # on any other signal a correction of +-1 moved the anchor by 20 segments and the spiral
    # searched 3.7x more space than exists (L5 NH20: 1 ms segments, only 20 of them). Latent
    # rather than harmless: corr is 0 unless --cl-autoseg actually engages, which is why a
    # working launch never showed it. Derive both.
    CL_SEG_S = float(args.long_code_epoch_s) / max(int(args.long_code_segments), 1)
    _clseg_spiral = ([0] + [v for n in range(1, int(args.long_code_segments) // 2 + 1)
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
        _kscan_seq = [float(x) for x in args.cl_kscan_chips.split(",") if x.strip()]
        _kscan_frac = True
    elif args.cl_kscan_segs:
        # explicit segment list -- built for the FULL-75 sweep after the +-2 scan was
        # over-read as exoneration (it exonerated |N|<=2 ONLY; the anchor's startup
        # latency jitter is tens of ms, i.e. potentially several 20 ms segments)
        _kscan_seq = [int(x) for x in args.cl_kscan_segs.split(",") if x.strip()]
        _kscan_frac = False
    else:
        _kscan_seq = [0, -1, 1, -2, 2]   # true k first (baseline), then neighbours
        _kscan_frac = False
    _kfmt = (lambda o: "c%+.2f" % o) if _kscan_frac else (lambda o: "k%+d" % o)
    _kscan = [0]     # [cycle counter], advanced once per CL cycle
    _kscan_deep = {} # offset -> best cl_deep seen for the probe PRN at that offset
    bp_pushed = {}   # prn -> utc0 of the bit_pred table last ATTACHED to a seed row. The
                     # combiner regenerates bit_pred once per EMIT (~1 Hz) but seeds push every
                     # --interval (0.25 s), so re-attaching each cycle is 75% redundant payload
                     # -- and the seed POST has a known too-big failure mode (~14.5k numbers,
                     # see --nav-bits-brdc). The tracker KEEPS its stored table on rows without
                     # nav_bits (u.has_bits guard), so skipping unchanged tables is free. This
                     # matters most at L5: 1 ms records make each table 4x an L1 pilot's.
    navbits = None   # LNAV decode-and-predict (P7a); created lazily on the first nav_obs row
    cnav = None      # CNAV decoder (--nav-decoder cnav, L2C-CM/L5-I); created lazily likewise
    inav = None      # I/NAV decoder (--inav-combiner, Galileo E1B); created lazily
    _inav_log_t = [0.0]
    fnav = None      # F/NAV decoder (--fnav-combiner, Galileo E5a-I); created lazily
    _fnav_log_t = [0.0]
    bcnav2 = None    # B-CNAV2 decoder (--bcnav2-combiner, BeiDou B2a-D); created lazily
    _bcnav2_log_t = [0.0]
    bcnav1 = None    # B-CNAV1 decoder (--bcnav1-combiner, BeiDou B1C-D); created lazily
    _bcnav1_log_t = [0.0]
    cnav2 = None     # CNAV-2 decoder (--cnav2-combiner, GPS L1C-D); created lazily
    _cnav2_log_t = [0.0]
    bcnav3 = None    # B-CNAV3 decoder (--nav-decoder bcnav3, BeiDou B2b PRIMARY chain); lazy
    _bcnav3_log_t = [0.0]
    navbits_log_t = 0.0
    # CONSTRUCTED bits for satellites too weak to decode (2026-07-25). The decoder above needs
    # 620 contiguous parity-clean bits, which is exactly what a weak satellite cannot give; this
    # one needs no decode at all, because subframes 1-3 ARE the ephemeris and BRDC has it for
    # every satellite in the sky. Same predict() contract, so it is simply a lower-priority
    # source in the chain below.
    navbrdc = None
    navhealth = None  # continuous predicted-vs-air agreement monitor (navbit_health)
    cp_held = set()  # PRNs whose cp anchor is FROZEN this cycle (locked -> DLL owns the residual)
    utc0_sample0 = 0.0  # CL time-assist: wall UTC of capture sample 0 (fetched lazily in the loop)
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
    _adr_span_now = {}
    # #92 THE HANDOVER. Owns the bound, the transport and the cumulative posted deltas --
    # which are an INSTRUMENT CORRECTION, not bookkeeping: the shadow's ramp series subtracts
    # them, or a ledger transfer masquerades as the drift #93 is trying to measure.
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
    rr_full_ok = False   # this poll's feed ran on the UNCAPPED fields -- the command requires
                         # it (a capped measurement past +-5 Hz is noise; feeding a loop with
                         # it is what walked arm 1)
    _span_fed_t = {}     # prn -> t of the last span-mode FED fine value (the non-overlap
                         # throttle; unused when --rrate-phase-span-s is 0)
    adr_ring = {}        # prn -> [((arc, records, res_cycles, trim), cmd, t), ...] -- the
                         # long-span baseline ring (--rrate-phase-span-s); oldest first,
                         # pruned to 2x span. Empty/unused when the flag is 0.
    adr_prev = {}        # prn -> ((arc, records, res_cycles), cmd_then) at the last poll --
                         # the PLL fine observable differences THESE (#33 phase-step feed)
    rr_kcoh_t = {}       # prn -> t of the last ACCEPTED kcoh rate feed (#83 P3 step 1);
                         # holds the coarse deweight like rr_fine_t, same hold window.
    rr_kcoh_fed = {}     # {"last": <the kcoh dict object last fed>} -- the estimator is
                         # THROTTLED (_run_est), so _est_last serves the same dict for
                         # several cycles; re-feeding it would count one measurement as
                         # many and the filter would grow confident on repetition.
    rr_fine_t = {}       # prn -> t of the last ACCEPTED fine measurement. The FLL->PLL
                         # handoff reads this: while the lock is fresh the coarse feed is
                         # de-weighted, and on expiry it silently returns to full weight.
    rr_cmd_applied = {}  # prn -> carrier command POSTED last poll (#33 P3). The rrate feed's
                         # reference: deep_rate_hz is measured on records already derotated
                         # by this, so the feed adds it back to reference y at the base seed.
                         # Rebuilt each poll at POST time; a sat that stops being commanded
                         # drops out, so a stale command can never re-enter a measurement.
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
    code_bias_ema = None   # smoothed receiver code-rate clock offset (l-a), dimensionless (~2.6 ppm airspy)
    if args.code_bias_init is not None:
        code_bias_ema = args.code_bias_init * 1e-6
        _log("code-rate clock offset warm-started at %+.3f ppm (--code-bias-init)" % (code_bias_ema * 1e6))
    elif args.code_bias_file:
        try:
            with open(args.code_bias_file) as f:
                code_bias_ema = float(f.read().strip()) * 1e-6
            _log("code-rate clock offset loaded %+.3f ppm from %s" % (code_bias_ema * 1e6, args.code_bias_file))
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
    code_bias_cal = code_bias_ema  # l-a calibration reference (None if cold)
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
    _cnav_sig = ("GPS_L2C_CNAV" if abs((args.carrier_hz or 0) - 1227.6e6) < 5e6
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
        ("GPS_L1_LNAV",    "G", _svp_lnav, "toe", lambda: navbits) if _decfb else None,
        ("GAL_E1B_INAV",   "E", _svp_inav, "t0e", lambda: inav) if _decfb else None,
        ("GAL_E5AI_FNAV",  "E", _svp_fnav, "t0e", lambda: fnav) if _decfb else None,
        (_cnav_sig,        "G", _svp_cnav, "toe", lambda: cnav) if _decfb else None,
        ("BDS_B1C_BCNAV1", "C", _svp_bc1,  "t_oe", lambda: bcnav1) if _decfb else None,
        ("BDS_B2A_BCNAV2", "C", _svp_bc2,  "t_oe", lambda: bcnav2) if _decfb else None,
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
    HIST_LEN = 8           # snapshots kept for the slope fit
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

    def _stage_cl_sibling():
        """S4: the CM/CL SIBLING CHAIN -- seed the long-code tracker from this chain's solution.
        
        The sibling despreads the SAME satellite on a different code (GPS L2 CM/CL), so it needs no
        search of its own: everything it wants -- the visible set, the predicted Doppler, the receiver
        clock -- has already been solved here. It consumes; it never feeds back. That is why this
        whole stage has ZERO outputs into the rest of the cycle, and why it is the first block that
        could be lifted out of the loop body unchanged.
        
        ⚠️ THE ANCHOR EPOCH IS EVALUATED SEPARATELY, NOT EXTRAPOLATED. Linear extrapolation back to
        utc0 is no cure for orbit curvature (tens of ms over hours), so this runs a SECOND model
        evaluation at the fixed anchor epoch, cached per ephemeris refresh -- the anchor never moves,
        only the ephemeris does."""
        if cl_tracker and utc0_sample0 and args.almanac and _ctx.pred:
            # tau AND the SV clock must be evaluated AT THE ANCHOR EPOCH (utc0_sample0),
            # because that is where cp is referenced. The first deploy evaluated them at
            # "now": invisible at launch, but the per-sat error grows at range_rate/c (up to
            # +-2.7 us/s) -- +-10 ms/h, a guaranteed universal mis-pin by hour 2-3 -- and
            # LINEAR extrapolation back is no cure (orbit curvature ~tens of ms over hours).
            # So: a second model evaluation at the FIXED anchor epoch, cached per ephemeris
            # refresh (the anchor never moves; only the ephemeris does).
            if brdc_alm is not None:
                _k0 = (round(utc0_sample0, 3), brdc_alm.get("eph_t"))
                if cl_pred0.get("key") != _k0:
                    try:
                        cl_pred0["val"] = brdc_predict(
                            brdc_alm, args.lat, args.lon, args.alt, alm_sys, alm_min_prn,
                            datetime.fromtimestamp(utc0_sample0, tz=timezone.utc),
                            args.carrier_hz)
                        cl_pred0["key"] = _k0
                        _log("CL: anchor-epoch geometry rebuilt (%d sats)"
                             % len(cl_pred0["val"]))
                    except Exception as e:
                        _log("CL: anchor-epoch predict failed (%s); now-epoch fallback "
                             "(fine will drift ~ms/10min)" % e)
            _pred0 = cl_pred0.get("val") or {}
            cl_payload = []
            _fines = []
            for d in payload:
                pv = _ctx.pred.get(d["prn"])
                # No geometry -> no k -> no CL row (fail closed; CM unaffected). Below the
                # elevation mask -> ALSO no row: those seeds are the below-horizon NOISE
                # PROBES, whose cp is deliberately noise -- deriving CL from them wastes a
                # tracker slot and their "fine" poisoned the first margin analysis (the
                # same el<0 trap the obs-aggregate rule exists for).
                if pv is None or pv[2] < args.mask_deg:
                    continue
                _g = _pred0.get(d["prn"])
                tau0 = (_g[3] if _g is not None else pv[3]) / C_LIGHT
                clk0 = _g[4] if _g is not None else pv[4]
                # Their segment-search correction (cl_segsearch) on OUR parameterised epoch:
                # LC_EPOCH/LC_SEG replaced the hardcoded 1.5 s / 75 segments so the CL assist
                # is not L2C-CL-only. Defaults are 1.5/75, so this is a no-op on the prototype.
                t_sv = (utc0_sample0 - tau0 + clk0 + args.cl_time_adjust - cl_toff[0]
                        + cl_segsearch["corr"] * CL_SEG_S)
                cl_chips = (t_sv % LC_EPOCH) * args.chip_rate_hz
                cp_cm = d["code_phase_chips"] % CODE_LEN
                k = int(round((cl_chips - cp_cm) / CODE_LEN))
                fine_ms = (cl_chips - cp_cm - k * CODE_LEN) / args.chip_rate_hz * 1e3
                k %= LC_SEG
                _fines.append(fine_ms)
                if abs(fine_ms) > 5.0:
                    # Half the +-10 ms budget gone AFTER centering: the seed still goes out
                    # (a wrong k reads as CL noise, which the verify below names;
                    # withholding would silently dark the chain instead of showing it).
                    _log_rl("clthin-%d" % d["prn"],
                            "CL PIN MARGIN THIN: PRN %d fine %+.2f ms of +-10 (post-center; "
                            "clock-offset est %+.2f ms)"
                            % (d["prn"], fine_ms, cl_toff[0] * 1e3))
                # K-SCAN: for the one probe PRN, offset the seed by the current step --
                # whole segments (segment mode) or fractional chips (comb mode) -- so its
                # CL row despreads at the shifted position. Everything else (fine, k
                # report, auto-center) uses the true k untouched; only the probe's seed is
                # shifted, so the scan cannot perturb the fleet's pin.
                _cp_extra = 0.0
                k_seed = k
                if args.cl_kscan_prn and d["prn"] == args.cl_kscan_prn:
                    _off = _kscan_seq[(_kscan[0] // max(args.cl_kscan_dwell, 1))
                                      % len(_kscan_seq)]
                    if _kscan_frac:
                        _cp_extra = _off
                    else:
                        k_seed = (k + int(_off)) % LC_SEG
                dcl = {kk: d[kk] for kk in ("prn", "doppler_hz", "code_phase_rate", "ref_hop",
                                            "doppler_rate_hz_s", "carrier_trim_hz") if kk in d}
                # Their k-scan probe (k_seed/_cp_extra) on OUR parameterised segment count.
                dcl["code_phase_chips"] = ((cp_cm + k_seed * CODE_LEN + _cp_extra)
                                           % (float(LC_SEG) * CODE_LEN))
                cl_payload.append(dcl)
                kp = cl_k.get(d["prn"])
                if kp is not None and kp != k:
                    msg = ("CL k-step PRN %d: %d -> %d (fine %+.2f ms)"
                           % (d["prn"], kp, k, fine_ms))
                    if (k - kp) % LC_SEG in (1, LC_SEG - 1):
                        _log_rl("clk-%d" % d["prn"], msg)  # geometry advancing: routine
                    else:
                        _log("CL K-JUMP (not +-1 -- clock/anchor fault?): " + msg)
                cl_k[d["prn"]] = k
                cl_report.append("PRN %d k=%d fine %+.1f ms" % (d["prn"], k, fine_ms))
            # AUTO-CENTER: the across-sat MEDIAN fine is the common receiver-clock/anchor
            # offset (class-1 continuous state -- measured +4.5 ms on first light, i.e. half
            # the +-10 ms pin budget spent on a knowable constant). A slow EMA of the median
            # (tau ~10 s at 5 Hz) folds it back into the next cycle's t_sv, re-centering
            # every sat's margin; the +-8 ms clamp keeps a broken clock from walking the pin
            # off a segment. Median (not mean): one sat mid k-step must not drag the fleet.
            # The k pins themselves stay integer and per-cycle -- this only recenters the
            # window they are rounded in.
            if _fines:
                _med = sorted(_fines)[len(_fines) // 2] * 1e-3
                cl_toff[0] = max(-8e-3, min(8e-3, cl_toff[0] + 0.02 * _med))
                _log_rl("cltoff", "CL clock-offset est %+.2f ms (median fine %+.2f ms, "
                        "%d sats)" % (cl_toff[0] * 1e3, _med * 1e3, len(_fines)))
            if cl_payload:
                try:
                    _post("%s/set_seeds" % cl_tracker, cl_payload)
                except Exception as e:
                    _log("CL set_seeds %s failed: %s" % (cl_tracker, e))
            # VERIFY (the other half of the class-2 pin): CL deep_snr vs CM per PRN. Equal
            # power split -> a right k reads ~CM's deep; a wrong k despreads noise. Read
            # beside the k it verifies, in this log, so the pin and its evidence never
            # separate.
            if cl_combiner:
                try:
                    cls_ = {int(r["prn"]): r for r in _get("%s/get_status" % cl_combiner)}
                    pairs = []
                    for prn in sorted(cl_k):
                        cm_d = (status.get(prn) or {}).get("deep_snr") or 0.0
                        cl_d = (cls_.get(prn) or {}).get("deep_snr") or 0.0
                        pairs.append("%d:%.0f/%.0f" % (prn, cm_d, cl_d))
                    if pairs:
                        _log_rl("clverify", "CL verify (PRN:cm/cl deep): " + " ".join(pairs))
                    # SEGMENT AUTO-SEARCH, judged on this same verify data. Step only when
                    # the fleet is unambiguously dead (>=2 strong CM sats, ZERO green CL) --
                    # a partly-green fleet must never be stepped away from -- and latch the
                    # moment >=2 strong sats read green. Disabled while a k-scan diagnostic
                    # is shifting the probe's seed.
                    if (args.cl_autoseg and not args.cl_kscan_prn
                            and not cl_segsearch["latched"]):
                        _strong = [prn for prn in cl_k
                                   if ((status.get(prn) or {}).get("deep_snr") or 0.0) > 50.0]
                        _green = [prn for prn in _strong
                                  if ((cls_.get(prn) or {}).get("deep_snr") or 0.0)
                                  > ((status.get(prn) or {}).get("deep_snr") or 0.0) / 3.0]
                        _nowv = _now()
                        if cl_segsearch["t_step"] == 0.0:
                            cl_segsearch["t_step"] = _nowv
                        elif len(_strong) >= 2 and len(_green) >= 2:
                            cl_segsearch["latched"] = True
                            _log("CL SEG-SEARCH LATCHED: correction %+d segment(s) "
                                 "(compensating a %+.0f ms utc0_sample0 anchor error); "
                                 "%d/%d strong sats green"
                                 % (cl_segsearch["corr"], -cl_segsearch["corr"] * CL_SEG_S * 1e3,
                                    len(_green), len(_strong)))
                        elif (len(_strong) >= 2 and not _green
                              and _nowv - cl_segsearch["t_step"] > args.cl_autoseg_dwell):
                            cl_segsearch["idx"] = ((cl_segsearch["idx"] + 1)
                                                   % len(_clseg_spiral))
                            cl_segsearch["corr"] = _clseg_spiral[cl_segsearch["idx"]]
                            cl_segsearch["t_step"] = _nowv
                            _log("CL SEG-SEARCH: fleet dead under strong CM (%d strong, 0 "
                                 "green) -- trying correction %+d segment(s)"
                                 % (len(_strong), cl_segsearch["corr"]))
                    # K-SCAN readout. The seeded offset for THIS cycle is
                    # seq[(cycle//dwell) % n]; the combiner is integrating that same offset
                    # (the seed goes out just above, then we read deep next). CL deep takes
                    # tens of seconds to build, and the tracker needs a few cycles to re-lock
                    # after a segment jump -- so only RECORD in the back half of each dwell,
                    # and only DECLARE a result after a full sweep has completed, requiring a
                    # winner that clears noise by a real margin.
                    if args.cl_kscan_prn:
                        _p = args.cl_kscan_prn
                        _dw = max(args.cl_kscan_dwell, 1)
                        _idx = (_kscan[0] // _dw) % len(_kscan_seq)
                        _cur = _kscan_seq[_idx]
                        _pos = _kscan[0] % _dw            # position within this dwell
                        _cl = (cls_.get(_p) or {}).get("deep_snr") or 0.0
                        _cm = (status.get(_p) or {}).get("deep_snr") or 0.0
                        if _pos >= _dw // 2:              # settled back half only
                            _kscan_deep[_cur] = max(_kscan_deep.get(_cur, 0.0), _cl)
                        _log_rl("kscan-%d" % _p,
                                "CL KSCAN PRN %d: %s (dwell %d/%d) cl_deep %.0f cm %.0f"
                                % (_p, _kfmt(_cur), _pos, _dw, _cl, _cm), every_s=5.0)
                        # a full sweep completes when we return to seq index 0 having filled
                        # every offset; declare once per sweep.
                        if (_idx == 0 and _pos == 0 and _kscan[0] > 0
                                and len(_kscan_deep) >= len(_kscan_seq)):
                            _best = max(_kscan_deep, key=_kscan_deep.get)
                            _bd = _kscan_deep[_best]
                            _2nd = sorted(_kscan_deep.values())[-2]
                            _cmp = (status.get(_p) or {}).get("deep_snr") or 0.0
                            _clear = _bd > 20.0 and _bd > 3.0 * max(_2nd, 1.0)
                            _win_says = (
                                ("SUB-CHIP/COMB FAULT, offset %s chips" % _kfmt(_best)
                                 if _best != 0 else
                                 "true cp CORRECT -- fault is not a sub-chip seed offset")
                                if _kscan_frac else
                                ("WHOLE-SEGMENT ANCHOR BUG, magnitude %s" % _kfmt(_best)
                                 if _best != 0 else
                                 "true k CORRECT -- fault is NOT the segment pin"))
                            _log("CL KSCAN PRN %d SWEEP: %s -> %s"
                                 % (_p, " ".join("%s:%.0f" % (_kfmt(o), _kscan_deep[o])
                                                 for o in sorted(_kscan_deep)),
                                    ("best %s clears noise %.0fx: %s" % (
                                        _kfmt(_best), _bd / max(_2nd, 1.0), _win_says))
                                    if _clear else
                                    "NO offset in this range despreads (best %s only %.0f "
                                    "vs cm %.0f) -- %s" % (
                                        _kfmt(_best), _bd, _cmp,
                                        "not a sub-chip seed offset either; the fault is "
                                        "past the seed (replica/carrier/comb in the C++)"
                                        if _kscan_frac else
                                        "not a small whole-segment error; widen the range "
                                        "or look past the pin")))
                            _kscan_deep.clear()          # fresh accumulation next sweep
                    _kscan[0] += 1
                except Exception as e:
                    _log_rl("clverify", "CL verify poll failed: %s" % e)

    def _stage_carrier_loop():
        """3d: THE SHARED CARRIER LOOP -- integrate the combiner's full-band carrier residual into a
        per-PRN trim, and command it.
        
        The code twin of 3c. Nothing here leaves the stage: the trim is delivered through `car_trim`
        and the seeds, so this block has ZERO live outputs into the rest of the cycle.
        
        ⚠️ IT IS OFF IN PRODUCTION. `--carrier-gain` is 0.0 on CHORD, so on the live instrument this
        entire routine is dead code, and has been since the loop was plumbed. That is a deliberate
        state (the loop was measured to make things worse -- see #71), not an oversight, but it means
        NOTHING HERE IS EXERCISED BY THE FIXTURES EITHER: the digest gate is silent about every line
        below. A latent state-clobbering bug lived here undisturbed until 2026-08-26 for exactly that
        reason. Treat changes to this routine as unverified until the loop is armed on sky."""
        if args.carrier_gain > 0.0:
            for _p in [p for p in _carrier.trim_force if p in seeds]:
                _carrier.trim[_p] = _carrier.trim_force.pop(_p)
                _log("TRIM FORCE (bench): PRN %d car_trim POISONED to %+.1f Hz"
                     % (_p, _carrier.trim[_p]))
            # (computed once above -- see the shared-call note; {} when carrier_source
            #  is not "rate", which preserves the old carrier_hz_resid fallback below)
            rate_resid, rate_consensus = _rr_resid, _rr_cons
            car_report = []
            for prn in list(seeds):
                if prn in probe_set:
                    # NOISE PROBES ARE EXEMPT. A probe is a satellite we deliberately point at
                    # empty sky: there IS no carrier, so its "residual" is pure noise, and with
                    # no lock gate the loop integrates that noise straight into the trim until
                    # it random-walks to the clamp. Harmless to the probe (it measures noise
                    # either way) -- but the trim moves the REPORTED Doppler, which made the
                    # probes look like the churniest objects in the sky, and the beam map's
                    # churn gate then excised 100% of them (2026-07-14). The pedestal
                    # calibrator was being destroyed by a loop chasing noise it should never
                    # have been fed. Nothing downstream wants a carrier trim on a probe.
                    continue
                rec = status.get(prn, {})
                # (ALIAS ESCAPE v1/v2 lived here until the 07-19 audit A4. The alias-capture
                #  disease it targeted -- resid estimator ambiguous mod 1/(2*T_rec), NCO
                #  parked on the alias for 40+ min -- is owned by two surviving mechanisms:
                #  a stale f_ref offset is snapped by the tight tracker fence (free under
                #  --dop-continuous), and a walked trim latch is the watchdog's lifecycle
                #  rescue. v1's fleet-kill postmortem lives in git history at 069e8770.)
                resid = float(rec.get("carrier_hz_resid", 0.0))
                if args.carrier_source == "rate":
                    # THE MEASUREMENT THE LOOP WAS ALWAYS MISSING (2026-08-04). carrier_hz_resid
                    # is signal-free -- 0.519 Hz on signal vs 0.492 Hz on noise -- which is why
                    # closing this loop on it made every metric worse (8.18.4) and why leaving it
                    # open left the carrier free-running (the ramp). deep_rate_hz comes from the
                    # combiner's phase-rate search: peak/median 17.9-22.0 on signal against
                    # 2.8-6.1 on noise, and split-half agreement of 0.0-0.5 Hz on strong sats
                    # against the <1 Hz a 1.05 s window needs.
                    #
                    # It is a DELTA, not an absolute: the search runs on records the tracker
                    # already derotated by car_trim, so what it reports is what remains. That is
                    # exactly the residual this loop integrates -- no reference change.
                    rr = rate_resid.get(prn)
                    if rr is None and args.carrier_rate_inherit:
                        rr = rate_consensus # failed its own gate: take the fleet's answer
                    if rr is None:
                        continue
                    resid = rr
                if resid == 0.0:
                    continue
                # TWO-MODE LOOP. BOOTSTRAP (never certified since seed): legacy behavior --
                # accept any residual at full gain, because at seed the trim is 0, the true
                # residual IS the ~30 Hz clock offset, and the window cannot cohere until the
                # trim pulls in (gating on certification here would deadlock acquisition).
                # TRACK (certified once): a residual is only a measurement when its window
                # was COHERENT -- gate on certification AND significance, and slew-clamp.
                # The sig gate alone let re-lock transition rows (amp back, phase not) kick
                # converged trims 30 Hz a shot: E36 at 48 dB-Hz visited the +-100 Hz rails.
                # Dedup FIRST -- at most one gate-check + integration per fresh emit: the
                # combiner emits a new residual every ~1.5 s window while this loop polls at
                # ~5 Hz; integrating a stale value 5-7x over-applies the gain and oscillates
                # (observed +-20 Hz swings), and the --carrier-refade counter below must
                # count EMITS, not polls. A changed value marks a fresh emit.
                if resid == _carrier.last.get(prn):
                    continue
                _carrier.last[prn] = resid
                coh_ok = (rec.get("coherence_s") or 0.0) > 0.0
                # ---- VERIFYING: an applied step hypothesis is judged by OUTCOME ----
                # (explain-apply-verify, 2026-07-22): a trim correction is a falsifiable
                # hypothesis -- either coherence returns / the residual collapses within
                # CARRIER_VERIFY_EMITS, or it was WRONG and is reverted + escalated to a
                # full re-acquire. This bounded closed loop is exactly what the two
                # retracted open-loop escapes (v2 EMA unwrap, loose step-accept) were
                # missing: a wrong correction costs one reverted step, never compounds.
                if prn in _carrier.verify:
                    v = _carrier.verify[prn]
                    v["emits"] += 1
                    if coh_ok or abs(resid) < CARRIER_EXPLAIN_HZ:
                        del _carrier.verify[prn]
                        _carrier.fade.pop(prn, None)
                        _log("CARRIER STEP VERIFIED PRN %d: healed in %d emit(s) "
                             "(coh=%s, resid %+.2f Hz)" % (prn, v["emits"], coh_ok, resid))
                        # fall through: this emit integrates normally below
                    elif v["emits"] >= CARRIER_VERIFY_EMITS:
                        _carrier.trim[prn] = v["prev_trim"]  # revert the refuted hypothesis
                        del _carrier.verify[prn]
                        _carrier.locked.discard(prn)         # escalate: BOOTSTRAP re-acquire
                        _carrier.step_t[prn] = t0 + 50.0     # ~60 s hypothesis lockout
                        _carrier.fade.pop(prn, None)
                        _log("CARRIER STEP REFUTED PRN %d: no heal after %d emits (resid "
                             "%+.2f Hz) -> trim reverted to %+.2f, BOOTSTRAP re-pull"
                             % (prn, CARRIER_VERIFY_EMITS, resid, v["prev_trim"]))
                        continue
                    else:
                        continue  # verdict pending: hold, no further corrections
                # ---- POST-BLEED VERIFY (trim-bleed, explain-apply-verify) ----
                # A re-pin is a falsifiable hypothesis too: after folding the trim into f_ref, the
                # sat must stay coherent. Unlike a step, there is NOTHING to revert (the re-pin is
                # phase-continuous and car_trim correctly re-grows from 0 via the normal loop), so
                # this only OBSERVES the outcome and lets the lockout prevent churn. It falls
                # through to integrate normally either way (trim re-grows to the small remnant).
                if prn in _carrier.bleed_verify:
                    bv = _carrier.bleed_verify[prn]
                    bv["emits"] += 1
                    if bv["emits"] >= args.carrier_bleed_verify_emits:
                        del _carrier.bleed_verify[prn]
                        # Judge by the SETTLED residual, not coh_ok (which blips for ~1 emit on the
                        # deep-window reset a re-pin causes, good bleed or not). A residual at/under
                        # the bar means the fold left the carrier aligned; a large one is a real
                        # miss (the loop re-grows the trim from 0 either way -- and even a mild miss
                        # already REDUCED the standing trim, so the bar is generous).
                        if abs(resid) <= args.carrier_bleed_ok_hz:
                            _log("CARRIER BLEED VERIFIED PRN %d: resid settled %+.2f Hz "
                                 "(<= %.2f) over %d emits, trim now %+.2f"
                                 % (prn, resid, args.carrier_bleed_ok_hz, bv["emits"],
                                    _carrier.trim.get(prn, 0.0)))
                        else:
                            _log("CARRIER BLEED REFUTED PRN %d: resid %+.2f Hz (> %.2f) after "
                                 "%d emits -- loop re-grows trim, %.0f s lockout"
                                 % (prn, resid, args.carrier_bleed_ok_hz, bv["emits"],
                                    args.carrier_bleed_lockout_s))
                sig = (max(rec.get("deep_snr") or 0.0, rec.get("amp_snr") or 0.0)
                       if coh_ok else 0.0)
                tracking = prn in _carrier.locked
                if coh_ok and sig >= args.carrier_min_sig > 0.0:
                    _carrier.locked.add(prn)
                gated = (tracking and args.carrier_min_sig > 0.0
                         and (not coh_ok or sig < args.carrier_min_sig))
                fade_gated = gated  # incoherent/weak: the resid estimator is NOT trusted
                if not gated and tracking and args.carrier_innov_hz > 0.0 \
                        and abs(resid) > args.carrier_innov_hz:
                    gated = True  # certified-but-implausible residual: the estimator is lying
                if gated:
                    # Presence first (shared by the hypothesis stage AND refade below):
                    # amp OR a fresh strong detection -- see the refade note.
                    _df = det_fresh.get(prn)
                    present = ((rec.get("amp_snr") or 0.0) >= args.hold_snr
                               or (_df is not None and t0 - _df[1] < 10.0
                                   and prn in best
                                   and best[prn][0] >= 2.0 * args.acquire_snr))
                    # ---- STRONG-INCOHERENT HYPOTHESIS (explain-apply-verify) ----
                    # The innovation gate above is COHERENT physics (a cohering sat cannot
                    # carry a multi-Hz residual, so such a reading is a lie). An INCOHERENT
                    # sat has no such bound -- and when the observables close on one story
                    # (signal PRESENT + M consecutive residuals AGREE + the agreed value is
                    # big enough to EXPLAIN the decoherence), the residual is the
                    # explanation, not a lie (type specimen C19 2026-07-22: full amp, dark,
                    # parked at +3.03 Hz for minutes while every gate held). Apply the FULL
                    # agreed correction ONCE and enter VERIFYING (top of loop): heal within
                    # 3 emits or be reverted + escalated. Applies to BOTH gate flavors --
                    # the verify stage is what makes fade-gated acceptance safe (the loose
                    # v1 escape lacked it and churned the fleet; a wrong hypothesis now
                    # costs one reverted step + a 60 s lockout).
                    if args.carrier_step_accept > 0 and present:
                        hist = _carrier.step_hist.setdefault(prn, [])
                        hist.append((t0, resid))
                        del hist[:-args.carrier_step_accept]
                        band = max(2.0, args.carrier_innov_hz)
                        if (len(hist) >= args.carrier_step_accept
                                and t0 - hist[0][0] < 30.0
                                and t0 - _carrier.step_t.get(prn, 0.0) >= 10.0):
                            vals = sorted(r for _, r in hist)
                            med = vals[len(vals) // 2]
                            if (vals[-1] - vals[0] < band
                                    and abs(med) >= CARRIER_EXPLAIN_HZ):
                                prev_trim = _carrier.trim.get(prn, 0.0)
                                _carrier.trim[prn] = max(-args.carrier_max_hz,
                                                    min(args.carrier_max_hz,
                                                        prev_trim + med))
                                _carrier.step_t[prn] = t0
                                _carrier.step_hist[prn] = []
                                _carrier.verify[prn] = {"prev_trim": prev_trim, "emits": 0}
                                _log("CARRIER STEP HYPOTHESIS PRN %d: %d agreeing gated "
                                     "resids (med %+.2f Hz, spread %.2f) -> trim %+.2f, "
                                     "VERIFYING (heal in %d emits or revert)"
                                     % (prn, args.carrier_step_accept, med,
                                        vals[-1] - vals[0], _carrier.trim[prn],
                                        CARRIER_VERIFY_EMITS))
                                continue
                    # --carrier-refade: the two gates otherwise form an ABSORBING state for a
                    # sat whose NCO really stepped (hold release / escape re-anchor without
                    # trim precomp): coherence never returns while the residual exceeds the
                    # innovation gate, so the trim can never unwind. Sustained gating WITH
                    # the sat still present = carrier genuinely lost, not a fade: demote to
                    # BOOTSTRAP and let the next residual re-pull at full gain (seconds).
                    # A genuinely faded sat (no amp AND no detection) keeps coasting on the
                    # feed-forward -- the pathology --carrier-min-sig exists to protect.
                    # Presence = amp OR a fresh strong detection (2026-07-19): the amp-only
                    # bar left WEAK-chain sats (L2C at cn0 27-31: amp 5-20, under the
                    # hold bar while decohered) latched with a STANDING sub-innovation
                    # residual (~2 Hz measured: kills 1-s deep windows, too small for the
                    # innov gate, coherence gate holds the trim, refade never fired) --
                    # the C20 absorbing state one level down. The search still sees these
                    # sats fine; det presence is the same test the watchdog trusts.
                    # (`present` computed at the top of the gated branch, shared with the
                    # hypothesis stage.)
                    _carrier.fade[prn] = _carrier.fade.get(prn, 0) + 1 if present else 0
                    # FLICKER GUARD (2026-07-20): a SUB-innovation residual on a sat that
                    # cohered seconds ago is certification-bar sig flicker, not a stepped
                    # NCO -- the re-pull has nothing to pull (settled-era E1/B1C: ~700
                    # REACQs/3 h at mean |resid| 1.7 Hz, all no-ops). Suppress the demotion
                    # for those; a STANDING decoherence (the L2C C20 absorbing state:
                    # sub-gate resid, dark for minutes) still demotes once the sat has been
                    # incoherent longer than the window. Inactive when the watchdog is off
                    # (wd_coh_t empty -> old behavior).
                    _flicker = (args.refade_flicker_s > 0.0
                                and abs(resid) < args.carrier_innov_hz
                                and t0 - _watchdog.coh_t.get(prn, 0.0) < args.refade_flicker_s)
                    if (args.carrier_refade > 0 and not _flicker
                            and _carrier.fade.get(prn, 0) >= args.carrier_refade):
                        _carrier.locked.discard(prn)
                        _carrier.fade.pop(prn, None)
                        _log("CARRIER REACQ PRN %d: %d consecutive gated emits at full amp "
                             "(last resid %+.2f Hz) -> BOOTSTRAP re-pull"
                             % (prn, args.carrier_refade, resid))
                    continue  # this emit stays held: coast on the feed-forward
                _carrier.fade.pop(prn, None)
                _carrier.step_hist.pop(prn, None)  # ungated emit: gated-run agreement is stale
                if not tracking and args.carrier_det_gate_s > 0.0:
                    # BOOTSTRAP WALK GATE: no fresh detection = no evidence the estimator
                    # has a signal to measure; its residual is noise and integrating it
                    # random-walks the trim (see --carrier-det-gate-s). Hold and coast.
                    _fr = det_fresh.get(prn)
                    if _fr is None or t0 - _fr[1] > args.carrier_det_gate_s:
                        continue
                if prn not in _carrier.trim and args.carrier_fleet_seed:
                    # start on the fleet's clock, not at 0: the converged trim is the chain's
                    # deterministic frac-N LO offset, common-mode across sats
                    #
                    # ⚠️ THIS LOCAL WAS CALLED `fleet` UNTIL 2026-08-26, AND THAT WAS A LATENT
                    # BUG (found by the refactor's interface analysis, never seen on sky).
                    # `fleet` is the DLL's per-PRN state dict for the whole cycle; assigning a
                    # sorted LIST of carrier trims over it left every later consumer --
                    # including the FLEET-TRIM arming block's `fleet[_p].get("present")` and
                    # the fast-trim PRN set -- indexing a list of floats by PRN. It has never
                    # fired only because --carrier-gain is 0.0 in production, so this whole
                    # block is dead; arming the carrier loop (which is exactly what #52 wants)
                    # would have taken the C++ trim loop down with it, and the traceback would
                    # have pointed at the trim code rather than here.
                    _car_seed_vals = sorted(_carrier.trim.values())
                    if len(_car_seed_vals) >= 3:
                        _carrier.trim[prn] = _car_seed_vals[len(_car_seed_vals) // 2]
                prev_trim = _carrier.trim.get(prn, 0.0)
                trim = (1.0 - args.carrier_leak) * prev_trim + args.carrier_gain * resid
                if tracking and args.carrier_max_step > 0.0:
                    trim = prev_trim + max(-args.carrier_max_step,
                                           min(args.carrier_max_step, trim - prev_trim))
                _carrier.trim[prn] = max(-args.carrier_max_hz, min(args.carrier_max_hz, trim))
                car_report.append("PRN %d resid %+.2f Hz trim %+.2f" % (prn, resid, _carrier.trim[prn]))
                # ---- f_ref TRIM-BLEED SHADOW (log-only, no action) ----
                # This emit is COHERENT and TRACKING (it reached the integrator ungated). If the
                # trim has held a STANDING value across the stability window, f_ref is pinned
                # off-true by ~that trim and a re-pin (f_ref += trim) would clear it. Log the
                # candidate so the trigger can be validated on live data before it is ever armed.
                # Recency-windowed like car_step_hist (a decoherence gap ages the window out ->
                # not "converged"), so no per-gate-branch cleanup is needed.
                if (args.carrier_bleed_shadow or args.carrier_bleed) and tracking and coh_ok:
                    bh = _carrier.bleed_hist.setdefault(prn, [])
                    bh.append((t0, _carrier.trim[prn]))
                    del bh[:-args.carrier_bleed_stable_emits]
                    vals = [v for _, v in bh]
                    # FLAT-TRIM gate (2026-08-03): a truly converged trim is FLAT; a still-settling
                    # one DRIFTS (low spread but a monotonic climb toward a higher plateau). Bleeding
                    # a drifting trim folds a mid-convergence value into f_ref and leaves the
                    # remainder as a residual -> the REFUTED class from the L2C live arm (PRN21 bled
                    # at +5.37 while climbing -> +2.86 resid). Spread alone can't tell drift from
                    # noise; the least-squares SLOPE can (noise averages out, a drift does not).
                    slope = 0.0
                    if len(bh) >= 2:
                        tb = sum(t for t, _ in bh) / len(bh)
                        vb = sum(vals) / len(vals)
                        den = sum((t - tb) ** 2 for t, _ in bh)
                        if den > 0.0:
                            slope = sum((t - tb) * (v - vb) for t, v in bh) / den
                    converged = (len(bh) >= args.carrier_bleed_stable_emits
                                 and t0 - bh[0][0] < 90.0
                                 and abs(_carrier.trim[prn]) >= args.carrier_bleed_hz
                                 and max(vals) - min(vals) <= args.carrier_bleed_stable_hz
                                 and abs(slope) <= args.carrier_bleed_max_slope)
                    # ARMED: re-pin f_ref and zero the trim (one bleed per lockout, never while a
                    # step- or bleed-hypothesis is already under verify for this PRN).
                    if (converged and args.carrier_bleed and prn not in _carrier.verify
                            and prn not in _carrier.bleed_verify
                            and t0 >= _carrier.bleed_lock_t.get(prn, 0.0)):
                        prev_trim = _carrier.trim[prn]
                        _carrier.trim[prn] = 0.0             # f_ref re-pin absorbs the offset
                        _carrier.repin_pending[prn] = prev_trim  # tracker does f_ref += prev_trim
                        _carrier.bleed_verify[prn] = {"emits": 0, "prev_trim": prev_trim, "t": t0}
                        _carrier.bleed_lock_t[prn] = t0 + args.carrier_bleed_lockout_s
                        _carrier.bleed_hist[prn] = []
                        _carrier.bleed_log_t[prn] = t0
                        _log("CARRIER BLEED PRN %d: re-pinning f_ref (%+.2f Hz absorbed, slope "
                             "%+.3f Hz/s), trim->0, VERIFYING (heal in %d emits)"
                             % (prn, prev_trim, slope, args.carrier_bleed_verify_emits))
                    elif converged and t0 - _carrier.bleed_log_t.get(prn, 0.0) >= 60.0:
                        _carrier.bleed_log_t[prn] = t0
                        _log("CAR-BLEED CANDIDATE PRN %d: trim %+.2f Hz stable %d emits "
                             "(spread %.2f, slope %+.3f Hz/s), coherent -> %s"
                             % (prn, _carrier.trim[prn], len(bh), max(vals) - min(vals), slope,
                                "locked out" if args.carrier_bleed
                                else "would re-pin f_ref, predict trim->~0 (shadow, no action)"))
            if car_report:
                _log("CAR: " + "; ".join(car_report))
            for k in list(_carrier.trim):
                if k not in seeds:
                    del _carrier.trim[k]
                    _carrier.locked.discard(k)  # a re-seeded sat re-enters via BOOTSTRAP
                    _carrier.verify.pop(k, None)  # a dropped sat's hypothesis dies with it
                    _carrier.step_hist.pop(k, None)
                    _carrier.fade.pop(k, None)
                    _carrier.bleed_hist.pop(k, None)  # its convergence history dies with it
                    _carrier.bleed_log_t.pop(k, None)
                    _carrier.bleed_verify.pop(k, None)
                    _carrier.bleed_lock_t.pop(k, None)
                    _carrier.repin_pending.pop(k, None)

    def _stage_narrow_search():
        """2b: ALMANAC-NARROW THE SEARCH -- push per-PRN predicted Doppler to the detectors.
        
        Each search then scans a window around the prediction instead of the blind grid, which is what
        makes acquisition affordable. Pure output: it POSTs hints and returns nothing to the cycle.
        
        ⚠️ THE HINT IS ONLY AS GOOD AS THE CLOCK IT CARRIES. When the receiver clock bias is stale the
        margin must widen rather than the hint narrow -- a confidently wrong narrow window is worse
        than no hint at all, because the search then cannot find what it was told to look near."""
        if (args.narrow_search and args.almanac and _ctx.pred) or (_xb_pred and args.xband_seed):
            margin = (args.search_margin_hz
                      if _cb.ema is not None and not _cb.stale
                      else args.search_margin_wide_hz)
            hints = [dict(prn=p, doppler_hz=_ctx.pred[p][0] + _cb.value, margin_hz=margin)
                     for p in sorted(_ctx.pred) if (_ctx.up is None or p in _ctx.up)
                     and (_capable is None or p in _capable)] if (args.almanac and _ctx.pred) else []
            # RESCUE: for a sat the sibling band tracks but BRDC did NOT just hint (no pred /
            # no almanac), add a cross-band hint so the search narrows instead of going blind.
            # Wider margin than a BRDC hint -- the cross-band seed accuracy is the inter-band
            # MAD (~10 Hz) plus this band's own unsolved-LO width -- but far better than the
            # blind grid. Provably rescue-only: a sat BRDC covered is already in `hints`.
            if args.xband_seed and _xb_pred:
                _hinted = {h["prn"] for h in hints}
                _xb_margin = max(margin, args.xband_hint_margin_hz)
                for _p, _xd in sorted(_xb_pred.items()):
                    if _p in _hinted or (_capable is not None and _p not in _capable):
                        continue
                    if _ctx.up is not None and _p not in _ctx.up:
                        continue
                    hints.append(dict(prn=_p, doppler_hz=_xd, margin_hz=_xb_margin))
                    _log_rl("xbandseed-%d" % _p,
                            "XBAND RESCUE HINT PRN %d: %+.0f Hz (sibling tracks it, BRDC does "
                            "not) -> search narrows instead of blind" % (_p, _xd),
                            every_s=30.0)
            # SECONDARY-CODE ALIGNMENT HINT, the Doppler hint's twin and the bigger saving:
            # the acquire builds a FULL surface per alignment, so 20 of them are ~92% of a pass.
            # We echo back the stage's OWN last reported nh with the hop it was measured at, and
            # it propagates by counter arithmetic (one index per code period, an exact hop
            # count). Self-referential on purpose: no GPS time, no range model, nothing to
            # bootstrap -- a PRN with no hint simply scans all 20 as before, and one detection
            # establishes it. Only worth anything because the revisit is now short: over the
            # 1276 s revisit this fleet had before 2026-08-04 the prediction would drift 4.3
            # periods and be worse than useless.
            # OFF BY DEFAULT, and it must stay off until the reported nh is reproducible.
            # Measured 2026-08-04: propagating the search's own reported alignment forward gets
            # 4 of 26 right, whether or not the code phase is carried. PRN 3 reported 11->11
            # over 78.2 s and 11->10 over 77.7 s -- gaps differing by exactly 0 mod 20 periods,
            # so no propagation law fits. Posting a wrong hint NARROWS the scan away from the
            # truth, which is worse than not hinting: self-healing (the TTL restores the full
            # scan) but at the cost of a revisit each time.
            # ALIGNMENT HINT from the EPHEMERIS, not from echoing our own report back.
            #
            # nh is the overlay chip index at TRANSMIT, so BRDC gives it outright:
            #   nh = round((gpst(t) - range/c + clk_sv) / period) mod overlay_len
            # (the convention --nh-assist already uses for the combiner, proven to 0.01 chip).
            # What that leaves is ONE global constant -- the receiver clock reference -- shared
            # by every satellite and measured from any detection at all. Measured on sky
            # 2026-08-04: PRN 3 gave offset 16 on nine consecutive detections, and pooled across
            # six satellites 100% of samples landed within +-2 of 16.
            #
            # Strictly better than echoing our own nh back: no propagation law (mine got 4 of 26
            # right), no dependence on the previous detection, and it works for a satellite
            # NEVER detected -- which is the real prize, since a first acquisition otherwise
            # pays the full 20-way scan.
            #
            # OPEN: the +-2 jitter should not exist. nh is deterministic given ephemeris and
            # time, so a spread means something upstream is not -- likely the same cause that
            # broke propagation (suspect the 16-period acquire window against a 20-chip
            # overlay, 16 !== 0 mod 20). The span absorbs it; it does not explain it.
            nh_hints = []
            if args.nh_hint and _ctx.pred and utc0_sample0:
                try:
                    import gnss_ephemeris as _nh2
                    _per = args.code_length / args.chip_rate_hz
                    def _pred_nh(_p, _t):
                        _v = _ctx.pred[_p]
                        return int(round((_nh2.gpst_of_utc(_t) - _v[3] / _nh2.C_LIGHT
                                          + (_v[4] if len(_v) > 4 else 0.0)) / _per)) % args.nh_overlay_len
                    # (a) re-measure the constant from every fresh detection we have
                    #
                    # ⚠️ "FRESH" MEANS A NEW DETECTION, NOT A NEW CYCLE. This loop used to
                    # append every nh_seen entry every cycle, so an unchanged detection was
                    # re-appended every --interval seconds and the 64-sample window filled
                    # with the SAME value repeated. nh_hint_min_samples then read that as 64
                    # independent confirmations of an offset that might rest on one stale
                    # detection -- a sample count that measures uptime rather than evidence.
                    # ref_hop identifies the detection, so admit a PRN only when its ref_hop
                    # has actually moved.
                    #
                    # AND THE HISTORY AGES. The comment at nh_seen's refresh says a wrong hint
                    # is "recoverable -- the hint expires and the full scan returns", but
                    # nothing implemented that expiry: nh_offset kept its last value forever
                    # and the ±2-of-20 narrowing kept being pushed. That closes a loop with no
                    # restoring force -- bad clock -> bad hint -> search narrowed onto the
                    # wrong overlay phase -> no detections -> the clock cannot be re-solved --
                    # and the only escape is the code clock random-walking back onto truth by
                    # chance. Observed 2026-08-10 17:06-17:21: the clock wandered
                    # 6996 -> 5095 -> 1797 -> 1555 -> 9210 -> 134 chips and only then snapped
                    # to 150.8 and locked, ~15 min of a self-reinforcing outage that read as a
                    # frontend sensitivity loss (docs 11.33).
                    for _p, (_nh, _rh) in _nho.seen.items():
                        if _p not in _ctx.pred or _nho.last_rh.get(_p) == _rh:
                            continue
                        _nho.last_rh[_p] = _rh
                        _t = utc0_sample0 + _rh / args.hops_per_sec
                        _nho.off_hist.append((t0, (_pred_nh(_p, _t) - _nh) % args.nh_overlay_len))
                    del _nho.off_hist[:-64]
                    _fresh = [o for (_ts, o) in _nho.off_hist
                              if t0 - _ts <= args.nh_hint_max_age_s]
                    if len(_fresh) < args.nh_hint_min_samples and _nho.offset[0] is not None:
                        _log("nh hint EXPIRED: %d sample(s) inside %.0f s (need %d) -- "
                             "dropping the offset so the search widens instead of staying "
                             "narrowed on a stale one"
                             % (len(_fresh), args.nh_hint_max_age_s, args.nh_hint_min_samples))
                        _nho.offset[0] = None
                    # (b) circular median: the offsets cluster, so rotate to the mode before
                    # taking it, or a cluster straddling the 0/20 wrap averages to nonsense.
                    if len(_fresh) >= args.nh_hint_min_samples:
                        _mode = max(set(_fresh), key=_fresh.count)
                        _rot = [((o - _mode + args.nh_overlay_len // 2) % args.nh_overlay_len)
                                - args.nh_overlay_len // 2 for o in _fresh]
                        _rot.sort()
                        _nho.offset[0] = (_mode + _rot[len(_rot) // 2]) % args.nh_overlay_len
                    # (c) hint EVERY visible sat, detected or not, at a hop the stage can
                    # propagate over a few seconds rather than a minute
                    if _nho.offset[0] is not None:
                        _rh_now = int(round((t0 - utc0_sample0) * args.hops_per_sec))
                        nh_hints = [dict(prn=int(_p),
                                         nh=(_pred_nh(_p, t0) - _nho.offset[0]) % args.nh_overlay_len,
                                         ref_hop=_rh_now)
                                    for _p in _ctx.pred if _ctx.pred[_p][2] >= args.mask_deg]
                        _log_rl("nhhint", "nh hint: offset %d (%d samples) -> %d sat(s), span %d"
                                % (_nho.offset[0], len(_fresh), len(nh_hints),
                                   args.nh_hint_span), every_s=60.0)
                except Exception as e:
                    _log_rl("nhhint-err", "nh hint failed: %s" % e, every_s=60.0)
            pushed = 0
            for d_ep in detectors:
                try:
                    _post("%s/set_doppler_hints" % d_ep, hints)
                    pushed += 1
                except Exception as e:
                    _log("set_doppler_hints %s failed: %s" % (d_ep, e))
                # SEPARATE try: a detector running a binary without /set_nh_hint 404s here, and
                # sharing the block above would make that failure look like the DOPPLER hint
                # failing -- silently un-narrowing the search on every cycle during a rolling
                # upgrade. Rate-limited because a stale binary fails every single cycle.
                if nh_hints:
                    try:
                        _post("%s/set_nh_hint" % d_ep, nh_hints)
                    except Exception as e:
                        _log_rl("nhpost-%s" % d_ep,
                                "set_nh_hint %s failed (old binary?): %s" % (d_ep, e),
                                every_s=120.0)
            _log_rl("narrow",
                    "narrowed search: %d hints +-%d Hz (%s) -> %d/%d detectors"
                    % (len(hints), int(margin),
                       ("bias solved" if _cb.ema is not None and not _cb.stale
                        else "bias STALE, wide re-solve" if _cb.ema is not None
                        else "pre-solve wide"),
                       pushed, len(detectors)))

    def _stage_detections_to_seeds():
        """1: DETECTIONS -> SEEDS. Turn this cycle's best-SNR detection per PRN into a tracker seed.
        
        The longest single stage in the cycle, and the one every other stage depends on: it converts
        the search's (snr, doppler, code phase, ref_hop) into the seed triple the trackers consume,
        via cp_to_seed_currency.
        
        ⚠️ AN ALIAS-BIN DETECTION IS HARMLESS -- DO NOT "FIX" IT. The search's Doppler is ambiguous
        mod 1/(2*t_rec) (25 Hz on 20 ms records). The search back-projects cp0 to sample 0 with the
        SAME reported dop, and cp_to_seed_currency adds that projection back with the same numbers, so
        the round trip is exact whatever bin the Doppler rode. A "fold" that replaced dop before the
        currency conversion broke exactly that cancellation, by K*t_abs*k*q -- it was live for part of
        a day in 2026-07-20 and is the reason this comment is here."""
        for prn, (snr, dop, cp, ref_hop, det_nh, cp_long, cp_at_ref) in best.items():
            # DETECTION ALIAS CENSUS (2026-07-20; was briefly a FOLD, corrected same day):
            # the search's Doppler estimate is ambiguous mod 1/(2*t_rec) -- 25 Hz on L2C's
            # 20 ms records, 50 Hz B1C. An alias-bin detection is HARMLESS to the cp
            # bookkeeping: the search back-projects cp0 to sample 0 with the SAME reported
            # dop (GnssChannelizedSearch: det.doppler_hz and the drift term share one
            # variable), and cp_to_seed_currency adds that projection back with the same
            # numbers -- the round trip is exact whatever bin the dop rode. The v1 fold
            # REPLACED dop before the currency conversion, breaking exactly that
            # cancellation by K*t_abs*k*q: the TRACK-vs-MODEL monitor caught held-sat
            # candidates 12-57 chips off their healthy tracks within the hour (L2C
            # 18/23/32, 15:20) -- and a candidate that wrong silently DISABLES the escape
            # referee (sign-flipping cp_err never sustains 5 consecutive). So: measure,
            # never modify. The census still maps which chains/sats ride alias bins (the
            # B1C zombie-birth investigation continues on that data).
            if (args.det_alias_fold and args.almanac and prn in _ctx.pred
                    and _cb.ema is not None and not _cb.stale):
                _aref = _ctx.pred[prn][0] + _cb.value
                _k = round((dop - _aref) / Q_ALIAS_HZ)
                if _k != 0 and abs(dop - _aref) < 3.5 * Q_ALIAS_HZ:
                    _log_rl("afold-%d" % prn,
                            "ALIAS BIN PRN %d: det dop %+.1f = model %+.1f %+d bin(s) of "
                            "%.0f Hz (census only; cp round-trip is exact)"
                            % (prn, dop, _aref, _k, Q_ALIAS_HZ),
                            every_s=30.0)
            v_dr = dr_pd.get((args.dr_constellation, prn)) if dr_pd else None
            if _ctx.up is not None and prn not in _ctx.up:
                # accept the detection anyway if BRDC says it's up: the TLE up-set
                # mismaps some BDS birds (PRN 39: TLE el<5 vs BRDC el 10)
                if v_dr is None or v_dr["el"] < args.mask_deg:
                    continue
            # #79: THE SEARCH IS THE ADMISSION AUTHORITY for the trim presence gate. Stamped
            # here -- after the visibility filter, so a spurious below-horizon detection
            # cannot arm a correction, and before every seeding-policy filter below, because
            # eligibility asks "is this satellite up and detectable", not "did this cycle
            # like the detection well enough to re-seed from it".
            if args.dll_deep_gate_from_search > 0.0 and snr >= args.dll_deep_gate_from_search:
                _dls.deep_gate_seen[prn] = t0
            _dop_src = "pred" if (args.almanac and prn in _ctx.pred) else "DET(grid)"
            seed_dop = (_ctx.pred[prn][0] + _cb.value) if (args.almanac and prn in _ctx.pred) else dop
            # Dead-reckon armed: prefer the BRDC doppler for EVERY seed -- the same model
            # that owns the undetected sats. Mixing sources stepped the seed doppler by
            # the TLE-vs-BRDC error at every DR<->search handoff (~25 Hz on a stale TLE
            # = the whole E1 hold fence; observed E32 RELEASE ddop -25, 2026-07-13).
            # THE DOPPLER AND THE CODE PHASE DO NOT HAVE TO SHARE A TRUST DECISION.
            # dr_untrusted is set for two unlike reasons: a stale ephemeris (the orbit
            # itself is doubtful -- true for range AND range-rate) or a code-phase
            # integrity residual over --dr-max-integrity-chips (1.0 chip ~ 30 m, which is
            # ordinary iono + b_sat and says nothing about the range rate). Under
            # --dr-doppler-ignores-integrity only the first demotes the Doppler, so a
            # satellite flipping trust no longer switches its seed between two BRDC
            # evaluations -- and the seed IS the replica's carrier phase.
            _unt = dr_untrusted.get(prn)
            _dop_trusted = (_unt is None
                            or (args.dr_doppler_ignores_integrity
                                and not str(_unt).startswith("ephemeris")))
            if v_dr is not None and _dop_trusted:
                _dop_src = "dr" if _unt is None else "dr(code-untrusted)"
                seed_dop = (args.doppler_sign * (-v_dr["range_rate_mps"] / C_LIGHT
                                                 * args.carrier_hz) + _cb.value)
                if _unt is not None:
                    _log_rl("dopkeep-%d" % prn,
                            "PRN %d: code model untrusted (%s) but KEEPING the BRDC Doppler "
                            "-- an integrity residual is a code-phase statement, and "
                            "switching the seed switches the replica's carrier phase"
                            % (prn, _unt), every_s=120.0)
            # ...unless the measured Doppler is explicitly preferred (--seed-doppler det). Last
            # word, so it overrides the model AND the DR: those two exist to keep the seed
            # smooth and to own undetected sats, but neither helps a sat we HAVE measured, and
            # a model error rides the seed's whole extrapolation age at 0.0087 chips/Hz/s.
            if args.seed_doppler == "det":
                _dop_src = "det"
                seed_dop = dop
            # SEED-STEP ATTRIBUTION (2026-07-18, the one-grid-step NCO disease): any seed
            # doppler step > 10 Hz vs the sat's previous seed is loud, with its source --
            # a ~exact-doppler_step jump here is the smoking gun for a grid/quantization
            # slip upstream (the hint-anchored search grid was one such; fixed same day).
            _prev_sd = seeds.get(prn, {}).get("doppler_hz")
            if _prev_sd is None and args.almanac and not _cb.available:
                # S2d: the condition is now "no bias FROM ANY SOURCE", not "this chain has
                # not solved its own". Same guard, wider supply. It is the exact deadlock
                # that cost 2h45m on 2026-07-27: L5 GPS could measure one satellite, never
                # reached --bias-min-sats, so this line withheld every first seed -- while
                # its two siblings sat in the same band holding the right answer (+32 Hz)
                # and its OWN l-a fit held a better one still. With the fused state that
                # chain is `bias_available` from cycle 1 and never enters this branch.
                #
                # BIAS-UNSOLVED FIRST-SEED GUARD (2026-07-18): a first seed sent before the
                # clock-freq bias solves carries the FULL dongle offset (~150 Hz at L1,
                # measured: PRN 5 first seed 1146.5 vs det 974.1 at bias +0.0). A strong sat
                # can lock and enter hold-freeze on that wrong currency within seconds --
                # the startup race behind the stochastic diseased-node births (rails at
                # first seeding, fleet-dependent). Detections don't need seeds, so holding
                # off costs ~2 s of acquisition and delays nothing else: the bias solves
                # from the same detections this cycle already collected.
                continue
            if _prev_sd is None:
                # FIRST seed: full attribution (the rail onsets coincide with first seeding,
                # and a first seed has no previous value for the step tripwire to fire on).
                _log("PRN %d FIRST SEED dop %.1f (src=%s, det=%.1f, pred=%s, bias %+.1f, trim %+.1f)"
                     % (prn, seed_dop, _dop_src, dop,
                        ("%.1f" % (_ctx.pred[prn][0])) if (args.almanac and prn in _ctx.pred) else "n/a",
                        _cb.value, _carrier.trim.get(prn, 0.0)))
            elif abs(seed_dop - _prev_sd) > 10.0 and prn in cp_held:
                # HELD sat: the candidate walks while the emitted tuple stays frozen /
                # translated -- this "step" is never applied as-is, and logging it every
                # cycle flooded the logs (195k lines on L1 GPS in 3 h, 2026-07-18). The
                # tripwire's purpose (grid/quantization slips on LIVE seeds) is served by
                # the un-held branch below.
                pass
            elif abs(seed_dop - _prev_sd) > 10.0:
                _log("PRN %d SEED DOP STEP %+.1f Hz (%.1f -> %.1f, src=%s, det=%.1f)"
                     % (prn, seed_dop - _prev_sd, _prev_sd, seed_dop, _dop_src, dop))

            # Maintain a per-PRN cp0-vs-hop history (only distinct snapshots; the search
            # holds its detection between updates) and fit the first-order code drift.
            # DOPPLER-RATE HISTORY. The seed's doppler_rate_hz_s drives BOTH the tracker's
            # carrier NCO (out.doppler_hz = doppler + dop_rate*dt) and the quadratic CODE term
            # (quad = 0.5*(chip/f_c)*dop_rate*dt^2), so its error shows up twice. BRDC computes
            # it by range-rate differencing over a 4 s epoch pair -- a numerical derivative, and
            # we use it at the worst moment, near zenith where the curvature peaks. Measured
            # 2026-08-04: BRDC gave PRN 3 -0.4699 Hz/s where the Doppler track said -0.578, and
            # that 0.108 Hz/s residual is ~0.7 rad of phase curvature inside the 1.05 s deep
            # window -- a direct contributor to the 2.55 rad that costs the coherent sum 29 dB.
            #
            # We measure the Doppler every pass, so fit its slope instead. Well-conditioned at
            # an 11 s revisit: four points span ~44 s over which the Doppler moves ~25 Hz
            # against ~1.5 Hz per-detection noise. Same gap rule as the cp history -- a gap
            # means re-acquisition and the old slope is stale.
            dh_ = _cpt.dop_hist.get(prn, [])
            if dh_ and (ref_hop - dh_[-1][0]) > MAX_GAP_HOPS:
                dh_ = []
            if not dh_ or ref_hop != dh_[-1][0]:
                dh_.append((ref_hop, dop))
                dh_ = dh_[-HIST_LEN:]
            _cpt.dop_hist[prn] = dh_

            h = _cpt.hist.get(prn, [])
            if h and (ref_hop - h[-1][0]) > MAX_GAP_HOPS:
                h = []  # gap too large -> re-acquisition, old slope is stale
            # SNR gate (2026-08-02). The slope this fit is trying to resolve is ~0.0148
            # chips/s -- the drift from a ~1.7 Hz Doppler error. A detection near the
            # acquire threshold has a phase that is simply noise (measured on CHORD: below
            # snr ~60 the within-period residual runs ~2000 chips against a few chips above
            # it), so ONE such point does not degrade the fit, it destroys it. Default 0
            # keeps every point, which is the prototype's behaviour and right there: its
            # detections sit far above threshold and its revisit is seconds.
            if not h or ref_hop != h[-1][0]:
                if snr >= args.fit_min_snr:
                    h.append((ref_hop, cp, dop))
                    h = h[-HIST_LEN:]
                elif h:
                    _log_rl("fitsnr-%d" % prn,
                            "PRN %d cp-fit: skipping snr %.0f point (< --fit-min-snr %.0f); "
                            "%d in history" % (prn, snr, args.fit_min_snr, len(h)),
                            every_s=120.0)
            _cpt.hist[prn] = h

            # The bare-detection cp is in the DETECTION's Doppler currency; the tracker will
            # despread at seed_dop. Convert (else a sat first acquired at t_abs inherits a
            # t_abs*f_chip*(dop-seed_dop)/f_c offset -- chips off-peak for any mid-run
            # acquisition, before tracking even starts).
            ((_, cp_seed_cur),) = cp_to_seed_currency([(ref_hop, cp, dop)], seed_dop)
            seed = Seed.born("det", epoch=ref_hop,
                             doppler_hz=seed_dop, code_phase_chips=cp_seed_cur,
                             code_phase_rate=0.0, ref_hop=ref_hop)
            # The doppler SOURCE was arbitrated above (_dop_src: pred | dr | det), BEFORE
            # the tuple existed, so the constructor's blanket "det" under-describes the
            # one field three estimators fight over. Re-attribute -- same value, provenance
            # only: the audit trail then separates a model-seeded doppler from a measured
            # one, the very distinction Phase 3's per-PRN primacy gate decides on.
            seed.put("dop_sel:" + _dop_src, epoch=ref_hop, doppler_hz=seed_dop)
            # 2nd-order carrier feed-forward: hand the tracker the almanac Doppler RATE (Hz/s, sign-
            # applied like doppler_hz); the tracker integrates it in its NCO (never a replica
            # retune -- that walks the absolutely-anchored code/carrier off-peak) so the deep-
            # integration residual stays flat even at zenith (max Doppler acceleration).
            v2_dr = dr_pd2.get((args.dr_constellation, prn)) if dr_pd2 else None
            v0_dr = dr_pd0.get((args.dr_constellation, prn)) if dr_pd0 else None
            if v2_dr is not None and v0_dr is not None:
                # BRDC doppler rate, CENTRAL difference over the +/-2 s pair straddling now_w
                # (task #52). Centred, so the rate is tagged at now_w rather than 2 s late.
                seed.put("dop_model", epoch=ref_hop,
                         doppler_rate_hz_s=(args.doppler_sign
                                            * (-(v2_dr["range_rate_mps"]
                                                 - v0_dr["range_rate_mps"]) / 4.0)
                                            / C_LIGHT * args.carrier_hz))
            elif v_dr is not None and v2_dr is not None:
                # Fallback for the first cycle, before pd0 exists: the OLD forward form, and
                # it is deliberately still here rather than silently emitting nothing -- but it
                # is 2 s mis-tagged, so it must not be the steady state.
                seed.put("dop_model", epoch=ref_hop,
                         doppler_rate_hz_s=(args.doppler_sign
                                            * (-(v2_dr["range_rate_mps"]
                                                 - v_dr["range_rate_mps"]) / 2.0)
                                            / C_LIGHT * args.carrier_hz))
            elif args.almanac and prn in _ctx.pred:
                seed.put("dop_model", epoch=ref_hop, doppler_rate_hz_s=_ctx.pred[prn][1])
            # MEASURED rate beats the model's, and it is the LAST word here for the same reason
            # --seed-doppler det is: the model exists to own sats we have not measured. Gated on
            # enough points over enough baseline that the slope is real rather than fitted to
            # detection noise.
            # CROSS-CHECK THE FIT AGAINST THE MODEL BEFORE ADOPTING IT. Measured on sky
            # 2026-08-05, seeded rate vs the Doppler actually observed over 140 s: PRN 8 seeded
            # 2.18x too small (-0.195 against -0.424), PRN 30 seeded the WRONG SIGN (+0.205
            # against -0.155), and PRNs 5/7/16 seeded None while drifting at -0.15..-0.45 Hz/s.
            # The rate is fitted from the MEASURED Doppler, which carries ~6 Hz of per-pass
            # search scatter (8.20.5), and it was the last word unconditionally -- a noisy
            # estimator overriding a model one, the same trade --seed-doppler det got wrong.
            # dop_rate_max only bounds the MAGNITUDE, so a wrong-signed or half-size fit inside
            # +-0.8 sails through. Its error costs twice: the carrier NCO extrapolation AND the
            # quadratic code term both use it.
            _model_dr = seed.get("doppler_rate_hz_s")
            _dr = fit_dop_rate(_cpt.dop_hist.get(prn, []), args.hops_per_sec,
                               args.dop_rate_min_pts, args.dop_rate_min_span_s,
                               args.dop_rate_max)
            if (_dr is not None and _model_dr is not None and args.dop_rate_model_tol > 0.0
                    and abs(_dr - _model_dr) > args.dop_rate_model_tol):
                # The two disagree by more than the model's own accuracy: trust the MODEL, which
                # comes from an orbit rather than from detection noise, and say so.
                dop_rate_rejected[prn] = (_dr, _model_dr)
            elif _dr is not None:
                seed.put("dop_fit", epoch=ref_hop, doppler_rate_hz_s=_dr)
                dop_rate_fitted[prn] = _dr
            elif args.force_doppler_rate is not None:
                # Replay-bench override: a recorded capture's sky is at another epoch (no almanac),
                # so inject a known rate into every seed to exercise the NCO feed-forward offline.
                seed.put("dop_force", epoch=ref_hop,
                         doppler_rate_hz_s=args.force_doppler_rate)
            fit = fit_cp_rate(
                cp_to_seed_currency(h, seed_dop,
                                    float(seed.get("doppler_rate_hz_s", 0.0) or 0.0)),
                CODE_LEN)
            if fit is not None:
                rate, h0, cp_ref = fit
                seed.put("cp_fit", epoch=h0,
                         code_phase_rate=rate, ref_hop=h0, code_phase_chips=cp_ref)
                fitted.add(prn)
                _cpt.fit_slope[prn] = rate * args.hops_per_sec   # chips/s, for CARRIER-FROM-CODE
                # This fit contributes an (l-a) sample: its code_frac minus the sat's carrier_frac.
                # Only strong, geometry-clean detections (SNR gate) -- weak/noisy slopes would bias it.
                la = code_clock_bias_sample(rate, seed_dop, args.hops_per_sec,
                                            args.chip_rate_hz, args.carrier_hz)
                # PER-SAMPLE gate: a single noisy/unwrap-blown slope fit is a large l-a outlier
                # that the few-sat median can't reject -- and a wandering pooled l-a swings the
                # seeded code rate (+-1 ppm = +-1 chip/s), walking the deep integration off-peak
                # within its ~1 s window (the 2026-07-07 L1 deep decay). Bound to --code-bias-max.
                if snr >= args.acquire_snr and abs(la) < args.code_bias_max * 1e-6:
                    la_samples.append(la)
                _log_rl("cpfit-%d" % prn,
                        "PRN %d cp-fit: %.2f chips @ hop %d, slope %+.3f chips/s "
                        "(%d pts, l-a %+.3f ppm)"
                        % (prn, cp_ref, h0, rate * args.hops_per_sec, len(h), la * 1e6))
            # L2C CL TIME-ASSIST: the trackers despread the CL pilot, whose absolute code phase
            # is cp_CL = cp_CM + k*10230 with k the CL segment index -- COMPUTED, not searched.
            # CL's 1.5 s epoch is locked to GPS time in Z-count (1.5 s) units, and GPS-UTC (18 s)
            # and the GPS epoch (unix 315964800) are both exact multiples of 1.5 s, so unix time
            # mod 1.5 IS GPS time mod 1.5 (--cl-time-adjust is the escape hatch if that ever
            # changes). cp is absolute-anchored at capture sample 0, so evaluate the TRANSMIT
            # time of sample 0 (utc0_sample0 - range/c) and SNAP the predicted CL chip count to
            # the measured cp_CM (mod 10230): the code phase supplies the fine time; the wall
            # clock only picks the segment (needs ~10 ms absolute accuracy, and the logged
            # residual MEASURES the anchor+host-clock error). Second-order cp0/range drift keeps
            # this valid for ~hour-long runs.
            # MEASURED overlay alignment (search reports `nh`) beats the time-assist below.
            # Both cp0 and the seed are arguments in the same C = arg + n*cps convention, so
            # matching the primary phase AND the overlay index gives simply
            #     cp_long = cp_primary + nh * CODE_LEN   (mod LC_SEG*CODE_LEN)
            # -- no wall clock, no range model, no half-period accuracy requirement, and
            # per-satellite rather than one fitted constant for the whole sky.
            if cp_long >= 0.0 and LC_SEG > 1:
                # The SEARCH already reduced this at the overlaid code's own length, so it
                # carries the period. Reconstructing it here -- from `nh`, or from absolute
                # time via --cl-assist -- means re-deriving a convention the search already
                # knows, which is where every previous attempt went wrong.
                seed.put("nh_lift", epoch=ref_hop,
                         code_phase_chips=((cp_long + args.nh_period_offset * CODE_LEN)
                                           % (LC_SEG * CODE_LEN)))
                cl_report.append("PRN %d long-cp (search)" % prn)
                # And carry the PHASE at the search's own epoch. cp0 back-references to sample
                # 0 through a Doppler-scaled rate, which multiplies the reported Doppler's
                # error by ~5900 chips/Hz = 0.58 overlay PERIODS per Hz -- so the period that
                # survives that route is noise. A phase at its own epoch has no such lever.
                if cp_at_ref >= 0.0:
                    LLc = LC_SEG * CODE_LEN
                    ph = cp_at_ref % LLc
                    # PERIOD CONTINUITY IS A CHECK, NOT AN AUTHORITY (2026-08-02).
                    #
                    # This existed because the search could not report the overlay period: it
                    # lifted by best_nh alone and dropped the coarse lag's whole-period count,
                    # so the reported period was wrong ~half the time. Since 4371ff4eb the
                    # search MEASURES it (from AcquisitionResult::peak_tau_samples), verified
                    # 16/16 on injection and 9/9 on sky above snr 60. The source is now right,
                    # so overriding it from history is strictly a way to be wrong.
                    #
                    # And this loop could only ever be wrong in one direction. It stored its
                    # OWN correction in ph_hist and predicted the next pass from that, so a
                    # single bad correction was permanent: every later measurement got snapped
                    # into agreement with the original error. Self-consistent, absolutely
                    # wrong -- and with no residual gate, a marginal detection whose phase is
                    # noise still yielded a confident-looking integer m. Measured 2026-08-02:
                    # PRN 21 was period-consistent at the source across three consecutive
                    # detections and collected four "corrections" anyway.
                    #
                    # So: compute it, log the disagreement, apply NOTHING, and store the
                    # MEASURED phase. That turns the resolver into a regression detector for
                    # the search's period -- a nonzero m on a STRONG satellite now means the
                    # source broke, which is worth an alarm rather than a silent repair.
                    # --period-continuity correct restores the old override if ever needed.
                    prev = _cpt.ph_hist.get(prn)
                    if prev is not None and args.period_continuity != "off":
                        h0, ph0, dop0 = prev
                        dh = ref_hop - h0
                        gap_s = dh / args.hops_per_sec
                        if 0 < gap_s <= 900.0:
                            rate = (args.chip_rate_hz / args.hops_per_sec
                                    * (1.0 + args.code_doppler_sign * 0.5 * (dop0 + dop)
                                       / args.carrier_hz))
                            ph_pred = (ph0 + dh * rate) % LLc  # NB not `pred` -- that is the
                            # almanac prediction dict in this scope, and shadowing it breaks
                            # the alias census a hundred lines down with a TypeError.
                            m = int(round(((ph_pred - ph) % LLc) / CODE_LEN)) % LC_SEG
                            if m:
                                resid = ((ph + m * CODE_LEN - ph_pred + LLc / 2) % LLc) - LLc / 2
                                # Only a STRONG disagreement is evidence about the source; a
                                # marginal detection disagreeing tells us about the detection.
                                sev = ("SOURCE PERIOD DISAGREES"
                                       if snr >= args.period_check_snr else "weak det")
                                _log_rl("phcont-%d" % prn,
                                        "PRN %d period continuity %s: %+d periods "
                                        "(snr %.0f, gap %.0f s, residual %+.1f chips) "
                                        "-- NOT applied (%s)"
                                        % (prn, sev, m, snr, gap_s, resid,
                                           args.period_continuity),
                                        every_s=60.0)
                            if args.period_continuity == "correct":
                                ph = (ph + m * CODE_LEN) % LLc
                    # Feed history only from detections whose phase means something. Below the
                    # bar the phase is noise (measured: snr < 60 gives ~2000-chip within-period
                    # residuals against a few chips above it), and a noise entry poisons every
                    # comparison until that PRN is seen again -- 90-270 s at CHORD's revisit.
                    if snr >= args.period_check_snr or prn not in _cpt.ph_hist:
                        _cpt.ph_hist[prn] = (ref_hop, ph, dop)
                    # --nh-period-offset: applied HERE, after the continuity check has had its
                    # say, and to the phase rather than the argument -- propagate_seed prefers
                    # phase_ref_chips whenever it is >= 0, so offsetting only code_phase_chips
                    # would change nothing the tracker ever reads. ph_hist keeps the UNSHIFTED
                    # phase so the continuity check still compares like with like.
                    ph = (ph + args.nh_period_offset * CODE_LEN) % LLc
                    seed.put("nh_lift", epoch=ref_hop, code_phase_at_ref_chips=ph)
            elif det_nh >= 0 and LC_SEG > 1:
                seed.put("nh_lift", epoch=ref_hop,
                         code_phase_chips=((seed["code_phase_chips"] % CODE_LEN)
                                           + (det_nh % LC_SEG) * CODE_LEN)
                                          % (LC_SEG * CODE_LEN))
                cl_report.append("PRN %d nh=%d (measured)" % (prn, det_nh))
            elif args.cl_assist and utc0_sample0 and args.almanac and prn in _ctx.pred:
                tau = _ctx.pred[prn][3] / C_LIGHT
                cl_chips = (((utc0_sample0 - tau + args.cl_time_adjust) % LC_EPOCH)
                            * args.chip_rate_hz)
                cp_cm = seed["code_phase_chips"]
                k = int(round((cl_chips - cp_cm) / CODE_LEN))
                fine_ms = (cl_chips - cp_cm - k * CODE_LEN) / args.chip_rate_hz * 1e3
                seed.put("cl_assist", epoch=ref_hop,
                         code_phase_chips=(cp_cm + (k % LC_SEG) * CODE_LEN)
                                          % (LC_SEG * CODE_LEN))
                cl_report.append("PRN %d k=%d fine %+.1f ms" % (prn, k % LC_SEG, fine_ms))
            # HOLD-ON-LOCK: once a PRN shows a real lock, FREEZE its cp anchor + rate and let the
            # DLL trim own the sub-chip residual. The search's per-fix cp is only good to ~1-2
            # chips (hop-resolution coarse + refine), so re-anchoring from the fit at every cycle
            # injects that jitter into the despread 5x per second -- the replay signature: the
            # incoherent amplitude flaps full-scale<->zero on seconds timescales and the deep
            # never builds past the lucky windows (measured 2026-07-11, 07-04 capture). Classic
            # acquisition->track handoff: the fit is for CAPTURE, the (frozen anchor + DLL) for
            # TRACKING. Doppler keeps refreshing (carrier fence semantics unchanged); on lock
            # loss the coast/drop path clears the seed and re-acquisition seeds fresh from the fit.
            # The seed's (cp0, rate, ref_hop, doppler) form the tracker's code CURRENCY: cp0 is
            # an absolute-sample-0 phase whose meaning shifts by t_abs*f_chip/f_c chips PER HZ
            # of doppler change. The currency must therefore be PIECEWISE-CONSTANT while
            # locked -- a merely-smooth doppler (detection grid jitter, clock-bias EMA wobble)
            # multiplies t_abs and walks/jitters the despread off-peak (blind replay: +-12 Hz
            # grid jitter -> +-0.5 chip at t=60 s; live soak: 0.2 Hz EMA wobble -> 0.5 chip at
            # t=1 h). Freeze the WHOLE tuple; release on amp collapse or when the frozen
            # doppler goes stale enough to decohere the 1-record despread itself.
            prev = seeds.get(prn)
            # TRACK-vs-SEARCH consistency (the hold's independent referee): the freeze
            # trusts the DLL to own the sub-chip residual, but a sharp-ACF (BOC)
            # discriminator has a FINITE capture range (~1/3 chip at 0.5-eff-chip
            # spacing) with further STABLE equilibria beyond it (prompt R ~ -0.25,
            # -12 dB). An excursion past capture (a jittery era, one bad re-anchor)
            # leaves the DLL contentedly servoing the WRONG lobe: deep stays coherent
            # (and floor-certified), amp bleeds ~12 dB, and the hold never lets go --
            # observed 2026-07-12 pm: C34 amp 440->0 over 35 min at deep 100+, the
            # track parked ~0.75 chips off while the search kept finding the true
            # peak at >100 sigma. The search FIT (sub-0.1-chip stable at 20 MSPS) is
            # the referee: persistent disagreement beyond the capture range releases
            # the freeze so the seed re-anchors on the true peak.
            cp_err = None
            # #76/#83 2(d): the EFFECTIVE trim -- what the tracker is actually adding to
            # the model phase right now: the Python slow integrator's value PLUS the C++
            # fleet loop's standing trim, read back last cycle (#76; ~one cycle old, since
            # the readback rides the policy block which runs after this loop). The handover
            # (180429f07) means at most one arm is normally nonzero per PRN, but the sum is
            # correct in every regime, including the release ramp when both exist. Until
            # #76 every consumer here used dll_trim alone: an armed PRN was judged as if
            # untrimmed while up to 3 chips of command stood at the trackers.
            _trim_eff = (_dls.trim.get(prn, 0.0)
                         + ((_dls.readback.get(prn) or {}).get("trim_chips", 0.0)))
            # THE DETECTION'S OWN PHYSICAL PHASE at its epoch: cp0 and dop were published
            # together, so undoing the pair reintroduces no translation -- which is the
            # entire #42 fix. (cp_at_ref would be better conditioned but lives in the C++
            # last-sample convention and carries the anchor Doppler term; see
            # track_vs_fit_chips.) Hoisted out of the hold branch: the innovation below
            # wants it for every detection.
            _cpe_recon = (cp + (ref_hop / args.hops_per_sec) * args.chip_rate_hz
                          * (1.0 + args.code_doppler_sign * dop / args.carrier_hz)
                          ) % CODE_LEN
            # ── #83 2(d): THE INNOVATION, computed for EVERY accepted detection ──
            # Measurement minus forecast (chips, wrapped to one period), evaluated BEFORE
            # this cycle overwrites the seed with the very detection being judged. SERVED
            # (log + publisher rows) and, when the PRN is held, it IS the escape referee's
            # statistic below -- one number, two consumers, so the referee and the
            # published innovation can never disagree. This is also the number Phase 3's
            # per-PRN model-primacy gate reads (p95 |innov| < ~1 chip over 10 min) and the
            # one-controller carrier design's code-side measurement.
            _inv = None
            # ⚠️ DR-OWNED SEEDS ARE EXCLUDED (2026-08-17 13:20, found by the flip's first
            # arm): their ref_hop is stamped from WALL time (int(t_now_abs*hops_per_sec))
            # while a detection's ref_hop rides the F-ENGINE hop counter -- a cross-axis
            # dh, so the "innovation" reads the wall-vs-F-engine offset's sub-second part
            # times the chip rate (ms -> thousands of chips; integer seconds vanish, 1 s =
            # 100.0 periods exactly). Measured on flipped G26: INNOV p95 2598 while MINNOV
            # p95 1.27, q 3.12 and trim -1.4 all said the tap was fine. The same axis
            # mismatch is the leading suspect for the 2(b) DR-chain audit steps. MINNOV is
            # the dr-owned satellites' referee; INNOV resumes when the search re-anchors.
            # Under --dr-fengine-axis the dr stamps ride the F-engine axis and
            # --innov-dr-seeds re-admits them (both flags, or the exclusion stands).
            if (prev is not None
                    and ((args.innov_dr_seeds and args.dr_fengine_axis)
                         or dr_state is None or prn not in dr_state["seeded"])
                    and all(k in prev for k in ("code_phase_chips", "code_phase_rate",
                                                "ref_hop", "doppler_hz"))):
                # FORECAST WHAT THE TRACKER RUNS, not the cp0 fiction. The first deploy of
                # this block used track_vs_fit_chips (= dr_seed_phys, the cp0-argument
                # path): every PRN read thousands of chips, sign-flipping, p95 ~5000 --
                # wrap-uniform, two uncorrelated phases -- while q and the trims said the
                # taps were fine. The tracker prefers code_phase_at_ref_chips
                # (propagate_seed), and seed writers legitimately move doppler_hz without
                # re-projecting cp0, so the cp0-implied phase swings by the t_abs lever
                # (~5600 chips/Hz). tracker_phase_at picks the same reference
                # propagate_seed does (#45 step 7 -- same lesson as #43's 90,000-chip
                # fiction). The measurement moves to the same LAST-SAMPLE convention with
                # its OWN doppler (the hop-epoch convention: 52.37 chips if mixed).
                _fc = tracker_phase_at(prev, ref_hop, args.hops_per_sec,
                                       args.chip_rate_hz, args.carrier_hz,
                                       args.code_doppler_sign, CODE_LEN,
                                       args.search_fft_len or None)
                _hop_off_det = (args.chip_rate_hz / args.hops_per_sec
                                * (1.0 + args.code_doppler_sign * dop
                                   / args.carrier_hz))
                if args.search_fft_len:
                    _hop_off_det *= 1.0 - 1.0 / args.search_fft_len
                _inv = ((_cpe_recon + _hop_off_det - _fc - _trim_eff + CODE_LEN / 2.0)
                        % CODE_LEN) - CODE_LEN / 2.0
                _ih = innov_hist.setdefault(prn, [])
                _ih.append((t0, _inv))
                # Bounds MEMORY only: the 10-minute statistic is cut by time at read
                # (the publish block), so this cap can never shorten the window.
                del _ih[:-120]
            # ── #83 P3-3b: MODEL-PRIMARY PRNs stop here ──
            # This detection has already done everything it is allowed to do: the #79
            # deep-gate stamp (above), INNOV (above), and the joint feed + MINNOV run in
            # their own block from `offs`. The SEED stays the MODEL's -- the dr slew path
            # owns it via the eligibility exceptions there -- so the re-anchor, the
            # hold/escape machinery and the commit below are all bypassed. The search
            # remains the referee THROUGH MINNOV: its exit gate (p95 hysteresis +
            # starvation) hands the PRN back, and the next detection re-anchors normally.
            if prn in mp_flipped:
                mp_last_det[prn] = t0
                continue
            if (prev is not None and prn in cp_held
                    and all(k in seed for k in ("code_phase_chips", "code_phase_rate",
                                                "ref_hop"))):
                # AT-EPOCH COMPARISON (#42 -> #45 step 1, 2026-08-12). The sample-0
                # currency comparison that lived here manufactured -t_abs*k*d(clock_bias)
                # chips of phantom whenever the seed-vs-detection dop bias moved (an EMA
                # DESIGNED to move): 145 CP_ERR-DECOMP specimens in 7 min, seven false
                # ESCAPES in one evening against tracks healthy at 40 dB-Hz. The
                # candidate tuple reaching this point can be PAIR-INCONSISTENT (label
                # stepped, cp0 unmoved) and no translation survives that. So: compare
                # physical phases at the DETECTION's epoch instead -- the search's own
                # cp_at_ref (dt=0, its Doppler estimate does not enter) against where
                # dr_seed_phys puts the tracker's despread. test_track_vs_fit.py pins
                # the discriminating pair (bias step: old ~1700/Hz, new ~0; real lobe
                # park: both read +-3.27) and drives the SHIPPED function.
                # #83 2(d) ADDENDUM (2026-08-17): the referee now CONSUMES THE INNOVATION
                # above -- forecast by tracker_phase_at, the reference propagate_seed
                # actually prefers -- instead of track_vs_fit_chips' dr_seed_phys (the
                # cp0-argument path). On sky the cp0 fiction read wrap-uniform thousands
                # of chips (p95 ~5000) on satellites tracking cleanly: seed writers move
                # doppler_hz without re-projecting cp0, and the implied phase swings on
                # the t_abs lever (~5600 chips/Hz). That is the named mechanism behind
                # the +-700-4500-chip CP_ERR reports of 08-16/17 -- the referee had been
                # accusing healthy tracks with a number drawn from a uniform
                # distribution, and only the sign-consistency + median gates kept it
                # from escaping constantly. Confirmed as a discriminating pair on the
                # holds fixture: same replay, forecast swapped, innovations collapse to
                # p95 1.6-2.3 chips.
                cp_err = _inv
                if cp_err is not None and abs(cp_err) > args.hold_max_cp_err:
                    _log_rl("cperr-%d" % prn,
                            "CP_ERR PRN %d: %+.2f chips at det hop %d (at-epoch: "
                            "search cp_at_ref vs held propagation; trim %+.2f "
                            "= py %+.2f + cpp %+.2f, hold_age %.0f s)"
                            % (prn, cp_err, ref_hop, _trim_eff,
                               _dls.trim.get(prn, 0.0),
                               (_dls.readback.get(prn) or {}).get("trim_chips", 0.0),
                               (ref_hop - prev["ref_hop"]) / args.hops_per_sec),
                            every_s=60.0)
                if cp_err is not None:
                    _cpt.err_hist.setdefault(prn, []).append(cp_err)
                    del _cpt.err_hist[prn][:-9]
                # MEDIAN GATE (2026-07-19): the per-detection cp noise is 0.03-0.5 chips
                # (per-sat conditions -- multipath/BOC refine; measured same-instrument at
                # t_abs 100 s AND 27000 s, i.e. FLAT in run age: the earlier 'growth law'
                # was the logged cp_ref coordinate wobbling with dop_seed x t_abs, which
                # the currency translation above cancels in cp_err by construction). The
                # 5-consecutive-sign rule alone still lets a noisy-conditions sat sustain
                # a false accusation; a 9-sample median cannot be dragged over the bar by
                # single-point noise, only by a persistent physical walk.
                _cpt.err_hist.setdefault(prn, []).append(cp_err)
                del _cpt.err_hist[prn][:-9]
            # Count only SIGN-CONSISTENT disagreements: a real wrong-lobe park is
            # one-signed; search-fit noise on a weak sat alternates (first deploy:
            # weak GPS sats escaped every minute on -0.5/+1.3/-0.5 flip-flops, and
            # each escape re-injects the seed jitter + forces an overlay re-align).
            # FIT-QUALITY gate (2026-07-12 evening): only a trustworthy fit may accuse the
            # track -- >=6 history points (fit noise ~ per-fix/sqrt(n)) and a solid current
            # detection (2x the acquire gate). Ungated, weak-sat fit noise drove 627 escapes
            # in 2.7 h and each one re-anchored the seed (the churn behind the GPS "wobble").
            # + FIT SPAN >= 30 s (2026-07-19 eve): point COUNT is not maturity -- L2C's
            # snapshots arrive fast enough that 6 points span ~13 s, over which the code-
            # Doppler QUADRATIC is unresolvable, so the fit carries the curvature-bias class
            # (the birth zombies the watchdog had to keep cleaning). A 30 s floor makes the
            # curvature term observable on every chain; on L1 (6 points ~ 60-80 s) it is a
            # no-op. This gate feeds BOTH the escape referee and hold admission.
            fit_span_s = ((h[-1][0] - h[0][0]) / args.hops_per_sec) if len(h) >= 2 else 0.0
            fit_trusted = (fit is not None and len(h) >= 6
                           and fit_span_s >= args.fit_maturity_span_s
                           and snr >= 2.0 * args.acquire_snr)
            # AMP VETO (see --escape-amp-veto): a full-amplitude hold is on the main peak
            # by construction -- refuse the fit's accusation rather than drag it off.
            amp_now = float((status.get(prn) or {}).get("amp_snr", 0) or 0)
            amp_veto = (args.escape_amp_veto > 0.0
                        and amp_now > args.escape_amp_veto)
            # INTEGRITY VETO (2026-07-19 eve, audit follow-up): never re-anchor onto a fit
            # the BRDC model itself disputes. The dead-reckon machinery already computes a
            # per-sat integrity residual (search-vs-model, solved clock removed, normally
            # +-0.2 chips): if a FRESH residual says the search's own position is off by
            # more than the escape bar, the fit is the suspect, not the track.
            integ_veto = False
            if dr_state is not None and dr_state.get("integ"):
                _iv = dr_state["integ"].get(prn)
                if (_iv is not None and t0 - _iv[1] < 10.0
                        and abs(_iv[0]) > args.hold_max_cp_err):
                    integ_veto = True
            cp_err_med_ok = (cp_err is not None and len(_cpt.err_hist.get(prn, [])) >= 5
                             and abs(statistics.median(_cpt.err_hist[prn]))
                             > args.hold_max_cp_err)
            if (cp_err is not None and abs(cp_err) > args.hold_max_cp_err
                    and cp_err_med_ok and fit_trusted and not amp_veto
                    and not integ_veto):
                n_prev = _cpt.escape.get(prn, 0)
                same_sign = (n_prev == 0) or (cp_err * _cpt.escape_sign.get(prn, 0.0) > 0)
                _cpt.escape[prn] = n_prev + 1 if same_sign else 1
                _cpt.escape_sign[prn] = cp_err
            else:
                _cpt.escape[prn] = 0
            if _cpt.escape.get(prn, 0) >= 5:
                _log("ESCAPE PRN %d: track %+.2f chips off the search fit (5 consecutive,"
                     " sign-consistent) -> release hold + DLL trim, re-anchor on the fit"
                     % (prn, cp_err))
                _cpt.escape[prn] = 0
                _cpt.err_hist.pop(prn, None)
                _dls.trim.pop(prn, None)
                _dls.last.pop(prn, None)
                cp_held.discard(prn)
                _hold.miss.pop(prn, None)
                # The re-anchor refreshes the seed doppler next cycle = an NCO f_ref step
                # the TRACK-mode trim was not built for (same latch as the hold release,
                # and it bypasses that branch because cp_held is discarded HERE): demote
                # to BOOTSTRAP so the carrier re-pulls instead of parking off-frequency.
                if prn in _carrier.locked:
                    _carrier.locked.discard(prn)
                    _carrier.fade.pop(prn, None)
                    _log("CARRIER REACQ PRN %d: escape re-anchor -> BOOTSTRAP re-pull" % prn)
            # TRACK-vs-MODEL MONITOR (2026-07-20, log-only; audit follow-up census): the
            # referee's reference is the search FIT; the model-referenced track residual
            # is r_i - cp_err (search-vs-model minus search-vs-track, same chip units).
            # Log when the MODEL says the track is past the escape bar -- especially when
            # the fit-referenced referee stays quiet (veto / immature fit): those are the
            # cases an upgraded model-referenced referee would catch. Decide on enforcement
            # from this census, not from theory (the referee has bitten guessers before).
            if (cp_err is not None and dr_state is not None and dr_state.get("integ")):
                _iv2 = dr_state["integ"].get(prn)
                if _iv2 is not None and t0 - _iv2[1] < 10.0:
                    _tm = _iv2[0] - cp_err
                    if abs(_tm) > args.hold_max_cp_err:
                        _log_rl("tvm-%d" % prn,
                                "TRACK-vs-MODEL PRN %d: %+.2f chips past the escape bar "
                                "(fit-ref cp_err %+.2f, integ %+.2f; fit-referee %s) -- "
                                "monitor only"
                                % (prn, _tm, cp_err, _iv2[0],
                                   "AMP-VETOED" if amp_veto else
                                   "INTEG-VETOED" if integ_veto else
                                   "fit-untrusted" if not fit_trusted else "active"),
                                every_s=120.0)
            # HOLD ADMISSION REQUIRES FIT MATURITY (2026-07-19 eve, the Tier-3 burn-in fix):
            # a birth-window anchor (wide margins, unsolved bias, <6-point fit) can be chips
            # wrong, and granting it hold protection created the zombie cohorts that made
            # every relaunch take 5-20 min to heal (or forever, pre-watchdog: the L2C 18%
            # mornings). Until the sat's own cp fit is trusted -- the SAME predicate the
            # escape referee requires before it may accuse a track -- the seed keeps riding
            # the maturing fit (the weak-sat path, measured born-clean tonight), which is
            # self-correcting. Only a mature anchor earns protection; expected burn-in
            # collapses to the fit-maturation time (~6 search snapshots, ~60-80 s).
            # Already-held sats are unaffected (the cp_held alternative below).
            elif (prev is not None
                    and ((sig_of_last(status.get(prn)) >= args.hold_snr
                          and (prn in cp_held or fit_trusted))
                         or (prn in cp_held and _hold.miss.get(prn, 0) < 3))):
                # PERSISTENT-loss release (2026-07-12 evening): a single blank/stale status
                # read (sig 0.0 -- a poll racing the emit, a slow combiner cycle) used to
                # release the hold instantly: 562 of 736 releases in 2.7 h fired at
                # amp_snr < 8, mostly 0.0, and every release re-fit the seed (dop jump) and
                # paid the ~0.9 Hz x ~5 s carrier re-anchor transient -- the 2 s-median
                # churn behind the GPS coherence wobble (settled sats measure 0.06 Hz).
                # A held sat now rides through up to 3 consecutive sub-gate reads; doppler
                # STALENESS still releases immediately (real currency decoherence).
                if sig_of_last(status.get(prn)) >= args.hold_snr:
                    _hold.miss[prn] = 0
                else:
                    _hold.miss[prn] = _hold.miss.get(prn, 0) + 1
                ddop = seed["doppler_hz"] - prev["doppler_hz"]
                # SAFETY NET (design (b)): bound a single cycle's Doppler move. A real MEO
                # Doppler moves <1 Hz per 0.2 s cycle; this only fires on a bad model, and it
                # bounds the damage rather than forbidding motion.
                if abs(ddop) > args.dop_max_rate_hz:
                    if prn not in _cpt.dop_clamped:
                        _cpt.dop_clamped.add(prn)
                        _log("DOP-CLAMP PRN %d: model wanted %+.1f Hz in one cycle (max %.1f)"
                             " -- clamping. A real MEO moves <1 Hz/cycle: SUSPECT THE MODEL."
                             % (prn, ddop, args.dop_max_rate_hz))
                    ddop = math.copysign(args.dop_max_rate_hz, ddop)
                    seed.put("dop_clamp", doppler_hz=prev["doppler_hz"] + ddop)
                # DESIGN (b): translate EVERY cycle (no fence). The freeze branch survives only
                # for --no-dop-continuous, and for the zero-motion case where it is a no-op.
                if (not args.dop_continuous and abs(ddop) <= args.hold_max_dop_hz) or ddop == 0.0:
                    # Currency frozen: the whole tuple rides unchanged.
                    seed.put("hold_freeze", epoch=prev["ref_hop"],
                             doppler_hz=prev["doppler_hz"],
                             code_phase_chips=prev["code_phase_chips"],
                             code_phase_rate=prev["code_phase_rate"],
                             ref_hop=prev["ref_hop"])
                    # #80 FIX (2026-08-16): the at-ref phase is PART of the tuple and the
                    # tracker PREFERS it (gnssSeedTransport.cpp:325 -- phase_ref_chips >= 0
                    # wins over cp_chips unconditionally). Leaving the DETECTION's fresh
                    # phase beside the frozen ref_hop commanded the despread off-peak by
                    # the full inter-snapshot code advance (~4665 chips mod-period per
                    # 6.29 s revisit, wrapping): measured on sky 2026-08-16 23:15 --
                    # CP_ERR +1005..+4516 chips on ALL six held sats at hold_age 12-15 s,
                    # prompt amps collapsed 21-77 -> 3-16 while the re-searching deep fold
                    # kept sig above the release bar, so the mispaired hold NEVER let go
                    # (#48's "prompt on noise", mechanism). Freeze means freeze: the phase
                    # rides from prev (where it is paired with prev's ref_hop), or ships
                    # not at all -- the tracker then falls back to the frozen (cp0, dop)
                    # argument pair, which is the hold's original contract.
                    if "code_phase_at_ref_chips" in prev:
                        seed.put("hold_freeze", epoch=prev["ref_hop"],
                                 code_phase_at_ref_chips=prev["code_phase_at_ref_chips"])
                    else:
                        seed.pop("code_phase_at_ref_chips", None)
                else:
                    # ---- CURRENCY TRANSLATION (2026-07-14): a DOPPLER UPDATE IS NOT A LOSS
                    # OF LOCK. The replica is anchored at sample 0, so the tracker builds the
                    # code phase as
                    #     cp(t) = cp0 + t*f_chip*(1 + sign*dop/f_carrier)
                    # -- the Doppler is applied over the ENTIRE elapsed time, RETROACTIVELY.
                    # cp0 is therefore not "the code phase": it is a coordinate that only
                    # means anything PAIRED WITH A DOPPLER. Change dop by ddop and leave cp0
                    # alone and the physical code jumps by t*f_chip*ddop/f_carrier -- ~20
                    # chips for a 10 Hz fence step at an hour of run age, and the lever arm
                    # GROWS with t.
                    #
                    # Historically the fence RELEASED the hold and re-anchored cp on the
                    # search fit. The 20-chip currency jump was bookkeeping and harmless; the
                    # damage was throwing away a good anchor and replacing it with a MEASURED
                    # one. The fit's noise is nothing on GPS's broad triangular peak, but a
                    # BOC(1,1) peak is ~3x sharper: measured 2026-07-14, C19 collapsed 42 ->
                    # 20 dB-Hz and took ~5 s of DLL re-convergence, EVERY
                    # hold_max_dop_hz/|dop_rate| = 10 Hz / 0.55 Hz/s = 18 s. ~25% of every
                    # emit, on every locked BOC satellite, on all three constellations.
                    #
                    # So: keep the anchor and re-express it in the new Doppler's currency.
                    # Pure arithmetic, no measurement, no fit noise, nothing for the deep
                    # integration to trip over. The new slope is the BETTER one (dop is a
                    # better estimate of the truth), so code_phase_rate is left alone -- only
                    # the retroactive part needs absorbing.
                    t_now = ref_hop / args.hops_per_sec
                    # doppler_hz keeps its NEW value -- that is the point -- and is
                    # re-attributed here because the translation is what makes the
                    # (cp0, dop) pair valid at the shipped ref_hop by construction.
                    seed.put("translate", epoch=prev["ref_hop"],
                             code_phase_chips=(
                                 prev["code_phase_chips"]
                                 - t_now * args.chip_rate_hz * args.code_doppler_sign
                                 * ddop / args.carrier_hz) % CODE_LEN,
                             code_phase_rate=prev["code_phase_rate"],
                             ref_hop=prev["ref_hop"],
                             doppler_hz=seed["doppler_hz"])
                    # #80 FIX, translate arm (the LIVE arm under --dop-continuous): same
                    # as the freeze arm above. prev's at-ref phase is a PHYSICAL phase at
                    # prev's ref_hop -- a doppler update does not move it (the new doppler
                    # enters the forward propagation, not the anchor), so it rides
                    # unchanged where the fresh detection's phase must not.
                    if "code_phase_at_ref_chips" in prev:
                        seed.put("translate", epoch=prev["ref_hop"],
                                 code_phase_at_ref_chips=prev["code_phase_at_ref_chips"])
                    else:
                        seed.pop("code_phase_at_ref_chips", None)
                    if prn not in _cpt.translated:
                        _cpt.translated.add(prn)
                        _log("TRANSLATE PRN %d: dop %+.0f -> %+.0f (%+.2f Hz) -> cp0 shifted "
                             "%+.2f chips; SAME physical code phase, anchor KEPT%s"
                             % (prn, prev["doppler_hz"], seed["doppler_hz"], ddop,
                                -t_now * args.chip_rate_hz * args.code_doppler_sign
                                * ddop / args.carrier_hz,
                                " (continuous: every cycle, no fence)"
                                if args.dop_continuous else ""))
                if prn not in cp_held:
                    _log("HOLD PRN %d: seed currency frozen (amp_snr %.1f >= %.1f, dop %+.0f)"
                         % (prn, sig_of_last(status.get(prn)), args.hold_snr,
                            prev["doppler_hz"]))
                cp_held.add(prn)
            else:
                if prn in cp_held:
                    ddop_rel = (seed["doppler_hz"] - prev["doppler_hz"]) if prev else 0.0
                    _log("RELEASE PRN %d: seed currency unfrozen (amp_snr %.1f, ddop %+.0f)"
                         % (prn, sig_of_last(status.get(prn)), ddop_rel))
                    # A release used to STEP the tracker's f_ref by ddop while the TRACK-mode
                    # trim carried the hold-era compensation -> instant residual ~ -ddop,
                    # latched by the coh/innovation gates (C20 parked at -6.2 Hz for 40 min,
                    # 2026-07-18). The arithmetic pre-shift (--trim-precomp-carrier) was
                    # bench-rejected in both signs and DELETED (07-19 audit A4); the safe
                    # rescuer below stands: the broker KNOWS the NCO stepped -- demote to
                    # BOOTSTRAP and re-pull the trim at full gain (seconds, no arithmetic).
                    # Under --dop-continuous ddop_rel is ~0 and this never fires.
                    if abs(ddop_rel) > 1.0 and prn in _carrier.locked:
                        _carrier.locked.discard(prn)
                        _carrier.fade.pop(prn, None)
                        _log("CARRIER REACQ PRN %d: hold released with dop step %+.1f Hz "
                             "-> BOOTSTRAP re-pull" % (prn, ddop_rel))
                cp_held.discard(prn)
                _hold.miss.pop(prn, None)
            seeds[prn] = seed
            _hold.low_hits[prn] = 0


    def _stage_coast_drop():
        """COAST / DROP: retire seeds for satellites that have set, and coast the ones merely fading.
        
        ⚠️ MODEL-OWNED SATS ARE EXEMPT FROM THE TLE UP-SET. The up-set mismaps some BeiDou birds, so
        for dead-reckoned satellites the BRDC elevation governs the drop instead -- in the dead-reckon
        stage, not here. Dropping a bird the model is still tracking would hand the chain a hole it
        would then try to re-acquire from scratch."""
        for prn in list(seeds):
            if prn in probe_set:
                continue
            if (_ctx.up is not None and prn not in _ctx.up
                    and not (dr_state is not None and prn in dr_state["seeded"])):
                # (model-owned sats are exempt: the TLE up-set mismaps some BDS birds;
                # their BRDC elevation governs the drop, in the dead-reckon block)
                _log("drop PRN %d (set below horizon)" % prn)
                del seeds[prn]
                cp_held.discard(prn)
                _hold.miss.pop(prn, None)
                _hold.low_hits.pop(prn, None)
                continue
            if prn in best:  # re-detected -> re-anchored in the seed loop above (coast reset there)
                continue
            # not re-detected this poll but still visible -> COAST: forecast the Doppler forward.
            if dr_state is not None and prn in dr_state["seeded"]:
                # model-owned (dead-reckoned) seed: its doppler is FROZEN between re-pins
                # (each pin refreshes dop+cp+rate TOGETHER, currency-consistent); the TLE
                # pred must not touch it -- the BDS TLE<->PRN mapping mismaps some birds
                # (PRN 39: TLE el <5 vs BRDC el 10), so BRDC governs these sats entirely.
                pass
            elif args.almanac and prn in _ctx.pred:
                # ⚠️ THE SEED'S DOPPLER IS VALID AT ref_hop, NOT AT NOW (2026-08-16).
                # gnss::propagate_seed computes
                #     dop_applied(t) = sd.doppler_hz + sd.dop_rate * (t - ref_hop)
                # so writing the forecast's NOW value into `doppler_hz` while leaving
                # `ref_hop` at the last re-detection makes the tracker add dop_rate*age on
                # top of a value that is ALREADY current. The applied Doppler then advances
                # at ~2x the true rate until the next detection re-anchors ref_hop, and
                # snaps back when it does -- the sawtooth measured on sky: drift -0.63 Hz/s
                # against a model -0.35, gap 60-81 Hz just before the snap, and
                # 69 Hz = 0.60 chips/s of CODE rate, which drags the prompt tap off-peak in
                # under two seconds. That is the churn (deep sig 1600 -> 3 on G23).
                #
                # Same disease as REC_PHI0 and as #44 itself: a VALUE updated to now while
                # its EPOCH says otherwise ([[#45]] value/currency/epoch). Two consequences,
                # both fixed here:
                #   1. store the Doppler BACK-PROPAGATED to ref_hop, so propagate_seed
                #      reconstructs the forecast exactly at now for ANY age;
                #   2. compare like with like when re-tagging cp -- the old EFFECTIVE
                #      Doppler at now, not the ref_hop-epoch number, or the ddop handed to
                #      retag_seed_doppler is inflated by that same dop_rate*age and kicks
                #      the code phase as well.
                # age = 0 whenever the sample-0 anchor is unknown, which reduces this to the
                # previous behaviour rather than guessing.
                _rate_new = _ctx.pred[prn][1]
                # the rate the TRACKER will actually propagate with: absent key => 0, and
                # then there is no double-count to undo.
                _rate_eff = _rate_new if "doppler_rate_hz_s" in seeds[prn] else 0.0
                _age = 0.0
                if utc0_sample0:
                    _age = max(0.0, (_drp.now_w - utc0_sample0)
                               - seeds[prn].get("ref_hop", 0) / args.hops_per_sec)
                new_dop = _ctx.pred[prn][0] + _cb.value           # the forecast AT NOW
                _old_rate = seeds[prn].get("doppler_rate_hz_s", 0.0)
                # what the tracker is APPLYING at this instant, i.e. the currency cp is in
                old_dop = seeds[prn].get("doppler_hz", new_dop) + _old_rate * _age
                if new_dop != old_dop:
                    # CURRENCY-CORRECT the coast, UNCONDITIONALLY (2026-07-19 audit A4): cp0
                    # is meaningful only in its doppler's currency -- updating the forecast
                    # dop WITHOUT re-expressing cp walks the despread by t_abs*f_chip*ddop/
                    # f_c chips at soak age (the t_abs lever the code-currency rule forbids;
                    # why long coasts silently lost the code peak).
                    #
                    # #44 (2026-08-12): the first fix translated at the SEED'S ANCHOR epoch
                    # (ref_hop) -- which preserves the phase at the anchor and steps the
                    # phase NOW by anchor_age * k_c * ddop per forecast update, i.e. the
                    # residual half of the very symptom it targeted (~k_c*r*age^2/2 chips
                    # of silent walk-off over a coast). The correct epoch is NOW, same as
                    # the hold-path TRANSLATE has always used; both now go through
                    # retag_seed_doppler so the algebra lives once, beside dr_cp0/
                    # dr_seed_phys, with its own regression test. When the sample-0 anchor
                    # is not known (no utc0_sample0), keep the old anchor-epoch behaviour:
                    # a partial correction still beats the raw-dop overwrite it replaced.
                    _t_retag = ((_drp.now_w - utc0_sample0) if utc0_sample0
                                else seeds[prn].get("ref_hop", 0) / args.hops_per_sec)
                    seeds[prn].put(
                        "coast_retag", epoch=seeds[prn].get("ref_hop"),
                        code_phase_chips=retag_seed_doppler(
                            seeds[prn].get("code_phase_chips", 0.0), old_dop, new_dop,
                            _t_retag, args.chip_rate_hz, args.carrier_hz,
                            args.code_doppler_sign, CODE_LEN),
                        # STORE AT ref_hop, not at now (see the note above).
                        doppler_hz=new_dop - _rate_eff * _age)
                if "doppler_rate_hz_s" in seeds[prn]:
                    seeds[prn].put("coast_retag", epoch=seeds[prn].get("ref_hop"),
                                   doppler_rate_hz_s=_rate_new)
            rec = status.get(prn, {})
            if have_sig:
                metric, thresh = sig_of(rec), args.lock_snr
            else:
                metric, thresh = float(rec.get("amplitude", 0.0)), args.drop_amplitude
            # FOLD-INDEPENDENT HOLD (#58). OR-ed against its own bar, never max()-ed into
            # `metric`: prompt hold is a power ratio and `metric` is a debiased sigma.
            if metric >= thresh or (args.lock_prompt_hold > 0.0
                                    and _hold.prev.get(prn, 0.0) >= args.lock_prompt_hold):
                _hold.low_hits[prn] = 0  # lock holding through the dropout -> reset coast
            else:
                _hold.low_hits[prn] = _hold.low_hits.get(prn, 0) + 1
                # dead-reckoned seeds are MODEL-owned: visible + predicted = keep despreading
                # (their whole point is sats with no signal above the search threshold)
                if (_hold.low_hits[prn] >= coast_polls and not args.coast_to_horizon
                        and not (dr_state is not None and prn in dr_state["seeded"])):
                    _log("drop PRN %d (coast %.0fs expired, %s=%.2f)"
                         % (prn, args.coast_budget, "sig" if have_sig else "|A|", metric))
                    del seeds[prn]
                    _hold.low_hits.pop(prn, None)

    def _stage_rate_feed_fine():
        """#33 RATE FEED (fine): the per-record fine rate, and the #93 ADR-span capture.
        
        Same reference discipline as the coarse feed: the standing command is added back so the
        observable describes the sky. This is also where the shadow's direct per-satellite ADR span
        rate is captured (`_adr_span_now`), which #93's v2 compares against the rrate ROW -- the row
        was measured on 2026-08-25 to be ~97% carrier-only (GAP 3's F1 verdict)."""
        if args.rrate_state and _drp.t_now_abs is not None:
            try:
                _rec_dt = 2048.0 / args.hops_per_sec
                _jpp = []
                _n_fine = 0
                _jrf = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference)
                for _p, _rec in (status or {}).items():
                    if not isinstance(_rec, dict):
                        continue
                    _cmd_now = rr_cmd_applied.get(_p, 0.0)
                    # GAP-1 LONG SPAN (2026-08-25): with --rrate-phase-span-s set, the
                    # difference baseline is the newest ring snapshot AT LEAST span_s old,
                    # not last poll's. res_cycles' error TELESCOPES (measured 08-24,
                    # fixtures/gap1_tau_scaling.py: per-sat sigma_rate 1.12 Hz at 2 s ->
                    # 0.31 at 16 s -> 0.18 at 32 s, converging 1/tau on both instances
                    # checked), so a longer baseline buys noise down linearly while the
                    # staleness cost is only drift*(span/2) ~ 0.02 Hz/s * span/2 -- the
                    # 16-32 s window nets ~0.35 Hz effective, under the 0.5 Hz quietness
                    # bar that parked arms 16b/17. All of adr_fine_rate's structural gates
                    # (same accumulator arc, counter advanced, span-vs-wall) apply
                    # unchanged; a ring entry that predates an arc break is rejected by
                    # the arc gate exactly like a last-poll one.
                    _ring = adr_ring.setdefault(_p, [])
                    if args.rrate_phase_span_s > 0.0:
                        while _ring and (t0 - _ring[0][2]) > 2.0 * args.rrate_phase_span_s:
                            _ring.pop(0)
                        _pv = None
                        for _e in reversed(_ring):
                            if (t0 - _e[2]) >= args.rrate_phase_span_s:
                                _pv = _e
                                break
                    else:
                        _pv = adr_prev.get(_p)
                    if _pv is not None:
                        _snap = {"adr_arc": _pv[0][0], "adr_records": _pv[0][1],
                                 "res_cycles": _pv[0][2]}
                        _snap["trim_cycles"] = _pv[0][3] if len(_pv[0]) > 3 else None
                        # wall_dt arms the serving-churn discriminator (see
                        # adr_fine_rate): the row is best-of-instance and the winner
                        # churns; a cross-instance span is wrong by up to 12x while
                        # passing the arc gate.
                        _wdt = (t0 - _pv[2]) if len(_pv) > 2 else None
                        _fr = adr_fine_rate(_rec, _snap, _rec_dt, wall_dt=_wdt)
                        if _fr is not None:
                            _fy, _nrec, _applied = _fr
                            _co = _rr2_resid.get(_p) if _rr2_resid else None
                            if _co is not None:
                                _jpp.append((_p, _fy, _co))
                            # ── #93 SHADOW v2 CAPTURE (read-only) ──────────────────
                            # The sat's total carrier residual rate vs the seeded
                            # model, reference-corrected EXACTLY as the feed would
                            # (measured applied command when served and plausible,
                            # else the posted-command span-mean) -- but on the shadow
                            # path, so the unfed chain (e5a) measures it too. Sign
                            # falls back to +1 where uncalibrated; the fold-safety
                            # bound is the same 20 Hz as the feed.
                            _sg93 = args.rrate_phase_sign or 1.0
                            if abs(_sg93 * _fy) < 20.0:
                                _ap93 = (_applied
                                         if (_applied is not None
                                             and abs(_applied) <= 50.0) else None)
                                _cm93 = ((_sg93 * _ap93) if _ap93 is not None
                                         else 0.5 * (_cmd_now + _pv[1]))
                                _adr_span_now[_p] = (_sg93 * _fy + _cm93, t0, _nrec)
                            # COMMAND MOTION over the span enters the REFERENCE, not a
                            # stillness gate (a strict gate starved the feed to ~1/min --
                            # the coarse loop nudges the command every poll). Span-mean
                            # is the best estimate of the applied command for an unknown
                            # application time within the span; its worst-case error is
                            # dcmd/2, which goes into sigma rather than into a silent
                            # bias. Gate only at ONE slew step (0.6): beyond that the
                            # command jumped for a non-loop reason (re-seed, probe) and
                            # the span is not a measurement.
                            _dcmd = _cmd_now - _pv[1]
                            # SPAN-MODE NON-OVERLAP THROTTLE (2026-08-25): at poll cadence
                            # consecutive 16 s-span values share 14/16 of their data --
                            # feeding them as independent makes the row ~sqrt(8)x
                            # overconfident. In span mode a sat feeds at most once per
                            # span, so fed measurements are disjoint windows. The SHADOW
                            # (JRRP) stays per-poll; only the feed is throttled.
                            _span_ok = (args.rrate_phase_span_s <= 0.0
                                        or t0 - _span_fed_t.get(_p, 0.0)
                                        >= args.rrate_phase_span_s)
                            if (args.rrate_phase_feed and args.rrate_phase_sign != 0.0
                                    and _span_ok
                                    and abs(_dcmd) <= 0.6
                                    and ((_rec.get("coherence_s") or 0.0) > 0.0
                                         or (_rec.get("coh_frac") or 0.0) >= 0.3)):
                                _yf = args.rrate_phase_sign * _fy
                                # NO CONVERGENCE REGIME (00:2x, measured): res_cycles is
                                # UNWRAPPED -- summed per-record increments, no mod-2pi
                                # anywhere -- so the fine value is valid at ANY residual
                                # below the per-record fold bound (0.25 cyc / 10.5 ms
                                # ~ +-23 Hz). The old 0.3 gate was FLL->PLL folklore: it
                                # kept fine to ~1 sat/poll while the two sats inside it
                                # held their commands to +-0.02 Hz and everyone else
                                # wandered at the coarse floor. Gate only at fold safety.
                                if abs(_yf) < 20.0:
                                    _k = (args.dr_constellation, int(_p))
                                    # reference: the MEASURED applied command when the
                                    # combiner serves it (same span, same stream, no
                                    # assumption -- and no motion penalty needed, it is
                                    # exact); the posted-command span-mean otherwise.
                                    # TRIPWIRE: since the honest PrnCtl::ctrim_hz export
                                    # (2026-08-14) trim_cycles integrates the trim the
                                    # tracker ACTUALLY applied on every chain. A tracker
                                    # still running an older binary serves the airspy
                                    # identity's (ctrim - f_offset)/2 = MHz-scale garbage
                                    # (arm 9's rail runaway) -- this bound rejects that
                                    # loudly. A plausible applied command is tens of Hz.
                                    if _applied is not None and abs(_applied) > 50.0:
                                        _applied = None
                                    if _applied is not None:
                                        _cmd_mid = args.rrate_phase_sign * _applied
                                        _sig_f = args.rrate_phase_sigma
                                    else:
                                        _cmd_mid = 0.5 * (_cmd_now + _pv[1])
                                        _sig_f = (args.rrate_phase_sigma ** 2
                                                  + (0.5 * _dcmd) ** 2) ** 0.5
                                    if args.rrate_phase_span_s > 0.0:
                                        # sigma is defined AT THE 1-POLL SPAN and the
                                        # noise telescopes (1/span, measured); the
                                        # staleness term prices the span-mean lagging a
                                        # ~0.02 Hz/s drifting rate by span/2. The
                                        # measurement is timestamped at t_now (the filter
                                        # predicts forward only), so the lag lives HERE,
                                        # in the weight, not in the epoch.
                                        _span_s = _nrec * _rec_dt
                                        _sig_f = ((_sig_f * args.interval
                                                   / max(_span_s, args.interval)) ** 2
                                                  + (0.02 * 0.5 * _span_s) ** 2) ** 0.5
                                    if _jrf.update_rrate(
                                            _k, _yf + _cmd_mid, _drp.t_now_abs, args.carrier_hz,
                                            sigma_hz=_sig_f) is not None:
                                        _n_fine += 1
                                        # ACCEPTED fine measurements arm the handoff --
                                        # not attempts, so a sat whose fine values the
                                        # gate keeps rejecting stays coarse-governed.
                                        rr_fine_t[_p] = t0
                                        _span_fed_t[_p] = t0
                    adr_prev[_p] = ((_rec.get("adr_arc"), _rec.get("adr_records") or 0,
                                     _rec.get("res_cycles"), _rec.get("trim_cycles")),
                                    _cmd_now, t0)
                    if args.rrate_phase_span_s > 0.0:
                        _ring.append(adr_prev[_p])
                # A sat that has left the seed set is RE-ACQUIRING when it returns, which
                # is the coarse feed's job -- drop its fine lock rather than let a stale
                # one de-weight the very measurements that must pull it back in.
                for _dead in [k for k in rr_fine_t if k not in seeds]:
                    rr_fine_t.pop(_dead, None)
                    adr_prev.pop(_dead, None)
                    adr_ring.pop(_dead, None)
                if _jpp:
                    _log_rl("jrrp",
                            "JRRP[%s%s] fine|coarse Hz (fine in INTERNAL sign): %s%s"
                            % (args.dr_constellation,
                               (" span %.0fs" % args.rrate_phase_span_s)
                               if args.rrate_phase_span_s > 0.0 else "",
                               " ".join("%d:%+.3f|%+.3f" % t for t in _jpp),
                               (" -- %d fine-fed" % _n_fine) if _n_fine else ""),
                            every_s=60.0)
            except Exception as e:
                _log_rl("jrrp-err", "phase-step feed skipped: %s" % e, every_s=300.0)

    def _stage_fleet_trim_arming():
        """4b: FLEET-TRIM ARMING -- tell the gather which PRNs its C++ loop may actuate.
        
        ⚠️ PRESENCE WITH A HOLD, NOT PRESENCE AT AN INSTANT. A satellite flickering across the deep
        gate would otherwise be armed and released every cycle, and an arming change costs the trim.
        
        ⚠️ THE HANDOVER'S HALF-STEP IS RECORDED BEFORE THE POST, NOT AFTER. `_ft_armed_last` is what
        next cycle's Python integrator stands down against, so authority is never held by both arms
        and never by neither. Recording after a POST would mean a failed POST leaves both sides
        believing the other is driving."""
        if args.fleet_trim_url:
            _now_present = [_p for _p in (_dllp.fleet or {}) if _dllp.fleet[_p].get("present")]
            for _p in _now_present:
                _dls.hold[_p] = time.time()
            # PRESENCE WITH A HOLD, not presence sampled at an instant -- see the flag.
            _armed = sorted(_p for _p, _t in _dls.hold.items()
                            if time.time() - _t < args.fleet_trim_hold_s)
            # THE HANDOVER'S HALF-STEP: record what we are about to hand the fast loop, so
            # NEXT cycle's slow integrator stands down for exactly the PRNs the C++ side is
            # actuating. Recorded before the POST rather than after, because a failed POST
            # leaves the controller running its LAST policy -- which is this one either way,
            # and the trims expire at the trackers if it never recovers.
            _dls.armed_last.clear()
            _dls.armed_last.update(_armed)
            _pol = {"chains": {telem_chain: {
                "armed": _armed,
                # BANDWIDTH, not per-update gain -- the controller converts with its measured
                # rate. The slow DLL's dll_gain/dll_leak_present are NOT reused here: those
                # constants are per-update at THIS process's cadence, and reusing them at
                # 23.84 Hz is exactly the limit cycle of 2026-08-15. See the two flags.
                "gain_per_s": args.fleet_trim_bandwidth,
                "leak_per_s": args.fleet_trim_leak_per_s,
                "clamp": 3.0,
                "spacing": args.dll_spacing,
                "targets": ["%s/set_trim" % t for t in trackers]}}}
            try:
                _post("%s/set_policy" % args.fleet_trim_url.rstrip("/"), _pol, timeout=2.0)
                _dls.stat["posts"] += 1
                _dls.stat["armed"] = len(_armed)
            except Exception as _e:
                # NEVER take the cycle down for the fast loop. A controller that cannot be
                # reached simply stops being refreshed, and its trims EXPIRE at the trackers
                # (trim_ttl_s) rather than standing forever.
                _dls.stat["fail"] += 1
                _dls.stat["last_err"] = str(_e)
            _log_rl("fleet-trim",
                    "FLEET-TRIM %s: %d PRN(s) armed to %s, %d posts / %d failed%s"
                    % (log_tag() or args.signal, len(_armed), args.fleet_trim_url,
                       _dls.stat["posts"], _dls.stat["fail"],
                       ("  last err %s" % _dls.stat["last_err"])
                       if _dls.stat["last_err"] else ""),
                    every_s=30.0)
            # #76 THE READBACK -- close the loop this block opened. GET the controller's
            # standing trims right after handing it policy, so this cycle's view of "where
            # does the tracker's tap actually sit" is seed + trim rather than seed alone.
            # READ-ONLY: nothing here feeds control yet (that is #83 2(d)); it fills
            # _ft_readback and the log. On failure the dict is CLEARED, not held: a stale
            # trim served as current is the exact blindness this exists to remove, and
            # "missing = unknown" is the truthful state ([[chord-stale-artifacts]]).
            if args.fleet_trim_readback:
                try:
                    _rb = _get("%s/get_dll" % args.fleet_trim_url.rstrip("/"), timeout=2.0)
                    _rows = (_rb or {}).get(telem_chain) or {}
                    _dls.readback.clear()
                    for _p, _r in _rows.items():
                        if isinstance(_r, dict) and "trim_chips" in _r:
                            _dls.readback[int(_p)] = _r
                    _dls.stat["rb"] += 1
                    _log_rl("fleet-trim-rb",
                            "FLEET-TRIM READBACK %s: %s"
                            % (log_tag() or args.signal,
                               " ".join("%d:%+.3f%s"
                                        % (_p, _dls.readback[_p]["trim_chips"],
                                           "" if _dls.readback[_p].get("armed") else "(rel)")
                                        for _p in sorted(_dls.readback))
                               or "no standing trim"),
                            every_s=30.0)
                except Exception as _e:
                    # Same rule as the POST above: never take the cycle down for the fast
                    # loop. Counted and surfaced, and the dict stays empty until a poll
                    # succeeds again.
                    _dls.readback.clear()
                    _dls.stat["rb_fail"] += 1
                    _dls.stat["last_err"] = str(_e)
                # -- GAP 3 SHADOW (#33, READ-ONLY): does the row PREDICT the trim ramp? --
                # Predicted per-sat code-rate aiding = rrate[m/s] * f_chip / c  (identically
                # K*residual_doppler_hz, K = chip_rate/carrier ~ 0.0087). Measured = LS slope
                # of the STANDING trim from consecutive readbacks. The window RESETS on any
                # step > 0.3 chips (a #92 re-anchor/wipe is a discontinuity, not a rate) and
                # on release (a released trim's slope is the LEAK, not the sky). Mean trim is
                # logged beside the slope so the leak drag (leak_per_s * trim) stays
                # computable offline -- in the tracking regime the equilibrium terms cancel
                # in the derivative and the raw slope ~ gain/(gain+leak) of the sky rate.
                # Nothing here feeds control; the consume step (GAP 3 step 3) is a separate
                # default-off arm with its own pre-registration.
                try:
                    _t_rb = time.time()
                    _jg = None
                    try:
                        _jg = rx.joint_receiver(band_id, CODE_LEN,
                                                rereference=args.joint_rereference)
                    except Exception:
                        _jg = None
                    _g3_rows = []
                    for _p in sorted(_dls.readback):
                        _r = _dls.readback[_p]
                        if not _r.get("armed"):
                            # A RELEASED trim's slope is the LEAK, not the sky: drop the
                            # series rather than pausing it.
                            _g3_ramp.drop(_p)
                            continue
                        # #93: subtract the cumulative #92 handover deltas -- the
                        # gather applied them to the trim, so removing them makes the
                        # series continuous across re-bases (longer windows) and keeps
                        # sub-0.3-chip adjustments out of the slope.
                        _g3_ramp.update(_p, _t_rb,
                                        _handover.corrected(_p, float(_r["trim_chips"])))
                        _fit3 = _g3_ramp.fit(_p)
                        if _fit3 is None:
                            continue
                        _msl, _ym, _spn, _n = _fit3
                        _prd = None
                        if _jg is not None:
                            _k3 = (args.dr_constellation, int(_p))
                            if _jg.rrate_sigma(_k3) < 99.0:
                                _prd = _jg.rrate(_k3) * args.chip_rate_hz / C_LIGHT
                        # #93 v2: the DIRECT ADR span rate, converted carrier Hz ->
                        # code chips/s (K = f_chip/f_carrier ~ 0.0087). Fresh within
                        # 60 s or absent.
                        _av = _adr_span_now.get(_p)
                        _adr = ((_av[0] * args.chip_rate_hz / args.carrier_hz)
                                if (_av is not None and _t_rb - _av[1] <= 60.0)
                                else None)
                        _g3_rows.append((_p, _prd, _adr, _msl, _ym, _spn))
                    _g3_ramp.retain(_dls.readback)
                    if _g3_rows:
                        _log_rl("gap3-shadow",
                                "GAP3-SHADOW %s (chips/s; p=row a=ADR-span "
                                "m=trim slope @mean trim): %s"
                                % (log_tag() or args.signal,
                                   " ".join("%d:p%s/a%s/m%+.5f@%+.2f(%ds)"
                                            % (_p, ("%+.5f" % _pr)
                                               if _pr is not None else "--",
                                               ("%+.5f" % _ad)
                                               if _ad is not None else "--",
                                               _ms, _tmn, int(_sp))
                                            for _p, _pr, _ad, _ms, _tmn, _sp
                                            in _g3_rows)),
                                every_s=60.0)
                except Exception as _e:
                    _log_rl("gap3-shadow-err", "GAP3-SHADOW error (shadow only): %s" % _e,
                            every_s=300.0)


    def _stage_rate_feed_coarse():
        """#33 RATE FEED (coarse): feed the deep-rate residual to the joint receiver's per-sat rate state.
        
        ⚠️ THE REFERENCE IS THE WHOLE POINT. `deep_rate` is measured on records the tracker ALREADY
        derotated by the commanded trim, so what the search reports is only what REMAINS. The standing
        command is added back so `y` is referenced to the sky rather than to the current command --
        feeding the residual raw makes the estimator measure its own actuator (see #33 GAP 2, the
        mirror)."""
        if args.rrate_state and _rr2_resid and _drp.t_now_abs is not None:
            try:
                _jrr = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference)
                _n_ok = 0
                _n_gov = 0   # sats in the PHASE-GOVERNED regime this poll
                _n_rec_fed = 0
                for _p, _rv in sorted(_rr2_resid.items()):
                    # THE REFERENCE. deep_rate is measured on records the tracker already
                    # derotated by the commanded trim, so what the search reports is what
                    # REMAINS. Add the standing command back so y is referenced to the
                    # chain's BASE seed -- the same value regardless of what the last
                    # command happened to be. A frequency from a reference, never from an
                    # argument: feeding the bare residual would make every commanded sat
                    # read as "solved" and the filter would unlearn its own correction.
                    # ⚠️ ONLY on the command-AWARE plant (--rrate-feed-applied). On the
                    # folded assembler the observable never lost the command, and adding
                    # it back is the arm-12 integrator runaway. See the flag's help.
                    _y = _rv + (rr_cmd_applied.get(_p, _carrier.trim.get(_p, 0.0))
                                if args.rrate_feed_applied else 0.0)
                    _k = (args.dr_constellation, int(_p))
                    # FLL->PLL HANDOFF: a phase-governed satellite takes its coarse
                    # measurements at inflated sigma (see --rrate-coarse-deweight). Never
                    # SKIPPED -- the coarse feed is what re-acquires this row after a
                    # cycle slip, and a fine lock that has gone quiet expires on its own.
                    _sig_c = 0.2
                    # ---- PREFER THE RECORD-STREAM RATE (2026-08-14) -----------------
                    # deep_rate_full_hz is the DEEP FOLD's argmax, and its measured
                    # structure function is FLAT with lag -- rms change 1.44 m/s at 3 s
                    # rising to 2.07 at 24 s and no further. Flat means independent NOISE
                    # on every sample (a genuinely moving state grows as sqrt(lag)), so
                    # that feed carries ~1.4 m/s = 5.6 Hz of noise around a state that
                    # barely moves. The filter was right to reject 98% of it, and
                    # loosening q_rr would only have admitted the noise.
                    #
                    # fcoh's rate_hz fits the phase slope of the fleet-summed per-record
                    # series instead, with a split-half sigma measured the same way --
                    # 15-500 mHz on these satellites, 4-100x better. Same records, no
                    # extra poll. Falls through to the fold's value when the arc is too
                    # short to fit, so a thin poll degrades rather than starving the row.
                    # ⚠️ OFF (2026-08-14, measured): the record-stream fit is NOT better.
                    # Structure functions on the SAME satellites at the SAME instants, once
                    # the 2048x units bug was fixed:
                    #     lag      3 s    6 s   12 s   24 s
                    #     record  1.21   1.83   2.37   2.40   m/s
                    #     fold    1.26   1.87   2.28   2.15
                    # Indistinguishable. Both pick a peak from a spectrum over the same ~1 s
                    # window of the same records with deterministic algorithms, so when the
                    # peak selection goes wrong they go wrong TOGETHER -- correlated errors,
                    # identical structure functions.
                    #
                    # Its split-half sigma (1.13 Hz) said otherwise and was WRONG to trust:
                    # two halves of one window fitted on one grid both land on the same wrong
                    # peak and agree. Split-half measures PRECISION, never accuracy. The
                    # structure function at 3 s reads 1.21 m/s where 0.29 m/s of independent
                    # noise would give 0.41 -- that gap is the shared bias split-half cannot
                    # see.
                    #
                    # Left in place, unused, because rec_rate_hz remains a useful published
                    # diagnostic and because the next estimator worth trying is of a
                    # DIFFERENT KIND (the fine feed's res_cycles phase accumulation), not a
                    # better spectral fit over the same window. Set _use_rec to re-enable.
                    _use_rec = False
                    _fcr = (_dllp.fcoh or {}).get(_p) or {}
                    _rrec, _srec = _fcr.get("rate_hz"), _fcr.get("rate_sigma_hz")
                    if _use_rec and _rrec is not None and _srec is not None:
                        _y = _rrec + (rr_cmd_applied.get(_p, _carrier.trim.get(_p, 0.0))
                                      if args.rrate_feed_applied else 0.0)
                        # never claim better than the fold's grid can resolve, and never
                        # worse than the old blanket 0.2 -- a split-half of exactly 0 is
                        # two halves landing in one bin, not infinite precision.
                        _sig_c = min(max(_srec, 0.02), 0.2)
                        _n_rec_fed += 1
                    if (args.rrate_coarse_deweight > 1.0
                            and (t0 - rr_fine_t.get(_p, -1e9) <= args.rrate_fine_hold_s
                                 or t0 - rr_kcoh_t.get(_p, -1e9)
                                 <= args.rrate_fine_hold_s)):
                        _sig_c *= args.rrate_coarse_deweight
                        _n_gov += 1
                    if _jrr.update_rrate(_k, _y, _drp.t_now_abs, args.carrier_hz,
                                         sigma_hz=_sig_c) is not None:
                        _n_ok += 1
                _jrr.gauge_rrate()
                _rows = " ".join(
                    "%d:%+.2f+-%s" % (_p, _jrr.rrate((args.dr_constellation, int(_p))),
                                      ("%.2f" % _s if (_s := _jrr.rrate_sigma(
                                          (args.dr_constellation, int(_p)))) < 99.0
                                       else "inf"))
                    for _p in sorted(_rr2_resid))
                _log_rl("jrr",
                        "JRR[%s%s] rrate m/s: %s | f_car %+.3f+-%.3f Hz "
                        "(%d/%d accepted this poll%s; n=%d rej=%d)"
                        % (args.dr_constellation, "" if rr_full_ok else " CAPPED-FALLBACK",
                           _rows, _jrr.f_carrier(),
                           _jrr.f_carrier_sigma(), _n_ok, len(_rr2_resid),
                           ((", %d PHASE-GOVERNED" % _n_gov) if _n_gov else "")
                           + ((", %d/%d from RECORD STREAM" % (_n_rec_fed, len(_rr2_resid)))
                              if _n_rec_fed else ", fold-fed"),
                           _jrr.n_rrate, _jrr.rrate_rejected), every_s=60.0)
            except Exception as e:
                _log_rl("jrr-err", "rrate feed skipped: %s" % e, every_s=300.0)










    def _dr_joint_shadow():
        """3e-shadow: THE JOINT RECEIVER FEED (#33 GAP 2) -- feed per-sat offsets to the joint filter.
        
        Shadow by construction: it estimates, logs, and (where armed) publishes a state, but the
        seeding sub-stage below decides what actually reaches the trackers.
        
        ⚠️ FEEDING AND CONSUMING THE SAME QUANTITY IS A MIRROR, NOT A MEASUREMENT. The model-primary
        feed once measured its own seeded lock-and-arm rate (#33 GAP 2's nested self-reference); the
        spec-anchored feed exists so what goes in is not what this loop just put there.
        
        ⚠️ ITS `legacy clk` COLUMN IS ONE CYCLE STALE -- see the `_DrProducts` note. Do not compare it
        against the joint clock printed beside it without accounting for the lag."""
        if args.joint_shadow and _drp.offs:
            try:
                # P3: ONE state for the whole receiver, band carried as a
                # measurement LABEL rather than as a separate filter. Two per-band
                # states cannot estimate the offset between them, which is exactly
                # why the second band needed a hand-wired cross-band clock bootstrap
                # (#34) -- tau_band now has somewhere to live.
                # WARM START (2026-08-11): hand the filter the legacy clock at
                # CREATION. Born at 0 against a true ~151 chips, every measurement
                # lands beyond escape_max_step and the escape hatch refuses the
                # correction as non-physical -- 100% rejection, forever (observed
                # 15:25 UTC: 34 rejections agreeing to 0.22 chips, all implying
                # +151.7, all refused). Only read on creation, so this cannot fight
                # the filter once it is running. #28's lesson, applied one filter on.
                _js = rx.joint_receiver(band_id, CODE_LEN,
                                        clk0=float(dr_state.get("clk") or 0.0))
                # #33 gap 3: refresh the coupling constant on the SHARED object
                # every cycle rather than only at construction -- any consumer
                # site can be the creator (thread startup order), and a kwarg
                # passed only here would silently lose that race. Idempotent:
                # the flag lives in the common yaml section, same value from
                # every chain. A change of value is logged by the filter's own
                # F-matrix taking effect (0.0 = identity, no code path change).
                _js.rr_bsat_chips_per_m = float(args.rr_bsat_chips_per_m)
                # SNR GATE, caller-side. The broker's own comment 60 lines up
                # measures it: below --period-check-snr a detection's phase is
                # noise, "~2000-chip within-period residuals against a few chips
                # above it". The estimator this replaces was a circular MEDIAN and
                # shrugged those off; a mean-gauged filter cannot, and feeding them
                # ungated on 2026-08-09 walked the clock rate to -0.028 ppm and put
                # 17 chips/min of fictitious drift into every unlocked seed.
                _snr = {p: v[0] for p, v in best.items()}
                # P2c: withhold the masked sats, then ask the state where it thinks
                # they are. `predicted` is clk + b_i, so the residual below is exactly
                # "what the shared state got wrong about a satellite it is no longer
                # being told about".
                # FEED WARMUP (2026-08-12, the zombie's root): no measurements
                # until the establishment window has passed -- see the flag help.
                if time.time() - broker_t0 < args.joint_feed_warmup_s:
                    _log_rl("jwarm", "JFEED WARMUP: withholding the joint feed "
                            "(%.0f s of %.0f remain) -- establishment-phase "
                            "measurements must not become birth geometry"
                            % (args.joint_feed_warmup_s
                               - (time.time() - broker_t0),
                               args.joint_feed_warmup_s), every_s=60.0)
                else:
                    # ── #83 P3-3a: THE MODEL INNOVATION (MINNOV) ──
                    # The same residual P2C measures one coasted satellite at a
                    # time -- wrap(d - predicted - tau), against the PRIOR state,
                    # before this cycle's measurements are consumed -- computed
                    # continuously for every ESTABLISHED satellite: "could the
                    # joint state have placed this satellite without this
                    # detection?". This is the per-PRN model-primacy flip gate's
                    # number (INNOV judges the commanded seed; MINNOV judges the
                    # MODEL). Established-rows-only: a birth row's prediction is
                    # its own first measurement. SERVED ONLY (publisher + log).
                    for _p3, _d3 in _drp.offs:
                        _k3 = (_drp.tag, _p3)
                        if (_snr.get(_p3, 0.0) >= args.joint_min_snr
                                and _k3 in _js._idx
                                and _js._n.get(_k3, 0) >= args.joint_mask_after):
                            _mi = _js.wrap(_d3 - _js.predicted(_k3)
                                           - _js.tau(band_id))
                            _mh = minnov_hist.setdefault(_p3, [])
                            _mh.append((t0, _mi))
                            del _mh[:-120]
                    _js.cycle([((_drp.tag, p), d, args.joint_sigma, band_id)
                               for p, d in _drp.offs
                               if _snr.get(p, 0.0) >= args.joint_min_snr
                               and _track_ok(p)
                               and not _p2c_hold(_js, (_drp.tag, p))],
                              _drp.t_now_abs)
                # The filter has no logger; drain what it wants an operator to see.
                # An escape or an incoherent run is a tracking event worth a line --
                # on 2026-08-10 the single most damaging update of the day fired
                # completely silently and was only found by its consequences.
                for _n in _js.drain_notes():
                    _log_rl("joint-note", "JOINT %s: %s" % (band_id, _n),
                            every_s=10.0)
                _drained = True
                _p2c_tick(_js, _drp.t_now_abs)
                for _p, _d in _drp.offs:
                    if _p2c_hold(_js, (_drp.tag, _p)):
                        _r = _js.wrap(_d - _js.predicted((_drp.tag, _p)) - _js.tau(band_id))
                        if p2c["key"] == (_drp.tag, _p):
                            p2c["samples"].append((_drp.t_now_abs - p2c["t0"], _r))
                        _log_rl("p2c-%d" % _p,
                                "P2C %s PRN %d MASKED %.0fs: coast residual %+.3f chips "
                                "(b %+.3f, sigma %.3f, tau %+.4f) -- flat = the state "
                                "carries it"
                                % (band_id, _p, _js.age((_drp.tag, _p), _drp.t_now_abs) or 0.0,
                                   _r, _js.bias((_drp.tag, _p)), _js.sigma((_drp.tag, _p)),
                                   _js.tau(band_id)),
                                every_s=30.0)
                if _drp.now_w >= dr_state.get("joint_log_next", 0.0):
                    dr_state["joint_log_next"] = _drp.now_w + 30.0
                    # tau_band is reported WITH its observability count, never alone.
                    # It separates from b_sat only through satellites seen in BOTH
                    # bands, so a tau printed next to "dual 0" is an artefact of the
                    # priors -- which is precisely the state the disjoint E5a/E5b PRN
                    # lists put the instrument in while every chain looked healthy.
                    _tb = "".join(
                        "  tau[%s] %+.3f+-%.3f (dual %d)"
                        % (_b, _js.tau(_b), _js.tau_sigma(_b),
                           _js.tau_observability(_b))
                        for _b in sorted(_js._band_idx))
                    _amb = "  ⚠AMBIGUOUS(clk near wrap)" if rx.joint_ambiguous() else ""
                    _log("JOINT[shadow] " + _js.summary(_drp.t_now_abs) + _tb + _amb)
            except Exception as e:      # shadow must never take the broker down
                _log_rl("jointerr", "JOINT[shadow] disabled this cycle: %s" % e,
                        every_s=300.0)
        # -- P3: THE MODEL-PRIMARY MEASUREMENT (task #33) ------------------------
        # Everything above needs `offs`, which only a chain with DETECTIONS has. So
        # until now only GPS fed the joint state -- 4 shadow lines against 0 for every
        # model-primary chain -- and with one band contributing there was no second
        # band for tau_band to be a delay AGAINST. This is the other half of the same
        # measurement, and the plan has specified it since section 3a:
        #
        #     y_i = (where the replica actually is) - (pure model, no clock, no bias)
        #         = dr_seed_phys(seed) + dll_trim - cp_predicted
        #
        # dr_seed_phys is the tracker's own phase model evaluated at h1 (the same
        # function the slew block uses to ask "where does the tracker think the code
        # is"), dll_trim is the fleet loop's standing correction to it, and
        # cp_predicted is BRDC geometry alone. The difference satisfies the identical
        # y_i = clk + b_i + tau_band as the search-anchored form, which is what makes
        # two seeding disciplines poolable into one state at all.
        #
        # SEEDS ARE LAST CYCLE'S HERE, deliberately: this runs before the seed loop
        # rebuilds them, and "where the replica actually is" IS the seed the trackers
        # are flying right now. Reading a seed the broker has not yet shipped would
        # measure an intention rather than the instrument.
        #
        # No SNR gate to mirror -- a model-primary chain has no detection SNR. The
        # protection is the filter's own innovation gate plus birth_max, which is why
        # those were built with an escape hatch.
        elif args.joint_model_primary and args.joint_shadow and seeds and not _drp.offs:
            try:
                _js = rx.joint_receiver(band_id, CODE_LEN,   # warm start, see above
                                        clk0=float(dr_state.get("clk") or 0.0))
                _js.rr_bsat_chips_per_m = float(args.rr_bsat_chips_per_m)  # see 3a
                _h1 = int(round(_drp.t_now_abs * args.hops_per_sec))
                _th = _h1 / args.hops_per_sec
                _mm = []
                _fd_skip = 0
                for _prn, _sd in seeds.items():
                    _v = _drp.pd.get((_drp.tag, _prn))
                    if _v is None or "ref_hop" not in _sd:
                        continue
                    # THE GATE THIS FEED NEVER HAD. A dead-reckoned seed carries no
                    # detection SNR, which is why it was originally fed ungated --
                    # but "no detection SNR" is not "no quality signal": the tracker
                    # says plainly whether it is seeing anything. Without this the
                    # feed offers EVERY seeded satellite every cycle, including ones
                    # despreading pure noise, and on 2026-08-09 it walked clk to
                    # +445 against a true ~150 with 67-82% of updates rejected.
                    if not _track_ok(_prn):
                        continue
                    # ── THE DLL MUST BE IN ITS LINEAR RANGE (2026-08-22) ──────────
                    # y = held + trim - cp_predicted. The tracker despreads at S+T
                    # (seed + applied trim) and the DLL drives T so that S+T sits on
                    # the peak -- so S+T is SKY-ANCHORED and y is a real measurement:
                    # move S and the DLL moves T oppositely, leaving y invariant.
                    # THAT is why feeding and consuming on one chain is legitimate in
                    # principle, and it is the answer to "a Kalman filter is supposed
                    # to feed and consume".
                    # IT ONLY HOLDS WHILE THE DLL CAN DO ITS HALF. Past ~1 chip the
                    # correlation triangle has no gradient, so T stops compensating,
                    # S+T follows S, and y reports the consumer's own output -- the
                    # loop closes with no observation in it. Measured 2026-08-22:
                    # median |model - held| is 1.8-2.5 chips, i.e. ALREADY outside the
                    # measurable range for most satellites, which is why the feed
                    # diverged when it was armed with no gate at all.
                    # So feed only satellites the loop is demonstrably holding: q at
                    # the lock bar AND a trim well inside the clamp. A railed or
                    # near-railed trim means the seed error exceeds what the
                    # discriminator can see, and that satellite's y is not evidence.
                    # ── #85: SPEC ANCHORING ──────────────────────────────
                    # sky = held + applied_trim + spec_tau IDENTICALLY (the fit
                    # measures sky-minus-replica and the replica IS held+trim), so
                    # with a fresh, significant fit y is sky-anchored at ANY
                    # displacement -- including past the DLL's linear range, where
                    # the trim gate below would otherwise exclude the satellite.
                    # A consumed clock moving the seed moves trim+spec_tau
                    # oppositely; y is invariant. THAT is the mirror's removal.
                    _sp = ((dr_state.get("spec_y") or {}).get(_prn)
                           if args.joint_feed_spec else None)
                    _sp_ok = (_sp is not None
                              and _drp.t_now_abs - _sp[2]
                              <= args.joint_feed_spec_max_age_s
                              and _sp[1] >= args.joint_feed_min_ratio)
                    if args.joint_feed_max_trim > 0.0 and not _sp_ok:
                        _fl_i = (_dllp.fleet or {}).get(_prn) or {}
                        _q_i = _fl_i.get("q")
                        _tr_i = abs(float((_dls.readback.get(_prn) or {})
                                          .get("trim_chips") or 0.0))
                        if (_q_i is None or _q_i < args.lock_q
                                or _tr_i >= args.joint_feed_max_trim):
                            _fd_skip += 1
                            continue
                    _held = dr_seed_phys(_sd, _h1, args.hops_per_sec,
                                         args.chip_rate_hz, args.carrier_hz,
                                         args.code_doppler_sign, _drp.mod)
                    # ⚠️ THE TRIM THE TRACKER ACTUALLY APPLIED, not the one this
                    # process happens to hold (2026-08-21). Authority over the code
                    # trim is per-PRN: Python integrates only for PRNs the C++ fleet
                    # loop is NOT actuating (`if prn not in _ft_armed_last`), so on a
                    # chain the fast loop owns -- gal_e5a owns nearly all of it --
                    # dll_trim is ZERO for exactly the satellites carrying a real
                    # standing trim. Measured 15:01 on gal_e5a: C++ held 11:+1.601
                    # 19:+1.795 25:+2.920 29:+1.984 while dll_trim read +0.00/-0.03.
                    # So this measurement was systematically wrong by up to ~3 chips
                    # per satellite, and -- because the trims MOVE (PRN 11 3.000 ->
                    # 1.601, PRN 25 1.652 -> 2.920 inside a minute) -- their drifting
                    # mean was injected into the shared state as a spurious CLOCK
                    # RATE (~0.011 chips/s from the readback alone). A feed that does
                    # not know what the actuator did is measuring its own loop.
                    # #76's readback is exactly this number; it is one cycle old
                    # (populated later in the cycle), which is causal and correct.
                    _trim_applied = (
                        float((_dls.readback.get(_prn) or {}).get("trim_chips") or 0.0)
                        if _prn in _dls.armed_last
                        else _dls.trim.get(_prn, 0.0))
                    _y = ((_held + _trim_applied
                           + (_sp[0] if _sp_ok else 0.0)
                           - cp_predicted(_v, _th)) % _drp.mod)
                    if _p2c_hold(_js, (_drp.tag, _prn)):
                        if True:
                            _r = _js.wrap(_y - _js.predicted((_drp.tag, _prn)) - _js.tau(band_id))
                            if p2c["key"] == (_drp.tag, _prn):
                                p2c["samples"].append((_drp.t_now_abs - p2c["t0"], _r))
                            _log_rl("p2c-%d" % _prn,
                                    "P2C %s PRN %d MASKED %.0fs: coast residual %+.3f "
                                    "chips (b %+.3f, tau %+.4f)"
                                    % (band_id, _prn,
                                       _js.age((_drp.tag, _prn), _drp.t_now_abs) or 0.0,
                                       _r, _js.bias((_drp.tag, _prn)), _js.tau(band_id)), every_s=30.0)
                        continue
                    _mm.append(((_drp.tag, _prn), _y, args.joint_sigma, band_id))
                # ── #85: THE SET GATE. Eligibility is a property of the SET --
                # 1-2 measurements have spread ~ 0 by construction and a single
                # bad y IS the poll (the 01:xx degenerate feed). Withhold rather
                # than feed thin; the state coasts on clk_rate, which is exactly
                # what it is for.
                if _mm and len(_mm) < args.joint_feed_min_set:
                    _log_rl("jfeed-thin",
                            "JFEED %s: only %d satellite(s) qualify (< %d) -- "
                            "WITHHELD; a thin set feeds its own noise as clock"
                            % (band_id, len(_mm), args.joint_feed_min_set),
                            every_s=60.0)
                    _mm = []
                if _mm:
                    # ── JFEED: THE FORK THIS EXISTS TO SETTLE (task #33, 2026-08-11)
                    # Turning this feed on gave 9 Galileo b_sat all equal to +0.44
                    # (0.01 chip spread) against 8.9 chips across 3 GPS ones. Two
                    # explanations need opposite fixes and the log could not tell
                    # them apart, because cycle() returns only a COUNT and update()
                    # returns None on reject -- every per-satellite fact was thrown
                    # away at the call site:
                    #
                    #   (a) DEGENERATE AT SOURCE. y = dr_seed_phys(seed) + dll_trim
                    #       - cp_predicted, and on a model-primary chain the seed is
                    #       itself BUILT from model + clock -- so y may just report
                    #       back the clock that built it, carrying no per-sat
                    #       information. Self-reference: right in the mean, wrong in
                    #       the variance. Signature: spread(y - clk) ~ 0.
                    #   (b) DISTINCT BUT REJECTED. The measurements differ per sat,
                    #       the innovation gate throws them out, and b_sat stays
                    #       frozen at its birth value while clk drags it along.
                    #       Signature: spread(y - clk) is chips-scale, |r| large,
                    #       acc low.
                    #
                    # So log the RAW y, the innovation r the gate actually tests,
                    # and the accepted count -- all three before/around the update,
                    # since predicted() moves once cycle() runs. Costs one line per
                    # 10 s and touches no control flow.
                    _diag = None
                    if _drp.now_w >= dr_state.get("jfeed_log_next", 0.0):
                        dr_state["jfeed_log_next"] = _drp.now_w + 10.0
                        _diag = [(_k[1], _yy,
                                  _js.wrap(_yy - _js.predicted(_k) - _js.tau(_bd)),
                                  _js.bias(_k))
                                 for _k, _yy, _sg, _bd in _mm]
                        # TERM DECOMPOSITION. The innovation came back at a common
                        # -135 chips on every satellite with y ramping 0.15 chips/s
                        # (300x the clock rate), so the question is no longer "which
                        # satellite" but "which TERM of y". Print the three summands
                        # against the LEGACY offset the same chain is successfully
                        # seeding from -- E5a tracks fine at deep_snr 80+, so the
                        # legacy number is the working reference and y has to be
                        # compared to it, not to zero.
                        # ⚠️⚠️ THIS DIAGNOSTIC PRINTED THE WRONG VARIABLE UNTIL
                        # 2026-08-21 23:3x, AND IT NEARLY BOUGHT A WRONG CONCLUSION.
                        # It logged `dll_trim` -- the PYTHON dict -- while the actual
                        # measurement `_y` a few lines above uses `_trim_applied`,
                        # the C++ READBACK. For an armed PRN dll_trim is zero BY
                        # DESIGN (Python stands down; see #51), so the line read
                        # "dll_trim +0.000" on 22 of 27 samples and invited the
                        # reading "there is no sky term in y" -- when the fleet DLL
                        # was reporting disc +0.13..+0.51 chips at q>3 on those very
                        # satellites. A diagnostic that prints a different quantity
                        # than the code under test is worse than no diagnostic.
                        #
                        # WHAT THIS NOW ANSWERS, and it is the question KV put:
                        # a Kalman loop is SUPPOSED to feed and consume -- that is
                        # vector tracking -- and closing the loop is stabilising
                        # PROVIDED the fed-back quantity is an OBSERVATION. A
                        # discriminator is an observation (it correlates the replica
                        # against the sky). `held - cp_pred` is a difference of two
                        # PREDICTIONS and observes nothing. So the diagnosis turns
                        # entirely on HOW MUCH SKY IS IN y, which is what `trim` and
                        # `disc` measure. Print them, per satellite, next to the term
                        # they are supposed to correct.
                        # `fleet` carries the previous cycle's per-PRN disc (it is
                        # rebuilt later in this cycle) -- one cycle old, the same
                        # causality the readback already has, and guarded because it
                        # does not exist on the first pass.
                        try:
                            _dfl = _dllp.fleet
                        except NameError:
                            _dfl = {}
                        for _dk, _dyy, _dsg, _dbd in _mm[:4]:
                            _dsd = seeds.get(_dk[1]) or {}
                            _dv = _drp.pd.get(_dk)
                            if _dv is None or "ref_hop" not in _dsd:
                                continue
                            _dheld = dr_seed_phys(_dsd, _h1, args.hops_per_sec,
                                                  args.chip_rate_hz, args.carrier_hz,
                                                  args.code_doppler_sign, _drp.mod)
                            _dcp = cp_predicted(_dv, _th)
                            _darm = _dk[1] in _dls.armed_last
                            _dtrim = (float((_dls.readback.get(_dk[1]) or {})
                                            .get("trim_chips") or 0.0)
                                      if _darm else _dls.trim.get(_dk[1], 0.0))
                            _drow = _dfl.get(_dk[1]) or {}
                            _ddisc = _drow.get("disc")
                            _dq = _drow.get("q")
                            _log("JFEED-TERMS %s PRN %d [%s]: held %+.3f  "
                                 "trim_applied %+.4f (py %+.4f)  disc %s q %s  "
                                 "cp_pred %+.3f -> y %+.3f | legacy clk %+.3f + b "
                                 "%+.3f = %+.3f | joint clk %+.3f"
                                 % (band_id, _dk[1],
                                    "ARMED-cpp" if _darm else "python",
                                    _dheld, _dtrim, _dls.trim.get(_dk[1], 0.0),
                                    ("%+.4f" % _ddisc) if _ddisc is not None else "-",
                                    ("%.2f" % _dq) if _dq is not None else "-",
                                    _dcp,
                                    ((_dheld + _dtrim - _dcp) % _drp.mod),
                                    _drp.clk_now, bsat.get(_dk[1], _drp.now_w),
                                    _drp.clk_now + bsat.get(_dk[1], _drp.now_w), _js.clk))
                    _nok = _js.cycle(_mm, _drp.t_now_abs)
                    if _diag:
                        _ys = [_js.wrap(d[1] - _js.clk) for d in _diag]
                        _sp = max(_ys) - min(_ys)
                        _log("JFEED %s: %d meas, %d accepted (%.0f%%)  "
                             "spread(y-clk) %.4f chips  -> %s | %s"
                             % (band_id, len(_mm), _nok,
                                100.0 * _nok / max(1, len(_mm)), _sp,
                                "DEGENERATE (no per-sat info)" if _sp < 0.05
                                else "per-sat info PRESENT",
                                " ".join("%s%d y%+.3f r%+.3f b%+.3f"
                                         % (_drp.tag, p, y, r, b)
                                         for p, y, r, b in sorted(_diag))))
                if _drp.now_w >= dr_state.get("joint_log_next", 0.0):
                    dr_state["joint_log_next"] = _drp.now_w + 30.0
                    _tb = "".join(
                        "  tau[%s] %+.3f+-%.3f (dual %d)"
                        % (_b, _js.tau(_b), _js.tau_sigma(_b),
                           _js.tau_observability(_b))
                        for _b in sorted(_js._band_idx))
                    _amb = "  ⚠AMBIGUOUS(clk near wrap)" if rx.joint_ambiguous() else ""
                    _log("JOINT[shadow] " + _js.summary(_drp.t_now_abs) + _tb + _amb)
            except Exception as e:
                _log_rl("jointerr-mp",
                        "JOINT[shadow] model-primary feed skipped: %s" % e,
                        every_s=300.0)




    def _dr_seed():
        """3e-seed: BIRTH, SLEW or HAND BACK every visible satellite's seed from the model.
        
        The stage that actuates: it propagates the ephemeris and clock to a code phase and Doppler and
        writes the seed each tracker will despread with. It computes `_drp.clk_now`.
        
        ⚠️ NAME THE EPOCH OF EVERY TERM. Every walkoff this project has chased was born here, in a mix
        of two time bases: the F-engine axis against the wall clock (the 08-17 axis fix), and
        `cp_predicted` extrapolating from AXIS age while `predict_all` ran at WALL -- error = K*dop*lag,
        a defect born WITH the fix that preceded it (chord-trim-clamp-limit-cycle).
        
        ⚠️ A RE-BASE MOVES THE SEED WHILE THE C++ TRIM STILL CARRIES THE OLD CHIPS. That is #92's
        handover, and without it the tap leaves the sky and the trim rebuilds over ~25 minutes -- E3's
        sawtooth. The bound on it is a safety argument, not a tuning knob."""
        if dr_state["clk"] is not None:
            _drp.clk_now = (dr_state["clk"]
                       + _drp.drift * (_drp.now_w - dr_state["clk_t"])) % CODE_LEN
            # -- P2b CONSUMER "clk" (2026-08-11, the decay root's fix) -------------
            # clk_now above is the circular MEDIAN of per-sat offsets whose per-sat
            # biases span ~11 chips; with 4-7 sats in the solve, every membership
            # change steps it 1-2 chips, and the set churns on the search's revisit
            # timescale -- measured 22:20-22:50: a +-1-2 chip ~600 s oscillation
            # that dragged every model-primary seed off-peak together while the
            # JOINT filter (clk + per-sat b estimated jointly) read the same clock
            # FLAT to +-0.3 (gnss_broker_20260811_drclk.log, DRCLK vs JOINT).
            #
            # So on chains that opt in (`joint-consume: clk`), replace the LEVEL
            # with the joint clock, applied as a WRAPPED DELTA to the median (the
            # lesson of the slew consumer: the two agree only mod CODE_LEN, and the
            # legacy path keeps ownership of the long-code segment). Stateless per
            # cycle -- nothing is written back to dr_state, so a refused/degraded
            # joint falls back to the median seamlessly on the next cycle.
            #
            # GATES: --joint-min-sats in the state (a thin fleet lets one bias leak
            # into the clock), sigma bound (P grows while unfed, so ONE gate covers both
            # estimator health and staleness), and a delta bound against wrap
            # aliases / a diverged filter. The log line prints BOTH clocks every
            # time so the counterfactual median stays visible on the treated chain
            # -- a consumer whose firing cannot be seen is how this month's gates
            # failed.
            if "clk" in joint_consume:
                _jrC = _joint_state(rx, band_id, args)
                if _jrC is not None and len(_jrC._idx) >= args.joint_min_sats:
                    # sigma <= 0 is DEGENERATE (a zero-gain zombie claims perfect
                    # knowledge), not excellent -- refuse it explicitly. The first
                    # version wrote `sigma() or inf`, which also flipped a
                    # legitimate 0.0 to inf by truthiness accident; with the
                    # state_filter P floor sigma cannot reach 0 anymore, but this
                    # gate must not depend on that.
                    _jsigC = _jrC.sigma()
                    if _jsigC is None or _jsigC <= 0.0:
                        _jsigC = float("inf")
                    _jdC = ((_jrC.clk - _drp.clk_now + CODE_LEN / 2.0) % CODE_LEN
                            ) - CODE_LEN / 2.0
                    _jokC = (_jsigC <= args.joint_clk_max_sigma
                             and abs(_jdC) <= args.joint_clk_max_chips)
                    _log_rl("jclk",
                            "JOINT-CLK: legacy %.3f joint %.3f chips (delta %+.3f,"
                            " sigma %.3f, n %d) -> %s"
                            % (_drp.clk_now, _jrC.clk % CODE_LEN, _jdC, _jsigC,
                               len(_jrC._idx),
                               "ADOPTED" if _jokC else
                               "REFUSED (bounds %.1f chips / %.2f sigma)"
                               % (args.joint_clk_max_chips,
                                  args.joint_clk_max_sigma)),
                            every_s=30.0)
                    if _jokC:
                        _drp.clk_now = (_drp.clk_now + _jdC) % CODE_LEN
            # DRCLK (2026-08-11): the "dead-reckon clock ..." line above is gated on
            # `offs`, i.e. on this chain having its OWN detections -- so the four
            # model-primary chains, the ones whose seeds ride clk_now raw, never log
            # the clock they are consuming. This line is the missing term in the
            # commanded = cp_predicted + clk_now + b_sat + trim decomposition of the
            # per-sat disc walk (BSAT and DLL lines carry the other two). Logged
            # AFTER the joint-clk adoption on purpose: DRCLK is the clock the seeds
            # actually consume; the JOINT-CLK line above keeps the pre-adoption
            # median visible.
            _log_rl("drclk",
                    "DRCLK clk_now %.3f chips drift %+.4f chips/s (la %+.4f ppm)"
                    % (_drp.clk_now, dr_state.get("drift") or 0.0, _drp.la * 1e6),
                    every_s=30.0)
            # -- P2b CONSUMER 3, THE SHADOW ARM -------------------------------------
            # Compare the two offset ESTIMATORS directly, over every satellite the
            # joint state holds, independent of whether any seeding path ran this
            # cycle. The first cut logged inside the seeding loop and printed
            # nothing, because that loop skips `prn in best` -- every DETECTED
            # satellite -- while the joint state is fed from `best` and so holds
            # only detected satellites. The two sets are very nearly disjoint by
            # construction, and the overlap is only the handful in transition, so a
            # comparison gated on the loop measures the transition rate rather than
            # the estimators. An A/B must not depend on which code path happened to
            # execute.
            _jr3 = _joint_state(rx, band_id, args)
            if _jr3 is not None and _drp.now_w >= dr_state.get("jslew_log_next", 0.0):
                dr_state["jslew_log_next"] = _drp.now_w + 30.0
                _cmp = []
                for (_ct, _p) in list(_jr3._idx):
                    if _ct != _drp.tag:
                        continue
                    _lo = _drp.clk_now + bsat.get(_p, _drp.now_w)
                    _jo = _jr3.predicted((_ct, _p))
                    _dd = ((_jo - _lo + _drp.mod / 2.0) % _drp.mod) - _drp.mod / 2.0
                    _cmp.append((_p, _dd, _jr3.sigma((_ct, _p)) or 0.0))
                if _cmp:
                    _cmp.sort(key=lambda x: -abs(x[1]))
                    _log("SEED-OFFSET %s: joint-vs-legacy over %d sat(s), "
                         "median %+.3f chips | %s"
                         % (band_id, len(_cmp),
                            sorted(abs(c[1]) for c in _cmp)[len(_cmp) // 2],
                            " ".join("PRN%d %+.2f(s%.2f)" % c for c in _cmp[:6])))
            planned = []
            for (ctag, prn), v in sorted(_drp.pd.items()):
                # SIGNAL CAPABILITY first (see --dr-min-prn / --signal-capability): a
                # satellite that does not broadcast this signal must never be seeded,
                # however visible and however well-predicted it is. The model will
                # happily hand us a code phase for a signal that isn't there.
                if prn < dr_min_prn:
                    continue
                if _capable is not None and prn not in _capable:
                    continue  # block does not carry this signal (GPS L1C/L5/L2C)
                # +0.5 deg hysteresis vs the drop mask: a sat riding the mask
                # otherwise flickers drop->reseed every few cycles
                # #83 P3-3b: a MODEL-PRIMARY PRN is dr-owned even while the search
                # detects it -- that is the whole point of the flip. (It can never
                # be in cp_held: the flip's ENTER discards the hold.)
                # dr_untrusted is the LEGACY a0 integrity (EMA clock, no b_sat,
                # ~1-chip bar); a flipped sat's seed is the JOINT model's and its
                # referee is MINNOV -- gating the slew on the legacy flag would
                # orphan the sat seedless (its detections bypass re-anchor).
                if (ctag != _drp.tag or v["el"] < args.mask_deg + 0.5
                        or (prn in best and prn not in mp_flipped)
                        or prn in probe_set or prn in cp_held
                        or (prn in dr_untrusted
                            and prn not in mp_flipped)):  # model wrong for this sat
                    continue
                if prn in seeds and prn not in dr_state["seeded"]:
                    continue  # search-anchored coast: not ours to touch
                # TASK #30, LAYER 2: A LOCKED MODEL-PRIMARY SAT GETS A SLEWED REFRESH,
                # not a hold and not a repin. Both extremes are measured failures:
                #   HOLD  (search-fed logic on a chain with no search): the model's
                #         residual code rate is unmodeled, so the prompt drifts through
                #         the +-1 chip window in 3-11 min -- the rise-peak-fall deep_snr
                #         envelope, fleet disc railed +0.7, E >> L, lock lost, reseed.
                #   REPIN (8c7d176b3, reverted): re-anchoring from clk_now every 10 s
                #         injects the clock EMA's ~+-1 chip jitter as code-phase STEPS;
                #         median deep_snr fell 221 -> 17 and 21% of deep folds lost
                #         certification. Seed continuity beats seed freshness.
                # The slew keeps continuity AND tracks the model: converge the seed's
                # physical code phase toward the live model at <= _DR_SLEW_CAP chips
                # per cycle (0.025 chips/s at 2 s -- above every drift measured, an
                # order below the 0.25-chip DLL trims the GPS fold shrugs off), with
                # gain _DR_SLEW_K low-passing the clock EMA jitter the repin injected
                # raw. Search-FED chains keep the plain hold: their DLL owns code and
                # their search owns Doppler, and their digest is the control.
                # THE RE-PIN DECISION, and the one #58 makes wrong. `_held` adds a
                # fold-independent path: a satellite whose PROMPT is on the signal is
                # locked whatever the deep fold currently says about it. Without this
                # the fold's ~50%% certification flicker toggles lock state and
                # re-pins a satellite the despread never lost -- and the re-pin then
                # steps the phase reference the fold is trying to integrate across.
                # ⚠️ AND THE ONE METRIC THIS PROJECT ACTUALLY JUDGES LOCK ON.
                # The viewer's own q annotation says it: "Judge lock HERE, not on
                # sig/C-N0 -- the deep fold re-searches and will certify a satellite
                # whose prompt tap is on noise (#47). ~2.2 is the working lock bar."
                # This decision used neither q nor anything correlated with it:
                # sig_of is amp_snr (measured 0-1 on tracking satellites) upgraded by
                # a deep fold that certifies as little as 17% of polls, and
                # lock_prompt_hold wants 3x the FLEET MEDIAN prompt, which most of the
                # fleet cannot clear by construction (measured 0.76-1.61).
                # So both gates failed together, every cycle, and 24% of re-births
                # landed on satellites at q >= 2.2 -- five of them stepping the seed
                # 1.3-4.5 chips, i.e. clean outside the +-1 chip correlation triangle,
                # which destroys the very lock the gate exists to protect. PRN 13 was
                # thrown +2.42 chips at q 3.01 and +4.47 at q 2.24.
                _q_locked = (args.lock_q > 0.0
                             and _hold.q.get(prn, 0.0) >= args.lock_q)
                _held = (_q_locked
                         or (args.lock_prompt_hold > 0.0
                             and _hold.prev.get(prn, 0.0) >= args.lock_prompt_hold))
                if _held and sig_of(status.get(prn, {})) < args.lock_snr:
                    _log_rl("hold-prompt",
                            "HOLD-BY-PROMPT: PRN %d held through a deep-fold dropout "
                            "(prompt %.1fx noise, sig %.1f < %.1f) -- no re-pin"
                            % (prn, _hold.prev.get(prn, 0.0),
                               sig_of(status.get(prn, {})), args.lock_snr),
                            every_s=60.0)
                _slew = (prn in seeds
                         and (not detectors or prn in mp_flipped)
                         and prn in dr_state["seeded"]
                         and (sig_of(status.get(prn, {})) >= args.lock_snr or _held))
                if (not _slew and prn in seeds
                        and (sig_of(status.get(prn, {})) >= args.lock_snr or _held)):
                    continue  # sub-threshold LOCK: the DLL owns the residual now
                if (not _slew and prn in dr_state["seeded"]
                        and _drp.now_w - dr_state["pin"].get(prn, 0.0) < args.dr_repin_s):
                    continue
                # doppler + rate from BRDC range-rate (NOT the TLE pred: the BDS
                # TLE<->PRN mapping mismaps some birds, and BRDC is the precision
                # source anyway); clock_bias still comes from the TLE-vs-measured
                # solve -- it's a receiver constant, common to both models.
                v2 = _drp.pd2.get((ctag, prn))
                dop_geo = -v["range_rate_mps"] / dr_eph_mod.C_LIGHT * args.carrier_hz
                dop_seed = args.doppler_sign * dop_geo + _cb.value
                # ⚠️ THE HALVED DRATE (found 2026-08-22, the per-sat ramp's root).
                # This line predates the task #52 pair centring: when pd2 sat at
                # now_w+4 the /4.0 was a correct forward difference, but pd2 moved
                # to now_w+2 (and pd0 to now_w-2) and this consumer kept /4.0 over
                # a 2 s span -- so every model-primary seed's doppler_rate_hz_s was
                # EXACTLY HALF the truth. The missing half leaks into held-vs-model
                # as (f_chip/f_c) * (drate/2) * T/2 per re-pin interval T: measured
                # ramp = 0.0549 chips/s per Hz/s of drate over Galileo (r=+0.68,
                # 29 chain-PRNs), implying T ~ 25 s -- the observed re-pin cadence.
                # drate < 0 for EVERY satellite (Doppler falls through a pass), so
                # the per-sat ramps all share a sign and pooled they masqueraded as
                # a constellation-common drift. The centred pair is what :4670 (the
                # dop_model seed slot) already uses; this is the same fix, same
                # first-cycle fallback.
                drate = 0.0
                v0 = (dr_state.get("pd0") or {}).get((ctag, prn))
                if v2 is not None and v0 is not None:
                    drate = (args.doppler_sign
                             * (-(v2["range_rate_mps"] - v0["range_rate_mps"]) / 4.0)
                             / dr_eph_mod.C_LIGHT * args.carrier_hz)
                elif v2 is not None:
                    drate = (args.doppler_sign
                             * (-(v2["range_rate_mps"] - v["range_rate_mps"]) / 2.0)
                             / dr_eph_mod.C_LIGHT * args.carrier_hz)
                # inverse of cp_loc above: physical cp -> sample-0 cp0 removes the
                # nominal advance AND the code-Doppler drift (the seed currency)
                # + b_sat (task #33): the per-sat slow bias the P1 fit measures --
                # iono/tropo/BRDC, the term holding seeds at a model that is right
                # about the orbit and wrong about the path. tau is sky-minus-replica,
                # so the replica moves BY tau to meet the sky: phys + b. Zero until
                # the fit has fed it (and exactly 0.0 in every transcript replay).
                # -- P2b CONSUMER 3: the clock+bias offset, ONE definition for both
                # the birth/re-pin phase (cp0, below) and the slew target (_model,
                # further down). They are the same physical quantity -- "what to add
                # to the pure model for this satellite" -- and must not be able to
                # disagree.
                #
                # ⚠️ WHICH CHAINS REACH WHICH PATH, because it decides what this
                # consumer can be tested on at all. The slew path requires `not
                # detectors`, so ONLY the model-primary chains (E5a/E5b/B2a/B2b)
                # slew; search-fed GPS takes cp0 at birth and re-pin and is then
                # skipped entirely once locked ("the DLL owns the residual now").
                # But the joint state is fed ONLY by search-anchored chains, so it
                # contains ONLY GPS satellites unless --joint-model-primary is on.
                # Scoped to the slew path alone this consumer could therefore NEVER
                # FIRE: the chains that slew have no satellites in the state, and
                # the chain with satellites in the state does not slew. Covering
                # cp0 as well is what gives it a live arm today, on GPS births.
                if _drp.hold:
                    continue      # clock is still a prime; see the withhold note
                _leg_off = _drp.clk_now + bsat.get(prn, _drp.now_w)
                _off = _leg_off
                _off_sigma = None      # set only when a JOINT offset is adopted
                _jr3 = _joint_state(rx, band_id, args)
                _joff = (_jr3.predicted((_drp.tag, prn))
                         if (_jr3 is not None and (_drp.tag, prn) in _jr3._idx)
                         else None)
                if _joff is not None:
                    # ⚠️ WRAP AT THE MODULUS THE TWO ACTUALLY SHARE, AND APPLY THE
                    # RESULT AS A DELTA. clk_now is reduced mod CODE_LEN (10230 for
                    # L5) and so is the joint clk, but the SEED lives mod _DR_MOD
                    # (204600 with --dr-long-code -- 20 primary periods). Three
                    # consequences, all learned the hard way today:
                    #   * wrapping the difference at _DR_MOD does not fold out a
                    #     CODE_LEN wrap, so when clk_now crosses zero the shadow
                    #     read -10058 chips of "disagreement" on every satellite at
                    #     once (observed 16:58, and the giveaway was that it was
                    #     COMMON to all six -- a per-sat estimator error cannot be);
                    #   * the plausibility bound would then REFUSE the joint offset
                    #     for the whole wrap neighbourhood -- a gate failing
                    #     periodically for a reason unrelated to estimator health;
                    #   * and SUBSTITUTING _joff for _leg_off would displace the
                    #     seed by a whole primary period, because the two are only
                    #     equivalent mod CODE_LEN while the seed cares mod _DR_MOD.
                    # So the joint state contributes only the SMALL correction it
                    # actually measures, and the legacy path keeps ownership of
                    # which long-code segment we are in.
                    _d3 = ((_joff - _leg_off + CODE_LEN / 2.0) % CODE_LEN) - CODE_LEN / 2.0
                    _ok3 = abs(_d3) <= args.joint_slew_max_chips
                    _log_rl("jslew-%d" % prn,
                            "SEED-OFFSET PRN %d (%s): joint %+.3f vs legacy %+.3f "
                            "chips (diff %+.3f mod %.0f, sigma %.3f)%s"
                            % (prn, "slew" if _slew else "cp0", _joff, _leg_off,
                               _d3, CODE_LEN, _jr3.sigma((_drp.tag, prn)) or 0.0,
                               "" if _ok3 else "  REFUSED (> %.1f chips)"
                               % args.joint_slew_max_chips),
                            every_s=60.0)
                    if "slew" in joint_consume and _ok3:
                        _off = _leg_off + _d3
                        # ...and how well we know it, for the rate limit below.
                        _off_sigma = _jr3.sigma((_drp.tag, prn))
                cp0 = ((cp_predicted(v, _drp.t_fc_abs) + _off)
                       - _drp.t_fc_abs * args.chip_rate_hz
                         * (1.0 + args.code_doppler_sign
                            * dop_seed / args.carrier_hz)) % _drp.mod
                if args.dr_dry_run:
                    planned.append("PRN %d el %.0f cp0 %.1f dop %+.0f rate %+.2f"
                                   % (prn, v["el"], cp0, dop_seed, drate))
                    continue
                if _slew:
                    # Where the TRACKER's propagation puts this seed right now
                    # (dr_seed_phys inverts the currency + extrapolation exactly --
                    # round-trip 1e-4 chips, re-anchor continuity 0.0 even across a
                    # Doppler change; see the selftest). The model side is the same
                    # sum the birth cp0 uses, BEFORE back-referencing.
                    #
                    # ⚠️ EVERY TERM AT THE SAME EPOCH: t_h = h1/hps, the ROUNDED
                    # hop's time -- never t_now_abs. The physical code phase runs at
                    # ~52.4 chips per hop, so comparing a model evaluated at
                    # t_now_abs against a held phase evaluated at h1/hps injects
                    # (t_now_abs*hps - h1) * 52.4 = up to +-26 chips of PHANTOM
                    # difference from the sub-hop rounding alone. Caught on the
                    # first live cycle (2026-08-09 23:04): every fresh B2a seed
                    # reported model-held = +23.335 chips two seconds after birth,
                    # identical across PRNs -- rounding, wearing the clock's
                    # common-mode costume. (Birth gets away with mixing t_now_abs
                    # and a rounded ref_hop because there the SAME t builds the
                    # subtraction, so the skew cancels against the tracker's
                    # back-reference to first order. A cross-epoch DIFFERENCE has no
                    # such cancellation.)
                    h1 = int(round(_drp.t_fc_abs * args.hops_per_sec))
                    t_h = h1 / args.hops_per_sec
                    _held = dr_seed_phys(
                        seeds[prn], h1, args.hops_per_sec, args.chip_rate_hz,
                        args.carrier_hz, args.code_doppler_sign, _drp.mod)
                    # The clock+bias offset is _off, computed once above so the birth
                    # phase and the slew target cannot disagree. b_sat is "how wrong
                    # the pure model is for THIS satellite", which is why this is the
                    # consumer aimed at the ~600 s plant oscillation (slew-to-model
                    # fighting trim-to-sky, with the model per-sat +-1-6 chips out).
                    _model = (cp_predicted(v, t_h) + _off) % _drp.mod
                    _dcp = ((_model - _held + _drp.mod / 2.0) % _drp.mod
                            ) - _drp.mod / 2.0
                    # ── THE RATE LIMIT, AND WHY IT IS THE WHOLE STORY ──
                    # _DR_SLEW_CAP is 0.05 chips per event and 47% of steps sat
                    # exactly on it: satellites 5-8 chips from their model were
                    # being corrected at 0.05 a go, every 10-30 s -- of order an
                    # HOUR to close, against a ~600 s oscillation. The seed can
                    # never reach the model within a cycle, so the plant is a
                    # rate-limited integrator chasing a moving multi-chip error: a
                    # limit cycle almost by construction, and a better account of
                    # the ~600 s oscillation than "slew fights trim".
                    #
                    # It also made P2b's slew consumer UNOBSERVABLE. For a
                    # satellite already railed at the cap, moving the target by
                    # +3-8 chips leaves the step at +-0.05 in the same direction --
                    # identical seed behaviour before and after, which is exactly
                    # what the viewer showed. Measured 2026-08-11 16:52, and
                    # visible in the FIRST log line read that morning (PRN 35:
                    # model-held -10.571 chips, step -0.050). Dividing the two
                    # numbers in any log line would have found it.
                    #
                    # ACQUISITION AUTHORITY, TRACKING RESTRAINT. A seed that is
                    # far from its target needs the authority to pull in; one that
                    # has arrived must not be yanked around by noise. So the
                    # ceiling slides with DISTANCE, and drops back to the flat cap
                    # once the seed is within dr_slew_near_chips.
                    #
                    # KEYED ON DISTANCE, NOT ON TRACK AGE, though "new satellites
                    # slew hard, established ones tighten" is the behaviour either
                    # would give -- because a new seed IS the far one. Distance
                    # re-arms and age does not, and that difference is the whole
                    # point here: the ~600 s oscillation happens on ESTABLISHED
                    # tracks that have drifted off. An age schedule would tighten
                    # exactly those satellites and then deny them the correction
                    # they need, locking in the limit cycle it was meant to break.
                    #
                    # SIGMA IS A TRUST GATE, NOT A MULTIPLIER. The first cut scaled
                    # the cap as n x sigma, borrowed from the escape guard -- but
                    # there sigma bounds the size of a CLAIM, while here it is the
                    # precision of a KNOWN target, so proportional scaling made a
                    # well-measured offset move SLOWER. Inverted. What sigma
                    # actually decides is whether to trust the target enough to
                    # move fast toward it at all; past dr_slew_trust_sigma we keep
                    # the crawl. An offset with no sigma (no joint state, or
                    # refused) is untrusted by construction.
                    _cap = _drp.slew_cap
                    if (args.dr_slew_cap_acq > _drp.slew_cap
                            and _off_sigma is not None
                            and math.isfinite(_off_sigma)
                            and _off_sigma <= args.dr_slew_trust_sigma
                            and abs(_dcp) > args.dr_slew_near_chips):
                        _cap = args.dr_slew_cap_acq
                    _step = max(-_cap, min(_cap, _drp.slew_k * _dcp))
                    # Re-anchor at h1 with the FRESH model doppler/rate (kills the
                    # dt^2 linearization error a held seed accumulates) but at the
                    # HELD phase plus the bounded step -- never at the model's own
                    # phase, which carries the clock EMA jitter raw. NOT popping
                    # dll_trim: same trajectory, later epoch; the trim's residual is
                    # still valid (the lesson of the reverted repin).
                    seeds[prn] = Seed.born(
                        "dr_slew", epoch=h1,
                        doppler_hz=dop_seed,
                        code_phase_chips=dr_cp0(
                            _held + _step, t_h, dop_seed,
                            args.chip_rate_hz, args.carrier_hz,
                            args.code_doppler_sign, _drp.mod),
                        code_phase_rate=cp_rate_from_code_bias(
                            dop_seed, _drp.la, args.hops_per_sec,
                            args.chip_rate_hz, args.carrier_hz),
                        ref_hop=h1, doppler_rate_hz_s=drate)
                    # #45 STEP 6: ship the PHASE as well. propagate_seed prefers it
                    # and it carries no sample-0 lever, so a later dop edit cannot
                    # desynchronise the pair (#42's writer, #44's coast). Both are
                    # emitted so a tracker that ignores the field is unaffected.
                    if args.seed_phase_transport:
                        seeds[prn].put(
                            "phase_xport", epoch=h1,
                            code_phase_at_ref_chips=seed_phase_at_ref(
                                _held + _step, dop_seed, args.chip_rate_hz,
                                args.hops_per_sec, args.carrier_hz,
                                args.code_doppler_sign, _drp.mod,
                                args.search_fft_len or None))
                    dr_state["pin"][prn] = _drp.now_w
                    _log_rl("drslew-%d" % prn,
                            "dead-reckon SLEW PRN %d: model-held %+.3f chips, "
                            "step %+.3f (cap %.2f), dop %+.0f rate %+.2f"
                            % (prn, _dcp, _step, _cap, dop_seed, drate),
                            every_s=120.0)
                    continue
                if prn not in seeds:
                    _log("dead-reckon SEED PRN %d (elev %.0f, cp0 %.1f, dop %+.0f,"
                         " rate %+.2f)" % (prn, v["el"], cp0, dop_seed, drate))
                _dls.trim.pop(prn, None)  # any old trim served the OLD anchor
                _dls.last.pop(prn, None)
                _rh_birth = int(round(_drp.t_fc_abs * args.hops_per_sec))
                # ── BIRTH-STEP DECOMPOSITION (2026-08-22) ────────────────────────
                # Measured overnight: when several satellites are re-born in the SAME
                # cycle they all step by the SAME amount -- E5 +142.84, E13 +142.76,
                # E16 +142.78, E26 +142.80, E31 +142.87 chips, agreeing to 0.11 chips
                # (0.08%) with completely unrelated ddop (-0.76..-6.96 Hz) and seed
                # ages. Nothing per-satellite can do that: it is ONE SHARED CONSTANT
                # entering every seed at once, and its size is the receiver clock.
                # Three candidate explanations were falsified BEFORE this line was
                # written, which is why it logs terms rather than a verdict:
                #   * a rate rewrite under a stale anchor -> implied rate change
                #     0.44 chips/s vs the 0.001-0.02 the filter carries, and
                #     r(|step|, seed age) = -0.11 when the mechanism needs
                #     proportionality;
                #   * a Doppler edit through the cp-currency lever -> r = +0.08,
                #     step-equivalent 0.055 Hz against an actual ddop of 4.08 Hz;
                #   * a feedback loop -> the steps are discrete and SHARED, and a
                #     loop cannot synchronise independent satellites to 0.08%.
                # So: print where the OLD seed said the signal was, where the NEW one
                # puts it, and every term that differs between the two branches. The
                # slew branch anchors on `_held` (the TRACKER's own propagation); this
                # branch anchors on model + `_off`. If the clock term is what flips,
                # `step` lands on `_off` (or on `_d3`, the joint-vs-legacy correction
                # that comes and goes with ADOPTED/REFUSED) and the line says so
                # outright instead of leaving it to be inferred from a magnitude.
                _pv = seeds.get(prn)
                if _pv is not None and "ref_hop" in _pv:
                    try:
                        _oldphys = dr_seed_phys(_pv, _rh_birth, args.hops_per_sec,
                                                args.chip_rate_hz, args.carrier_hz,
                                                args.code_doppler_sign, _drp.mod)
                        _newphys = (cp_predicted(v, _drp.t_fc_abs) + _off) % _drp.mod
                        _bstep = ((_newphys - _oldphys + _drp.mod / 2.0) % _drp.mod
                                  ) - _drp.mod / 2.0
                        # ── #92 THE HANDOVER (--fleet-trim-rebase-adjust) ──
                        # The seed is about to move by _bstep while the C++
                        # standing trim carries the SAME chips: post the
                        # compensating -_bstep to the gather in the SAME cycle
                        # so the tap (seed + trim) never leaves the sky (E3's
                        # ~25-min sawtooth). The bound, the transport and the
                        # cumulative-delta bookkeeping are in
                        # gnss_broker/handover.py. Gated on _ft_armed_last: a
                        # PRN the fleet loop is not actuating has no standing
                        # trim to hand over, and posting for seeding churn
                        # buries the one refusal that would MEAN something.
                        _handover.offer(prn, _bstep, prn in _dls.armed_last,
                                        telem_chain, args.fleet_trim_url,
                                        _post, _log)
                        _log_rl("birthstep-%d" % prn,
                                "BIRTH-STEP PRN %d: old_phys %+.3f -> new_phys %+.3f"
                                "  step %+.3f chips | off %+.3f = leg %+.3f"
                                " (clk %+.3f + b %+.3f) %s |"
                                " age %.1f s ddop %+.3f Hz"
                                " | WHY-BIRTH: sig_of %.2f vs lock_snr %.1f,"
                                " hold_prev %.2f vs %.1f -> held %s;"
                                " in_seeds %s in_seeded %s | prev [%s]"
                                % (prn, _oldphys, _newphys, _bstep, _off, _leg_off,
                                   _drp.clk_now, bsat.get(prn, _drp.now_w),
                                   ("+ d3 %+.3f [joint %s]"
                                    % (_d3, "ADOPTED" if _ok3 else "REFUSED"))
                                   if _joff is not None else "[joint absent]",
                                   (_rh_birth - int(_pv["ref_hop"])) / args.hops_per_sec,
                                   dop_seed - float(_pv.get("doppler_hz", dop_seed)),
                                   # ⚠️ WHY WAS THE SLEW BRANCH NOT TAKEN? Reaching
                                   # this line means `_slew` was False for a satellite
                                   # the tracker may well be holding -- and on
                                   # 2026-08-22 that cost E13 a 3.75-chip snap off the
                                   # correlation triangle and 28 minutes of q < 1.
                                   # sig_of() is amp_snr, or max(amp, deep) ONLY when
                                   # the deep fold certifies (coherence_s > 0) -- and
                                   # measured live, amp_snr sits at 0-1 against a
                                   # lock_snr of 3.0 while the fold certifies as
                                   # little as 17% of polls. So print every input to
                                   # the decision rather than inferring which failed.
                                   sig_of(status.get(prn, {})), args.lock_snr,
                                   _hold.prev.get(prn, 0.0), args.lock_prompt_hold,
                                   _held, prn in seeds,
                                   prn in dr_state["seeded"],
                                   _pv.owners() if hasattr(_pv, "owners") else "?"),
                                every_s=20.0)
                    except Exception as _e:      # diagnostics never break seeding
                        _log_rl("birthstep-err", "BIRTH-STEP unavailable: %s" % _e,
                                every_s=300.0)
                seeds[prn] = Seed.born(
                    "dr_birth", epoch=_rh_birth,
                    doppler_hz=dop_seed, code_phase_chips=cp0,
                    code_phase_rate=cp_rate_from_code_bias(
                        dop_seed, _drp.la, args.hops_per_sec,
                        args.chip_rate_hz, args.carrier_hz),
                    ref_hop=_rh_birth,
                    doppler_rate_hz_s=drate)
                # #45 STEP 6, birth/re-pin arm. cp0 was just built FROM this phase
                # (cp_predicted + _off at t_now_abs), so shipping it costs nothing
                # and removes the round trip the tracker would otherwise redo.
                if args.seed_phase_transport:
                    seeds[prn].put(
                        "phase_xport", epoch=_rh_birth,
                        code_phase_at_ref_chips=seed_phase_at_ref(
                            (cp_predicted(v, _drp.t_fc_abs) + _off) % _drp.mod,
                            dop_seed, args.chip_rate_hz, args.hops_per_sec,
                            args.carrier_hz, args.code_doppler_sign, _drp.mod,
                            args.search_fft_len or None))
                dr_state["seeded"].add(prn)
                dr_state["pin"][prn] = _drp.now_w
            if planned:
                _log("dead-reckon DRY RUN, would seed: %s" % "; ".join(planned))
            # model-owned sats drop on the BRDC elevation (they're exempt from
            # the TLE horizon drop -- see the coast loop), or on capability
            for prn in list(dr_state["seeded"]):
                v = _drp.pd.get((_drp.tag, prn))
                if prn < dr_min_prn or (_capable is not None and prn not in _capable):
                    _log("dead-reckon drop PRN %d (does not broadcast this signal)" % prn)
                    seeds.pop(prn, None)
                    _hold.low_hits.pop(prn, None)
                elif v is None or v["el"] < args.mask_deg:
                    _log("dead-reckon drop PRN %d (set below BRDC horizon)" % prn)
                    seeds.pop(prn, None)
                    _hold.low_hits.pop(prn, None)


    def _stage_nav_bits():
        """NAV BITS: decode the broadcast navigation message and cross-check it against BRDC.
        
        Holds ELEVEN lazily-created decoder objects (LNAV, CNAV, CNAV2, FNAV, INAV, BCNAV1/2/3, the
        BRDC source and the agreement monitor). They are created on the first row that needs them and
        must persist across cycles -- hence the nonlocal list, which is the whole reason this block
        resisted extraction until its state was named.
        
        ⚠️ OFF IN PRODUCTION (`--nav-bits` is not set in the CHORD yaml), so the fixture gate is BLIND
        to every line below it. Changes here are unverified until the flag is armed."""
        nonlocal bcnav1, bcnav2, bcnav3, cnav, cnav2, fnav, inav, navbits, navbits_log_t, navbrdc, navhealth
        if args.nav_bits:
            for _p, _r in status.items():
                if "nav_obs" not in _r:
                    continue
                if navhealth is None:
                    from navbit_health import BitAgreement
                    navhealth = BitAgreement(log=_log)
                # Route nav_obs to the decoder this band actually speaks. L1CA carries LNAV
                # (periodic subframes -> future bits); L2C-CM / L5-I carry CNAV (FEC+CRC, no
                # fixed schedule -> decode + shadow-serve decoded spans). Sending CNAV symbols
                # to the LNAV frame-sync just spins forever finding no preamble.
                if args.nav_decoder == "cnav":
                    if cnav is None:
                        from cnav_predictor import CnavPredictor
                        cnav = CnavPredictor(log=_log)
                        _log("CNAV decoder armed (combiner exports nav_obs)")
                    cnav.ingest(_p, _r["nav_obs"])
                elif args.nav_decoder == "bcnav3":
                    # BeiDou B2b B-CNAV3: the PRIMARY chain's own nav decoder (NB-LDPC block, not a
                    # bit-prediction scheme -> no peel role, pure decode + BRDC xcheck).
                    if bcnav3 is None:
                        from bcnav3_predictor import Bcnav3Predictor
                        bcnav3 = Bcnav3Predictor(log=_log)
                        _log("B-CNAV3 decoder armed (combiner exports nav_obs)")
                    bcnav3.ingest(_p, _r["nav_obs"])
                else:
                    if navbits is None:
                        from navbit_predictor import LnavPredictor
                        navbits = LnavPredictor(log=_log)
                        _log("nav-bit predictor armed (combiner exports nav_obs)")
                    navbits.ingest(_p, _r["nav_obs"])
                # ...and ask the question no single gate was asking continuously (decoder-
                # agnostic): did the bits we PUBLISHED last cycle actually match the ones coming
                # out of the sky? Scored only where the satellite is strong enough that a
                # disagreement means OUR bits are wrong rather than the air reading being wrong.
                navhealth.score(_p, _r["nav_obs"], _r.get("deep_snr"))
            # AUXILIARY CNAV CHAIN (--cnav-combiner, S4): a second combiner polled only for its
            # CNAV symbols. At L5 this broker's own combiner is the Q PILOT -- its nav_obs are
            # deterministic overlay predictions handled above -- while CNAV arrives on the
            # derived L5-I data sibling, which no broker otherwise reads. Failure here is
            # deliberately NON-FATAL and quiet-ish: this is an observability path, and a chain
            # that has not acquired yet simply returns nothing. It must never disturb the
            # tracking loop that shares this cycle.
            if cnav_combiner:
                try:
                    _aux = {int(r["prn"]): r for r in _get("%s/get_status" % cnav_combiner)
                            if r.get("prn")}
                except Exception as e:
                    _aux = {}
                    _log_rl("cnavaux", "cnav aux combiner %s unreadable: %s"
                            % (cnav_combiner, e))
                for _p, _r in _aux.items():
                    if "nav_obs" not in _r:
                        continue
                    if cnav is None:
                        from cnav_predictor import CnavPredictor
                        cnav = CnavPredictor(log=_log)
                        _log("CNAV decoder armed on aux chain %s" % cnav_combiner)
                    cnav.ingest(_p, _r["nav_obs"])
            # S5 D-component #1: Galileo E1B I/NAV, the exact CNAV-aux analogue.
            if inav_combiner:
                try:
                    _iaux = {int(r["prn"]): r for r in _get("%s/get_status" % inav_combiner)
                             if r.get("prn")}
                except Exception as e:
                    _iaux = {}
                    _log_rl("inavaux", "inav aux combiner %s unreadable: %s"
                            % (inav_combiner, e))
                for _p, _r in _iaux.items():
                    if "nav_obs" not in _r:
                        continue
                    if inav is None:
                        from inav_predictor import InavPredictor
                        inav = InavPredictor(log=_log)
                        _log("I/NAV decoder armed on aux chain %s" % inav_combiner)
                    inav.ingest(_p, _r["nav_obs"])
                # 60 s health + BRDC cross-check (Kepler only; alm_sys is 'E' for the GAL broker).
                # I/NAV rides E1B on L1 and E5b-I on the mid band -- SAME message + decoder; label
                # the decode-health obs by the actual carrier (from the aux combiner name).
                if inav is not None and _now() - _inav_log_t[0] > 60.0:
                    _inav_log_t[0] = _now()
                    _inav_sig = "GAL_E5BI_INAV" if "e5b" in inav_combiner else "GAL_E1B_INAV"
                    for _p in sorted(inav._p):
                        h = inav.health(_p)
                        if not h or not h["words"]:
                            continue
                        eph = inav.ephemeris(_p)
                        xc = (_inav_brdc_xcheck(brdc_alm, alm_sys, _p, eph, _log)
                              if (eph is not None and brdc_alm is not None) else "")
                        _log("inav PRN %d: %d pages, %d words, have %s, eph %s%s"
                             % (_p, h["pages"], h["words"], h["have"],
                                "YES" if eph is not None else "no", xc))
                        _dh_obs(_inav_sig, _p, h, eph, xc)
            # S5 D-component #2: Galileo E5a-I F/NAV, the exact I/NAV-aux analogue on L5.
            if fnav_combiner:
                try:
                    _faux = {int(r["prn"]): r for r in _get("%s/get_status" % fnav_combiner)
                             if r.get("prn")}
                except Exception as e:
                    _faux = {}
                    _log_rl("fnavaux", "fnav aux combiner %s unreadable: %s"
                            % (fnav_combiner, e))
                for _p, _r in _faux.items():
                    if "nav_obs" not in _r:
                        continue
                    if fnav is None:
                        from fnav_predictor import FnavPredictor
                        fnav = FnavPredictor(log=_log)
                        _log("F/NAV decoder armed on aux chain %s" % fnav_combiner)
                    fnav.ingest(_p, _r["nav_obs"])
                # 60 s health + BRDC cross-check (Kepler only; alm_sys is 'E' for the GAL broker)
                if fnav is not None and _now() - _fnav_log_t[0] > 60.0:
                    _fnav_log_t[0] = _now()
                    for _p in sorted(fnav._p):
                        h = fnav.health(_p)
                        if not h or not h["words"]:
                            continue
                        eph = fnav.ephemeris(_p)
                        xc = (_fnav_brdc_xcheck(brdc_alm, alm_sys, _p, eph, _log)
                              if (eph is not None and brdc_alm is not None) else "")
                        _log("fnav PRN %d: %d pages, %d words, have %s, eph %s%s"
                             % (_p, h["pages"], h["words"], h["have"],
                                "YES" if eph is not None else "no", xc))
                        _dh_obs("GAL_E5AI_FNAV", _p, h, eph, xc)
            # S5 D-component #3: BeiDou B2a B-CNAV2 (first LDPC), the F/NAV-aux analogue on BDS.
            if bcnav2_combiner:
                try:
                    _baux = {int(r["prn"]): r for r in _get("%s/get_status" % bcnav2_combiner)
                             if r.get("prn")}
                except Exception as e:
                    _baux = {}
                    _log_rl("bcnav2aux", "bcnav2 aux combiner %s unreadable: %s"
                            % (bcnav2_combiner, e))
                for _p, _r in _baux.items():
                    if "nav_obs" not in _r:
                        continue
                    if bcnav2 is None:
                        from bcnav2_predictor import Bcnav2Predictor
                        bcnav2 = Bcnav2Predictor(log=_log)
                        _log("B-CNAV2 decoder armed on aux chain %s" % bcnav2_combiner)
                    bcnav2.ingest(_p, _r["nav_obs"])
                # 60 s health + BRDC cross-check (alm_sys is 'C' for the BDS broker)
                if bcnav2 is not None and _now() - _bcnav2_log_t[0] > 60.0:
                    _bcnav2_log_t[0] = _now()
                    for _p in sorted(bcnav2._p):
                        h = bcnav2.health(_p)
                        if not h or not h["words"]:
                            continue
                        eph = bcnav2.ephemeris(_p)
                        xc = (_bcnav2_brdc_xcheck(brdc_alm, alm_sys, _p, eph, _log)
                              if (eph is not None and brdc_alm is not None) else "")
                        _log("bcnav2 PRN %d: %d frames, %d crc, have %s, eph %s%s"
                             % (_p, h["pages"], h["words"], h["have"],
                                "YES" if eph is not None else "no", xc))
                        _dh_obs("BDS_B2A_BCNAV2", _p, h, eph, xc)
            # S5 D-component #4 (LAST): BeiDou B1C B-CNAV1, the B-CNAV2-aux analogue on L1 BDS.
            if bcnav1_combiner:
                try:
                    _c1aux = {int(r["prn"]): r for r in _get("%s/get_status" % bcnav1_combiner)
                              if r.get("prn")}
                except Exception as e:
                    _c1aux = {}
                    _log_rl("bcnav1aux", "bcnav1 aux combiner %s unreadable: %s"
                            % (bcnav1_combiner, e))
                for _p, _r in _c1aux.items():
                    if "nav_obs" not in _r:
                        continue
                    if bcnav1 is None:
                        from bcnav1_predictor import Bcnav1Predictor
                        bcnav1 = Bcnav1Predictor(log=_log)
                        _log("B-CNAV1 decoder armed on aux chain %s" % bcnav1_combiner)
                    bcnav1.ingest(_p, _r["nav_obs"])
                if bcnav1 is not None and _now() - _bcnav1_log_t[0] > 60.0:
                    _bcnav1_log_t[0] = _now()
                    for _p in sorted(bcnav1._p):
                        h = bcnav1.health(_p)
                        if not h or not h["words"]:
                            continue
                        eph = bcnav1.ephemeris(_p)
                        xc = (_bcnav1_brdc_xcheck(brdc_alm, alm_sys, _p, eph, _log)
                              if (eph is not None and brdc_alm is not None) else "")
                        _log("bcnav1 PRN %d: %d frames, %d crc, have %s, eph %s%s"
                             % (_p, h["pages"], h["words"], h["have"],
                                "YES" if eph is not None else "no", xc))
                        _dh_obs("BDS_B1C_BCNAV1", _p, h, eph, xc)
            # GPS L1C-D CNAV-2: the bcnav1-aux analogue on the L1C broker (alm_sys 'G'). The broker's
            # own --combiner is the L1C-P pilot; the CNAV-2 data symbols come off the derived L1C-D.
            if cnav2_combiner:
                try:
                    _c2aux = {int(r["prn"]): r for r in _get("%s/get_status" % cnav2_combiner)
                              if r.get("prn")}
                except Exception as e:
                    _c2aux = {}
                    _log_rl("cnav2aux", "cnav2 aux combiner %s unreadable: %s"
                            % (cnav2_combiner, e))
                for _p, _r in _c2aux.items():
                    if "nav_obs" not in _r:
                        continue
                    if cnav2 is None:
                        from cnav2_predictor import Cnav2Predictor
                        cnav2 = Cnav2Predictor(log=_log)
                        _log("CNAV-2 decoder armed on aux chain %s" % cnav2_combiner)
                    cnav2.ingest(_p, _r["nav_obs"])
                if cnav2 is not None and _now() - _cnav2_log_t[0] > 60.0:
                    _cnav2_log_t[0] = _now()
                    for _p in sorted(cnav2._p):
                        h = cnav2.health(_p)
                        if not h or not h["words"]:
                            continue
                        eph = cnav2.ephemeris(_p)
                        xc = (_cnav2_brdc_xcheck(brdc_alm, alm_sys, _p, eph, _log)
                              if (eph is not None and brdc_alm is not None) else "")
                        _log("cnav2 PRN %d: %d frames CRC-OK, toi=%s, eph %s%s"
                             % (_p, h["words"], h["toi"], "YES" if eph is not None else "no", xc))
                        _dh_obs("GPS_L1CD_CNAV2", _p, h, eph, xc)
            # Recalibrate the constructed source: it needs the ephemeris, this cycle's geometry
            # (range + sat clock per PRN), and at least one SYNCED satellite to pin the common
            # capture-clock -> GPS offset. GPS LNAV only; other constellations get their own
            # source when their encoders exist.
            if (args.nav_bits_brdc and navbits is not None and alm_sys == "G"
                    and brdc_alm is not None and _ctx.pred):
                if navbrdc is None:
                    from navbit_brdc import BrdcLnavSource
                    navbrdc = BrdcLnavSource(log=_log)
                    _log("constructed-bit source armed (BRDC LNAV, un-synced PRNs)")
                try:
                    navbrdc.update(brdc_alm["eph"], _ctx.pred, navbits)
                except Exception as e:
                    _log("navbrdc update failed: %s" % e)
            if navbits is not None and _now() - navbits_log_t > 60.0:
                navbits_log_t = _now()
                for _p in sorted(navbits._p):
                    h = navbits.health(_p)
                    if not h:
                        continue
                    # Now that LNAV extracts the orbit set (subframes 1-3), surface the live
                    # ephemeris + a BRDC position cross-check exactly as CNAV/I-NAV do -- this is
                    # the live validator of the LNAV_EPH_FIELDS bit offsets (near-zero dpos, since
                    # BRDC IS the LNAV message) and the on-node BRDC-fallback source for L1.
                    eph_s = ""
                    e = None
                    if h["synced"]:
                        e = navbits.ephemeris(_p)
                        if e is not None:
                            eph_s = " eph toe=%.0f e=%.3e" % (e["toe"], e["e"])
                            if brdc_alm is not None:
                                eph_s += _lnav_brdc_xcheck(brdc_alm, alm_sys, _p, e, _log)
                        _log("navbit PRN %d: %d sf decoded, %d pages, predict-mismatch %s%s"
                             % (_p, h["decoded_sf"], h["pages"],
                                ("%.4f" % h["mismatch"]) if h["mismatch"] is not None else "n/a",
                                eph_s))
                    else:
                        # NOT synced == this PRN is NOT peeled (peel_require_bits). Say so, with
                        # the reason: contiguous run vs total history vs what sync needs.
                        _log("navbit PRN %d: NO SYNC (contig run %d/%d, hist %d)%s"
                             % (_p, h["run"], h["need"], h["hist"],
                                " -> CONSTRUCTED from BRDC"
                                if (navbrdc is not None and navbrdc.ready())
                                else " -> not peeled"))
                    _dh_obs("GPS_L1_LNAV", _p, h, e, eph_s)
                if navbrdc is not None:
                    # The calibration IS the trust boundary: a bad offset makes every
                    # constructed bit confidently wrong, so state it every cycle. `verify`
                    # scores constructed bits against a SYNCED satellite's own received bits
                    # -- the live form of the offline 113820/113820 test.
                    if navbrdc.ready():
                        chk = []
                        for _p in sorted(navbits._p):
                            r = navbrdc.verify(_p, navbits)
                            if r and r[0] >= 200:
                                chk.append("%d:%.1f%%" % (_p, 100.0 * r[1] / r[0]))
                        _log("navbrdc: offset %.6f s, spread %.2f ms, %d cal sats "
                             "(%d outliers dropped); verify %s"
                             % (navbrdc.offset, (navbrdc.spread or 0.0) * 1e3,
                                navbrdc.n_cal, navbrdc.n_rej, " ".join(chk) or "n/a"))
                    else:
                        _log("navbrdc: NOT ready (%s)" % navbrdc.why_not())
            # CNAV decode health + the live ephemeris (types 10+11). eph toe/e prove a decoded
            # ephemeris set. S4 EPHEMERIS CROSS-CHECK: propagate the live-decoded CNAV ephemeris
            # and the independently-downloaded BRDC (LNAV) ephemeris to the SAME absolute instant
            # (the CNAV toe) and report the ECEF position residual. Two independent nav messages,
            # two encodings (CNAV FEC+CRC vs LNAV parity), one truth -- a small residual VALIDATES
            # the whole decode chain against an outside reference, and is the foundation for
            # eventually trusting live CNAV ephemeris over the 2 h-latency download. Cheap (Kepler
            # propagation, no Viterbi), so it rides the existing 60 s health cadence.
            if cnav is not None and _now() - navbits_log_t > 60.0:
                navbits_log_t = _now()
                for _p in sorted(cnav._p):
                    h = cnav.health(_p)
                    if not h:
                        continue
                    eph_s = ""
                    e = None
                    if h["eph"]:
                        e = cnav.ephemeris(_p)
                        if e is not None:
                            eph_s = " eph toe=%.0f e=%.3e" % (e["toe"], e["e"])
                            if brdc_alm is not None:
                                eph_s += _cnav_brdc_xcheck(brdc_alm, alm_sys, _p, e, _log)
                    if h["synced"]:
                        _log("cnav PRN %d: %d msgs decoded, %d stored, %d emits, g2=%s%s"
                             % (_p, h["decoded"], h["messages"], h["emits"],
                                h["g2"], eph_s))
                    else:
                        _log("cnav PRN %d: NO DECODE (%d emits accumulated, g2=%s)"
                             % (_p, h["emits"], h["g2"]))
                    _dh_obs(_cnav_sig, _p, h, e, eph_s)
            # B-CNAV3 decode health (BeiDou B2b PRIMARY chain). Frame decode is convention-complete
            # (LDPC + CRC); the message-type histogram + SOW logged here is exactly what maps the
            # ephemeris field table (Phase 2), after which eph + a BRDC dpos xcheck join (like cnav).
            if bcnav3 is not None and _now() - _bcnav3_log_t[0] > 60.0:
                _bcnav3_log_t[0] = _now()
                for _p in sorted(bcnav3._p):
                    h = bcnav3.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = bcnav3.ephemeris(_p)
                    xc = (_bcnav3_brdc_xcheck(brdc_alm, alm_sys, _p, eph, _log)
                          if (eph is not None and brdc_alm is not None) else "")
                    _log("bcnav3 PRN %d: %d frames CRC-OK, have %s, sow=%s, eph %s%s"
                         % (_p, h["words"], h["have"], h["sow"],
                            "YES" if eph is not None else "no", xc))
                    _dh_obs("BDS_B2B_BCNAV3", _p, h, eph, xc)

    def _stage_push_seeds():
        """4: PUSH the consensus seeds to every tracker.
        
        The cycle's actuation point: whatever the stages above decided, this is what the trackers
        receive. The DLL trim is applied at POST time only, so the stored seed stays the model's view
        and the trim stays the loop's -- keeping the two ledgers separable is what makes #92's handover
        expressible at all.
        
        ⚠️ PUBLISHED DOPPLER IS THE COMMAND, NOT AN ESTIMATE. Continuity of what the tracker is told
        beats freshness of what we last measured; a seed that jumps to the newest estimate every cycle
        makes the tracker chase the estimator's noise.

        ⚠️ THE RAILED/RELEASED COUNTERS ARE `+=` ON A COUNTER INITIALISED BY THE CYCLE, so they
        must be nonlocal. An augmented assignment READS before it writes -- which is exactly what
        broker_iface missed when it cleared this stage for promotion, because it recorded only
        the Store. It cost gal_e5a a chain death at 12:49 on 2026-08-26."""
        nonlocal _rr_railed, _rr_released
        for prn, v in sorted(seeds.items()):
            d = dict(prn=prn, **v)
            if _dls.trim.get(prn):
                d["code_phase_chips"] = d["code_phase_chips"] + _dls.trim[prn]
                # ⚠️ AUDIT §4.6 (#83 2(b) precondition): THE TRIM MUST MOVE THE PHASE TOO.
                # propagate_seed PREFERS code_phase_at_ref_chips whenever the payload
                # carries it -- which the search-fed path always does -- so a trim written
                # only into cp0 was written into the field the tracker ignores: the Python
                # slow trim has been a NO-OP on every phase-carrying seed, and enabling
                # --seed-phase-transport on the DR chains would have silently disabled
                # their only code loop. One trim, both currencies, same instant.
                if d.get("code_phase_at_ref_chips", -1.0) is not None \
                        and d.get("code_phase_at_ref_chips", -1.0) >= 0.0:
                    _tmod = (LC_SEG * CODE_LEN) if LC_SEG > 1 else CODE_LEN
                    d["code_phase_at_ref_chips"] = (
                        d["code_phase_at_ref_chips"] + _dls.trim[prn]) % _tmod
            if _carrier.trim.get(prn):
                d["carrier_trim_hz"] = _carrier.trim[prn]
            if _jrc is not None and prn not in probe_set:
                # Probes excepted for the trim loop's own reason: no carrier, and a moving
                # trim moves the REPORTED Doppler, which the beam map's churn gate reads
                # as sky. A sat whose row has not converged keeps the trim-loop value
                # (usually 0 on CHORD): commanding from a wide row would inject the
                # filter's own transient into the NCO.
                _k = (args.dr_constellation, int(prn))
                # ARM-12 GUARD: the row's own satellite must be DETECTED this cycle
                # (kcoh sig >= --rrate-cmd-min-sig) before it may command. The sigma
                # gate alone admitted E4's confident-noise row (see the flag's help).
                # No kcoh row (throttled estimator at startup, or the sat absent from
                # the fold) counts as NOT detected: no evidence, no command.
                _cmd_sig_ok = True
                if args.rrate_cmd_min_sig > 0.0:
                    _cmd_sig_ok = (((_dllp.kcoh or {}).get(prn) or {}).get("sig") or 0.0) \
                                  >= args.rrate_cmd_min_sig
                if _cmd_sig_ok and _jrc.rrate_sigma(_k) <= args.rrate_cmd_max_sigma:
                    _cmd = _jrc.carrier_correction_hz(_k, args.carrier_hz)
                    if args.carrier_max_hz > 0.0:
                        _cmd = max(-args.carrier_max_hz, min(args.carrier_max_hz, _cmd))
                    # SLEW toward the target from the command actually POSTED last poll
                    # (--rrate-cmd-slew-hz): the feed's reference is only exact for a
                    # command that holds still over the emit lag, so the step is bounded
                    # and the bound is what makes the closed loop stable. Railed steps
                    # are counted into the JRR-CMD line -- a rail that never clears is
                    # a target out of reach, not convergence in progress.
                    _prev = rr_cmd_applied.get(prn, 0.0)
                    if args.rrate_cmd_slew_hz > 0.0:
                        _stp = max(-args.rrate_cmd_slew_hz,
                                   min(args.rrate_cmd_slew_hz, _cmd - _prev))
                        if abs(_cmd - _prev) > args.rrate_cmd_slew_hz:
                            _rr_railed += 1
                        _cmd = _prev + _stp
                    d["carrier_trim_hz"] = _cmd
                    _rr_cmd_new[prn] = _cmd
            # RELEASE-SLEW (arm 12). A sat that exits the command set -- sig bar lost,
            # sigma widened, receiver row gone -- used to fall back to car_trim/0 in ONE
            # poll: an instant step of its whole standing command, the exact reference
            # discontinuity the slew bound exists to prevent, taken at release instead
            # of at pull-in. Walk it back at the same bounded rate; drop out only once
            # within one step of zero.
            if (args.rrate_command and prn not in _rr_cmd_new and prn not in probe_set
                    and rr_cmd_applied.get(prn)):
                _prev = rr_cmd_applied[prn]
                _stp = args.rrate_cmd_slew_hz if args.rrate_cmd_slew_hz > 0.0 else 0.0
                if _stp > 0.0 and abs(_prev) > _stp:
                    _cmd = _prev - math.copysign(_stp, _prev)
                    d["carrier_trim_hz"] = _cmd
                    _rr_cmd_new[prn] = _cmd
                    _rr_released += 1
                # else: within one step of zero (or slew disabled) -- final sub-slew
                # step back to car_trim/0, and the sat leaves rr_cmd_applied.
            if prn in _carrier.repin_pending:
                # ONE-SHOT trim-bleed re-pin: the tracker does f_ref += this amount this frame.
                # Consume it here so it rides exactly this post (car_trim was zeroed above, so no
                # carrier_trim_hz accompanies it -- the trim moves wholly into f_ref, leaving the
                # combined carrier invariant).
                d["carrier_repin"] = _carrier.repin_pending.pop(prn)
            # Peel sign source. PILOTS (P7b): the combiner publishes bit_pred directly --
            # its secondary overlay is DETERMINISTIC, so the chips are projected from the
            # pinned dead-reckon anchor with no decode and no round trip; forward verbatim.
            # DATA signals (P7a): the LNAV predictor's output. bit_pred wins where both
            # exist (a known overlay beats a decoded guess).
            _row = status.get(prn) or {}
            _bsrc = "none"
            # A source the health monitor has condemned for THIS satellite is skipped at
            # SELECTION time, so the chain genuinely falls back (pred -> lnav -> brdc)
            # instead of re-picking the vetoed source and going dark every cycle.
            _src_ok = (lambda _s: navhealth is None
                       or navhealth.verdict(prn, _s) != "bad")
            # Wire schema is COMPONENT-KEYED: nav_bits = {"P": table, "D": table, ...},
            # "P" = the component this chain's replica correlates (relational, not a signal
            # name -- see docs/navbit_supply_architecture.md C1). The tracker also accepts a
            # bare table as "P"; we publish the keyed form so the first data-channel producer
            # is an ADDITION ("D": ...) rather than a schema change.
            if _row.get("bit_pred", {}).get("bits") and _src_ok("pred"):
                # Attach only when the table CHANGED (utc0 moves once per combiner emit);
                # on other cycles the tracker keeps its stored copy -- see bp_pushed above.
                if bp_pushed.get(prn) != _row["bit_pred"].get("utc0"):
                    d["nav_bits"] = {"P": _row["bit_pred"]}
                    bp_pushed[prn] = _row["bit_pred"].get("utc0")
                _bsrc = "pred"
            elif navbits is not None:
                # Predict from the freshest capture-clock UTC this PRN has reported: the
                # tracker consumes by ITS record UTC (same capture clock), so wall-clock
                # never enters. 4 s horizon >> the ~1 s status staleness + push cadence.
                _utc = _row.get("utc")
                if _utc:
                    nb_lnav = navbits.predict(prn, float(_utc), horizon_s=4.0)
                    nb_brdc = (navbrdc.predict(prn, float(_utc), horizon_s=30.0)
                               if navbrdc is not None else None)
                    if navhealth is not None:      # shadow-remember BOTH candidates
                        navhealth.remember(prn, nb_lnav, "lnav")
                        navhealth.remember(prn, nb_brdc, "brdc")
                    nb = nb_lnav if (nb_lnav is not None and _src_ok("lnav")) else None
                    if nb is None and _src_ok("brdc"):
                        # CONSTRUCTED fallback: this PRN never synced, so the decoder has
                        # nothing. Bits come from BRDC instead; alignment comes from the
                        # tracking geometry (range + sat clock) plus the common clock offset
                        # calibrated off a satellite that DID sync -- never from decoding,
                        # which is precisely what this satellite cannot do.
                        # 30 s, not the decoder's 4 s: only sf1-3 are constructible, so a 4 s
                        # window that lands inside sf4/5 is entirely unknown, predict() returns
                        # None, and the PRN reads `nobits` forever. A full frame (30 s) always
                        # spans the 18 s of sf1-3, so every push carries usable bits. Costs
                        # 1500 int8 per PRN.
                        nb = nb_brdc
                        _bsrc = "brdc" if nb is not None else "none"
                    elif nb is not None:
                        _bsrc = "lnav"
                    if nb is not None:
                        d["nav_bits"] = {"P": nb}
                        _bsrc = _bsrc if _bsrc != "none" else "lnav"
            elif cnav is not None:
                # CNAV serve: the L2C-CM chain's replica correlates the DATA component, so a
                # decoded CNAV table is its "P" (the direct L1CA/LNAV analog). But this SERVES
                # already-decoded spans, it does not PREDICT the future (the CNAV type schedule
                # is not fixed -- see cnav_predictor), so at a forward horizon predict() usually
                # returns None and nothing is attached. That is correct: CL, not CNAV, is L2C's
                # peel win; CNAV's prizes are the live ephemeris + the S4 L5 cross-band feed.
                # Shadow-remembered and vetoed like every other source before it feeds a wipe.
                _utc = _row.get("utc")
                if _utc:
                    nb = cnav.predict(prn, float(_utc), horizon_s=4.0)
                    if navhealth is not None:
                        navhealth.remember(prn, nb, "cnav")
                    if nb is not None and _src_ok("cnav"):
                        d["nav_bits"] = {"P": nb}
                        _bsrc = "cnav"
            # VETO: a source that does not match the air for this satellite must not feed the
            # subtracter. Wrong bits are worse than no bits -- no bits means no subtraction,
            # wrong bits mean subtracting at the wrong sign on ~40% of records (measured
            # 2026-07-26: HURTING 0-1 -> 7-8). Then remember what we are about to publish, so
            # next cycle's observations score THIS table rather than a re-derived one.
            if "nav_bits" in d and navhealth is not None and navhealth.veto(prn, _bsrc):
                # FALL BACK, do not go dark: a bad decoded table must not darken a PRN whose
                # constructed table is fine. The chain re-runs with the vetoed source skipped
                # next cycle via the per-(prn,source) verdict; this cycle just drops the bits
                # (one cycle of nobits is the safe direction).
                d.pop("nav_bits")
                _bsrc = "vetoed:" + _bsrc
            elif "nav_bits" in d and navhealth is not None:
                navhealth.remember(prn, d["nav_bits"].get("P"), _bsrc)
            bit_src[_bsrc] = bit_src.get(_bsrc, 0) + 1
            if _bsrc != "none" and "nav_bits" in d:
                bit_known[_bsrc] = bit_known.get(_bsrc, 0) + sum(
                    1 for t in d["nav_bits"].values() for b in t["bits"] if b)
            payload.append(d)

    def _stage_dead_reckon():
        """3e: DEAD-RECKONED SEEDING -- the model-primary spine, and the shell of its pipeline.
        
        Refreshes the ephemeris, builds the per-satellite predictions and code offsets, then runs the
        five sub-stages in order: joint shadow -> clock solve -> clock quality -> clock adopt -> seed.
        Its working set lives on `_drp`, so each arrow of that pipeline is a named field rather than
        an implicit shared local.
        
        On the searchless chains (Galileo, BeiDou) this IS the acquisition path: it can seed a
        satellite that has never been detected, from the broadcast ephemeris and the receiver clock
        alone.
        
        ⚠️ `t_now_abs` IS None, NEVER 0.0, until the first ephemeris refresh. A missing axis time is
        UNKNOWN, and a confident wrong timestamp is worse than a skipped measurement -- as 0.0 it
        killed chain threads through the #85 stash."""
        if (dr_state is not None and args.almanac and _ctx.pred and utc0_sample0
                and _now() >= dr_state["next"]):
            _drp.now_w = _now()
            dr_state["next"] = _drp.now_w + args.dr_refresh_s
            # ── DO NOT SEED ON A GUESSED CLOCK (2026-08-22) ──────────────────────────────
            # Measured with the BIRTH-STEP diagnostic: every satellite born before the clock
            # bootstraps carries NO clock, and the first re-birth after BOOTSTRAP adds it --
            # so the whole fleet steps by one clock at once. 40 events, all within 57 s of
            # broker start and none after 60 s, with step/off = 0.9958: the step IS `_off`,
            # to 0.4%. Five satellites agreeing to 0.08% with unrelated Doppler is what gave
            # it away -- nothing per-satellite can do that.
            # The fix is to wait, because the wait is SHORT and the clock is imminent: the
            # solve snaps on its first median (gps_l5 bootstraps ~1 s after start) and the
            # model-primary chains adopt it cross-band within a few seconds of that.
            # ⚠️ BUT IT MUST NOT BE AN INDEFINITE WAIT, and that is the whole design of the
            # prime. --dr-clock-chips exists precisely so a DETECTOR-LESS chain can seed at
            # all: `offs` is always empty there, so the solve never runs and no bootstrap is
            # ever coming. Refusing to seed until a measurement arrives would leave exactly
            # those chains dark forever -- trading a 150-chip startup step for a dead chain.
            # So: withhold while the clock is admittedly a guess, but only for
            # --dr-clock-wait-s, then seed on the prime and SAY SO. A guard that can deadlock
            # the instrument is worse than the artefact it removes.
            _drp.hold = False
            if dr_state.get("clk_primed") and args.dr_clock_wait_s > 0.0:
                _waited = _drp.now_w - dr_state.get("clk_t", _drp.now_w)
                if _waited < args.dr_clock_wait_s:
                    _log_rl("clkwait", "dead-reckon: WITHHOLDING seeds -- the clock is still "
                                       "the %.2f-chip PRIME, not a measurement (%.0f of %.0f s"
                                       " waited). Seeding now would anchor every cp0 without "
                                       "a clock and step the whole fleet by ~%.0f chips at the"
                                       " first re-birth after BOOTSTRAP."
                            % (dr_state.get("clk") or 0.0, _waited, args.dr_clock_wait_s,
                               abs(dr_state.get("clk") or 0.0) or 150.0),
                            every_s=10.0)
                    dr_state["next"] = _drp.now_w + min(2.0, args.dr_refresh_s)
                    _drp.hold = True
                # ⚠️ elif, NOT a second if. Written as a bare `if` this fired in the SAME
                # MILLISECOND as the withhold above -- "seeding on the PRIME after waiting
                # 30 s" logged 0 s in, because reaching the warning never depended on the
                # wait having expired. An alarm that cannot distinguish "waiting" from
                # "gave up waiting" is worse than none: it would have taught us to ignore it.
                elif not dr_state.get("clk_wait_warned"):
                    dr_state["clk_wait_warned"] = True
                    _log("dead-reckon: ⚠️ seeding on the %.2f-chip PRIME after waiting %.0f s "
                         "-- no clock measurement arrived. Expected on a DETECTOR-LESS chain "
                         "with no sibling to adopt from; on a chain that HAS detectors it "
                         "means the solve is not running, and every seed below is anchored on "
                         "a guess." % (dr_state.get("clk") or 0.0,
                                       _drp.now_w - dr_state.get("clk_t", _drp.now_w)))
            # THE DEAD-RECKON MODEL MUST WORK AT THE CODE THE TRACKER ACTUALLY DESPREADS.
            # This was CODE_LEN / chip_rate -- ONE PRIMARY PERIOD -- so t0m and every predicted
            # phase were reduced mod 1 ms and the secondary segment was discarded before a seed
            # was ever built. A dead-reckoned PRN therefore always landed in segment 0: right
            # 1 time in LC_SEG.
            #
            # GPS survives that because the BLIND SEARCH re-seeds it with a measured `nh`, so
            # the model's missing segment is overwritten before it matters. E5a/B2a have no
            # blind search at all (per-PRN secondaries, CHORD_MULTIBAND.md section 5) -- the
            # dead-reckon seed IS the answer -- so for them this modulo is the whole bug:
            # measured 2026-08-08, every E5a seed had code_phase_chips < 10230 against a
            # 1,023,000-chip code, i.e. 1-in-100 of the right CS period, i.e. noise.
            #
            # Reducing mod the LONG period instead keeps the numbers small enough for double
            # precision (the reason the reduction exists) while carrying the segment. Placing
            # it needs absolute time to half a primary period, 0.5 ms; the F-engine anchor is
            # GPS-disciplined to microseconds and BRDC range/clock are nanosecond-class, so
            # there are three orders of margin.
            _drp.t_code = (LC_SEG * CODE_LEN) / args.chip_rate_hz if args.dr_long_code \
                     else CODE_LEN / args.chip_rate_hz
            # The seed is reduced at the SAME length the prediction was: one constant, used
            # twice, so they cannot drift apart.
            _drp.mod = (LC_SEG * CODE_LEN) if args.dr_long_code else CODE_LEN
            # Layer-2 slew constants (task #30). CAP: 0.05 chips per 2 s cycle = 0.025
            # chips/s of correction authority -- above the 0.003-0.02 chips/s drift band
            # measured on sky, an order below the 0.25-chip DLL trims a fold tolerates, and
            # 20x below the ~1-chip clock-EMA jitter the reverted 10 s repin injected whole.
            # K: first-order low-pass; with the cap it bounds, it is deliberately unfussy.
            # The per-event slew rate limit. 0.05 chips was hardcoded here and is the
            # single most consequential constant in the seeding path (see the call site):
            # at 0.05 a go, every 10-30 s, a satellite 5-8 chips from its model takes of
            # order an HOUR to arrive. A flag because the decisive experiment is
            # --dr-slew-cap 0 -- slewing OFF, the seed left where the loop put it -- which
            # is the one arm nobody has run and the one that separates "the seed is wrong"
            # from "the model is wrong and the slew drags good seeds onto it".
            _drp.slew_cap = args.dr_slew_cap
            _drp.slew_k = 0.25
            if dr_state["eph"] is None or _drp.now_w - dr_state["eph_t"] > 7200:
                try:
                    dr_state["eph"] = dr_eph_mod.parse_rinex_nav(dr_eph_mod.fetch_brdc())
                    dr_state["eph_t"] = _drp.now_w
                    dr_state["t0m"] = dr_eph_mod.gpst_of_utc(utc0_sample0) % _drp.t_code
                    _log("dead-reckon: BRDC loaded (%d sats)" % len(dr_state["eph"]))
                    # MEASURED CODE BIASES, refreshed on the ephemeris cadence (A0b, part 2).
                    # Daily product, ~5 days of latency, biases stable over weeks -- so the
                    # refresh rate is irrelevant and the fetch is cached. Optional by design:
                    # no token or no network -> dcb stays None and group_delay_s falls back to
                    # the broadcast term, which is what every run before 2026-08-23 did.
                    if args.dcb_bias:
                        try:
                            import gnss_dcb as _dcbm
                            _p = _dcbm.fetch_dcb()
                            _t = _dcbm.parse_dcb(_p)
                            dr_state["dcb"] = _t or None
                            if _t:
                                _n = sum(1 for k in _t if k[0] == args.dr_constellation)
                                _log("dead-reckon: DCB loaded (%s; %d sats this "
                                     "constellation) -- measured code biases override the "
                                     "broadcast TGD/BGD per satellite"
                                     % (os.path.basename(_p or "?"), _n))
                            else:
                                _log("dead-reckon: no DCB product (no token/network) -- "
                                     "falling back to the broadcast group delay")
                        except Exception as _de:
                            dr_state["dcb"] = None
                            _log("dead-reckon: DCB load failed (%s); broadcast term only"
                                 % _de)
                except Exception as e:
                    _log("dead-reckon: BRDC unavailable (%s); retry in 10 min" % e)
                    dr_state["eph_t"] = _drp.now_w - 7200 + 600
            # DECODED-EPH FALLBACK: keep predicting off our own decode when the network BRDC is
            # gone (or always, under --decoded-eph-fallback-force, the live A/B harness).
            _use_decoded = _decfb is not None and (
                args.decoded_eph_fallback_force
                or (args.decoded_eph_fallback and not dr_state["eph"]))
            if dr_state["eph"] or _use_decoded:
                _drp.tag = args.dr_constellation
                _drp.t_now_abs = _drp.now_w - utc0_sample0
                # ── #83 THE AXIS FIX (see --dr-fengine-axis) ── "now" from the F-engine
                # hop counter: newest telemetry hop at its fetch instant, plus the wall
                # ELAPSED since -- so NTP's absolute offset never enters and its slew
                # contributes only (drift x sub-cycle seconds) = nanoseconds. Every label
                # this block stamps (h1, _rh_birth) and every phase it evaluates then
                # lives on the axis the tracker actually runs. The static difference
                # between this axis and the old wall one lands in the solved receiver
                # clock exactly as the old anchor error did (common-mode), so nothing
                # steps at the flip of the flag except the labels' MEANING.
                if args.dr_fengine_axis and fe_axis[0] is not None:
                    # FILTERED offset, not the freshest sample (2026-08-23): the raw form
                    # _feh/hps + (now_w - _few) re-samples the pipeline lag every poll and
                    # the jitter lands in every rebirth as lag x range_rate -- see fe_off.
                    if fe_off[0] is not None:
                        _drp.t_now_abs = fe_off[0] + _drp.now_w
                    else:
                        _feh, _few = fe_axis[0]
                        _drp.t_now_abs = _feh / args.hops_per_sec + (_drp.now_w - _few)
                # ── THE FORECAST EPOCH (see --dr-forecast-lead-s) ──
                # A seed is a FORECAST WITH A LABEL: (ref_hop, phase, doppler, rate). Its
                # correctness is defined entirely on the hop axis, so producing one needs no
                # notion of "now" at all -- and asking for one is what dragged the wall clock,
                # the NTP offset and the telemetry lag into the seed path in the first place.
                # Instead choose the hop we are forecasting TO: H = newest telemetry hop +
                # lead. Then the pipeline lag never enters the arithmetic; it only sets how
                # large the lead must be (it must exceed transport + install, or the seed
                # lands after the epoch it describes), and the lag's jitter merely eats
                # margin. H is an exact integer hop, so int(round(t_fc_abs*hps)) == H and the
                # label carries no rounding of its own.
                # ⚠️ ONE EPOCH, USED CONSISTENTLY. The phase and the label must be built from
                # the SAME t or the tracker's back-reference no longer cancels (see the note
                # at h1 below). Filter/bookkeeping sites deliberately keep t_now_abs: ages,
                # update_rrate and the joint cycle are measurements at NOW, and shifting them
                # into the future would inflate every age by the lead.
                _drp.t_fc_abs = _drp.t_now_abs
                if args.dr_forecast_lead_s > 0.0 and fe_axis[0] is not None:
                    # forecast from the FILTERED now (t_now_abs above), not the raw newest
                    # hop -- same jitter, same fix. H stays an exact integer hop so the
                    # label still carries no rounding of its own.
                    _fch = int(round((_drp.t_now_abs + args.dr_forecast_lead_s)
                                     * args.hops_per_sec))
                    _drp.t_fc_abs = _fch / args.hops_per_sec
                # ── THE EPHEMERIS EPOCH STAYS ON WALL TIME, AND HERE IS THE MEASUREMENT ──
                # Orbit evaluation (predict_all / predict_from_decoders below) is the ONE
                # consumer in this file that needs absolute UTC to be TRUE rather than
                # merely self-consistent: everything else either differences two wall reads
                # (immune to a slewing offset) or uses wall on both sides of a round trip
                # (it cancels). A wrong epoch here is a wrong satellite POSITION and no
                # round trip removes it, so moving it onto the F-engine axis looked
                # obviously right -- utc0_sample0 is GPS-disciplined, cf06's wall carries
                # ~1.45 ms of NTP error (chrony root dispersion 1.489 ms, measured).
                #
                # ⚠️ TRIED 2026-08-17, MEASURED WORSE BY ~65x, REVERTED. The F-engine "now"
                # we can OBSERVE is not the F-engine's now: it is the newest telemetry
                # pow_hop, which is behind the sky by the gather/merge/serve latency.
                # Measured over the 520 cycles of broker_onsky_e5a: (utc0 + hop/hps) - wall
                # = -99.6 ms MEDIAN with a 59 ms interquartile spread. Feeding that to the
                # ephemeris trades 1.45 ms of NTP error for ~100 ms of pipeline lag -- 80 m,
                # 2.7 chips at L5 -- and the armed replay showed exactly that: seed cp0
                # moved 0.6-2.4 chips per PRN, scaling with each satellite's range rate.
                # The axis is drift-free, which is why it is right for LABELS (a label and
                # its phase are stamped at the same t and the tracker propagates from the
                # label -- staleness cancels); it is wrong for an EPOCH, where staleness is
                # the whole error. Fixing this properly needs lag compensation, i.e. an
                # estimate of the pipeline delay -- not a substitution.
                #
                # What survives is the reason the question was asked: a wall STEP is bounded
                # by nothing, and would move every model seed on all five chains at once,
                # fleet-common and unattributable. The two axes are compared below as a
                # TRIPWIRE (log only, never a control input) -- it cannot see a step smaller
                # than the lag jitter, and does not pretend to.
                if fe_axis[0] is not None and utc0_sample0:
                    # unpacked HERE, not borrowed from the axis-fix block above: that block
                    # runs only under --dr-fengine-axis and this tripwire must work either way
                    _axh, _axw = fe_axis[0]
                    _dax = (utc0_sample0 + _axh / args.hops_per_sec) - _axw
                    _dprev = dr_state.get("ax_off")
                    if _dprev is not None and abs(_dax - _dprev) > args.clock_step_guard_s:
                        _log("*** WALL-vs-F-ENGINE OFFSET JUMPED %+.3f s (%.3f -> %.3f, "
                             "bar %.3f s). TWO CAUSES, and this line cannot tell them "
                             "apart: (a) cf06's wall clock stepped -- the F-engine axis is "
                             "GPS-disciplined, so every model-evaluated seed on every chain "
                             "just moved with it, ~%.0f chips at 800 m/s of range rate, "
                             "fleet-common; (b) the telemetry lag jumped -- the observable "
                             "F-engine 'now' is the newest pow_hop and trails the sky by "
                             "the gather/serve latency (-99.6 ms median, 59 ms IQR "
                             "measured), so anything near that scale is lag, not the clock. "
                             "Discriminate with chronyc tracking. Nothing here corrects "
                             "either -- this exists so the next hour is attributable."
                             % (_dax - _dprev, _dprev, _dax, args.clock_step_guard_s,
                                abs(_dax - _dprev) * 800.0 / 29.3))
                    dr_state["ax_off"] = _dax
                _drp.la = (args.code_bias_force * 1e-6 if args.code_bias_force is not None
                      else (code_bias_ema if code_bias_ema is not None else None))
                # TASK #30, LAYER 1: A DETECTOR-LESS CHAIN CANNOT MEASURE (l-a) -- BORROW THE
                # BAND SIBLING'S. code_bias_ema fills only from this chain's own cp-fit pool,
                # which needs detections, so on E5a/B2a it stayed None and every dead-reckon
                # seed shipped code_phase_rate = 0.0 (visible in SEEDDBG). The receiver code
                # clock is PER BAND, not per chain -- gps_l5 measures it continuously on the
                # same 1176.45 MHz and has contributed it to the Receiver since M3, but
                # rx.code_bias() had NO consumer until now. The unmodeled rate this leaves is
                # ~0.003-0.01 chips/s, which walks the prompt through the +-1 chip correlation
                # window in 3-11 minutes: measured on sky as E5a's rise-peak-fall deep_snr
                # envelope with the fleet disc railed at +0.7 and E >> L. Rates are SMOOTH --
                # this is the correction that cannot inject a step, unlike the reverted repin.
                if _drp.la is None:
                    _sh_cb = rx.code_bias(band_id, exclude=chain_id, t_now=_drp.now_w)
                    _sh_band = band_id
                    # LAYER 2 (task #34): FALL BACK ACROSS THE BAND. The same-band lookup above
                    # is a bootstrap trap for a band with no chain that can solve its own clock,
                    # and 1207.14 MHz was exactly that: gal_e5b and bds_b2b adopted (l-a) ZERO
                    # times while their 1176.45 siblings adopted it 102 and 107 times, so all 19
                    # of their PRNs shipped code_phase_rate = 0.0 and walked open-loop.
                    #
                    # This is SOUND, not a compromise, and the reason is that (l-a) is a
                    # FRACTIONAL FREQUENCY: tau_band -- the per-carrier group delay that makes
                    # the code PHASE band-specific -- is a CONSTANT, and a constant contributes
                    # nothing to a rate. See Receiver.code_bias_any_band. The code PHASE keeps
                    # its per-band scoping (dr_clock is untouched); adopting a phase across
                    # carriers would inject the very tau_band offset the second band exists to
                    # measure.
                    #
                    # Logged distinctly from the same-band case: "cross-band" in the message is
                    # how you tell, from the log alone, that a chain is running on a borrowed
                    # rate rather than one measured in its own band.
                    if _sh_cb is None:
                        _sh_cb = rx.code_bias_any_band(exclude=chain_id, t_now=_drp.now_w)
                        _sh_band = "cross-band"
                    if _sh_cb is not None:
                        _drp.la = float(_sh_cb.value)
                        _log_rl("la-adopt",
                                "dead-reckon: (l-a) %+.4f ppm ADOPTED from in-process chain "
                                "'%s' (%s %s) -> seeds carry the code-clock rate"
                                % (_drp.la * 1e6, _sh_cb.src,
                                   "same band" if _sh_band == band_id else "CROSS-BAND, rate only;",
                                   _sh_band if _sh_band == band_id else "phase stays per-band"),
                                every_s=300.0)
                    else:
                        _drp.la = 0.0
                # clock drift (chips/s): EMPIRICAL from consecutive raw solves (EMA'd
                # below), falling back to the f_chip*(l-a) model until measured -- the
                # modeled value left a persistent EMA lag (~0.6 chips at first deploy),
                # outside the BOC DLL capture range.
                _drp.drift = dr_state.get("drift")
                if _drp.drift is None:
                    _drp.drift = args.chip_rate_hz * _drp.la
                try:
                    # two epochs, 4 s apart: range_rate difference -> doppler RATE (the
                    # TLE almanac's rate is unused here -- BRDC governs model-owned sats)
                    if _use_decoded:
                        _ents = _decoded_entries(_drp.now_w)
                        _drp.pd = _decfb.predict_from_decoders(
                            _ents, args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(_drp.now_w, tz=timezone.utc), mask_deg=-90.0)
                        # CENTRED PAIR (task #52): +/-2 s about now_w, not [now, now+4].
                        _drp.pd2 = _decfb.predict_from_decoders(
                            _ents, args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(_drp.now_w + 2.0, tz=timezone.utc),
                            mask_deg=-90.0)
                        pd0 = _decfb.predict_from_decoders(
                            _ents, args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(_drp.now_w - 2.0, tz=timezone.utc),
                            mask_deg=-90.0)
                        if _now() - _decfb_log_t[0] > 60.0:
                            _decfb_log_t[0] = _now()
                            ab = ""
                            if args.decoded_eph_fallback_force and dr_state["eph"]:
                                # A/B: compare decoded vs BRDC predict, worst common sat.
                                pb = dr_eph_mod.predict_all(
                                    dr_state["eph"], args.lat, args.lon, args.alt,
                                    datetime.fromtimestamp(_drp.now_w, tz=timezone.utc),
                                    mask_deg=-90.0)
                                cm = set(_drp.pd) & set(pb)
                                if cm:
                                    dr_m = max(abs(_drp.pd[k]["range_m"] - pb[k]["range_m"])
                                               for k in cm)
                                    dd = max(abs(_drp.pd[k]["range_rate_mps"]
                                                 - pb[k]["range_rate_mps"]) for k in cm)
                                    ab = (" | A/B vs BRDC over %d sats: worst range %.1f m, "
                                          "range-rate %.3f m/s (%.2f Hz@fc)"
                                          % (len(cm), dr_m, dd,
                                             dd / 299792458.0 * args.carrier_hz))
                            _log("dead-reckon: predicting from DECODED eph (%s; %d entries -> "
                                 "%d sats)%s"
                                 % ("FORCE A/B" if args.decoded_eph_fallback_force
                                    else "BRDC network DOWN -> fallback",
                                    len(_ents), len(_drp.pd), ab))
                    else:
                        # A0b (2026-08-23): `signal=` makes the returned sat_clk_s refer to
                        # THIS chain's code rather than the constellation's own broadcast
                        # reference (GPS L1/L2, GAL E1/E5a or E1/E5b, BDS B3I). It is the
                        # clock cp_predicted consumes, so it moves the seed directly.
                        # Measured before arming: ~+0.15 chips common at L5, +-0.3 per-sat --
                        # a b_sat-scale correction, NOT a constellation-offset one.
                        _drp.pd = dr_eph_mod.predict_all(
                            dr_state["eph"], args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(_drp.now_w, tz=timezone.utc), mask_deg=-90.0,
                            signal=args.signal, dcb=dr_state.get("dcb"))
                        # CENTRED PAIR (task #52): +/-2 s about now_w, not [now, now+4]. The
                        # old form was a FORWARD difference, so it estimated the rate at
                        # now+2 and handed it to the seed as if it were the rate at now -- the
                        # same first-order time-tag mistake as the velocity in
                        # gnss_ephemeris.sat_pos_clk (ced7f8b51), one level up. Bias ~1.6e-3
                        # Hz/s against a 29 mHz/s single-window budget: below budget, which is
                        # why it survived, but it is free to remove and the centred form also
                        # cuts the truncation error 3x at the same 4 s baseline.
                        _drp.pd2 = dr_eph_mod.predict_all(
                            dr_state["eph"], args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(_drp.now_w + 2.0, tz=timezone.utc),
                            mask_deg=-90.0, signal=args.signal, dcb=dr_state.get("dcb"))
                        pd0 = dr_eph_mod.predict_all(
                            dr_state["eph"], args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(_drp.now_w - 2.0, tz=timezone.utc),
                            mask_deg=-90.0, signal=args.signal, dcb=dr_state.get("dcb"))
                except Exception as e:
                    _drp.pd, _drp.pd2, pd0 = {}, {}, {}
                    _log("dead-reckon: predict failed: %s" % e)
                if _drp.pd:
                    # cache for the SEED loop (next cycle): BRDC doppler/rate for
                    # search-anchored sats, so both masters share one currency
                    dr_state["pd"], dr_state["pd2"] = _drp.pd, _drp.pd2
                    dr_state["pd0"] = pd0

                # THE EPHEMERIS'S OWN EPOCH, in capture age. predict_all above ran at
                # now_w (WALL), so in capture-age units its rows are valid at
                # now_w - utc0_sample0 -- which is t_now_abs on the WALL axis and is NOT
                # t_now_abs under --dr-fengine-axis, where t_now_abs lags wall by the
                # telemetry lag (~0.15 s live). cp_predicted used to extrapolate from
                # t_now_abs unconditionally; under the axis fix that misplaces every
                # satellite by rdot*lag metres = K*dop*lag chips, PER SAT, dop-signed --
                # measured 2026-08-24 as the standing trim law trim ~ -1.0e-3 chips/Hz
                # x dop (tau 115 ms = the live axis lag), the walkoff limit cycle's root.
                # The clock median hides the common part, which is why 08-23's eps read
                # +420 ms against an 8-18 s lag. Born WITH the axis fix (08-17): on the
                # wall axis the two epochs coincide and the old form was exact.
                _drp.t_eph_age = _drp.now_w - utc0_sample0

                # -- receiver-clock solve (the bootstrap) + per-sat integrity residuals:
                # physical cp at each detection hop (undo the sample-0 back-reference),
                # minus the prediction, epoch-normalized to now (the offset drifts at
                # f_chip*(l-a)); the circular median over sats is the receiver clock.
                _drp.offs = []
                # DETECTION AGE per sat (task #33, docs 11.22 follow-up). The epoch
                # normalization below ages each latched detection by drift*(t_now - t_i)
                # with drift = f_chip*(l-a): the l-a EMA's measured noise is +-0.05-0.07
                # chips/s, and detections latch for up to the 1276 s per-PRN revisit --
                # so the normalization term can carry CHIPS of error that scale with THIS
                # SAT'S staleness. Logged next to each residual so "integrity" can be
                # judged against age: residuals that grow with age are measuring the
                # normalization, not the sky.
                det_age = {}
                off_inputs = {}
                for prn, (snr, dop, cp, ref_hop, _nh, _cpl, _car) in sorted(best.items()):
                    v = _drp.pd.get((_drp.tag, prn))
                    if v is None:
                        continue
                    t_i = ref_hop / args.hops_per_sec
                    det_age[prn] = _drp.t_now_abs - t_i
                    # The search's sample-0 back-reference subtracts BOTH the nominal code
                    # advance (t*f_chip mod L -- the 'off' term) and the code-Doppler drift;
                    # add BOTH back. Omitting the nominal term hid inside the solved clock
                    # whenever all detections shared a snapshot hop (every offline check),
                    # but across live scans it scatters by f_chip/hops_per_sec * (ref_hop
                    # mod code-period) -- the wandering "13 chips/s clock" of the first
                    # live deploy (2026-07-13).
                    # #45 STEP 5 (2026-08-12): this reconstruction STAYS the clock-solve
                    # measurement. The payload's cp_at_ref is better conditioned in
                    # principle, but it is referenced at the hop's last sample AND carries
                    # the replica anchor's Doppler term (+52.3711 + 1.39e-4*dop chips vs
                    # this quantity, measured on 26,815 banked detections), so consuming it
                    # means importing the search's fft_len and anchor geometry here -- the
                    # coupling this pass removes -- to fix a conditioning problem the sky
                    # says does not exist: the detection's (cp0, dop) pair is published
                    # together and its embed cancels exactly, giving 0.27-chip measured
                    # continuity. Better conditioned on paper is not better when the price
                    # is a second component's geometry.
                    cp_loc = (cp + t_i * args.chip_rate_hz
                              * (1.0 + args.code_doppler_sign * dop / args.carrier_hz)
                              ) % CODE_LEN
                    d_i = (cp_loc - cp_predicted(v, t_i)
                           + _drp.drift * (_drp.t_now_abs - t_i)) % CODE_LEN
                    _drp.offs.append((prn, d_i))
                    # WHICH INPUT MOVED. Record cp_loc (NOT raw cp): raw cp swings
                    # ~uniform mod L between passes by construction -- the search embeds
                    # -t_abs*chip_rate*sign*dop/carrier in cp0 and the detection Doppler
                    # jitters +-60 Hz pass to pass, 1696 chips/Hz at t_abs ~2.2 days.
                    # That embed cancels exactly in cp_loc (verified live 2026-08-11:
                    # d(cp_loc) median 0.27 chips over 1423 consecutive detections), so
                    # cp_loc is the ONLY interpretable form of "what the search said".
                    # The 2026-08-11 WHAT-MOVED that printed raw dcp cost half a day:
                    # its thousands-of-chips swings were read as a search fault.
                    off_inputs[prn] = (cp_loc, t_i, dop)
                # ERRATIC-TRACK GUARD (#39). A satellite whose solve offset JUMPS between
                # consecutive cycles is not measuring anything: d_i = clk + b_i and both
                # terms are stable, so a real satellite moves far less than 10 chips per
                # cycle. MEASURED 2026-08-10: PRN 2 -- which is not L5-capable, so this was
                # a noise or cross-correlation track -- entered at -3226 chips and then read
                # +2430 +2997 -4988 +3024 -4193 +2897 +1297 +3231 on successive cycles while
                # every real satellite sat at 1-6. It dragged the median, the clock latched
                # at 19:47, fleet-coherent alignment collapsed 0.86 -> 0.15 and engagement
                # went to zero. Catching it HERE means one satellite is dropped instead of
                # the whole solve being refused, which is what makes the refusal guard a
                # last resort rather than the first line.
                if args.dr_max_off_jump_chips > 0.0 and _drp.offs:
                    _keep, _drop = split_erratic_offsets(
                        _drp.offs, dr_state.setdefault("off_hist", {}), _drp.now_w,
                        args.dr_max_off_jump_chips, args.dr_off_jump_max_age_s, CODE_LEN)
                    if _drop:
                        _prevI = dr_state.setdefault("off_inputs_prev", {})
                        _det = []
                        for _p, _j in _drop:
                            # NOT `_now`: that name is a FUNCTION in this scope, and
                            # assigning it here made every reference to _now() in main()
                            # a local-before-assignment -- the broker died on startup at
                            # a line 2000 lines away from this one.
                            _cur = off_inputs.get(_p)
                            _was = _prevI.get(_p)
                            if _cur and _was:
                                _dcl = ((_cur[0] - _was[0] + CODE_LEN / 2.0) % CODE_LEN
                                        - CODE_LEN / 2.0)
                                _det.append("PRN %d: dcp_loc %+.1f  dt_i %+.3fs  "
                                            "ddop %+.3fHz"
                                            % (_p, _dcl, _cur[1] - _was[1],
                                               _cur[2] - _was[2]))
                        if _det:
                            _log_rl("offjumpwhy", "clock solve: WHAT MOVED -- " +
                                    " | ".join(_det), every_s=60.0)
                    dr_state["off_inputs_prev"] = dict(off_inputs)
                    if _drop and len(_keep) >= args.dr_min_sats:
                        _drp.offs = _keep
                        _log_rl("offjump",
                                "clock solve: EXCLUDED %s -- offset jumped %s chips since "
                                "the last cycle (bound %.0f). d_i = clk + b_i, both "
                                "stable; the detection-Doppler embed cancels exactly in "
                                "cp_loc (2026-08-11), so a jump is a real discontinuity "
                                "in what the search reported, the model, or the drift "
                                "normalization -- see WHAT MOVED"
                                % (", ".join("PRN %d" % p for p, _ in _drop),
                                   ", ".join("%.0f" % j for _, j in _drop),
                                   args.dr_max_off_jump_chips), every_s=60.0)
                    elif _drop:
                        # DROPPING THEM WOULD STARVE THE SOLVE. Say so rather than silently
                        # keeping them: if EVERY satellite jumped, the clock itself moved
                        # (or the model did), and that is a different fault from one bad
                        # track -- the MAD guard below is the right net for it.
                        _log_rl("offjumpkeep",
                                "clock solve: %d PRN(s) jumped but excluding them leaves "
                                "%d < --dr-min-sats %d -- keeping all; if this persists the "
                                "CLOCK moved, not one track"
                                % (len(_drop), len(_keep), args.dr_min_sats), every_s=60.0)
                dr_state["offs_t"] = _drp.now_w  # freshness stamp for the referee's integrity veto
                # -- P2a SHADOW: the joint receiver-state solve (task #33, section 3a) ----
                # `offs` IS the measurement, and always was. d_i = clk + b_i: the physical
                # code phase the search measured, minus the pure model, with no clock
                # removed. Today the next 40 lines take its circular median as the clock
                # and treat the per-sat spread (+-3-7 chips, docs 11.22) as an error to be
                # gated on; the joint solve reads the SAME numbers as clock PLUS per-sat
                # bias and estimates both, separated by process noise rather than by a
                # threshold. Shadow: logged beside the median it will replace, consumed by
                # NOTHING, so every transcript digest is untouched.


                # ⚠️ NOTES MUST ESCAPE EVEN WITH NO DETECTIONS (2026-08-21). The shadow
                # block below is gated on `offs` -- this chain's OWN detections -- so on a
                # model-primary chain the filter's notes were never drained and never
                # logged. Measured: the state rejected 494 measurements in a row with
                # updates frozen at 3, the DEAF latch fired exactly as designed, and not
                # one line reached the operator. Identical in shape to the DRCLK defect
                # recorded a few hundred lines below (the four model-primary chains never
                # log the clock they consume). Drain first, unconditionally.
                if args.rrate_state or args.joint_shadow:
                    try:
                        _jsn = rx.joint_receiver(band_id, CODE_LEN,
                                                 rereference=args.joint_rereference)
                        for _n in _jsn.drain_notes():
                            _log_rl("joint-note", "JOINT %s: %s" % (band_id, _n),
                                    every_s=10.0)
                    except Exception:
                        pass
                _dr_joint_shadow()
                deadreckon.dr_clock_solve(_ctx)
                deadreckon.dr_clock_quality(_ctx)
                # ---- ADOPT A BAND SIBLING'S CLOCK (--dr-clock-adopt) -------------------------
                # Runs AFTER the EMA above on purpose: a chain that solved its own clock this
                # cycle keeps it, and only a chain that cannot solve one (no detectors -> `offs`
                # empty -> the block above never ran) takes the sibling's. So enabling the flag
                # on a detector-bearing chain is a no-op rather than a silent override.
                #
                # IN-PROCESS SIBLING FIRST (task #27 M3). A chain co-hosted in this process
                # has already CONTRIBUTED its clock to the Receiver, so there is nothing to
                # serialise, flush, age or slew-test: the number is the same object the
                # sibling is using this cycle. The file route below stays for genuinely
                # cross-process siblings (the airspy benches, and a transitional split
                # deployment) and is unchanged. With one chain the lookup returns None and
                # this branch does not exist.
                _drp.rx_sib = (rx.dr_clock(band_id, exclude=chain_id, t_now=t0)
                           if (args.dr_clock_adopt and not _drp.offs) else None)
                # CROSS-BAND BOOTSTRAP (task #34). Without this a band whose chains all lack
                # detectors NEVER gets a clock: measured on sky, gal_e5b and bds_b2b sat at the
                # startup prime of 0.00 chips while gps_l5 had bootstrapped 150.74 and both
                # 1176.45 siblings had adopted it. 150 chips of error against a +-1 chip peak,
                # so dll_disc read -0.0008 (despreading noise) where E5a railed at -0.59.
                #
                # Layer 2 (the cross-band RATE) was necessary and NOT sufficient, and the
                # distinction is the whole point: a rate keeps you on the peak, a phase puts
                # you there. With the clock 150 chips out there is no peak to hold.
                #
                # ⚠️ THE ADOPTED PHASE CARRIES tau_band -- that is accepted, not overlooked. It
                # replaces a 150-chip error with a tau_band-sized one, which is inside the
                # DLL's pull-in, and the loop's steady-state residual then IS tau_band. The
                # per-band scoping exists to protect that measurement, and by blocking the
                # bootstrap it was the reason the measurement could never be made. Logged as
                # BOOTSTRAP, never as "ADOPTED ... same band", so the log distinguishes a
                # borrowed phase from a measured one.
                #
                # MODULUS: a clock may be reduced to a SHORTER code (150.74 mod 10230 is
                # well-defined) but never lengthened -- a value known mod 10230 says nothing
                # about which of the 100 periods of a 1023000-chip code it sits in. So a donor
                # whose code is shorter than ours is refused, which is the same-length guard
                # generalised rather than dropped.
                _rx_xband = False
                if _drp.rx_sib is None and args.dr_clock_adopt and not _drp.offs:
                    _cand = rx.dr_clock_any_band(exclude=chain_id, t_now=t0)
                    if _cand is not None and (_cand.extra.get("code_length") or 0) >= CODE_LEN:
                        _drp.rx_sib, _rx_xband = _cand, True
                if _drp.rx_sib is not None and _rx_xband:
                    _v = float(_drp.rx_sib.value) % CODE_LEN
                    if dr_state.get("clk") is None or abs(
                            ((_v - dr_state["clk"] + CODE_LEN / 2) % CODE_LEN)
                            - CODE_LEN / 2) > 0.5:
                        _log("dead-reckon: clock BOOTSTRAP %.2f chips from in-process chain "
                             "'%s' (CROSS-BAND -- carries tau_band; the DLL residual IS that "
                             "measurement)" % (_v, _drp.rx_sib.src))
                    dr_state["clk"] = _v
                    dr_state["clk_t"] = _drp.rx_sib.t
                    dr_state.pop("clk_primed", None)
                elif _drp.rx_sib is not None and _drp.rx_sib.extra.get("code_length") == CODE_LEN:
                    if dr_state.get("clk") is None or abs(
                            ((float(_drp.rx_sib.value) - dr_state["clk"] + CODE_LEN / 2)
                             % CODE_LEN) - CODE_LEN / 2) > 0.5:
                        _log("dead-reckon: clock ADOPTED %.2f chips from in-process chain "
                             "'%s' (same band %s, no file transport)"
                             % (float(_drp.rx_sib.value), _drp.rx_sib.src, band_id))
                    dr_state["clk"] = float(_drp.rx_sib.value) % CODE_LEN
                    dr_state["clk_t"] = _drp.rx_sib.t
                    # An adopted clock IS a measurement -- the sibling measured it -- so the
                    # prime is spent. If this chain ever gains detectors it should refine by
                    # EMA from here, not snap away from a good number.
                    dr_state.pop("clk_primed", None)
                    if _drp.rx_sib.extra.get("drift") is not None:
                        dr_state["drift"] = float(_drp.rx_sib.extra["drift"])
                elif _drp.rx_sib is not None:
                    # Same band, different code length: the chips are modular in a different
                    # period, so the number is numerically fine and physically meaningless.
                    # Refuse loudly rather than adopt a plausible wrong value.
                    _log_rl("clkadopt-len",
                            "dead-reckon: chain '%s' publishes a clock mod %.0f chips but "
                            "this chain's code is %.0f -- NOT adoptable across code lengths"
                            % (_drp.rx_sib.src, _drp.rx_sib.extra.get("code_length") or -1, CODE_LEN),
                            every_s=60.0)
                deadreckon.dr_clock_adopt(_ctx)
                # ---- MODEL-HEALTH GATES (design (b)): the resilience the fence never gave.
                # A fence on Doppler MOTION cannot detect a model that is simply WRONG -- a
                # stale ephemeris, a manoeuvre, a bad orbit. These can, and they use signals we
                # already compute every cycle:
                #   (1) EPHEMERIS FRESHNESS -- toe age.
                #   (2) INTEGRITY RESIDUAL -- measured-vs-predicted code phase, normally
                #       +-0.2 chips. If it blows up, the model is lying about this satellite.
                # A demoted satellite falls back to the SEARCH-measured Doppler, and because a
                # Doppler-source change is just another currency translation, the fallback
                # costs nothing: no re-anchor, no lock loss.
                # A single bad sample must never demote a satellite, and a gate that flaps is
                # worse than no gate: at startup the clock EMA is still converging, so the
                # residuals legitimately sit near the threshold and the first version of this
                # flapped UNTRUSTED/TRUSTED every few seconds (observed). Same discipline the
                # escape referee already uses: PERSISTENCE (N consecutive) + HYSTERESIS
                # (demote high, restore low) + do not judge at all until the clock has settled.
                if dr_state["clk"] is not None and dr_state.get("drift") is not None:
                    dr_state.setdefault("integ", {})
                    for prn_i, d_i in _drp.offs:
                        r_i = (((d_i - dr_state["clk"] + CODE_LEN / 2.0) % CODE_LEN)
                               - CODE_LEN / 2.0)
                        # exported for the escape referee's integrity veto (chips, this
                        # chain's code; search-vs-model with the solved clock removed)
                        dr_state["integ"][prn_i] = (r_i, _drp.now_w)
                        v_i = _drp.pd.get((_drp.tag, prn_i))
                        age_i = abs(v_i["toe_age_s"]) if v_i else 1e9
                        why = None
                        if age_i > args.dr_max_eph_age_s:
                            why = "ephemeris %.1f h old" % (age_i / 3600.0)
                        elif abs(r_i) > args.dr_max_integrity_chips:
                            why = "integrity residual %+.2f chips" % r_i
                        if why:
                            dr_bad[prn_i] = dr_bad.get(prn_i, 0) + 1
                        elif abs(r_i) < 0.5 * args.dr_max_integrity_chips:
                            dr_bad[prn_i] = 0          # hysteresis: restore well INSIDE
                        if (dr_bad.get(prn_i, 0) >= 3) and prn_i not in dr_untrusted:
                            dr_untrusted[prn_i] = why
                            # ⚠️ THIS MESSAGE USED TO SAY "falling back to the SEARCH-measured
                            # Doppler", which is not what happens under the default
                            # --seed-doppler auto: the seed loop skips the v_dr branch for an
                            # untrusted sat and lands on `pred` (the almanac predictor, which
                            # is BRDC too when --almanac-source brdc), NOT on the search's
                            # `dop`. The search Doppler reaches the seed only with
                            # --seed-doppler det. Corrected 2026-08-10 while tracing where
                            # GPS's carrier seed picks up its jitter -- a log line that names
                            # the wrong source sends the next reader to the wrong code.
                            _log("MODEL-UNTRUSTED PRN %d (%s, 3 consecutive) -> Doppler source "
                                 "falls back dr -> %s for this sat"
                                 % (prn_i, why,
                                    "pred (almanac)" if args.seed_doppler != "det" else "det"))
                        elif dr_bad.get(prn_i, 0) == 0 and prn_i in dr_untrusted:
                            del dr_untrusted[prn_i]
                            _log("MODEL-TRUSTED again PRN %d (integrity %+.2f chips)"
                                 % (prn_i, r_i))
                if dr_state["clk"] is not None and _drp.offs and _drp.now_w >= dr_state["log_next"]:
                    dr_state["log_next"] = _drp.now_w + 30.0
                    resid = ["PRN %d %+.2f a%.0f%s" % (p, r, det_age.get(p, -1),
                                                       " BAD" if abs(r) > 1.0 else "")
                             for p, d in _drp.offs
                             for r in [((d - dr_state["clk"] + CODE_LEN / 2) % CODE_LEN)
                                       - CODE_LEN / 2]]
                    _log("dead-reckon clock %.2f chips (%.3f us mod %.0f ms, drift "
                         "%+.3f chips/s); integrity: %s"
                         % (dr_state["clk"], dr_state["clk"] / args.chip_rate_hz * 1e6,
                            _drp.t_code * 1e3, dr_state.get("drift") or 0.0, "; ".join(resid)))
                # -- seed / re-pin every visible, undetected, unlocked sat from the model --
                _dr_seed()
            # a fresh detection re-anchors via the seed loop (search = fallback); a
            # dropped seed (set below horizon) clears the model-owned state with it.
            # #83 P3-3b: EXCEPT a model-primary PRN -- its detections feed the filter and
            # the referee, never the seed, so they must not evict its dr ownership (they
            # would every single cycle, orphaning the flip the moment it started).
            for prn in list(dr_state["seeded"]):
                if (prn in best and prn not in mp_flipped) or prn not in seeds:
                    dr_state["seeded"].discard(prn)
                    dr_state["pin"].pop(prn, None)

    def _stage_fleet_dll():
        """3c: THE FLEET DELAY-LOCK LOOP -- the instrument suite, then the control loop.
        
        Polls the fleet discriminator into `_dllp.fleet` (the cycle's central per-satellite state),
        runs the eight _instr_* diagnostics hung off that same poll, and finally calls the one routine
        that actuates: _stage_dll_control.
        
        ⚠️ THE q FLOOR IS MEASURED FROM THIS CYCLE'S OWN POPULATION, NEVER A CONSTANT. Summing
        tightens the noise distribution instead of raising q, so the correct bar FALLS as instances are
        added: any fixed constant is right for exactly one fleet size."""
        if args.dll_gain > 0.0:
            # FLEET COMBINE (docs/CHORD_GNSS_SHARED_DLL.md). Sum the RAW powers across every
            # instance that reports the same window, then form ONE discriminator. Ratios do not
            # sum -- (SUM E - SUM L)/(SUM E + SUM L) is not any function of the per-instance
            # dll_disc values -- which is exactly why the combiner publishes e_pow/l_pow/p_pow.
            # #79: the effective deep-gate set = hand-listed PRNs UNION the ones the search
            # is currently detecting. `True` (--dll-deep-gate all) stays absorbing.
            _dllp.deep_gate_eff = _deep_gate
            if args.dll_deep_gate_from_search > 0.0 and _deep_gate is not True:
                _fresh_dg = {p for p, _ts in _dls.deep_gate_seen.items()
                             if t0 - _ts <= args.dll_deep_gate_search_hold_s}
                if _fresh_dg:
                    _dllp.deep_gate_eff = set(_deep_gate or ()) | _fresh_dg
                    if _fresh_dg != _dg_auto_last[0]:
                        _log("DEEP GATE (auto, #79): search-admitted PRN %s at snr >= %.0f "
                             "(hold %.0f s)%s"
                             % (",".join(str(p) for p in sorted(_fresh_dg)),
                                args.dll_deep_gate_from_search,
                                args.dll_deep_gate_search_hold_s,
                                "" if not _deep_gate else
                                " + hand-listed %s"
                                % ",".join(str(p) for p in sorted(_deep_gate))))
                        _dg_auto_last[0] = set(_fresh_dg)
            _dllp.inst_hops = {}
            _dllp.fleet = fleet_dll(dll_combiners, dll_hop_window, args.dll_min_instances,
                              args.dll_quality_sigma, args.dll_quality_min,
                              deep_gate_prns=_dllp.deep_gate_eff,
                              deep_gate_margin=args.dll_deep_gate_margin,
                              # the noise ANCHOR for the presence floor (#49): without it
                              # the bar is built from the tracked population and becomes a
                              # peer competition. See the note in fleet_dll.
                              probe_prns=probe_set,
                              # #70: collect the per-instance newest hop from the poll we are
                              # already making. No extra HTTP -- fleet_dll parses pow_hop for
                              # the currency check and then aggregates the axis away.
                              src_hops=_dllp.inst_hops)
            # TASK #63: THE SAME DISCRIMINATOR, FORMED HERE FROM THE UN-SUMMED COMB. The powers
            # above were built by each tracker summing across its own channels -- "the one
            # combine the broker can never undo" -- and everything derived from them inherits
            # that. combdll rebuilds them from the per-channel Early/Prompt/Late the transport
            # ships, in the one place that can see the whole band.
            #
            # THE POLLED ARM IS STILL COMPUTED ABOVE, ON PURPOSE, AND FOR TWO REASONS: the deep
            # statistics are not in the comb (they come from the combiner's fold, so they are
            # handed across as coh_rows), and running both makes the swap self-monitoring --
            # the log line below is the same paired comparison the offline A/B makes, on the
            # loop's own cycles. Any cycle where the gather has nothing for this chain simply
            # keeps the polled fleet.
            instruments.instr_tap_walk(_ctx)
            # PROMPT HOLD for the NEXT cycle's lock gate (fold-independent, see
            # --lock-prompt-hold). Mutated in place, never rebound: the gate closes over it.
            _hold.prev.clear()
            _hold.q.clear()
            for _p, _fl in (_dllp.fleet or {}).items():
                _pm = _fl.get("p_med")
                if _pm:
                    _hold.prev[_p] = _fl["p_pow"] / _pm
                # ...and the FLEET DISCRIMINATOR QUALITY, from the same dict, for the same
                # next-cycle gate. See --lock-q: q is the metric this project judges lock on
                # everywhere EXCEPT, until now, the one decision that can destroy a lock.
                if _fl.get("q") is not None:
                    _hold.q[_p] = float(_fl["q"])
            # CROSS-NODE COHERENT COMBINE. Separate from the DLL on purpose: the DLL sums
            # POWERS (phase-free, which is what makes it cheap) while this removes the common
            # per-record PHASE, and the two answer different questions off the same endpoints.
            # It touches NO loop -- purely an observable, published for the viewer and the beam
            # map -- so a fault here can degrade what is displayed but can never move the code
            # or carrier loops.
            _dllp.fcoh = {}
            # TASK #59. The gather feed, when it is asked for AND actually has this chain's
            # windows. `source=None` falls straight through to the REST poll below, so a gather
            # that is down, restarting, or simply not carrying this chain costs a rate-limited
            # log line and nothing else -- the two feeds run side by side until the new one has
            # been shown ON SKY to be at least as good, and neither is load-bearing for the
            # other.
            _tsrc = None
            if telem_client is not None:
                # THE ALIGNMENT CHECK, PUBLISHED. `spread` is max-min of the instances' newest
                # window index: 0 or 1 is the transport working. Logged whether or not anything
                # consumes the feed, because the entire argument for this transport is that
                # misalignment becomes visible immediately instead of surfacing weeks later as
                # a physics anomaly (#46, #52, #53 all did).
                _st = telem_client.stats()
                _cs = _st["chains"].get(telem_chain)
                if not _cs:
                    _msg = "chain %s: nothing yet" % telem_chain
                elif not _cs.get("live"):
                    _msg = ("chain %s: ALL %d instances stale (%s)"
                            % (telem_chain, _cs["instances"], ",".join(_cs["stale"])))
                else:
                    # SPREAD IS OVER LIVE INSTANCES ONLY; the stale ones are NAMED. A stopped
                    # instance keeps its last window forever, so folding it into the spread
                    # turns every instance death into a four-digit alarm about alignment --
                    # which is the one number here that must stay trustworthy.
                    _msg = ("chain %s: %d live, win %d..%d spread %d%s"
                            % (telem_chain, _cs["live"], _cs["win_min"], _cs["win_max"],
                               _cs["spread"],
                               (" | STALE %s" % ",".join(_cs["stale"])) if _cs["stale"] else ""))
                _log_rl("telem-stat",
                        "TELEM %s frames %d gaps %d bad %d | %s"
                        % ("up" if _st["connected"] else "DOWN", _st["frames"], _st["gaps"],
                           _st["bad"], _msg),
                        every_s=30.0)
            if args.telem_coherent and telem_client is not None:
                try:
                    _tsrc = telem_client.coherent_source(
                        telem_chain, prns=set(seeds) or None, n_win=args.telem_windows, lag=1)
                    if not _tsrc[0]:
                        _tsrc = None
                        _log_rl("telem-empty",
                                "telem: no windows for chain %r yet -- falling back to "
                                "/get_records (gather stats: %s)"
                                % (telem_chain, telem_client.stats()))
                except Exception as e:
                    _tsrc = None
                    _log_rl("telem-src", "telem: source failed (%s); using /get_records" % e)
            if args.fleet_coherent:
                try:
                    _dllp.fcoh = fleet_coherent(dll_combiners, args.coh_min_instances,
                                          args.coh_min_records, prns=set(seeds) or None,
                                          log=None, floor_margin=args.coh_floor_margin,
                                          seed=int(_now()),
                                          # lets it fit the record-stream carrier rate off
                                          # the records it already fetched (#33 coarse feed).
                                          # ⚠️ HOPS, NOT RECORDS. get_records' first tuple
                                          # element is a HOP COUNT -- phaseslope.py divides
                                          # it by 195312.5, not by the record rate. Passing
                                          # hops_per_sec/2048 made the time axis 2048x too
                                          # long, so every fitted rate came out 2048x too
                                          # SMALL: +-0.005 Hz where the fold read +-10,
                                          # ratio 1907-2140 across satellites.
                                          hop_rate_hz=args.hops_per_sec,
                                          # #59: when set, the poll is skipped entirely and
                                          # this identical estimator runs on the gathered
                                          # records instead.
                                          source=_tsrc)
                except Exception as e:
                    _log_rl("fleet-coh", "fleet coherent: skipped this cycle (%s)" % e)
            # PATH B, same estimator, separate population. Reported side by side rather than
            # merged: blending the two streams would make the very comparison this exists to
            # support impossible, and their per-record noise is NOT independent (both despread
            # the same antenna voltages), so a merged sum would not buy the sqrt(2) it appears
            # to. Publishes nothing and touches no loop.
            _dllp.fcoh_n2 = {}
            if args.fleet_coherent and n2_combiners:
                try:
                    _dllp.fcoh_n2 = fleet_coherent(n2_combiners, args.coh_min_instances,
                                             args.coh_min_records, prns=set(seeds) or None,
                                             log=None, floor_margin=args.coh_floor_margin,
                                             seed=int(_now()))
                except Exception as e:
                    _log_rl("fleet-coh-n2", "fleet coherent (path B): skipped (%s)" % e)
            instruments.instr_coherent_rows(_ctx)
            # FLEET PHASE-SLOPE DELAY FIT (task #32, docs/CHORD_JOINT_TRACKING.md P1).
            # MEASUREMENT ONLY: touches no seed and no loop -- it is logged and published so
            # it can be judged against the disc, the E/L asymmetry and GPS's search-measured
            # code phase BEFORE anything consumes it (the #30 rule: measure the statistic
            # before the loop). Gated on its OWN flag, which is what keeps every recorded
            # transcript replaying byte-identically: replay is strict-ordered, an
            # unrecorded GET is a TRANSCRIPT DIVERGENCE, and old transcripts' argv does not
            # carry --spectrum-endpoints, so replays never issue the new polls.
            _dllp.spec_fit = {}
            instruments.instr_spectrum_fit(_ctx)
            # THE SERVED C/N0 (task #57): per-record prompt power off the gather feed,
            # q-gated, debiased against the noise probes. Fits NOTHING -- the deep fold's
            # per-integration rate re-search is a fit on something the tracking loop already
            # fixed, and its ~20 dB of paired self-scatter is why cn0_coh_db cannot be the
            # radiometry. Measurement-only: touches no loop, issues no polls (the telemetry
            # client is a push stream, so recorded transcripts replay byte-identically).
            # Needs the probes as its noise anchor; without them it publishes nothing rather
            # than falling back to the peer competition (#49's lesson).
            # Throttle (--estimator-every-s): both telemetry-walk estimators run together,
            # at most this often; the last values keep being served in between. Defined
            # HERE because this is the FIRST of the two blocks in cycle order.
            _dllp.run_est = (telem_client is not None and probe_set
                        and _now() >= _est_next[0])
            if _dllp.run_est:
                _est_next[0] = _now() + args.estimator_every_s
            # ⚠️ THE THROTTLE EXISTS FOR THE *WALK*, NOT FOR THE ESTIMATOR. Its whole
            # justification (see --estimator-every-s) is that these are "pure-Python walks over
            # ~1500 record decodes each", which at every-cycle cadence across five chains ate
            # ~75% of the interpreter and starved the telemetry reader. With --comb-taps-cpp
            # armed, PROMPT-CN0 no longer walks anything: it fetches an already-reduced series
            # from the gather. So the reason to throttle IT is gone, and throttling it now only
            # costs the served C/N0 its freshness -- the rows keep serving a value up to
            # --estimator-every-s old for no saving at all.
            #
            # KCOH still walks and stays throttled. They were gated together because they had
            # the same cost; they no longer do, so they no longer share a gate.
            _dllp.run_pcn0 = _dllp.run_est or (args.comb_taps_cpp >= 2 and args.fleet_trim_url
                                     and telem_client is not None and probe_set)
            _dllp.pcn0 = _est_last["pcn0"]
            instruments.instr_prompt_cn0(_ctx)
            # THE KNOWN-RATE COHERENT C/N0 (task #57 step 3): the ~1 s fold with the rate
            # INJECTED from the PREVIOUS cycle's record-stream fit (_kcoh_rates, updated
            # below AFTER the fold so this integration never consumes a rate estimated
            # from itself). No search, no q gate -- a fixed-rate fold over noise is noise,
            # which is why this cannot fire on noise the way the deep fold did, and why it
            # reaches the deep-sidelobe satellites a per-record gate never passes.
            # Measurement-only; rides the same telemetry the comb DLL uses.
            # Same throttle gate as PROMPT-CN0 above (_run_est, set there -- the first of
            # the two telemetry-walk blocks in cycle order).
            _dllp.kcoh = _est_last["kcoh"]
            instruments.instr_kcoh(_ctx)
            # NOW the rates for the NEXT cycle, from THIS cycle's record-stream fit.
            for _p, _fc2 in (_dllp.fcoh or {}).items():
                _r2 = _fc2.get("rate_hz")
                if _r2 is not None:
                    _kcoh_rates[_p] = float(_r2)
            # THE PER-ELEMENT COMPLEX GAIN (task #57 step 2): amplitude AND phase per
            # antenna, per instance -- the beam/peel coefficients. Assembled from the
            # combiners' /get_elements (raw leave-one-out cross-products accumulated per
            # record node-side, where the element axis lives), significance-anchored on the
            # noise probes per (instance, element). Measurement-only; touches no loop.
            # ── #8: RF-PATH HEALTH, POLLED ──────────────────────────────────────────────
            # Clip fraction and per-band power from each GPU's voltage tap. Measurement
            # only: it steers nothing and gates nothing, exactly like the element poll below.
            #
            # ⚠️ THIS IS THE ONE NUMBER THAT SEPARATES "LOUD" FROM "RAILED", and we have
            # never had it. The fleet's amplitudes swing 5-10x an hour (#56) with the root
            # upstream of both tracking and the combine, and on 08-18 something lit up the
            # band hard enough to take chains down for hours -- with no way to tell whether
            # the front end saturated or merely saw a large linear signal. Those two have
            # different fixes and we could not distinguish them.
            #
            # Grouped into LOBES by channel contiguity (rf_lobes), because the tap's channel
            # list is the union of every chain's covering set and therefore arrives as one
            # run per band. A band-selective source is only diagnosable against a band that
            # was quiet in the SAME sample.
            instruments.instr_rf_stats(_ctx)

            # ⚠️ NEVER THROTTLE IN REPLAY. A transcript is an ordered recording of every GET,
            # checked by URL as it is consumed, so making FEWER calls than the recording
            # desynchronises the stream: the next endpoint reads the previous one's response
            # and `_get` catches the divergence per call, so the replay limps on producing
            # garbage instead of stopping. Found the hard way -- this moved the holds digest
            # and it looked like the (unrelated) _solve refactor until the two arms were
            # diffed and BOTH gave the same moved digest. "TRANSCRIPT DIVERGENCE at get #116"
            # was in the replay's own stderr the whole time.
            #
            # The general rule: broker_equiv can gate what a call COMPUTES, never how OFTEN it
            # is made. Any cadence change is live-only and needs a live measurement.
            instruments.instr_element_poll(_ctx)
            # #83 2(d): the served innovation -- freshest value + 10-minute p95 per PRN.
            # The statistic is cut by TIME here at read, not by count at write: a PRN
            # detected once in ten minutes reports that one sample, and a PRN that set
            # 20 minutes ago stops being served instead of fossilizing its last window.
            _dllp.innov_pub = {}
            for _p, _ih in list(innov_hist.items()):
                if _now() - _ih[-1][0] > 1200.0:
                    del innov_hist[_p]
                    continue
                _win = [(tt, vv) for tt, vv in _ih if _now() - tt <= 600.0]
                if not _win:
                    continue
                _av = sorted(abs(vv) for _, vv in _win)
                _dllp.innov_pub[_p] = {
                    "innov_chips": _win[-1][1],
                    "innov_age_s": _now() - _win[-1][0],
                    "innov_p95_10m": _av[max(0, math.ceil(0.95 * len(_av)) - 1)],
                    "innov_n_10m": len(_win),
                }
            # #83 P3-3a: the MODEL innovation rides the same rows -- minnov_* keys. A PRN
            # can carry either or both (INNOV needs a standing seed, MINNOV an established
            # joint row); absent keys mean "not measurable", never zero.
            for _p, _mh in list(minnov_hist.items()):
                if _now() - _mh[-1][0] > 1200.0:
                    del minnov_hist[_p]
                    continue
                _win = [(tt, vv) for tt, vv in _mh if _now() - tt <= 600.0]
                if not _win:
                    continue
                _av = sorted(abs(vv) for _, vv in _win)
                _dllp.innov_pub.setdefault(_p, {}).update({
                    "minnov_chips": _win[-1][1],
                    "minnov_p95_10m": _av[max(0, math.ceil(0.95 * len(_av)) - 1)],
                    "minnov_n_10m": len(_win)})
            # ── #83 P3-3b: THE FLIP DECISION (see --model-primacy-max) ──
            # One writer for mp_flipped, once per cycle, from the MEASURED p95s built
            # above. ENTER: p95 < gate with enough samples; the best-p95 eligible PRNs
            # fill the cap, the rest are the in-poll controls. EXIT: p95 beyond the
            # hysteresis bound, or the referee starved (no detection while flipped for
            # --model-primacy-starve-s). Every transition is one loud line -- a flip
            # whose firing cannot be seen is how gates fail here.
            instruments.instr_model_primacy(_ctx)
            if _dllp.innov_pub:
                _log_rl("innov",
                        "INNOV %s: %s"
                        % (log_tag() or args.signal,
                           " ".join("%d:%+.2f(p95 %.2f, n%d)"
                                    % (_p, v["innov_chips"], v["innov_p95_10m"],
                                       v["innov_n_10m"])
                                    for _p, v in sorted(_dllp.innov_pub.items())
                                    if "innov_chips" in v)),
                        every_s=60.0)
                _mv = ["%d:%+.2f(p95 %.2f, n%d)"
                       % (_p, v["minnov_chips"], v["minnov_p95_10m"], v["minnov_n_10m"])
                       for _p, v in sorted(_dllp.innov_pub.items()) if "minnov_chips" in v]
                if _mv:
                    _log_rl("minnov",
                            "MINNOV %s (model vs sky, flip-gate statistic): %s"
                            % (log_tag() or args.signal, " ".join(_mv)),
                            every_s=60.0)
            if publisher is not None:
                # Published BEFORE the trim update so the row shows the state the loop acted
                # on, not the state after it acted -- otherwise a reader can never see the
                # input that produced a given correction.
                publisher.update(_dllp.fleet, seeds, _dls.trim, len(dll_combiners), last_dets, _dllp.fcoh,
                                 pcn0=_dllp.pcn0, kcoh=_dllp.kcoh, innov=_dllp.innov_pub,
                                 cpp_trim={_p: (_r.get("trim_chips") or 0.0)
                                           for _p, _r in _dls.readback.items()})
            _dllp.report = []
            codeloop.stage_dll_control(_ctx)
            if _dllp.report:
                _log("DLL: " + "; ".join(_dllp.report))
            # CODE-DERIVED CARRIER ERROR, logged only -- not applied yet.
            #
            # Carrier and code are locked in ratio: a Doppler error dF drifts the code at
            # dF * f_chip/f_carrier (validated offline, STATE 7.2: measured 0.01476 chips/s
            # against 0.01476 predicted). Inverting, the fitted cp slope gives the carrier error
            # directly, at f_carrier/f_chip = 115.03 Hz per chip/s.
            #
            # This matters because carrier_hz_resid is signal-free (2026-08-04: |resid| median
            # 0.519 Hz on satellites with signal, 0.492 Hz on satellites without), so the carrier
            # loop was integrating noise and is now off. The code side, by contrast, is strong --
            # sustained q ~ 3.2 and an 8-point cp fit.
            #
            # LOGGED, NOT APPLIED. Three loops today were found eating a statistic that did not
            # measure what its name said; this one gets compared against the estimator it would
            # replace before it is allowed to move anything.
            if args.carrier_from_code:
                _k = args.carrier_hz / args.chip_rate_hz
                _rows = []
                for _p in sorted(_cpt.fit_slope):
                    _rec = status.get(_p, {})
                    _meas = float(_rec.get("carrier_hz_resid", 0.0))
                    _sig = sig_of(_rec)
                    if _sig < args.lock_snr:
                        continue
                    _rows.append("PRN %d code->%+.2f Hz meas %+.2f Hz (sig %.1f)"
                                 % (_p, _cpt.fit_slope[_p] * _k, _meas, _sig))
                if _rows:
                    _log_rl("carfromcode", "CARRIER-FROM-CODE (shadow): " + "; ".join(_rows[:6]),
                            every_s=30.0)
            if dop_rate_rejected:
                _log("dop-rate: %d fit(s) REJECTED against the model (kept the model): %s"
                     % (len(dop_rate_rejected),
                        ", ".join("PRN %d fit %+.3f vs model %+.3f" % (k, v[0], v[1])
                                  for k, v in sorted(dop_rate_rejected.items())[:5])))
            _absent = sorted(p for p in seeds
                             if seeds[p].get("doppler_rate_hz_s") is None)
            if _absent:
                # No carrier extrapolation AND no quadratic code term for these sats.
                _log("dop-rate: %d seeded PRN(s) have NO doppler rate at all: %s"
                     % (len(_absent), _absent[:8]))
            if dop_rate_fitted:
                _log_rl("doprate", "doppler-rate FIT seeded on %d sat(s): %s"
                        % (len(dop_rate_fitted),
                           "; ".join("PRN %d %+.4f Hz/s" % (k, v)
                                     for k, v in sorted(dop_rate_fitted.items())[:5])),
                        every_s=60.0)
            # The BAR, every cycle it is measured. A threshold on a noisy statistic that is
            # never printed is a threshold nobody can audit -- and this one legitimately moves
            # with the fleet size, so a reader has to be able to see where it went.
            if _dllp.fleet:
                any_fl = next(iter(_dllp.fleet.values()))
                _log_rl("dll-floor",
                        # WHICH ARM produced these numbers. Since #63 this line can describe
                        # either the polled powers or the ones formed here from the comb, and
                        # a floor with no provenance is exactly the kind of number that gets
                        # compared across a switch without anyone noticing it changed source.
                        "fleet DLL [%s]: %d PRN(s) over %d combiner(s), %d present, "
                        "q floor %.2f%s"
                        % (any_fl.get("src") or "polled", len(_dllp.fleet), len(dll_combiners),
                           sum(1 for v in _dllp.fleet.values() if v["present"]), any_fl["q_floor"],
                           "" if any_fl["q_med"] is None
                           else " (noise median %.2f, sigma %.3f)"
                                % (any_fl["q_med"], any_fl["q_sigma"])))
            # ── THE INSTANCE LIVENESS GUARD (#70, 2026-08-18) ──────────────────────────
            # WHY THIS EXISTS, and why the guard we already had could not do it. On 08-18's
            # full-fleet restart FOUR instances came up wedged -- cx42/gnss0, cx43/gnss0,
            # cx44/gnss1, cx51/gnss0 -- each with its DPDK capture window frozen and the
            # ENTIRE 195,313 pkt/s stream being dropped, and each serving plausible,
            # well-formed rows to every poll for as long as it was left alone. An earlier
            # instance of the same fault ran on cx19 for 25 HOURS and 18.7 billion dropped
            # packets before a human noticed.
            #
            # ⚠️ "ALL 12 RESPOND" IS NOT "ALL 12 ARE ALIVE". That is the whole trap, and it
            # is why this keys on a COUNTER and never on reachability: healthy is ~5.9M
            # hops per 30 s, wedged is exactly 0, and both answer 200.
            #
            # ⚠️ AND IT IS NOT --fe-axis-stale-s, which watches the MAXIMUM hop over
            # instances. That question ("has the time base frozen?") is real and that guard
            # caught the cx19 collapse -- but eleven healthy instances keep the maximum
            # climbing, so it was correctly silent through all four wedges. The axis it
            # cannot resolve is per-instance, and this is that axis. Both stay.
            #
            # FREE: the hop comes from the fleet DLL poll above, which already parses
            # pow_hop for its currency check and then aggregates the per-instance axis away.
            # The decision is a pure function in fits.py (test_instance_stall.py) with a
            # CONTROL CLAUSE -- if most of the fleet is also standing still this is global
            # (a paused F-engine, a replay, a clock step) and accusing an instance would
            # point the next hour in the wrong direction, so it says nothing instead.
            # ── LINK 1 OF THE WALKOFF CHAIN, MEASURED PER INSTANCE ────────────────────
            # ⚠️ THE POPULATION IS THE POINT, AND I GOT IT WRONG ONCE. The obvious place for
            # this is beside `_fh = max(pow_hop)` in the status block -- but that `status` is
            # `{prn: row}` from ONE combiner, and every PRN in a poll carries the SAME
            # pow_hop, so it reports spread 0.00 s across 32 rows and says nothing about the
            # fleet ([[identical-numbers-are-not-agreement]]). The axis lag that link 1 is
            # about is ACROSS INSTANCES: on 2026-08-23, gps_l5/gal_e5a read -18.0..-19.3 s
            # while bds_b2a read -7.8 s at the same instant. `_inst_hops_now` is the right
            # population -- the newest hop per combiner, from the poll fleet_dll already makes.
            #
            # Sign convention matches _dax below: NEGATIVE = that instance lags the wall.
            # POSITIVE = a hop the F-engine has not reached, i.e. link 2's FUTURE HOP, which
            # is impossible for a processed record and names the instance serving it.
            if utc0_sample0 and _dllp.inst_hops:
                _w = _now()
                _ia = sorted(((utc0_sample0 + float(h) / args.hops_per_sec) - _w, str(k))
                             for k, h in _dllp.inst_hops.items() if h)
                if _ia:
                    _fut = [k for d, k in _ia if d > 0.5]
                    _log_rl("axis-inst",
                            "AXIS INST: n=%d  lag median %+.2f s  worst %+.2f s (%s)  "
                            "freshest %+.2f s (%s)  spread %.2f s%s"
                            % (len(_ia), _ia[len(_ia) // 2][0], _ia[0][0], _ia[0][1],
                               _ia[-1][0], _ia[-1][1], _ia[-1][0] - _ia[0][0],
                               "" if not _fut else
                               "  *** %d FUTURE instance(s) >0.5 s AHEAD: %s"
                               % (len(_fut), ",".join(sorted(_fut)[:4]))),
                            every_s=30.0)
            if args.instance_stall_s > 0 and _dllp.inst_hops:
                _ih, _stalled = instance_stall_verdict(
                    dr_state.get("inst_hops", {}), _dllp.inst_hops, t0,
                    args.instance_stall_s)
                dr_state["inst_hops"] = _ih
                if _stalled:
                    _log_rl("inststall",
                            "⚠️ INSTANCE STALLED: %d of %d serving but NOT ADVANCING -- %s. "
                            "These answer 200 with plausible rows; their pow_hop has not "
                            "moved (healthy is ~5.9M hops/30 s). The usual cause is a frozen "
                            "DPDK capture window dropping the whole stream, which no amount "
                            "of waiting clears -- check the node log for RESYNC lines, then "
                            "restart that node. Bandwidth is degraded fleet-wide until then."
                            % (len(_stalled), len(_ih),
                               ", ".join("%s stuck at hop %d for %.0f s" % (u, h, dt_)
                                         for u, h, dt_ in _stalled[:4])),
                            every_s=300.0)

            # ── THE q STALL GUARD (#70/#87, 2026-08-18) ────────────────────────────────
            # WHY THIS EXISTS: on 08-18 three chains sat in a degraded steady state for
            # 3.5 h after the RFI outage railed their C++ trims -- gal_e5b at q_med 1.09
            # against its own 2.09 baseline -- and NOTHING said so. The numbers were in
            # this very log the whole time; it took a same-time-of-day comparison, run by
            # hand because someone asked for a status check, to see it. A degradation
            # that only a human comparison can find is a degradation that runs for hours.
            #
            # WHAT IT WATCHES: this chain's own q duty (fraction of judged sats at
            # q >= --q-stall-bar) over a trailing window, against the BEST duty this chain
            # has reached in this process's lifetime. Self-referential on purpose: chains
            # differ 4x in duty by construction (#49), so a fleet-common bar would either
            # cry wolf on bds_b2b or never fire on gps_l5. The baseline only RISES, so a
            # chain that degrades cannot quietly redefine "normal" downward -- which is
            # exactly how the 3.5 h went unnoticed.
            #
            # ⚠️ IT IS A NOTICE, NOT A CONTROL. It changes nothing and gates nothing: a
            # guard that acts on a statistic this noisy would be a new failure mode, and
            # the honest recovery (a NODE restart, which clears the C++ trim state a
            # broker restart cannot) is not the broker's to perform.
            if args.q_stall_window > 0 and _dllp.fleet:
                _qs = [v["q"] for v in _dllp.fleet.values() if v.get("q") is not None]
                if _qs:
                    _duty = sum(1 for q in _qs if q >= args.q_stall_bar) / float(len(_qs))
                    _qh = dr_state.setdefault("q_hist", [])
                    _qh.append((t0, _duty))
                    del _qh[:max(0, len(_qh) - 4000)]
                    # The decision itself is a PURE function in fits.py so it can be
                    # tested against a constructed collapse (test_q_stall.py) -- the
                    # on-sky fixtures run ~11 cycles at a duty that never falls, so a
                    # replay cannot distinguish "did not fire" from "cannot fire".
                    _qb, _qv = q_stall_verdict(_qh, t0, args.q_stall_window,
                                               args.q_stall_frac, args.q_stall_min_best,
                                               dr_state.get("q_duty_best"))
                    dr_state["q_duty_best"] = _qb
                    if _qv is not None:
                        _log_rl("qstall",
                                "⚠️ q STALL: duty %.2f over the last %.0f s vs %.2f best "
                                "this session (%.0f%% of it, bar q>=%.1f, %d sat(s)). "
                                "This chain has been degraded, not absent -- the usual "
                                "cause is railed C++ trim state, which a BROKER restart "
                                "does NOT clear (#87). Judge against the same time of day, "
                                "then a NODE restart."
                                % (_qv[0], args.q_stall_window, _qv[1], 100.0 * _qv[2],
                                   args.q_stall_bar, len(_qs)),
                                every_s=args.q_stall_notice_s)
            for k in list(_dls.trim):
                if k not in seeds:
                    del _dls.trim[k]
                    _dls.last_hop.pop(k, None)

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
        rx=rx, publisher=publisher, telem_client=telem_client, detectors=detectors,
        dll_combiners=dll_combiners, spectrum_endpoints=spectrum_endpoints,
        spec_writer=_spec_writer, state_dir=_state_dir, xb_read_dir=_xb_read_dir,
        sig_of=sig_of, combiner=combiner, gating=gating, capable=_capable,
        receiver_state=receiver_state, alm_now=_alm_now, cb=_cb,
        almanac_sats=almanac_sats, brdc_alm=brdc_alm, det_fresh=det_fresh,
        state_w=state_w, clk_persist_t=_clk_persist_t,
        car=_carrier, wd=_watchdog, nho=_nho, dls=_dls, hold=_hold, cpt=_cpt,
        trackers=trackers, joint_consume=joint_consume, broker_t0=broker_t0,
        dr_eph_mod=dr_eph_mod, dr_min_prn=dr_min_prn,
        hist_len=HIST_LEN, max_gap_hops=MAX_GAP_HOPS, q_alias_hz=Q_ALIAS_HZ,
        carrier_explain_hz=CARRIER_EXPLAIN_HZ, carrier_verify_emits=CARRIER_VERIFY_EMITS,
        fuse_cached=_fuse_cached, cp_to_seed_currency=cp_to_seed_currency,
        sig_of_last=sig_of_last,
        dllp=_dllp, drp=_drp, handover=_handover, adm_gate=_adm_gate, g3_ramp=_g3_ramp,
        seeds=seeds, dr_state=dr_state, bsat=bsat, cp_held=cp_held,
        dr_untrusted=dr_untrusted,
        est_last=_est_last, kcoh_rates=_kcoh_rates, rf_last=_rf_last,
        elem_arch_t=_elem_arch_t, elem_poll_t=_elem_poll_t,
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
                t_det = (utc0_sample0 + _b[3] / args.hops_per_sec
                         if utc0_sample0 else t0)
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
        _xb_pred = {}   # prn -> cross-band predicted Doppler for THIS band (bias-removed)
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
                        _xb_pred[_p] = (_ds - _LOsib) * _ratio + _LOown - _bias
                        # SHADOW: accumulate the residual for every dual-tracked sat
                        _own = status.get(_p) or {}
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
                                % (xband, len(_xb_pred), len(_xb_resid), _med,
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
        _stage_narrow_search()

        # refresh / add consensus seeds: code phase from the search, Doppler from the
        # orbit prediction when available (precise enough for coherent integration),
        # else the coarse search grid.
        la_samples = []   # per-sat (l-a) estimates this cycle, from sats with a good code-rate fit
        fitted = set()    # PRNs that got their own >=3-snapshot slope fit this cycle
        cl_report = []    # CL time-assist per-PRN (k, fine-time residual) log lines this cycle
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
        if args.time0_endpoint and utc0_sample0 and _now_anchor - _anchor_chk[0] > 60.0:
            _anchor_chk[0] = _now_anchor
            try:
                _fresh = float(_get("%s/%s" % (base, args.time0_endpoint.strip("/")))
                               .get("time0_ns", 0.0)) / 1e9
                if _fresh and abs(_fresh - utc0_sample0) > 1e-3:
                    _log("*** TIME ANCHOR CHANGED: frame0 was %.9f, endpoint now reports %.9f "
                         "(%+.3f days). The F-engine has been restarted. EVERY SEED THIS BROKER "
                         "SENDS IS WRONG BY THAT AMOUNT, and every node still running cached the "
                         "old epoch too. Restart the nodes AND this broker."
                         % (utc0_sample0, _fresh, (_fresh - utc0_sample0) / 86400.0))
            except Exception:
                pass   # endpoint down is the normal outage case, already logged elsewhere

        if (args.cl_assist or args.cl_tracker or dr_state is not None) and not utc0_sample0:
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
                    utc0_sample0 = rx.time_anchor(
                        lambda: float(_get("%s/%s" % (base, args.time0_endpoint.strip("/")))
                                      .get("time0_ns", 0.0)) / 1e9,
                        chain_id) or 0.0
                    if utc0_sample0:
                        _log("time anchor: CHORD F-engine frame0 = %.9f s (GPS-disciplined)"
                             % utc0_sample0)
                        _anchor_seen[0] = utc0_sample0
                else:
                    utc0_sample0 = rx.time_anchor(
                        lambda: float(_get("%s/%s/adcstat" % (base, args.adc_stage))
                                      .get("utc0_sample0", 0.0)),
                        chain_id) or 0.0
                    if utc0_sample0:
                        _log("CL time-assist: capture sample-0 UTC anchor %.3f" % utc0_sample0)
            except Exception as e:
                _log("time anchor unavailable (%s); retrying" % e)
        dr_pd = (dr_state or {}).get("pd") or {}
        dr_pd2 = (dr_state or {}).get("pd2") or {}
        dr_pd0 = (dr_state or {}).get("pd0") or {}
        _stage_detections_to_seeds()

        # 3. COAST / drop (the trajectory-predictor promotion). A visible sat is coasted through a
        # signal dropout (radar sweep, brief fade): its seed is held and its Doppler forecast
        # forward from the orbit + clock each poll, so the tracker keeps despreading at the
        # PREDICTED trajectory and re-peaks when the signal returns -- the lock survives the gap
        # instead of being pruned and re-acquired. The code prediction holds for ~the coast budget;
        # drop ONLY when the sat SETS (the unambiguous "gone") or |A| stays down for the whole
        # budget (genuine loss / prediction breakdown). |A| recovering resets the coast.
        coast_polls = max(1, int(round(args.coast_budget / max(args.interval, 1e-3))))
        try:
            status = {int(r["prn"]): r for r in _get("%s/get_status" % combiner)}
            if args.almanac_epoch:
                _u = [float(r["utc"]) for r in status.values() if r.get("utc")]
                if _u:
                    _alm_file_pos[0] = max(_u) - args.almanac_epoch_utc0
            # #83 THE AXIS FIX: capture the newest F-engine hop AT FETCH TIME. The pair
            # (hop, wall-at-fetch) lets the dr block build its "now" on the F-engine axis
            # with wall entering only as the elapsed-since-fetch difference.
            _fh = max((float(r.get("pow_hop") or 0.0) for r in status.values()),
                      default=0.0)
            if _fh > 0.0:
                # ⚠️ THE TIME BASE MUST NEVER FREEZE SILENTLY (2026-08-18, the cx19 collapse).
                # t_now_abs is built from this hop, and this hop comes from ONE combiner. When
                # cx19/gnss0 deadlocked its capture window, its pow_hop stopped advancing while
                # the broker kept polling it happily -- so t_now_abs froze, det_age went
                # NEGATIVE and grew at 1 s/s, every integrity residual read as enormous, the
                # clock solve declared "the CLOCK moved" and latched, and all four chains that
                # ADOPT that clock lost their seeds together (#75). Not one line said the time
                # base had stopped. This is that line.
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
                fe_axis[0] = (_fh, _now())
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
            status = {}
            _log("get_status failed: %s" % e)
        _ctx.begin_cycle(status=status)
        # P7a nav-bit predictor: fold in this cycle's bit observations (rows carry nav_obs
        # only when the combiner runs bit_export, so non-GPS chains skip this for free).
        _stage_nav_bits()
        # Lock metric: the detection SIGNIFICANCE (sigma above noise) -- the deep nav-wiped SNR when
        # available, else the noise-debiased incoherent SNR -- not the raw |A|. The incoherent |A| is
        # biased by the noise floor (~the floor for weak sats), so judging "still locked" by |A| >
        # drop_amplitude let phantoms coast forever (|A| never falls below the floor). sig ~1 = noise,
        # >>1 = a real lock. Falls back to |A| only if the combiner reports no significance at all.
        have_sig = any(sig_of(r) > 0 for r in status.values())
        # NOISE PROBES (--noise-probes N): keep the N deepest-below-horizon PRNs seeded so
        # the combiner emits GENUINE noise records for them -- the beam map's pedestal
        # calibration (clip bias of the moment debias + the coherent estimator's selection
        # residue) needs signal-free samples, and an almanac-gated broker otherwise never
        # tracks one (2026-07-12: the GPS pedestal fell back to a signal percentile,
        # x=10.6 ~ 40 dB-Hz, blinding the map's low end). Probes are exempt from the
        # set-below-horizon drop and invisible to hints/hold/DLL/carrier (their sig ~ 0
        # fails every gate naturally); dop/cp are arbitrary for noise -- predicted values
        # keep the despread configuration representative.
        probe_set = set()
        _ctx.begin_cycle(probe_set=probe_set)
        if args.noise_probes > 0 and args.almanac and _ctx.pred:
            deep_low = sorted((p for p, v in _ctx.pred.items() if v[2] < -15.0),
                              key=lambda p: _ctx.pred[p][2])[:args.noise_probes]
            probe_set = set(deep_low)
            for p in deep_low:
                if p not in seeds:
                    _log("noise probe PRN %d seeded (elev %.0f)" % (p, _ctx.pred[p][2]))
                seeds[p] = Seed.born(
                    "probe", epoch=0,
                    doppler_hz=_ctx.pred[p][0] + _cb.value,
                    code_phase_chips=0.0,
                    code_phase_rate=cp_rate_from_code_bias(
                        _ctx.pred[p][0], code_bias_ema or 0.0, args.hops_per_sec,
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
        _stage_coast_drop()

        # 3e. DEAD-RECKONED CODE-PHASE SEEDING (--dead-reckon): the search only exists to
        # measure what the model already knows. BRDC ephemeris (~2 m orbits + ~5 ns sat
        # clocks) plus the receiver clock solved from the sats we DO detect predict every
        # other visible sat's code phase to well inside the DLL capture range (0.10 chip
        # rms validated, gnss_deadreckon_check.py 2026-07-13) -- so seed them all:
        # sub-threshold sats despread on-peak with no detection ever required (the
        # sidelobe-mapping mode). The search demotes to bootstrap (clock solve), fallback
        # (a detection re-anchors via the normal seed loop, which also removes the PRN
        # from the model-owned set below) and integrity check (residuals logged here).
        _stage_dead_reckon()

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
        _stage_fleet_dll()

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
        _rr_resid, _rr_cons = {}, None
        if (args.carrier_source == "rate"
                and (args.carrier_gain > 0.0
                     or (args.joint_shadow and args.detectors
                         and not args.rrate_state))):
            try:
                _rr_resid, _rr_cons = rate_residuals(
                    status, args.carrier_rate_min_q, args.carrier_rate_clip_hz,
                    _log if args.carrier_gain > 0.0 else None,
                    prev_hop=rate_prev_hop, max_gap=args.carrier_rate_max_gap,
                    prev_val=rate_prev_val, max_step=args.carrier_rate_max_step,
                    unit_hop=rate_unit_hop)
            except Exception as e:
                _log_rl("rate-resid-err", "rate_residuals skipped: %s" % e, every_s=300.0)
        if (args.joint_shadow and args.detectors and args.carrier_source == "rate"
                and args.carrier_gain <= 0.0):
            try:
                _fr_resid, _fr_cons = _rr_resid, _rr_cons
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
        rr_full_ok = False
        if args.rrate_state and args.carrier_source == "rate":
            rr_full_ok = any(isinstance(_r, dict) and _r.get("deep_rate_full_q") is not None
                             for _r in (status or {}).values())
            try:
                _fd = ("deep_rate_full_hz", "deep_rate_full_q") if rr_full_ok \
                    else ("deep_rate_hz", "deep_rate_q")
                _rr2_resid, _ = rate_residuals(
                    status, args.carrier_rate_min_q, args.carrier_rate_clip_hz, None,
                    prev_hop=rrate_prev_hop, max_gap=args.carrier_rate_max_gap,
                    prev_val=rrate_prev_val, max_step=args.carrier_rate_max_step,
                    unit_hop=rate_unit_hop, rate_field=_fd[0], q_field=_fd[1])
            except Exception as e:
                _rr2_resid = {}
                _log_rl("jrr-err", "rrate residuals skipped: %s" % e, every_s=300.0)
        else:
            _rr2_resid = {}
        if args.rrate_state and _rr2_resid:
            # REGIME GATE (arm 4's lesson, 2026-08-13 15:4x). The full-band field revealed
            # a population the capped view called noise: STRONG but DECOHERED sats (PRN 27
            # at amp 56, coh_frac 0.02) carrying REAL multi-Hz carrier residuals that SWING
            # +-10 Hz poll-to-poll -- seed/f_ref churn, not orbit error. The rrate model is
            # a slow per-sat drift; fed those swings it rejects 2:1 and the escape
            # snap-moves rows. Feed only sats whose regime the model fits: COHERING ones
            # (same bar as the joint code feed's _track_ok). The decohered population is
            # its own open question -- the full field finally makes it VISIBLE (#48).
            _rr2_resid = {
                _p: _rv for _p, _rv in _rr2_resid.items()
                if ((status.get(_p) or {}).get("coherence_s") or 0.0) > 0.0
                or ((status.get(_p) or {}).get("coh_frac") or 0.0) >= 0.3}
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
                _jrk = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference)
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
                    _yk = _rem + (rr_cmd_applied.get(_p, _carrier.trim.get(_p, 0.0))
                                  if args.rrate_feed_applied else 0.0)
                    _sigk = min(0.3, max(0.03, 2.0 / math.sqrt(_kv["sig"])))
                    _k2 = (args.dr_constellation, int(_p))
                    if _jrk.update_rrate(_k2, _yk, _drp.t_now_abs, args.carrier_hz,
                                         sigma_hz=_sigk) is not None:
                        rr_kcoh_t[_p] = t0
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
        _stage_rate_feed_coarse()
        # 3d''. PLL FINE STAGE (#33 phase-step feed). The ADR's residual half
        # (res_cycles), differenced over the poll span: ~5 mHz where the rate spectrum's
        # floor is ~60. SHADOW ALWAYS (the JRRP line calibrates the r2c sign against the
        # coarse observable on the uncommanded chains' standing residuals); FEED only when
        # --rrate-phase-feed is set AND the sign is calibrated AND, per sat: same arc,
        # counter advanced (adr_fine_rate's structural gates), the coarse loop converged,
        # and the command HELD over the span -- the fine value's reference is only exact
        # under a constant command, and gating beats averaging a moving one.
        _stage_rate_feed_fine()

        # S2 OBSERVER: the shadow surface for sky validation (present even when the feed
        # produced nothing this poll, so a dead feed shows n=0 rather than vanishing).
        if state_w is not None and args.rrate_state:
            try:
                _jro = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference)
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

        _stage_carrier_loop()

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
        if len(la_samples) >= args.code_bias_min_sats:
            raw_cb = statistics.median(la_samples)
            if abs(raw_cb) < args.code_bias_max * 1e-6:
                code_bias_ema = (raw_cb if code_bias_ema is None
                                 else code_bias_ema + args.code_bias_alpha * (raw_cb - code_bias_ema))
                # SAT-SCALED bar, same rationale as the carrier-bias alarm above (few-fit
                # chains' l-a median is ~1/sqrt(n) noisy; the fixed bar cried wolf on the
                # weak chains 2026-07-20). A real dongle-clock event is large + sustained.
                _labar = args.code_bias_alarm_ppm * max(1.0, (5.0 / max(len(la_samples), 1)) ** 0.5)
                if (code_bias_cal is not None
                        and abs(code_bias_ema - code_bias_cal) > _labar * 1e-6):
                    _log_rl("laalarm",
                            "CLOCK DRIFT ALARM: l-a %+.3f ppm vs calibration %+.3f "
                            "(|d| > %.2f ppm, %d fits) -- dongle clock news, INVESTIGATE"
                            % (code_bias_ema * 1e6, code_bias_cal * 1e6,
                               _labar, len(la_samples)), every_s=60.0)
                _log_rl("la-pool",
                        "code-rate clock offset (l-a) %+.3f ppm (raw %+.3f, %d fitted "
                        "sats, EMA a=%.2f)"
                        % (code_bias_ema * 1e6, raw_cb * 1e6, len(la_samples),
                           args.code_bias_alpha))
                # CONTRIBUTE (task #27 M3). PER BAND, not receiver-wide: cable and PFB group
                # delay are per carrier, so this number does not survive a retune. That is
                # exactly what --state-dongle asserts by hand today.
                rx.contribute_code_bias(chain_id, band_id, code_bias_ema,
                                        len(la_samples), t0)
                if args.code_bias_file:
                    try:
                        with open(args.code_bias_file, "w") as f:
                            f.write("%.4f\n" % (code_bias_ema * 1e6))
                    except Exception:
                        pass
        # S2 OBSERVER: the code-side twin. Outside the min-sats gate, same reason as the
        # carrier export. This one is the honest cross-chain comparison of the two: l-a has
        # NO sibling fusion at all, so its spread across a band's chains is a real measure
        # of estimator scatter, where the carrier's is partly manufactured by the fusion.
        if state_w is not None:
            try:
                _raw_la = statistics.median(la_samples) if la_samples else None
                state_w.observe(
                    "code",
                    ppm=(code_bias_ema * 1e6) if code_bias_ema is not None else None,
                    raw_ppm=(_raw_la * 1e6) if _raw_la is not None else None,
                    mad_ppm=(lambda m: m * 1e6 if m is not None else None)(
                        receiver_state.mad(la_samples, _raw_la)),
                    n=len(la_samples),
                    cal_ppm=(code_bias_cal * 1e6) if code_bias_cal is not None else None,
                    forced=args.code_bias_force is not None)
            except Exception:
                pass
        cb_to_seed = (args.code_bias_force * 1e-6 if args.code_bias_force is not None
                      else code_bias_ema)
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
                           (code_bias_ema or 0.0) * 1e6), every_s=60.0)
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
        _jrc = None
        if args.rrate_command:
            if not rr_full_ok:
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
                    _j = rx.joint_receiver(band_id, CODE_LEN, rereference=args.joint_rereference)
                    # No receiver-wide term solved yet -> nothing to command. The sigma
                    # gate below then handles per-sat convergence one row at a time.
                    _jrc = _j if _j.f_carrier_sigma() != float("inf") else None
                except Exception:
                    _jrc = None
        _rr_cmd_new = {}
        _rr_railed = 0
        _rr_released = 0

        # 4. push consensus seeds to every tracker (DLL trim applied at POST time only)
        payload = []
        bit_src, bit_known = {}, {}
        _stage_push_seeds()
        # The commands actually shipped this poll become the rrate feed's reference next
        # poll. REBUILT, not updated: a sat that stopped being commanded (row widened, seed
        # dropped) must fall back to referencing car_trim/0, or a stale command would
        # silently re-enter its measurements forever.
        rr_cmd_applied.clear()
        rr_cmd_applied.update(_rr_cmd_new)
        if _rr_cmd_new:
            _log_rl("jrr-cmd",
                    "JRR-CMD[%s]: %s Hz (rrate rows -> carrier_trim_hz, %d sat(s), "
                    "%d slew-railed, %d releasing)"
                    % (args.dr_constellation,
                       " ".join("%d:%+.2f" % kv for kv in sorted(_rr_cmd_new.items())),
                       len(_rr_cmd_new), _rr_railed, _rr_released), every_s=60.0)
        # WHERE THE PEEL'S SIGNS ACTUALLY CAME FROM this cycle. Without this the only symptom
        # of a source that silently supplies nothing is `nobits` in a health line 10 s later on
        # a different process, which is what made the 30 s-horizon bug hard to see.
        _log_rl("bitsrc", "nav_bits by source: %s; known bits: %s"
                % (dict(sorted(bit_src.items())), dict(sorted(bit_known.items()))))
        if navhealth is not None:
            _rep = navhealth.report()
            if _rep:
                _log_rl("navhealth", _rep)
        if navbits is not None:
            _log_rl("fleet", navbits.fleet.stats())
        if os.environ.get("GNSS_SEED_DEBUG"):
            for d in payload:
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
        for d in payload:
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
        _stage_fleet_trim_arming()

        if args.fast_trim_hz > 0.0:
            with fast_lock:
                fast_tmpl.clear()
                for _d in payload:
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
                _post("%s/set_seeds" % t_ep, payload)
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
        _stage_cl_sibling()

        # (S5 cross-band read + shadow accumulation + rescue hints moved EARLY, block 2a-xband
        # above -- it must run before the search-hint POST it feeds.)

        _log_rl("active", "active=%s (%d); seeded %d/%d trackers" % (sorted(seeds), len(seeds), ok, len(trackers)))
        if cl_report:
            _log_rl("clreport", "CL: " + "; ".join(cl_report))

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
