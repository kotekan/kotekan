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
    expand_token, resolve_prefix, parse_endpoints,
)
from gnss_broker.fits import (                # noqa: E402
    fit_cp_rate, fit_dop_rate, code_clock_bias_sample, rate_residuals,
    cp_rate_from_code_bias, dr_cp0, dr_seed_phys,
)
from gnss_broker.fleet import (               # noqa: E402
    fleet_dll, _coherent_sum, fleet_coherent, fleet_spectrum, fit_spectrum_delay,
)
from gnss_broker.publish import FleetPublisher            # noqa: E402
from gnss_broker import signals                            # noqa: E402
from gnss_broker import receiver                           # noqa: E402
from gnss_broker.state_filter import SatBiasFilter         # noqa: E402
from gnss_broker.sky import (                 # noqa: E402
    brdc_predict, visible_prns, _dh_dpos,
    _cnav_brdc_xcheck, _cnav2_brdc_xcheck, _inav_brdc_xcheck, _lnav_brdc_xcheck,
    _fnav_brdc_xcheck, _bcnav2_brdc_xcheck, _bcnav3_brdc_xcheck, _bcnav1_brdc_xcheck,
)

def main(argv=None, rx=None, publisher=None):
    # `--signal help` before the parser, because --trackers is required and listing the
    # known signals must not depend on being able to name a fleet first.
    if "help" in (argv if argv is not None else sys.argv[1:]):
        _a = list(argv if argv is not None else sys.argv[1:])
        if "--signal" in _a and _a.index("--signal") + 1 < len(_a) \
                and _a[_a.index("--signal") + 1] == "help":
            print("known signals (derived from lib/stages/gnss/gnssSignal.hpp):\n"
                  + signals.describe())
            return
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rest-url", default="http://localhost:12048",
                    help="default kotekan REST base for bare stage names")
    ap.add_argument("--detectors", "--searches", dest="detectors", default="",
                    help="detection endpoints: aggregator/search stage names, absolute "
                         "URLs, or {a..b} ranges (e.g. aggregate  or  search_{00..49}). "
                         "OPTIONAL: leave empty to run PURELY MODEL-PRIMARY, seeding every "
                         "visible satellite from the BRDC model with no acquisition search at "
                         "all (--almanac --dead-reckon). That is the CHORD configuration -- the "
                         "chain there has no search stage, because the station position is known "
                         "to millimetres and the model already lands inside the DLL capture "
                         "range, so a blind search would only re-derive what the ephemeris "
                         "already says.")
    ap.add_argument("--trackers", required=True,
                    help="tracker endpoints to seed (e.g. track_{00..49} or "
                         "http://nodeA:12048/track_{0..24},http://nodeB:12048/track_{0..24})")
    ap.add_argument("--combiner", default="combiner",
                    help="combiner endpoint for full-band |A| status (name or URL)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="control-loop period, s (keep below ~0.5 s drift time)")
    ap.add_argument("--acquire-snr", type=float, default=12.0,
                    help="min detection SNR to (re)seed a PRN")
    ap.add_argument("--drop-amplitude", type=float, default=0.3,
                    help="combined |A| below which a tracked PRN is coasting (fallback metric when "
                         "the combiner reports no deep significance / nav-bit wipe is off)")
    ap.add_argument("--lock-snr", type=float, default=3.0,
                    help="detection significance (sigma above noise) above which a sat counts as "
                         "locked -- the primary, noise-relative lock metric (vs the noise-biased |A|; "
                         "noise sits at ~1, a real lock at >>3)")
    # (--trim-precomp / --trim-precomp-carrier / --trim-precomp-coast DELETED, 07-19 audit
    #  A4: the carrier pre-shift was bench-rejected in both signs -- the BOOTSTRAP re-pull
    #  owns step recovery, and under --dop-continuous steps no longer occur; the coast cp
    #  currency translation became unconditional -- it is the same algebra as the hold-path
    #  TRANSLATE, and the flag's OFF default was shipping the known-bad legacy overwrite.)
    ap.add_argument("--coast-to-horizon", action="store_true",
                    help="never drop a visible sat for low signal -- coast on the pure model "
                         "(almanac doppler + pooled code rate, currency-corrected) until it "
                         "SETS. The beam-map mode: sidelobe/null transits keep despreading on "
                         "the predicted trajectory so the unbiased incoherent/coherent power "
                         "observables sample the WHOLE beam, not just where the sat is locked. "
                         "The model holds the code peak for ~1-2 min per the pooled l-a "
                         "uncertainty; the search re-anchors whenever the signal returns.")
    ap.add_argument("--noise-probes", type=int, default=0,
                    help="keep this many deepest-below-horizon PRNs seeded as NOISE PROBES: "
                         "the combiner then emits genuine signal-free records for the beam "
                         "map's pedestal calibration (an almanac-gated broker otherwise never "
                         "tracks one and the pedestal falls back to a signal percentile). "
                         "Probes fail every lock gate naturally; ~2 is plenty.")
    ap.add_argument("--hold-max-cp-err", type=float, default=0.4,
                    help="release a HELD seed when the tracked code phase (held cp + DLL "
                         "trim) disagrees with the search FIT by more than this (chips) on "
                         "3 consecutive fixes. This is the DLL's capture half-range: a "
                         "sharp-ACF (BOC) power discriminator has stable FALSE equilibria "
                         "~0.75 chips out (prompt -12 dB) that the hold would otherwise "
                         "servo forever while the search sees the true peak.")
    ap.add_argument("--escape-amp-veto", type=float, default=100.0,
                    help="VETO a hold-escape while the held track's incoherent amp_snr exceeds "
                         "this ABSOLUTE value. Physics: the false lobes the escape exists to "
                         "catch sit at prompt -12 dB, so a hold despreading at full amplitude "
                         "CANNOT be on one -- an accusing fit is wrong by construction "
                         "(2026-07-18: the phantom-sloped L2C fit dragged healthy 200-800-amp "
                         "holds off-peak every ~60 s; the strongest observed sats' false lobes "
                         "read <~50). amp_snr ONLY, never deep (deep stays coherent ON the "
                         "wrong lobe -- the C34 signature). 0 disables the veto.")
    ap.add_argument("--hold-snr", type=float, default=8.0,
                    help="incoherent amp_snr above which a tracked PRN's cp anchor is FROZEN "
                         "(hold-on-lock: DLL owns the sub-chip residual; fit re-anchors only on "
                         "loss). Uses amp_snr ONLY -- deep_snr's off-peak value is the nav-wipe "
                         "rectification floor (~7), which would freeze bad anchors instantly.")
    ap.add_argument("--hold-max-dop-hz", type=float, default=None,
                    help="release a held seed when the fresh Doppler departs the FROZEN one by "
                         "more than this: the stale replica carrier decoheres the SINGLE-RECORD "
                         "despread. Default = 0.1 cycle per record = 0.1*chip_rate/code_length "
                         "-- it MUST scale with the record period (100 Hz for GPS 1 ms records, "
                         "25 Hz for E1C 4 ms, 10 Hz for B1C 10 ms; the GPS-calibrated 100 Hz on "
                         "B1C let the despread walk into the sinc NULL: amp oscillated 778<->0 "
                         "on ~1 min cycles, the first tri-constellation night's BDS symptom).")
    ap.add_argument("--coast-budget", type=float, default=30.0,
                    help="seconds a VISIBLE sat is coasted (seed held + Doppler forecast forward) "
                         "through a signal dropout before dropping it -- so a radar sweep / brief "
                         "fade doesn't lose the lock. The code prediction stays good for ~tens of s "
                         "on a free-running TCXO; raise it with a disciplined clock (OCXO). |A| "
                         "recovering resets the coast; setting below the horizon drops immediately.")
    ap.add_argument("--lat", type=float, help="receiver latitude (enables gating)")
    ap.add_argument("--lon", type=float, help="receiver longitude")
    ap.add_argument("--alt", type=float, default=0.0, help="receiver altitude, m")
    ap.add_argument("--mask-deg", type=float, default=5.0, help="elevation mask, deg")
    ap.add_argument("--almanac", action="store_true",
                    help="orbit-predict each PRN's Doppler (needs --lat/--lon + skyfield): "
                         "seed trackers with the precise predicted Doppler (geometry + a common "
                         "clock-freq bias solved from the measured sats) instead of the coarse "
                         "search grid, and gate to visible sats. Code phase still from the search.")
    ap.add_argument("--tle", default=None, help="GPS TLE file/URL (default: Celestrak gps-ops)")
    ap.add_argument("--almanac-source", default="brdc", choices=("brdc", "tle"),
                    help="orbit source for the almanac assist. brdc (default) = the IGS "
                         "broadcast-ephemeris file (gnss_ephemeris): PRN-INDEXED (immune to the "
                         "celestrak name->PRN label rot that mislabels BeiDou IGSOs -- C31/C38/"
                         "C39 ranges were 16-20k km off on 2026-07-17, corrupting nh hints by "
                         "5-7 overlay chips and hiding C38's true 84-deg pass), ~m orbits vs "
                         "TLE's ~km, includes the sat clock, and needs no skyfield. tle = the "
                         "legacy celestrak path (also the automatic fallback when BRDC is "
                         "unreachable at startup).")
    ap.add_argument("--constellation", default=None, choices=("G", "E", "C", "R"),
                    help="constellation letter for the BRDC almanac (which file covers ALL "
                         "systems, unlike the per-group TLE URLs). Default: --dr-constellation. "
                         "C keeps only PRN >= 19 (BDS-3): the BDS-2 birds transmit B1I at "
                         "1561 MHz, outside our band -- same capability cut the BEIDOU-3 "
                         "TLE name filter encodes.")
    ap.add_argument("--carrier-hz", type=float, default=1575.42e6, help="carrier for Doppler pred")
    ap.add_argument("--doppler-sign", type=float, default=1.0,
                    help="multiply predicted Doppler (set -1 if the convention is inverted)")
    ap.add_argument("--narrow-search", action="store_true",
                    help="push per-PRN predicted Doppler to the detectors' /set_doppler_hints so the "
                         "search scans only doppler +- margin (almanac-narrowed acquisition) instead "
                         "of its blind grid; needs --almanac. Far cheaper + more sensitive.")
    ap.add_argument("--search-margin-hz", type=float, default=500.0,
                    help="narrow search half-window once the clock-freq bias is solved (Hz)")
    ap.add_argument("--search-margin-wide-hz", type=float, default=3000.0,
                    help="wider search half-window BEFORE the bias is solved (covers the unknown "
                         "TCXO offset; shrinks to --search-margin-hz once a few sats pin the clock)")
    ap.add_argument("--clock-bias-file", default=None,
                    help="persist the solved carrier clock-frequency bias (Hz, plain text) "
                         "across runs, and warm-start from it: the bias is then SOLVED from "
                         "cycle 1 (narrow margins, seeding enabled immediately). Audit rec D: "
                         "on the GPSDO this is a per-chain constant, not an estimate.")
    ap.add_argument("--state-read-dir", default=None,
                    help="directory of sibling receiver-state JSON to READ (--dr-clock-adopt). "
                         "Defaults to --state-file's directory. Separate because an adopting "
                         "chain publishes nothing: it contributes no estimate, so requiring it "
                         "to name a write path just to read one would be backwards.")
    ap.add_argument("--dr-clock-adopt", action="store_true",
                    help="ADOPT the receiver clock from a BAND SIBLING instead of holding the "
                         "primed --dr-clock-chips constant forever. Reads the sibling state "
                         "files receiver_state.StateWriter already publishes (same --state-dir "
                         "and --state-dongle scope as --clock-bias-siblings) and refreshes "
                         "dr_state['clk'] and its drift from the freshest one, every cycle.\n"
                         "\n"
                         "WHY: a chain with --detectors EMPTY cannot solve its own clock. The "
                         "bootstrap takes a median of measured code-phase residuals from "
                         "/get_detections, so with no detector `offs` is always empty, the EMA "
                         "never runs, and the primed constant stands for the WHOLE RUN -- which "
                         "the --dr-clock-chips help already says in as many words. That is "
                         "every multi-constellation chain (E5a/B2a have no blind acquisition at "
                         "all), so today the number is read out of the GPS broker's log by a "
                         "human and pasted in, and it is wrong after any F-engine restart.\n"
                         "\n"
                         "THIS IS AN ADOPTION, NOT A FUSION, and that is why it is safe to turn "
                         "on now while receiver_state's covariance-weighted fuser is still "
                         "deferred ('score, don't steer'). The adopting chain has NO estimate of "
                         "its own, so there is nothing to weight and no manufactured agreement "
                         "-- the failure mode that module's docstring warns about (siblings "
                         "reading each other, so their spread measures the fusion rather than "
                         "the estimator) cannot arise. A chain WITH detections ignores this "
                         "flag; it solves its own.\n"
                         "\n"
                         "SCOPE IS THE LO, exactly as receiver_state defines it: E5a and GPS L5 "
                         "share one carrier, one cable and one F-engine, so the clock transfers "
                         "EXACTLY. It does NOT transfer across a retune -- E5b at 1207 MHz has a "
                         "different PFB group delay -- which is why the dongle key gates it and "
                         "why this must never be pointed across bands.")
    ap.add_argument("--dr-long-code", action=argparse.BooleanOptionalAction, default=True,
                    help="dead-reckon the code phase at the FULL secondary-overlaid code length "
                         "(--long-code-segments * code length) rather than one primary period. "
                         "Required for any chain without a blind search: E5a/B2a carry per-PRN "
                         "secondaries, so nothing measures the segment and the model must supply "
                         "it. With this off (the historical behaviour) every dead-reckoned seed "
                         "lands in segment 0 -- correct 1 time in LC_SEG, which for E5a's CS100 "
                         "is 1 in 100. GPS masked the bug because its blind search re-seeds with "
                         "a measured nh. Needs absolute time to half a primary period (0.5 ms); "
                         "the GPS-disciplined anchor is microsecond-class.")
    ap.add_argument("--dr-clock-adopt-max-slew", type=float, default=2.0,
                    help="with --dr-clock-adopt: refuse a sibling whose clock is MOVING faster "
                         "than this (chips/s), measured across two consecutive reads one cycle "
                         "apart. Deliberately NOT the sibling's own quality fields: "
                         "integ_mad_chips is a MAD over a mixed satellite population and read "
                         "3.3-3.9 chips while the clock itself was stable to 0.2 (2026-08-08), "
                         "and `untrusted` counts a persistent demoted set not comparable to `n`. "
                         "Watching the number is the honest test. Default 2.0 chips/s: real "
                         "drift is ~0.02, a non-converged estimate moves by thousands.")
    ap.add_argument("--dr-clock-adopt-max-age-s", type=float, default=60.0,
                    help="with --dr-clock-adopt: refuse a sibling record older than this. "
                         "Staleness is a REFUSAL, not a fallback (receiver_state.read_state) -- "
                         "an old clock is a different epoch's number, and after an F-engine "
                         "restart it is exactly the wrong one. On refusal the chain keeps what "
                         "it has and says so.")
    ap.add_argument("--clock-bias-siblings", nargs="*", default=None,
                    help="BAND-SHARED bias (2026-07-22): the other chains' --clock-bias-file "
                         "paths on the SAME band. All chains of a band despread the SAME "
                         "carrier frequency through ONE LO, so the clock-freq bias is one "
                         "physical number measured independently per chain; a chain with 2 "
                         "tracked sats estimates it at +-30 Hz swing (measured: the L5-GPS "
                         "bias wandered -16..+44 Hz in 70 s while L1-GPS sat at -152 +-1 Hz "
                         "with 5 sats), and that swing feeds EVERY seed's predicted Doppler "
                         "-> fence re-pins -> the 30-45 s NCO kick cycle that decoheres the "
                         "chain. Fusing the siblings' persisted estimates (sat-count "
                         "weighted, <60 s fresh) gives each chain the band's full sat count "
                         "(~8-10) -- an identifiability fix, not a tuning knob. Chains in "
                         "drift-alarm stop persisting, so poisoned estimates go quiet "
                         "automatically.")
    ap.add_argument("--state-file", default=None,
                    help="S2 / Mechanism B OBSERVER: publish this chain's receiver-state "
                         "estimates as JSON (atomically replaced, ~1 Hz) for cross-chain "
                         "comparison. WRITE-ONLY -- exporting changes no estimate and no "
                         "seed. It exists because eight brokers estimate the same four "
                         "physical quantities independently, four of them measure the SAME "
                         "per-dongle clock error by different routes (search-Doppler median, "
                         "carrier-loop trim, l-a slope, DR drift), nothing has ever compared "
                         "carrier-side against code-side, and NOTHING anywhere carries a "
                         "variance -- so the covariance-weighted fuser this is a prerequisite "
                         "for cannot yet be written. Exports each chain's PRE-fusion value "
                         "and its scatter, because the persisted .hz files already read each "
                         "other and their agreement is therefore partly manufactured.")
    ap.add_argument("--decoded-eph-fallback", type=int, default=1,
                    help="when the network BRDC fetch fails, keep the dead-reckon predict alive "
                         "off THIS broker's own on-node decoded ephemeris (decoded_eph.py, which "
                         "reuses each decoder's BRDC-validated propagator). Geometry+Doppler are "
                         "exact; precise sat-clock is Galileo-complete, best-effort elsewhere. "
                         "1=on (default), 0=off (pre-fallback behaviour: predict goes dark).")
    ap.add_argument("--decoded-eph-fallback-force", type=int, default=0,
                    help="ALWAYS predict from decoded eph, even when BRDC is available -- the live "
                         "A/B validation harness (compare against the BRDC predict in the log). "
                         "Exercises the BeiDou BDT frame + CNAV clock the offline test can't. "
                         "Do NOT leave on in production.")
    ap.add_argument("--decode-health-file", default=None,
                    help="publish this broker's NAV-DECODE health as JSON (atomically "
                         "replaced) for the viewer + the eventual decoded-eph BRDC fallback: "
                         "per (signal, PRN) whether SYNCED, whether a full ephemeris was "
                         "EXTRACTED, the BRDC-agreement residual dpos_m, and decode freshness. "
                         "WRITE-ONLY, sibling of --state-file; the viewer reads the directory "
                         "and merges. Off when unset.")
    ap.add_argument("--state-dongle", default=None,
                    help="fusion scope key for --state-file: what physically shares an LO "
                         "(one airspy per band, so this is the band tag, identical for every "
                         "chain on it). Do NOT fuse across dongles -- the per-band offsets "
                         "(-151/-15/+31 Hz) are frac-N synthesis constants, not a common "
                         "reference error, and averaging them is meaningless.")
    ap.add_argument("--state-fuse", type=int, default=1,
                    help="with --state-file: also compute and PUBLISH this dongle's fused "
                         "fractional LO estimate (S2c). STILL WRITE-ONLY -- no seed, gate "
                         "or estimator consumes it; the broker logs what the fused prior "
                         "WOULD have said beside what the chain actually uses, so the flip "
                         "is decided on a soak of evidence rather than on argument. Fuses "
                         "in ppm because carrier (Hz at this band) and code (l-a in ppm) "
                         "measure the same FRACTIONAL error, and only from siblings' RAW "
                         "values -- fusing their smoothed ones would feed the estimate back "
                         "on itself and its covariance would be fiction.")
    ap.add_argument("--state-consume", type=int, default=0,
                    help="S2d, RESCUE-ONLY (revised 2026-07-29): consume the dongle's fused "
                         "LO estimate EXACTLY when this chain has no estimate of its own "
                         "(cold start, below min-sats, warm-start file lost). The original "
                         "always-on scope was tried and REVERTED the same day -- car_trim "
                         "rose 30-36% at matched node age, because the LO is a CONSTANT and "
                         "the chain's own EMA (minutes of time-averaging) beats one cycle "
                         "of cross-chain averaging; rescored against the EMA, fusion lost "
                         "7 of 8 chains. In the rescue case there is no EMA to lose to, and "
                         "the fused state's unique value is the cross-FAMILY rescue "
                         "(code->carrier) that --clock-bias-siblings structurally cannot "
                         "provide. With the chain solved, this flag is a PROVEN no-op. The "
                         "'untested rescue path' worry is answered by scoring it always "
                         "(the SHADOW log line) and exercising it deliberately "
                         "(diag/receiver_state_rescue_test.py + the isolated-broker "
                         "method), not by running it always.")
    ap.add_argument("--state-fuse-floor-ppm", type=float, default=0.001,
                    help="covariance FLOOR: no source may claim a standard error below "
                         "this. ON by default, and the default was chosen from a live "
                         "capture, not from theory. The 15-min cross-chain scan said a "
                         "floor was unnecessary (pairwise |z| = 0.2 -- pure noise), but "
                         "that scan compared MEDIANS over 150 samples and the fuser runs "
                         "PER CYCLE: on the very first live fusion the L5 GPS chain claimed "
                         "se = 0.00018 ppm (0.21 Hz, 100x tighter than any sibling) while "
                         "sitting 22.7 sigma off, because its handful of satellites "
                         "happened to agree that cycle. Its inverse-variance weight alone "
                         "would have dragged the dongle's answer from +32.3 Hz to +14.7 Hz "
                         "with all three chains actually at +31.5..+33.8. MAD over 2-6 "
                         "satellites is a poor scatter estimate and can come out near zero "
                         "by chance; the floor is the statement that no chain can beat "
                         "~1.5 Hz from a few sats no matter what its MAD says.")
    ap.add_argument("--state-fuse-reject-sigma", type=float, default=5.0,
                    help="drop a source this far from the ROBUST (median) centre, then "
                         "refit. Judged against the median, never the inverse-variance "
                         "mean: with one bad source among three the mean is dragged far "
                         "enough that the GOOD sources also exceed the bar, the survivor "
                         "list comes back empty and a naive implementation then rejects "
                         "nothing while publishing the contaminated estimate. 0 disables.")
    ap.add_argument("--state-flush-s", type=float, default=1.0,
                    help="--state-file publish cadence (s). Current state only; history is "
                         "the scorer's job (8 brokers appending at 1 Hz is ~200 MB/day in a "
                         "cache directory that has to survive reboots).")
    ap.add_argument("--clock-bias-alarm-hz", type=float, default=10.0,
                    help="CLOCK DRIFT ALARM bar: loud log if the live bias EMA departs the "
                         "warm-start calibration by more than this (GPSDO unlock / thermal "
                         "event -- hardware news, not something to silently absorb)")
    ap.add_argument("--code-bias-alarm-ppm", type=float, default=0.05,
                    help="same alarm for the code-rate clock (l-a) vs its warm-start value")
    ap.add_argument("--bias-stale-s", type=float, default=300.0,
                    help="STALE-BIAS RESCUE: if the solved bias EMA has gone this long without "
                         "a multi-sat measurement, widen the search margins and RE-SOLVE from "
                         "the next detections (snap to the fresh median, recalibrate, persist). "
                         "Closes the 2026-07-20 lockout: a GPSDO unlock walked the EMA -2 ppm, "
                         "every lock died, and the EMA latched mid-walk -- hints then sat kHz "
                         "off truth at narrow margins with nothing left to update them. The "
                         "held value still centers the (wide) hints and seeds, so a healthy "
                         "chain that merely has a sparse sky loses nothing. 0 disables.")
    ap.add_argument("--fit-maturity-span-s", type=float, default=30.0,
                    help="cp-fit HISTORY SPAN required before the fit is trusted (escape "
                         "referee + hold admission). 30 s makes the code-Doppler quadratic "
                         "observable on every chain. BENCH NOTE (2026-07-19): 100-s replay "
                         "legs cannot afford 30 s of maturity + overlay consensus -- short "
                         "legs go bimodal on B1C deep (some sats never sync, deep ~15 vs "
                         "220). Benches pass ~10; the A/B verdict discipline requires it.")
    ap.add_argument("--bias-alpha", type=float, default=0.05,
                    help="EMA weight for the clock-freq bias (smaller = steadier seed Doppler; "
                         "~0.05 => few-second time constant, dithers out the 500 Hz grid)")
    ap.add_argument("--bias-min-sats", type=int, default=2,
                    help="detected sats needed before a cycle's median residual may update the "
                         "clock-freq bias (and hence NARROW the search). A single sat's residual "
                         "is unfalsifiable: one bad prediction (wrong TLE mapping, non-transmitting "
                         "sat cross-corr) gets swallowed as 'clock bias' and shifts every other "
                         "hint out of the narrow window -- a self-locking deadlock where no second "
                         "sat can ever acquire to correct it (2026-07-12: BDS-2 C14 froze the whole "
                         "B1C constellation at -1550 Hz).")
    ap.add_argument("--tle-name-filter", type=str, default=None,
                    help="regex on the TLE NAME; almanac keeps only matching sats. Encodes "
                         "signal capability the TLE group can't (e.g. 'BEIDOU-3' for B1C: "
                         "BDS-2 birds don't transmit it -- their predictions poison the clock "
                         "bias and their PRNs only manufacture cross-correlation locks).")
    ap.add_argument("--code-length", type=float, default=1023.0,
                    help="spreading-code length (chips) for cp0 unwrap/fit (L1 C/A = 1023)")
    ap.add_argument("--hops-per-sec", type=float, default=125000.0,
                    help="F-engine hops/s (Fs/fft_len) -- cp slope chips/s log + the code clock-bias estimate")
    ap.add_argument("--det-alias-fold", type=int, default=1,
                    help="ALIAS-BIN CENSUS (diagnostic only): log detections whose Doppler "
                         "sits a record-alias quantum 1/(2*t_rec) off the model+bias "
                         "reference (25 Hz L2C, 50 Hz B1C). The v1 of this flag FOLDED the "
                         "dop before the cp currency conversion -- WRONG: the search "
                         "back-projects cp0 with the same reported dop, so the round trip "
                         "is exact for any bin, and folding broke the cancellation by "
                         "K*t_abs*k*q (12-57 chip candidates on held L2C sats, caught by "
                         "the track-vs-model monitor within the hour). 0 silences the log.")
    ap.add_argument("--watchdog-weak-sig", type=float, default=30.0,
                    help="WEAK-TRACK RESEED bar: a sat the search sees at full det snr "
                         "(>= --watchdog-det-snr) whose TRACK significance stays under "
                         "this for a whole --watchdog-s window is a coherent-but-weak "
                         "zombie (correlating ~20 dB off-peak; nonzero coherence evades "
                         "the zero-coherence watchdog, correct cp evades the referee, "
                         "folded resid evades refade -- the C21/C42 class). Healthy weak "
                         "chains are exempt via the det bar (their dets are weak too). "
                         "0 disables.")
    ap.add_argument("--chip-rate-hz", type=float, default=1.023e6,
                    help="spreading chip rate (L1 C/A 1.023e6) -- for the code-rate clock-bias estimate")
    ap.add_argument("--code-doppler-sign", type=float, default=1.0,
                    help="must match the search stage's code_doppler_sign: the sign of the "
                         "Doppler-dependent cp0 back-reference the search applies. Used to "
                         "re-express cp0 history in the seed's Doppler currency before the "
                         "slope fit (cp_to_seed_currency).")
    ap.add_argument("--code-bias-alpha", type=float, default=0.05,
                    help="EMA weight for the receiver code-rate clock offset (l-a); slow -> tracks TCXO/OCXO drift")
    ap.add_argument("--code-bias-max", type=float, default=3.0,
                    help="reject per-sat l-a samples beyond +-this (ppm) before the median -- a "
                         "noisy/unwrap-blown slope fit is an outlier a few-sat median can't reject; "
                         "the seeded code rate must stay ~0.1 ppm stable or the deep decoheres")
    ap.add_argument("--code-bias-min-sats", type=int, default=2,
                    help="fitted sats needed before the pooled code-rate clock offset is trusted + seeded to weak sats")
    ap.add_argument("--code-bias-init", type=float, default=None,
                    help="warm-start the receiver code-rate clock offset (l-a) in PPM, e.g. from a prior "
                         "strong-signal (L1 C/A) run -- so a weak band (L1C) seeds on-peak from cycle 1 "
                         "instead of self-calibrating. Live samples still refine it if any sats fit.")
    ap.add_argument("--code-bias-force", type=float, default=None,
                    help="DIAGNOSTIC: pin the (l-a) code-rate clock offset to this PPM -- the "
                         "live fit/EMA still runs and logs (so the fit stays observable) but the "
                         "SEEDED rate uses this value only. 2026-07-18: built to test the L2C "
                         "phantom-l-a hypothesis (fit says +0.022 ppm, air truth says 0.000).")
    ap.add_argument("--code-bias-file", type=str, default=None,
                    help="persist the converged (l-a) ppm here: read at startup (unless --code-bias-init "
                         "is set) and rewritten each update, so the offset carries across runs/bands")
    ap.add_argument("--fit-gap-s", type=float, default=16.0,
                    help="reset the cp-fit history across a detection gap longer than this "
                         "(seconds of capture time; converted via --hops-per-sec)")
    ap.add_argument("--carrier-gain", type=float, default=0.0,
                    help="SHARED carrier loop gain (0 = off): integrate the combiner's full-band "
                         "carrier_hz_resid into a per-PRN carrier_trim_hz commanded to every "
                         "subband tracker's NCO -- one loop at full-band SNR instead of N "
                         "noise-driven per-channel FLLs. Trackers need carrier_shared: true.")
    ap.add_argument("--carrier-source", choices=("rate", "resid"), default="rate",
                    help="what the shared carrier loop integrates. 'rate' (default, 2026-08-04) "
                         "= the combiner's phase-rate search (deep_rate_hz), measured at "
                         "peak/median 17.9-22.0 on signal vs 2.8-6.1 on noise and pinned to "
                         "~0.2 Hz by split-half on strong sats. 'resid' = the legacy "
                         "carrier_hz_resid, which is SIGNAL-FREE (0.519 Hz on signal, 0.492 on "
                         "noise) -- kept only to reproduce the old behaviour, never for science.")
    ap.add_argument("--carrier-rate-min-q", type=float, default=10.0,
                    help="hard gate on deep_rate_q before a rate residual is believed. The 8.18.5 "
                         "gap: 17.9-22.0 on signal, 2.8-6.1 on noise. A weak sat does not merely "
                         "scatter, it lands on the WRONG spectral bin (measured: amp_snr 9.5 was "
                         "41.7 Hz out by split-half, where amp_snr 83.7 was 0.000).")
    ap.add_argument("--carrier-rate-clip-hz", type=float, default=25.0,
                    help="clip rate residuals this far from the fleet MEDIAN before the weighted "
                         "consensus (<=0 disables). Median first, because a wrong-bin outlier is "
                         "arbitrarily far and would drag any mean.")
    ap.add_argument("--carrier-rate-inherit", action="store_true", default=False,
                    help="a PRN failing the q gate takes the fleet's amp_snr-weighted consensus "
                         "instead of no correction. OFF since 2026-08-04: this was built on "
                         "'the dominant term is common-mode', which the sky refutes. Measured "
                         "across 131 emits with >=3 PRNs each, the spread of deep_rate_hz "
                         "BETWEEN PRNs within a single emit has median 60.8 Hz -- near the full "
                         "+-47.7 Hz range of the search. That is expected in hindsight: the "
                         "residual is measured against each tracker's OWN f_ref, which re-pins "
                         "per PRN on its own schedule, so there is no shared zero to average "
                         "towards. (The claim that IS true, and a different one, is that "
                         "different NODES agree on a given PRN.) Inheriting hands a satellite a "
                         "number belonging to someone else's reference; free-running is better.")
    ap.add_argument("--no-carrier-rate-inherit", dest="carrier_rate_inherit",
                    action="store_false", help="disable the consensus fallback (per-PRN only)")
    ap.add_argument("--carrier-trim-const", type=float, default=None,
                    help="DIAGNOSTIC: command this fixed carrier_trim_hz to every seeded PRN, "
                         "independent of --carrier-gain. Use with --carrier-gain 0 to measure the "
                         "open-loop step response of deep_rate_hz: sweep 0 / +X / -X and the "
                         "measured rate should move by exactly -X. Same sign => the loop is "
                         "inverted; no movement => the trim never reaches the despread.")
    ap.add_argument("--carrier-rate-max-gap", type=float, default=2.0,
                    help="max window gap, in RECORDS, across which a rate measurement is still "
                         "believed. Measured: contiguous emits step -0.24..-0.72 Hz, while any "
                         "gap steps by tens of Hz because the tracker re-anchored. A gap "
                         "re-baselines rather than integrating a reference change.")
    ap.add_argument("--carrier-rate-max-step", type=float, default=3.0,
                    help="reject a rate residual that jumped more than this since the last "
                         "believed sample, and re-baseline. Catches the tracker's f_ref FENCE "
                         "re-anchor, which fires mid-tracking with no window gap to mark it: the "
                         "phase-continuity fold keeps the phase smooth but the FREQUENCY steps. "
                         "Measured separation -- contiguous emits step 0.24-0.72 Hz, the smallest "
                         "re-anchor jump was 7.6 Hz.")
    ap.add_argument("--carrier-max-hz", type=float, default=40.0,
                    help="clamp on the shared carrier trim (Hz)")
    ap.add_argument("--carrier-leak", type=float, default=0.05,
                    help="shared carrier integrator leak (same role as --dll-leak)")
    ap.add_argument("--carrier-min-sig", type=float, default=0.0,
                    help="HOLD the trim (skip the update) when the combiner's lock significance "
                         "is below this (0 = old behavior). The probe exemption above, "
                         "generalized: a FADED real satellite is the same pathology -- its "
                         "residual is noise, and integrating it at full gain random-walks the "
                         "trim, which drags the model phase off, which DEEPENS the fade. "
                         "Measured 2026-07-17 on the 1176 MHz chains: a ~4 s noise-driven "
                         "limit cycle (dip -> trim walk 1-9 Hz -> decoherence -> dip), median "
                         "certified stretch 4 s, every dip an ADR phase break of ~8 cycles -- "
                         "the gf-TEC floor. A held trim coasts on the almanac Doppler-rate "
                         "feed-forward, which carries the true dynamics through the fade. "
                         "Updates additionally require a CERTIFIED coherent window "
                         "(coherence_s > 0): a residual measured on a decohered window is "
                         "garbage at ANY amplitude (the sig gate alone let re-lock transition "
                         "rows kick converged trims to the +-100 Hz rails -- E36 at 48 dB-Hz).")
    ap.add_argument("--carrier-max-step", type=float, default=0.0,
                    help="slew clamp on the trim, Hz per update (0 = unclamped). A healthy "
                         "converged loop corrects 0.02-0.2 Hz per update and the clock it "
                         "tracks is GPSDO-smooth, so any large requested step is a bad "
                         "measurement by construction; clamping bounds the damage a single "
                         "garbage residual can do to less than a deep window can absorb. "
                         "Convergence from a fleet-seeded start needs only a few Hz total.")
    ap.add_argument("--carrier-step-accept", type=int, default=0,
                    help="EXPLAIN-APPLY-VERIFY hypothesis stage (0 = off): M = this many "
                         "consecutive fresh GATED residuals that agree (spread < max(2 Hz, "
                         "innov)). For a PRESENT-but-gated sat, when the agreed median is "
                         "also large enough to EXPLAIN the decoherence (>= ~1/(2*T_emit) = "
                         "0.5 Hz), the observables close on one story -- 'the NCO is off by "
                         "med' -- and the FULL correction is applied ONCE, entering a "
                         "VERIFY window: coherence returns / residual collapses within 3 "
                         "emits, or the hypothesis is REVERTED, the sat escalated to a "
                         "BOOTSTRAP re-acquire, and hypotheses locked out 60 s. The "
                         "coherent-state innovation gate is untouched (it is a physics "
                         "bound: a cohering sat cannot carry a multi-Hz residual). The "
                         "closed verify loop is what the two retracted open-loop escapes "
                         "lacked: a wrong correction costs one bounded, reverted step. "
                         "Type specimen: C19 2026-07-22, parked at +3.03 Hz / full amp / "
                         "dark for minutes while every gate held.")
    ap.add_argument("--carrier-innov-hz", type=float, default=0.0,
                    help="TRACK-mode innovation gate (0 = off): REJECT any residual larger "
                         "than this outright. After feed-forward, a converged sat's true "
                         "residual is sub-Hz (the trim tracks a slowly-drifting almanac-"
                         "Doppler error); the resid estimator nevertheless emits tens-of-Hz "
                         "values that pass certification (measured 07-17: 'certified' +40 Hz "
                         "-- impossible for a genuinely coherent window). A slew clamp only "
                         "slows the poisoning (E36 at 49 dB-Hz walked to 18 Hz off the fleet "
                         "at 1 Hz/update and collapsed 20 dB); rejection stops it. Real step "
                         "changes re-enter via re-seed -> BOOTSTRAP.")
    # (ALIAS ESCAPE v1/v2 DELETED, 07-19 audit A4. v1 killed the fleet in 15 min
    #  (8208dba6/069e8770); v2 shipped gated-off and never armed. Its two jobs are owned
    #  by surviving mechanisms: a stale/aliased f_ref offset is snapped by the TIGHT
    #  tracker fence (fll_reacq_hz ~15 Hz, free under --dop-continuous), and a walked/
    #  aliased TRIM latch is the watchdog's lifecycle rescue below.)
    ap.add_argument("--carrier-bleed-shadow", type=int, default=1,
                    help="f_ref TRIM-BLEED SHADOW (1 = on, log-only): report where a converged, "
                         "coherent, STANDING carrier trim WOULD be re-pinned into f_ref (which "
                         "would drop the sinc(ctrim*T_rec) despread loss, ~0.13-0.26 dB on L2C). "
                         "Takes NO action -- it validates the TRIGGER on live data before any "
                         "tracker change (the fleet-collapse lesson: a carrier correction that "
                         "fires on the wrong sat is catastrophic). Gated on coherent + converged "
                         "so it can never flag the BOOTSTRAP/incoherent noise-walk that killed "
                         "alias-escape v1/v2.")
    ap.add_argument("--carrier-bleed-hz", type=float, default=2.0,
                    help="Shadow: minimum standing |trim| (Hz) to consider a bleed candidate. "
                         "Below this the sinc loss is negligible and not worth a re-pin.")
    ap.add_argument("--carrier-bleed-stable-emits", type=int, default=8,
                    help="Shadow: consecutive coherent emits the trim must hold steady over to "
                         "count as CONVERGED (a moving trim is still settling -- not a candidate). "
                         "Longer window makes the drift SLOPE (below) measurable above noise.")
    ap.add_argument("--carrier-bleed-stable-hz", type=float, default=0.4,
                    help="Shadow: max peak-to-peak trim spread (Hz) over the stability window to "
                         "call it converged.")
    ap.add_argument("--carrier-bleed-max-slope", type=float, default=0.02,
                    help="Shadow: max |least-squares trim slope| (Hz/s) over the window. A still-"
                         "settling trim drifts (low spread but a monotonic climb); bleeding it folds "
                         "a mid-convergence value and leaves a residual (the 2026-08-03 REFUTED "
                         "class). Requiring a FLAT slope, not just low spread, rejects those.")
    ap.add_argument("--carrier-bleed", type=int, default=0,
                    help="ARM the trim-bleed (0 = shadow-only). On a verified candidate: zero "
                         "car_trim and flag the tracker to re-adopt the seed (f_ref = dop, phase-"
                         "continuous reanchored==2) -- folding the frozen sub-fence pin offset into "
                         "f_ref so the despread runs on-true. EXPLAIN-APPLY-VERIFY: heal (stay "
                         "coherent) or it is logged REFUTED and the loop re-grows the trim from 0. "
                         "Default OFF -- validate on the replay bench (GNSS_TRIM_FORCE) first.")
    ap.add_argument("--carrier-bleed-verify-emits", type=int, default=3,
                    help="Emits to watch after a bleed before judging the re-pin by its residual.")
    ap.add_argument("--carrier-bleed-ok-hz", type=float, default=1.5,
                    help="Post-bleed |residual| (Hz) at the end of the verify window to count "
                         "VERIFIED. Judged on the SETTLED residual, not coh_ok -- a re-pin resets "
                         "the deep coherent window for ~1 emit (coherence_s blips) even on a good "
                         "bleed, so coh_ok would false-refute; and a mild remainder still reduced "
                         "the standing trim (net win).")
    ap.add_argument("--carrier-bleed-lockout-s", type=float, default=120.0,
                    help="Minimum seconds between bleeds of the same PRN (anti-churn).")
    ap.add_argument("--watchdog-s", type=float, default=0.0,
                    help="TRACK WATCHDOG (0 = off): a sat with a fresh detection at "
                         ">= --watchdog-det-snr that has ZERO coherent emits for this many "
                         "seconds (and has been seeded at least that long) is dropped from "
                         "seeds entirely -- the full re-seed lifecycle (fresh dop blend, "
                         "fleet trim prior, tracker state reset via the active[] gap) is "
                         "the only rescue that fixes every cause (aliased NCO, walked "
                         "trim, poisoned anchor) without guessing which one it is. The "
                         "2026-07-18 targeted-correction attempts (trim-step v1/v2) both "
                         "guessed and both lost; the lifecycle rescue never did.")
    ap.add_argument("--watchdog-det-snr", type=float, default=50.0,
                    help="watchdog presence bar: only judge sats the search currently "
                         "sees at this significance -- a sat this strong that cannot "
                         "cohere is broken by definition; weak sats legitimately take "
                         "minutes and must never be churned by the watchdog.")
    ap.add_argument("--carrier-det-gate-s", type=float, default=0.0,
                    help="BOOTSTRAP walk gate (0 = off): in BOOTSTRAP mode, integrate a "
                         "residual only if a fresh detection exists within this many "
                         "seconds. A never-detected (almanac-only) or long-undetected seed "
                         "has no signal for the estimator: its 'residual' is noise, and "
                         "integrating it random-walks the trim to the clamp (C40 walked to "
                         "-42 Hz over the 07-18 evening; the E36 innovation gate protects "
                         "only TRACK mode). Held trims coast on the fleet prior + Doppler-"
                         "rate feed-forward, which is the better model anyway.")
    ap.add_argument("--refade-flicker-s", type=float, default=30.0,
                    help="suppress the --carrier-refade demotion when the residual is SUB-"
                         "innovation AND the sat cohered within this many seconds: that is "
                         "certification-bar sig flicker (settled-era E1/B1C: ~700 no-op "
                         "re-pulls/3 h at |resid| ~1.7 Hz), not a stepped NCO. A STANDING "
                         "decoherence (the L2C C20 absorbing state, dark for minutes at a "
                         "sub-gate resid) still demotes after the window. Needs the track "
                         "watchdog on for coherence timestamps; 0 disables the guard.")
    ap.add_argument("--carrier-refade", type=int, default=10,
                    help="TRACK-mode DEMOTION: after this many consecutive gated residuals "
                         "(fade-hold or innovation-reject) while the sat is still PRESENT "
                         "(amp_snr >= --hold-snr), drop it back to BOOTSTRAP so the loop "
                         "re-acquires at full gain (0 = never). Without this the two TRACK "
                         "gates form an ABSORBING state: a seed-doppler step (un-precomped "
                         "hold release / escape re-anchor) leaves a residual above the "
                         "innovation gate, decoherence turns off coh_ok, and the sat parks "
                         "carrier-dead with a perfectly measurable residual forever -- "
                         "measured 2026-07-18 on B1C: C20 latched at -6.2 Hz for 40 min at "
                         "full amp while deep sat on the floor; the strongest (fastest-"
                         "slewing) sats latch first, the weak ones never certify into TRACK "
                         "and self-heal, inverting the fleet. The presence bar keeps a "
                         "genuinely faded sat coasting on the feed-forward (the pathology "
                         "--carrier-min-sig exists for); the innov gate's designed escape "
                         "('re-seed -> BOOTSTRAP') never fires for a held strong sat.")
    ap.add_argument("--carrier-fleet-seed", action="store_true",
                    help="initialize a new (or re-seeded) sat's trim to the MEDIAN of the "
                         "converged fleet trims instead of 0. The converged trim is the "
                         "chain's deterministic frac-N LO offset (same for every sat, stable "
                         "across restarts -- e.g. the L5 chain sits ~+30 Hz), so the fleet "
                         "median is the right prior; the carrier twin of the code-bias "
                         "seeding above (strong sats calibrate the clock so weak ones start "
                         "on it).")
    ap.add_argument("--force-doppler-rate", type=float, default=None,
                    help="REPLAY BENCH ONLY: attach this doppler_rate_hz_s to every seed (a "
                         "recorded capture's sky is at another epoch, so no almanac rate) to "
                         "exercise the tracker's NCO Doppler-rate feed-forward offline.")
    ap.add_argument("--dll-gain", type=float, default=0.25,
                    help="code delay-lock-loop gain (0 = off): each poll, nudge a persistent "
                         "per-PRN cp TRIM by gain * tau_est from the combiner's E/L discriminator. "
                         "The trim rides on top of the search-fit cp, converging to the fit's "
                         "grid-quantization bias -- sub-chip code tracking with no per-record "
                         "decisions (R1, docs/gnss_architecture_audit.md).")
    ap.add_argument("--dll-spacing", type=float, default=0.5,
                    help="tracker Early/Late spacing in chips (must match dll_spacing_chips)")
    ap.add_argument("--dll-leak-present", type=float, default=0.05,
                    help="DLL integrator leak on the FLEET path, where `present` has already "
                         "confirmed signal (--dll-combiners). The ordinary --dll-leak caps the "
                         "reachable correction at (gain/leak)*0.25 = 1.25 chips, which is below "
                         "the residuals actually seen and leaves the loop pushing at a railed "
                         "discriminator without arriving. 0.01 lifts that ceiling to 6.25 chips "
                         "-- past the 3.27-chip tracker grating-lobe spacing -- while still "
                         "mean-reverting over ~100 updates. The leak was a stand-in for a signal "
                         "test; the fleet path now has a real one, so it does not need to be "
                         "this strong. The single-combiner fallback keeps --dll-leak.\n"
                         "MEASURED 2026-08-04 AND REVERTED: 0.01 made locks strictly WORSE -- "
                         "best-q median 2.05 -> 1.03 and samples above q 2.0 went 4/8 -> 0/8, "
                         "every satellite driven to noise. The leak is not only a ceiling: when "
                         "the discriminator is RAILED (|disc| ~ 0.9, which is most of the time "
                         "here) tau saturates at 0.25 chips and carries only the SIGN, not the "
                         "distance -- so the integrator pushes one way indefinitely and the leak "
                         "is what bounds the excursion. Default restored to 0.05; the knob is "
                         "kept so the experiment is repeatable, not because it should be moved.")
    ap.add_argument("--dll-leak", type=float, default=0.05,
                    help="DLL integrator leak (0 = pure integrator): trim mean-reverts each "
                         "update so discriminator NOISE can't random-walk it to the clamp. DC "
                         "loop gain = dll_gain/dll_leak; ~1/leak windows of smoothing.")
    ap.add_argument("--nh-hint", action="store_true",
                    help="hint the search's secondary-code alignment from the EPHEMERIS, so the "
                         "acquire scans nh_hint_span alignments instead of all 20 (~92%% of a "
                         "pass). nh at transmit is round((gpst - range/c + clk_sv)/period) mod "
                         "overlay_len -- the same convention --nh-assist uses for the combiner "
                         "-- leaving ONE global constant, the receiver clock reference, which "
                         "any detection measures and every satellite shares. Works for sats "
                         "NEVER detected, which is where a full 20-way scan hurts most. "
                         "Requires --almanac and the capture time anchor.")
    ap.add_argument("--nh-hint-span", type=int, default=2,
                    help="alignments scanned either side of the prediction. Measured 2026-08-04: "
                         "the offset is 16 with a +-2 spread, so span 2 (5 of 20 = 4x) covered "
                         "100%% of samples while span 1 (6.7x) covered 85%%. The stage keeps its "
                         "own nh_hint_span; this only sizes the LOG. The +-2 spread on a "
                         "deterministic quantity is an OPEN anomaly, not an accepted tolerance.")
    ap.add_argument("--nh-hint-min-samples", type=int, default=6,
                    help="DISTINCT detections (by ref_hop) pooled before the nh offset is "
                         "trusted and hints are sent. The constant is common to all "
                         "satellites, so this fills quickly. Counts distinct detections, not "
                         "cycles: re-counting an unchanged detection every --interval made "
                         "this measure UPTIME rather than evidence (fixed 2026-08-10).")
    ap.add_argument("--dr-max-solve-mad-chips", type=float, default=100.0,
                    help="refuse the receiver-clock solve when the per-satellite offsets "
                         "scatter by more than this (MAD, chips). A circular median over "
                         "--dr-min-sats satellites is a MEASUREMENT only if its inputs "
                         "agree; on a starved fleet the detections are noise, scatter "
                         "~uniformly over the code, and their median is an arbitrary number "
                         "that used to be snapped straight into the clock and shipped as "
                         "every seed. Not a tuning knob -- real offsets cluster inside ~+-10 "
                         "chips against a uniform-noise MAD of ~2557 on a 10230-chip code. "
                         "Refusing leaves the clock UNSET, which is strictly better than "
                         "wrong: a wrong clock also acquires nothing, and additionally "
                         "poisons the search hints that would have recovered it (docs 11.33).")
    ap.add_argument("--nh-hint-max-age-s", type=float, default=600.0,
                    help="drop nh-offset samples older than this, and stop hinting entirely "
                         "when fewer than --nh-hint-min-samples remain. THE HINT MUST BE ABLE "
                         "TO EXPIRE. It narrows the search to +-nh_hint_span of 20 overlay "
                         "phases, so a WRONG offset points the scan away from the signal and "
                         "no detection can arrive to correct it -- a closed loop whose only "
                         "escape was the code clock random-walking back onto truth by chance "
                         "(observed 2026-08-10: ~15 min of self-reinforcing outage that read "
                         "as a frontend sensitivity loss, docs 11.33). Default 600 s is ~half "
                         "the 1276 s per-PRN revisit, so a healthy fleet always has fresh "
                         "samples and only a genuinely starved one widens.")
    ap.add_argument("--publish-port", type=int, default=0,
                    help="serve the FLEET-MERGED per-PRN state on this port (0 = off): "
                         "GET /get_status returns rows in GnssCoherentCombiner's schema, so "
                         "the viewer consumes them unchanged, but built from ALL "
                         "--dll-combiners instead of one instance's 6.7%% of the L5 lobe. The "
                         "browser cannot merge 14 origins across 8 hosts itself, and the "
                         "broker is already the shared-knowledge node (pooled l-a, clock-freq "
                         "bias, fused LO, cross-band assist) -- this is the same kind of "
                         "object. Coherent statistics are best-of-instance, not merged, and "
                         "say so via coh_src; see FleetPublisher.")
    ap.add_argument("--carrier-from-code", action="store_true",
                    help="SHADOW: log the carrier error DERIVED from the fitted code slope "
                         "(x f_carrier/f_chip) beside the measured carrier_hz_resid, without "
                         "applying it. carrier_hz_resid is signal-free at CHORD SNR, so the "
                         "carrier loop was integrating noise; the code side is strong. Compare "
                         "the two before letting the derived value drive anything.")
    ap.add_argument("--dop-rate-model-tol", type=float, default=0.12,
                    help="reject a FITTED doppler rate that disagrees with the model's by\n"
                         "more than this (Hz/s) and keep the model. --dop-rate-max only\n"
                         "bounds magnitude, so a wrong-SIGNED or half-size fit passes it;\n"
                         "measured on sky, PRN 8 was seeded 2.18x small and PRN 30 with the\n"
                         "wrong sign. BRDC range-rate differencing is good to ~0.05 Hz/s, so\n"
                         "0.12 admits real fit noise and rejects a fit that is simply wrong.\n"
                         "0 disables the cross-check (the pre-2026-08-05 behaviour).")
    ap.add_argument("--dop-rate-max", type=float, default=0.8,
                    help="reject a fitted doppler rate beyond this (Hz/s) and fall back to the "
                         "almanac's. PHYSICAL, not tuned: GPS Doppler acceleration peaks near "
                         "0.94 Hz/s at L1, so ~0.70 at L5 (x 1176.45/1575.42); 0.8 leaves "
                         "margin. Seeding a noise-fitted rate ADDS curvature error -- observed "
                         "on deploy, PRN 20 fitted at -1.16 Hz/s.")
    ap.add_argument("--dop-rate-min-pts", type=int, default=4,
                    help="detections needed before the MEASURED doppler rate is seeded in place "
                         "of the almanac's. Below this the model is the better bet.")
    ap.add_argument("--dop-rate-min-span-s", type=float, default=30.0,
                    help="baseline the doppler-rate fit must span (s). Slope error goes as "
                         "sigma/(T*sqrt(N/12)), so at ~1.5 Hz detection scatter 4 points over "
                         "44 s give ~0.06 Hz/s and 8 over 88 s ~0.02 Hz/s, against BRDC's "
                         "measured 0.108 Hz/s error. A short baseline fits noise.")
    ap.add_argument("--spectrum-endpoints", default="",
                    help="FLEET PHASE-SLOPE DELAY FIT (task #32, docs/CHORD_JOINT_TRACKING.md "
                         "P1): comma-separated GnssGpuRecordAssemble endpoints ({a..b} ranges "
                         "expanded) whose /get_spectrum -- the per-channel prompt sums BEFORE "
                         "the cross-channel combine -- feed the joint (tau, phi_i) fit. A "
                         "delay is a phase ramp across frequency; the fleet's UNION of combs "
                         "is what suppresses the 3.27-chip grating lobes a single instance's "
                         "stride-16 comb cannot resolve. MEASUREMENT ONLY today: logged as "
                         "SPEC-FIT and published as spec_tau_chips/spec_peak_ratio, feeding "
                         "no loop. Empty = off, and every recorded transcript replays "
                         "byte-identically (replay is strict-ordered; this flag not being in "
                         "an old transcript's argv is what keeps the new GETs out of replay).")
    ap.add_argument("--joint-shadow", action="store_true",
                    help="P2a (task #33, docs/CHORD_JOINT_TRACKING.md section 3a): run the "
                         "JOINT receiver-state solve -- one [clk, clk_rate, b_sat[i]] "
                         "estimated together from the dead-reckon integrity residuals, "
                         "receiver-scope so all three constellations feed ONE clock -- and "
                         "LOG it beside the circular-median clock it is meant to replace. "
                         "Consumed by nothing: this flag cannot change a seed, a POST or a "
                         "transcript digest, by construction. The comparison it exists to "
                         "make: the median treats the +-3-7 chip per-sat spread (docs "
                         "11.22) as error to be gated on; the joint solve reads the same "
                         "numbers as clock PLUS per-sat bias. If the biases hold steady "
                         "while the clock stays smooth, the revised P2 is right and the "
                         "consumers can start switching one per commit (P2b).")
    ap.add_argument("--dr-max-drift-chips-s", type=float, default=1.0,
                    help="reject a dead-reckon clock DRIFT estimate beyond this (chips/s). "
                         "The estimate is a difference of two clock solves, so any "
                         "discontinuity -- a node or F-engine restart -- enters as motion "
                         "that never happened, and the a=0.05 EMA then bleeds it off over "
                         "~10 min while sweeping every model-primary seed off peak. The "
                         "true drift is ~4e-4 chips/s on this GPS-disciplined clock, so 1.0 "
                         "rejects nothing real (2026-08-09: +223 and -36 chips/s observed "
                         "after node restarts).")
    ap.add_argument("--joint-consume", default="",
                    help="P2b: comma-separated JOINT-state consumers to switch LIVE, one "
                         "name per commit so each is A/B-able on its own. Empty (default) "
                         "= pure shadow. Names: 'rate' (seeded code-rate clock from "
                         "clk_rate instead of the l-a EMA -- which measured ~80x biased, "
                         "see the call site); 'slew' (the dead-reckon SLEW TARGET's "
                         "clock+bias offset from clk + b_sat instead of the clock EMA plus "
                         "the SatBiasFilter -- the half of the state P2c validates, and the "
                         "half that survived the 2026-08-10 gauge collapse); later 'clock'. "
                         "The shadow comparison for each is logged whether or not it is "
                         "consumed, so the A/B exists before the switch. Implies "
                         "--joint-shadow.")
    ap.add_argument("--joint-mask-prn", default="",
                    help="P2c, THE DISCRIMINATING TEST FOR THE ARCHITECTURE: comma-separated "
                         "PRNs whose measurements are WITHHELD from the joint solve while "
                         "their residual is still logged. Vector tracking's claim is that a "
                         "satellite rides the SHARED receiver state -- so a masked sat must "
                         "COAST on clk + clk_rate*dt with its b_sat frozen, and its "
                         "predicted-minus-actual must stay flat. If it drifts instead, the "
                         "state is not carrying it and the per-sat loops are doing the work "
                         "the joint solve claims to have replaced. Measurement-only: the "
                         "masked sat is still SEEDED normally, so nothing on sky changes. "
                         "Mask a STRONG sat -- masking a weak one tests nothing, since its "
                         "own measurements were not holding it up anyway.")
    ap.add_argument("--joint-model-primary", action="store_true",
                    help="Feed MODEL-PRIMARY chains (E5a/B2a/E5b/B2b) into the joint solve. "
                         "⚠️ DEFAULT OFF AFTER A MEASURED REGRESSION (2026-08-10). The "
                         "model-primary measurement carries NO QUALITY GATE -- a detection has "
                         "an SNR to threshold on, a dead-reckoned seed does not -- so a "
                         "satellite whose tracker has not converged contributes a y of hundreds "
                         "of chips. birth_max does not stop it: that gate is conditioned on "
                         "P[0,0] < 100, which is false for minutes after a restart, so early "
                         "births are unvetted. Measured with it on: clk walked to +445 chips "
                         "against a true ~150, biases reached -427/-330/+281 where they should "
                         "be a few, tau_band blew to +792 chips, and 67-82% of updates were "
                         "rejected. It looked healthy for the first hour only because the "
                         "satellites then up happened to be good; the failure arrives with the "
                         "next rise. Re-enable once the measurement is gated on tracking "
                         "quality (deep_snr/coh_frac), which is what #33 P2b needs anyway.")
    ap.add_argument("--joint-mask-after", type=int, default=50,
                    help="P2c: engage --joint-mask-prn only once that satellite has this many "
                         "ACCEPTED updates. Masking from startup is a vacuous test -- the sat "
                         "is never born into the state, so there is no b_sat to freeze and "
                         "nothing to coast on. It must be established FIRST and withheld "
                         "after, which is the whole point: we are testing whether the state "
                         "REMEMBERS a satellite, not whether it can invent one. "
                         "⚠️ THE TWO SEEDING DISCIPLINES FEED AT VERY DIFFERENT RATES, which is "
                         "why this defaults low: a model-primary chain offers every seeded sat "
                         "EVERY cycle, so the count accrues in seconds, while a search-anchored "
                         "GPS sat only enters when a detection clears --joint-min-snr and can "
                         "take hours. Calibrated on the wrong path a threshold either fires "
                         "instantly or never -- 200 fired in 6 s on Galileo and had not fired "
                         "on GPS in 14 min. 50 is reachable on both and ample to establish a "
                         "bias (sigma_b0 10 against a 0.3-chip measurement converges in tens).")
    ap.add_argument("--joint-min-snr", type=float, default=60.0,
                    help="minimum detection SNR for a residual to enter the JOINT solve. "
                         "The default matches --period-check-snr, whose docstring records "
                         "the measurement: below it a detection's phase is noise with "
                         "~2000-chip residuals. Ungated, ONE such detection walked the "
                         "state to -0.028 ppm on 2026-08-09 (docs 11.23).")
    ap.add_argument("--joint-min-deep-snr", type=float, default=10.0,
                    help="minimum TRACKER significance (deep_snr, floor-cleared) for a "
                         "residual to enter the joint solve. --joint-min-snr gates on the "
                         "SEARCH SNR carried in `best`, which is a different and much "
                         "staler number: detections latch for up to the 1276 s per-PRN "
                         "revisit, so a satellite that has left the beam keeps presenting "
                         "the strong SNR of its last good scan while its tracker output is "
                         "already noise. That is exactly how PRN 25 got in on 2026-08-10 -- "
                         "elevation RISING 45->65 deg (it transited out of the beam, it did "
                         "not set), deep_snr 30->2.4, coh_frac 0.94->0.19, code phase "
                         "uniform over the 10230-chip code -- and destroyed the state. "
                         "MEASURED, not guessed: over 6827 GPS rows the population is "
                         "sharply bimodal, noise at deep_snr 2-3 / coh_frac 0.2 and signal "
                         "at 24+ / 0.92+; 10 sits in the empty gap and admits 70.5%%.")
    ap.add_argument("--joint-min-coh-frac", type=float, default=0.3,
                    help="minimum coherent fraction for a residual to enter the joint solve. "
                         "Nearly redundant with --joint-min-deep-snr (both gates select the "
                         "same 70.5%% of rows, which is the two measuring the same physics) "
                         "and kept as a cheap independent guard, not as a second opinion.")
    ap.add_argument("--joint-slew-max-chips", type=float, default=5.0,
                    help="P2b consumer 3: refuse the joint slew target if it disagrees with "
                         "the legacy (clock EMA + SatBiasFilter) target by more than this. "
                         "Both describe the same receiver, so a few chips is the whole "
                         "plausible range -- b_sat runs +-3-6 chips and the clock is common "
                         "to both. A larger gap means one estimator is broken, and the "
                         "legacy path is the one with years behind it. Same discipline as "
                         "--joint-max-rate-ppm on consumer 1, which would have refused the "
                         "-0.028 ppm runaway that froze the trackers no matter what went "
                         "wrong upstream.")
    ap.add_argument("--joint-min-sats", type=int, default=4,
                    help="a JOINT consumer refuses to act on a state carrying fewer than "
                         "this many satellites: with the mean(b)=0 gauge a thin fleet lets "
                         "one satellite's bias leak into the clock at 1/N.")
    ap.add_argument("--joint-max-rate-ppm", type=float, default=0.01,
                    help="a JOINT rate consumer REFUSES a |clk_rate| beyond this (ppm) and "
                         "falls back to the l-a EMA. CHORD's reference is GPS-disciplined "
                         "and measures 4e-5 ppm, so 0.01 is ~250x headroom and still "
                         "refuses the -0.028 ppm runaway that froze the trackers.")
    ap.add_argument("--joint-sigma", type=float, default=0.3,
                    help="measurement sigma (chips) for --joint-shadow. The search cp noise "
                         "is 0.03-0.5 chips per-sat-conditions; 0.3 is the middle of that "
                         "and NOT tuned against the answer -- the state's own covariance "
                         "reports whether it was right.")
    ap.add_argument("--bsat-gain", type=float, default=0.02,
                    help="Per-cycle gain of the b_sat loop (task #33): the per-satellite "
                         "path-bias filter fed by the fleet phase-slope tau. 0 disables the "
                         "UPDATES (b freezes at 0 / its last value) while the fit and its "
                         "publishing keep running -- the control setting for the oscillation "
                         "causality experiment: the 2026-08-09 tau test found the closed "
                         "system limit-cycling at ~10 min with the DLL trim swinging IN "
                         "PHASE with b (corr +0.65, trim +-1.1 chips vs b +-0.5), so 'does "
                         "the plant oscillate with b frozen?' is what separates 'b causes "
                         "it' from 'b follows a pre-existing trim/slew oscillation'.")
    ap.add_argument("--dll-combiners", default="",
                    help="FLEET-COMBINED DLL (docs/CHORD_GNSS_SHARED_DLL.md): comma-separated "
                         "combiner endpoints ({a..b} ranges expanded) whose RAW Early/Prompt/"
                         "Late powers are SUMMED before the discriminator is formed, closing "
                         "ONE code loop at full signal bandwidth. On CHORD the F-engine comb "
                         "spreads L5 across every node, so a single instance correlates 7 x "
                         "195.3 kHz of a 20.46 MHz lobe -- 6.7%%, -11.8 dB -- and no instance "
                         "can do better; that gap, not sensitivity or the quality gate, is why "
                         "the code lock is episodic. The combine is legitimate because the "
                         "discriminator is NON-COHERENT: (E-L)/(E+L) is built from POWERS, "
                         "which add with no phase alignment and no sample sync, only the same "
                         "window (pow_hop, exact). Empty = today's behaviour, --combiner alone. "
                         "REQUIRES trim_gain 0 on the trackers: E/L are measured relative to "
                         "the phase each instance despread at, so independent local trims make "
                         "the sum SMEAR instead of sharpen (design section 5).")
    ap.add_argument("--dll-hop-window-s", type=float, default=5.0,
                    help="fleet DLL: how far back from the newest window an instance's "
                         "measurement may be and still join the sum, in seconds (converted "
                         "once to an integer hop count -- the key itself is never a float). "
                         "Instances free-run, so their emit phases differ by up to one emit "
                         "period; the code moves 0.0033 chips/s (measured), so even a full "
                         "second of spread is 0.003 chips, far below anything the "
                         "discriminator resolves. DEFAULT 5 s, NOT the 0.084 s spread measured "
                         "on 2026-08-03: instances free-run, so their emit phases random-walk "
                         "apart without bound, and within hours that spread had grown to 0.503 "
                         "s. A 0.5 s window then straddled it and the fleet flapped 14 -> 8 "
                         "instances sample to sample, silently halving the combined bandwidth. "
                         "Size this for DRIFT, not for a snapshot -- 5 s costs 0.017 chips.")
    ap.add_argument("--dll-quality-sigma", type=float, default=3.0,
                    help="fleet DLL: how many sigma above the MEASURED q noise floor a PRN must "
                         "sit before its trim integrates. q = 2P/(E+L) is a peak-SHARPNESS "
                         "metric, not an SNR: exactly 1.0 with no peak (all three taps see "
                         "equal noise power), 4.0 for a clean lock at 0.5-chip spacing. Summing "
                         "instances does NOT raise it -- every tap's mean scales alike -- it "
                         "SHRINKS its spread as 1/sqrt(K), so the right bar falls as the fleet "
                         "grows and no constant can be correct for more than one fleet size. "
                         "The floor is therefore re-measured each cycle as median + this many "
                         "MAD-sigma over the live q population (most tracked PRNs are "
                         "signal-free at any moment, so the median IS the no-peak value), and "
                         "logged every time it is used.")
    ap.add_argument("--dll-quality-min", type=float, default=2.2,
                    help="fleet DLL: FALLBACK q bar, used only when fewer than 8 PRNs report "
                         "and the noise population cannot be characterised. 2.2 is the measured "
                         "single-instance bar (noise mean ~1.0 with a tail to 1.87 on sky "
                         "2026-08-03), so the fallback is the conservative one-node answer.")
    ap.add_argument("--dll-min-instances", type=int, default=2,
                    help="fleet DLL: instances that must report the same window before their "
                         "sum is used. Below this the PRN falls back to the single --combiner "
                         "discriminator, so a partially-down fleet degrades instead of "
                         "stalling.")
    ap.add_argument("--fleet-coherent", action=argparse.BooleanOptionalAction, default=True,
                    help="CROSS-NODE COHERENT COMBINE (fleet_coherent). Polls /get_records from "
                         "the --dll-combiners endpoints and removes the per-record COMMON SKY "
                         "PHASE -- ~0.75 rad, 0.984-coherent across instances but only ~0.57 "
                         "autocorrelated in time, which is why no within-node temporal estimator "
                         "can touch it and why the deep folds sat at ~14 sigma against a ~100 "
                         "sigma thermal ceiling. Each instance is corrected against the OTHERS "
                         "(leave-one-out), and the fleet total uses a one-way split so the noise "
                         "estimate survives. Measured 2026-08-05: 14 -> 170-700 on the bright "
                         "sats, against a shuffled-null floor of ~4. Costs one extra REST poll "
                         "per combiner and ~0.2 s per cycle; publishes into the coherent fields "
                         "the FleetPublisher previously had to take best-of-one-instance.")
    ap.add_argument("--n2-combiners", default="",
                    help="CROSS-NODE COHERENT COMBINE FOR PATH B: comma-separated *_n2combine "
                         "endpoints ({a..b} ranges expanded), run through the SAME "
                         "fleet_coherent as --dll-combiners but kept as a separate population "
                         "so the two paths stay independently measurable. Empty = off.\n"
                         "\n"
                         "This is not an optimisation, it is how path B works at scale. A node "
                         "holds every 8th science channel and each GPU a stride-16 comb, so one "
                         "instance sees 7 of the 105 channels under the L5 lobe -- and on a "
                         "NARROW signal (L2C, E5b) at full CHORD a node may hold ONE channel. "
                         "The fleet combine is then the measurement, not an improvement to it.\n"
                         "\n"
                         "THE ESTIMATOR IS PARTITION-INVARIANT, which is the useful property "
                         "here and was verified rather than assumed. Holding the total aperture "
                         "fixed at 12 bands and 128 records and splitting it 2x6, 4x3, 6x2 and "
                         "12x1 through this same function gives fleet deep_snr 81.9/80.9/81.6/"
                         "82.0 at a fleet per-record SNR of 10.4, and 26.2/26.4/26.3/26.6 at "
                         "3.5. Node count does not enter. The reason is that the leave-one-out "
                         "reference is the sum of the OTHER instances: K instances of B/K bands "
                         "give it (K-1)/K of the total aperture, so slicing finer makes the "
                         "reference slightly BETTER, while costing K-1 alignment constants "
                         "instead of one. Those cancel. So a narrow signal spread one channel "
                         "per node over many nodes combines exactly as well as the same channels "
                         "gathered on a few -- which is what makes L2C and E5b viable at full "
                         "CHORD.\n"
                         "\n"
                         "WHAT DOES BITE is the FLEET TOTAL per-record SNR, a bands x elements "
                         "question with no node term in it. Down to ~1.7 the partitions are "
                         "still indistinguishable (11.4/11.8/11.8/11.6); at 0.87 the finest "
                         "split finally loses ground (4.3 -> 2.9), and that is already inside "
                         "the floor's shadow. Drive it with more bands or more elements "
                         "(gnssElemCal / elem_sum), not with a different node layout.\n"
                         "\n"
                         "NB the `align` figure in the log falls with instance count at fixed "
                         "total (0.917 at 2x6 down to 0.611 at 12x1, same total SNR) because "
                         "each alignment is estimated from one instance's data. It is a "
                         "per-instance diagnostic, NOT a health metric for the combine -- "
                         "deep_snr over those same four partitions is flat to 2%.")
    ap.add_argument("--coh-min-instances", type=int, default=3,
                    help="fleet coherent: instances that must see a PRN before it is combined. "
                         "Below this the PRN keeps its single-instance coherent numbers.")
    ap.add_argument("--coh-min-records", type=int, default=32,
                    help="fleet coherent: common records (hops) required per PRN.")
    ap.add_argument("--coh-floor-margin", type=float, default=3.0,
                    help="fleet coherent: multiple of the MEASURED null floor a fleet deep_snr "
                         "must clear to be published as a detection. The floor is re-measured "
                         "every cycle by shuffling each instance's records (real amplitudes, no "
                         "common phase), exactly as the fleet DLL measures its q floor -- never "
                         "a constant, because the right bar depends on fleet size and window "
                         "length. Characterised: null 99th pct 4.15 / max 5.73 vs a weakest real "
                         "detection of 22.2, so 3x sits in a clean gap.")
    ap.add_argument("--cl-assist", action="store_true",
                    help="LEGACY single-chain CL mode (superseded by --cl-tracker, which keeps "
                         "CM running as the in-run control): lift each seed's code_phase_chips "
                         "IN PLACE by k*10230 with the CL segment k COMPUTED from absolute "
                         "capture time (the airspy /adcstat utc0_sample0 anchor) + almanac "
                         "range. CL's 1.5 s epoch is GPS-time-locked, so k is arithmetic, not "
                         "a 75-way search. Needs --almanac; the main trackers must despread "
                         "GPS_L2C_CL. Mutually exclusive with --cl-tracker.")
    ap.add_argument("--adc-stage", default="airspy_in",
                    help="airspy input stage name for the utc0_sample0 anchor GET (CL assist)")
    ap.add_argument("--time0-endpoint", default=None,
                    help="CHORD: REST path (relative to --rest-url, e.g. telescope/time0_ns) "
                         "serving the F-engine's GPS-disciplined absolute time of frame 0. Used "
                         "INSTEAD of the airspy /adcstat anchor. The airspy node stamps sample 0 "
                         "with host wall-clock -- good to milliseconds -- which is why it must "
                         "then SOLVE the receiver clock from measured code phases. CHORD's "
                         "frame 0 is disciplined to GPS via IRIG-B/PPS and is exact, so the "
                         "anchor is a fact rather than an estimate.")
    ap.add_argument("--dr-clock-drift", type=float, default=None,
                    help="CHORD: prime the dead-reckon clock DRIFT (chips/s). The drift "
                         "estimator needs consecutive multi-sat solutions 0.5-30 s apart; a "
                         "search whose passes take minutes never provides them, so drift pins "
                         "to zero on stale repeats and the clock freezes while the true "
                         "receiver clock walks (measured 0.044 chips/s = +5 Hz at L5 on the "
                         "CHORD GPSDO). Priming it makes the age-correction terms treat "
                         "minutes-old detections consistently, which is what lets a "
                         "slow-cadence search bootstrap the fast tracker loop.")
    ap.add_argument("--dr-clock-chips", type=float, default=None,
                    help="CHORD: prime the dead-reckon receiver clock (chips) instead of "
                         "bootstrapping it from measured code phases. THIS IS WHAT LETS A NODE "
                         "WITH NO SEARCH STAGE COLD-START. The bootstrap needs --dr-min-sats "
                         "satellites already tracking to take a median of their residuals, but "
                         "nothing can track until it is seeded -- on the airspy node the search "
                         "stage breaks that circle. With a GPS-disciplined F-engine the offset "
                         "is known a priori (0 plus a fixed instrumental/cable delay), so pass "
                         "0.0 to start and calibrate the constant later from the measured "
                         "integrity residual, which the broker logs every cycle. THE PRIME IS "
                         "A SEED, NOT A MEASUREMENT: a chain with detectors REPLACES it with "
                         "its first multi-sat solve rather than EMA-ing toward one, so a wrong "
                         "prime costs one cycle rather than ~20.")
    ap.add_argument("--seed-doppler", default="auto", choices=("auto", "det"),
                    help="which Doppler the SEED carries. 'auto' (default, unchanged) prefers "
                         "the almanac/DR model + solved clock bias, which is smooth and owns "
                         "the undetected sats. 'det' uses the search's MEASURED Doppler "
                         "instead. Pick 'det' when the model is not trustworthy to well under "
                         "a Hz at the seed's age: cp0 is an ARGUMENT, so a seed's phase moves "
                         "chip_rate/carrier chips per second per Hz of Doppler error (0.0087 "
                         "chips/Hz/s) for as long as the tracker extrapolates it. Measured on "
                         "CHORD 2026-08-01: model-vs-measured +231 Hz with a 244 Hz spread, "
                         "against a measured Doppler good to ~3 Hz (proven by the residual "
                         "code rate of a real lock) -- at a 456 s anchor that is 340 chips of "
                         "seed error versus 12. 'det' also makes cp_to_seed_currency a no-op, "
                         "since cp0 was fit at exactly that Doppler.")
    ap.add_argument("--long-code-segments", type=int, default=75,
                    help="number of primary periods in the overlaid/long code the TRACKERS "
                         "despread (L2C CL = 75 x 10230; GPS L5 Q5 with NH20 baked in = 20 x "
                         "10230). The time-assist below picks which one, so this must match the "
                         "tracker's `signal`, not the search's.")
    ap.add_argument("--long-code-epoch-s", type=float, default=1.5,
                    help="the long code's GPS-time-locked repeat period, seconds (L2C CL 1.5; "
                         "L5 NH20 0.02). The assist needs unix-time mod EPOCH == GPS-time mod "
                         "EPOCH, which holds when GPS-UTC (whole seconds) and the GPS epoch "
                         "offset (315964800) are both multiples of it -- true for 1.5 and for "
                         "0.02. Absolute-time accuracy needed is ~EPOCH/2.")
    ap.add_argument("--bias-min-snr", type=float, default=0.0,
                    help="detections below this SNR do not enter the clock-freq bias median. "
                         "The bias is common-mode, so its uncertainty is (per-sat Doppler "
                         "error)/sqrt(N) -- one noise satellite is costly when N is small. "
                         "Ungated on CHORD the raw estimate scatters 10.5 Hz; the acquire's "
                         "own error predicts 0.8 Hz at N=2. 0 (default) keeps every point.")
    ap.add_argument("--nh-period-offset", type=int, default=0,
                    help="EXPERIMENT (2026-08-02): shift every seed's overlay period by N "
                         "primary code periods. The oracle measures the seeded period as a "
                         "CONSTANT 4 too high -- 4/4 strong detections, 3 satellites, seed ages "
                         "151-680 s, oracle ratios 30.1/25.8/17.4/9.8 -- and -4 == +16 mod 20, "
                         "where 16 code periods = 3125 hops = Mp = the anchor the search builds "
                         "repl0 at. This exists to TEST that, not to fix it: a constant that "
                         "works without a mechanism is how refine_span:4096 got baked in. "
                         "Applied to BOTH code_phase_at_ref_chips (which propagate_seed prefers) "
                         "and code_phase_chips. 0 = off.")
    ap.add_argument("--fit-min-snr", type=float, default=0.0,
                    help="detections below this SNR do not enter the cp-rate fit history. The "
                         "fit resolves a ~0.0148 chips/s residual; a near-threshold detection's "
                         "phase is noise, so one bad point destroys the slope rather than "
                         "degrading it. 0 (default) keeps every point -- right for the "
                         "prototype, whose detections sit well above threshold and whose "
                         "revisit is seconds. CHORD wants ~60 alongside --fit-gap-s 900.")
    ap.add_argument("--period-continuity", default="check",
                    choices=("check", "correct", "off"),
                    help="what to do when the search's reported overlay period disagrees with "
                         "the one predicted from the previous pass. 'check' (default) LOGS the "
                         "disagreement and applies nothing -- correct since the search began "
                         "measuring the period from the acquire's coarse lag (4371ff4eb); a "
                         "nonzero disagreement on a strong satellite then means the SOURCE "
                         "regressed, which wants an alarm, not a silent repair. 'correct' "
                         "restores the old override (it stored its own correction and "
                         "predicted from that, so one bad call was permanent). 'off' skips the "
                         "comparison entirely.")
    ap.add_argument("--period-check-snr", type=float, default=60.0,
                    help="detections below this SNR do not enter the period-continuity history "
                         "and their disagreements are logged as 'weak det' rather than as "
                         "source regressions. Measured on CHORD 2026-08-02: above ~60 the "
                         "within-period phase is self-consistent to a few chips across a "
                         "400 s gap, below it the residuals are ~2000 chips, i.e. noise.")
    ap.add_argument("--cl-time-adjust", type=float, default=0.0,
                    help="seconds added to the CL time-assist clock -- escape hatch for a future "
                         "non-multiple-of-1.5s GPS-UTC offset or a known host-clock bias")
    ap.add_argument("--cl-tracker", default=None,
                    help="L2C CL SIBLING-CHAIN mode (Mechanism A of the shared-knowledge plan; "
                         "supersedes --cl-assist's in-place lift): derive one CL pilot seed row "
                         "per CM row -- same doppler/dop-rate/carrier-trim/ref_hop (SAME carrier, "
                         "SAME 511.5 kcps chip clock), code_phase lifted by k*10230 with the "
                         "segment k pinned from absolute capture time + model range + SV clock "
                         "(t_sv = t_gpst - range/c + clk, the nh-assist convention proven to "
                         "0.01 chip by c31_convention.py) and SNAPPED to the measured CM cp -- "
                         "and POST them to THIS tracker stage's /set_seeds. The CM chain is "
                         "untouched: it stays up as the in-run control, and CL certification is "
                         "judged against it. Needs --almanac + the airspy utc0_sample0 anchor.")
    ap.add_argument("--cl-combiner", default=None,
                    help="the CL chain's combiner stage: polled each cycle so the CL-vs-CM "
                         "deep_snr comparison (the segment-pin VERIFY -- a wrong k despreads "
                         "as noise) lands in this broker's own log next to the k it verifies.")
    ap.add_argument("--cl-kscan-prn", type=int, default=0,
                    help="DIAGNOSTIC (default 0 = OFF): step the CL segment for THIS probe PRN "
                         "through {k, k-1, k+1, k-2, k+2} and log which offset despreads best. "
                         "Convention-free test for the whole-segment anchor bug that fine_ms "
                         "cannot see (fine is the residual after round()). Only the probe PRN's "
                         "SEED is shifted; the fleet's pin, the fine, and the auto-center are "
                         "untouched, so this is safe to leave off and harmless when on. Pick a "
                         "strong CM sat as the probe.")
    ap.add_argument("--xband-combiner", default=None,
                    help="S5 CROSS-BAND ASSIST (SHADOW): a SIBLING band's combiner stage "
                         "(e.g. l1_gps_combiner) whose per-sat tracked Doppler this broker "
                         "reads to predict THIS band's Doppler by the exact carrier ratio. "
                         "The satellite-motion part is geometry -- common to both bands and "
                         "scaling as f_this/f_sibling -- so `(D_sib - LO_sib)*ratio + LO_this` "
                         "predicts this band's observed Doppler; the LO terms come from each "
                         "band's own S2 fused state (the dongle LOs are INDEPENDENT -- "
                         "measured, no GPSDO common-mode -- so neither can be borrowed). "
                         "SHADOW: logs the prediction beside this band's actual acquisition "
                         "for every dual-tracked sat and accumulates the residual; nothing is "
                         "seeded from it yet. The eventual flip is RESCUE-ONLY (seed a sat "
                         "this band cannot predict itself -- cold start / stale BRDC), the "
                         "S2d lesson applied.")
    ap.add_argument("--xband-lo-dongle", default=None,
                    help="the sibling band's S2 state dongle key (e.g. gps_l1), to read its "
                         "fused LO for --xband-combiner")
    ap.add_argument("--xband-carrier-hz", type=float, default=None,
                    help="the sibling band's carrier frequency (Hz), for the Doppler ratio")
    ap.add_argument("--xband-seed", type=int, default=1,
                    help="S5b THE FLIP (default ON; a provable no-op in normal operation): "
                         "emit a SEARCH DOPPLER HINT from the cross-band prediction for a sat "
                         "the SIBLING band tracks but THIS band has NO prediction of its own "
                         "for (not in BRDC pred / no almanac). Cross-band transfers Doppler "
                         "(carrier ratio) but NOT code phase (the codes differ), so it hints "
                         "the SEARCH -- narrowing its Doppler window -- it does not seed the "
                         "tracker. RESCUE-ONLY by construction: for any sat BRDC already "
                         "predicts, the BRDC hint stands and NO cross-band hint is added, so "
                         "with fresh BRDC the cross-band hint list is EMPTY. It fires only "
                         "when BRDC is missing a sat the sibling sees (outage / deep cold "
                         "start / a band too weak to hold its own almanac lock) -- the "
                         "S2d-learned rescue-only scope, structural not just gated. 0 = pure "
                         "shadow (log the residual, emit no hints).")
    ap.add_argument("--xband-hint-margin-hz", type=float, default=60.0,
                    help="search margin for a cross-band RESCUE hint (Hz): the cross-band "
                         "seed accuracy is the inter-band MAD (~10 Hz) plus this band's own "
                         "unsolved-LO width, so wider than a BRDC hint but far tighter than "
                         "the blind grid")
    ap.add_argument("--cl-autoseg", type=int, default=1,
                    help="CL segment AUTO-SEARCH (default ON; the durable fix for the "
                         "~40%%-of-launches CL failure): when the CL-vs-CM verify reads a "
                         "dead fleet under strong CM, step an integer-segment correction "
                         "through 0,-1,+1,-2,... (one 20 ms segment per step) and LATCH on "
                         "green. Compensates the whole-segment utc0_sample0 anchor error "
                         "(stamped from system_clock::now() on the first USB transfer, "
                         "tens of ms of per-launch jitter; the auto-center absorbs only "
                         "the fractional part). A working launch latches 0 immediately.")
    ap.add_argument("--cl-autoseg-dwell", type=float, default=30.0,
                    help="seconds per correction step (tracker re-lock + combiner deep "
                         "build; the k-scan measured green appearing well inside 30 s)")
    ap.add_argument("--cl-kscan-chips", default="",
                    help="FRACTIONAL scan mode: CSV of CHIP offsets added to the probe "
                         "PRN's seeded cp (e.g. '0,0.25,-0.25,0.5,-0.5,0.75,-0.75,1,-1') "
                         "instead of whole-segment steps. The comb/sub-chip test: CM/CL "
                         "are chip-interleaved at 1.023 Mcps (one comb slot = 0.5 chip of "
                         "the 511.5 kcps code), and slot parity couples with code phase "
                         "when the replica timeline shifts, so scan a fine grid rather "
                         "than betting on +-0.5 -- a half-chip code offset degrades ~6 dB "
                         "rather than nulling, so any partial despread stands far above "
                         "the ~2 noise floor and names the true offset.")
    ap.add_argument("--cl-kscan-segs", default="",
                    help="explicit CSV of SEGMENT offsets for the scan (e.g. the full-75 "
                         "sweep '0,-1,1,...,-37,37'). The default +-2 neighbourhood only "
                         "exonerates SMALL anchor errors; utc0_sample0 is stamped from "
                         "system_clock::now() on the FIRST USB transfer and carries tens "
                         "of ms of per-launch startup latency -- several 20 ms segments.")
    ap.add_argument("--cl-kscan-dwell", type=int, default=20,
                    help="CL k-scan: broker cycles to dwell per offset (the CL combiner's deep "
                         "integration must respond before stepping). 20 cycles ~= 4 s at the "
                         "0.2 s interval, matching L2C's coherence window.")
    ap.add_argument("--nh-assist", action="store_true",
                    help="secondary-overlay TIME-ASSIST for a per-PRN-overlay pilot (B1C/E5a/B2a): "
                         "POST each visible sat's PREDICTED absolute overlay-chip index (from "
                         "almanac range + BeiDou/Galileo-time, one convention, the combiner "
                         "self-calibrates the constant) to the combiner's /set_nh_hint. The weak "
                         "sats that cannot win the combiner's L-way (1800 for B1C) alignment "
                         "search get the geometrically-correct alignment for free. Needs --almanac; "
                         "the combiner needs nh_assist: true. Fail-safe: a wrong hint just fails "
                         "its floor and the blind search result stands.")
    ap.add_argument("--nh-overlay-len", type=int, default=1800,
                    help="secondary-overlay length in chips (B1C pilot 1800; E5a/B2a CS100 = 100)")
    ap.add_argument("--bias-det-fresh-s", type=float, default=30.0,
                    help="clock-bias solve uses only detections FRESHER than this (seconds). "
                         "A stale detection's (meas - pred) grows at the satellite's Doppler "
                         "rate -- stale-age x dop_rate reads as a fake, GROWING clock bias "
                         "that the seeds then chase (measured: a ~90 s-stale detection walked "
                         "the bias +4 -> +68 Hz and dragged a 55-sigma tracker off the sky).")
    ap.add_argument("--almanac-epoch-utc0", type=float, default=1.0,
                    help="the tracker's capture_utc0 (gnss_node.yaml shared value, 1.0): "
                         "subtracted from combiner row utc to get the FILE POSITION that "
                         "advances the --almanac-epoch clock at the data's own rate.")
    ap.add_argument("--almanac-epoch", type=float, default=0.0,
                    help="REPLAY BENCH ONLY: unix time of the capture's sample 0. The almanac "
                         "clock is OFFSET to this and then ADVANCES with wall time, so the "
                         "predicted sky moves as the file plays (a frozen epoch actively pulls "
                         "trackers off satellites -- measured 2026-07-27). Assumes ~realtime "
                         "replay pacing. 0 = live (use now).")
    ap.add_argument("--dead-reckon", action="store_true",
                    help="seed CODE PHASE from broadcast ephemeris (BRDC) for every visible "
                         "sat the search hasn't detected: predict the absolute transmit-time "
                         "code phase, add the receiver clock solved each cycle from the "
                         "detected sats (measured-vs-predicted circular median -- the "
                         "gnss_deadreckon_check.py bootstrap, ~100 ns), and express it in "
                         "the seed's Doppler currency. The search demotes to bootstrap "
                         "(clock solve), fallback (a detection re-anchors via the normal "
                         "seed loop) and integrity check (per-sat residuals logged); "
                         "dead-reckoning only has to land within the DLL capture range "
                         "(~0.4 chips; validated 0.10 chip rms 2026-07-13). Needs --almanac.")
    ap.add_argument("--dr-constellation", default="G", choices=("G", "E", "C"),
                    help="RINEX constellation letter for this broker's band")
    ap.add_argument("--signal-capability", default=None,
                    help="restrict ALL seeds + hints to GPS PRNs whose satellite block actually "
                         "broadcasts this signal (GPS_L1C_P -> Block III; GPS_L5_Q -> IIF+; "
                         "GPS_L2C_CM -> IIR-M+). The GENERAL form of --dr-min-prn: that numeric "
                         "cutoff only works for BDS-2's contiguous low PRNs; GPS III sats are "
                         "interspersed among IIF/IIR (4/11/14/18/20/21/23/28) so no cutoff can "
                         "express 'L1C only'. Read ONCE at startup from the live Celestrak block "
                         "names (gps_beamtrack.signal_capable_prns); on fetch failure or empty "
                         "result the filter is DISABLED with a warning (phantoms return, but the "
                         "chain lives -- better than killing L1C during a network outage). "
                         "GPS and GLONASS: for R the block marker is the Celestrak 'K' suffix on "
                         "the Uragan number, and GLO_L3OC_* -> K satellites only (the CDMA "
                         "signals are GLONASS-K's; the FDMA L1OF/L2OF are on every satellite, so "
                         "they get no filter). E/C use --tle-name-filter instead.")
    ap.add_argument("--dr-min-prn", type=int, default=None,
                    help="dead-reckon only PRNs >= this: a SIGNAL-CAPABILITY gate, not a "
                         "visibility one. Default 19 for BeiDou (B1C/B2a are BDS-3 ONLY; the "
                         "BDS-2 birds C1-C18 broadcast B1I at 1561 MHz, which is not even "
                         "inside our band), 1 otherwise. The search cannot make this mistake "
                         "-- it never detects a satellite that isn't transmitting -- but "
                         "DEAD RECKONING CAN: it seeds from the model, and the model is happy "
                         "to predict a code phase for a signal that does not exist. The "
                         "tracker then despreads noise at that phase and the cross-correlation "
                         "against real B1C satellites reports 20-60 sigma. Measured 2026-07-14: "
                         "C11/C12/C13 produced 11309 phantom rows (5.5%% of all BeiDou map "
                         "points) at a plausible-looking 25-30 dB-Hz. A model that can invent "
                         "a satellite needs a capability gate the search never needed.")
    ap.add_argument("--dr-repin-s", type=float, default=10.0,
                    help="re-anchor a dead-reckoned (undetected, unlocked) seed from the "
                         "model this often: fresh cp/doppler/rate together (a DR seed's "
                         "doppler is FROZEN between pins -- currency-consistent by "
                         "construction -- so this also bounds the doppler staleness; "
                         "10 s * max MEO rate ~0.6 Hz/s = 6 Hz, under every band's fence)")
    ap.add_argument("--dr-refresh-s", type=float, default=2.0,
                    help="dead-reckon cadence (clock solve + integrity + pin checks)")
    ap.add_argument("--dr-clock-alpha", type=float, default=0.2,
                    help="EMA weight for the solved receiver clock (wrap-aware chips; the "
                         "held value propagates at f_chip*(l-a) between solves)")
    ap.add_argument("--dr-min-sats", type=int, default=2,
                    help="detections needed for a receiver-clock solve (one sat is "
                         "unfalsifiable -- same reasoning as --bias-min-sats)")
    ap.add_argument("--dop-continuous", action="store_true", default=False,
                    help="DESIGN (b): update the seed Doppler every cycle (model-primary "
                         "seeding) and currency-translate cp0 on every update. VALIDATED "
                         "2026-07-19 (A/B replay legs, 17:37 capture): B1C parity-or-better "
                         "(coh duty up on every sat, C37 63->79/100, reacq 2->1) and GPS "
                         "better (RELEASE 9->0, CARRIER REACQ 6->1, coh duty up 4/6 sats). "
                         "History: the 2026-07-14 attempt measured E/C WORSE (E 42.0 -> 34.9 "
                         "dB-Hz) because the tracker's f_ref re-pin was not yet code- and "
                         "phase-continuous -- freezing the seed was double-dutying as f_ref "
                         "stabilization. The 07-14 NCO phase fold (reanchored==2) plus "
                         "max_anchor_age_s 0 completed the primitive; the fence became "
                         "moot (f_ref rate-follows the model, seed steps vanished). "
                         "run_band.sh made this the single-band default; the run_3band "
                         "transition silently dropped it (fleet ran frozen 07-18/19 -- the "
                         "release/escape churn era). Fleet default restored 2026-07-19.\n"
                         "Original rationale: update the seed Doppler EVERY cycle and "
                         "currency-translate cp0 each time, instead of freezing it and taking "
                         "a discrete step at hold_max_dop_hz. The fence was never a safety "
                         "mechanism -- it was a GRANULARITY threshold, and the "
                         "piecewise-constant-currency rule it enforced was a defence against a "
                         "NOISY, search-grid Doppler. Dead reckoning made the Doppler "
                         "model-derived and smooth, and the currency translation makes a "
                         "Doppler update cost exactly nothing (the cp0 shift cancels the "
                         "retroactive term by construction, so even jitter moves the code by "
                         "ZERO). We made the defence obsolete ourselves. Resilience now comes "
                         "from checking whether the model is RIGHT (--dr-max-eph-age-s, "
                         "--dop-max-rate-hz, the integrity residual, a railed carrier trim) "
                         "rather than whether it is MOVING. --no-dop-continuous restores the "
                         "fence.")
    ap.add_argument("--no-dop-continuous", dest="dop_continuous", action="store_false",
                    help="restore the discrete hold_max_dop_hz fence (pre-2026-07-14)")
    ap.add_argument("--dop-max-rate-hz", type=float, default=None,
                    help="SAFETY NET (not a fence): clamp how far the seed Doppler may move in "
                         "ONE cycle. Bounds the damage from a garbage prediction slamming the "
                         "tracker, without imposing discrete steps. A real MEO Doppler moves "
                         "<1 Hz per 0.2 s cycle, so this only ever fires on a bad model.")
    ap.add_argument("--dr-max-eph-age-s", type=float, default=14400.0,
                    help="do not trust the BRDC Doppler for a satellite whose ephemeris toe is "
                         "older than this (4 h). Stale/absent model -> fall back to the "
                         "search-measured Doppler. The fallback is SEAMLESS: switching Doppler "
                         "SOURCE is just another currency translation, not a loss of lock.")
    ap.add_argument("--dr-max-integrity-chips", type=float, default=1.0,
                    help="demote a satellite to search-anchored when its dead-reckon integrity "
                         "residual (measured-vs-predicted code phase, already computed every "
                         "cycle and normally +-0.2 chips) exceeds this. THIS is the resilience "
                         "the fence never provided: it detects a model that is WRONG, which a "
                         "fence on Doppler MOTION cannot.")
    ap.add_argument("--dr-dry-run", action="store_true",
                    help="compute + log the clock solve, integrity residuals and planned "
                         "dead-reckoned seeds WITHOUT injecting any (validation mode)")
    ap.add_argument("--nav-bits-brdc", type=int, default=0,
                    help="CONSTRUCT nav bits from BRDC for satellites that never sync "
                         "(navbit_brdc.BrdcLnavSource). DEFAULT OFF: first live trial "
                         "2026-07-25 collapsed the GPS chain to 0/14 peeling with every PRN on "
                         "`nobits` and every gain reset -- the 30 s tables make the seed POST "
                         "~14.5k numbers and the whole seed push appears to stop landing. The "
                         "bit CONTENT is validated (113820/113820 offline, hold-one-out 100%%); "
                         "it is the transport that needs sizing before this goes back on.")
    ap.add_argument("--nav-bits", type=int, default=1,
                    help="LNAV decode-and-predict from the combiner's nav_obs export "
                         "(bit_export: true), pushed as nav_bits with each seed row -- the "
                         "fused peel's sign source (P7a: ~11 -> >=26-35 dB). Self-gating: "
                         "chains whose combiner exports no nav_obs are untouched. 0 = off.")
    ap.add_argument("--nav-decoder", default="lnav", choices=("lnav", "cnav", "bcnav3"),
                    help="which decoder consumes nav_obs. lnav (default): the LNAV "
                         "decode-and-predict (GPS L1CA, the periodic-subframe future-bit source). "
                         "bcnav3: the BeiDou B2b B-CNAV3 decoder (GF(64) NB-LDPC(162,81) + CRC-24Q, "
                         "the B2b PRIMARY chain's own nav; pure decode + BRDC xcheck, no bit-pred). "
                         "cnav: the CNAV decoder (GPS L2C-CM / L5-I) -- FEC+CRC-24Q, NOT a "
                         "future-bit source (the type schedule is not fixed); it decodes live "
                         "ephemeris/messages and shadow-serves the signs of DECODED spans. Set "
                         "cnav on the L2C broker so its nav_obs (CNAV symbols) go to the right "
                         "decoder instead of churning the LNAV frame-sync forever.")
    ap.add_argument("--cnav-combiner", default=None,
                    help="AUXILIARY combiner polled purely for CNAV nav_obs, in ADDITION to "
                         "--combiner (S4). Exists because a band's CNAV can live on a chain "
                         "that is not this broker's own: at L5 the broker's combiner is the "
                         "Q PILOT (whose nav_obs are deterministic overlay predictions, which "
                         "belong to the LNAV/pilot path), while the CNAV symbols come from the "
                         "derived L5-I DATA sibling. Pointing --nav-decoder at the main "
                         "combiner cannot express that split. Symbols from here go to the CNAV "
                         "decoder regardless of --nav-decoder, and are cross-checked against "
                         "BRDC on the usual 60 s health cadence -- giving a SECOND, independent "
                         "decode of the same message set L2C already decodes, so an ephemeris "
                         "can be verified three ways (L2C vs L5 vs BRDC).")
    ap.add_argument("--inav-combiner", default=None,
                    help="AUXILIARY combiner polled for Galileo E1B I/NAV nav_obs (S5 "
                         "D-component #1), the exact analogue of --cnav-combiner: the GAL "
                         "broker's own --combiner is the E1C PILOT (deterministic overlay "
                         "signs), while the I/NAV DATA symbols come off the derived E1B "
                         "sibling chain. Symbols from here go to the InavPredictor and the "
                         "decoded ephemeris is cross-checked against BRDC on the 60 s health "
                         "cadence -- an independent E1B decode validating the Galileo "
                         "ephemeris (and the codec's ICD conventions) against the almanac.")
    ap.add_argument("--fnav-combiner", default=None,
                    help="AUXILIARY combiner polled for Galileo E5a-I F/NAV nav_obs (S5 "
                         "D-component #2), the --inav-combiner analogue on the L5 band: the "
                         "GAL/E5a broker's own --combiner is the E5a-Q PILOT (deterministic "
                         "CS100 overlay signs), while the F/NAV DATA symbols come off the "
                         "derived E5a-I sibling chain (CS20 secondary + navwipe). Symbols go "
                         "to the FnavPredictor and the decoded ephemeris is cross-checked "
                         "against BRDC on the 60 s health cadence -- an independent E5a decode "
                         "validating the Galileo ephemeris (and galileo_fnav's ICD "
                         "conventions) against the almanac, beside E1B's I/NAV decode.")
    ap.add_argument("--bcnav2-combiner", default=None,
                    help="AUXILIARY combiner polled for BeiDou B2a B-CNAV2 nav_obs (S5 "
                         "D-component #3, the FIRST non-binary FEC), the --fnav-combiner "
                         "analogue on the BDS broker: the BDS broker's own --combiner is the "
                         "B2a-P PILOT (deterministic Weil overlay signs), while the B-CNAV2 "
                         "DATA symbols come off the derived B2a-D sibling chain (CS5 secondary "
                         "+ navwipe). Symbols go to the Bcnav2Predictor (GF(64) NB-LDPC codec) "
                         "and the decoded ephemeris is cross-checked against BRDC on the 60 s "
                         "health cadence -- an independent BDS-3 decode validating the "
                         "ephemeris (and the LDPC + CNAV-eph conventions) against the almanac.")
    ap.add_argument("--bcnav1-combiner", default=None,
                    help="AUXILIARY combiner polled for BeiDou B1C B-CNAV1 nav_obs (S5 "
                         "D-component #4, the LAST), the --bcnav2-combiner analogue on the L1 "
                         "BDS broker: the broker's own --combiner is the B1C-P PILOT, while the "
                         "B-CNAV1 DATA symbols come off the derived B1C-D sibling. Symbols go to "
                         "the Bcnav1Predictor (reusing the GF(64) NB-LDPC codec, different H "
                         "matrices for SF2/SF3) and the decoded ephemeris is cross-checked "
                         "against BRDC -- completing the civil D-component set (GPS+GAL+BDS).")
    ap.add_argument("--cnav2-combiner", default=None,
                    help="AUXILIARY combiner polled for GPS L1C-D CNAV-2 nav_obs, the "
                         "--bcnav1-combiner analogue on the L1C broker: the broker's own --combiner "
                         "is the L1C-P PILOT (deterministic L1CO overlay signs), while the CNAV-2 "
                         "DATA symbols come off the derived L1C-D sibling. Symbols go to the "
                         "Cnav2Predictor (binary-LDPC systematic extract + CRC-24Q, 18 s frame) and "
                         "the decoded ephemeris is cross-checked against BRDC -- the 4th GPS-family "
                         "decode (LNAV / CNAV / CNAV-2).")
    ap.add_argument("--once", action="store_true",
                    help="run a single control-loop iteration and exit (for tests)")
    # -- named signals (task #27 M2); see gnss_broker/signals.py --------------------------
    ap.add_argument("--signal", default=None, metavar="KEY",
                    help="name the chain instead of retyping its twelve constants: "
                         "`--signal gps_l5` sets carrier/chip rate/code length/long-code "
                         "segments+epoch/overlay length/constellation from "
                         "lib/stages/gnss/gnssSignal.hpp, which is what the trackers' "
                         "replica bank is actually built from. Every one of those flags "
                         "fails SILENTLY when mistyped (a wrong overlay length puts the "
                         "seed in a random one of N periods and errors nowhere), which is "
                         "why naming beats typing. An explicit flag that DISAGREES with "
                         "the named signal is a hard error, never a silent override. "
                         "`--signal help` lists them.")
    # -- the refactor gate (task #27 M0); see the _Transcript note at the top of this file --
    ap.add_argument("--transcript-write", default=None, metavar="FILE",
                    help="record every clock read, GET and POST to FILE (JSONL) while running "
                         "normally. Cheap and side-effect-free: safe to leave on in production "
                         "for the minute it takes to capture a golden run.")
    ap.add_argument("--transcript-read", default=None, metavar="FILE",
                    help="replay a recorded transcript instead of talking to the fleet, then "
                         "exit when it is exhausted. No network, no sleeps, deterministic "
                         "clock. The POST stream this produces is the equivalence gate -- see "
                         "scripts/gnss/broker_equiv.py.")
    args = ap.parse_args(argv)
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
    # Integer hop tolerance, derived once from the record geometry. Kept as an int so every
    # comparison downstream is integer arithmetic on the F-engine's own counter.
    dll_hop_window = max(0, int(round(args.dll_hop_window_s * args.hops_per_sec)))
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

    def _p2c_hold(js, key):
        """True once this sat is BOTH masked and established -- see --joint-mask-after."""
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
            js = rx_.joint_receiver(band, CODE_LEN)
            return js if len(js._idx) >= a.joint_min_sats else None
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
    low_hits = {}    # prn -> consecutive low-|A| poll count
    status = {}      # prn -> last combiner get_status record (previous cycle; lock gate below)
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
    dll_trim = {}       # prn -> persistent cp trim (chips) from the E/L delay-lock loop
    dll_last = {}       # prn -> last integrated disc (dedup: one integration per emit)
    last_dets = []      # most recent raw /get_detections, re-served by the publisher so
                        # the viewer has ONE origin for both search and combiner data
    nh_off_hist = []    # pooled (t, predicted - reported): ONE receiver-clock constant.
                        # TIMESTAMPED so the offset can EXPIRE -- see the accumulator.
    nh_last_rh = {}     # prn -> ref_hop already folded in, so a stale detection cannot be
                        # re-counted every cycle and fake a full window of agreement
    nh_offset = [None]  # the calibrated constant, once enough samples agree
    nh_seen = {}        # prn -> (nh, ref_hop) last REPORTED by the search: calibrates the offset
                        # as its own alignment hint, so nothing external has to be trusted
    dll_last_hop = {}   # prn -> last integrated WINDOW (fleet DLL): the exact dedup dll_last
                        # approximates. A changed float disc means "probably a new emit"; a
                        # changed hop means it, and it cannot false-negative on a disc that
                        # happens to repeat.
    cp_translated = set()  # PRNs whose first currency translation has been logged (once each)
    dop_clamped = set()    # PRNs that have tripped the one-cycle Doppler rate limit
    dr_untrusted = {}      # prn -> reason: the model is WRONG for this sat; use the search
    dr_bad = {}            # prn -> consecutive model-health failures (persistence, not a hair trigger)
    cp_escape = {}      # prn -> consecutive track-vs-search cp disagreements (hold referee)
    cp_escape_sign = {} # prn -> last disagreement (sign-consistency: real parks are one-signed)
    cp_err_hist = {}      # prn -> last 9 cp_err samples (median gate: noise cannot sustain it)
    hold_miss = {}      # prn -> consecutive sub-gate status reads while held (blank-poll rides)
    rate_prev_hop = {}  # prn -> last pow_hop used by the carrier loop (continuity gate)
    rate_prev_val = {}  # prn -> last rate residual (slew gate: catches f_ref re-pins)
    rate_unit_hop = [0]  # [emit spacing in hops], LEARNED -- see rate_residuals' continuity gate
    car_trim = {}       # prn -> persistent NCO frequency command (Hz): the shared carrier loop
    car_last = {}       # prn -> last SEEN residual (dedup: one gate-check/integration per emit)
    car_locked = set()  # prns certified coherent since seed: BOOTSTRAP -> TRACK mode latch
    car_fade = {}       # prn -> consecutive TRACK-mode gated emits (--carrier-refade demotion)
    car_step_hist = {}  # prn -> [(t, resid)] recent GATED residuals (--carrier-step-accept)
    car_step_t = {}     # prn -> last step-accept time (rate limit / refute lockout)
    car_verify = {}     # prn -> {prev_trim, emits}: an applied step hypothesis under VERIFY
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
    car_bleed_hist = {}   # prn -> [recent trim values on coherent ungated emits] (convergence)
    car_bleed_log_t = {}  # prn -> last shadow-log walltime (rate limit)
    # ARMED action (--carrier-bleed): when a candidate fires, zero car_trim and flag the tracker to
    # re-adopt the seed (f_ref = dop, phase-continuous) -- folding the frozen offset into f_ref.
    car_repin_pending = {}     # prn -> bleed amount (Hz) to carry as carrier_repin in this seed
    car_bleed_verify = {}      # prn -> {emits, prev_trim, t}: post-bleed EXPLAIN-APPLY-VERIFY
    car_bleed_lock_t = {}      # prn -> earliest walltime to bleed this prn again (anti-churn)
    det_fresh = {}      # prn -> (ref_hop, walltime) of the last NEW detection (alias escape)
    wd_birth = {}       # prn -> when it entered seeds (track-watchdog judgment window)
    wd_coh_t = {}       # prn -> last coherent walltime (track-watchdog's own clock)
    wd_strong_t = {}    # prn -> last time track sig cleared --watchdog-weak-sig (zombie ref)
    wd_weak_n = {}      # prn -> consecutive weak-track fires (exponential backoff; survives
                        # reseeds BY DESIGN -- the backoff exists to remember failed rescues;
                        # cleared only when the sat clears the weak bar)
    _trim_force = {}    # BENCH-ONLY fault injection ("20:-60" = PRN 20 at -60 Hz): applied
    for _spec in os.environ.get("GNSS_TRIM_FORCE", "").split(","):
        # when the PRN is first SEEDED (a startup preload would be swept by the not-in-
        # seeds trim cleanup before the sat ever seeds). Reproduces the alias-capture
        # regime on the replay bench: BOOTSTRAP itself converges a -60 Hz NCO error to
        # the -50 Hz alias (the estimator reads the error mod 1/(2*T_rec)).
        if ":" in _spec:
            _p, _v = _spec.split(":")
            _trim_force[int(_p)] = float(_v)
            _log("TRIM FORCE (bench): PRN %s armed, car_trim %+.1f Hz at first seed"
                 % (_p, float(_v)))
    cp_fit_slope = {}    # prn -> fitted cp slope, chips/s (x f_carrier/f_chip = carrier error)
    dop_rate_fitted = {} # prn -> the fitted rate actually seeded (for the log)
    dop_rate_rejected = {} # prn -> (fitted, model) when the fit disagreed with the model
    dop_hist = {}    # prn -> [(ref_hop, doppler_hz), ...] for the MEASURED doppler-rate fit
    cp_hist = {}     # prn -> [(ref_hop, cp0, dop_det), ...] recent distinct snapshots (slope fit)
    # PERIOD CONTINUITY. The search's sub-period phase is sound, but the OVERLAY PERIOD it
    # reports re-randomises every pass (measured: exact integer-period jumps of +1/+3/-4/+3/
    # +3/-1/-2, with only +-10 chips of sub-period residual). A satellite's code phase evolves
    # deterministically, so the period is pinned by requiring continuity with the previous pass:
    # predict this pass's phase from the last accepted one and take the integer that lands
    # nearest. The margin is not marginal -- over a 270 s revisit with a 1.5 Hz Doppler error the
    # prediction is good to ~3.5 chips against a 5115-chip half-period tolerance, a factor 1500.
    #
    # NOTE what this does and does not give: it makes the period SELF-CONSISTENT, not absolute.
    # A whole sequence can still sit a constant integer off -- one calibration per satellite (or
    # one common one), rather than a fresh coin flip every pass.
    ph_hist = {}     # prn -> (ref_hop, resolved_phase, dop)

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
    clock_bias_ema = None  # smoothed common clock-frequency bias (slow TCXO drift), Hz
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
    clock_bias_cal = None  # startup calibration value, Hz (drift-alarm reference)
    if args.clock_bias_file:
        try:
            with open(args.clock_bias_file) as f:
                # format: "<bias_hz> [n_sats] [unix_ts]" -- extended for --clock-bias-siblings;
                # the extra fields are ignored here (warm-start wants only the value).
                clock_bias_ema = float(f.read().split()[0])
            clock_bias_cal = clock_bias_ema
            _log("clock-freq bias warm-started %+.1f Hz from %s (margins narrow, seeding "
                 "enabled from cycle 1)" % (clock_bias_ema, args.clock_bias_file))
        except Exception:
            pass
    code_bias_cal = code_bias_ema  # l-a calibration reference (None if cold)
    _clk_persist_t = [0.0]         # last clock-bias-file write (10 s rate limit)
    _bias_meas_t = _now()     # last multi-sat bias measurement (stale-rescue clock;
                                   # birth-stamped so warm-start gets a full grace window)
    bias_stale = False             # solved-but-unmeasured for > --bias-stale-s
    bias_available = False         # is ANY usable bias in hand (own or fused)? S2d gate
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
        gpst_now = _decfb.gpst_of_utc(datetime.fromtimestamp(now_w, tz=timezone.utc))
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

    while True:
        # Start of cycle: sample the frozen cycle clock ONCE (see the _Transcript note).
        # Every `_now()` below this returns exactly t0, so the whole pass -- every gate,
        # every age, every EMA -- evaluates at one instant instead of smearing over the
        # cycle's own processing time.
        t0 = _TR.tick()
        t_wall = time.time()   # REAL clock, kept solely for the sleep below
        # 1. collect best-SNR detection per PRN across all detection sources
        best = {}  # prn -> (snr, dop, cp, ref_hop, nh, cp_long, cp_at_ref)
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
                nh_seen[_p] = (_b[4], _b[3])
        for _p in list(nh_seen):
            if _p not in seeds and _p not in best:
                del nh_seen[_p]          # sat set: stop hinting a PRN we no longer track

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
        if args.state_consume and clock_bias_ema is None and _fused_hz is not None:
            clock_bias = _fused_hz
            bias_available = True
            _log_rl("fusrescue",
                    "FUSED-STATE RESCUE: this chain is UNSOLVED; consuming the dongle's "
                    "fused LO %+.1f Hz (%d src: %dc/%dd over %s) until it solves itself"
                    % (_fused_hz, _fus_now["n_src"], _fus_now["n_carrier"],
                       _fus_now["n_code"], ",".join(_fus_now["chains"])),
                    every_s=10.0)
        else:
            clock_bias = clock_bias_ema if clock_bias_ema is not None else 0.0
            bias_available = clock_bias_ema is not None
            if args.state_consume and clock_bias_ema is None and _fus_now is None:
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
                        % (("%+.1f Hz" % clock_bias_ema) if clock_bias_ema is not None
                           else "UNSOLVED"), every_s=30.0)

        # 2. orbit-predicted Doppler + visibility (almanac assist), else plain gate
        pred = {}          # prn -> (doppler_hz, rate_hz_s, elev_deg) [sign-applied]
        # STALE-BIAS RESCUE (--bias-stale-s): a solved bias nobody has measured for minutes
        # is a LIABILITY, not a constant -- if it latched away from truth (mid-walk during
        # the 2026-07-20 GPSDO unlock) the narrow hints it centers are what PREVENT the
        # measurements that would fix it. Widen and re-solve; hold the value for seeding.
        bias_stale = (args.bias_stale_s > 0.0 and clock_bias_ema is not None
                      and t0 - _bias_meas_t > args.bias_stale_s)
        if bias_stale:
            _log_rl("clkstale",
                    "CLOCK BIAS STALE: no multi-sat measurement for %.0f s (holding %+.0f Hz "
                    "for seeding) -- margins WIDE until re-solved"
                    % (t0 - _bias_meas_t, clock_bias), every_s=60.0)
        up = None
        if args.almanac:
            try:
                t_pred = _alm_now()
                if brdc_alm is not None:
                    raw = brdc_predict(brdc_alm, args.lat, args.lon, args.alt,
                                       alm_sys, alm_min_prn, t_pred, args.carrier_hz)
                else:
                    from gps_beamtrack import predict_dopplers
                    raw = predict_dopplers(args.lat, args.lon, args.alt, t_utc=t_pred,
                                           _sats=almanac_sats,
                                           f_carrier_hz=args.carrier_hz)
                # doppler_sign flips to the receiver's observed convention -- apply it to BOTH the
                # Doppler and its rate so the 2nd-order feed-forward ramps the right way. (Range
                # is geometry, no sign; it feeds the CL time-assist propagation delay. The 5th
                # element -- broadcast sat clock -- is BRDC-only; the TLE path has none.)
                pred = {p: (args.doppler_sign * v[0], args.doppler_sign * v[1], v[2], v[3],
                            (v[4] if len(v) > 4 else 0.0))
                        for p, v in raw.items()}
            except Exception as e:
                _log("almanac predict failed: %s" % e)
            up = {p for p, v in pred.items() if v[2] >= args.mask_deg}
            # nh TIME-ASSIST: POST each visible sat's predicted absolute overlay-chip index to the
            # combiner. period = one primary code period (= one overlay chip); the overlay counter
            # runs on the SATELLITE's clock, so the predicted chip at transmit is
            # round((gpst(now) - range/c + clk_sv)/period) mod overlay_len -- the convention
            # proven to 0.01 chip offline (c31_convention.py). clk_sv is the 5th pred element
            # (BRDC only; 0.0 on the TLE fallback, a <=~0.1-chip omission the consensus absorbs).
            # NO absolute-convention care (the combiner self-calibrates the constant from its
            # confidently-locked sats). Differential + slowly-varying, so the exact reference
            # instant is immaterial to <<1 chip.
            if args.nh_assist and pred:
                try:
                    import gnss_ephemeris as _nh_eph
                    period = args.code_length / args.chip_rate_hz
                    t_ref = args.almanac_epoch or _now()
                    hints = [{"prn": int(p),
                              "nh": int(round((_nh_eph.gpst_of_utc(t_ref)
                                               - v[3] / _nh_eph.C_LIGHT
                                               + (v[4] if len(v) > 4 else 0.0)) / period))
                                    % args.nh_overlay_len}
                             for p, v in pred.items() if v[2] >= args.mask_deg
                             and (_capable is None or p in _capable)]
                    if hints:
                        _post("%s/set_nh_hint" % combiner, hints)
                except Exception as e:
                    _log("nh-assist POST failed: %s" % e)
            # Common clock-frequency bias = median(measured - predicted) over detected
            # sats. A tight residual spread confirms the sign convention; a wild spread
            # (resid ~ -2x predicted) means flip --doppler-sign.
            # FRESH detections only. The search stage's REST table keeps serving a detection
            # long after it was made, and best[] holds it; the prediction meanwhile ADVANCES
            # at the satellite's dop_rate, so a stale detection's (meas - pred) grows linearly
            # with its age -- stale-age x dop_rate wearing a clock-bias costume. Measured on
            # the L5 replay bench 2026-07-27: one ~90 s-stale PRN20 detection walked the
            # "solved" bias +4 -> +68 Hz at exactly dop_rate, the seeds chased it, and the
            # tracker was dragged off a 55-sigma satellite. Live mostly hides this because
            # strong sats re-detect every few seconds -- but any sparse-detection stretch
            # (fades, one-sat horizons) exposes the same feedback. det_fresh already tracks
            # exactly this (its comment even computes the hazard: "at 0.4 Hz/s slew, 100 s of
            # staleness fakes a 40 Hz mismatch") -- the bias solve just never consulted it.
            # SNR gate as well as freshness (2026-08-02). This median is a COMMON-MODE
            # estimate: its uncertainty is (per-sat Doppler error)/sqrt(N), so one satellite
            # whose Doppler is noise does real damage when N is 2. The acquire's own
            # interpolation error is ~1.2 Hz rms (measured sawtooth), which at N=2 predicts
            # ~0.8 Hz -- but the raw estimate scatters 10.5 Hz, because every detection above
            # acquire_snr enters, and ~half of CHORD's sit between the threshold (30) and the
            # noise ceiling (19), where the reported Doppler is near-random inside the hint
            # window. Gated, this is what decides whether predicted Doppler (BRDC 0.1 Hz +
            # this) beats the detection's own (~2 Hz): at N=8 it should reach ~0.4 Hz.
            # Default 0 keeps every point -- the prototype's behaviour, right for detections
            # that sit far above threshold.
            resid = [best[p][1] - pred[p][0] for p in best
                     if p in pred and t0 - det_fresh.get(p, (None, 0.0))[1] < args.bias_det_fresh_s
                     and best[p][0] >= args.bias_min_snr]
            # BAND-SHARED bias fusion (--clock-bias-siblings) is read BEFORE the min-sats
            # gate, and the gate counts LOCAL + SIBLING sats.
            #
            # ⚠️ IT USED TO LIVE INSIDE THE GATE, which made the rescue unreachable by
            # exactly the chains it was written for. The LO is a property of the BAND (one
            # airspy, one LO): every chain on it is measuring ONE physical number, so the
            # falsifiability the gate wants is satisfied by the band's detections in
            # AGGREGATE, not by each constellation independently.
            # Measured 2026-07-27 (power-outage cold start): L5 GPS could measure exactly
            # one satellite, fell short of --bias-min-sats 2, and therefore never entered
            # the block that would have read its siblings -- while L5 GAL and L5 BDS sat in
            # those very files with 13-16 sats each and the correct answer (+32.1 / +32.9 Hz;
            # GPS itself settled at +24..+34 once it finally solved). 603 cycles UNSOLVED,
            # search_snr == 0.0, 2h45m of nothing, ended only when a second GPS satellite
            # physically rose high enough. The band knew the answer the whole time.
            n_sib = 0
            _sib_bw = 0.0   # sum(bias * n) over fresh siblings
            _sib_w = 0.0    # sum(n)
            if args.clock_bias_siblings:
                for _sp in args.clock_bias_siblings:
                    try:
                        _parts = open(_sp).read().split()
                        _b = float(_parts[0])
                        _n = int(_parts[1]) if len(_parts) > 1 else 1
                        _ts = float(_parts[2]) if len(_parts) > 2 else 0.0
                    except Exception:
                        continue
                    # Freshness still required: a stale sibling is a different epoch's LO.
                    if t0 - _ts < 60.0 and _n >= 1:
                        _sib_bw += _b * _n
                        _sib_w += _n
                        n_sib += _n
            # Two INDEPENDENT ways to clear the bar, never a pooled sum:
            #   * this chain has >= bias_min_sats of its own -> trust local, blend siblings in
            #   * the BAND (siblings) has >= bias_min_sats    -> use the band consensus ALONE
            # Summing the two would let a single untrusted local residual into the average --
            # a bad -450 Hz detection dragged the fused answer 32.5 -> 15.3 Hz in the unit
            # test, which is exactly the garden path --bias-min-sats exists to prevent. A lone
            # local residual stays untrusted; it just no longer BLOCKS the band's answer.
            _local_ok = len(resid) >= args.bias_min_sats
            _sib_ok = n_sib >= args.bias_min_sats
            if _local_ok or _sib_ok:
                # The per-cycle median is quantized to the 500 Hz search grid and jumps
                # hundreds of Hz as the detected-sat set flickers; the TRUE bias is a slow
                # TCXO drift. EMA-smooth it (sub-grid dither across sats/cycles averages
                # out the quantization) so every sat's seed Doppler is stable -- a jittery
                # common bias was wrecking coherent integration (residual carrier +-260 Hz).
                # GATED on --bias-min-sats: one sat's residual is unfalsifiable and a bad
                # one narrows the search into a self-locking deadlock (see the arg help);
                # below the gate the EMA holds its last multi-sat value (or stays unsolved
                # -> margins stay WIDE, which is exactly what lets more sats in).
                # One LO, one number: fuse this chain's median with the siblings' persisted
                # estimates, sat-count weighted. With ZERO local sats the band consensus
                # stands alone -- that is not a guess, it is the same physical LO measured
                # by ~27 satellites on the neighbouring chains.
                if _local_ok:
                    raw_bias = statistics.median(resid)
                    if n_sib:
                        _w = float(len(resid))
                        raw_bias = (raw_bias * _w + _sib_bw) / (_w + _sib_w)
                else:
                    # Local count below the bar: the band's answer stands ALONE. The local
                    # residual is deliberately discarded, not down-weighted.
                    raw_bias = _sib_bw / _sib_w
                if clock_bias_ema is None or bias_stale:
                    # First solve, or stale-rescue re-solve: SNAP to the fresh median. An
                    # EMA crawl (a=0.05) from a mid-walk latch kHz off truth would spend
                    # minutes converging through exactly the hint region it just vacated.
                    if bias_stale:
                        _log("CLOCK BIAS RE-SOLVED %+.1f Hz after %.0f s stale (held %+.1f)"
                             % (raw_bias, t0 - _bias_meas_t, clock_bias_ema))
                        if (clock_bias_cal is not None
                                and abs(raw_bias - clock_bias_cal) > args.clock_bias_alarm_hz):
                            _log("CLOCK BIAS RECALIBRATED %+.1f -> %+.1f Hz -- hardware "
                                 "news (GPSDO re-settled?); new warm-start reference"
                                 % (clock_bias_cal, raw_bias))
                        clock_bias_cal = raw_bias
                        bias_stale = False
                    clock_bias_ema = raw_bias
                else:
                    clock_bias_ema += args.bias_alpha * (raw_bias - clock_bias_ema)
                _bias_meas_t = t0
                clock_bias = clock_bias_ema
                # CONTRIBUTE (task #27 M3). Receiver scope -- one reference, one frequency
                # error -- so every co-hosted chain can have this without solving it. A
                # chain that HAS its own estimate never reads back, which is why publishing
                # here cannot change single-chain behaviour.
                rx.contribute_carrier_bias(chain_id, clock_bias_ema, len(resid), t0)
                # `alarming` (TIGHT bar) gates the PERSIST only -- conservative: never write a
                # walking bias to the cal file (the 2026-07-20 GPSDO-walk-poisoning guard).
                alarming = (clock_bias_cal is not None
                            and abs(clock_bias_ema - clock_bias_cal) > args.clock_bias_alarm_hz)
                # rec D persist (10 s rate limit). The file is a CALIBRATION, not an EMA
                # mirror -- NEVER overwrite it while the live bias is in alarm (2026-07-20:
                # the GPSDO free-run walk was faithfully persisted all the way to -2 ppm
                # and poisoned the next warm-start kHz off truth).
                if (args.clock_bias_file and not alarming
                        and t0 - _clk_persist_t[0] > 10.0):
                    _clk_persist_t[0] = t0
                    # COLD CAL STAMP -- wait for a TRUSTWORTHY sat count. A cal stamped from a
                    # noisy 1-2 sat first solve lands far from the settled bias and then cries
                    # wolf forever (2026-07-21: L5 cold-solved +75 with 2 sats, settled to -5
                    # -> a phantom 80 Hz "drift" alarmed all morning). No stamp yet = no alarm
                    # yet, which is correct: a chain with <3 sats has no trustworthy reference.
                    if clock_bias_cal is None and len(resid) >= max(args.bias_min_sats + 1, 3):
                        clock_bias_cal = clock_bias_ema
                        _log("clock-freq bias calibrated %+.1f Hz (%d sats, cold start) -> %s"
                             % (clock_bias_ema, len(resid), args.clock_bias_file))
                    if clock_bias_cal is not None:
                        try:
                            with open(args.clock_bias_file, "w") as f:
                                # value + sat count + timestamp: siblings weight by count
                                # and ignore stale entries (--clock-bias-siblings).
                                f.write("%.2f %d %.2f\n" % (clock_bias_ema, len(resid), t0))
                        except Exception:
                            pass
                # ALARM LOG on a SAT-SCALED bar: the median-of-residuals noise is ~1/sqrt(n),
                # so the fixed bar (tuned for strong chains) cried wolf on the weak-sat chains
                # (L5/E5a/B2a ~730 false alarms/night 2026-07-20 while the strong chains were
                # silent -- and a fleet-wide GPSDO event hits ALL chains, so a weak chain's
                # solo alarm is almost always noise). Widen it below 5 sats; a real event is
                # large + sustained so it still trips. Persist above keeps the TIGHT bar.
                if clock_bias_cal is not None:
                    _abar = args.clock_bias_alarm_hz * max(1.0, (5.0 / max(len(resid), 1)) ** 0.5)
                    if abs(clock_bias_ema - clock_bias_cal) > _abar:
                        _log_rl("clkalarm",
                                "CLOCK DRIFT ALARM: carrier bias %+.1f Hz vs calibration %+.1f "
                                "(|d| > %.0f Hz, %d sats) -- GPSDO unlock / thermal event? INVESTIGATE"
                                % (clock_bias_ema, clock_bias_cal, _abar, len(resid)),
                                every_s=60.0)
            # S2 OBSERVER: publish the carrier-side estimate. OUTSIDE the solve gate on
            # purpose -- an unsolved chain is exactly the case the fused state exists to
            # rescue, so it has to be visible, and `null` is how that is said (never 0).
            # `raw_hz` is this chain's OWN median, computed here rather than reusing
            # `raw_bias` above, which is already sibling-FUSED. Scoring cross-chain
            # agreement on a fused number measures the fusion, not the estimator.
            if state_w is not None:
                try:
                    _raw_local = statistics.median(resid) if resid else None
                    state_w.observe(
                        "carrier",
                        hz=clock_bias_ema,
                        raw_hz=_raw_local,
                        mad_hz=receiver_state.mad(resid, _raw_local),
                        n=len(resid),
                        sib_hz=(_sib_bw / _sib_w) if _sib_w else None,
                        sib_n=n_sib,
                        cal_hz=clock_bias_cal,
                        stale=bool(bias_stale),
                        meas_age_s=round(t0 - _bias_meas_t, 2))
                except Exception:
                    pass
            for p in sorted(best):
                if p in pred:
                    _log_rl("meas-%d" % p,
                            "PRN %d: meas %+.0f  pred %+.0f  resid %+.0f Hz (elev %.0f)"
                            % (p, best[p][1], pred[p][0], best[p][1] - pred[p][0], pred[p][2]))
            if _local_ok or _sib_ok:
                _log_rl("clkbias",
                        "clock-freq bias %+.0f Hz (raw %+.0f, %d sats%s + %d sib, EMA a=%.2f) "
                        "-> seeding predicted Doppler"
                        % (clock_bias, raw_bias, len(resid),
                           "" if _local_ok else " LOCAL-UNTRUSTED(band consensus)",
                           n_sib, args.bias_alpha))
            else:
                # Say WHY, with both counts -- "1 sat" alone sent this investigation looking
                # at the clock, the sky and the front end before anyone asked whether the
                # BAND had already solved it (2026-07-27).
                _log_rl("clkbias",
                        "clock-freq bias %s (%d local + %d sibling sats < --bias-min-sats "
                        "%d: residual not trusted)"
                        % ("held %+.0f Hz" % clock_bias if clock_bias_ema is not None
                           else "UNSOLVED (wide margins)", len(resid), n_sib,
                           args.bias_min_sats))
        elif gating:
            up = visible_prns(args.lat, args.lon, args.alt, args.mask_deg, 0.0)

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
        if (args.narrow_search and args.almanac and pred) or (_xb_pred and args.xband_seed):
            margin = (args.search_margin_hz
                      if clock_bias_ema is not None and not bias_stale
                      else args.search_margin_wide_hz)
            hints = [dict(prn=p, doppler_hz=pred[p][0] + clock_bias, margin_hz=margin)
                     for p in sorted(pred) if (up is None or p in up)
                     and (_capable is None or p in _capable)] if (args.almanac and pred) else []
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
                    if up is not None and _p not in up:
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
            if args.nh_hint and pred and utc0_sample0:
                try:
                    import gnss_ephemeris as _nh2
                    _per = args.code_length / args.chip_rate_hz
                    def _pred_nh(_p, _t):
                        _v = pred[_p]
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
                    for _p, (_nh, _rh) in nh_seen.items():
                        if _p not in pred or nh_last_rh.get(_p) == _rh:
                            continue
                        nh_last_rh[_p] = _rh
                        _t = utc0_sample0 + _rh / args.hops_per_sec
                        nh_off_hist.append((t0, (_pred_nh(_p, _t) - _nh) % args.nh_overlay_len))
                    del nh_off_hist[:-64]
                    _fresh = [o for (_ts, o) in nh_off_hist
                              if t0 - _ts <= args.nh_hint_max_age_s]
                    if len(_fresh) < args.nh_hint_min_samples and nh_offset[0] is not None:
                        _log("nh hint EXPIRED: %d sample(s) inside %.0f s (need %d) -- "
                             "dropping the offset so the search widens instead of staying "
                             "narrowed on a stale one"
                             % (len(_fresh), args.nh_hint_max_age_s, args.nh_hint_min_samples))
                        nh_offset[0] = None
                    # (b) circular median: the offsets cluster, so rotate to the mode before
                    # taking it, or a cluster straddling the 0/20 wrap averages to nonsense.
                    if len(_fresh) >= args.nh_hint_min_samples:
                        _mode = max(set(_fresh), key=_fresh.count)
                        _rot = [((o - _mode + args.nh_overlay_len // 2) % args.nh_overlay_len)
                                - args.nh_overlay_len // 2 for o in _fresh]
                        _rot.sort()
                        nh_offset[0] = (_mode + _rot[len(_rot) // 2]) % args.nh_overlay_len
                    # (c) hint EVERY visible sat, detected or not, at a hop the stage can
                    # propagate over a few seconds rather than a minute
                    if nh_offset[0] is not None:
                        _rh_now = int(round((t0 - utc0_sample0) * args.hops_per_sec))
                        nh_hints = [dict(prn=int(_p),
                                         nh=(_pred_nh(_p, t0) - nh_offset[0]) % args.nh_overlay_len,
                                         ref_hop=_rh_now)
                                    for _p in pred if pred[_p][2] >= args.mask_deg]
                        _log_rl("nhhint", "nh hint: offset %d (%d samples) -> %d sat(s), span %d"
                                % (nh_offset[0], len(_fresh), len(nh_hints),
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
                       ("bias solved" if clock_bias_ema is not None and not bias_stale
                        else "bias STALE, wide re-solve" if clock_bias_ema is not None
                        else "pre-solve wide"),
                       pushed, len(detectors)))

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
            if (args.det_alias_fold and args.almanac and prn in pred
                    and clock_bias_ema is not None and not bias_stale):
                _aref = pred[prn][0] + clock_bias
                _k = round((dop - _aref) / Q_ALIAS_HZ)
                if _k != 0 and abs(dop - _aref) < 3.5 * Q_ALIAS_HZ:
                    _log_rl("afold-%d" % prn,
                            "ALIAS BIN PRN %d: det dop %+.1f = model %+.1f %+d bin(s) of "
                            "%.0f Hz (census only; cp round-trip is exact)"
                            % (prn, dop, _aref, _k, Q_ALIAS_HZ),
                            every_s=30.0)
            v_dr = dr_pd.get((args.dr_constellation, prn)) if dr_pd else None
            if up is not None and prn not in up:
                # accept the detection anyway if BRDC says it's up: the TLE up-set
                # mismaps some BDS birds (PRN 39: TLE el<5 vs BRDC el 10)
                if v_dr is None or v_dr["el"] < args.mask_deg:
                    continue
            _dop_src = "pred" if (args.almanac and prn in pred) else "DET(grid)"
            seed_dop = (pred[prn][0] + clock_bias) if (args.almanac and prn in pred) else dop
            # Dead-reckon armed: prefer the BRDC doppler for EVERY seed -- the same model
            # that owns the undetected sats. Mixing sources stepped the seed doppler by
            # the TLE-vs-BRDC error at every DR<->search handoff (~25 Hz on a stale TLE
            # = the whole E1 hold fence; observed E32 RELEASE ddop -25, 2026-07-13).
            if v_dr is not None and prn not in dr_untrusted:
                _dop_src = "dr"
                seed_dop = (args.doppler_sign * (-v_dr["range_rate_mps"] / C_LIGHT
                                                 * args.carrier_hz) + clock_bias)
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
            if _prev_sd is None and args.almanac and not bias_available:
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
                        ("%.1f" % (pred[prn][0])) if (args.almanac and prn in pred) else "n/a",
                        clock_bias, car_trim.get(prn, 0.0)))
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
            dh_ = dop_hist.get(prn, [])
            if dh_ and (ref_hop - dh_[-1][0]) > MAX_GAP_HOPS:
                dh_ = []
            if not dh_ or ref_hop != dh_[-1][0]:
                dh_.append((ref_hop, dop))
                dh_ = dh_[-HIST_LEN:]
            dop_hist[prn] = dh_

            h = cp_hist.get(prn, [])
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
            cp_hist[prn] = h

            # The bare-detection cp is in the DETECTION's Doppler currency; the tracker will
            # despread at seed_dop. Convert (else a sat first acquired at t_abs inherits a
            # t_abs*f_chip*(dop-seed_dop)/f_c offset -- chips off-peak for any mid-run
            # acquisition, before tracking even starts).
            ((_, cp_seed_cur),) = cp_to_seed_currency([(ref_hop, cp, dop)], seed_dop)
            seed = {"doppler_hz": seed_dop, "code_phase_chips": cp_seed_cur,
                    "code_phase_rate": 0.0, "ref_hop": ref_hop}
            # 2nd-order carrier feed-forward: hand the tracker the almanac Doppler RATE (Hz/s, sign-
            # applied like doppler_hz); the tracker integrates it in its NCO (never a replica
            # retune -- that walks the absolutely-anchored code/carrier off-peak) so the deep-
            # integration residual stays flat even at zenith (max Doppler acceleration).
            v2_dr = dr_pd2.get((args.dr_constellation, prn)) if dr_pd2 else None
            if v_dr is not None and v2_dr is not None:
                # BRDC doppler rate (range-rate differencing over the 4 s epoch pair),
                # same source as the doppler above
                seed["doppler_rate_hz_s"] = (args.doppler_sign
                                             * (-(v2_dr["range_rate_mps"]
                                                  - v_dr["range_rate_mps"]) / 4.0)
                                             / C_LIGHT * args.carrier_hz)
            elif args.almanac and prn in pred:
                seed["doppler_rate_hz_s"] = pred[prn][1]
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
            _dr = fit_dop_rate(dop_hist.get(prn, []), args.hops_per_sec,
                               args.dop_rate_min_pts, args.dop_rate_min_span_s,
                               args.dop_rate_max)
            if (_dr is not None and _model_dr is not None and args.dop_rate_model_tol > 0.0
                    and abs(_dr - _model_dr) > args.dop_rate_model_tol):
                # The two disagree by more than the model's own accuracy: trust the MODEL, which
                # comes from an orbit rather than from detection noise, and say so.
                dop_rate_rejected[prn] = (_dr, _model_dr)
            elif _dr is not None:
                seed["doppler_rate_hz_s"] = _dr
                dop_rate_fitted[prn] = _dr
            elif args.force_doppler_rate is not None:
                # Replay-bench override: a recorded capture's sky is at another epoch (no almanac),
                # so inject a known rate into every seed to exercise the NCO feed-forward offline.
                seed["doppler_rate_hz_s"] = args.force_doppler_rate
            fit = fit_cp_rate(
                cp_to_seed_currency(h, seed_dop,
                                    float(seed.get("doppler_rate_hz_s", 0.0) or 0.0)),
                CODE_LEN)
            if fit is not None:
                rate, h0, cp_ref = fit
                seed["code_phase_rate"] = rate
                seed["ref_hop"] = h0
                seed["code_phase_chips"] = cp_ref
                fitted.add(prn)
                cp_fit_slope[prn] = rate * args.hops_per_sec   # chips/s, for CARRIER-FROM-CODE
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
                seed["code_phase_chips"] = ((cp_long + args.nh_period_offset * CODE_LEN)
                                            % (LC_SEG * CODE_LEN))
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
                    prev = ph_hist.get(prn)
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
                    if snr >= args.period_check_snr or prn not in ph_hist:
                        ph_hist[prn] = (ref_hop, ph, dop)
                    # --nh-period-offset: applied HERE, after the continuity check has had its
                    # say, and to the phase rather than the argument -- propagate_seed prefers
                    # phase_ref_chips whenever it is >= 0, so offsetting only code_phase_chips
                    # would change nothing the tracker ever reads. ph_hist keeps the UNSHIFTED
                    # phase so the continuity check still compares like with like.
                    ph = (ph + args.nh_period_offset * CODE_LEN) % LLc
                    seed["code_phase_at_ref_chips"] = ph
            elif det_nh >= 0 and LC_SEG > 1:
                seed["code_phase_chips"] = ((seed["code_phase_chips"] % CODE_LEN)
                                            + (det_nh % LC_SEG) * CODE_LEN) % (LC_SEG * CODE_LEN)
                cl_report.append("PRN %d nh=%d (measured)" % (prn, det_nh))
            elif args.cl_assist and utc0_sample0 and args.almanac and prn in pred:
                tau = pred[prn][3] / C_LIGHT
                cl_chips = (((utc0_sample0 - tau + args.cl_time_adjust) % LC_EPOCH)
                            * args.chip_rate_hz)
                cp_cm = seed["code_phase_chips"]
                k = int(round((cl_chips - cp_cm) / CODE_LEN))
                fine_ms = (cl_chips - cp_cm - k * CODE_LEN) / args.chip_rate_hz * 1e3
                seed["code_phase_chips"] = (cp_cm + (k % LC_SEG) * CODE_LEN) % (LC_SEG * CODE_LEN)
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
            if (prev is not None and prn in cp_held
                    and all(k in seed for k in ("code_phase_chips", "code_phase_rate",
                                                "ref_hop"))):
                h_now = seed["ref_hop"]
                cp_prev = (prev["code_phase_chips"]
                           + prev["code_phase_rate"] * (h_now - prev["ref_hop"]))
                # The tracker's commanded cp also carries the QUADRATIC doppler-rate code
                # feed-forward from ref_hop (0.5*sgn*dop_rate*f_chip/f_c*dt^2 -- chips at
                # hold ages of minutes); model it or long-held sats false-escape.
                dt_anchor = (h_now - prev["ref_hop"]) / args.hops_per_sec
                cp_prev += (0.5 * args.code_doppler_sign
                            * float(prev.get("doppler_rate_hz_s", 0.0) or 0.0)
                            * args.chip_rate_hz / args.carrier_hz * dt_anchor * dt_anchor)
                # held cp (prev currency) -> the fresh seed's doppler currency
                t_abs = h_now / args.hops_per_sec
                cp_prev += (t_abs * args.chip_rate_hz * args.code_doppler_sign
                            * (prev["doppler_hz"] - seed["doppler_hz"]) / args.carrier_hz)
                cp_err = ((seed["code_phase_chips"] - cp_prev - dll_trim.get(prn, 0.0)
                           + CODE_LEN / 2.0) % CODE_LEN) - CODE_LEN / 2.0
                # MEDIAN GATE (2026-07-19): the per-detection cp noise is 0.03-0.5 chips
                # (per-sat conditions -- multipath/BOC refine; measured same-instrument at
                # t_abs 100 s AND 27000 s, i.e. FLAT in run age: the earlier 'growth law'
                # was the logged cp_ref coordinate wobbling with dop_seed x t_abs, which
                # the currency translation above cancels in cp_err by construction). The
                # 5-consecutive-sign rule alone still lets a noisy-conditions sat sustain
                # a false accusation; a 9-sample median cannot be dragged over the bar by
                # single-point noise, only by a persistent physical walk.
                cp_err_hist.setdefault(prn, []).append(cp_err)
                del cp_err_hist[prn][:-9]
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
            cp_err_med_ok = (cp_err is not None and len(cp_err_hist.get(prn, [])) >= 5
                             and abs(statistics.median(cp_err_hist[prn]))
                             > args.hold_max_cp_err)
            if (cp_err is not None and abs(cp_err) > args.hold_max_cp_err
                    and cp_err_med_ok and fit_trusted and not amp_veto
                    and not integ_veto):
                n_prev = cp_escape.get(prn, 0)
                same_sign = (n_prev == 0) or (cp_err * cp_escape_sign.get(prn, 0.0) > 0)
                cp_escape[prn] = n_prev + 1 if same_sign else 1
                cp_escape_sign[prn] = cp_err
            else:
                cp_escape[prn] = 0
            if cp_escape.get(prn, 0) >= 5:
                _log("ESCAPE PRN %d: track %+.2f chips off the search fit (5 consecutive,"
                     " sign-consistent) -> release hold + DLL trim, re-anchor on the fit"
                     % (prn, cp_err))
                cp_escape[prn] = 0
                cp_err_hist.pop(prn, None)
                dll_trim.pop(prn, None)
                dll_last.pop(prn, None)
                cp_held.discard(prn)
                hold_miss.pop(prn, None)
                # The re-anchor refreshes the seed doppler next cycle = an NCO f_ref step
                # the TRACK-mode trim was not built for (same latch as the hold release,
                # and it bypasses that branch because cp_held is discarded HERE): demote
                # to BOOTSTRAP so the carrier re-pulls instead of parking off-frequency.
                if prn in car_locked:
                    car_locked.discard(prn)
                    car_fade.pop(prn, None)
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
                         or (prn in cp_held and hold_miss.get(prn, 0) < 3))):
                # PERSISTENT-loss release (2026-07-12 evening): a single blank/stale status
                # read (sig 0.0 -- a poll racing the emit, a slow combiner cycle) used to
                # release the hold instantly: 562 of 736 releases in 2.7 h fired at
                # amp_snr < 8, mostly 0.0, and every release re-fit the seed (dop jump) and
                # paid the ~0.9 Hz x ~5 s carrier re-anchor transient -- the 2 s-median
                # churn behind the GPS coherence wobble (settled sats measure 0.06 Hz).
                # A held sat now rides through up to 3 consecutive sub-gate reads; doppler
                # STALENESS still releases immediately (real currency decoherence).
                if sig_of_last(status.get(prn)) >= args.hold_snr:
                    hold_miss[prn] = 0
                else:
                    hold_miss[prn] = hold_miss.get(prn, 0) + 1
                ddop = seed["doppler_hz"] - prev["doppler_hz"]
                # SAFETY NET (design (b)): bound a single cycle's Doppler move. A real MEO
                # Doppler moves <1 Hz per 0.2 s cycle; this only fires on a bad model, and it
                # bounds the damage rather than forbidding motion.
                if abs(ddop) > args.dop_max_rate_hz:
                    if prn not in dop_clamped:
                        dop_clamped.add(prn)
                        _log("DOP-CLAMP PRN %d: model wanted %+.1f Hz in one cycle (max %.1f)"
                             " -- clamping. A real MEO moves <1 Hz/cycle: SUSPECT THE MODEL."
                             % (prn, ddop, args.dop_max_rate_hz))
                    ddop = math.copysign(args.dop_max_rate_hz, ddop)
                    seed["doppler_hz"] = prev["doppler_hz"] + ddop
                # DESIGN (b): translate EVERY cycle (no fence). The freeze branch survives only
                # for --no-dop-continuous, and for the zero-motion case where it is a no-op.
                if (not args.dop_continuous and abs(ddop) <= args.hold_max_dop_hz) or ddop == 0.0:
                    # Currency frozen: the whole tuple rides unchanged.
                    seed["doppler_hz"] = prev["doppler_hz"]
                    seed["code_phase_chips"] = prev["code_phase_chips"]
                    seed["code_phase_rate"] = prev["code_phase_rate"]
                    seed["ref_hop"] = prev["ref_hop"]
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
                    seed["code_phase_chips"] = (
                        prev["code_phase_chips"]
                        - t_now * args.chip_rate_hz * args.code_doppler_sign
                          * ddop / args.carrier_hz) % CODE_LEN
                    seed["code_phase_rate"] = prev["code_phase_rate"]
                    seed["ref_hop"] = prev["ref_hop"]
                    # seed["doppler_hz"] keeps its NEW value -- that is the point.
                    if prn not in cp_translated:
                        cp_translated.add(prn)
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
                    if abs(ddop_rel) > 1.0 and prn in car_locked:
                        car_locked.discard(prn)
                        car_fade.pop(prn, None)
                        _log("CARRIER REACQ PRN %d: hold released with dop step %+.1f Hz "
                             "-> BOOTSTRAP re-pull" % (prn, ddop_rel))
                cp_held.discard(prn)
                hold_miss.pop(prn, None)
            seeds[prn] = seed
            low_hits[prn] = 0

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
        except Exception as e:
            status = {}
            _log("get_status failed: %s" % e)
        # P7a nav-bit predictor: fold in this cycle's bit observations (rows carry nav_obs
        # only when the combiner runs bit_export, so non-GPS chains skip this for free).
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
                    and brdc_alm is not None and pred):
                if navbrdc is None:
                    from navbit_brdc import BrdcLnavSource
                    navbrdc = BrdcLnavSource(log=_log)
                    _log("constructed-bit source armed (BRDC LNAV, un-synced PRNs)")
                try:
                    navbrdc.update(brdc_alm["eph"], pred, navbits)
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
        # Lock metric: the detection SIGNIFICANCE (sigma above noise) -- the deep nav-wiped SNR when
        # available, else the noise-debiased incoherent SNR -- not the raw |A|. The incoherent |A| is
        # biased by the noise floor (~the floor for weak sats), so judging "still locked" by |A| >
        # drop_amplitude let phantoms coast forever (|A| never falls below the floor). sig ~1 = noise,
        # >>1 = a real lock. Falls back to |A| only if the combiner reports no significance at all.
        def sig_of(r):
            # deep counts only when floor-cleared (coherence_s > 0): a floored deep (~7)
            # otherwise keeps phantom coasts alive forever, exactly like raw |A| did.
            amp = float(r.get("amp_snr", 0) or 0)
            if float(r.get("coherence_s", 0) or 0) > 0.0:
                return max(amp, float(r.get("deep_snr", 0) or 0))
            return amp
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
        if args.noise_probes > 0 and args.almanac and pred:
            deep_low = sorted((p for p, v in pred.items() if v[2] < -15.0),
                              key=lambda p: pred[p][2])[:args.noise_probes]
            probe_set = set(deep_low)
            for p in deep_low:
                if p not in seeds:
                    _log("noise probe PRN %d seeded (elev %.0f)" % (p, pred[p][2]))
                seeds[p] = {"doppler_hz": pred[p][0] + clock_bias,
                            "code_phase_chips": 0.0,
                            "code_phase_rate": cp_rate_from_code_bias(
                                pred[p][0], code_bias_ema or 0.0, args.hops_per_sec,
                                args.chip_rate_hz, args.carrier_hz),
                            "ref_hop": 0,
                            "doppler_rate_hz_s": pred[p][1]}
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
        if args.watchdog_s > 0.0:
            for prn in list(seeds):
                if prn in probe_set:
                    continue
                wd_birth.setdefault(prn, t0)
                _r = status.get(prn) or {}
                if (_r.get("coherence_s") or 0.0) > 0.0:
                    wd_coh_t[prn] = t0
                else:
                    wd_coh_t.setdefault(prn, t0)
                _fr = det_fresh.get(prn)
                # TRIM-RAIL RESCUE (2026-07-20): a trim parked at the +-carrier-max-hz rail
                # is pathology by construction -- converged trims are a few Hz around the
                # chain's common LO offset, and a railed loop plus the tracker fence can
                # self-sustain an NCO alias (E1: 4 ms records = 125 Hz ambiguity; the GPSDO
                # walk railed every trim at +100 and the fleet sat incoherent for 4 h while
                # the det bar (100) hid them from this watchdog at E1's det snr ~45). The
                # rail IS the evidence, so a railed sat is judged at the ordinary presence
                # bar (2x acquire) instead of the strong det bar.
                _railed = (args.carrier_max_hz > 0.0
                           and abs(car_trim.get(prn, 0.0)) >= 0.95 * args.carrier_max_hz)
                _det_bar = (2.0 * args.acquire_snr if _railed else args.watchdog_det_snr)
                _reseed = None
                if (t0 - wd_birth[prn] > args.watchdog_s
                        and t0 - wd_coh_t.get(prn, t0) > args.watchdog_s
                        and _fr is not None and t0 - _fr[1] < 10.0
                        and prn in best and best[prn][0] >= _det_bar):
                    _reseed = ("det snr %.0f but ZERO coherent emits for >%.0f s%s"
                               % (best[prn][0], args.watchdog_s,
                                  " (trim RAILED %+.0f Hz)" % car_trim[prn]
                                  if _railed else ""))
                # WEAK-TRACK RESEED (2026-07-20): the coherent-but-weak zombie -- track
                # correlating ~20 dB off-peak with just enough coherence to hide from the
                # zero-coherence test above (C21/C42: sig 11-18 vs det snr strong, 70 min,
                # every rescuer blind). Judge track significance against the det bar:
                # strong det + persistently floor-level track = broken by construction.
                _tsig = max(_r.get("deep_snr") or 0.0, _r.get("amp_snr") or 0.0)
                if (args.watchdog_weak_sig > 0.0
                        and _fr is not None and t0 - _fr[1] < 10.0
                        and prn in best and best[prn][0] >= args.watchdog_det_snr):
                    if _tsig >= args.watchdog_weak_sig:
                        wd_strong_t[prn] = t0
                        wd_weak_n.pop(prn, None)  # cleared the bar -> backoff resets
                    elif (_reseed is None
                          # 3x birth grace (2026-07-20 13:12 soak): a reseed resets the
                          # deep ladder and sig takes 60-120 s to rebuild past the bar, so
                          # a 1x window re-fired on its own aftermath -- metronomic churn
                          # on healthy ramping sats (E3 at 50 dB-Hz reseeded 3x at birth).
                          # A real zombie (70 min) doesn't care about a 135 s judgment.
                          # EXPONENTIAL BACKOFF (14:05 soak): track sig alone cannot
                          # separate a zombie from a LEGIT-WEAK sat (E1 PRN 8: 29 dB-Hz,
                          # det snr 112 -- det snr barely scales with C/N0 on E1, so the
                          # weak-det exemption fails there) and the bar churned weak sats
                          # at exactly grace cadence. A real zombie is cured by fire #1;
                          # a sat that fires AGAIN earns doubled grace each time (135 s ->
                          # 270 -> 540 -> ... capped 16x), so persistent-weak sats are
                          # left alone while one-shot rescues stay fast.
                          and t0 - wd_birth[prn] > (3.0 * args.watchdog_s
                                                    * (2 ** min(wd_weak_n.get(prn, 0), 4)))
                          and t0 - wd_strong_t.get(prn, wd_birth[prn]) > args.watchdog_s):
                        wd_weak_n[prn] = wd_weak_n.get(prn, 0) + 1
                        _reseed = ("det snr %.0f but track sig %.0f < %.0f for >%.0f s "
                                   "(coherence %.2f, fire #%d -- WEAK-TRACK zombie)"
                                   % (best[prn][0], _tsig, args.watchdog_weak_sig,
                                      args.watchdog_s, _r.get("coherence_s") or 0.0,
                                      wd_weak_n[prn]))
                if _reseed is not None:
                    _log("WATCHDOG RESEED PRN %d: %s -> drop + fresh seed (tracker "
                         "state resets via the active-list gap)" % (prn, _reseed))
                    del seeds[prn]
                    dll_trim.pop(prn, None)
                    dll_last.pop(prn, None)
                    cp_held.discard(prn)
                    hold_miss.pop(prn, None)
                    cp_escape.pop(prn, None)
                    cp_err_hist.pop(prn, None)
            for k in list(wd_birth):
                if k not in seeds:   # any unseeding path re-stamps birth on re-entry
                    wd_birth.pop(k, None)
                    wd_coh_t.pop(k, None)
                    wd_strong_t.pop(k, None)
        for prn in list(seeds):
            if prn in probe_set:
                continue
            if (up is not None and prn not in up
                    and not (dr_state is not None and prn in dr_state["seeded"])):
                # (model-owned sats are exempt: the TLE up-set mismaps some BDS birds;
                # their BRDC elevation governs the drop, in the dead-reckon block)
                _log("drop PRN %d (set below horizon)" % prn)
                del seeds[prn]
                cp_held.discard(prn)
                hold_miss.pop(prn, None)
                low_hits.pop(prn, None)
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
            elif args.almanac and prn in pred:
                new_dop = pred[prn][0] + clock_bias
                old_dop = seeds[prn].get("doppler_hz", new_dop)
                if new_dop != old_dop:
                    # CURRENCY-CORRECT the coast, UNCONDITIONALLY (2026-07-19 audit A4): cp0
                    # is meaningful only in its doppler's currency -- updating the forecast
                    # dop WITHOUT re-expressing cp walks the despread by t_abs*f_chip*ddop/
                    # f_c chips at soak age (the t_abs lever the code-currency rule forbids;
                    # why long coasts silently lost the code peak). Same algebra as the
                    # hold-path TRANSLATE, which the dop-continuous A/B legs validated; the
                    # old --trim-precomp-coast gate (OFF in prod) was shipping the known-bad
                    # legacy raw-dop overwrite. The carrier pre-shift that used to ride here
                    # is gone with the trim-precomp flags (bench-rejected; the BOOTSTRAP
                    # re-pull owns step recovery).
                    t_abs = seeds[prn].get("ref_hop", 0) / args.hops_per_sec
                    seeds[prn]["code_phase_chips"] = (
                        seeds[prn].get("code_phase_chips", 0.0)
                        + t_abs * args.chip_rate_hz * args.code_doppler_sign
                          * (old_dop - new_dop) / args.carrier_hz) % CODE_LEN
                    seeds[prn]["doppler_hz"] = new_dop
                if "doppler_rate_hz_s" in seeds[prn]:
                    seeds[prn]["doppler_rate_hz_s"] = pred[prn][1]
            rec = status.get(prn, {})
            if have_sig:
                metric, thresh = sig_of(rec), args.lock_snr
            else:
                metric, thresh = float(rec.get("amplitude", 0.0)), args.drop_amplitude
            if metric >= thresh:
                low_hits[prn] = 0  # lock holding through the dropout -> reset coast
            else:
                low_hits[prn] = low_hits.get(prn, 0) + 1
                # dead-reckoned seeds are MODEL-owned: visible + predicted = keep despreading
                # (their whole point is sats with no signal above the search threshold)
                if (low_hits[prn] >= coast_polls and not args.coast_to_horizon
                        and not (dr_state is not None and prn in dr_state["seeded"])):
                    _log("drop PRN %d (coast %.0fs expired, %s=%.2f)"
                         % (prn, args.coast_budget, "sig" if have_sig else "|A|", metric))
                    del seeds[prn]
                    low_hits.pop(prn, None)

        # 3e. DEAD-RECKONED CODE-PHASE SEEDING (--dead-reckon): the search only exists to
        # measure what the model already knows. BRDC ephemeris (~2 m orbits + ~5 ns sat
        # clocks) plus the receiver clock solved from the sats we DO detect predict every
        # other visible sat's code phase to well inside the DLL capture range (0.10 chip
        # rms validated, gnss_deadreckon_check.py 2026-07-13) -- so seed them all:
        # sub-threshold sats despread on-peak with no detection ever required (the
        # sidelobe-mapping mode). The search demotes to bootstrap (clock solve), fallback
        # (a detection re-anchors via the normal seed loop, which also removes the PRN
        # from the model-owned set below) and integrity check (residuals logged here).
        if (dr_state is not None and args.almanac and pred and utc0_sample0
                and _now() >= dr_state["next"]):
            now_w = _now()
            dr_state["next"] = now_w + args.dr_refresh_s
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
            t_code = (LC_SEG * CODE_LEN) / args.chip_rate_hz if args.dr_long_code \
                     else CODE_LEN / args.chip_rate_hz
            # The seed is reduced at the SAME length the prediction was: one constant, used
            # twice, so they cannot drift apart.
            _DR_MOD = (LC_SEG * CODE_LEN) if args.dr_long_code else CODE_LEN
            # Layer-2 slew constants (task #30). CAP: 0.05 chips per 2 s cycle = 0.025
            # chips/s of correction authority -- above the 0.003-0.02 chips/s drift band
            # measured on sky, an order below the 0.25-chip DLL trims a fold tolerates, and
            # 20x below the ~1-chip clock-EMA jitter the reverted 10 s repin injected whole.
            # K: first-order low-pass; with the cap it bounds, it is deliberately unfussy.
            _DR_SLEW_CAP = 0.05
            _DR_SLEW_K = 0.25
            if dr_state["eph"] is None or now_w - dr_state["eph_t"] > 7200:
                try:
                    dr_state["eph"] = dr_eph_mod.parse_rinex_nav(dr_eph_mod.fetch_brdc())
                    dr_state["eph_t"] = now_w
                    dr_state["t0m"] = dr_eph_mod.gpst_of_utc(utc0_sample0) % t_code
                    _log("dead-reckon: BRDC loaded (%d sats)" % len(dr_state["eph"]))
                except Exception as e:
                    _log("dead-reckon: BRDC unavailable (%s); retry in 10 min" % e)
                    dr_state["eph_t"] = now_w - 7200 + 600
            # DECODED-EPH FALLBACK: keep predicting off our own decode when the network BRDC is
            # gone (or always, under --decoded-eph-fallback-force, the live A/B harness).
            _use_decoded = _decfb is not None and (
                args.decoded_eph_fallback_force
                or (args.decoded_eph_fallback and not dr_state["eph"]))
            if dr_state["eph"] or _use_decoded:
                tag = args.dr_constellation
                t_now_abs = now_w - utc0_sample0
                la = (args.code_bias_force * 1e-6 if args.code_bias_force is not None
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
                if la is None:
                    _sh_cb = rx.code_bias(band_id, exclude=chain_id, t_now=now_w)
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
                        _sh_cb = rx.code_bias_any_band(exclude=chain_id, t_now=now_w)
                        _sh_band = "cross-band"
                    if _sh_cb is not None:
                        la = float(_sh_cb.value)
                        _log_rl("la-adopt",
                                "dead-reckon: (l-a) %+.4f ppm ADOPTED from in-process chain "
                                "'%s' (%s %s) -> seeds carry the code-clock rate"
                                % (la * 1e6, _sh_cb.src,
                                   "same band" if _sh_band == band_id else "CROSS-BAND, rate only;",
                                   _sh_band if _sh_band == band_id else "phase stays per-band"),
                                every_s=300.0)
                    else:
                        la = 0.0
                # clock drift (chips/s): EMPIRICAL from consecutive raw solves (EMA'd
                # below), falling back to the f_chip*(l-a) model until measured -- the
                # modeled value left a persistent EMA lag (~0.6 chips at first deploy),
                # outside the BOC DLL capture range.
                drift = dr_state.get("drift")
                if drift is None:
                    drift = args.chip_rate_hz * la
                try:
                    # two epochs, 4 s apart: range_rate difference -> doppler RATE (the
                    # TLE almanac's rate is unused here -- BRDC governs model-owned sats)
                    if _use_decoded:
                        _ents = _decoded_entries(now_w)
                        pd = _decfb.predict_from_decoders(
                            _ents, args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(now_w, tz=timezone.utc), mask_deg=-90.0)
                        pd2 = _decfb.predict_from_decoders(
                            _ents, args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(now_w + 4.0, tz=timezone.utc),
                            mask_deg=-90.0)
                        if _now() - _decfb_log_t[0] > 60.0:
                            _decfb_log_t[0] = _now()
                            ab = ""
                            if args.decoded_eph_fallback_force and dr_state["eph"]:
                                # A/B: compare decoded vs BRDC predict, worst common sat.
                                pb = dr_eph_mod.predict_all(
                                    dr_state["eph"], args.lat, args.lon, args.alt,
                                    datetime.fromtimestamp(now_w, tz=timezone.utc),
                                    mask_deg=-90.0)
                                cm = set(pd) & set(pb)
                                if cm:
                                    dr_m = max(abs(pd[k]["range_m"] - pb[k]["range_m"])
                                               for k in cm)
                                    dd = max(abs(pd[k]["range_rate_mps"]
                                                 - pb[k]["range_rate_mps"]) for k in cm)
                                    ab = (" | A/B vs BRDC over %d sats: worst range %.1f m, "
                                          "range-rate %.3f m/s (%.2f Hz@fc)"
                                          % (len(cm), dr_m, dd,
                                             dd / 299792458.0 * args.carrier_hz))
                            _log("dead-reckon: predicting from DECODED eph (%s; %d entries -> "
                                 "%d sats)%s"
                                 % ("FORCE A/B" if args.decoded_eph_fallback_force
                                    else "BRDC network DOWN -> fallback",
                                    len(_ents), len(pd), ab))
                    else:
                        pd = dr_eph_mod.predict_all(
                            dr_state["eph"], args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(now_w, tz=timezone.utc), mask_deg=-90.0)
                        pd2 = dr_eph_mod.predict_all(
                            dr_state["eph"], args.lat, args.lon, args.alt,
                            datetime.fromtimestamp(now_w + 4.0, tz=timezone.utc),
                            mask_deg=-90.0)
                except Exception as e:
                    pd, pd2 = {}, {}
                    _log("dead-reckon: predict failed: %s" % e)
                if pd:
                    # cache for the SEED loop (next cycle): BRDC doppler/rate for
                    # search-anchored sats, so both masters share one currency
                    dr_state["pd"], dr_state["pd2"] = pd, pd2

                def cp_predicted(v, t_abs):
                    """Physical code phase (chips) of the predicted signal at capture age
                    t_abs, EXCLUDING the receiver clock. One predict_all per cycle: the
                    range is propagated to other epochs through range_rate (fine over the
                    few-second detection staleness). All mod arithmetic on small numbers
                    (t0m is the sample-0 GPST pre-reduced mod the code period)."""
                    t_tx = (dr_state["t0m"] + t_abs
                            - (v["range_m"] + v["range_rate_mps"] * (t_abs - t_now_abs))
                              / dr_eph_mod.C_LIGHT
                            + v["sat_clk_s"])
                    return (t_tx % t_code) * args.chip_rate_hz

                # -- receiver-clock solve (the bootstrap) + per-sat integrity residuals:
                # physical cp at each detection hop (undo the sample-0 back-reference),
                # minus the prediction, epoch-normalized to now (the offset drifts at
                # f_chip*(l-a)); the circular median over sats is the receiver clock.
                offs = []
                # DETECTION AGE per sat (task #33, docs 11.22 follow-up). The epoch
                # normalization below ages each latched detection by drift*(t_now - t_i)
                # with drift = f_chip*(l-a): the l-a EMA's measured noise is +-0.05-0.07
                # chips/s, and detections latch for up to the 1276 s per-PRN revisit --
                # so the normalization term can carry CHIPS of error that scale with THIS
                # SAT'S staleness. Logged next to each residual so "integrity" can be
                # judged against age: residuals that grow with age are measuring the
                # normalization, not the sky.
                det_age = {}
                for prn, (snr, dop, cp, ref_hop, _nh, _cpl, _car) in sorted(best.items()):
                    v = pd.get((tag, prn))
                    if v is None:
                        continue
                    t_i = ref_hop / args.hops_per_sec
                    det_age[prn] = t_now_abs - t_i
                    # The search's sample-0 back-reference subtracts BOTH the nominal code
                    # advance (t*f_chip mod L -- the 'off' term) and the code-Doppler drift;
                    # add BOTH back. Omitting the nominal term hid inside the solved clock
                    # whenever all detections shared a snapshot hop (every offline check),
                    # but across live scans it scatters by f_chip/hops_per_sec * (ref_hop
                    # mod code-period) -- the wandering "13 chips/s clock" of the first
                    # live deploy (2026-07-13).
                    cp_loc = (cp + t_i * args.chip_rate_hz
                              * (1.0 + args.code_doppler_sign * dop / args.carrier_hz)
                              ) % CODE_LEN
                    d_i = (cp_loc - cp_predicted(v, t_i)
                           + drift * (t_now_abs - t_i)) % CODE_LEN
                    offs.append((prn, d_i))
                dr_state["offs_t"] = now_w  # freshness stamp for the referee's integrity veto
                # -- P2a SHADOW: the joint receiver-state solve (task #33, section 3a) ----
                # `offs` IS the measurement, and always was. d_i = clk + b_i: the physical
                # code phase the search measured, minus the pure model, with no clock
                # removed. Today the next 40 lines take its circular median as the clock
                # and treat the per-sat spread (+-3-7 chips, docs 11.22) as an error to be
                # gated on; the joint solve reads the SAME numbers as clock PLUS per-sat
                # bias and estimates both, separated by process noise rather than by a
                # threshold. Shadow: logged beside the median it will replace, consumed by
                # NOTHING, so every transcript digest is untouched.

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

                if args.joint_shadow and offs:
                    try:
                        # P3: ONE state for the whole receiver, band carried as a
                        # measurement LABEL rather than as a separate filter. Two per-band
                        # states cannot estimate the offset between them, which is exactly
                        # why the second band needed a hand-wired cross-band clock bootstrap
                        # (#34) -- tau_band now has somewhere to live.
                        _js = rx.joint_receiver(band_id, CODE_LEN)
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
                        _js.cycle([((tag, p), d, args.joint_sigma, band_id)
                                   for p, d in offs
                                   if _snr.get(p, 0.0) >= args.joint_min_snr
                                   and _track_ok(p)
                                   and not _p2c_hold(_js, (tag, p))],
                                  t_now_abs)
                        # The filter has no logger; drain what it wants an operator to see.
                        # An escape or an incoherent run is a tracking event worth a line --
                        # on 2026-08-10 the single most damaging update of the day fired
                        # completely silently and was only found by its consequences.
                        for _n in _js.drain_notes():
                            _log_rl("joint-note", "JOINT %s: %s" % (band_id, _n),
                                    every_s=10.0)
                        for _p, _d in offs:
                            if _p2c_hold(_js, (tag, _p)):
                                _r = _js.wrap(_d - _js.predicted((tag, _p)) - _js.tau(band_id))
                                _log_rl("p2c-%d" % _p,
                                        "P2C %s PRN %d MASKED %.0fs: coast residual %+.3f chips "
                                        "(b %+.3f, sigma %.3f, tau %+.4f) -- flat = the state "
                                        "carries it"
                                        % (band_id, _p, _js.age((tag, _p), t_now_abs) or 0.0,
                                           _r, _js.bias((tag, _p)), _js.sigma((tag, _p)),
                                           _js.tau(band_id)),
                                        every_s=30.0)
                        if now_w >= dr_state.get("joint_log_next", 0.0):
                            dr_state["joint_log_next"] = now_w + 30.0
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
                            _log("JOINT[shadow] " + _js.summary(t_now_abs) + _tb + _amb)
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
                elif args.joint_model_primary and args.joint_shadow and seeds and not offs:
                    try:
                        _js = rx.joint_receiver(band_id, CODE_LEN)
                        _h1 = int(round(t_now_abs * args.hops_per_sec))
                        _th = _h1 / args.hops_per_sec
                        _mm = []
                        for _prn, _sd in seeds.items():
                            _v = pd.get((tag, _prn))
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
                            _held = dr_seed_phys(_sd, _h1, args.hops_per_sec,
                                                 args.chip_rate_hz, args.carrier_hz,
                                                 args.code_doppler_sign, _DR_MOD)
                            _y = ((_held + dll_trim.get(_prn, 0.0)
                                   - cp_predicted(_v, _th)) % _DR_MOD)
                            if _p2c_hold(_js, (tag, _prn)):
                                if True:
                                    _r = _js.wrap(_y - _js.predicted((tag, _prn)) - _js.tau(band_id))
                                    _log_rl("p2c-%d" % _prn,
                                            "P2C %s PRN %d MASKED %.0fs: coast residual %+.3f "
                                            "chips (b %+.3f, tau %+.4f)"
                                            % (band_id, _prn,
                                               _js.age((tag, _prn), t_now_abs) or 0.0,
                                               _r, _js.bias((tag, _prn)), _js.tau(band_id)), every_s=30.0)
                                continue
                            _mm.append(((tag, _prn), _y, args.joint_sigma, band_id))
                        if _mm:
                            _js.cycle(_mm, t_now_abs)
                        if now_w >= dr_state.get("joint_log_next", 0.0):
                            dr_state["joint_log_next"] = now_w + 30.0
                            _tb = "".join(
                                "  tau[%s] %+.3f+-%.3f (dual %d)"
                                % (_b, _js.tau(_b), _js.tau_sigma(_b),
                                   _js.tau_observability(_b))
                                for _b in sorted(_js._band_idx))
                            _amb = "  ⚠AMBIGUOUS(clk near wrap)" if rx.joint_ambiguous() else ""
                            _log("JOINT[shadow] " + _js.summary(t_now_abs) + _tb + _amb)
                    except Exception as e:
                        _log_rl("jointerr-mp",
                                "JOINT[shadow] model-primary feed skipped: %s" % e,
                                every_s=300.0)
                if len(offs) >= args.dr_min_sats:
                    ref = offs[0][1]
                    cen = sorted(((d - ref + CODE_LEN / 2) % CODE_LEN) - CODE_LEN / 2
                                 for _, d in offs)
                    raw = (cen[len(cen) // 2] + ref) % CODE_LEN
                    # DO THE SATELLITES AGREE? A median over >= dr_min_sats is only a
                    # measurement if its inputs cluster. Detections from a starved fleet are
                    # noise, their offsets scatter ~uniformly over CODE_LEN, and the circular
                    # median of that is an arbitrary number -- which was then accepted
                    # unconditionally, snapped straight into dr_state["clk"], and shipped as
                    # every satellite's seed.
                    #
                    # THAT IS THE OTHER HALF OF THE 2026-08-10 LOOP (docs 11.33): a wrong
                    # clock produced wrong seeds and wrong hints, which prevented the
                    # detections that would have corrected it, and the only escape was the
                    # clock random-walking back onto truth (6996 -> 5095 -> 1797 -> 1555 ->
                    # 9210 -> 134 chips over ~15 min, then a snap to 150.8 and lock). Three
                    # broker restarts that afternoon each re-rolled this dice on a thin fleet.
                    #
                    # THE SEPARATION IS ENORMOUS, so the bound is not a tuning knob: real
                    # per-sat offsets cluster inside ~+-10 chips (the joint state measures the
                    # biases at +-5, docs 11.22 quotes +-3-7), while uniform noise over a
                    # 10230-chip code has a MAD of ~2557. Anything from 50 to 500 separates
                    # them; 100 is the middle of that range in log terms.
                    _mad = sorted(abs(c - cen[len(cen) // 2]) for c in cen)[len(cen) // 2]
                    if _mad > args.dr_max_solve_mad_chips:
                        _log_rl("clkmad",
                                "clock solve REFUSED: %d sats scatter MAD %.0f chips (bound "
                                "%.0f) -- this is a median over NOISE, not a measurement; "
                                "holding clk %s"
                                % (len(offs), _mad, args.dr_max_solve_mad_chips,
                                   ("%.2f" % dr_state["clk"]) if dr_state.get("clk")
                                   is not None else "UNSET"),
                                every_s=30.0)
                        raw = None
                if len(offs) >= args.dr_min_sats and raw is not None:
                    prev_raw = dr_state.get("raw_prev")
                    # A primed drift is authoritative (the GPSDO rate is a band constant):
                    # never EMA it toward pair-differences of solutions built from UNCHANGED
                    # detections, which difference to ~zero and drag a correct prime away
                    # (measured 2026-07-31: primed +0.0439 walked to -61 within a minute).
                    if prev_raw is not None and 0.5 < now_w - prev_raw[1] < 30.0                             and args.dr_clock_drift is None:
                        d_est = (((raw - prev_raw[0] + CODE_LEN / 2) % CODE_LEN)
                                 - CODE_LEN / 2) / (now_w - prev_raw[1])
                        # PLAUSIBILITY BOUND. d_est is a DIFFERENCE OF TWO CLOCK SOLVES, so
                        # anything that displaces the solve -- a node restart, an F-engine
                        # restart, detections straddling either -- lands here as clock
                        # "motion" that never happened. Measured 2026-08-09: every node
                        # restart poisoned this EMA (+223 chips/s once, -36 another), and
                        # because the EMA is a=0.05 at ~30 s the poison then bleeds off over
                        # ~10 MINUTES, during which `drift` sweeps every model-primary seed
                        # off its peak. GPS survives (search-anchored); E5a/B2a/E5b/B2b do
                        # not -- E5a measured 96 -> 5 while GPS sat at 352 in the same
                        # minute. The bound is physics, not tuning: this clock is
                        # GPS-disciplined and its true drift is ~4e-4 chips/s (the joint
                        # state measures it), while the noisiest legitimate estimator we
                        # have scatters +-0.07. A whole chip per second is ~2500x the truth
                        # and 14x that scatter, so nothing real is being rejected.
                        if abs(d_est) > args.dr_max_drift_chips_s:
                            _log_rl("driftrej",
                                    "clock drift estimate %+.1f chips/s REJECTED (bound "
                                    "%.2f): the solve jumped, the clock did not -- holding "
                                    "drift %+.4f" % (d_est, args.dr_max_drift_chips_s,
                                                     dr_state.get("drift") or 0.0),
                                    every_s=30.0)
                        else:
                            dr_state["drift"] = (d_est if dr_state.get("drift") is None
                                                 else dr_state["drift"]
                                                 + 0.05 * (d_est - dr_state["drift"]))
                    dr_state["raw_prev"] = (raw, now_w)
                    # SNAP ON THE FIRST MEASUREMENT, whether or not a prime is standing.
                    # `raw` is a circular median over >= --dr-min-sats satellites, so it is a
                    # measurement of the same quantity the prime guessed; EMA-ing from the
                    # guess buys nothing but the walk-in. Snapping lands within the median's
                    # own scatter (a few chips at 6 sats) on cycle one instead of ~20 cycles.
                    if dr_state["clk"] is None or dr_state.pop("clk_primed", False):
                        was = dr_state["clk"]
                        dr_state["clk"] = raw
                        _log("dead-reckon: receiver clock BOOTSTRAP %.2f chips = %.3f us "
                             "(mod %.0f ms; %d sats%s)"
                             % (raw, raw / args.chip_rate_hz * 1e6, t_code * 1e3, len(offs),
                                "" if was is None else
                                "; REPLACES the %.2f-chip prime -- a prime is a seed, not a "
                                "measurement" % was))
                    else:
                        clk = (dr_state["clk"]
                               + drift * (now_w - dr_state["clk_t"])) % CODE_LEN
                        step = ((raw - clk + CODE_LEN / 2) % CODE_LEN) - CODE_LEN / 2
                        dr_state["clk"] = (clk + args.dr_clock_alpha * step) % CODE_LEN
                    dr_state["clk_t"] = now_w
                    # CONTRIBUTE (task #27 M3). THIS IS THE SEAM --dr-clock-adopt PAPERS
                    # OVER: dr_state straddles the boundary -- clk and drift are the
                    # RECEIVER's, seeded/pin/pd are this chain's per-PRN bookkeeping. A
                    # co-hosted chain with no detectors reads this directly instead of a
                    # JSON file written at flush cadence and gated on a two-read slew test.
                    # Carried WITH its code length, because chips are modular and a value
                    # mod 10230 is meaningless to a 1023000-chip code.
                    rx.contribute_dr_clock(chain_id, band_id, dr_state["clk"],
                                           dr_state.get("drift"), now_w, CODE_LEN)
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
                _rx_sib = (rx.dr_clock(band_id, exclude=chain_id, t_now=t0)
                           if (args.dr_clock_adopt and not offs) else None)
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
                if _rx_sib is None and args.dr_clock_adopt and not offs:
                    _cand = rx.dr_clock_any_band(exclude=chain_id, t_now=t0)
                    if _cand is not None and (_cand.extra.get("code_length") or 0) >= CODE_LEN:
                        _rx_sib, _rx_xband = _cand, True
                if _rx_sib is not None and _rx_xband:
                    _v = float(_rx_sib.value) % CODE_LEN
                    if dr_state.get("clk") is None or abs(
                            ((_v - dr_state["clk"] + CODE_LEN / 2) % CODE_LEN)
                            - CODE_LEN / 2) > 0.5:
                        _log("dead-reckon: clock BOOTSTRAP %.2f chips from in-process chain "
                             "'%s' (CROSS-BAND -- carries tau_band; the DLL residual IS that "
                             "measurement)" % (_v, _rx_sib.src))
                    dr_state["clk"] = _v
                    dr_state["clk_t"] = _rx_sib.t
                    dr_state.pop("clk_primed", None)
                elif _rx_sib is not None and _rx_sib.extra.get("code_length") == CODE_LEN:
                    if dr_state.get("clk") is None or abs(
                            ((float(_rx_sib.value) - dr_state["clk"] + CODE_LEN / 2)
                             % CODE_LEN) - CODE_LEN / 2) > 0.5:
                        _log("dead-reckon: clock ADOPTED %.2f chips from in-process chain "
                             "'%s' (same band %s, no file transport)"
                             % (float(_rx_sib.value), _rx_sib.src, band_id))
                    dr_state["clk"] = float(_rx_sib.value) % CODE_LEN
                    dr_state["clk_t"] = _rx_sib.t
                    # An adopted clock IS a measurement -- the sibling measured it -- so the
                    # prime is spent. If this chain ever gains detectors it should refine by
                    # EMA from here, not snap away from a good number.
                    dr_state.pop("clk_primed", None)
                    if _rx_sib.extra.get("drift") is not None:
                        dr_state["drift"] = float(_rx_sib.extra["drift"])
                elif _rx_sib is not None:
                    # Same band, different code length: the chips are modular in a different
                    # period, so the number is numerically fine and physically meaningless.
                    # Refuse loudly rather than adopt a plausible wrong value.
                    _log_rl("clkadopt-len",
                            "dead-reckon: chain '%s' publishes a clock mod %.0f chips but "
                            "this chain's code is %.0f -- NOT adoptable across code lengths"
                            % (_rx_sib.src, _rx_sib.extra.get("code_length") or -1, CODE_LEN),
                            every_s=60.0)
                if (_rx_sib is None and args.dr_clock_adopt and not offs and _xb_read_dir
                        and args.state_dongle):
                    try:
                        import receiver_state as _rs  # optional module, imported where used
                        sibs = _rs.read_dongle(
                            _xb_read_dir, args.state_dongle,
                            max_age_s=args.dr_clock_adopt_max_age_s, t_now=t0,
                            exclude=(os.path.basename(args.state_file).rsplit(".", 1)[0]
                                     if args.state_file else None))
                    except Exception:
                        sibs = []
                    # Freshest wins. Not a weighted mean: this is an adoption of ONE physical
                    # number measured by a chain that can measure it, and averaging two siblings'
                    # copies of the same quantity would just average their noise back in.
                    best_sib = None
                    _refused = False
                    for rec in sibs:
                        # GROUP NAME IS "rxclock", read off a live state file rather than
                        # inferred from the observe() call site -- the first attempt guessed
                        # "dr" from the surrounding variable names and silently found nothing,
                        # logging "no fresh sibling" while a perfectly good record sat there.
                        dr = (rec.get("rxclock") or {})
                        if dr.get("chips") is None:
                            continue
                        if best_sib is None or float(rec.get("t", 0)) > float(best_sib[0]):
                            best_sib = (rec.get("t", 0), rec, dr)
                    # QUALITY GATE: JUDGE THE CLOCK BY WATCHING IT, not by the sibling's
                    # aggregate quality fields. Two earlier versions of this gate were wrong in
                    # opposite directions and both cost time:
                    #
                    #   * age alone -> adopted a clock bouncing 8690 -> 2787 -> 9382 chips
                    #     minutes after the sibling restarted.
                    #   * `untrusted >= n` -> refuses forever. Those count DIFFERENT
                    #     populations: n = len(_ir), the satellites contributing an integrity
                    #     residual THIS cycle; untrusted = len(dr_untrusted), the persistent
                    #     demoted set. untrusted > n is normal after a restart, not impossible,
                    #     and comparing them is meaningless.
                    #   * integ_mad_chips > 1.0 -> also refuses a good clock. That MAD is taken
                    #     over a mixed population including badly-tracked satellites; measured
                    #     2026-08-08 it sat at 3.3-3.9 chips while the CLOCK ITSELF was stable
                    #     to 0.2 chips across consecutive reads.
                    #
                    # So gate on the quantity being adopted: has it stopped moving? Two reads a
                    # cycle apart answer that directly, in seconds, with no appeal to anyone's
                    # self-reported quality. A converged clock moves by its drift; a
                    # non-converged one moves by thousands of chips.
                    if best_sib is not None:
                        _, rec, dr = best_sib
                        _cand = float(dr["chips"]) % CODE_LEN
                        _prev = dr_state.get("adopt_prev")
                        if _prev is None:
                            dr_state["adopt_prev"] = (_cand, t0)
                            _log_rl("clkadopt-watch",
                                    "dead-reckon: watching sibling '%s' clock %.2f chips -- "
                                    "adopting once a second read confirms it is not moving "
                                    "(one cycle, not a burn-in)" % (rec.get("chain", "?"), _cand))
                            best_sib = None
                            _refused = True
                        else:
                            _pc, _pt = _prev
                            _dt = max(t0 - _pt, 1e-6)
                            _move = abs(((_cand - _pc + CODE_LEN / 2) % CODE_LEN)
                                        - CODE_LEN / 2) / _dt
                            dr_state["adopt_prev"] = (_cand, t0)
                            if _move > args.dr_clock_adopt_max_slew:
                                _log_rl("clkadopt-q",
                                        "dead-reckon: REFUSED sibling '%s' clock -- moving "
                                        "%.1f chips/s (limit %.1f). It has not converged; "
                                        "holding %.2f chips."
                                        % (rec.get("chain", "?"), _move,
                                           args.dr_clock_adopt_max_slew,
                                           dr_state["clk"] if dr_state.get("clk") is not None
                                           else float("nan")))
                                best_sib = None
                                _refused = True
                    if best_sib is not None:
                        _, rec, dr = best_sib
                        new_clk = float(dr["chips"]) % CODE_LEN
                        prev = dr_state.get("clk")
                        moved = (abs(((new_clk - prev + CODE_LEN / 2) % CODE_LEN)
                                     - CODE_LEN / 2) if prev is not None else None)
                        dr_state["clk"] = new_clk
                        dr_state["clk_t"] = t0
                        if dr.get("drift_chips_s") is not None:
                            dr_state["drift"] = float(dr["drift_chips_s"])
                        # Loud on a real MOVE, quiet on the steady state. A jump is the
                        # signature of an F-engine restart re-establishing frame 0, which is
                        # precisely the event the hand-primed constant used to survive wrongly.
                        if moved is None or moved > 0.5:
                            _log("dead-reckon: clock ADOPTED %.2f chips from band sibling "
                                 "'%s' (%s%s, age %.1f s)"
                                 % (new_clk, rec.get("chain", "?"),
                                    "cold" if moved is None else "moved %.2f chips" % moved,
                                    ", drift %+.4f chips/s" % dr["drift_chips_s"]
                                    if dr.get("drift_chips_s") is not None else "",
                                    t0 - float(rec.get("t", t0))))
                        else:
                            _log_rl("clkadopt", "dead-reckon: clock adopted %.2f chips from "
                                                "'%s' (steady)" % (new_clk, rec.get("chain", "?")))
                    elif args.dr_clock_adopt and not _refused:
                        # NOT after a quality refusal -- that path logs its own reason. Saying
                        # "no fresh sibling" when a sibling was found and REJECTED describes the
                        # wrong failure, and the two want opposite responses: absent means check
                        # the publisher, rejected means wait for it to converge.
                        _log_rl("clkadopt-none",
                                "dead-reckon: --dr-clock-adopt found no fresh sibling for dongle "
                                "'%s' in %s (<%.0f s) -- HOLDING the primed clock %.2f chips, "
                                "which does not survive an F-engine restart"
                                % (args.state_dongle, _xb_read_dir,
                                   args.dr_clock_adopt_max_age_s,
                                   dr_state.get("clk") if dr_state.get("clk") is not None else float("nan")))
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
                    for prn_i, d_i in offs:
                        r_i = (((d_i - dr_state["clk"] + CODE_LEN / 2.0) % CODE_LEN)
                               - CODE_LEN / 2.0)
                        # exported for the escape referee's integrity veto (chips, this
                        # chain's code; search-vs-model with the solved clock removed)
                        dr_state["integ"][prn_i] = (r_i, now_w)
                        v_i = pd.get((tag, prn_i))
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
                            _log("MODEL-UNTRUSTED PRN %d (%s, 3 consecutive) -> falling back "
                                 "to the SEARCH-measured Doppler for this sat (seamless: a "
                                 "source change is just another currency translation)"
                                 % (prn_i, why))
                        elif dr_bad.get(prn_i, 0) == 0 and prn_i in dr_untrusted:
                            del dr_untrusted[prn_i]
                            _log("MODEL-TRUSTED again PRN %d (integrity %+.2f chips)"
                                 % (prn_i, r_i))
                if dr_state["clk"] is not None and offs and now_w >= dr_state["log_next"]:
                    dr_state["log_next"] = now_w + 30.0
                    resid = ["PRN %d %+.2f a%.0f%s" % (p, r, det_age.get(p, -1),
                                                       " BAD" if abs(r) > 1.0 else "")
                             for p, d in offs
                             for r in [((d - dr_state["clk"] + CODE_LEN / 2) % CODE_LEN)
                                       - CODE_LEN / 2]]
                    _log("dead-reckon clock %.2f chips (%.3f us mod %.0f ms, drift "
                         "%+.3f chips/s); integrity: %s"
                         % (dr_state["clk"], dr_state["clk"] / args.chip_rate_hz * 1e6,
                            t_code * 1e3, dr_state.get("drift") or 0.0, "; ".join(resid)))
                # -- seed / re-pin every visible, undetected, unlocked sat from the model --
                if dr_state["clk"] is not None:
                    clk_now = (dr_state["clk"]
                               + drift * (now_w - dr_state["clk_t"])) % CODE_LEN
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
                    if _jr3 is not None and now_w >= dr_state.get("jslew_log_next", 0.0):
                        dr_state["jslew_log_next"] = now_w + 30.0
                        _cmp = []
                        for (_ct, _p) in list(_jr3._idx):
                            if _ct != tag:
                                continue
                            _lo = clk_now + bsat.get(_p, now_w)
                            _jo = _jr3.predicted((_ct, _p))
                            _dd = ((_jo - _lo + _DR_MOD / 2.0) % _DR_MOD) - _DR_MOD / 2.0
                            _cmp.append((_p, _dd, _jr3.sigma((_ct, _p)) or 0.0))
                        if _cmp:
                            _cmp.sort(key=lambda x: -abs(x[1]))
                            _log("SEED-OFFSET %s: joint-vs-legacy over %d sat(s), "
                                 "median %+.3f chips | %s"
                                 % (band_id, len(_cmp),
                                    sorted(abs(c[1]) for c in _cmp)[len(_cmp) // 2],
                                    " ".join("PRN%d %+.2f(s%.2f)" % c for c in _cmp[:6])))
                    planned = []
                    for (ctag, prn), v in sorted(pd.items()):
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
                        if (ctag != tag or v["el"] < args.mask_deg + 0.5 or prn in best
                                or prn in probe_set or prn in cp_held
                                or prn in dr_untrusted):   # model is wrong for this sat
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
                        _slew = (prn in seeds and not detectors
                                 and prn in dr_state["seeded"]
                                 and sig_of(status.get(prn, {})) >= args.lock_snr)
                        if (not _slew and prn in seeds
                                and sig_of(status.get(prn, {})) >= args.lock_snr):
                            continue  # sub-threshold LOCK: the DLL owns the residual now
                        if (not _slew and prn in dr_state["seeded"]
                                and now_w - dr_state["pin"].get(prn, 0.0) < args.dr_repin_s):
                            continue
                        # doppler + rate from BRDC range-rate (NOT the TLE pred: the BDS
                        # TLE<->PRN mapping mismaps some birds, and BRDC is the precision
                        # source anyway); clock_bias still comes from the TLE-vs-measured
                        # solve -- it's a receiver constant, common to both models.
                        v2 = pd2.get((ctag, prn))
                        dop_geo = -v["range_rate_mps"] / dr_eph_mod.C_LIGHT * args.carrier_hz
                        dop_seed = args.doppler_sign * dop_geo + clock_bias
                        drate = 0.0
                        if v2 is not None:
                            drate = (args.doppler_sign
                                     * (-(v2["range_rate_mps"] - v["range_rate_mps"]) / 4.0)
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
                        _leg_off = clk_now + bsat.get(prn, now_w)
                        _off = _leg_off
                        _jr3 = _joint_state(rx, band_id, args)
                        _joff = (_jr3.predicted((tag, prn))
                                 if (_jr3 is not None and (tag, prn) in _jr3._idx)
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
                                       _d3, CODE_LEN, _jr3.sigma((tag, prn)) or 0.0,
                                       "" if _ok3 else "  REFUSED (> %.1f chips)"
                                       % args.joint_slew_max_chips),
                                    every_s=60.0)
                            if "slew" in joint_consume and _ok3:
                                _off = _leg_off + _d3
                        cp0 = ((cp_predicted(v, t_now_abs) + _off)
                               - t_now_abs * args.chip_rate_hz
                                 * (1.0 + args.code_doppler_sign
                                    * dop_seed / args.carrier_hz)) % _DR_MOD
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
                            h1 = int(round(t_now_abs * args.hops_per_sec))
                            t_h = h1 / args.hops_per_sec
                            _held = dr_seed_phys(
                                seeds[prn], h1, args.hops_per_sec, args.chip_rate_hz,
                                args.carrier_hz, args.code_doppler_sign, _DR_MOD)
                            # The clock+bias offset is _off, computed once above so the birth
                            # phase and the slew target cannot disagree. b_sat is "how wrong
                            # the pure model is for THIS satellite", which is why this is the
                            # consumer aimed at the ~600 s plant oscillation (slew-to-model
                            # fighting trim-to-sky, with the model per-sat +-1-6 chips out).
                            _model = (cp_predicted(v, t_h) + _off) % _DR_MOD
                            _dcp = ((_model - _held + _DR_MOD / 2.0) % _DR_MOD
                                    ) - _DR_MOD / 2.0
                            _step = max(-_DR_SLEW_CAP,
                                        min(_DR_SLEW_CAP, _DR_SLEW_K * _dcp))
                            # Re-anchor at h1 with the FRESH model doppler/rate (kills the
                            # dt^2 linearization error a held seed accumulates) but at the
                            # HELD phase plus the bounded step -- never at the model's own
                            # phase, which carries the clock EMA jitter raw. NOT popping
                            # dll_trim: same trajectory, later epoch; the trim's residual is
                            # still valid (the lesson of the reverted repin).
                            seeds[prn] = {
                                "doppler_hz": dop_seed,
                                "code_phase_chips": dr_cp0(
                                    _held + _step, t_h, dop_seed,
                                    args.chip_rate_hz, args.carrier_hz,
                                    args.code_doppler_sign, _DR_MOD),
                                "code_phase_rate": cp_rate_from_code_bias(
                                    dop_seed, la, args.hops_per_sec,
                                    args.chip_rate_hz, args.carrier_hz),
                                "ref_hop": h1, "doppler_rate_hz_s": drate}
                            dr_state["pin"][prn] = now_w
                            _log_rl("drslew-%d" % prn,
                                    "dead-reckon SLEW PRN %d: model-held %+.3f chips, "
                                    "step %+.3f (cap %.2f), dop %+.0f rate %+.2f"
                                    % (prn, _dcp, _step, _DR_SLEW_CAP, dop_seed, drate),
                                    every_s=120.0)
                            continue
                        if prn not in seeds:
                            _log("dead-reckon SEED PRN %d (elev %.0f, cp0 %.1f, dop %+.0f,"
                                 " rate %+.2f)" % (prn, v["el"], cp0, dop_seed, drate))
                        dll_trim.pop(prn, None)  # any old trim served the OLD anchor
                        dll_last.pop(prn, None)
                        seeds[prn] = {
                            "doppler_hz": dop_seed, "code_phase_chips": cp0,
                            "code_phase_rate": cp_rate_from_code_bias(
                                dop_seed, la, args.hops_per_sec,
                                args.chip_rate_hz, args.carrier_hz),
                            "ref_hop": int(round(t_now_abs * args.hops_per_sec)),
                            "doppler_rate_hz_s": drate}
                        dr_state["seeded"].add(prn)
                        dr_state["pin"][prn] = now_w
                    if planned:
                        _log("dead-reckon DRY RUN, would seed: %s" % "; ".join(planned))
                    # model-owned sats drop on the BRDC elevation (they're exempt from
                    # the TLE horizon drop -- see the coast loop), or on capability
                    for prn in list(dr_state["seeded"]):
                        v = pd.get((tag, prn))
                        if prn < dr_min_prn or (_capable is not None and prn not in _capable):
                            _log("dead-reckon drop PRN %d (does not broadcast this signal)" % prn)
                            seeds.pop(prn, None)
                            low_hits.pop(prn, None)
                        elif v is None or v["el"] < args.mask_deg:
                            _log("dead-reckon drop PRN %d (set below BRDC horizon)" % prn)
                            seeds.pop(prn, None)
                            low_hits.pop(prn, None)
            # a fresh detection re-anchors via the seed loop (search = fallback); a
            # dropped seed (set below horizon) clears the model-owned state with it
            for prn in list(dr_state["seeded"]):
                if prn in best or prn not in seeds:
                    dr_state["seeded"].discard(prn)
                    dr_state["pin"].pop(prn, None)

        # 3c. Delay-lock loop (R1): close the CODE loop from the combiner's window-averaged E/L
        # powers. disc = (<|E|^2>-<|L|^2>)/(<|E|^2>+<|L|^2>) ~ -4*tau at 0.5-chip spacing, where
        # tau = (true - commanded) cp: the correlation triangle gives |E|=R(d+tau), |L|=R(d-tau),
        # so a LATE true peak strengthens L and drives disc negative -> tau_est = -disc/4,
        # scaled by (spacing/0.5). The per-PRN TRIM integrates gain*tau_est and is applied to
        # the seed at POST time only -- the stored seed stays pure fit/coast state, so the trim
        # converges to the search fit's quantization bias instead of double-counting on coasted
        # seeds. Gated on lock significance (an unlocked PRN's disc is noise). Bounded +-3 chips.
        if args.dll_gain > 0.0:
            # FLEET COMBINE (docs/CHORD_GNSS_SHARED_DLL.md). Sum the RAW powers across every
            # instance that reports the same window, then form ONE discriminator. Ratios do not
            # sum -- (SUM E - SUM L)/(SUM E + SUM L) is not any function of the per-instance
            # dll_disc values -- which is exactly why the combiner publishes e_pow/l_pow/p_pow.
            fleet = fleet_dll(dll_combiners, dll_hop_window, args.dll_min_instances,
                              args.dll_quality_sigma, args.dll_quality_min)
            # CROSS-NODE COHERENT COMBINE. Separate from the DLL on purpose: the DLL sums
            # POWERS (phase-free, which is what makes it cheap) while this removes the common
            # per-record PHASE, and the two answer different questions off the same endpoints.
            # It touches NO loop -- purely an observable, published for the viewer and the beam
            # map -- so a fault here can degrade what is displayed but can never move the code
            # or carrier loops.
            fcoh = {}
            if args.fleet_coherent:
                try:
                    fcoh = fleet_coherent(dll_combiners, args.coh_min_instances,
                                          args.coh_min_records, prns=set(seeds) or None,
                                          log=None, floor_margin=args.coh_floor_margin,
                                          seed=int(_now()))
                except Exception as e:
                    _log_rl("fleet-coh", "fleet coherent: skipped this cycle (%s)" % e)
            # PATH B, same estimator, separate population. Reported side by side rather than
            # merged: blending the two streams would make the very comparison this exists to
            # support impossible, and their per-record noise is NOT independent (both despread
            # the same antenna voltages), so a merged sum would not buy the sqrt(2) it appears
            # to. Publishes nothing and touches no loop.
            fcoh_n2 = {}
            if args.fleet_coherent and n2_combiners:
                try:
                    fcoh_n2 = fleet_coherent(n2_combiners, args.coh_min_instances,
                                             args.coh_min_records, prns=set(seeds) or None,
                                             log=None, floor_margin=args.coh_floor_margin,
                                             seed=int(_now()))
                except Exception as e:
                    _log_rl("fleet-coh-n2", "fleet coherent (path B): skipped (%s)" % e)
            if fcoh or fcoh_n2:
                # One line, both paths, only PRNs one of them actually detected. per_inst is
                # printed alongside the fleet number because the fleet total alone cannot
                # distinguish "the combine worked" from "one instance was already strong".
                rows = []
                for prn in sorted(set(fcoh) | set(fcoh_n2)):
                    a, b = fcoh.get(prn), fcoh_n2.get(prn)
                    if not ((a and a.get("present")) or (b and b.get("present"))):
                        continue

                    def _f(v):
                        # per_inst is url -> deep_snr (a FLOAT, not a row); best_inst_snr is
                        # already reduced for us. '*' marks a value that did NOT clear the
                        # shuffled-null floor -- printed anyway, because a number just under
                        # the bar is the useful one when judging whether a fleet is big enough.
                        if not v:
                            return "--"
                        return "%.0f/%d%s (best inst %.0f, floor %.1f)" % (
                            v.get("deep_snr", 0.0), v.get("n_src", 0),
                            "" if v.get("present") else "*", v.get("best_inst_snr", 0.0),
                            v.get("floor", 0.0))

                    rows.append("PRN %d A %s | B %s" % (prn, _f(a), _f(b)))
                if rows:
                    _log("FLEET-COH: " + "; ".join(rows))
            # FLEET PHASE-SLOPE DELAY FIT (task #32, docs/CHORD_JOINT_TRACKING.md P1).
            # MEASUREMENT ONLY: touches no seed and no loop -- it is logged and published so
            # it can be judged against the disc, the E/L asymmetry and GPS's search-measured
            # code phase BEFORE anything consumes it (the #30 rule: measure the statistic
            # before the loop). Gated on its OWN flag, which is what keeps every recorded
            # transcript replaying byte-identically: replay is strict-ordered, an
            # unrecorded GET is a TRANSCRIPT DIVERGENCE, and old transcripts' argv does not
            # carry --spectrum-endpoints, so replays never issue the new polls.
            spec_fit = {}
            if spectrum_endpoints:
                try:
                    _spec = fleet_spectrum(spectrum_endpoints, prns=set(seeds) or None)
                    for prn, pts in _spec.items():
                        r = fit_spectrum_delay(pts, args.chip_rate_hz, args.hops_per_sec)
                        if r is not None:
                            spec_fit[prn] = {"tau_chips": r[0], "peak": r[1],
                                             "floor": r[2], "n_pts": r[3], "n_inst": r[4]}
                except Exception as e:
                    _log_rl("fleet-spec", "fleet spectrum: skipped this cycle (%s)" % e)
                if spec_fit:
                    _log_rl("specfit", "SPEC-FIT: " + "; ".join(
                        "PRN %d tau %+.3f chips (p/f %.1fx, %d ch/%d inst)"
                        % (prn, v["tau_chips"], v["peak"] / max(v["floor"], 1e-12),
                           v["n_pts"], v["n_inst"])
                        for prn, v in sorted(spec_fit.items())), every_s=30.0)
                    # FEED b_sat (task #33). Presence-gated by the same lock metric the
                    # reseed logic trusts (deep-certified sig), because P1 measured weak-sat
                    # tau to be self-reference-biased toward zero -- a weak "tau ~ 0" is not
                    # a measurement. The filter adds its own thin-fleet and lobe gates.
                    for prn, v in spec_fit.items():
                        if sig_of(status.get(prn, {})) >= args.lock_snr:
                            bsat.update(prn, v["tau_chips"], v["n_inst"], t0)
                    _log_rl("bsat", "BSAT: " + bsat.summary(t0), every_s=60.0)
                    # Ride the fleet dict into the publisher: the viewer/status row grows
                    # spec_tau_chips + spec_peak_ratio next to the disc it will one day
                    # replace. EXISTING rows only -- publisher.update() indexes fleet rows'
                    # p_pow/disc/q directly, so a bare fit-only row would crash it; a PRN
                    # the DLL poll missed this cycle is still in the SPEC-FIT log line.
                    for prn, v in spec_fit.items():
                        if prn in fleet:
                            fleet[prn]["spec_tau"] = v["tau_chips"]
                            fleet[prn]["spec_ratio"] = v["peak"] / max(v["floor"], 1e-12)
                            fleet[prn]["bsat"] = bsat.get(prn, t0)
            if publisher is not None:
                # Published BEFORE the trim update so the row shows the state the loop acted
                # on, not the state after it acted -- otherwise a reader can never see the
                # input that produced a given correction.
                publisher.update(fleet, seeds, dll_trim, len(dll_combiners), last_dets, fcoh)
            dll_report = []
            for prn in list(seeds):
                rec = status.get(prn, {})
                fl = fleet.get(prn)
                if fl is not None:
                    # THE FLEET PATH. Gate on the summed q against a floor MEASURED from this
                    # cycle's own q population, not against a constant and not against the
                    # single-instance significance. sig_of(rec) is one node's 6.7% view and
                    # would keep vetoing precisely the satellites this exists for; a fixed bar
                    # is worse still, because summing tightens the noise distribution instead
                    # of raising q, so the correct bar FALLS as instances are added and any
                    # constant is right for exactly one fleet size (see fleet_dll).
                    #
                    # THE RATE GATE (design section 7, and it is not optional): the clock offset
                    # is 3.45 chips/s = 11 chips per loop round trip, fed forward by the seed's
                    # code_phase_rate. The trim only ever absorbs the residual. With no live
                    # rate the trim would face the whole 11 chips -- far outside the +-0.5 chip
                    # pull-in, unrecoverable, and it would read as the DLL diverging rather than
                    # the feed-forward being absent. So HOLD instead of integrating.
                    if not float(seeds[prn].get("code_phase_rate", 0.0) or 0.0):
                        _log_rl("dll-norate-%d" % prn,
                                "fleet DLL PRN %d: no live code_phase_rate, holding trim" % prn)
                        continue
                    if not fl["present"]:
                        continue
                    # One integration per new WINDOW -- an exact integer test, where the
                    # single-instance path below can only watch for a changed float.
                    if fl["hop"] == dll_last_hop.get(prn):
                        continue
                    dll_last_hop[prn] = fl["hop"]
                    disc = fl["disc"]
                else:
                    disc = float(rec.get("dll_disc", 0.0))
                    if sig_of(rec) < args.lock_snr or disc == 0.0:
                        continue
                    # One integration per NEW measurement: the combiner emits a fresh disc each
                    # integration window (~1 s) while this loop polls at ~5 Hz -- integrating the
                    # stale value 5x per emit over-applies the gain (part of the 2026-07-07 L1
                    # runaway). A changed value marks a fresh emit.
                    if disc == dll_last.get(prn):
                        continue
                    dll_last[prn] = disc
                tau = -max(-1.0, min(1.0, disc)) / 4.0 * (args.dll_spacing / 0.5)
                # LEAKY integrator: mean-reverts, so discriminator NOISE cannot random-walk the
                # trim to the clamp (a pure integrator did -- L1 trims reached +-1 to 3 chips
                # over 2 min, dragging the code off the +-0.5-chip peak and collapsing the deep
                # while the search stayed strong). Steady state gain*tau/leak still nulls a real
                # static fit bias; noise averages over ~1/leak windows.
                # LEAK GATE. The leak exists because a PURE integrator random-walked the trim
                # into the clamp on discriminator noise (L1, 2026-07-07). But it also CAPS the
                # correction: steady state is trim = (gain/leak)*tau and |tau| <= 0.25 chips, so
                # at gain 0.25 / leak 0.05 nothing beyond 1.25 chips is reachable. Measured
                # 2026-08-04: PRN 9 parked at trim +1.00 with disc railed at -0.984 -- the loop
                # pushing as hard as it can and still not arriving.
                #
                # What the leak was standing in for was a signal test, and the fleet path now
                # HAS one: `present` is a self-calibrating significance on the summed prompt
                # power (fleet_dll), and a PRN that fails it is skipped entirely rather than
                # leaked. So on the fleet path the leak can be much smaller -- it keeps the slow
                # mean-reversion that stops a long-lived bias accumulating, without capping the
                # correction below the errors we actually have. The single-combiner fallback
                # keeps the original leak: it has no such gate, and that is where the runaway
                # happened.
                leak = args.dll_leak_present if fl is not None else args.dll_leak
                trim = (1.0 - leak) * dll_trim.get(prn, 0.0) + args.dll_gain * tau
                dll_trim[prn] = max(-3.0, min(3.0, trim))
                dll_report.append(
                    "PRN %d disc %+.3f trim %+.2f%s"
                    % (prn, disc, dll_trim[prn],
                       "" if fl is None
                       else " [fleet %d/%d q %.2f p %.1fx]"
                            % (fl["n_src"], len(dll_combiners), fl["q"],
                               fl["p_pow"] / fl["p_med"] if fl.get("p_med") else 0.0)))
            if dll_report:
                _log("DLL: " + "; ".join(dll_report))
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
                for _p in sorted(cp_fit_slope):
                    _rec = status.get(_p, {})
                    _meas = float(_rec.get("carrier_hz_resid", 0.0))
                    _sig = sig_of(_rec)
                    if _sig < args.lock_snr:
                        continue
                    _rows.append("PRN %d code->%+.2f Hz meas %+.2f Hz (sig %.1f)"
                                 % (_p, cp_fit_slope[_p] * _k, _meas, _sig))
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
            if fleet:
                any_fl = next(iter(fleet.values()))
                _log_rl("dll-floor",
                        "fleet DLL: %d PRN(s) over %d combiner(s), %d present, q floor %.2f%s"
                        % (len(fleet), len(dll_combiners),
                           sum(1 for v in fleet.values() if v["present"]), any_fl["q_floor"],
                           "" if any_fl["q_med"] is None
                           else " (noise median %.2f, sigma %.3f)"
                                % (any_fl["q_med"], any_fl["q_sigma"])))
            for k in list(dll_trim):
                if k not in seeds:
                    del dll_trim[k]
                    dll_last_hop.pop(k, None)

        # 3d. SHARED CARRIER LOOP (the carrier twin of 3c): integrate the combiner's full-band
        # cross-record phase-walk residual into a commanded NCO frequency per PRN. The residual
        # is measured AFTER the current trim (the NCO derotates before records ship), so the
        # plain integrator converges: trim += gain * resid. No lock gate: the observable is
        # vector-averaged over the emit window at FULL-BAND SNR (the whole point -- per-channel
        # amplitude gates would exclude exactly the weak-band cases this loop exists for);
        # the clamp bounds any noise walk.
        if args.carrier_gain > 0.0:
            for _p in [p for p in _trim_force if p in seeds]:
                car_trim[_p] = _trim_force.pop(_p)
                _log("TRIM FORCE (bench): PRN %d car_trim POISONED to %+.1f Hz"
                     % (_p, car_trim[_p]))
            rate_resid, rate_consensus = {}, None
            if args.carrier_source == "rate":
                rate_resid, rate_consensus = rate_residuals(
                    status, args.carrier_rate_min_q, args.carrier_rate_clip_hz, _log,
                    prev_hop=rate_prev_hop, max_gap=args.carrier_rate_max_gap,
                    prev_val=rate_prev_val, max_step=args.carrier_rate_max_step,
                    unit_hop=rate_unit_hop)
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
                if resid == car_last.get(prn):
                    continue
                car_last[prn] = resid
                coh_ok = (rec.get("coherence_s") or 0.0) > 0.0
                # ---- VERIFYING: an applied step hypothesis is judged by OUTCOME ----
                # (explain-apply-verify, 2026-07-22): a trim correction is a falsifiable
                # hypothesis -- either coherence returns / the residual collapses within
                # CARRIER_VERIFY_EMITS, or it was WRONG and is reverted + escalated to a
                # full re-acquire. This bounded closed loop is exactly what the two
                # retracted open-loop escapes (v2 EMA unwrap, loose step-accept) were
                # missing: a wrong correction costs one reverted step, never compounds.
                if prn in car_verify:
                    v = car_verify[prn]
                    v["emits"] += 1
                    if coh_ok or abs(resid) < CARRIER_EXPLAIN_HZ:
                        del car_verify[prn]
                        car_fade.pop(prn, None)
                        _log("CARRIER STEP VERIFIED PRN %d: healed in %d emit(s) "
                             "(coh=%s, resid %+.2f Hz)" % (prn, v["emits"], coh_ok, resid))
                        # fall through: this emit integrates normally below
                    elif v["emits"] >= CARRIER_VERIFY_EMITS:
                        car_trim[prn] = v["prev_trim"]  # revert the refuted hypothesis
                        del car_verify[prn]
                        car_locked.discard(prn)         # escalate: BOOTSTRAP re-acquire
                        car_step_t[prn] = t0 + 50.0     # ~60 s hypothesis lockout
                        car_fade.pop(prn, None)
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
                if prn in car_bleed_verify:
                    bv = car_bleed_verify[prn]
                    bv["emits"] += 1
                    if bv["emits"] >= args.carrier_bleed_verify_emits:
                        del car_bleed_verify[prn]
                        # Judge by the SETTLED residual, not coh_ok (which blips for ~1 emit on the
                        # deep-window reset a re-pin causes, good bleed or not). A residual at/under
                        # the bar means the fold left the carrier aligned; a large one is a real
                        # miss (the loop re-grows the trim from 0 either way -- and even a mild miss
                        # already REDUCED the standing trim, so the bar is generous).
                        if abs(resid) <= args.carrier_bleed_ok_hz:
                            _log("CARRIER BLEED VERIFIED PRN %d: resid settled %+.2f Hz "
                                 "(<= %.2f) over %d emits, trim now %+.2f"
                                 % (prn, resid, args.carrier_bleed_ok_hz, bv["emits"],
                                    car_trim.get(prn, 0.0)))
                        else:
                            _log("CARRIER BLEED REFUTED PRN %d: resid %+.2f Hz (> %.2f) after "
                                 "%d emits -- loop re-grows trim, %.0f s lockout"
                                 % (prn, resid, args.carrier_bleed_ok_hz, bv["emits"],
                                    args.carrier_bleed_lockout_s))
                sig = (max(rec.get("deep_snr") or 0.0, rec.get("amp_snr") or 0.0)
                       if coh_ok else 0.0)
                tracking = prn in car_locked
                if coh_ok and sig >= args.carrier_min_sig > 0.0:
                    car_locked.add(prn)
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
                        hist = car_step_hist.setdefault(prn, [])
                        hist.append((t0, resid))
                        del hist[:-args.carrier_step_accept]
                        band = max(2.0, args.carrier_innov_hz)
                        if (len(hist) >= args.carrier_step_accept
                                and t0 - hist[0][0] < 30.0
                                and t0 - car_step_t.get(prn, 0.0) >= 10.0):
                            vals = sorted(r for _, r in hist)
                            med = vals[len(vals) // 2]
                            if (vals[-1] - vals[0] < band
                                    and abs(med) >= CARRIER_EXPLAIN_HZ):
                                prev_trim = car_trim.get(prn, 0.0)
                                car_trim[prn] = max(-args.carrier_max_hz,
                                                    min(args.carrier_max_hz,
                                                        prev_trim + med))
                                car_step_t[prn] = t0
                                car_step_hist[prn] = []
                                car_verify[prn] = {"prev_trim": prev_trim, "emits": 0}
                                _log("CARRIER STEP HYPOTHESIS PRN %d: %d agreeing gated "
                                     "resids (med %+.2f Hz, spread %.2f) -> trim %+.2f, "
                                     "VERIFYING (heal in %d emits or revert)"
                                     % (prn, args.carrier_step_accept, med,
                                        vals[-1] - vals[0], car_trim[prn],
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
                    car_fade[prn] = car_fade.get(prn, 0) + 1 if present else 0
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
                                and t0 - wd_coh_t.get(prn, 0.0) < args.refade_flicker_s)
                    if (args.carrier_refade > 0 and not _flicker
                            and car_fade.get(prn, 0) >= args.carrier_refade):
                        car_locked.discard(prn)
                        car_fade.pop(prn, None)
                        _log("CARRIER REACQ PRN %d: %d consecutive gated emits at full amp "
                             "(last resid %+.2f Hz) -> BOOTSTRAP re-pull"
                             % (prn, args.carrier_refade, resid))
                    continue  # this emit stays held: coast on the feed-forward
                car_fade.pop(prn, None)
                car_step_hist.pop(prn, None)  # ungated emit: gated-run agreement is stale
                if not tracking and args.carrier_det_gate_s > 0.0:
                    # BOOTSTRAP WALK GATE: no fresh detection = no evidence the estimator
                    # has a signal to measure; its residual is noise and integrating it
                    # random-walks the trim (see --carrier-det-gate-s). Hold and coast.
                    _fr = det_fresh.get(prn)
                    if _fr is None or t0 - _fr[1] > args.carrier_det_gate_s:
                        continue
                if prn not in car_trim and args.carrier_fleet_seed:
                    # start on the fleet's clock, not at 0: the converged trim is the chain's
                    # deterministic frac-N LO offset, common-mode across sats
                    fleet = sorted(car_trim.values())
                    if len(fleet) >= 3:
                        car_trim[prn] = fleet[len(fleet) // 2]
                prev_trim = car_trim.get(prn, 0.0)
                trim = (1.0 - args.carrier_leak) * prev_trim + args.carrier_gain * resid
                if tracking and args.carrier_max_step > 0.0:
                    trim = prev_trim + max(-args.carrier_max_step,
                                           min(args.carrier_max_step, trim - prev_trim))
                car_trim[prn] = max(-args.carrier_max_hz, min(args.carrier_max_hz, trim))
                car_report.append("PRN %d resid %+.2f Hz trim %+.2f" % (prn, resid, car_trim[prn]))
                # ---- f_ref TRIM-BLEED SHADOW (log-only, no action) ----
                # This emit is COHERENT and TRACKING (it reached the integrator ungated). If the
                # trim has held a STANDING value across the stability window, f_ref is pinned
                # off-true by ~that trim and a re-pin (f_ref += trim) would clear it. Log the
                # candidate so the trigger can be validated on live data before it is ever armed.
                # Recency-windowed like car_step_hist (a decoherence gap ages the window out ->
                # not "converged"), so no per-gate-branch cleanup is needed.
                if (args.carrier_bleed_shadow or args.carrier_bleed) and tracking and coh_ok:
                    bh = car_bleed_hist.setdefault(prn, [])
                    bh.append((t0, car_trim[prn]))
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
                                 and abs(car_trim[prn]) >= args.carrier_bleed_hz
                                 and max(vals) - min(vals) <= args.carrier_bleed_stable_hz
                                 and abs(slope) <= args.carrier_bleed_max_slope)
                    # ARMED: re-pin f_ref and zero the trim (one bleed per lockout, never while a
                    # step- or bleed-hypothesis is already under verify for this PRN).
                    if (converged and args.carrier_bleed and prn not in car_verify
                            and prn not in car_bleed_verify
                            and t0 >= car_bleed_lock_t.get(prn, 0.0)):
                        prev_trim = car_trim[prn]
                        car_trim[prn] = 0.0             # f_ref re-pin absorbs the offset
                        car_repin_pending[prn] = prev_trim  # tracker does f_ref += prev_trim
                        car_bleed_verify[prn] = {"emits": 0, "prev_trim": prev_trim, "t": t0}
                        car_bleed_lock_t[prn] = t0 + args.carrier_bleed_lockout_s
                        car_bleed_hist[prn] = []
                        car_bleed_log_t[prn] = t0
                        _log("CARRIER BLEED PRN %d: re-pinning f_ref (%+.2f Hz absorbed, slope "
                             "%+.3f Hz/s), trim->0, VERIFYING (heal in %d emits)"
                             % (prn, prev_trim, slope, args.carrier_bleed_verify_emits))
                    elif converged and t0 - car_bleed_log_t.get(prn, 0.0) >= 60.0:
                        car_bleed_log_t[prn] = t0
                        _log("CAR-BLEED CANDIDATE PRN %d: trim %+.2f Hz stable %d emits "
                             "(spread %.2f, slope %+.3f Hz/s), coherent -> %s"
                             % (prn, car_trim[prn], len(bh), max(vals) - min(vals), slope,
                                "locked out" if args.carrier_bleed
                                else "would re-pin f_ref, predict trim->~0 (shadow, no action)"))
            if car_report:
                _log("CAR: " + "; ".join(car_report))
            for k in list(car_trim):
                if k not in seeds:
                    del car_trim[k]
                    car_locked.discard(k)  # a re-seeded sat re-enters via BOOTSTRAP
                    car_verify.pop(k, None)  # a dropped sat's hypothesis dies with it
                    car_step_hist.pop(k, None)
                    car_fade.pop(k, None)
                    car_bleed_hist.pop(k, None)  # its convergence history dies with it
                    car_bleed_log_t.pop(k, None)
                    car_bleed_verify.pop(k, None)
                    car_bleed_lock_t.pop(k, None)
                    car_repin_pending.pop(k, None)

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
                _ct = sorted(car_trim.values())
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
                seed["code_phase_rate"] = cp_rate_from_code_bias(
                    seed["doppler_hz"], cb_to_seed, args.hops_per_sec,
                    args.chip_rate_hz, args.carrier_hz)
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
                car_trim[prn] = _ctc

        # 4. push consensus seeds to every tracker (DLL trim applied at POST time only)
        payload = []
        bit_src, bit_known = {}, {}
        for prn, v in sorted(seeds.items()):
            d = dict(prn=prn, **v)
            if dll_trim.get(prn):
                d["code_phase_chips"] = d["code_phase_chips"] + dll_trim[prn]
            if car_trim.get(prn):
                d["carrier_trim_hz"] = car_trim[prn]
            if prn in car_repin_pending:
                # ONE-SHOT trim-bleed re-pin: the tracker does f_ref += this amount this frame.
                # Consume it here so it rides exactly this post (car_trim was zeroed above, so no
                # carrier_trim_hz accompanies it -- the trim moves wholly into f_ref, leaving the
                # combined carrier invariant).
                d["carrier_repin"] = car_repin_pending.pop(prn)
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
        if cl_tracker and utc0_sample0 and args.almanac and pred:
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
                pv = pred.get(d["prn"])
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
        if state_w is not None:
            try:
                if dr_state is not None:
                    _iw = _now()
                    _ir = [v[0] for v in dr_state.get("integ", {}).values()
                           if isinstance(v, (list, tuple)) and _iw - v[1] < 30.0]
                    # ⚠️ CHIPS ARE NOT COMPARABLE ACROSS CHAINS -- a chip is a different
                    # duration per signal (1.023 Mcps at L1 C/A, 10.23 at L5), and this
                    # value is mod a different CODE_LEN too. Publishing only chips is the
                    # count-where-a-time-was-meant trap that has bitten this node three
                    # times; carry the rate and the derived TIME/FRACTIONAL forms so a
                    # consumer never has to guess. drift_ppm is the important one: it is
                    # directly comparable to l-a and to the carrier bias in ppm, and that
                    # comparison is what shows dr drift to be a RESIDUAL (measured 0.000
                    # -0.11 of l-a on all 8 chains) rather than a third estimator.
                    _cr = float(args.chip_rate_hz) if args.chip_rate_hz else None
                    _dr = dr_state.get("drift")
                    _im = receiver_state.mad(_ir)
                    state_w.observe(
                        "rxclock",
                        chips=dr_state.get("clk"),
                        chip_rate_hz=_cr,
                        us=((dr_state["clk"] / _cr * 1e6)
                            if (_cr and dr_state.get("clk") is not None) else None),
                        drift_chips_s=_dr,
                        drift_ppm=((_dr / _cr * 1e6) if (_cr and _dr is not None) else None),
                        n=len(_ir),
                        integ_mad_chips=_im,
                        integ_mad_us=((_im / _cr * 1e6) if (_cr and _im is not None) else None),
                        untrusted=len(dr_untrusted),
                        age_s=(round(t0 - dr_state["clk_t"], 2)
                               if dr_state.get("clk_t") else None))
                # ---- S2c: fuse this dongle's chains, PUBLISH, consume NOTHING ----
                # Sources are read from the state files, self included (self's record is up
                # to one flush old -- irrelevant for a quantity measured flat within noise
                # over 15 min, and it keeps the ordering trivial). No feedback loop exists:
                # sources_from() reads only `carrier.raw_hz` / `code.raw_ppm`, never the
                # `fused` group this writes back.
                if args.state_fuse and _state_dir:
                    # Reuse the cycle's cached fusion -- the same object the consumption
                    # path above was handed, so the published record and the value actually
                    # used can never disagree.
                    _fus = _fuse_cached(t0)
                    if _fus:
                        state_w.observe(
                            "fused",
                            lo_ppm=_fus["lo_ppm"], se_ppm=_fus["se_ppm"],
                            lo_ppm_norej=_fus["lo_ppm_norej"],
                            n_src=_fus["n_src"], n_carrier=_fus["n_carrier"],
                            n_code=_fus["n_code"], n_rejected=_fus["n_rejected"],
                            all_outliers=_fus["all_outliers"],
                            worst_sigma=_fus["worst_sigma"], chains=_fus["chains"],
                            hz_here=_fus["lo_ppm"] * 1e-6 * args.carrier_hz)
                        # SHADOW LINE: what the fused prior says, beside what this chain is
                        # actually using. The delta is the whole S2d decision, logged every
                        # minute so a soak can be read straight out of the broker log.
                        _fhz = _fus["lo_ppm"] * 1e-6 * args.carrier_hz
                        _own = clock_bias_ema
                        _log_rl("shadowfuse",
                                "SHADOW fused LO %+.5f ppm +-%.5f (%d src: %dc/%dd over %s"
                                "%s%s) -> %+.2f Hz here; chain uses %s; delta %s [%s]"
                                % (_fus["lo_ppm"], _fus["se_ppm"] or 0.0, _fus["n_src"],
                                   _fus["n_carrier"], _fus["n_code"],
                                   ",".join(_fus["chains"]),
                                   "; REJECTED %d" % _fus["n_rejected"]
                                   if _fus["n_rejected"] else "",
                                   "; ALL-OUTLIERS (no majority -- do not trust)"
                                   if _fus["all_outliers"] else "",
                                   _fhz,
                                   ("%+.2f Hz" % _own) if _own is not None else "UNSOLVED",
                                   ("%+.2f Hz" % (_fhz - _own)) if _own is not None
                                   else "n/a (this is exactly the case fusion rescues)",
                                   # three honest modes: actively rescuing an unsolved
                                   # chain / armed but idle (steady state, proven no-op) /
                                   # pure shadow. "CONSUMED" when the chain is solved would
                                   # be a lie under rescue-only semantics.
                                   ("RESCUING (own unsolved)"
                                    if args.state_consume and clock_bias_ema is None
                                    else "RESCUE-ARMED, idle"
                                    if args.state_consume else "SHADOW")),
                                every_s=60.0)
                state_w.flush(t0)
            except Exception as e:
                # NOT a bare pass. A silent except here once swallowed a broken format
                # string in the shadow line: the line simply stopped appearing and nothing
                # said why -- and a soak read from that log would have looked like "the
                # fuser stopped running" or, worse, like clean data. Rate-limited so a
                # persistent fault names itself once a minute instead of flooding.
                _log_rl("stateobs", "receiver-state observe/flush failed: %r" % (e,),
                        every_s=60.0)
        if dhw is not None:
            dhw.flush(t0)
        if args.once:
            return
        # REAL elapsed time, not the frozen cycle clock: this is the one place in the loop
        # that must know how long the pass actually took, or the control cadence would
        # become interval + processing rather than interval.
        dt = args.interval - (time.time() - t_wall)
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
        print(_TR.digest())
    finally:
        _TR.close()
