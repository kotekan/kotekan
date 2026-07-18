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
import math
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from gnss_stages import resolve_stage  # noqa: E402  (gps_* <-> bare stage-name aliasing)


def _get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, payload, timeout=5.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _log(msg):
    print("[broker] %s" % msg, file=sys.stderr, flush=True)


def fit_cp_rate(hist, code_len):
    """Least-squares fit cp0 vs capture hop -> (rate chips/hop, hop_ref, cp_ref).

    The search anchors each cp0 to its snapshot hop (capture time, shared with the
    tracker via fpga_seq), but cp0 drifts ~linearly with that hop (residual code-rate
    error). Fitting the slope lets the tracker extrapolate to its own window hop and
    sit on the peak -- a first-order code model that removes the seed-staleness bias.
    Returns None if there isn't enough spread to fit. cp0 is unwrapped (period
    code_len) along the sequence before fitting; the result is anchored at the latest
    hop. hop is centred to keep the (large) absolute index from losing precision.
    """
    if len(hist) < 3:
        return None
    hops = [h for h, _ in hist]
    cps = [c for _, c in hist]
    unw = [cps[0]]
    for c in cps[1:]:
        d = c - (unw[-1] % code_len)
        d -= code_len * round(d / code_len)  # nearest wrap
        unw.append(unw[-1] + d)
    h0 = hops[-1]
    dh = [float(h - h0) for h in hops]
    n = len(dh)
    sh, sc = sum(dh), sum(unw)
    shh = sum(x * x for x in dh)
    shc = sum(x * y for x, y in zip(dh, unw))
    den = n * shh - sh * sh
    if den == 0.0:
        return None
    rate = (n * shc - sh * sc) / den          # chips per hop
    cp_ref = (sc - rate * sh) / n             # fitted cp0 at h0
    return rate, h0, cp_ref % code_len


def code_clock_bias_sample(rate_chips_per_hop, doppler_hz, hops_per_sec, chip_hz, carrier_hz):
    """One satellite's estimate of the receiver LO-vs-ADC clock offset (l - a), dimensionless.

    CONVENTION (the 2026-07-04 L5 finding): the fitted slope here is the drift of the SEARCH's
    absolute-anchored cp0, which is a RESIDUAL rate -- the replica/search already apply the
    geometric code Doppler internally (chip_per_sample scales with the seeded Doppler; the
    search references cp back to sample 0 with the matching term). What remains in the cp0
    slope is f_chip * (l - a) (+ the small Doppler-quantization residue), so
        l - a  =  slope / f_chip
    with NO carrier_frac subtraction: v/c never appears in a residual slope. (The original
    formula subtracted doppler/carrier, valid for the OFFLINE raw-drift tools like
    l1_code_drift.py where the code drift is measured without a feed-forward -- but applied to
    residual-convention slopes it contaminated l-a by the per-sat carrier_frac: the estimates
    disagreed band-to-band (+0.25 ppm L1 / -0.63 ppm L5) where the residual reading agrees
    (+0.03 / -0.04 ppm, both near the GPSDO's measured +0.06).)
    doppler_hz/carrier_hz are kept in the signature for call-site stability; unused.
    """
    del doppler_hz, carrier_hz  # residual convention: geometry is already fed forward
    return rate_chips_per_hop * hops_per_sec / chip_hz


def cp_rate_from_code_bias(doppler_hz, code_bias, hops_per_sec, chip_hz, carrier_hz):
    """Seed the cp0 slope (chips/hop) for a not-yet-fittable sat from the calibrated (l - a).

    RESIDUAL convention (2026-07-04 L5 finding): cp0 is absolute-anchored and the replica applies
    the geometric code Doppler itself, so the correct seeded slope is ONLY the clock residual
        cp0_rate = f_chip * (l - a)
    The original formula added doppler/f_carrier -- the FULL physics code rate -- which the
    replica then applied AGAIN: unfitted sats slid off-peak at the code-Doppler rate. Fatal at
    L5 (+-30 chips/s -> off the +-1 chip peak in <1 s; the replay smoking gun: seeded cp 6690,
    displayed 6477 after 65 s = -3.28 chips/s = exactly dop/f * f_chip), historically masked at
    L1 (+-3 chips/s) by fast fits + the old per-record pull-in re-centering.
    doppler_hz/carrier_hz kept in the signature for call-site stability; unused."""
    del doppler_hz, carrier_hz  # residual convention: geometry is already fed forward
    return code_bias * chip_hz / hops_per_sec


def expand_token(tok):
    """Expand the first bash-style {a..b} range in a token, recursing for more.

    Zero-pads to the operand width iff either operand is written zero-padded
    (e.g. {00..49} -> 00..49, but {0..49} -> 0..49), matching shell brace ranges.
    """
    m = re.search(r"\{(\d+)\.\.(\d+)\}", tok)
    if not m:
        return [tok]
    lo, hi = m.group(1), m.group(2)
    padded = (len(lo) > 1 and lo[0] == "0") or (len(hi) > 1 and hi[0] == "0")
    width = max(len(lo), len(hi)) if padded else 0
    a, b = int(lo), int(hi)
    step = 1 if b >= a else -1
    out = []
    for i in range(a, b + step, step):
        out.append(tok[:m.start()] + str(i).zfill(width) + tok[m.end():])
    res = []
    for o in out:  # handle any further ranges in the same token
        res.extend(expand_token(o))
    return res


def resolve_prefix(entry, default_base):
    """Endpoint prefix (everything before /<verb>) for a list entry.

    Absolute http(s) entries are used as-is; bare names hang off --rest-url. Bare names
    are ALIAS-RESOLVED against the live pipeline (gnss_stages): the tri-constellation
    configs name the GPS chain gps_search/gps_track/gps_combiner, matching gal_*/bds_*,
    while the older single-constellation benches still use search/track/combiner -- either
    spelling works against either config.
    """
    entry = entry.strip()
    if entry.startswith("http://") or entry.startswith("https://"):
        return entry.rstrip("/")
    return default_base.rstrip("/") + "/" + resolve_stage(default_base,
                                                          entry.strip("/"))


def parse_endpoints(csv, default_base):
    """Comma list -> resolved endpoint prefixes, with {a..b} ranges expanded."""
    prefixes = []
    for raw in csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        for tok in expand_token(raw):
            prefixes.append(resolve_prefix(tok, default_base))
    return prefixes


def brdc_predict(state, lat, lon, alt_m, sysc, min_prn, t_utc, f_carrier_hz):
    """BRDC-almanac analog of gps_beamtrack.predict_dopplers for ONE constellation:
    {prn: (doppler_hz, doppler_rate_hz_s, elevation_deg, range_m, sat_clk_s)}.

    Same tuple layout/sign convention as the TLE path (+Doppler = approaching), extended
    with the broadcast SATELLITE CLOCK (the TLE class has none) -- the nh time-assist
    convention is t_sv = t_gpst - range/c + clk (proven to 0.01 chip, c31_convention.py).
    All elevations are returned (down to -90): the noise probes pick the deepest-BELOW-
    horizon sats. Rate = range-rate differencing over a 4 s epoch pair, like the
    dead-reckon block. Full-sky predict_all is ~1.2 ms, so per-cycle cost is trivial
    (the skyfield path it replaces was heavier).
    """
    ge = state["mod"]
    now = time.time()
    if state["eph"] is None or now - state["eph_t"] > 7200.0:
        # Current-day BRDC files GROW; fetch_brdc re-fetches a cache older than 2 h. Pass
        # t_utc so a replay (--almanac-epoch) gets the DAY-MATCHED file -- today's file
        # cannot predict another epoch (best_eph 4 h window).
        try:
            state["eph"] = ge.parse_rinex_nav(ge.fetch_brdc(t_utc))
            state["eph_t"] = now
            _log("brdc almanac: ephemeris refreshed (%d sats)" % len(state["eph"]))
        except Exception as e:
            state["eph_t"] = now - 7200.0 + 600.0  # coast on the old set, retry in 10 min
            if state["eph"] is None:
                raise
            _log("brdc almanac: refresh failed (%s); coasting on the previous set "
                 "(sats thin out as toe ages past 4 h)" % e)
    dt = 4.0
    pa = ge.predict_all(state["eph"], lat, lon, alt_m, t_utc, mask_deg=-90.0)
    pb = ge.predict_all(state["eph"], lat, lon, alt_m, t_utc + timedelta(seconds=dt),
                        mask_deg=-90.0)
    C = 299792458.0
    out = {}
    for (s, prn), v in pa.items():
        if s != sysc or prn < min_prn:
            continue
        dop = -v["range_rate_mps"] / C * f_carrier_hz
        v2 = pb.get((s, prn))
        rate = (-(v2["range_rate_mps"] - v["range_rate_mps"]) / dt / C * f_carrier_hz
                if v2 else 0.0)
        out[prn] = (dop, rate, v["el"], v["range_m"], v["sat_clk_s"])
    return out


def visible_prns(lat, lon, alt_m, mask_deg, look_ahead_s):
    """PRNs above the elevation mask now (and look_ahead_s ahead). Needs skyfield."""
    try:
        from gps_beamtrack import load_gps_sats, sat_elevation  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        _log("visibility gating unavailable (%s); not gating" % e)
        return None
    sats = load_gps_sats()
    up = set()
    for prn, sat in sats.items():
        if sat_elevation(sat, lat, lon, alt_m, look_ahead_s) >= mask_deg:
            up.add(prn)
    return up


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rest-url", default="http://localhost:12048",
                    help="default kotekan REST base for bare stage names")
    ap.add_argument("--detectors", "--searches", dest="detectors", required=True,
                    help="detection endpoints: aggregator/search stage names, absolute "
                         "URLs, or {a..b} ranges (e.g. aggregate  or  search_{00..49})")
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
    ap.add_argument("--trim-precomp", action="store_true",
                    help="EXPERIMENTAL (2026-07-12 night, default OFF pending bench "
                         "validation): pre-shift the carrier trim by -ddop at seed-doppler "
                         "steps and currency-translate coasted cp on forecast updates. "
                         "First on-sky deploy correlated with an E/C carrier collapse on "
                         "high-flap-rate sats (deep med 1-4 at healthy amp; GPS unaffected) "
                         "-- suspected sign/reference error; validate on the replay bench "
                         "before re-enabling.")
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
    ap.add_argument("--drop-hits", type=int, default=8,
                    help="(superseded by --coast-budget) old consecutive-low-|A|-polls drop count")
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
    ap.add_argument("--constellation", default=None, choices=("G", "E", "C"),
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
    ap.add_argument("--dll-leak", type=float, default=0.05,
                    help="DLL integrator leak (0 = pure integrator): trim mean-reverts each "
                         "update so discriminator NOISE can't random-walk it to the clamp. DC "
                         "loop gain = dll_gain/dll_leak; ~1/leak windows of smoothing.")
    ap.add_argument("--cl-assist", action="store_true",
                    help="trackers despread the L2C CL pilot: lift each seed's code_phase_chips by "
                         "k*10230 with the CL segment k COMPUTED from absolute capture time (the "
                         "airspy /adcstat utc0_sample0 anchor) + almanac range. CL's 1.5 s epoch is "
                         "GPS-time-locked, so k is arithmetic, not a 75-way search (which is "
                         "SNR-starved per-record in a narrow subband). Needs --almanac.")
    ap.add_argument("--adc-stage", default="airspy_in",
                    help="airspy input stage name for the utc0_sample0 anchor GET (CL assist)")
    ap.add_argument("--cl-time-adjust", type=float, default=0.0,
                    help="seconds added to the CL time-assist clock -- escape hatch for a future "
                         "non-multiple-of-1.5s GPS-UTC offset or a known host-clock bias")
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
    ap.add_argument("--almanac-epoch", type=float, default=0.0,
                    help="REPLAY BENCH ONLY: unix time to evaluate the almanac at, instead of "
                         "now() -- so a recorded capture's real sky (ranges, visibility) drives "
                         "the seeds + nh-assist. 0 = live (use now).")
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
                         "C11/C12/C13 produced 11309 phantom rows (5.5% of all BeiDou map "
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
                    help="DESIGN (b), default OFF -- IT DOES NOT WORK YET, AND THE REASON IS "
                         "INSTRUCTIVE. Measured 2026-07-14: continuous Doppler made E/C WORSE "
                         "than the fence+translate design (E 42.0 -> 34.9 dB-Hz, degraded "
                         "emits 12% -> 51%; C 40.8 -> 35.6, 17% -> 48%). Freezing the seed "
                         "Doppler was doing DOUBLE DUTY: it also kept the TRACKER's f_ref "
                         "stable. Let the Doppler drift continuously and the tracker's OWN "
                         "fence (fll_reacq_hz, 10 Hz at B1C) fires every 10/0.55 = 18 s -- the "
                         "identical cadence, merely relocated into the tracker, where the "
                         "re-pin is not translation-protected. (b) needs the tracker's f_ref "
                         "re-pin made code- and phase-continuous FIRST. Until then the fence "
                         "stays, and the translation on its step is what buys the +7.8 dB.\n"
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
    ap.add_argument("--once", action="store_true",
                    help="run a single control-loop iteration and exit (for tests)")
    args = ap.parse_args(argv)

    base = args.rest_url.rstrip("/")
    detectors = parse_endpoints(args.detectors, base)
    trackers = parse_endpoints(args.trackers, base)
    combiner = resolve_prefix(args.combiner, base)
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
            when = (datetime.fromtimestamp(args.almanac_epoch, tz=timezone.utc)
                    if args.almanac_epoch else datetime.now(timezone.utc))
            brdc_alm = {"mod": _alm_eph_mod,
                        "eph": _alm_eph_mod.parse_rinex_nav(_alm_eph_mod.fetch_brdc(when)),
                        "eph_t": time.time()}
            n = sum(1 for k in brdc_alm["eph"] if k[0] == alm_sys and k[1] >= alm_min_prn)
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

    dr_state = None
    if args.dead_reckon:
        if not args.almanac:
            _log("--dead-reckon needs --almanac; disabling")
        else:
            try:
                import gnss_ephemeris as dr_eph_mod
                dr_state = {"eph": None, "eph_t": 0.0, "t0m": None, "clk": None,
                            "clk_t": 0.0, "next": 0.0, "log_next": 0.0,
                            "pin": {}, "seeded": set()}
                _log("dead-reckon: BRDC cp seeding armed (%s, repin %.0f s%s)"
                     % (args.dr_constellation, args.dr_repin_s,
                        ", DRY RUN" if args.dr_dry_run else ""))
            except Exception as e:
                _log("dead-reckon unavailable (%s); disabled" % e)

    seeds = {}       # prn -> {"doppler_hz", "code_phase_chips", ...} (consensus)
    low_hits = {}    # prn -> consecutive low-|A| poll count
    status = {}      # prn -> last combiner get_status record (previous cycle; lock gate below)
    cp_held = set()  # PRNs whose cp anchor is FROZEN this cycle (locked -> DLL owns the residual)
    utc0_sample0 = 0.0  # CL time-assist: wall UTC of capture sample 0 (fetched lazily in the loop)
    dll_trim = {}       # prn -> persistent cp trim (chips) from the E/L delay-lock loop
    dll_last = {}       # prn -> last integrated disc (dedup: one integration per emit)
    cp_translated = set()  # PRNs whose first currency translation has been logged (once each)
    dop_clamped = set()    # PRNs that have tripped the one-cycle Doppler rate limit
    dr_untrusted = {}      # prn -> reason: the model is WRONG for this sat; use the search
    dr_bad = {}            # prn -> consecutive model-health failures (persistence, not a hair trigger)
    cp_escape = {}      # prn -> consecutive track-vs-search cp disagreements (hold referee)
    cp_escape_sign = {} # prn -> last disagreement (sign-consistency: real parks are one-signed)
    hold_miss = {}      # prn -> consecutive sub-gate status reads while held (blank-poll rides)
    car_trim = {}       # prn -> persistent NCO frequency command (Hz): the shared carrier loop
    car_last = {}       # prn -> last integrated residual (dedup: one integration per emit)
    car_locked = set()  # prns certified coherent since seed: BOOTSTRAP -> TRACK mode latch
    cp_hist = {}     # prn -> [(ref_hop, cp0, dop_det), ...] recent distinct snapshots (slope fit)

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
    CODE_LEN = float(args.code_length)
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
        t0 = time.time()
        # 1. collect best-SNR detection per PRN across all detection sources
        best = {}  # prn -> (snr, dop, cp, ref_hop)
        for d_ep in detectors:
            try:
                dets = _get("%s/get_detections" % d_ep)
            except Exception as e:
                _log("get_detections %s failed: %s" % (d_ep, e))
                continue
            for d in dets:
                prn, snr = int(d["prn"]), float(d["snr"])
                if snr < args.acquire_snr:
                    continue
                if prn not in best or snr > best[prn][0]:
                    best[prn] = (snr, float(d["doppler_hz"]), float(d["code_phase_chips"]),
                                 int(d.get("ref_hop", 0)))

        # 2. orbit-predicted Doppler + visibility (almanac assist), else plain gate
        pred = {}          # prn -> (doppler_hz, rate_hz_s, elev_deg) [sign-applied]
        # Hold the last smoothed bias through detection dropouts (the TCXO didn't move).
        clock_bias = clock_bias_ema if clock_bias_ema is not None else 0.0
        up = None
        if args.almanac:
            try:
                t_pred = (datetime.fromtimestamp(args.almanac_epoch, tz=timezone.utc)
                          if args.almanac_epoch else datetime.now(tz=timezone.utc))
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
                    t_ref = args.almanac_epoch or time.time()
                    hints = [{"prn": int(p),
                              "nh": int(round((_nh_eph.gpst_of_utc(t_ref)
                                               - v[3] / _nh_eph.C_LIGHT
                                               + (v[4] if len(v) > 4 else 0.0)) / period))
                                    % args.nh_overlay_len}
                             for p, v in pred.items() if v[2] >= args.mask_deg]
                    if hints:
                        _post("%s/set_nh_hint" % combiner, hints)
                except Exception as e:
                    _log("nh-assist POST failed: %s" % e)
            # Common clock-frequency bias = median(measured - predicted) over detected
            # sats. A tight residual spread confirms the sign convention; a wild spread
            # (resid ~ -2x predicted) means flip --doppler-sign.
            resid = [best[p][1] - pred[p][0] for p in best if p in pred]
            if len(resid) >= args.bias_min_sats:
                # The per-cycle median is quantized to the 500 Hz search grid and jumps
                # hundreds of Hz as the detected-sat set flickers; the TRUE bias is a slow
                # TCXO drift. EMA-smooth it (sub-grid dither across sats/cycles averages
                # out the quantization) so every sat's seed Doppler is stable -- a jittery
                # common bias was wrecking coherent integration (residual carrier +-260 Hz).
                # GATED on --bias-min-sats: one sat's residual is unfalsifiable and a bad
                # one narrows the search into a self-locking deadlock (see the arg help);
                # below the gate the EMA holds its last multi-sat value (or stays unsolved
                # -> margins stay WIDE, which is exactly what lets more sats in).
                raw_bias = statistics.median(resid)
                clock_bias_ema = (raw_bias if clock_bias_ema is None
                                  else clock_bias_ema + args.bias_alpha * (raw_bias - clock_bias_ema))
                clock_bias = clock_bias_ema
            for p in sorted(best):
                if p in pred:
                    _log("PRN %d: meas %+.0f  pred %+.0f  resid %+.0f Hz (elev %.0f)"
                         % (p, best[p][1], pred[p][0], best[p][1] - pred[p][0], pred[p][2]))
            if len(resid) >= args.bias_min_sats:
                _log("clock-freq bias %+.0f Hz (raw %+.0f, %d sats, EMA a=%.2f) -> seeding "
                     "predicted Doppler" % (clock_bias, raw_bias, len(resid), args.bias_alpha))
            elif resid:
                _log("clock-freq bias %s (%d sat < --bias-min-sats %d: residual not trusted)"
                     % ("held %+.0f Hz" % clock_bias if clock_bias_ema is not None
                        else "UNSOLVED (wide margins)", len(resid), args.bias_min_sats))
        elif gating:
            up = visible_prns(args.lat, args.lon, args.alt, args.mask_deg, 0.0)

        # 2b. Almanac-narrow the SEARCH: push per-PRN predicted Doppler to the detectors so each
        # scans only doppler +- margin instead of its blind grid -- far cheaper + more sensitive,
        # and it's what lets the not-yet-detected sats be acquired without a full sweep. The margin
        # is WIDE until the common clock-freq bias is solved (the geometric Doppler is then offset
        # by the unknown TCXO), NARROW once a few sats pin it. Sent for all predicted+visible sats.
        if args.narrow_search and args.almanac and pred:
            margin = (args.search_margin_hz if clock_bias_ema is not None
                      else args.search_margin_wide_hz)
            hints = [dict(prn=p, doppler_hz=pred[p][0] + clock_bias, margin_hz=margin)
                     for p in sorted(pred) if (up is None or p in up)]
            pushed = 0
            for d_ep in detectors:
                try:
                    _post("%s/set_doppler_hints" % d_ep, hints)
                    pushed += 1
                except Exception as e:
                    _log("set_doppler_hints %s failed: %s" % (d_ep, e))
            _log("narrowed search: %d hints +-%d Hz (%s) -> %d/%d detectors"
                 % (len(hints), int(margin),
                    "bias solved" if clock_bias_ema is not None else "pre-solve wide",
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
        if (args.cl_assist or dr_state is not None) and not utc0_sample0:
            try:
                utc0_sample0 = float(
                    _get("%s/%s/adcstat" % (base, args.adc_stage)).get("utc0_sample0", 0.0))
                if utc0_sample0:
                    _log("CL time-assist: capture sample-0 UTC anchor %.3f" % utc0_sample0)
            except Exception as e:
                _log("CL time-assist: adcstat anchor unavailable (%s); retrying" % e)
        dr_pd = (dr_state or {}).get("pd") or {}
        dr_pd2 = (dr_state or {}).get("pd2") or {}
        for prn, (snr, dop, cp, ref_hop) in best.items():
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
                seed_dop = (args.doppler_sign * (-v_dr["range_rate_mps"] / 299792458.0
                                                 * args.carrier_hz) + clock_bias)
            # SEED-STEP ATTRIBUTION (2026-07-18, the one-grid-step NCO disease): any seed
            # doppler step > 10 Hz vs the sat's previous seed is loud, with its source --
            # a ~exact-doppler_step jump here is the smoking gun for a grid/quantization
            # slip upstream (the hint-anchored search grid was one such; fixed same day).
            _prev_sd = seeds.get(prn, {}).get("doppler_hz")
            if _prev_sd is None and args.almanac and clock_bias_ema is None:
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
            h = cp_hist.get(prn, [])
            if h and (ref_hop - h[-1][0]) > MAX_GAP_HOPS:
                h = []  # gap too large -> re-acquisition, old slope is stale
            if not h or ref_hop != h[-1][0]:
                h.append((ref_hop, cp, dop))
                h = h[-HIST_LEN:]
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
                                             / 299792458.0 * args.carrier_hz)
            elif args.almanac and prn in pred:
                seed["doppler_rate_hz_s"] = pred[prn][1]
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
                _log("PRN %d cp-fit: %.2f chips @ hop %d, slope %+.3f chips/s (%d pts, l-a %+.3f ppm)"
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
            if args.cl_assist and utc0_sample0 and args.almanac and prn in pred:
                tau = pred[prn][3] / 299792458.0
                cl_chips = (((utc0_sample0 - tau + args.cl_time_adjust) % 1.5)
                            * args.chip_rate_hz)
                cp_cm = seed["code_phase_chips"]
                k = int(round((cl_chips - cp_cm) / CODE_LEN))
                fine_ms = (cl_chips - cp_cm - k * CODE_LEN) / args.chip_rate_hz * 1e3
                seed["code_phase_chips"] = (cp_cm + (k % 75) * CODE_LEN) % (75 * CODE_LEN)
                cl_report.append("PRN %d k=%d fine %+.1f ms" % (prn, k % 75, fine_ms))
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
            # Count only SIGN-CONSISTENT disagreements: a real wrong-lobe park is
            # one-signed; search-fit noise on a weak sat alternates (first deploy:
            # weak GPS sats escaped every minute on -0.5/+1.3/-0.5 flip-flops, and
            # each escape re-injects the seed jitter + forces an overlay re-align).
            # FIT-QUALITY gate (2026-07-12 evening): only a trustworthy fit may accuse the
            # track -- >=6 history points (fit noise ~ per-fix/sqrt(n)) and a solid current
            # detection (2x the acquire gate). Ungated, weak-sat fit noise drove 627 escapes
            # in 2.7 h and each one re-anchored the seed (the churn behind the GPS "wobble").
            fit_trusted = (fit is not None and len(h) >= 6
                           and snr >= 2.0 * args.acquire_snr)
            # AMP VETO (see --escape-amp-veto): a full-amplitude hold is on the main peak
            # by construction -- refuse the fit's accusation rather than drag it off.
            amp_now = float((status.get(prn) or {}).get("amp_snr", 0) or 0)
            amp_veto = (args.escape_amp_veto > 0.0
                        and amp_now > args.escape_amp_veto)
            if (cp_err is not None and abs(cp_err) > args.hold_max_cp_err
                    and fit_trusted and not amp_veto):
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
                dll_trim.pop(prn, None)
                dll_last.pop(prn, None)
                cp_held.discard(prn)
                hold_miss.pop(prn, None)
            elif (prev is not None
                    and (sig_of_last(status.get(prn)) >= args.hold_snr
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
                    _log("RELEASE PRN %d: seed currency unfrozen (amp_snr %.1f, ddop %+.0f)"
                         % (prn, sig_of_last(status.get(prn)),
                            (seed["doppler_hz"] - prev["doppler_hz"]) if prev else 0.0))
                cp_held.discard(prn)
                hold_miss.pop(prn, None)
            # TRIM PRE-COMPENSATION (2026-07-12 night): the NCO trim holds (f_true - f_ref);
            # when the seed doppler (= the tracker's f_ref) steps by ddop, the required trim
            # shifts by exactly -ddop. Un-compensated, every 100 Hz GPS staleness release
            # forced the loop to re-absorb the step from scratch: a ~10 Hz-residual,
            # several-emit transient synchronized across sats (observed as constellation-
            # wide coh-0 waves every ~200 s). Pre-shifting the trim makes the step seamless.
            if args.trim_precomp and prev is not None and prn in car_trim:
                dstep = prev.get("doppler_hz", 0.0) - seed.get("doppler_hz", 0.0)
                if dstep != 0.0:
                    car_trim[prn] = max(-args.carrier_max_hz,
                                        min(args.carrier_max_hz, car_trim[prn] + dstep))
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
        except Exception as e:
            status = {}
            _log("get_status failed: %s" % e)
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
                if new_dop != old_dop and args.trim_precomp:
                    # CURRENCY-CORRECT the coast (2026-07-12 evening): cp0 is meaningful only
                    # in its doppler's currency -- updating the forecast dop WITHOUT
                    # re-expressing cp walks the despread by t_abs*f_chip*ddop/f_c, chips/Hz
                    # at soak age (the exact t_abs lever the code-currency rule forbids;
                    # this is why long coasts silently lost the code peak). Same
                    # translation as cp_to_seed_currency, at the seed's ref_hop.
                    t_abs = seeds[prn].get("ref_hop", 0) / args.hops_per_sec
                    seeds[prn]["code_phase_chips"] = (
                        seeds[prn].get("code_phase_chips", 0.0)
                        + t_abs * args.chip_rate_hz * args.code_doppler_sign
                          * (old_dop - new_dop) / args.carrier_hz) % CODE_LEN
                    seeds[prn]["doppler_hz"] = new_dop
                    if prn in car_trim:  # trim pre-compensation, same as the seed loop
                        car_trim[prn] = max(-args.carrier_max_hz,
                                            min(args.carrier_max_hz,
                                                car_trim[prn] + (old_dop - new_dop)))
                elif new_dop != old_dop:
                    seeds[prn]["doppler_hz"] = new_dop  # legacy coast update (no translation)
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
                and time.time() >= dr_state["next"]):
            now_w = time.time()
            dr_state["next"] = now_w + args.dr_refresh_s
            t_code = CODE_LEN / args.chip_rate_hz
            if dr_state["eph"] is None or now_w - dr_state["eph_t"] > 7200:
                try:
                    dr_state["eph"] = dr_eph_mod.parse_rinex_nav(dr_eph_mod.fetch_brdc())
                    dr_state["eph_t"] = now_w
                    dr_state["t0m"] = dr_eph_mod.gpst_of_utc(utc0_sample0) % t_code
                    _log("dead-reckon: BRDC loaded (%d sats)" % len(dr_state["eph"]))
                except Exception as e:
                    _log("dead-reckon: BRDC unavailable (%s); retry in 10 min" % e)
                    dr_state["eph_t"] = now_w - 7200 + 600
            if dr_state["eph"]:
                tag = args.dr_constellation
                t_now_abs = now_w - utc0_sample0
                la = (args.code_bias_force * 1e-6 if args.code_bias_force is not None
                      else (code_bias_ema or 0.0))
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
                for prn, (snr, dop, cp, ref_hop) in sorted(best.items()):
                    v = pd.get((tag, prn))
                    if v is None:
                        continue
                    t_i = ref_hop / args.hops_per_sec
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
                if len(offs) >= args.dr_min_sats:
                    ref = offs[0][1]
                    cen = sorted(((d - ref + CODE_LEN / 2) % CODE_LEN) - CODE_LEN / 2
                                 for _, d in offs)
                    raw = (cen[len(cen) // 2] + ref) % CODE_LEN
                    prev_raw = dr_state.get("raw_prev")
                    if prev_raw is not None and 0.5 < now_w - prev_raw[1] < 30.0:
                        d_est = (((raw - prev_raw[0] + CODE_LEN / 2) % CODE_LEN)
                                 - CODE_LEN / 2) / (now_w - prev_raw[1])
                        dr_state["drift"] = (d_est if dr_state.get("drift") is None
                                             else dr_state["drift"]
                                             + 0.05 * (d_est - dr_state["drift"]))
                    dr_state["raw_prev"] = (raw, now_w)
                    if dr_state["clk"] is None:
                        dr_state["clk"] = raw
                        _log("dead-reckon: receiver clock BOOTSTRAP %.2f chips = %.3f us "
                             "(mod %.0f ms; %d sats)"
                             % (raw, raw / args.chip_rate_hz * 1e6, t_code * 1e3, len(offs)))
                    else:
                        clk = (dr_state["clk"]
                               + drift * (now_w - dr_state["clk_t"])) % CODE_LEN
                        step = ((raw - clk + CODE_LEN / 2) % CODE_LEN) - CODE_LEN / 2
                        dr_state["clk"] = (clk + args.dr_clock_alpha * step) % CODE_LEN
                    dr_state["clk_t"] = now_w
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
                    for prn_i, d_i in offs:
                        r_i = (((d_i - dr_state["clk"] + CODE_LEN / 2.0) % CODE_LEN)
                               - CODE_LEN / 2.0)
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
                    resid = ["PRN %d %+.2f%s" % (p, r, " BAD" if abs(r) > 1.0 else "")
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
                    planned = []
                    for (ctag, prn), v in sorted(pd.items()):
                        # SIGNAL CAPABILITY first (see --dr-min-prn): a satellite that does
                        # not broadcast this signal must never be seeded, however visible and
                        # however well-predicted it is. The model will happily hand us a code
                        # phase for a signal that isn't there.
                        if prn < dr_min_prn:
                            continue
                        # +0.5 deg hysteresis vs the drop mask: a sat riding the mask
                        # otherwise flickers drop->reseed every few cycles
                        if (ctag != tag or v["el"] < args.mask_deg + 0.5 or prn in best
                                or prn in probe_set or prn in cp_held
                                or prn in dr_untrusted):   # model is wrong for this sat
                            continue
                        if prn in seeds and prn not in dr_state["seeded"]:
                            continue  # search-anchored coast: not ours to touch
                        if (prn in seeds
                                and sig_of(status.get(prn, {})) >= args.lock_snr):
                            continue  # sub-threshold LOCK: the DLL owns the residual now
                        if (prn in dr_state["seeded"]
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
                        cp0 = ((cp_predicted(v, t_now_abs) + clk_now)
                               - t_now_abs * args.chip_rate_hz
                                 * (1.0 + args.code_doppler_sign
                                    * dop_seed / args.carrier_hz)) % CODE_LEN
                        if args.dr_dry_run:
                            planned.append("PRN %d el %.0f cp0 %.1f dop %+.0f rate %+.2f"
                                           % (prn, v["el"], cp0, dop_seed, drate))
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
                        if prn < dr_min_prn:
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
            dll_report = []
            for prn in list(seeds):
                rec = status.get(prn, {})
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
                trim = (1.0 - args.dll_leak) * dll_trim.get(prn, 0.0) + args.dll_gain * tau
                dll_trim[prn] = max(-3.0, min(3.0, trim))
                dll_report.append("PRN %d disc %+.3f trim %+.2f" % (prn, disc, dll_trim[prn]))
            if dll_report:
                _log("DLL: " + "; ".join(dll_report))
            for k in list(dll_trim):
                if k not in seeds:
                    del dll_trim[k]

        # 3d. SHARED CARRIER LOOP (the carrier twin of 3c): integrate the combiner's full-band
        # cross-record phase-walk residual into a commanded NCO frequency per PRN. The residual
        # is measured AFTER the current trim (the NCO derotates before records ship), so the
        # plain integrator converges: trim += gain * resid. No lock gate: the observable is
        # vector-averaged over the emit window at FULL-BAND SNR (the whole point -- per-channel
        # amplitude gates would exclude exactly the weak-band cases this loop exists for);
        # the clamp bounds any noise walk.
        if args.carrier_gain > 0.0:
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
                resid = float(rec.get("carrier_hz_resid", 0.0))
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
                coh_ok = (rec.get("coherence_s") or 0.0) > 0.0
                sig = (max(rec.get("deep_snr") or 0.0, rec.get("amp_snr") or 0.0)
                       if coh_ok else 0.0)
                tracking = prn in car_locked
                if coh_ok and sig >= args.carrier_min_sig > 0.0:
                    car_locked.add(prn)
                if tracking and args.carrier_min_sig > 0.0 \
                        and (not coh_ok or sig < args.carrier_min_sig):
                    continue  # fade: hold the trim, coast on the feed-forward
                if tracking and args.carrier_innov_hz > 0.0 \
                        and abs(resid) > args.carrier_innov_hz:
                    continue  # certified-but-implausible residual: the estimator is lying
                # Integrate each MEASUREMENT once, not each poll: the combiner emits a new
                # residual every ~1.5 s window while this loop polls at ~5 Hz -- integrating
                # the stale value 5-7x per measurement over-applies the gain and oscillates
                # (observed +-20 Hz swings). A changed value marks a fresh emit.
                if resid == car_last.get(prn):
                    continue
                car_last[prn] = resid
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
            if car_report:
                _log("CAR: " + "; ".join(car_report))
            for k in list(car_trim):
                if k not in seeds:
                    del car_trim[k]
                    car_locked.discard(k)  # a re-seeded sat re-enters via BOOTSTRAP

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
                _log("code-rate clock offset (l-a) %+.3f ppm (raw %+.3f, %d fitted sats, EMA a=%.2f)"
                     % (code_bias_ema * 1e6, raw_cb * 1e6, len(la_samples), args.code_bias_alpha))
                if args.code_bias_file:
                    try:
                        with open(args.code_bias_file, "w") as f:
                            f.write("%.4f\n" % (code_bias_ema * 1e6))
                    except Exception:
                        pass
        cb_to_seed = (args.code_bias_force * 1e-6 if args.code_bias_force is not None
                      else code_bias_ema)
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
                _log("seeded code rate from (l-a) %+.3f ppm%s -> %d sat(s)"
                     % (cb_to_seed * 1e6,
                        " [FORCED]" if args.code_bias_force is not None else "", n_seeded))

        # 4. push consensus seeds to every tracker (DLL trim applied at POST time only)
        payload = []
        for prn, v in sorted(seeds.items()):
            d = dict(prn=prn, **v)
            if dll_trim.get(prn):
                d["code_phase_chips"] = d["code_phase_chips"] + dll_trim[prn]
            if car_trim.get(prn):
                d["carrier_trim_hz"] = car_trim[prn]
            payload.append(d)
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
        _log("active=%s (%d); seeded %d/%d trackers" % (sorted(seeds), len(seeds), ok, len(trackers)))
        if cl_report:
            _log("CL assist: " + "; ".join(cl_report))

        if args.once:
            return
        dt = args.interval - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    main()
