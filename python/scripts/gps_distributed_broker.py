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
import json
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone


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

    The whole signal is Doppler-scaled by v/c at the antenna, but the receiver observes the CODE
    against its ADC sample clock and the CARRIER against its tuner-LO clock:
        code_frac    = fitted code rate / chip rate   = v/c - a   (a = ADC clock error)
        carrier_frac = measured Doppler  / carrier    = v/c - l   (l = LO  clock error)
    so code_frac - carrier_frac = l - a -- v/c CANCELS, leaving the common clock offset with NO
    ephemeris needed. It is the same for every satellite, so strong detections calibrate it and weak
    ones inherit it (the code-side twin of the existing carrier clock_bias). ~+2.6 ppm on the airspy.
    """
    code_frac = rate_chips_per_hop * hops_per_sec / chip_hz
    carrier_frac = doppler_hz / carrier_hz
    return code_frac - carrier_frac


def cp_rate_from_code_bias(doppler_hz, code_bias, hops_per_sec, chip_hz, carrier_hz):
    """Seed the code-phase rate (chips/hop) for ANY sat from carrier-aiding + the calibrated offset:
    code_rate = f_chip * (doppler/f_L1 + (l - a)). Gives a not-yet-fittable weak/new sat an accurate
    code rate from its FIRST detection, instead of waiting to build its own >=3-snapshot slope fit."""
    code_frac = doppler_hz / carrier_hz + code_bias
    return code_frac * chip_hz / hops_per_sec


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

    Absolute http(s) entries are used as-is; bare names hang off --rest-url.
    """
    entry = entry.strip()
    if entry.startswith("http://") or entry.startswith("https://"):
        return entry.rstrip("/")
    return default_base.rstrip("/") + "/" + entry.strip("/")


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
    ap.add_argument("--code-length", type=float, default=1023.0,
                    help="spreading-code length (chips) for cp0 unwrap/fit (L1 C/A = 1023)")
    ap.add_argument("--hops-per-sec", type=float, default=125000.0,
                    help="F-engine hops/s (Fs/fft_len) -- cp slope chips/s log + the code clock-bias estimate")
    ap.add_argument("--chip-rate-hz", type=float, default=1.023e6,
                    help="spreading chip rate (L1 C/A 1.023e6) -- for the code-rate clock-bias estimate")
    ap.add_argument("--code-bias-alpha", type=float, default=0.05,
                    help="EMA weight for the receiver code-rate clock offset (l-a); slow -> tracks TCXO/OCXO drift")
    ap.add_argument("--code-bias-min-sats", type=int, default=2,
                    help="fitted sats needed before the pooled code-rate clock offset is trusted + seeded to weak sats")
    ap.add_argument("--code-bias-init", type=float, default=None,
                    help="warm-start the receiver code-rate clock offset (l-a) in PPM, e.g. from a prior "
                         "strong-signal (L1 C/A) run -- so a weak band (L1C) seeds on-peak from cycle 1 "
                         "instead of self-calibrating. Live samples still refine it if any sats fit.")
    ap.add_argument("--code-bias-file", type=str, default=None,
                    help="persist the converged (l-a) ppm here: read at startup (unless --code-bias-init "
                         "is set) and rewritten each update, so the offset carries across runs/bands")
    ap.add_argument("--fit-gap-s", type=float, default=16.0,
                    help="reset the cp-fit history across a detection gap longer than this "
                         "(seconds of capture time; converted via --hops-per-sec)")
    ap.add_argument("--dll-gain", type=float, default=0.25,
                    help="code delay-lock-loop gain (0 = off): each poll, nudge a persistent "
                         "per-PRN cp TRIM by gain * tau_est from the combiner's E/L discriminator. "
                         "The trim rides on top of the search-fit cp, converging to the fit's "
                         "grid-quantization bias -- sub-chip code tracking with no per-record "
                         "decisions (R1, docs/gnss_architecture_audit.md).")
    ap.add_argument("--dll-spacing", type=float, default=0.5,
                    help="tracker Early/Late spacing in chips (must match dll_spacing_chips)")
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
    ap.add_argument("--once", action="store_true",
                    help="run a single control-loop iteration and exit (for tests)")
    args = ap.parse_args(argv)

    base = args.rest_url.rstrip("/")
    detectors = parse_endpoints(args.detectors, base)
    trackers = parse_endpoints(args.trackers, base)
    combiner = resolve_prefix(args.combiner, base)
    gating = args.lat is not None and args.lon is not None

    # Almanac assist: load the GPS constellation once (TLEs), predict per-cycle.
    almanac_sats = None
    if args.almanac:
        if not gating:
            _log("--almanac needs --lat/--lon; disabling")
            args.almanac = False
        else:
            try:
                from gps_beamtrack import load_gps_satellites, predict_dopplers
                from gps_beamtrack import DEFAULT_TLE_URL
                almanac_sats = load_gps_satellites(args.tle or DEFAULT_TLE_URL)
                _log("almanac: loaded %d GPS TLEs; predicting Doppler @ (%.4f, %.4f)"
                     % (len(almanac_sats), args.lat, args.lon))
            except Exception as e:
                _log("almanac unavailable (%s); falling back to search Doppler" % e)
                args.almanac = False

    seeds = {}       # prn -> {"doppler_hz", "code_phase_chips", ...} (consensus)
    low_hits = {}    # prn -> consecutive low-|A| poll count
    utc0_sample0 = 0.0  # CL time-assist: wall UTC of capture sample 0 (fetched lazily in the loop)
    dll_trim = {}       # prn -> persistent cp trim (chips) from the E/L delay-lock loop
    cp_hist = {}     # prn -> [(ref_hop, cp0), ...] recent distinct snapshots (for the slope fit)
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
                from gps_beamtrack import predict_dopplers
                raw = predict_dopplers(args.lat, args.lon, args.alt,
                                       t_utc=datetime.now(tz=timezone.utc), _sats=almanac_sats,
                                       f_carrier_hz=args.carrier_hz)
                # doppler_sign flips to the receiver's observed convention -- apply it to BOTH the
                # Doppler and its rate so the 2nd-order feed-forward ramps the right way. (Range
                # is geometry, no sign; it feeds the CL time-assist propagation delay.)
                pred = {p: (args.doppler_sign * d, args.doppler_sign * r, e, rng)
                        for p, (d, r, e, rng) in raw.items()}
            except Exception as e:
                _log("predict_dopplers failed: %s" % e)
            up = {p for p, v in pred.items() if v[2] >= args.mask_deg}
            # Common clock-frequency bias = median(measured - predicted) over detected
            # sats. A tight residual spread confirms the sign convention; a wild spread
            # (resid ~ -2x predicted) means flip --doppler-sign.
            resid = [best[p][1] - pred[p][0] for p in best if p in pred]
            if resid:
                # The per-cycle median is quantized to the 500 Hz search grid and jumps
                # hundreds of Hz as the detected-sat set flickers; the TRUE bias is a slow
                # TCXO drift. EMA-smooth it (sub-grid dither across sats/cycles averages
                # out the quantization) so every sat's seed Doppler is stable -- a jittery
                # common bias was wrecking coherent integration (residual carrier +-260 Hz).
                raw_bias = statistics.median(resid)
                clock_bias_ema = (raw_bias if clock_bias_ema is None
                                  else clock_bias_ema + args.bias_alpha * (raw_bias - clock_bias_ema))
                clock_bias = clock_bias_ema
            for p in sorted(best):
                if p in pred:
                    _log("PRN %d: meas %+.0f  pred %+.0f  resid %+.0f Hz (elev %.0f)"
                         % (p, best[p][1], pred[p][0], best[p][1] - pred[p][0], pred[p][2]))
            if resid:
                _log("clock-freq bias %+.0f Hz (raw %+.0f, %d sats, EMA a=%.2f) -> seeding "
                     "predicted Doppler" % (clock_bias, raw_bias, len(resid), args.bias_alpha))
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
        # CL time-assist anchor: wall-clock UTC of capture sample 0 (airspy stamps it at its
        # first USB callback; /adcstat serves it). 0.0 until the stream starts -- retry lazily.
        if args.cl_assist and not utc0_sample0:
            try:
                utc0_sample0 = float(
                    _get("%s/%s/adcstat" % (base, args.adc_stage)).get("utc0_sample0", 0.0))
                if utc0_sample0:
                    _log("CL time-assist: capture sample-0 UTC anchor %.3f" % utc0_sample0)
            except Exception as e:
                _log("CL time-assist: adcstat anchor unavailable (%s); retrying" % e)
        for prn, (snr, dop, cp, ref_hop) in best.items():
            if up is not None and prn not in up:
                continue
            seed_dop = (pred[prn][0] + clock_bias) if (args.almanac and prn in pred) else dop

            # Maintain a per-PRN cp0-vs-hop history (only distinct snapshots; the search
            # holds its detection between updates) and fit the first-order code drift.
            h = cp_hist.get(prn, [])
            if h and (ref_hop - h[-1][0]) > MAX_GAP_HOPS:
                h = []  # gap too large -> re-acquisition, old slope is stale
            if not h or ref_hop != h[-1][0]:
                h.append((ref_hop, cp))
                h = h[-HIST_LEN:]
            cp_hist[prn] = h

            seed = {"doppler_hz": seed_dop, "code_phase_chips": cp,
                    "code_phase_rate": 0.0, "ref_hop": ref_hop}
            # 2nd-order carrier feed-forward: hand the tracker the almanac Doppler RATE (Hz/s, sign-
            # applied like doppler_hz) so it ramps the replica frequency and the deep-integration
            # residual stays flat even at zenith (max Doppler acceleration). Anchored at ref_hop.
            if args.almanac and prn in pred:
                seed["doppler_rate_hz_s"] = pred[prn][1]
            fit = fit_cp_rate(h, CODE_LEN)
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
                if snr >= args.acquire_snr:
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
            return max(float(r.get("deep_snr", 0)), float(r.get("amp_snr", 0)))
        have_sig = any(sig_of(r) > 0 for r in status.values())
        for prn in list(seeds):
            if up is not None and prn not in up:
                _log("drop PRN %d (set below horizon)" % prn)
                del seeds[prn]
                low_hits.pop(prn, None)
                continue
            if prn in best:  # re-detected -> re-anchored in the seed loop above (coast reset there)
                continue
            # not re-detected this poll but still visible -> COAST: forecast the Doppler forward.
            if args.almanac and prn in pred:
                seeds[prn]["doppler_hz"] = pred[prn][0] + clock_bias
            rec = status.get(prn, {})
            if have_sig:
                metric, thresh = sig_of(rec), args.lock_snr
            else:
                metric, thresh = float(rec.get("amplitude", 0.0)), args.drop_amplitude
            if metric >= thresh:
                low_hits[prn] = 0  # lock holding through the dropout -> reset coast
            else:
                low_hits[prn] = low_hits.get(prn, 0) + 1
                if low_hits[prn] >= coast_polls:
                    _log("drop PRN %d (coast %.0fs expired, %s=%.2f)"
                         % (prn, args.coast_budget, "sig" if have_sig else "|A|", metric))
                    del seeds[prn]
                    low_hits.pop(prn, None)

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
                tau = -max(-1.0, min(1.0, disc)) / 4.0 * (args.dll_spacing / 0.5)
                dll_trim[prn] = max(-3.0, min(3.0, dll_trim.get(prn, 0.0) + args.dll_gain * tau))
                dll_report.append("PRN %d disc %+.3f trim %+.2f" % (prn, disc, dll_trim[prn]))
            if dll_report:
                _log("DLL: " + "; ".join(dll_report))
            for k in list(dll_trim):
                if k not in seeds:
                    del dll_trim[k]

        # 3b. Receiver code-rate clock offset (l-a): pool the strong sats' per-fit samples (robust
        # median), EMA-smooth it (slow -> tracks TCXO/OCXO drift), then SEED it to every sat that
        # could NOT fit its own slope (weak / just-acquired / coasting) as carrier-aiding + (l-a).
        # The code-side twin of the carrier clock_bias: strong detections calibrate the clock so weak
        # ones never drift off-peak. Ephemeris-free (v/c cancels). Bounded +-50 ppm against a rogue fit.
        if len(la_samples) >= args.code_bias_min_sats:
            raw_cb = statistics.median(la_samples)
            if -50e-6 < raw_cb < 50e-6:
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
        if code_bias_ema is not None:
            n_seeded = 0
            for prn, seed in seeds.items():
                if prn not in fitted:  # weak/new/coasting -> carrier-aiding + the calibrated offset
                    seed["code_phase_rate"] = cp_rate_from_code_bias(
                        seed["doppler_hz"], code_bias_ema, args.hops_per_sec,
                        args.chip_rate_hz, args.carrier_hz)
                    n_seeded += 1
            if n_seeded:
                _log("seeded code rate from (l-a) %+.3f ppm -> %d unfitted sat(s)"
                     % (code_bias_ema * 1e6, n_seeded))

        # 4. push consensus seeds to every tracker (DLL trim applied at POST time only)
        payload = []
        for prn, v in sorted(seeds.items()):
            d = dict(prn=prn, **v)
            if dll_trim.get(prn):
                d["code_phase_chips"] = d["code_phase_chips"] + dll_trim[prn]
            payload.append(d)
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
