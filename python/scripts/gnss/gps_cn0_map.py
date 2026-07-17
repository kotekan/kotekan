#!/usr/bin/env python3
"""C/N0 beam maps from the gps_status_logger JSONL -- TWO observables per emit, mapped
identically (no per-emit thresholding, linear pixel average FIRST, dB last, per-tag
below-horizon pedestal):

INCOHERENT (the commercial-receiver observable; unconditional -- needs only 1 record of
coherence, survives any carrier-loop state):
    x = s^2/N = (unbiased_amplitude^2) / (amplitude^2 - unbiased_amplitude^2)
    density y = (x - pedestal) / T_rec                          [Hz]

COHERENT (deep_pow_hz, exported since 2026-07-12; the fixed-FULL-window noise-debiased
coherent power -- NO ladder, NO detection gate, mean ~0 on noise so the pixel average is
unbiased at arbitrarily weak signal; coherent integration buys sqrt(T_coh/T_rec) =
~15 dB (GPS) / 12 (E1C) / 10 (B1C) deeper sidelobes per unit dwell. GPS is nav-bit
limited; the E/C pilots are the deep-sidelobe workhorses):
    density y = deep_pow_hz - pedestal                          [Hz]

Both pedestals are calibrated per constellation per epoch from BELOW-HORIZON sats
(elevation < mask via skyfield). Zero-point convention matches the pipeline (~ +3 dB of
true C/N0). Coherent products carry a _coh suffix; the coh/inc pixel ratio is a
systematics map (carrier-loop health convolved with nothing = 0 dB everywhere).

Usage:
  python3 gps_cn0_map.py            # auto-loads the three /tmp/gpswipe status logs
  python3 gps_cn0_map.py G=... E=... C=...
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # gps_beamtrack sibling

# Per-constellation signal geometry (must match the live configs): record period T_rec and
# combiner integration_length K. x = s2/N is a PER-RECORD power ratio, so cross-constellation
# merging happens in C/N0-density units, each tag carrying its OWN pedestal/noise scale.
_TLE = {"G": None,  # None -> gps_beamtrack.DEFAULT_TLE_URL
        "E": "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
        "C": "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"}
# Per-BAND, per-constellation geometry. The band matters: the same constellation carries a
# different signal (t_rec, integration_length K, name) on each tune, so a band-blind table
# mislabels the plot AND mis-scales C/N0 (x = s2/N is per RECORD -> a density needs the right
# t_rec). Mirrors livebeam_server.BAND_CHAINS / the live configs -- keep them in step.
BANDS = {
    "l1":  {"G": dict(name="GPS L1 C/A",   t_rec=1e-3,  K=1000),
            "E": dict(name="Galileo E1C",  t_rec=4e-3,  K=250),
            "C": dict(name="BeiDou B1C",   t_rec=10e-3, K=100)},
    "l2c": {"G": dict(name="GPS L2C-CM",   t_rec=20e-3, K=200)},
    "l5":  {"G": dict(name="GPS L5-Q",     t_rec=1e-3,  K=1000),
            "E": dict(name="Galileo E5a-Q", t_rec=1e-3, K=1000),
            "C": dict(name="BeiDou B2a-P", t_rec=1e-3,  K=1000)},
}
CONST = {}  # filled from BANDS[--band] in main()
# run_live's per-constellation status logger outputs, per band (recdir differs per band).
BAND_LOGS = {
    "l1":  {"G": "/tmp/gpswipe/status_log.jsonl",
            "E": "/tmp/gpswipe/status_log_gal.jsonl",
            "C": "/tmp/gpswipe/status_log_bds.jsonl"},
    "l2c": {"G": "/tmp/gps_l2c_gpu/status_log.jsonl"},
    "l5":  {"G": "/tmp/gps_l5_gpu/status_log.jsonl",
            "E": "/tmp/gps_l5_gpu/status_log_gal.jsonl",
            "C": "/tmp/gps_l5_gpu/status_log_bds.jsonl"},
}
DEFAULT_LOGS = BAND_LOGS["l1"]  # rebound in main() from --band


def load(path, tag):
    """One constellation's status log -> (prn, t, x, x_coh, dop). x = s2/N per record
    (dimensionless); x_coh = deep_pow_hz (already a density, Hz; NaN in pre-07-12 logs)."""
    K4 = CONST[tag]["K"] ** 0.25
    prn, t, x, xc, dop = [], [], [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = r.get("amplitude") or 0.0
        u = r.get("unbiased_amplitude") or 0.0
        if a > 0.0 and a > u:  # exact: N = a^2 - u^2 (new logs)
            v = (u * u) / (a * a - u * u)
        else:                   # fallback: invert the normalized significance (older logs)
            v = (r.get("amp_snr") or 0.0) / K4
        pc = r.get("deep_pow_hz")
        prn.append(int(r["prn"]))
        t.append(float(r["t"]))
        x.append(v)
        xc.append(float(pc) if pc is not None else np.nan)
        dop.append(r.get("doppler_hz") or 0.0)
    n = len(prn)
    return (np.array(prn, int), np.array(t, "<f8"), np.array(x, float),
            np.array(xc, float), np.array(dop, float))


def pedestal(vals, low, t, tbin_s):
    """Per-epoch pedestal + single-emit noise scale from below-horizon samples (fallback:
    a low percentile of everything -- signal-biased, flagged by the caller's printout)."""
    fin = np.isfinite(vals)
    glob = (float(np.median(vals[low & fin])) if (low & fin).sum() >= 200
            else float(np.nanpercentile(vals[fin], 20)) if fin.any() else 0.0)
    tb = ((t - t.min()) // tbin_s).astype(int) if len(t) else np.array([], int)
    ped_t = np.full((tb.max() + 1) if len(tb) else 1, glob)
    for b in np.unique(tb):
        sel = low & fin & (tb == b)
        if sel.sum() >= 100:
            ped_t[b] = np.median(vals[sel])
    if (low & fin).sum() >= 200:
        sig1 = float(np.std(vals[low & fin]))
    elif fin.any():
        v = vals[fin]
        sig1 = float(np.std(v[v < np.percentile(v, 50)]))
    else:
        sig1 = 1.0
    return ped_t[tb] if len(tb) else np.array([]), glob, sig1, (low & fin).sum() >= 200


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*",
                    help="status logs as TAG=PATH (G=..., E=..., C=...); bare paths are G. "
                         "Default: every existing " + ", ".join(DEFAULT_LOGS.values()))
    ap.add_argument("--band", default="l1", choices=sorted(BANDS),
                    help="which tune the logs are from: l1 (GPS C/A + E1C + B1C), l2c, or l5 "
                         "(L5-Q + E5a-Q + B2a-P). Sets the per-constellation signal names AND "
                         "the t_rec/K used to scale C/N0 -- a band-blind run mislabels and "
                         "mis-scales (2026-07-17).")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    ap.add_argument("--ped-mask", type=float, default=-5.0,
                    help="samples BELOW this elevation calibrate the noise pedestal (deg). "
                         "STRICTLY below the horizon (default -5): with coast-to-horizon the "
                         "0..10 deg population is real on-peak signal, and a signal-fed "
                         "pedestal imprints its wander on every sat (the 2026-07-13 morning "
                         "artifact). Noise probes supply the below-horizon samples.")
    ap.add_argument("--map-mask", type=float, default=5.0, help="map elevation mask (deg)")
    ap.add_argument("--sys", default=None,
                    help="constellations to map, e.g. 'E' or 'GE' (default: all present). "
                         "A single-constellation map is often far more legible -- the site "
                         "multipath, for instance, is unmistakable in Galileo alone, where it "
                         "lands in the COHERENT estimator (GPS hides it in the INCOHERENT one).")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filenames (keeps per-constellation maps apart)")
    ap.add_argument("--min-prn-c", type=int, default=19,
                    help="BeiDou signal-capability floor: B1C/B2a are BDS-3 only, so C1-C18 "
                         "(BDS-2, which broadcasts B1I at 1561 MHz -- outside our band) carry "
                         "no signal and any 'detection' of them is cross-correlation. Set to 1 "
                         "to disable the filter.")
    ap.add_argument("--tbin", type=float, default=60.0, help="time-series bin (s)")
    ap.add_argument("--naz", type=int, default=48)
    ap.add_argument("--nel", type=int, default=17)
    ap.add_argument("--min-bin", type=int, default=10, help="min emits per map cell")
    ap.add_argument("--dop-jump", type=float, default=5.0,
                    help="a seed-Doppler step above this (Hz) between consecutive emits marks a "
                         "RE-ANCHOR -- the code is briefly off-peak while the DLL re-trims")
    ap.add_argument("--churn-gate", type=float, default=2.0,
                    help="excise emits where the sat's LOCAL doppler-step rate (steps/min in "
                         "a +-60 s window) exceeds this. State-flagged and value-blind: high "
                         "churn marks the release/re-fit dither episodes that park BOC code "
                         "partially off-peak for tens of minutes (2026-07-13 morning: dips "
                         "showed 3 steps/min vs 0.6 healthy; -8 dB mean, 18x flutter). "
                         "0 disables.")
    ap.add_argument("--reanchor-pad", type=float, default=2.0,
                    help="exclude emits within this many seconds of a re-anchor (STATE-flagged, "
                         "value-blind: no selection bias on the map)")
    ap.add_argument("--ped-tbin", type=float, default=600.0,
                    help="pedestal time bin (s), per constellation")
    ap.add_argument("--residual", action="store_true",
                    help="also emit a residual beam map: each cell minus the azimuthal mean of its "
                         "elevation ring. Removes the (smooth) elevation response so the AZIMUTHAL "
                         "structure -- site multipath fringes -- stands out (diverging colour, 0=ring mean).")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args(argv)

    # ---- gather logs (composite: all constellations sharing the 1575.42 tune) ----
    # Repeat a tag to MERGE logs across nights (G=night1.jsonl G=night2.jsonl ...):
    # dwell accumulates instead of resetting per run; the per-epoch pedestal bins
    # handle the time gaps between sessions.
    global CONST
    CONST = BANDS[args.band]
    band_logs = BAND_LOGS[args.band]
    BANDLBL = args.band.upper()

    logs = {}
    if args.logs:
        bare = [s for s in args.logs if "=" not in s]
        if len(bare) > 1:
            # Bare paths all default to G. Handing THREE bare logs (G/E/C) silently pooled three
            # constellations into "GPS": PRNs collided (G1/E1/C1 are different birds), every sat
            # was positioned from the GPS TLE, and the plot was labelled GPS. That produced a
            # plausible-looking, meaningless map on 2026-07-17. Refuse it.
            ap.error("%d bare paths given -- bare = G, so these would all pool into GPS.\n"
                     "Tag them: %s" % (len(bare), " ".join(
                         "%s=%s" % (t, p) for t, p in zip("GEC", bare))))
        for spec in args.logs:
            tag, _, path = spec.rpartition("=")
            tag = tag or "G"
            if tag not in CONST:
                ap.error("tag %r is not in band %r (has %s)"
                         % (tag, args.band, "/".join(sorted(CONST))))
            logs.setdefault(tag, []).append(path)
    else:
        logs = {tag: [p] for tag, p in band_logs.items() if os.path.exists(p)}
    if not logs:
        ap.error("no status logs found")
    # honest title: name the constellations actually pooled, not a hardcoded "G+E+C"
    TAGLBL = "+".join(sorted(logs)) if len(logs) > 1 else CONST[list(logs)[0]]["name"]

    from gps_beamtrack import load_gps_satellites, DEFAULT_TLE_URL
    from skyfield.api import load as sky_load, wgs84
    ts = sky_load.timescale()
    obs = wgs84.latlon(args.lat, args.lon, elevation_m=args.alt)

    # ---- per-constellation: load, excise re-anchors, sky positions, pedestals -> densities ----
    TAG, PRN, T, ALT, AZ = [], [], [], [], []
    Y, YSIG, Y2, Y2SIG = [], [], [], []
    ped_ok = {}   # tag -> below-horizon pedestal was genuine (fallback poisons cross-cal)
    if args.sys:
        keep = set(args.sys.upper())
        logs = {k: v for k, v in logs.items() if k in keep}
        if not logs:
            print("no logs for --sys %s" % args.sys)
            return 1
        print("mapping constellations: %s" % ", ".join(sorted(logs)))
    for tag, paths in sorted(logs.items()):
        parts = [load(p_, tag) for p_ in paths]
        prn = np.concatenate([q[0] for q in parts])
        t = np.concatenate([q[1] for q in parts])
        x = np.concatenate([q[2] for q in parts])
        xc = np.concatenate([q[3] for q in parts])
        dop = np.concatenate([q[4] for q in parts])
        # SIGNAL CAPABILITY (2026-07-14): B1C is BDS-3 only. The BDS-2 birds (C1-C18) are real
        # satellites, genuinely overhead, with valid ephemerides -- they just broadcast B1I at
        # 1561 MHz, outside our band. Before the broker's dead-reckon seeder learned this, it
        # seeded them from the model, the tracker despread noise at the predicted phase, and
        # cross-correlation against the real B1C satellites reported 20-60 sigma at a
        # thoroughly plausible 25-30 dB-Hz: 5.5% of every BeiDou map point was a satellite
        # that was not transmitting. Old logs still carry those rows, so the map excludes them
        # here too -- a beam map is exactly the wrong place to discover you mapped a ghost.
        cap = np.ones(len(prn), bool) if tag != "C" else (prn >= args.min_prn_c)
        if (~cap).sum():
            print("%s: dropped %d rows from PRNs that do not broadcast this signal (< C%d)"
                  % (tag, (~cap).sum(), args.min_prn_c))
        prn, t, x, xc, dop = prn[cap], t[cap], x[cap], xc[cap], dop[cap]
        o = np.argsort(t)   # merged sessions must be time-ordered for the excision scan
        prn, t, x, xc, dop = prn[o], t[o], x[o], xc[o], dop[o]
        if len(x) == 0:
            print("%s: empty log(s) %s" % (tag, paths))
            continue
        # Sky positions FIRST: the excision below must know which rows are below-horizon
        # NOISE PROBES, because it must not touch them (see the spare-probes note there).
        sats = load_gps_satellites(_TLE[tag] or DEFAULT_TLE_URL)  # TLE is per-CONSTELLATION, not per-band
        alt = np.full(len(prn), np.nan)
        az = np.full(len(prn), np.nan)
        for p_ in np.unique(prn):
            sat = sats.get(int(p_))
            if sat is None:
                continue
            sel = prn == p_
            times = ts.from_datetimes(
                [datetime.fromtimestamp(u, tz=timezone.utc) for u in t[sel]])
            topo = (sat - obs).at(times).altaz()
            alt[sel] = topo[0].degrees
            az[sel] = topo[1].degrees
        ok = np.isfinite(alt)
        prn, t, x, xc, dop, alt, az = (prn[ok], t[ok], x[ok], xc[ok], dop[ok], alt[ok], az[ok])
        low = alt < args.ped_mask

        # state-flagged re-anchor excision (value-blind)
        keep = np.ones(len(x), bool)
        for p in np.unique(prn):
            idx = np.where(prn == p)[0]
            o = idx[np.argsort(t[idx])]
            if len(o) < 3:
                continue
            jumps = t[o][1:][np.abs(np.diff(dop[o])) > args.dop_jump]
            bad = np.zeros(len(o), bool)
            for tj in jumps:
                bad |= np.abs(t[o] - tj) <= args.reanchor_pad
            if args.churn_gate > 0 and len(jumps):
                # local step rate: steps within +-60 s of each emit, per minute
                rate = np.array([np.sum(np.abs(jumps - tt) <= 60.0) for tt in t[o]]) / 2.0
                bad |= rate > args.churn_gate
            keep[o[bad]] = False
        # NEVER excise the below-horizon NOISE PROBES. The excision exists to drop emits whose
        # CODE is off-peak during a re-anchor -- a probe has no peak to be off, and its samples
        # ARE the noise we are trying to measure. Worse, probes look maximally churny by
        # construction: the shared carrier loop has no lock gate, so on a signal-free probe it
        # integrates pure noise into the trim, the trim random-walks to its clamp, and the
        # REPORTED Doppler wanders with it. Measured 2026-07-14: the churn gate deleted 49075
        # of 49075 GPS probe rows and 49088 of 49088 Galileo ones -- 100%, silently -- so the
        # pedestal fell back to a signal-biased percentile and every C/N0 in the map inherited
        # the bias. The calibrator was destroyed by the loop chasing noise it should never have
        # been fed. (The broker now also skips probes in the carrier loop; this is the belt.)
        keep |= low
        n_cut = int((~keep).sum())
        prn, t, x, xc, alt, az, low = (prn[keep], t[keep], x[keep], xc[keep],
                                       alt[keep], az[keep], low[keep])
        t_rec = CONST[tag]["t_rec"]
        # incoherent: per-record ratio -> density (Hz)
        ped, glob, sig1, real_ped = pedestal(x, low, t, args.ped_tbin)
        ped_ok[tag] = real_ped
        y = (x - ped) / t_rec
        ysig = sig1 / t_rec
        print("%s (%s): %d emits (%d re-anchor-cut), inc ped x=%.4f%s, single-emit floor "
              "%.1f dB-Hz" % (tag, CONST[tag]["name"], len(x), n_cut, glob,
                              "" if real_ped else " (NO below-horizon sats: signal-biased "
                              "percentile fallback!)",
                              10 * np.log10(max(2 * sig1 / t_rec, 1e-9))))
        # coherent: deep_pow_hz is already a density (Hz)
        if np.isfinite(xc).any():
            ped2, glob2, sig2, real2 = pedestal(xc, low, t, args.ped_tbin)
            y2 = xc - ped2
            y2sig = sig2
            print("%s coherent: ped %.3f Hz%s, single-emit floor %.1f dB-Hz"
                  % (tag, glob2, "" if real2 else " (percentile fallback!)",
                     10 * np.log10(max(2 * sig2, 1e-9))))
        else:
            y2 = np.full(len(x), np.nan)
            y2sig = np.nan
            print("%s coherent: no deep_pow_hz in this log (pre-2026-07-12)" % tag)
        TAG.append(np.full(len(y), tag)); PRN.append(prn); T.append(t)
        ALT.append(alt); AZ.append(az)
        Y.append(y); YSIG.append(np.full(len(y), ysig))
        Y2.append(y2); Y2SIG.append(np.full(len(y), y2sig))
    tag = np.concatenate(TAG); prn = np.concatenate(PRN); t = np.concatenate(T)
    alt = np.concatenate(ALT); az = np.concatenate(AZ)
    y = np.concatenate(Y); ysig = np.concatenate(YSIG)
    y2 = np.concatenate(Y2); y2sig = np.concatenate(Y2SIG)
    sid = np.array([a + str(b) for a, b in zip(tag, prn)])   # "G12"/"E25"/"C19"
    span_h = (t.max() - t.min()) / 3600.0
    print("composite: %d emits, %d sats, span %.1f h" % (len(y), len(set(sid)), span_h))

    # CROSS-CONSTELLATION ZERO POINTS (item 6): per constellation, the map observable
    # carries direction-INDEPENDENT constants -- satellite EIRP convention, pilot power
    # fraction (E1C -3 dB of E1, B1C pilot 3/4), the antenna's ~4 dB SAW rolloff at the
    # BOC lobes -- that bias E/C low relative to GPS in the pooled composite while leaving
    # each constellation's beam SHAPE intact. Fit ONE dB offset per tag per observable
    # from sky cells where that tag and GPS both have enough emits (median of per-cell
    # dB differences -- robust to per-sat EIRP scatter), and apply it before pooling.
    def cross_cal(yv, ysv, label):
        m0 = (alt > args.map_mask) & np.isfinite(yv) & np.isfinite(ysv)
        az_e = np.linspace(0, 360, args.naz + 1)
        el_e = np.linspace(args.map_mask, 90, args.nel + 1)
        cell = {}
        ai = np.clip(np.digitize(az[m0], az_e) - 1, 0, args.naz - 1)
        ei = np.clip(np.digitize(alt[m0], el_e) - 1, 0, args.nel - 1)
        tg, yy = tag[m0], yv[m0]
        for t_, a_, e_, v_ in zip(tg, ai, ei, yy):
            cell.setdefault((a_, e_), {}).setdefault(t_, []).append(v_)
        out = np.ones(len(yv))
        for t_ in ("E", "C"):
            d = []
            for c, per in cell.items():
                if "G" in per and t_ in per and len(per["G"]) >= args.min_bin                         and len(per[t_]) >= args.min_bin:
                    mg, mx = np.mean(per["G"]), np.mean(per[t_])
                    if mg > 0 and mx > 0:
                        d.append(10 * np.log10(mg / mx))
            if len(d) >= 8:
                off = float(np.median(d))
                warn = "" if ped_ok.get("G", False) and ped_ok.get(t_, False) else \
                    "  [UNRELIABLE: pedestal fallback truncates the shared-cell selection]"
                print("cross-cal (%s): %s +%.2f dB from %d shared cells%s"
                      % (label, t_, off, len(d), warn))
                out[tag == t_] = 10 ** (off / 10.0)
            else:
                print("cross-cal (%s): %s skipped (%d shared cells < 8)"
                      % (label, t_, len(d)))
        return out
    y = y * cross_cal(y, ysig, "incoherent")
    if np.isfinite(y2).any():
        y2 = y2 * cross_cal(y2, y2sig, "coherent")

    def to_db(mean_y, mean_sig, nsamp):
        lim = 2.0 * mean_sig / np.sqrt(np.maximum(nsamp, 1))
        return np.where(mean_y > lim, 10.0 * np.log10(np.maximum(mean_y, 1e-12)), np.nan)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag_color = {"G": "tab:blue", "E": "tab:orange", "C": "tab:red"}
    t0 = t.min()
    written = []

    def plot_observable(yv, ysv, label, suffix):
        """The three standard products for one observable (density Hz + its noise scale)."""
        fin = np.isfinite(yv) & np.isfinite(ysv)
        if not fin.any():
            return
        # (1) per-sat C/N0 vs time (constellation-colored)
        fig, ax = plt.subplots(figsize=(11, 5))
        for s_ in sorted(set(sid)):
            m = (sid == s_) & fin
            if m.sum() < 20:
                continue
            tb = ((t[m] - t0) // args.tbin).astype(int)
            ub = np.unique(tb)
            my = np.array([yv[m][tb == b].mean() for b in ub])
            ms = np.array([ysv[m][tb == b].mean() for b in ub])
            ns = np.array([(tb == b).sum() for b in ub])
            ax.plot(ub * args.tbin / 3600.0, to_db(my, ms, ns), ".", ms=3,
                    color=tag_color[s_[0]], alpha=0.7)
        for k, c in tag_color.items():
            if k in set(tag):
                ax.plot([], [], "o", color=c, label=CONST[k]["name"])
        ax.set_xlabel("hours since %s UTC"
                      % datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))
        ax.set_ylabel("%s C/N0 (dB-Hz, est.)" % label)
        ax.set_title("%s: %s C/N0 (%.0f s bins) -- %.1f h, %d sats"
                     % (BANDLBL, label, args.tbin, span_h, len(set(sid))))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        f1 = os.path.join(args.outdir, "cn0_%s_vs_time%s.png" % (suffix, args.suffix))
        fig.savefig(f1, dpi=110)
        plt.close(fig)
        written.append(f1)

        # (2) elevation response, constellation-colored
        fig, ax = plt.subplots(figsize=(8, 5))
        eb = np.arange(args.map_mask, 90.1, 2.0)
        for s_ in sorted(set(sid)):
            m = (sid == s_) & (alt > args.map_mask) & fin
            if m.sum() < 50:
                continue
            idx = np.digitize(alt[m], eb) - 1
            ub = np.unique(idx)
            my = np.array([yv[m][idx == b].mean() for b in ub])
            ms = np.array([ysv[m][idx == b].mean() for b in ub])
            ns = np.array([(idx == b).sum() for b in ub])
            ax.plot(eb[ub] + 1.0, to_db(my, ms, ns), ".-", ms=4, lw=0.6,
                    color=tag_color[s_[0]], alpha=0.6)
        ax.set_xlabel("elevation (deg)")
        ax.set_ylabel("%s C/N0 (dB-Hz, est.)" % label)
        ax.set_title("Elevation response, %s -- %s" % (label, BANDLBL))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f2 = os.path.join(args.outdir, "cn0_%s_vs_elev%s.png" % (suffix, args.suffix))
        fig.savefig(f2, dpi=110)
        plt.close(fig)
        written.append(f2)

        # (3) COMPOSITE polar beam map, all constellations pooled per (az, el) cell
        m = (alt > args.map_mask) & fin
        az_e = np.linspace(0, 360, args.naz + 1)
        el_e = np.linspace(args.map_mask, 90, args.nel + 1)
        ai = np.clip(np.digitize(az[m], az_e) - 1, 0, args.naz - 1)
        ei = np.clip(np.digitize(alt[m], el_e) - 1, 0, args.nel - 1)
        ys, ss = yv[m], ysv[m]
        grid = np.full((args.nel, args.naz), np.nan)
        for i in range(args.nel):
            for j in range(args.naz):
                sel = (ei == i) & (ai == j)
                n = sel.sum()
                if n >= args.min_bin:
                    grid[i, j] = to_db(np.array([ys[sel].mean()]),
                                       np.array([ss[sel].mean()]), np.array([n]))[0]
        TH, RG = np.meshgrid(np.deg2rad(az_e), 90.0 - el_e)
        fig = plt.figure(figsize=(8.4, 7.5))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        vlo, vhi = (np.nanpercentile(grid, [2, 98]) if np.isfinite(grid).any() else (0, 1))
        pc = ax.pcolormesh(TH, RG, np.ma.masked_invalid(grid), cmap="turbo",
                           vmin=vlo, vmax=vhi, shading="flat")
        ax.set_rlim(0, 90)
        ax.set_rticks([30, 60, 90])
        ax.set_yticklabels(["60", "30", "0"])
        ax.set_title("%s: %s C/N0 beam map (%s; %.1f h, %d sats)"
                     % (BANDLBL, label, TAGLBL, span_h, len(set(sid))), pad=18)
        fig.colorbar(pc, ax=ax, label="C/N0 (dB-Hz, est.)", shrink=0.8, pad=0.09)
        fig.tight_layout()
        f3 = os.path.join(args.outdir, "cn0_%s_beammap%s.png" % (suffix, args.suffix))
        fig.savefig(f3, dpi=120)
        plt.close(fig)
        written.append(f3)
        if np.isfinite(grid).any():
            print("map (%s): %d cells; C/N0 %.1f .. %.1f dB-Hz"
                  % (label, int(np.isfinite(grid).sum()),
                     np.nanmin(grid), np.nanmax(grid)))

        # (4) RESIDUAL beam map: cell minus its elevation ring's azimuthal mean. Strips the smooth
        # elevation response, leaving the AZIMUTHAL structure = site multipath fringes (a reflector
        # localized in azimuth shows as a lobe here that the raw map buries under the el gradient).
        if args.residual and np.isfinite(grid).any():
            ring_mean = np.nanmean(grid, axis=1, keepdims=True)          # per-el-ring az mean
            resid = grid - ring_mean
            lim = np.nanpercentile(np.abs(resid), 98) if np.isfinite(resid).any() else 1.0
            fig = plt.figure(figsize=(8.4, 7.5))
            ax = fig.add_subplot(111, projection="polar")
            ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
            pc = ax.pcolormesh(TH, RG, np.ma.masked_invalid(resid), cmap="RdBu_r",
                               vmin=-lim, vmax=lim, shading="flat")
            ax.set_rlim(0, 90); ax.set_rticks([30, 60, 90]); ax.set_yticklabels(["60", "30", "0"])
            ax.set_title("Residual %s C/N0 (cell - el-ring az mean; %.1f h)" % (label, span_h), pad=18)
            fig.colorbar(pc, ax=ax, label="C/N0 - ring mean (dB)", shrink=0.8, pad=0.09)
            fig.tight_layout()
            f4 = os.path.join(args.outdir, "cn0_%s_resid%s.png" % (suffix, args.suffix))
            fig.savefig(f4, dpi=120); plt.close(fig); written.append(f4)
        return grid

    g_inc = plot_observable(y, ysig, "incoherent", "inc")
    g_coh = plot_observable(y2, y2sig, "coherent", "coh")

    # coh/inc consistency: per-sat median ratio (dB); 0 = loops honest, negative = the
    # coherent path lost signal the incoherent still sees (carrier/code-loop systematics).
    fin = np.isfinite(y2) & (y > 0) & (y2 > 0)
    if fin.any():
        print("coh/inc consistency (per-sat median, dB; ~0 = healthy):")
        for s_ in sorted(set(sid[fin])):
            m = fin & (sid == s_)
            if m.sum() < 20:
                continue
            r = 10 * np.median(np.log10(y2[m] / y[m]))
            print("  %-4s %+5.1f dB (n=%d)" % (s_, r, m.sum()))
    print("wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
