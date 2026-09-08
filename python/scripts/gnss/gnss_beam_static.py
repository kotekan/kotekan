#!/usr/bin/env python3
"""Publication-resolution beam maps and radial cuts from master cubes (gnss_beam_cube.py build).

    map     one PNG: the selected days x chains x channels x elements, projected around boresight
    radial  one PNG: radial cuts vs off-boresight angle, one curve per element or per channel
    gains   the per-element normalisation the cube implies, printed and plotted

WHY A SECOND TOOL. The viewer works at nside 16-32 because a browser has to download the cube;
a master is built at whatever nside the archive supports (128 is ~0.46 deg pixels, still ~8x
finer than the 3.5 deg main lobe) and nothing here is size-limited. Same accumulators, same
units, no re-reduction: this reads the SAME masters the viewer eats.

⚠️ EVERY AXIS IS ABSOLUTE. `--freqs` are absolute freq_ids, never per-chain bin indices, so
`--freqs 5988 --chains bds_b2a gal_e5a gps_l5` is ONE channel of the 1176 MHz band seen by
three chains -- not "bin 5988 of each list". `--elements` are element ids.

NORMALISATION -- THE PART THAT DECIDES WHETHER ELEMENTS ARE COMPARABLE
Each cube cell is already in PEDESTAL units: the sample was divided by that (element, channel,
5-min)'s own noise floor, so a cell is "power over this element's own noise in this channel".
That removes each element's RECEIVER scale but not its SKY response: a low-gain element sees a
weaker sky over the same noise, and the residual per-element scale is what `--elem-norm` fits.

  svd     (default) rank-1 fit of the LINEAR element x pixel matrix; each element's gain is its
          amplitude in the dominant mode. Uses every well-sampled pixel of the fit region, so
          it is driven by the whole pattern rather than by one bright pixel.
  svd-log the same fit in dB, i.e. a two-way (element + pixel) additive model. Insensitive to
          the main lobe dominating the fit; use it when the lobe is vetoed away or saturated.
  peak    divide by each element's own peak pixel. HERE FOR COMPARISON, and it is the reason
          this option list exists: with the main lobe vetoed out the peak sits on the ragged
          lobe EDGE, where a pixel's mean is built from few samples and swings by dB, so it
          normalises by the noise of one pixel and drives that scatter into every other pixel.
  none    no per-element gain; shows the raw spread between elements.

The gain vector is reported with the dominant mode's variance share -- if that share is low the
elements are NOT one pattern times a scale and no single number can normalise them.

@author Keith Vanderlinde
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_beam_cube import BORE_AZ, BORE_EL, angsep_deg  # noqa: E402

C_LIGHT = 299792458.0
DISH_M = 6.0


# ── loading ────────────────────────────────────────────────────────────────────────────────
def load_master(path, want_chains=None):
    """One master .npz -> {"day", "nside", "units", "pointing", "chains": {name: {...}}}."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    out = {"path": path, "day": meta["day"], "nside": int(meta["nside"]),
           "units": meta.get("units", "power"), "pointing": meta.get("pointing"), "chains": {}}
    for i, c in enumerate(meta["chains"]):
        if want_chains and c["chain"] not in want_chains:
            continue
        out["chains"][c["chain"]] = {
            "pix": z["pix_%d" % i], "n": z["n_%d" % i], "s1": z["s1_%d" % i],
            "freq_ids": [tuple(f) for f in c["freq_ids"]], "sys": c["sys"]}
    return out


def parse_freqs(spec):
    """"all" | "5988" | "5980-6000" | "5988,6000" -> None (=all) or a sorted set of freq_ids."""
    if not spec or spec == "all":
        return None
    ids = set()
    for part in str(spec).replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return ids


def sub_mask(freq_ids, want):
    """Boolean mask over a chain's subband axis: which bins cover a wanted absolute freq_id."""
    if want is None:
        return np.ones(len(freq_ids), bool)
    return np.array([any(lo <= f <= hi for f in want) for lo, hi in freq_ids], bool)


def parse_elems(spec, n_elem):
    if not spec or spec == "all":
        return list(range(n_elem))
    out = []
    for part in str(spec).replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [e for e in out if 0 <= e < n_elem]


# ── accumulation ───────────────────────────────────────────────────────────────────────────
def gather(masters, freqs, chain_offsets=None):
    """Sum the selected channels of every day and chain into per-element full-sky accumulators.

    Returns (N[E, npix] int64, S[E, npix] float64, nside, info). Pure addition on the stored
    accumulators -- days, chains and channels combine by summing, which is the whole reason the
    cube stores (n, s1) over linear power and never dB.
    """
    nside = masters[0]["nside"]
    n_elem = max(c["n"].shape[1] for m in masters for c in m["chains"].values())
    npix = 12 * nside * nside
    N = np.zeros((n_elem, npix), np.int64)
    S = np.zeros((n_elem, npix), np.float64)
    info = {"days": [], "chains": {}, "n_chan": {}}
    for m in masters:
        if m["nside"] != nside:
            sys.exit("master %s is nside %d, first is %d -- one nside per plot"
                     % (m["path"], m["nside"], nside))
        if m["units"] != masters[0]["units"] or m["pointing"] != masters[0]["pointing"]:
            sys.exit("master %s is %s/%s, first is %s/%s -- NEVER sum two currencies"
                     % (m["path"], m["units"], m["pointing"], masters[0]["units"],
                        masters[0]["pointing"]))
        info["days"].append(m["day"])
        for name, c in m["chains"].items():
            sm = sub_mask(c["freq_ids"], freqs)
            if not sm.any():
                continue
            g = 1.0 if not chain_offsets else 10.0 ** (-chain_offsets.get(name, 0.0) / 10.0)
            nn = c["n"][sm].sum(0)                    # (E, P) over the selected channels
            ss = c["s1"][sm].sum(0) * g
            e = nn.shape[0]
            np.add.at(N, (slice(0, e), c["pix"]), nn)
            np.add.at(S, (slice(0, e), c["pix"]), ss)
            info["chains"][name] = info["chains"].get(name, 0) + int(nn.sum())
            info["n_chan"][name] = int(sm.sum())
    return N, S, nside, info


def chain_offsets_from(masters, freqs, annulus, min_n, ref):
    """Per-chain dB offset onto `ref`, measured as the median level in the lobe annulus.

    The same alignment the exporter ships to the viewer, recomputed here over ALL the selected
    days at once (the exporter does it per day) so a multi-day map has one consistent zero.
    """
    lev = {}
    for name in sorted({n for m in masters for n in m["chains"]}):
        N, S, nside, _ = gather([{k: (v if k != "chains" else {name: v[name]})
                                  for k, v in m.items()} for m in masters if name in m["chains"]],
                                freqs)
        nn, ss = N.sum(0), S.sum(0)
        az, el = pix_azel(nside)
        th = angsep_deg(az, el, BORE_AZ, BORE_EL)
        ok = (nn >= min_n) & (th >= annulus[0]) & (th <= annulus[1])
        lev[name] = (10.0 * np.log10(np.median(ss[ok] / nn[ok]))) if ok.sum() >= 10 else None
    base = lev.get(ref) if lev.get(ref) is not None else next(
        (v for v in lev.values() if v is not None), None)
    return {k: (round(v - base, 3) if v is not None and base is not None else 0.0)
            for k, v in lev.items()}, lev


def pix_azel(nside):
    """(az, el) degrees of every RING pixel centre, in the cube's local horizon frame."""
    import healpy as hp
    theta, phi = hp.pix2ang(nside, np.arange(12 * nside * nside))
    return np.degrees(phi), 90.0 - np.degrees(theta)


# ── per-element normalisation ──────────────────────────────────────────────────────────────
def elem_gains(N, S, mode, min_n, fit_mask, quiet=False):
    """Per-element linear gain, one number per element, geometric mean 1.

    `fit_mask` selects the pixels the fit may use (see --fit-radius): the model is "one pattern
    times a per-element scale", and it is only true where every element actually sees the same
    sky, so the fit gets the well-sampled region and the far skirt rides along.
    """
    E = N.shape[0]
    g = np.ones(E)
    live = np.array([N[e][fit_mask].sum() > 0 for e in range(E)])
    with np.errstate(all="ignore"):
        X = np.where(N >= min_n, S / np.maximum(N, 1), np.nan)[:, fit_mask]
    keep = np.isfinite(X[live]).all(0) & (np.nanmin(X[live], 0) > 0)
    share = float("nan")
    if mode == "none":
        return g, live, share, int(keep.sum())
    if mode == "peak":
        with np.errstate(all="ignore"):
            pk = np.nanmax(np.where(np.isfinite(X), X, np.nan), axis=1)
        g = np.where(live & np.isfinite(pk) & (pk > 0), pk, 1.0)
    elif keep.sum() >= 32:
        M = X[np.ix_(live, keep)]
        if mode == "svd-log":
            M = 10.0 * np.log10(M)
            M = M - M.mean(0, keepdims=True)      # remove the pixel pattern; rows are the offsets
            u, s, _ = np.linalg.svd(M, full_matrices=False)
            share = float(s[0] ** 2 / (s ** 2).sum())
            rows = u[:, 0] * s[0] / np.sqrt(M.shape[1])
            if rows.mean() < 0 or (rows * M.mean(1)).sum() < 0:
                rows = -rows
            gl = np.ones(E)
            gl[live] = 10.0 ** (rows / 10.0)
            g = gl
        else:                                      # "svd": rank-1 in LINEAR power
            u, s, _ = np.linalg.svd(M, full_matrices=False)
            share = float(s[0] ** 2 / (s ** 2).sum())
            a = u[:, 0]
            if a.sum() < 0:
                a = -a
            if (a <= 0).any():
                if not quiet:
                    print("  WARNING: dominant mode has %d non-positive element amplitude(s); "
                          "falling back to svd-log for those" % int((a <= 0).sum()))
                a = np.where(a > 0, a, np.nan)
            gl = np.ones(E)
            gl[live] = a
            g = gl
    elif not quiet:
        print("  WARNING: only %d pixel(s) common to every element -- no gain fit, using 1.0"
              % int(keep.sum()))
    g = np.where(np.isfinite(g) & (g > 0), g, np.nan)
    ref = np.exp(np.nanmean(np.log(g[live])))      # geometric mean 1: the map level is preserved
    g = g / ref
    g[~live] = np.nan
    return g, live, share, int(keep.sum())


def collapse(N, S, gains, elems, smooth_deg=0.0, nside=None, weight="ivar"):
    """Gain-corrected, n-weighted mean power per pixel over the chosen elements.

    `smooth_deg` widens the bin instead of interpolating the answer: the NUMERATOR and the
    DENOMINATOR are smoothed with the same kernel and divided afterwards, so a smoothed pixel
    is the sample-weighted mean over the kernel -- the same estimator as a coarser nside, just
    centred everywhere. Interpolating the RATIO would invent a level in an empty pixel from a
    neighbour's, which at these fill fractions is most of the map.
    """
    num = np.zeros(N.shape[1])
    den = np.zeros(N.shape[1], np.float64)
    for e in elems:
        if not np.isfinite(gains[e]):
            continue
        # INVERSE-VARIANCE, NOT SAMPLE COUNT. Correcting an element by its gain divides its
        # NOISE by the same gain, so a -18 dB element arrives 60x amplified; weighting that by
        # sample count alone lets the elements that measure nothing dominate the average. A
        # gain-corrected sample has variance ~sigma^2/g^2, so its weight is g^2 -- which also
        # retires the dark elements without a threshold anyone has to choose.
        w = gains[e] ** 2 if weight == "ivar" else 1.0
        num += w * S[e] / gains[e]
        den += w * N[e]
    if smooth_deg > 0.0:
        import healpy as hp
        fwhm = np.radians(smooth_deg)
        num = hp.smoothing(num, fwhm=fwhm, verbose=False) if _hp_verbose() else hp.smoothing(
            num, fwhm=fwhm)
        den = hp.smoothing(den, fwhm=fwhm, verbose=False) if _hp_verbose() else hp.smoothing(
            den, fwhm=fwhm)
        num, den = np.maximum(num, 0.0), np.maximum(den, 0.0)
    with np.errstate(all="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), den


def _hp_verbose():
    """healpy dropped `verbose` in 1.15; probe once so the call works on either."""
    import inspect
    import healpy as hp
    return "verbose" in inspect.signature(hp.smoothing).parameters


# ── projection ─────────────────────────────────────────────────────────────────────────────
def uvec(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)], -1)


def patch(vals, nside, radius_deg, size, proj="azeq"):
    """Resample a full-sky array onto a square patch centred on boresight.

    Azimuthal equidistant by default so a radius on the image IS an angle in degrees -- the
    projection to read ring radii off. `x` is EAST, `y` is NORTH, and the image is drawn with
    east to the LEFT (looking up at the sky, azimuth running clockwise from north).
    """
    import healpy as hp
    b = uvec(BORE_AZ, BORE_EL)
    north = np.array([-np.sin(np.radians(BORE_EL)) * np.sin(np.radians(BORE_AZ)),
                      -np.sin(np.radians(BORE_EL)) * np.cos(np.radians(BORE_AZ)),
                      np.cos(np.radians(BORE_EL))])
    east = np.array([np.cos(np.radians(BORE_AZ)), -np.sin(np.radians(BORE_AZ)), 0.0])
    ax = np.linspace(-radius_deg, radius_deg, size)
    X, Y = np.meshgrid(ax, ax)
    r = np.hypot(X, Y)
    inside = r <= radius_deg
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = np.where(r > 0, X / r, 0.0), np.where(r > 0, Y / r, 0.0)
    if proj == "gnomonic":
        rr = np.arctan(np.radians(r))
    elif proj == "ortho":
        rr = np.arcsin(np.clip(np.sin(np.radians(r)), -1, 1))
    else:
        rr = np.radians(r)
    v = (b[None, None, :] * np.cos(rr)[..., None]
         + (east[None, None, :] * ux[..., None] + north[None, None, :] * uy[..., None])
         * np.sin(rr)[..., None])
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v / np.where(nrm > 0, nrm, 1.0)
    theta = np.arccos(np.clip(v[..., 2], -1, 1))
    phi = np.arctan2(v[..., 0], v[..., 1]) % (2 * np.pi)
    img = np.full(X.shape, np.nan)
    pix = hp.ang2pix(nside, theta, phi)
    img[inside] = vals[pix[inside]]
    return img, ax


def airy_db(theta_deg, freq_hz, dish_m=DISH_M):
    from scipy.special import j1
    x = np.pi * dish_m / (C_LIGHT / freq_hz) * np.sin(np.radians(theta_deg))
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(x == 0.0, 1.0, 2.0 * j1(x) / x)
    return 20.0 * np.log10(np.maximum(np.abs(a), 1e-9))


def freq_hz(freqs, masters):
    """Representative sky frequency of the selection, for the Airy overlay (0.1953125 MHz/id)."""
    if freqs:
        ids = sorted(freqs)
    else:
        ids = sorted({f for m in masters for c in m["chains"].values()
                      for lo, hi in c["freq_ids"] for f in (lo, hi)})
    return float(np.mean(ids)) * 0.1953125e6


# ── plots ──────────────────────────────────────────────────────────────────────────────────
def fig_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_map(img, ax_deg, title, sub, png, vmin, vmax, cbar_label, rings=()):
    plt = fig_setup()
    d = os.path.dirname(os.path.abspath(png))
    if d:
        os.makedirs(d, exist_ok=True)
    fig, a = plt.subplots(figsize=(7.6, 7.0), dpi=160)
    ext = [ax_deg[0], ax_deg[-1], ax_deg[0], ax_deg[-1]]
    im = a.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
    a.invert_xaxis()                       # east to the left: the view looking UP
    for r in rings:
        a.add_artist(plt.Circle((0, 0), r, fill=False, color="w", lw=0.5, alpha=0.35))
        a.annotate("%g°" % r, (0, r), color="w", fontsize=6, alpha=0.6,
                   ha="center", va="bottom")
    a.plot(0, 0, "+", color="w", ms=9, mew=1.0, alpha=0.8)
    a.set_xlabel("east offset from boresight  [deg]")
    a.set_ylabel("north offset from boresight  [deg]")
    a.set_title(title, fontsize=10)
    a.text(0.5, -0.105, sub, transform=a.transAxes, ha="center", va="top", fontsize=7,
           color="0.35")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label=cbar_label)
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def profile(ang, val, edges):
    """Median and 16/84 percentiles per annulus, plus the pixel count."""
    cen, med, lo, hi, cnt = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (ang >= a) & (ang < b) & np.isfinite(val)
        if s.sum() < 3:
            continue
        cen.append(0.5 * (a + b))
        v = val[s]
        med.append(np.median(v))
        lo.append(np.percentile(v, 16))
        hi.append(np.percentile(v, 84))
        cnt.append(int(s.sum()))
    return [np.array(x) for x in (cen, med, lo, hi, cnt)]


# ── commands ───────────────────────────────────────────────────────────────────────────────
def prepare(args):
    """Everything the three commands share: load, align chains, accumulate, fit gains."""
    freqs = parse_freqs(args.freqs)
    masters = [load_master(p, set(args.chains) if args.chains else None) for p in args.masters]
    masters = [m for m in masters if m["chains"]]
    if not masters:
        sys.exit("no master has any of the requested chain(s)")
    offs, lev = (None, {})
    if not args.no_chain_align:
        offs, lev = chain_offsets_from(masters, freqs, args.annulus, args.min_n, args.offset_ref)
    N, S, nside, info = gather(masters, freqs, offs)
    az, el = pix_azel(nside)
    th = angsep_deg(az, el, BORE_AZ, BORE_EL)
    fit = (th <= args.fit_radius) & (N.sum(0) > 0)
    g, live, share, npix_fit = elem_gains(N, S, args.elem_norm, args.min_n, fit)
    print("days %s   chains %s   nside %d   %s units"
          % ("+".join(sorted(set(info["days"]))),
             " ".join("%s(%d ch)" % (k, info["n_chan"][k]) for k in sorted(info["chains"])),
             nside, masters[0]["units"]))
    if offs:
        print("  chain align onto %s: %s" % (args.offset_ref,
              "  ".join("%s %+.2f dB" % (k, v) for k, v in sorted(offs.items()))))
    print("  elem-norm %s over %d pixel(s) within %.0f deg; dominant-mode variance share %s"
          % (args.elem_norm, npix_fit, args.fit_radius,
             "n/a" if not np.isfinite(share) else "%.3f" % share))
    gl = g[np.isfinite(g)]
    if len(gl):
        print("  element gains: %.2f..%.2f dB (%d live, %d dark)"
              % (10 * np.log10(gl.min()), 10 * np.log10(gl.max()), int(live.sum()),
                 int((~live).sum())))
    return dict(masters=masters, freqs=freqs, N=N, S=S, nside=nside, az=az, el=el, th=th,
                gains=g, live=live, share=share, info=info, offsets=offs, levels=lev)


def cmd_map(args):
    d = prepare(args)
    n_elem = d["N"].shape[0]
    sweep = None
    if args.sweep == "elem":
        sweep = [("e%03d" % e, [e], d["freqs"]) for e in parse_elems(args.elements, n_elem)]
    elif args.sweep == "freq":
        ids = sorted(d["freqs"]) if d["freqs"] else sorted(
            {lo for m in d["masters"] for c in m["chains"].values() for lo, _ in c["freq_ids"]})
        sweep = [("f%d" % f, parse_elems(args.elements, n_elem), {f}) for f in ids]
    else:
        sweep = [(None, parse_elems(args.elements, n_elem), d["freqs"])]

    outs = []
    vmin, vmax = args.vmin, args.vmax
    frames = []
    for tag, elems, freqs in sweep:
        if freqs is not d["freqs"]:
            N, S, nside, info = gather(d["masters"], freqs, d["offsets"])
        else:
            N, S, nside, info = d["N"], d["S"], d["nside"], d["info"]
        val, cnt = collapse(N, S, d["gains"], elems, args.smooth, nside, args.elem_weight)
        val = np.where(cnt >= args.min_n, val, np.nan)
        with np.errstate(all="ignore"):
            db = 10.0 * np.log10(val)
        if args.peak_norm:
            db = db - np.nanmax(db)
        img, axd = patch(db, nside, args.radius, args.size, args.proj)
        frames.append((tag, img, axd, elems, freqs, np.isfinite(db).sum()))
    if vmin is None or vmax is None:
        allv = np.concatenate([f[1][np.isfinite(f[1])] for f in frames if np.isfinite(f[1]).any()])
        hi = np.percentile(allv, 99.9)
        vmax = hi if vmax is None else vmax
        vmin = (hi - args.dr) if vmin is None else vmin
    for tag, img, axd, elems, freqs, npx in frames:
        base, ext = os.path.splitext(args.out)
        png = args.out if tag is None else "%s_%s%s" % (base, tag, ext or ".png")
        fsel = ("all channels" if not freqs else
                "freq_id " + (",".join(str(f) for f in sorted(freqs)) if len(freqs) <= 4
                              else "%d..%d" % (min(freqs), max(freqs))))
        esel = ("all elements" if len(elems) == d["N"].shape[0] else
                "element " + (",".join(str(e) for e in elems) if len(elems) <= 6
                              else "%d of %d" % (len(elems), d["N"].shape[0])))
        title = "%s · %s · %s" % ("+".join(sorted(d["info"]["chains"])), fsel, esel)
        sub = ("days %s · nside %d · %s · elem-norm %s · %d pixels · %s"
               % ("+".join(sorted(set(d["info"]["days"]))), d["nside"], args.proj,
                  args.elem_norm, npx, "peak-normalised" if args.peak_norm else "pedestal dB")
               + ("" if not args.smooth else " · smoothed %.2f deg FWHM" % args.smooth))
        outs.append(draw_map(img, axd, title, sub, png, vmin, vmax,
                             "dB rel. peak" if args.peak_norm else "dB (pedestal units)",
                             rings=args.rings))
        print("  -> %s" % png)
    return outs


def cmd_radial(args):
    d = prepare(args)
    plt = fig_setup()
    n_elem = d["N"].shape[0]
    edges = np.linspace(0.0, args.radius, int(args.radius / args.bin_deg) + 1)
    series = []
    if args.by == "element":
        for e in parse_elems(args.elements, n_elem):
            if not d["live"][e]:
                print("  element %d is dark -- skipped" % e)
                continue
            val, cnt = collapse(d["N"], d["S"], d["gains"], [e], args.smooth, d["nside"], args.elem_weight)
            series.append(("element %d" % e, val, cnt))
    else:
        ids = sorted(d["freqs"]) if d["freqs"] else []
        if not ids:
            sys.exit("--by freq needs --freqs (a list or range of absolute freq_ids)")
        elems = parse_elems(args.elements, n_elem)
        for f in ids:
            N, S, _, _ = gather(d["masters"], {f}, d["offsets"])
            val, cnt = collapse(N, S, d["gains"], elems, args.smooth, d["nside"], args.elem_weight)
            series.append(("freq_id %d (%.1f MHz)" % (f, f * 0.1953125), val, cnt))

    # Profile every series first: the y-range, the reference level and the residual panel all
    # need the whole set before anything is drawn.
    prof = []
    for lab, val, cnt in series:
        v = np.where(cnt >= args.min_n, val, np.nan)
        with np.errstate(all="ignore"):
            db = 10.0 * np.log10(v)
        cen, med, lo, hi, _ = profile(d["th"], db, edges)
        if len(cen) == 0:
            continue
        prof.append([lab, cen, med, lo, hi])
    if not prof:
        sys.exit("no series had enough pixels -- lower --min-n or widen --bin-deg")
    # PER-SERIES LEVEL. Channels are NOT on one level even in pedestal units: a cell is signal
    # over that channel's own noise, and the BPSK(10) spectrum puts ~12 dB more signal in a
    # centre channel than an edge one. Referencing each curve to its own level in a wide
    # annulus compares the SHAPE, which is the question a frequency sweep is asking.
    if args.series_ref:
        a, b = args.series_ref
        for pr in prof:
            m = (pr[1] >= a) & (pr[1] <= b)
            z = np.nanmedian(pr[2][m]) if m.any() else np.nanmax(pr[2])
            pr[2], pr[3], pr[4] = pr[2] - z, pr[3] - z, pr[4] - z
    elif args.peak_norm:
        for pr in prof:
            z = np.nanmax(pr[2])
            pr[2], pr[3], pr[4] = pr[2] - z, pr[3] - z, pr[4] - z

    grid = np.array(sorted({c for pr in prof for c in pr[1]}))
    stack = np.full((len(prof), len(grid)), np.nan)
    for i, pr in enumerate(prof):
        stack[i, np.searchsorted(grid, pr[1])] = pr[2]
    ref = np.nanmedian(stack, 0)

    nax = 2 if args.residual else 1
    fig, axes = plt.subplots(nax, 1, figsize=(8.6, 5.6 + 2.4 * (nax - 1)), dpi=160,
                             sharex=True, gridspec_kw={"height_ratios": [3, 1.4][:nax]})
    a = axes[0] if nax > 1 else axes
    cmap = plt.get_cmap("viridis")
    for i, (lab, cen, med, lo, hi) in enumerate(prof):
        col = cmap(i / max(1, len(prof) - 1))
        a.plot(cen, med, "-", color=col, lw=1.4, label=lab)
        if args.band:
            a.fill_between(cen, lo, hi, color=col, alpha=0.12, lw=0)
        if nax > 1:
            axes[1].plot(cen, med - ref[np.searchsorted(grid, cen)], "-", color=col, lw=1.1)
    ylo = min(np.nanmin(pr[2]) for pr in prof)
    yhi = max(np.nanmax(pr[2]) for pr in prof)
    pad = 0.12 * (yhi - ylo)
    ylo, yhi = (args.ymin if args.ymin is not None else ylo - pad,
                args.ymax if args.ymax is not None else yhi + pad)
    if not args.no_airy:
        t = np.linspace(0.02, args.radius, 2000)
        f0 = freq_hz(d["freqs"], d["masters"])
        ad = airy_db(t, f0, args.dish) - airy_db(np.array([1e-6]), f0, args.dish)[0]
        # Anchor the ideal pattern to the data at the innermost measured annulus. The nulls run
        # to -inf and would own the y-range, so it is CLIPPED to the panel, not fitted to it.
        ad = ad + np.nanmax([pr[2][0] for pr in prof]) - ad[np.argmin(np.abs(t - prof[0][1][0]))]
        a.plot(t, np.clip(ad, ylo, None), "k--", lw=1.0, alpha=0.55,
               label="Airy %.1f m at %.0f MHz (clipped)" % (args.dish, f0 / 1e6))
    a.set_ylim(ylo, yhi)
    a.set_ylabel("dB rel. %s" % ("peak" if args.peak_norm else
                                 ("%g-%g deg" % tuple(args.series_ref)) if args.series_ref
                                 else "pedestal"))
    a.grid(alpha=0.25, lw=0.5)
    a.legend(fontsize=7, ncol=2)
    a.set_title("%s · radial cut · by %s" % ("+".join(sorted(d["info"]["chains"])), args.by),
                fontsize=10)
    if nax > 1:
        axes[1].axhline(0, color="0.4", lw=0.8)
        axes[1].set_ylabel("− median [dB]")
        axes[1].grid(alpha=0.25, lw=0.5)
    (axes[-1] if nax > 1 else a).set_xlabel("angle from boresight  [deg]")
    (axes[-1] if nax > 1 else a).set_xlim(0, args.radius)
    (axes[-1] if nax > 1 else a).text(
        0.5, -0.22 if nax > 1 else -0.13,
        "days %s · nside %d · elem-norm %s · median per %.2f deg annulus%s"
        % ("+".join(sorted(set(d["info"]["days"]))), d["nside"], args.elem_norm, args.bin_deg,
           ", 16/84% band" if args.band else ""),
        transform=(axes[-1] if nax > 1 else a).transAxes, ha="center", va="top",
        fontsize=7, color="0.35")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print("  -> %s" % args.out)
    return [args.out]


def cmd_gains(args):
    d = prepare(args)
    plt = fig_setup()
    modes = ["svd", "svd-log", "peak"]
    fit = (d["th"] <= args.fit_radius) & (d["N"].sum(0) > 0)
    fig, a = plt.subplots(figsize=(8.4, 4.4), dpi=160)
    edges = np.linspace(args.scatter_range[0], args.scatter_range[1],
                        int((args.scatter_range[1] - args.scatter_range[0]) / 1.0) + 1)
    for m in modes:
        g, live, share, npx = elem_gains(d["N"], d["S"], m, args.min_n, fit, quiet=True)
        # THE NUMBER THAT DECIDES BETWEEN MODES. A normalisation is good if it makes the
        # elements agree, so profile every live element and report the spread about their
        # median: same data, same bins, one scalar per mode.
        curves = []
        want = set(parse_elems(args.elements, len(g)))
        for e in [x for x in np.flatnonzero(live) if int(x) in want]:
            val, cnt = collapse(d["N"], d["S"], g, [int(e)], args.smooth, d["nside"], args.elem_weight)
            with np.errstate(all="ignore"):
                db = 10.0 * np.log10(np.where(cnt >= args.min_n, val, np.nan))
            cen, med, _, _, _ = profile(d["th"], db, edges)
            if len(cen) == len(edges) - 1:
                curves.append(med)
        rms = float("nan")
        if len(curves) >= 3:
            C = np.array(curves)
            rms = float(np.sqrt(np.nanmean((C - np.nanmedian(C, 0)) ** 2)))
        with np.errstate(all="ignore"):
            a.plot(np.arange(len(g)), 10 * np.log10(g), "o-", ms=3, lw=1.0,
                   label="%s (mode share %s, element scatter %.2f dB)"
                   % (m, "n/a" if not np.isfinite(share) else "%.2f" % share, rms))
        print("  %-8s scatter %.3f dB over %g-%g deg (%d elements) | %s"
              % (m, rms, args.scatter_range[0], args.scatter_range[1], len(curves),
                 " ".join("%.2f" % x for x in 10 * np.log10(g))))
    a.set_xlabel("element")
    a.set_ylabel("gain [dB]")
    a.grid(alpha=0.25, lw=0.5)
    a.legend(fontsize=8)
    a.set_title("per-element normalisation, %s · days %s"
                % ("+".join(sorted(d["info"]["chains"])),
                   "+".join(sorted(set(d["info"]["days"])))), fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print("  -> %s" % args.out)
    return [args.out]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("masters", nargs="+", help="cube_<day>_nside<N>.npz")
        p.add_argument("--chains", nargs="*", default=["bds_b2a", "gal_e5a", "gps_l5"],
                       help="default: the three 1176 MHz chains")
        p.add_argument("--freqs", default="all",
                       help="ABSOLUTE freq_ids: 5988 | 5980-6000 | 5988,6000 | all")
        p.add_argument("--elements", default="all", help="0 | 0,3 | 0-7 | all")
        p.add_argument("--elem-norm", default="svd",
                       choices=["svd", "svd-log", "peak", "none"])
        p.add_argument("--fit-radius", type=float, default=30.0,
                       help="pixels within this angle of boresight fit the element gains")
        p.add_argument("--min-n", type=int, default=8, help="samples a pixel needs to be shown")
        p.add_argument("--annulus", type=float, nargs=2, default=(2.0, 12.0),
                       help="where the cross-chain offset is measured")
        p.add_argument("--offset-ref", default="gps_l5")
        p.add_argument("--no-chain-align", action="store_true")
        p.add_argument("--elem-weight", default="ivar", choices=["ivar", "n"],
                       help="how elements combine: inverse-variance (default) or sample count")
        p.add_argument("--smooth", type=float, default=0.0, metavar="FWHM_DEG",
                       help="widen the bin: smooth the accumulators, then divide "
                            "(0 = raw pixels). At nside 128 the sky is sampled along "
                            "satellite TRACKS, so a map is track-limited long before "
                            "it is resolution-limited")
        p.add_argument("--peak-norm", action="store_true")
        p.add_argument("--out", required=True)

    m = sub.add_parser("map", help="sky map around boresight")
    common(m)
    m.add_argument("--radius", type=float, default=25.0)
    m.add_argument("--size", type=int, default=900, help="image pixels per side")
    m.add_argument("--proj", default="azeq", choices=["azeq", "gnomonic", "ortho"])
    m.add_argument("--dr", type=float, default=35.0, help="colour range below the 99.9th pct")
    m.add_argument("--vmin", type=float, default=None)
    m.add_argument("--vmax", type=float, default=None)
    m.add_argument("--rings", type=float, nargs="*", default=[5, 10, 20])
    m.add_argument("--sweep", default=None, choices=["freq", "elem"],
                   help="one frame per channel/element, shared colour scale (for video)")
    m.set_defaults(fn=cmd_map)

    r = sub.add_parser("radial", help="radial cut(s) vs angle from boresight")
    common(r)
    r.add_argument("--by", default="element", choices=["element", "freq"])
    r.add_argument("--radius", type=float, default=40.0)
    r.add_argument("--bin-deg", type=float, default=0.5)
    r.add_argument("--band", action="store_true", help="shade the 16/84 percentile spread")
    r.add_argument("--no-airy", action="store_true")
    r.add_argument("--dish", type=float, default=DISH_M)
    r.add_argument("--series-ref", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="level each curve by its own median in this annulus, to compare SHAPES")
    r.add_argument("--residual", action="store_true",
                   help="second panel: each curve minus the median of the set")
    r.add_argument("--ymin", type=float, default=None)
    r.add_argument("--ymax", type=float, default=None)
    r.set_defaults(fn=cmd_radial)

    g = sub.add_parser("gains", help="compare the per-element normalisations")
    common(g)
    g.add_argument("--scatter-range", type=float, nargs=2, default=(5.0, 30.0),
                   metavar=("LO", "HI"),
                   help="angles over which element-to-element agreement is scored")
    g.set_defaults(fn=cmd_gains)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
