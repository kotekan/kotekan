#!/usr/bin/env python3
"""Beam data cube: response vs (chain, subband, element, healpix) -- built, and exported.

    build   archives -> one master cube per UTC day (sparse .npz accumulators)
    export  master cubes -> one browser cube per day (.bin + .json) for the viewer
    ls      what a master cube contains, without loading the values

WHY A CUBE AND NOT A MAP. gnss_beam_map.py already makes beautiful maps, and every one of them
is a COLLAPSE: over elements, over frequency, or both. The parts are what the instrument
actually measures, and the collapses are display defaults -- so the cube stores the parts and
lets the viewer do the summing. A combined number can always be rebuilt from the parts; the
parts can never be recovered from a combined number.

⚠️⚠️ WHICH FIELD IS THE BEAM, restated because it has cost a day already. The beam is
PROBE-DEBIASED `p2` = <|A_e|^2>. It is NOT `|u|/q`: that is a ratio to the array mean, so it
is flat across the sky BY CONSTRUCTION -- every element rises and falls with its own
reference -- and mapping it yields a smooth, plausible, entirely meaningless ~6 dB all-sky
"beam". The tell was physical, not statistical: a 6 m dish at 1176 MHz cannot have a 6 dB
full-sky pattern. See docs/CHORD_BEAM_MAPS.md §2.

⚠️ WHAT IS STORED IS AN ACCUMULATOR, NEVER A dB. Each cell keeps (n, s1, s2) over LINEAR
debiased power, so cells, subbands, elements, chains and DAYS all combine by ADDITION and
adding a night never requires reprocessing an old one. dB happens once, in the viewer, at the
end -- which is also the only way a "sum over elements" toggle can mean anything, since you
cannot add decibels.

TWO SOURCES, ONE OUTPUT FORMAT
  --source elem  (default, works today)  fixtures/obs/elem_<chain>_<YYYYMMDD>.jsonl
        The element archive. Per-element, but the covering channels were already summed
        upstream, so the subband axis comes out LENGTH 1 per chain. The 8 chains still give
        real coarse frequency (5 distinct bands), which is why this is worth building now.
  --source cube  fixtures/cube/cube_<chain>_<YYYYMMDD>.jsonl
        The /get_beam_cube archive: per (subband, element), the joint axis. Needs the node
        side armed (GnssGpuRecordAssemble `beam_cube: true`) -- until then there are no files
        and this source finds nothing. The output format is IDENTICAL either way, so the
        viewer does not change when the axis becomes real; only n_subband grows.

⚠️ THE RAILING VETO IS CROSS-CHAIN, AND THAT IS THE POINT. A satellite within ~5 deg of
boresight rails the 4+4b quantiser for EVERY chain at once -- they all ride the same nibbles.
A per-chain veto would clean the chain carrying the bright satellite and leave the other seven
contaminated WHILE THEY LOOK LIKE CLEAN CONTROLS. Here the veto is computed from geometry over
all three constellations, independent of what any chain happened to be tracking, and applied
to every chain of that epoch. Vetoing removes the main lobe itself; that is the honest outcome,
not a bug -- the main lobe is what breaks the measurement.

@author Keith Vanderlinde
"""
import argparse
import json
import math
import os
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all  # noqa: E402

LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
R_REF = 20.2e6            # m: the dB zero for range normalisation (semi-synchronous slant)
BORE_AZ, BORE_EL = 180.0, 81.41   # docs/CHORD_BEAM_MAPS.md §5 -- NOT telescope.dish_coelev_deg
MIN_ELEMS = 8             # an instance with fewer live elements says nothing about the beam
FLOOR_BIN_S = 300.0       # probe-pedestal bin: per (instance, element, 5 min)
GEOM_BIN_S = 60.0         # BRDC evaluated once per (prn, minute), interpolation-free

# Chain -> constellation letter, for the BRDC lookup. Declared, never guessed from the
# frequency: naming a band by nearest carrier is what invented a "GLONASS L3" we do not fly.
CHAIN_SYS = {"gps_l5": "G", "gps_l2c": "G", "gal_e5a": "E", "gal_e5b": "E", "gal_e6": "E",
             "bds_b2a": "C", "bds_b2b": "C", "bds_b3i": "C"}


# ─────────────────────────────────────────────────────────────────────────────────────────
# healpix. Only ang2pix RING is needed, so it is inlined rather than dragging in healpy --
# the builder then runs in venv-ft alongside everything else instead of needing /home/kvand/
# gnss/venv. Verified against healpy in test_beam_cube.py; if that test is not passing, do
# not trust a map made with this.
# ─────────────────────────────────────────────────────────────────────────────────────────
def ang2pix_ring(nside, theta, phi):
    """HEALPix RING pixel index for colatitude theta and longitude phi (radians, arrays)."""
    nside = int(nside)
    npix = 12 * nside * nside
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2.0 * np.pi) * (2.0 / np.pi)      # in [0, 4)
    pix = np.empty(theta.shape, dtype=np.int64)

    eq = za <= 2.0 / 3.0
    # --- equatorial belt -------------------------------------------------------------
    if np.any(eq):
        t, zz = tt[eq], z[eq]
        temp1 = nside * (0.5 + t)
        temp2 = nside * zz * 0.75
        jp = (temp1 - temp2).astype(np.int64)          # ascending edge
        jm = (temp1 + temp2).astype(np.int64)          # descending edge
        ir = nside + 1 + jp - jm                       # in {1, 2n+1}
        kshift = 1 - (ir & 1)
        ip = (jp + jm - nside + kshift + 1) // 2
        ip = np.mod(ip, 4 * nside)
        pix[eq] = 2 * nside * (nside - 1) + (ir - 1) * 4 * nside + ip
    # --- polar caps ------------------------------------------------------------------
    if np.any(~eq):
        t, zz, zza = tt[~eq], z[~eq], za[~eq]
        tp = t - np.floor(t)
        tmp = nside * np.sqrt(3.0 * (1.0 - zza))
        jp = (tp * tmp).astype(np.int64)
        jm = ((1.0 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.mod((t * ir).astype(np.int64), 4 * ir)
        north = zz > 0
        p = np.where(north, 2 * ir * (ir - 1) + ip,
                     npix - 2 * ir * (ir + 1) + ip)
        pix[~eq] = p
    return pix


def azel_to_pix(nside, az_deg, el_deg):
    """Local horizon frame: colatitude from zenith, longitude = azimuth (N=0, clockwise).

    Deliberately the LOCAL frame, not celestial: the beam is fixed to the dish, so a
    celestial pixelisation would smear a stationary pattern across the sky as the earth turns.
    """
    theta = np.radians(90.0 - np.asarray(el_deg, dtype=float))
    phi = np.radians(np.asarray(az_deg, dtype=float))
    return ang2pix_ring(nside, theta, phi)


def angsep_deg(az1, el1, az2, el2):
    a1, e1, a2, e2 = (np.radians(x) for x in (az1, el1, az2, el2))
    c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ─────────────────────────────────────────────────────────────────────────────────────────
# Geometry, shared by every chain of a day so the veto is genuinely cross-chain.
# ─────────────────────────────────────────────────────────────────────────────────────────
class Geometry:
    """BRDC positions on a 1-minute grid, plus the cross-chain railing veto for that minute."""

    def __init__(self, day_unix, veto_deg):
        self.eph = parse_rinex_nav(fetch_brdc(
            datetime.fromtimestamp(day_unix + 43200, tz=timezone.utc)))
        self.veto_deg = veto_deg
        self._pos = {}       # (sys, prn, minute) -> (el, az, range_m) or None
        self._all = {}       # minute -> {(sys, prn): v}
        self._veto = {}      # minute -> (vetoed, closest_deg, closest_name)

    def _minute(self, minute):
        if minute not in self._all:
            try:
                self._all[minute] = predict_all(self.eph, LAT, LON, ALT,
                                                minute * GEOM_BIN_S + GEOM_BIN_S / 2,
                                                mask_deg=-90.0, max_age=86400.0) or {}
            except Exception:
                self._all[minute] = {}
        return self._all[minute]

    def at(self, sys_, prn, t):
        key = (sys_, prn, int(t // GEOM_BIN_S))
        if key not in self._pos:
            v = self._minute(key[2]).get((sys_, prn))
            self._pos[key] = (v["el"], v["az"], v["range_m"]) if v else None
        return self._pos[key]

    def vetoed(self, t):
        """True if ANY satellite of ANY constellation is inside veto_deg of boresight.

        Pooled over all three constellations on purpose: the chains share the 4+4b nibbles,
        so a veto computed per chain would clean the one carrying the bright satellite and
        leave the rest contaminated while looking like clean controls.
        """
        if self.veto_deg <= 0.0:
            return False
        m = int(t // GEOM_BIN_S)
        if m not in self._veto:
            best, name = 1e9, None
            for (sy, prn), v in self._minute(m).items():
                if v is None or v.get("el") is None or v["el"] < 0.0:
                    continue
                d = float(angsep_deg(v["az"], v["el"], BORE_AZ, BORE_EL))
                if d < best:
                    best, name = d, "%s%d" % (sy, prn)
            self._veto[m] = (best < self.veto_deg, best, name)
        return self._veto[m][0]

    def veto_stats(self):
        n = len(self._veto)
        v = sum(1 for x in self._veto.values() if x[0])
        return n, v


# ─────────────────────────────────────────────────────────────────────────────────────────
# The accumulator. Sparse over pixels: a day of tracks touches a small fraction of the sky,
# and storing the full outer product dense is what turns a 2 MB cube into a 60 MB one.
# ─────────────────────────────────────────────────────────────────────────────────────────
class CubeAccum:
    def __init__(self, nside, n_elem):
        self.nside = int(nside)
        self.n_elem = int(n_elem)
        self.n_sub = 0
        self.cells = {}   # (subband, pixel) -> [n(E), s1(E), s2(E)]

    def add(self, sub, pix, power, live):
        """power: [n_elem] linear debiased power; live: [n_elem] bool (dark elements excluded)."""
        self.n_sub = max(self.n_sub, sub + 1)
        c = self.cells.get((sub, pix))
        if c is None:
            c = [np.zeros(self.n_elem, np.uint32), np.zeros(self.n_elem),
                 np.zeros(self.n_elem)]
            self.cells[(sub, pix)] = c
        c[0][live] += 1
        c[1][live] += power[live]
        c[2][live] += power[live] ** 2

    def arrays(self):
        """-> (pix[P], n[S,E,P], s1[S,E,P], s2[S,E,P]) with pix sorted and unique."""
        pixels = sorted({p for _, p in self.cells})
        idx = {p: i for i, p in enumerate(pixels)}
        S, E, P = max(1, self.n_sub), self.n_elem, len(pixels)
        n = np.zeros((S, E, P), np.uint32)
        s1 = np.zeros((S, E, P))
        s2 = np.zeros((S, E, P))
        for (sub, p), c in self.cells.items():
            j = idx[p]
            n[sub, :, j] = c[0]
            s1[sub, :, j] = c[1]
            s2[sub, :, j] = c[2]
        return np.asarray(pixels, np.int32), n, s1, s2


# ─────────────────────────────────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────────────────────────────────
def probe_floors(paths, tmin, tmax):
    """Probe pedestal per (instance, subband, element, 5-min bin), as a MEDIAN.

    Per ELEMENT and per SUBBAND, never medianed across either: elements differ in gain by
    design and bands differ in noise figure, so a floor averaged over them over-subtracts the
    quiet ones into negative power.
    """
    acc = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                if '"probe": true' not in line and '"probe":true' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("t")
                if t is None or not (tmin <= t <= tmax) or not d.get("probe"):
                    continue
                key0 = (d.get("inst"), int(t // FLOOR_BIN_S))
                for sub, row in enumerate(_rows(d)):
                    for e, v in enumerate(row):
                        if v > 0.0:
                            acc[(key0[0], sub, key0[1])][e].append(float(v))
    return {k: {e: float(np.median(v)) for e, v in d.items()} for k, d in acc.items()}


def _rows(d):
    """Per-subband list of per-element p2 for one archive row, for BOTH sources.

    elem archive: `p2` is [n_elem]                      -> one subband
    cube archive: `p2_sum` is [n_sub][n_elem] with `w`  -> per-subband MEAN power
    """
    if "p2_sum" in d:
        w = d.get("w") or []
        out = []
        for sub, els in enumerate(d["p2_sum"]):
            ww = float(w[sub]) if sub < len(w) else 0.0
            out.append([float(v) / ww for v in els] if ww > 0.0 else [0.0] * len(els))
        return out
    return [[float(v) for v in (d.get("p2") or [])]]


def build_day(args, chain, paths, day_unix, geom):
    tmin, tmax = day_unix, day_unix + 86400.0
    sys_ = CHAIN_SYS.get(chain)
    if sys_ is None:
        print("  %s: unknown chain, no constellation declared -> skipped" % chain)
        return None
    floors = probe_floors(paths, tmin, tmax)
    acc, freq_ids = None, None
    stats = defaultdict(int)

    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    stats["badjson"] += 1
                    continue
                t = d.get("t")
                if t is None or not (tmin <= t <= tmax):
                    continue
                if d.get("probe"):
                    stats["probe"] += 1
                    continue           # a probe is the FLOOR, never a beam sample
                rows = _rows(d)
                if not rows or len(rows[0]) < MIN_ELEMS:
                    stats["short"] += 1
                    continue
                if acc is None:
                    acc = CubeAccum(args.nside, len(rows[0]))
                    freq_ids = d.get("bin_freq_ids") or [[None, None]] * len(rows)
                stats["rows"] += 1
                if geom.vetoed(t):
                    stats["vetoed"] += 1
                    continue
                g = geom.at(sys_, int(d["prn"]), t)
                if not g or g[0] is None:
                    stats["nogeo"] += 1
                    continue
                el, az, rng = g
                if el <= args.mask_deg:
                    stats["low"] += 1
                    continue
                pix = int(azel_to_pix(args.nside, np.array([az]), np.array([el]))[0])
                # Range normalisation in the POWER domain: a satellite is ~2.5 dB stronger
                # overhead than at the horizon from spreading loss alone, which would
                # masquerade as beam. Power scales as r^2 relative to the reference range.
                rscale = 1.0 if args.no_range_norm else (rng / R_REF) ** 2
                for sub, row in enumerate(rows):
                    fl = floors.get((d.get("inst"), sub, int(t // FLOOR_BIN_S))) or {}
                    p = np.zeros(acc.n_elem)
                    live = np.zeros(acc.n_elem, bool)
                    for e in range(min(acc.n_elem, len(row))):
                        p2e, f = row[e], fl.get(e)
                        # DEBIAS IN POWER, and DROP a sample below its own pedestal rather
                        # than clamping it: clamping piles non-detections at an artificial
                        # floor and manufactures a beam edge exactly where the beam ends.
                        if p2e > 0.0 and f is not None and f > 0.0 and p2e > f:
                            p[e] = (p2e - f) * rscale
                            live[e] = True
                    if live.sum() >= MIN_ELEMS:
                        acc.add(sub, pix, p, live)
                        stats["cells"] += 1
                    else:
                        stats["thin"] += 1

    if acc is None or not acc.cells:
        print("  %s: no usable rows (%s)" % (chain, dict(stats)))
        return None
    pix, n, s1, s2 = acc.arrays()
    print("  %s: %d row(s) -> %d subband(s) x %d element(s) x %d pixel(s); "
          "vetoed %d, no-geom %d, below-mask %d, thin %d"
          % (chain, stats["rows"], n.shape[0], n.shape[1], len(pix),
             stats["vetoed"], stats["nogeo"], stats["low"], stats["thin"]))
    return dict(pix=pix, n=n, s1=s1, s2=s2, freq_ids=freq_ids[:n.shape[0]])


def find_inputs(args, chain, daystr):
    pat = ("cube_%s_%s.jsonl" if args.source == "cube" else "elem_%s_%s.jsonl") % (chain, daystr)
    p = os.path.join(args.archive, pat)
    return [p] if os.path.exists(p) else []


def cmd_build(args):
    os.makedirs(args.outdir, exist_ok=True)
    for daystr in args.days:
        day_unix = datetime.strptime(daystr, "%Y%m%d").replace(
            tzinfo=timezone.utc).timestamp()
        chains = args.chains or sorted(CHAIN_SYS)
        present = [(c, find_inputs(args, c, daystr)) for c in chains]
        present = [(c, p) for c, p in present if p]
        if not present:
            print("%s: no %s archives found in %s" % (daystr, args.source, args.archive))
            continue
        print("%s: %d chain(s)" % (daystr, len(present)))
        geom = Geometry(day_unix, args.veto_deg)
        out = {"nside": args.nside, "day": daystr, "source": args.source,
               "veto_deg": args.veto_deg, "range_norm": not args.no_range_norm,
               "chains": []}
        blobs = {}
        for chain, paths in present:
            r = build_day(args, chain, paths, day_unix, geom)
            if r is None:
                continue
            i = len(out["chains"])
            out["chains"].append({"chain": chain, "sys": CHAIN_SYS[chain],
                                  "n_sub": int(r["n"].shape[0]),
                                  "n_elem": int(r["n"].shape[1]),
                                  "n_pix": int(len(r["pix"])),
                                  "freq_ids": r["freq_ids"]})
            blobs["pix_%d" % i] = r["pix"]
            blobs["n_%d" % i] = r["n"]
            blobs["s1_%d" % i] = r["s1"]
            blobs["s2_%d" % i] = r["s2"]
        nm, nv = geom.veto_stats()
        print("  railing veto: %d/%d minute(s) vetoed (%.1f%%) at %.1f deg"
              % (nv, nm, 100.0 * nv / max(1, nm), args.veto_deg))
        if not out["chains"]:
            continue
        path = os.path.join(args.outdir, "cube_%s_nside%d.npz" % (daystr, args.nside))
        np.savez_compressed(path, meta=json.dumps(out), **blobs)
        print("  -> %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))


# ─────────────────────────────────────────────────────────────────────────────────────────
# export: master -> browser cube
# ─────────────────────────────────────────────────────────────────────────────────────────
def cmd_export(args):
    """One .bin + .json per day. The browser sums; the server does not pre-collapse anything.

    ⚠️ DOWNSAMPLING HAPPENS HERE, NOT IN THE BROWSER. Healpix downsampling is trivial
    arithmetic (RING -> NEST -> shift -> RING), but the browser cannot downsample a cube it
    has not downloaded, and at full resolution x 52 subbands x 32 elements a day is ~60 MB
    per chain. So the master keeps every pixel and this writes the viewer's working set.
    """
    os.makedirs(args.outdir, exist_ok=True)
    index = []
    for src in args.masters:
        z = np.load(src, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        nside0 = int(meta["nside"])
        nside = args.nside or nside0
        if nside > nside0:
            sys.exit("--nside %d is FINER than the master's %d: a cube cannot be "
                     "up-sampled, rebuild instead" % (nside, nside0))
        shift = 0
        while (nside0 >> shift) > nside:
            shift += 1
        chains, arrays = [], []
        for i, c in enumerate(meta["chains"]):
            pix, n, s1 = z["pix_%d" % i], z["n_%d" % i], z["s1_%d" % i]
            if shift:
                pix = ring_downgrade(pix, nside0, nside)
            # Bin subbands, then coalesce duplicate pixels created by the downgrade. Both are
            # plain sums BECAUSE the stored quantity is an accumulator -- this is the payoff
            # for never storing dB.
            n, s1, sub_fid = bin_subbands(n, s1, meta["chains"][i]["freq_ids"], args.subbands)
            pixu, inv = np.unique(pix, return_inverse=True)
            if len(pixu) != len(pix):
                nn = np.zeros(n.shape[:2] + (len(pixu),), n.dtype)
                ss = np.zeros(s1.shape[:2] + (len(pixu),))
                np.add.at(nn, (slice(None), slice(None), inv), n)
                np.add.at(ss, (slice(None), slice(None), inv), s1)
                n, s1 = nn, ss
            # PIXEL CENTRES SHIP WITH THE CUBE. The viewer needs an (az, el) for every pixel
            # to draw anything, and the alternative -- a second, hand-ported healpix in
            # JavaScript with nothing to check it against -- is exactly the kind of unverified
            # duplicate that put the beam in the wrong place once already. The page ports
            # ang2pix for speed but VERIFIES the port against these centres on load and
            # refuses to draw if they disagree, so the arithmetic is checked, not trusted.
            az, el = pix_centres(pixu, nside)
            chains.append({"chain": c["chain"], "sys": c["sys"],
                           "n_sub": int(n.shape[0]), "n_elem": int(n.shape[1]),
                           "n_pix": int(len(pixu)), "freq_ids": sub_fid})
            arrays.append((pixu.astype(np.int32),
                           np.stack([az, el], 1).astype(np.float32).ravel(),
                           n.astype(np.uint32), s1.astype(np.float32)))
        day = meta["day"]
        blob = b"".join(a.tobytes() for arr in arrays for a in arr)
        binp = os.path.join(args.outdir, "cube_%s.bin" % day)
        with open(binp, "wb") as fh:
            fh.write(blob)
        man = {"day": day, "nside": nside, "source": meta["source"],
               "veto_deg": meta["veto_deg"], "range_norm": meta["range_norm"],
               "chains": chains,
               # Byte offsets so the viewer slices one ArrayBuffer instead of parsing.
               "layout": "per chain, in order: pix int32[n_pix], centre float32[n_pix*2] "
                         "(az,el degrees), n uint32[n_sub*n_elem*n_pix], "
                         "s1 float32[n_sub*n_elem*n_pix]"}
        with open(os.path.join(args.outdir, "cube_%s.json" % day), "w") as fh:
            json.dump(man, fh)
        index.append({"day": day, "bytes": len(blob),
                      "chains": [c["chain"] for c in chains]})
        print("%s -> %s (%.2f MB, nside %d, %d chain(s))"
              % (src, binp, len(blob) / 1e6, nside, len(chains)))
    index.sort(key=lambda d: d["day"])
    with open(os.path.join(args.outdir, "index.json"), "w") as fh:
        json.dump({"days": index}, fh)
    print("index.json: %d day(s)" % len(index))


def pix_centres(pix, nside):
    """(az, el) in degrees for each RING pixel centre. EXPORT ONLY -- uses healpy.

    The build path deliberately does not import healpy (it runs in venv-ft, over whole days,
    and only needs ang2pix); the export runs once per day offline, so leaning on the reference
    implementation here costs nothing and removes a second place to get pix2ang wrong.
    """
    try:
        import healpy as hp
    except ImportError:
        sys.exit("export needs healpy for pixel centres: run it with "
                 "/home/kvand/gnss/venv/bin/python (NOT venv-ft). The build subcommand does "
                 "not need it.")
    theta, phi = hp.pix2ang(int(nside), np.asarray(pix, np.int64), nest=False)
    return np.degrees(phi) % 360.0, 90.0 - np.degrees(theta)


def ring_downgrade(pix, nside_in, nside_out):
    """RING -> RING at a coarser nside, via NEST where the downgrade is a bit shift."""
    return nest2ring(ring2nest(np.asarray(pix, np.int64), nside_in)
                     >> (2 * (int(np.log2(nside_in)) - int(np.log2(nside_out)))), nside_out)


def bin_subbands(n, s1, freq_ids, nbin):
    """Sum groups of subbands. nbin 0/None or >= n_sub keeps the axis as-is."""
    S = n.shape[0]
    if not nbin or nbin >= S:
        return n, s1, freq_ids
    edges = [int(round(i * S / nbin)) for i in range(nbin + 1)]
    nn = np.stack([n[a:b].sum(0) for a, b in zip(edges, edges[1:])])
    ss = np.stack([s1[a:b].sum(0) for a, b in zip(edges, edges[1:])])
    fid = [[freq_ids[a][0], freq_ids[b - 1][-1]] for a, b in zip(edges, edges[1:])]
    return nn, ss, fid


# --- RING <-> NEST. Only needed for the downgrade. ----------------------------------------
# nest2ring is the closed form (verified against healpy at nside 4..128, 20k random pixels
# each, zero mismatches). ring2nest is its INVERSE PERMUTATION rather than a second closed
# form: my hand-written ring->xyf inversion was wrong for 100% of pixels at every nside, and
# an exact inverse of a verified map cannot be wrong in a way the verified map is not. The
# permutation is npix entries (49,152 at nside 64) built once per nside and cached, which is
# cheaper than the arithmetic it replaces at the sizes this runs on.
_R2N_CACHE = {}


def _spread(v):
    v = np.asarray(v, np.int64) & 0xFFFFFFFF
    for s, m in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F),
                 (2, 0x3333333333333333), (1, 0x5555555555555555)):
        v = (v | (v << s)) & m
    return v


def _compress(v):
    v = np.asarray(v, np.int64) & 0x5555555555555555
    for s, m in ((1, 0x3333333333333333), (2, 0x0F0F0F0F0F0F0F0F), (4, 0x00FF00FF00FF00FF),
                 (8, 0x0000FFFF0000FFFF), (16, 0x00000000FFFFFFFF)):
        v = (v | (v >> s)) & m
    return v


def nest2ring(pix, nside):
    pix = np.asarray(pix, np.int64)
    order = int(round(np.log2(nside)))
    f = pix >> (2 * order)
    p = pix & ((1 << (2 * order)) - 1)
    x, y = _compress(p), _compress(p >> 1)
    jrll = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], np.int64)
    jpll = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], np.int64)
    jr = jrll[f] * nside - x - y - 1
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    out = np.zeros_like(pix)
    north, south = jr < nside, jr > 3 * nside
    eq = ~(north | south)

    def _jp(sel, nr):
        jp = (jpll[f[sel]] * nr + x[sel] - y[sel] + 1) // 2
        return np.where(jp > 4 * nr, jp - 4 * nr, np.where(jp < 1, jp + 4 * nr, jp))

    if np.any(north):
        nr = jr[north]
        out[north] = 2 * nr * (nr - 1) + _jp(north, nr) - 1
    if np.any(south):
        nr = 4 * nside - jr[south]
        out[south] = npix - 2 * nr * (nr + 1) + _jp(south, nr) - 1
    if np.any(eq):
        nr = np.int64(nside)
        kshift = (jr[eq] - nside) & 1
        jp = (jpll[f[eq]] * nr + x[eq] - y[eq] + 1 + kshift) // 2
        jp = np.where(jp > 4 * nside, jp - 4 * nside, np.where(jp < 1, jp + 4 * nside, jp))
        out[eq] = ncap + (jr[eq] - nside) * 4 * nside + jp - 1
    return out


def ring2nest(pix, nside):
    key = int(nside)
    if key not in _R2N_CACHE:
        npix = 12 * key * key
        nest = np.arange(npix, dtype=np.int64)
        inv = np.empty(npix, np.int64)
        inv[nest2ring(nest, key)] = nest      # ring index -> nest index, exactly
        _R2N_CACHE[key] = inv
    return _R2N_CACHE[key][np.asarray(pix, np.int64)]


def cmd_ls(args):
    for src in args.masters:
        z = np.load(src, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        print("%s  nside %d  source %s  veto %.1f deg  range_norm %s"
              % (m["day"], m["nside"], m["source"], m["veto_deg"], m["range_norm"]))
        for i, c in enumerate(m["chains"]):
            n = z["n_%d" % i]
            occ = int((n > 0).sum())
            print("   %-9s %s  sub %-3d elem %-3d pix %-6d  filled %d (%.1f%%)"
                  % (c["chain"], c["sys"], c["n_sub"], c["n_elem"], c["n_pix"],
                     occ, 100.0 * occ / max(1, n.size)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="archives -> master cube per day")
    b.add_argument("--days", nargs="+", required=True, help="YYYYMMDD")
    b.add_argument("--chains", nargs="*", help="default: every chain with an archive")
    b.add_argument("--source", choices=["elem", "cube"], default="elem")
    b.add_argument("--archive", default="/home/kvand/gnss/fixtures/obs")
    b.add_argument("--outdir", default="/home/kvand/gnss/fixtures/beamcube")
    b.add_argument("--nside", type=int, default=64)
    b.add_argument("--veto-deg", type=float, default=5.0,
                   help="cross-chain railing veto; 0 disables (and keeps the main lobe, "
                        "contaminated -- quote it as a lower bound if you do)")
    b.add_argument("--mask-deg", type=float, default=0.0)
    b.add_argument("--no-range-norm", action="store_true")
    b.set_defaults(func=cmd_build)

    e = sub.add_parser("export", help="master cubes -> browser cubes + index.json")
    e.add_argument("masters", nargs="+")
    e.add_argument("--outdir", default="/home/kvand/gnss/fixtures/beamcube/web")
    e.add_argument("--nside", type=int, default=16, help="0 = keep the master's")
    e.add_argument("--subbands", type=int, default=8, help="0 = keep every subband")
    e.set_defaults(func=cmd_export)

    l = sub.add_parser("ls", help="describe master cubes")
    l.add_argument("masters", nargs="+")
    l.set_defaults(func=cmd_ls)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
