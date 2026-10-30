#!/usr/bin/env python3
"""Beam data cube: response vs (chain, subband, element, healpix), built and exported.

    build   archives -> one master cube per UTC day (sparse .npz accumulators)
    export  master cubes -> per day a .json manifest and one .bin per chain, plus index.json,
            for the viewer

The cube stores what the instrument measures, per element and per channel; a map is a collapse
over some of those axes, and the viewer does the summing. A combined number can be rebuilt
from the parts, never the reverse.

The beam is the debiased power <|A_e|^2> (`p2` in the element archive, `incoh/w` in the cube
archive). It is not `|u|/q`: that is a ratio to the array mean, flat across the sky by
construction, and mapping it gives a plausible but meaningless ~6 dB all-sky pattern (a 6 m
dish at 1176 MHz cannot have one). See docs/CHORD_BEAM_MAPS.md §2.

What is stored is an accumulator, never a dB. Each cell keeps (n, s1, s2) over linear debiased
power, so cells, subbands, elements, chains and days combine by addition, and adding a night
never reprocesses an old one. dB is taken once, in the viewer, at the end.

Two sources, one output format:
  --source elem  (default)  fixtures/obs/elem_<chain>_<YYYYMMDD>.jsonl
        The element archive. Per element, but the covering channels are already summed
        upstream, so each chain's subband axis has length 1; the chains still give coarse
        frequency (5 distinct bands).
  --source l0    <archive>/<sender>/<YYYYMMDD>.h5   (archive = .../gnss_cube/l0/<pointing>)
        The beam-cube archive (docs/CHORD_CUBE_ARCHIVE.md): per (subband, element) rows on
        ~1 s windows, pushed by every instance and folded by gnss_cube_compact.py, one file per
        sender per day. A rung directory (.../rung12/<pointing>) is accepted too: its rows
        carry the same w and incoh, summed over 12 or 60 windows, so far fewer rows to read.
        Rows carry both sums: `incoh` is the beam and is what this reads; `coh` (L0) and
        `cohref` (rung) are the coherent sums, for calibration.
        The archive root is a pointing: one master cube is one pointing, the path names it,
        the attrs confirm it, and two pointings in one build are refused.
        No probe flag travels in the cube frame, so the pedestal comes from the rows: per
        (sender, subband, element, 5 min), the median mean power over every slot. The beam is
        narrow, so a satellite is in the main lobe for minutes of a pass and 10-20 dB under
        the noise in the sidelobes for the rest; nearly every slot is noise at any moment and
        the median does not see the bright one. The bias from many faint sidelobe satellites
        is a few percent of the pedestal, far under the ~20 dB the map spans.

The railing veto is cross-chain. A satellite within --veto-deg (default 5) of boresight rails
the 4+4b quantiser for every chain at once, since they share the same nibbles; a per-chain
veto would clean the chain tracking it and leave the others contaminated while they look like
clean controls. The veto is computed from BRDC geometry over every constellation, independent
of what any chain tracks, and applied to every chain for that minute. It removes the main lobe
itself: the main lobe is what rails the quantiser.

@author Keith Vanderlinde
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all  # noqa: E402

LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
R_REF = 20.2e6  # m: the dB zero for range normalisation (semi-synchronous slant)
import gnss_arraymap  # noqa: E402  (same directory)
import gnss_chain_health  # noqa: E402  (same directory)

# The p0_dec40p73 boresight (docs/CHORD_BEAM_MAPS.md §5; not telescope.dish_coelev_deg), for a
# master built without bore_az_deg/bore_el_deg. The build takes the boresight from the epoch.
BORE_AZ, BORE_EL = 180.0, 81.41
MIN_ELEMS = 8  # a sample with fewer live elements says nothing about the beam
FLOOR_BIN_S = 300.0  # pedestal bin: per (instance, subband, element, 5 min)
ENV_HALF_BINS = 24  # pedestal reference: low envelope over +-2 h of pedestal bins
ENV_PCT = 10.0
HEALTH_DIR = (
    "/home/kvand/gnss/fixtures/obs/health"  # gnss_chain_health.py masks, per day
)
GEOM_BIN_S = 60.0  # BRDC evaluated once per minute, at its midpoint; no interpolation

# Chain -> constellation letter, for the BRDC lookup. Declared, not inferred from the
# frequency: constellations share carriers.
CHAIN_SYS = {
    "gps_l5": "G",
    "gps_l2c": "G",
    "gal_e5a": "E",
    "gal_e5b": "E",
    "gal_e6": "E",
    "bds_b2a": "C",
    "bds_b2b": "C",
    "bds_b3i": "C",
    "glo_l3oc": "R",
    "glo_l2oc": "R",
    "glo_l2of": "R",
}

# Cube-archive sender ("cx19_gnss0_b2b_n2assemble") -> chain. The band token is the stage's
# infix; the GPS L5 stage carries none. Declared, like CHAIN_SYS: a sender with an unlisted
# infix maps to no chain, and find_inputs does not read its files.
SENDER_BAND_CHAIN = {
    "": "gps_l5",
    "l2c": "gps_l2c",
    "e5a": "gal_e5a",
    "e5b": "gal_e5b",
    "e6": "gal_e6",
    "b2a": "bds_b2a",
    "b2b": "bds_b2b",
    "b3i": "bds_b3i",
    "l3oc": "glo_l3oc",
    "l2oc": "glo_l2oc",
    "l2of": "glo_l2of",
}


def sender_chain(sender):
    parts = sender.split("_")
    if len(parts) < 3 or not parts[-1].startswith("n2assemble"):
        return None
    band = parts[2] if len(parts) >= 4 else ""
    return SENDER_BAND_CHAIN.get(band)


# ─────────────────────────────────────────────────────────────────────────────────────────
# healpix. Only ang2pix RING is needed, so it is inlined and the build needs no healpy
# (--source elem runs in venv-ft; --source l0 needs h5py, from /home/kvand/gnss/venv).
# tests/test_beam_cube.py checks it against healpy; do not trust a map while that test fails.
# ─────────────────────────────────────────────────────────────────────────────────────────
def ang2pix_ring(nside, theta, phi):
    """HEALPix RING pixel index for colatitude theta and longitude phi (radians, arrays)."""
    nside = int(nside)
    npix = 12 * nside * nside
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2.0 * np.pi) * (2.0 / np.pi)  # in [0, 4)
    pix = np.empty(theta.shape, dtype=np.int64)

    eq = za <= 2.0 / 3.0
    # --- equatorial belt -------------------------------------------------------------
    if np.any(eq):
        t, zz = tt[eq], z[eq]
        temp1 = nside * (0.5 + t)
        temp2 = nside * zz * 0.75
        jp = (temp1 - temp2).astype(np.int64)  # ascending edge
        jm = (temp1 + temp2).astype(np.int64)  # descending edge
        ir = nside + 1 + jp - jm  # in {1, 2n+1}
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
        p = np.where(north, 2 * ir * (ir - 1) + ip, npix - 2 * ir * (ir + 1) + ip)
        pix[~eq] = p
    return pix


def azel_to_pix(nside, az_deg, el_deg):
    """Local horizon frame: colatitude from zenith, longitude = azimuth (N=0, clockwise).

    The local frame, not celestial: the beam is fixed to the dish, so a celestial
    pixelisation would smear a stationary pattern across the sky as the earth turns.
    """
    theta = np.radians(90.0 - np.asarray(el_deg, dtype=float))
    phi = np.radians(np.asarray(az_deg, dtype=float))
    return ang2pix_ring(nside, theta, phi)


def angsep_deg(az1, el1, az2, el2):
    a1, e1, a2, e2 = (np.radians(x) for x in (az1, el1, az2, el2))
    c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ─────────────────────────────────────────────────────────────────────────────────────────
# Geometry, shared by every chain of a day so the veto is cross-chain.
# ─────────────────────────────────────────────────────────────────────────────────────────
class Geometry:
    """BRDC positions on a 1-minute grid, plus the cross-chain railing veto for that minute."""

    def __init__(self, day_unix, veto_deg, bore=(BORE_AZ, BORE_EL), keep_own=False):
        self.eph = parse_rinex_nav(
            fetch_brdc(datetime.fromtimestamp(day_unix + 43200, tz=timezone.utc))
        )
        self.veto_deg = veto_deg
        self.keep_own = keep_own
        self.bore = bore  # (az, el) deg, the array epoch's boresight
        self._pos = {}  # (sys, prn, minute) -> (el, az, range_m) or None
        self._all = {}  # minute -> {(sys, prn): v}
        self._veto = {}  # minute -> (vetoed, closest_deg, closest_name, names within)

    def _minute(self, minute):
        if minute not in self._all:
            try:
                self._all[minute] = (
                    predict_all(
                        self.eph,
                        LAT,
                        LON,
                        ALT,
                        minute * GEOM_BIN_S + GEOM_BIN_S / 2,
                        mask_deg=-90.0,
                        max_age=86400.0,
                    )
                    or {}
                )
            except Exception:
                self._all[minute] = {}
        return self._all[minute]

    def at(self, sys_, prn, t):
        key = (sys_, prn, int(t // GEOM_BIN_S))
        if key not in self._pos:
            v = self._minute(key[2]).get((sys_, prn))
            self._pos[key] = (v["el"], v["az"], v["range_m"]) if v else None
        return self._pos[key]

    def vetoed(self, t, own=None):
        """True if any satellite of any constellation is within veto_deg of boresight.

        Pooled over constellations because the chains share the 4+4b nibbles: a veto computed
        per chain would clean the one carrying the bright satellite and leave the rest
        contaminated while they look like clean controls.

        With keep_own, a row whose OWN satellite `own` ("G3", "E16": sys letter + PRN) is the
        CLOSEST to boresight survives: that is the main lobe, railed, which the plain veto
        removes entirely (the 10-28 element-archive maps kept it; its core is an upper bound on
        the beam width). Rows of every other satellite in such a minute are still dropped. The
        closest, not "the only one within": a boresight pass (G13 at 0.12 deg, 10-29) is lost
        otherwise whenever a second satellite sits at 3-5 deg, and the closest is the one doing
        the railing.
        """
        if self.veto_deg <= 0.0:
            return False
        m = int(t // GEOM_BIN_S)
        if m not in self._veto:
            best, name, within = 1e9, None, set()
            for (sy, prn), v in self._minute(m).items():
                if v is None or v.get("el") is None or v["el"] < 0.0:
                    continue
                d = float(angsep_deg(v["az"], v["el"], *self.bore))
                if d < best:
                    best, name = d, "%s%d" % (sy, prn)
                if d < self.veto_deg:
                    within.add("%s%d" % (sy, prn))
            self._veto[m] = (best < self.veto_deg, best, name, within)
        if self.keep_own and own is not None:
            return self._veto[m][0] and own != self._veto[m][2]
        return self._veto[m][0]

    def veto_stats(self):
        n = len(self._veto)
        v = sum(1 for x in self._veto.values() if x[0])
        return n, v


# ─────────────────────────────────────────────────────────────────────────────────────────
# The accumulator. Sparse over pixels: a day of tracks touches a small fraction of the sky.
# ─────────────────────────────────────────────────────────────────────────────────────────
class CubeAccum:
    def __init__(self, nside, n_elem):
        self.nside = int(nside)
        self.n_elem = int(n_elem)
        self.n_sub = 0
        self.cells = {}  # (subband, pixel) -> [n(E), s1(E), s2(E)]

    def add(self, sub, pix, power, live):
        """power: [n_elem] linear debiased power; live: [n_elem] bool (dark elements excluded)."""
        self.n_sub = max(self.n_sub, sub + 1)
        c = self.cells.get((sub, pix))
        if c is None:
            c = [
                np.zeros(self.n_elem, np.uint32),
                np.zeros(self.n_elem),
                np.zeros(self.n_elem),
            ]
            self.cells[(sub, pix)] = c
        c[0][live] += 1
        c[1][live] += power[live]
        c[2][live] += power[live] ** 2

    def add_bulk(self, sub, pix, power, live):
        """Vectorised add: sub[K], pix[K] per sample, power[K,E], live[K,E]."""
        if len(pix) == 0:
            return
        key = np.stack([np.asarray(sub, np.int64), np.asarray(pix, np.int64)], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        pw = np.where(live, power, 0.0)
        n = np.zeros((len(uk), self.n_elem), np.uint32)
        s1 = np.zeros((len(uk), self.n_elem))
        s2 = np.zeros((len(uk), self.n_elem))
        np.add.at(n, inv, live.astype(np.uint32))
        np.add.at(s1, inv, pw)
        np.add.at(s2, inv, pw ** 2)
        self.n_sub = max(self.n_sub, int(uk[:, 0].max()) + 1)
        for j, (sb, px) in enumerate(uk):
            c = self.cells.get((int(sb), int(px)))
            if c is None:
                self.cells[(int(sb), int(px))] = [n[j], s1[j], s2[j]]
            else:
                c[0] += n[j]
                c[1] += s1[j]
                c[2] += s2[j]

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
    """Probe pedestal per (instance, subband, element, 5-min bin), as a median.

    Per element and per subband, never medianed across either: elements differ in gain and
    bands differ in noise figure, so a floor averaged over them over-subtracts the quiet ones
    into negative power.
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
    """Per-subband list of per-element power for one elem-archive row: `p2` is [n_elem],
    so there is one subband."""
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
                    continue  # a probe is the floor, never a beam sample
                rows = _rows(d)
                if not rows or len(rows[0]) < MIN_ELEMS:
                    stats["short"] += 1
                    continue
                if acc is None:
                    acc = CubeAccum(args.nside, len(rows[0]))
                    freq_ids = d.get("bin_freq_ids") or [[None, None]] * len(rows)
                stats["rows"] += 1
                if geom.vetoed(t, "%s%d" % (sys_, int(d["prn"]))):
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
                # Range normalisation in power: from spreading loss alone a satellite is ~2 dB
                # stronger overhead than at the horizon, which would masquerade as beam.
                # Received power falls as 1/r^2, so each sample is scaled by (r/R_REF)^2.
                rscale = 1.0 if args.no_range_norm else (rng / R_REF) ** 2
                for sub, row in enumerate(rows):
                    fl = floors.get((d.get("inst"), sub, int(t // FLOOR_BIN_S))) or {}
                    p = np.zeros(acc.n_elem)
                    live = np.zeros(acc.n_elem, bool)
                    for e in range(min(acc.n_elem, len(row))):
                        p2e, f = row[e], fl.get(e)
                        # Debias in power, and drop a sample below its own pedestal rather
                        # than clamp it: clamping piles non-detections at an artificial floor
                        # and manufactures a beam edge exactly where the beam ends.
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
    print(
        "  %s: %d row(s) -> %d subband(s) x %d element(s) x %d pixel(s); "
        "vetoed %d, no-geom %d, below-mask %d, thin %d"
        % (
            chain,
            stats["rows"],
            n.shape[0],
            n.shape[1],
            len(pix),
            stats["vetoed"],
            stats["nogeo"],
            stats["low"],
            stats["thin"],
        )
    )
    return dict(pix=pix, n=n, s1=s1, s2=s2, freq_ids=freq_ids[: n.shape[0]])


def pedestal(P, tb, inday, ref="envelope"):
    """Per row of P[R,S,E]: the pedestal, the median over every slot in the row's 5-min bin per
    subband and element, and the reference a sample is divided by.

    The pedestal is the despread noise at that moment, so it carries every satellite near the
    beam: on 10-29 it rose x3-10 while a satellite crossed boresight and wandered 0.7-1.4 dB
    between quiet bins 10-20 min apart. It stays the right thing to subtract. "envelope" divides
    by its low envelope instead, the ENV_PCT percentile over +-ENV_HALF_BINS bins (0.6-2 dB over
    the day, < 0.1 dB bin to bin); "bin" divides by the pedestal itself.
    """
    floor = np.full_like(P, np.nan)
    refl = np.full_like(P, np.nan)
    bins = np.unique(tb[inday])
    fb = {}
    for b in bins:
        with np.errstate(all="ignore"):
            fb[b] = np.nanmedian(P[(tb == b) & inday], axis=0)
    for b in bins:
        m = (tb == b) & inday
        floor[m] = fb[b][None]
        if ref == "bin":
            refl[m] = fb[b][None]
            continue
        near = np.array([fb[c] for c in bins if abs(c - b) <= ENV_HALF_BINS])
        with np.errstate(all="ignore"):
            refl[m] = np.nanpercentile(near, ENV_PCT, axis=0)[None]
    return floor, refl


def _h5_rows(path):
    """One L0 or rung file -> (attrs, t[R], prn[R], P[R,S,E] mean power (nan: bin unused)).

    `incoh/w` is the mean |A_e|^2 per (record, channel) term in both formats: a rung row is
    the window sums added, and a ratio of sums is still a mean. Time is the window's UTC
    (L0: through the window table, exact, since every row idx is a window idx) or the
    mid-point of the row's first and last window (rung).
    """
    import h5py

    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        fmt = a.get("format")
        prn = f["rows/prn"][:].astype(int)
        w = f["rows/w"][:].astype(np.float64)
        incoh = f["rows/incoh"][:].astype(np.float64)
        if fmt == "gnss_cube_l0":
            idx = f["rows/idx"][:]
            widx, wutc = f["win/idx"][:], f["win/utc"][:]
            o = np.argsort(widx)
            t = np.interp(idx, widx[o], wutc[o])
        elif fmt == "gnss_cube_rung":
            t = 0.5 * (f["rows/utc0"][:] + f["rows/utc1"][:])
        else:
            raise SystemExit("%s: not a cube archive file (format=%r)" % (path, fmt))
    P = np.where(
        (w > 0.0)[:, :, None], incoh / np.maximum(w, 1e-30)[:, :, None], np.nan
    )
    return a, t, prn, P


def _h5_freq_ids(path):
    """Absolute F-engine channel per bin of one archive file (list of int), or None.

    Each bin must be one channel (freq_id_lo == freq_id_hi), or this exits. A rung without
    freq_id attrs takes them from the L0 file it was folded from, at the same
    <pointing>/<sender>/<day>.h5 under l0/."""
    import h5py

    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format")
        lo = (
            f["win/freq_id_lo"][0]
            if fmt == "gnss_cube_l0"
            else f.attrs.get("freq_id_lo")
        )
        hi = (
            f["win/freq_id_hi"][0]
            if fmt == "gnss_cube_l0"
            else f.attrs.get("freq_id_hi")
        )
    if lo is None:
        parts = os.path.normpath(path).split(os.sep)
        for i, seg in enumerate(parts):
            if seg.startswith("rung"):
                l0p = os.sep.join(parts[:i] + ["l0"] + parts[i + 1 :])
                if os.path.exists(l0p):
                    with h5py.File(l0p, "r") as f0:
                        lo, hi = f0["win/freq_id_lo"][0], f0["win/freq_id_hi"][0]
                break
    if lo is None:
        return None
    if hi is not None and list(lo) != list(hi):
        sys.exit(
            "%s: a bin spans channels %s..%s -- the archive promised one channel per bin"
            % (path, list(lo), list(hi))
        )
    return [int(x) for x in lo]


def prn_gain_table(args, chain, paths, day_unix, geom, dth=5.0, dphi=90.0, iters=12):
    """Per-PRN level offset in dB, solved jointly with a coarse beam by median polish.

    Received power goes as G_ant(direction) * EIRP_sat(off-nadir) / r^2. The build divides out
    each element's noise pedestal and corrects range, but not the satellite's own transmit
    power and antenna pattern, so without this each satellite paints its track at its own
    level and the map grows arcs along the bright ones. The per-PRN level is a property of the
    satellite and repeats from day to day.

    Solved jointly, not as each satellite's own mean: a satellite that only crosses a bright
    part of the beam would absorb that beam signal into its gain. Median polish alternates
    gain(prn) and beam(cell) on a coarse grid, so a satellite is referenced to the others
    crossing the same cells. Medians, so a railing pass does not drag its own gain up.

    The grid must be coarse enough that satellites share cells. With one PRN in a cell,
    beam = value - gain absorbs everything and the gain solves to zero, which looks like
    "nothing to correct". Over one day a healpix nside-16 cell is typically crossed by one
    satellite; the default dth 5 x dphi 90 deg cells about boresight are crossed by several.
    theta x phi rather than healpix because the beam is mostly radial; dphi 90 keeps
    quadrant-scale azimuthal structure (the cos(2*PA) elongation) in the beam, not the gains.
    A median occupancy under 3 PRNs per cell is reported on stderr.

    The gains are centred on their median, so the map's absolute level is unchanged.
    """
    sys_ = CHAIN_SYS[chain]
    lvl = defaultdict(list)  # (prn, coarse cell) -> [log10 power]
    raw = defaultdict(list)  # prn -> [log10 power], for the printed raw spread
    for path in paths:
        try:
            a, t, prn, P = _h5_rows(path)
        except SystemExit:
            raise
        except Exception:
            continue
        if P.size == 0:
            continue
        inday = (t >= day_unix) & (t < day_unix + 86400.0)
        if not inday.any():
            continue
        tb = ((t - day_unix) // FLOOR_BIN_S).astype(int)
        floor, refl = pedestal(P, tb, inday, getattr(args, "pedestal_ref", "envelope"))
        minute = (t // GEOM_BIN_S).astype(int)
        uk, inv = np.unique(np.stack([prn, minute], 1), axis=0, return_inverse=True)
        inv = inv.ravel()
        el = np.full(len(uk), np.nan)
        az = np.full(len(uk), np.nan)
        rng = np.full(len(uk), np.nan)
        vet = np.zeros(len(uk), bool)
        for j, (pn, mn) in enumerate(uk):
            tt = mn * GEOM_BIN_S + GEOM_BIN_S / 2
            vet[j] = geom.vetoed(tt, "%s%d" % (sys_, int(pn)))
            g = geom.at(sys_, int(pn), tt)
            if g and g[0] is not None:
                el[j], az[j], rng[j] = g
        vet_r, el_r, az_r, rng_r = vet[inv], el[inv], az[inv], rng[inv]
        ok = inday & ~vet_r & ~np.isnan(el_r) & (el_r > args.mask_deg)
        # The chain-health mask, as in build_day_h5: a masked bin is noise, not a level.
        ok &= ~gnss_chain_health.unhealthy(getattr(args, "_health", {}), chain, t)
        if not ok.any():
            continue
        rscale = (
            np.ones(int(ok.sum())) if args.no_range_norm else (rng_r[ok] / R_REF) ** 2
        )
        Pk, Fk, Rk = P[ok], floor[ok], refl[ok]
        with np.errstate(all="ignore"):
            live = (Pk > 0.0) & (Fk > 0.0) & (Rk > 0.0) & (Pk > Fk)
            pw = np.where(
                live,
                (Pk - Fk) / np.where(live, Rk, 1.0) * rscale[:, None, None],
                np.nan,
            )
            row = np.nanmean(
                pw, axis=(1, 2)
            )  # pool subbands+elements: one level per row
        # theta (off boresight) and phi (position angle about it), the beam's own coordinates.
        aa, ee = az_r[ok], el_r[ok]
        th = angsep_deg(aa, ee, *geom.bore)
        ca, ce = math.radians(geom.bore[0]), math.radians(geom.bore[1])
        u = np.stack(
            [
                np.cos(np.radians(ee)) * np.sin(np.radians(aa)),
                np.cos(np.radians(ee)) * np.cos(np.radians(aa)),
                np.sin(np.radians(ee)),
            ],
            -1,
        )
        nh = np.array(
            [-math.sin(ce) * math.sin(ca), -math.sin(ce) * math.cos(ca), math.cos(ce)]
        )
        ea = np.array([math.cos(ca), -math.sin(ca), 0.0])
        ph = np.degrees(np.arctan2(u @ ea, u @ nh)) % 360.0
        cellkey = np.floor(th / dth).astype(np.int64) * 1000 + np.floor(
            ph / dphi
        ).astype(np.int64)
        good = np.isfinite(row) & (row > 0.0)
        with np.errstate(all="ignore"):
            lg = np.log10(np.where(good, row, 1.0))
        for p_, x_, v_ in zip(prn[ok][good], cellkey[good], lg[good]):
            lvl[(int(p_), int(x_))].append(float(v_))
            raw[int(p_)].append(float(v_))
    if not lvl:
        return {}
    cell = {k: float(np.median(v)) for k, v in lvl.items() if len(v) >= 3}
    prns = sorted({k[0] for k in cell})
    pixs = sorted({k[1] for k in cell})
    by_prn = defaultdict(list)
    by_pix = defaultdict(list)
    for (p_, x_), v_ in cell.items():
        by_prn[p_].append((x_, v_))
        by_pix[x_].append((p_, v_))
    gain = {p_: 0.0 for p_ in prns}
    beam = {x_: 0.0 for x_ in pixs}
    for _ in range(iters):
        for x_ in pixs:
            r = [v - gain[p_] for p_, v in by_pix[x_]]
            if r:
                beam[x_] = float(np.median(r))
        for p_ in prns:
            r = [v - beam[x_] for x_, v in by_prn[p_]]
            if r:
                gain[p_] = float(np.median(r))
    mid = float(np.median(list(gain.values())))
    out = {p_: 10.0 * (gain[p_] - mid) for p_ in prns}  # dB, centred
    sp = np.array(list(out.values()))
    occ = np.array([len({k[0] for k in cell if k[1] == x_}) for x_ in pixs])
    # The raw per-PRN spread is printed for reference, not as a check: a satellite that stays
    # at high theta is faint because the beam is faint there, and the fit rightly leaves that
    # in the beam, so the solved and raw spreads are not expected to match.
    rawmed = {p_: float(np.median(v)) for p_, v in raw.items() if len(v) >= 200}
    rawsp = (
        10.0 * float(np.std(list(rawmed.values()))) if len(rawmed) > 3 else float("nan")
    )
    print(
        "  %s: per-PRN gains over %d satellite(s), %d coarse cell(s) "
        "(median %d PRN/cell): spread %.2f dB p-p, %.2f dB rms; raw per-PRN rms %.2f dB"
        % (
            chain,
            len(prns),
            len(pixs),
            int(np.median(occ)) if len(occ) else 0,
            sp.max() - sp.min(),
            sp.std(),
            rawsp,
        )
    )
    # The guard is occupancy: with too few PRNs per cell the fit is degenerate and returns ~0
    # whatever the truth, so a small answer is trusted only when the cells are populated.
    med_occ = int(np.median(occ)) if len(occ) else 0
    if med_occ < 3:
        print(
            "  WARNING: %s: median %d PRN per cell -- the (gain, beam) fit is under-determined "
            "and will return ~0 whatever the truth is. Coarsen dth/dphi before believing this."
            % (chain, med_occ),
            file=sys.stderr,
        )
    return out


def build_day_h5(args, chain, paths, day_unix, geom, prn_gain=None):
    """Master cube for one chain from cube-archive files (L0 or rung), one sender per file."""
    tmin, tmax = day_unix, day_unix + 86400.0
    sys_ = CHAIN_SYS[chain]
    acc, pointing = None, None
    stats = defaultdict(int)
    # The subband axis is the absolute F-engine channel, never the instance's bin index: each
    # instance holds the few channels its node was dealt, so bin 0 of two instances are two
    # different frequencies. The axis is the union of freq_ids over every sender file of the
    # chain, and each file's bins land on their own channel.
    fid_by_path = {}
    for path in paths:
        fid_by_path[path] = _h5_freq_ids(path)
        if fid_by_path[path] is None:
            sys.exit(
                "%s: no freq ids -- the subband axis cannot be built without them"
                % path
            )
    band = sorted(set(x for v in fid_by_path.values() for x in v))
    sub_of = {fid: i for i, fid in enumerate(band)}
    freq_ids = [[fid, fid] for fid in band]
    print(
        "  %s: %d distinct channel(s) freq_id %d..%d across %d sender file(s)"
        % (chain, len(band), band[0], band[-1], len(paths))
    )
    for path in paths:
        a, t, prn, P = _h5_rows(path)
        pt = str(a.get("pointing"))
        if pointing is None:
            pointing = pt
        elif pt != pointing:
            sys.exit(
                "%s: pointing %s in a build of %s -- one master is one pointing"
                % (path, pt, pointing)
            )
        R, S, E = P.shape
        sub_idx = [sub_of[f] for f in fid_by_path[path]]
        if acc is None:
            acc = CubeAccum(args.nside, E)
            acc.n_sub = len(band)
        inday = (t >= tmin) & (t < tmax)
        stats["rows"] += int(inday.sum())
        # -- pedestal (module docstring) and the reference a sample is divided by -------------
        tb = ((t - day_unix) // FLOOR_BIN_S).astype(int)
        floor, refl = pedestal(P, tb, inday, getattr(args, "pedestal_ref", "envelope"))
        # -- geometry per (prn, minute); veto per minute -------------------------------------
        minute = (t // GEOM_BIN_S).astype(int)
        keys = np.stack([prn, minute], 1)
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        el = np.full(len(uk), np.nan)
        az = np.full(len(uk), np.nan)
        rng = np.full(len(uk), np.nan)
        vet = np.zeros(len(uk), bool)
        for j, (pn, mn) in enumerate(uk):
            tt = mn * GEOM_BIN_S + GEOM_BIN_S / 2
            vet[j] = geom.vetoed(tt, "%s%d" % (sys_, int(pn)))
            g = geom.at(sys_, int(pn), tt)
            if g and g[0] is not None:
                el[j], az[j], rng[j] = g
        vet_r, el_r, az_r, rng_r = vet[inv], el[inv], az[inv], rng[inv]
        stats["vetoed"] += int((inday & vet_r).sum())
        stats["nogeo"] += int((inday & ~vet_r & np.isnan(el_r)).sum())
        ok = inday & ~vet_r & ~np.isnan(el_r)
        stats["low"] += int((ok & (el_r <= args.mask_deg)).sum())
        ok &= el_r > args.mask_deg
        # The cube archive carries no lock flag, so a chain that lost the fleet and kept
        # emitting records writes noise the pedestal cannot tell from a weak sky. The
        # chain-health mask (gnss_chain_health.py: the observables writers' fleet_present
        # fraction per 5-min bin) records it; masked bins are dropped, per chain.
        unhealthy = gnss_chain_health.unhealthy(getattr(args, "_health", {}), chain, t)
        stats["unhealthy"] += int((ok & unhealthy).sum())
        ok &= ~unhealthy
        if not ok.any():
            continue
        pix = azel_to_pix(args.nside, az_r[ok], el_r[ok]).astype(np.int64)
        rscale = (
            np.ones(int(ok.sum())) if args.no_range_norm else (rng_r[ok] / R_REF) ** 2
        )
        Pk, Fk, Rk = P[ok], floor[ok], refl[ok]
        # Debias in power and drop a sample under its own pedestal (never clamp; see
        # build_day), in units of the pedestal reference. The cube's |A_e|^2 is normalised by
        # the channel's own replica energy (A = G/E_c), so a bin's scale is 1/E_c^2: under a
        # BPSK(10) main lobe the centre channels sit ~20 dB below the edge ones, and a raw sum
        # over subbands would be just the edge bins. Dividing by each (bin, element)'s
        # pedestal reference makes a cell "received power per element over that element's
        # noise in that channel": dimensionless, and comparable across bins, elements and
        # chains.
        # --unbiased keeps the samples under the pedestal too: (P-F)/F averaged with its sign
        # is an unbiased pattern estimate whose noise falls with more samples, while the drop
        # leaves a positive floor of ~0.8 sigma per cell (the half-normal mean) that no amount
        # of data lowers. The drop is the default and the elem-archive convention; an unbiased
        # cell can go negative, and the viewer leaves blank a pixel whose summed s1 is <= 0.
        with np.errstate(all="ignore"):
            valid = (Pk > 0.0) & (Fk > 0.0) & (Rk > 0.0)
            live = valid if args.unbiased else (valid & (Pk > Fk))
            # Per-satellite level, if solved, applied in power alongside the range scale so it
            # divides out the satellite's EIRP before the sample reaches a pixel. A PRN with no
            # solved gain (too few cells) passes through at 1.0 and is counted in
            # stats["prn_ungained"].
            if prn_gain:
                gdb = np.array([prn_gain.get(int(p_), np.nan) for p_ in prn[ok]])
                stats["prn_ungained"] += int(np.isnan(gdb).sum())
                gsc = np.where(
                    np.isfinite(gdb), 10.0 ** (-np.nan_to_num(gdb) / 10.0), 1.0
                )
                rscale = rscale * gsc
            pw = np.where(
                live, (Pk - Fk) / np.where(live, Rk, 1.0) * rscale[:, None, None], 0.0
            )
        for sub in range(S):
            lv = live[:, sub, :]
            enough = lv.sum(1) >= MIN_ELEMS
            stats["thin"] += int((~enough & (lv.sum(1) > 0)).sum())
            if enough.any():
                acc.add_bulk(
                    np.full(int(enough.sum()), sub_idx[sub]),
                    pix[enough],
                    pw[enough, sub, :],
                    lv[enough],
                )
                stats["cells"] += int(enough.sum())
        print(
            "    %s: %d rows, %d used"
            % (os.path.basename(os.path.dirname(path)), int(inday.sum()), int(ok.sum()))
        )
    if acc is None or not acc.cells:
        print("  %s: no usable rows (%s)" % (chain, dict(stats)))
        return None
    pix, n, s1, s2 = acc.arrays()
    print(
        "  %s: %d row(s) -> %d subband(s) x %d element(s) x %d pixel(s); "
        "vetoed %d, no-geom %d, below-mask %d, thin %d; pointing %s"
        % (
            chain,
            stats["rows"],
            n.shape[0],
            n.shape[1],
            len(pix),
            stats["vetoed"],
            stats["nogeo"],
            stats["low"],
            stats["thin"],
            pointing,
        )
    )
    if stats["unhealthy"]:
        print(
            "  %s: %d row(s) dropped by the chain-health mask"
            % (chain, stats["unhealthy"])
        )
    assert n.shape[0] == len(band), (n.shape, len(band))
    return dict(
        pix=pix,
        n=n,
        s1=s1,
        s2=s2,
        freq_ids=freq_ids,
        pointing=pointing,
        units="pedestal"
        + ("_env" if getattr(args, "pedestal_ref", "envelope") == "envelope" else "")
        + ("_unbiased" if args.unbiased else ""),
    )


def find_inputs(args, chain, daystr):
    if args.source == "l0":
        import glob

        out = []
        for p in sorted(glob.glob(os.path.join(args.archive, "*", daystr + ".h5"))):
            if sender_chain(os.path.basename(os.path.dirname(p))) == chain:
                out.append(p)
        return out
    p = os.path.join(args.archive, "elem_%s_%s.jsonl" % (chain, daystr))
    return [p] if os.path.exists(p) else []


def cmd_build(args):
    os.makedirs(args.outdir, exist_ok=True)
    for daystr in args.days:
        day_unix = (
            datetime.strptime(daystr, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
        )
        # Which array was this day? Resolved first: a recabling or re-pointing inside the day
        # would make the master a sum of two arrays that nothing later can unmix, so
        # gnss_arraymap.for_day refuses a day that straddles an epoch change.
        try:
            epoch = gnss_arraymap.for_day(daystr)
        except gnss_arraymap.Refused as exc:
            sys.exit(
                "%s: %s\n  Add the interval to %s (then gnss_arraymap.py check) before "
                "building this day." % (daystr, exc, gnss_arraymap.CONFIG)
            )
        chains = args.chains or sorted(CHAIN_SYS)
        present = [(c, find_inputs(args, c, daystr)) for c in chains]
        present = [(c, p) for c, p in present if p]
        if not present:
            print(
                "%s: no %s archives found in %s" % (daystr, args.source, args.archive)
            )
            continue
        print("%s: %d chain(s)" % (daystr, len(present)))
        bore = (epoch.pointing.bore_az_deg, epoch.pointing.bore_el_deg)
        geom = Geometry(
            day_unix,
            args.veto_deg,
            bore,
            keep_own=getattr(args, "veto_keep_own", False),
        )
        out = {
            "nside": args.nside,
            "day": daystr,
            "source": args.source,
            "veto_deg": args.veto_deg,
            "veto_keep_own": bool(getattr(args, "veto_keep_own", False)),
            "pedestal_ref": getattr(args, "pedestal_ref", "envelope"),
            "range_norm": not args.no_range_norm,
            "array_epoch": epoch.name,
            "array_epoch_key": epoch.key(),
            "bore_az_deg": bore[0],
            "bore_el_deg": bore[1],
            "chains": [],
        }
        print(
            "  array epoch %s%s"
            % (epoch.key(), "" if epoch.verified else "  (UNVERIFIED)")
        )
        # Only build_day_h5 applies the chain-health mask, so an elem build neither loads nor
        # records one: a mask in the meta would claim a veto that did not happen.
        hm = args.health_mask if args.source == "l0" else None
        if args.source != "l0" and args.health_mask not in ("auto", ""):
            print("  health mask %s: not applied (--source elem)" % args.health_mask)
        if hm == "auto":
            hm = os.path.join(HEALTH_DIR, "mask_%s.json" % daystr)
        args._health = gnss_chain_health.load_mask(hm) if hm else {}
        if args._health:
            print(
                "  health mask %s: %s"
                % (
                    hm,
                    ", ".join(
                        "%s %d bin(s)" % (c, len(b))
                        for c, (_, b) in sorted(args._health.items())
                        if b
                    )
                    or "nothing masked",
                )
            )
        elif hm:
            print(
                "  health mask %s: not found -- no chain-health veto for this day" % hm
            )
        out["health_mask"] = hm or None
        blobs = {}
        for chain, paths in present:
            if args.source == "l0":
                # The gain solve is its own pass over the same files: it needs every
                # satellite's whole day before the first sample can be corrected, so it
                # cannot ride along with the accumulation it corrects.
                pg = (
                    prn_gain_table(args, chain, paths, day_unix, geom)
                    if args.prn_norm
                    else None
                )
                r = build_day_h5(args, chain, paths, day_unix, geom, prn_gain=pg)
                if pg:
                    out["prn_norm"] = True
                    out.setdefault("prn_gains_db", {})[chain] = {
                        str(k): round(v, 3) for k, v in sorted(pg.items())
                    }
            else:
                if args.prn_norm:
                    sys.exit(
                        "--prn-norm needs --source l0 (it solves from the cube rows)"
                    )
                r = build_day(args, chain, paths, day_unix, geom)
            if r is None:
                continue
            out["units"] = r.get("units", "power")
            if r.get("pointing"):
                if out.get("pointing", r["pointing"]) != r["pointing"]:
                    sys.exit(
                        "two pointings in one build: %s / %s"
                        % (out["pointing"], r["pointing"])
                    )
                out["pointing"] = r["pointing"]
            i = len(out["chains"])
            out["chains"].append(
                {
                    "chain": chain,
                    "sys": CHAIN_SYS[chain],
                    "n_sub": int(r["n"].shape[0]),
                    "n_elem": int(r["n"].shape[1]),
                    "n_pix": int(len(r["pix"])),
                    "freq_ids": r["freq_ids"],
                }
            )
            blobs["pix_%d" % i] = r["pix"]
            blobs["n_%d" % i] = r["n"]
            blobs["s1_%d" % i] = r["s1"]
            blobs["s2_%d" % i] = r["s2"]
        # The archive path names a pointing and the epoch records one; if they disagree, one
        # of the two is wrong and the map would be drawn around the wrong boresight.
        if out.get("pointing") and out["pointing"] != epoch.pointing.name:
            sys.exit(
                "%s: archive says pointing %s, epoch %s records %s -- fix one before "
                "building" % (daystr, out["pointing"], epoch.name, epoch.pointing.name)
            )
        nm, nv = geom.veto_stats()
        print(
            "  railing veto: %d/%d minute(s) vetoed (%.1f%%) at %.1f deg"
            % (nv, nm, 100.0 * nv / max(1, nm), args.veto_deg)
        )
        if not out["chains"]:
            continue
        path = os.path.join(args.outdir, "cube_%s_nside%d.npz" % (daystr, args.nside))
        np.savez_compressed(path, meta=json.dumps(out), **blobs)
        print("  -> %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))


# ─────────────────────────────────────────────────────────────────────────────────────────
# export: master -> browser cube
# ─────────────────────────────────────────────────────────────────────────────────────────
def cmd_export(args):
    """Per day a .json manifest and one .bin per chain; index.json over the masters given.

    index.json is rewritten from the masters on the command line alone, so pass every day.
    The browser does the sums; the export only downsamples pixels (RING -> NEST -> shift ->
    RING) and, with --subbands, groups channels. Downsampling happens here because the
    browser cannot downsample a cube it has not downloaded: the master keeps every pixel and
    this writes the viewer's working set.
    """
    os.makedirs(args.outdir, exist_ok=True)
    index = []
    for src in args.masters:
        z = np.load(src, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        nside0 = int(meta["nside"])
        nside = args.nside or nside0
        if nside > nside0:
            sys.exit(
                "--nside %d is finer than the master's %d: a cube cannot be "
                "up-sampled, rebuild instead" % (nside, nside0)
            )
        bore = (meta.get("bore_az_deg", BORE_AZ), meta.get("bore_el_deg", BORE_EL))
        shift = 0
        while (nside0 >> shift) > nside:
            shift += 1
        chains, arrays = [], []
        for i, c in enumerate(meta["chains"]):
            pix, n, s1 = z["pix_%d" % i], z["n_%d" % i], z["s1_%d" % i]
            if shift:
                pix = ring_downgrade(pix, nside0, nside)
            # Bin subbands, then coalesce duplicate pixels created by the downgrade: both are
            # plain sums because the stored quantity is a linear accumulator.
            n, s1, sub_fid = bin_subbands(
                n, s1, meta["chains"][i]["freq_ids"], args.subbands
            )
            pixu, inv = np.unique(pix, return_inverse=True)
            if len(pixu) != len(pix):
                nn = np.zeros(n.shape[:2] + (len(pixu),), n.dtype)
                ss = np.zeros(s1.shape[:2] + (len(pixu),))
                np.add.at(nn, (slice(None), slice(None), inv), n)
                np.add.at(ss, (slice(None), slice(None), inv), s1)
                n, s1 = nn, ss
            # Pixel centres ship with the cube: on load the page's JS ang2pix port must
            # reproduce every centre's pixel index (verifyPixelisation in beamcube.js), or the
            # page refuses to draw.
            az, el = pix_centres(pixu, nside)
            chains.append(
                {
                    "chain": c["chain"],
                    "sys": c["sys"],
                    "n_sub": int(n.shape[0]),
                    "n_elem": int(n.shape[1]),
                    "n_pix": int(len(pixu)),
                    "freq_ids": sub_fid,
                    "lobe_db": lobe_level_db(n, s1, az, el, args.offset_annulus, bore),
                }
            )
            arrays.append(
                (
                    pixu.astype(np.int32),
                    np.stack([az, el], 1).astype(np.float32).ravel(),
                    n.astype(np.uint32),
                    s1.astype(np.float32),
                )
            )
        # Cross-chain offsets. Pedestal units remove each bin's own noise-floor scale, but the
        # sky signal still differs per chain (satellite EIRP per band, code bandwidth caught by
        # the comb), so two chains are two beams with different zeros. Each chain's offset from
        # --offset-ref (the first chain, if that one is absent) is the difference of median
        # pixel levels in --offset-annulus (default 2..12 deg off boresight), where every chain
        # samples the same structure. The viewer applies it as an editable default; it is
        # never baked into the cube.
        ref = next((c for c in chains if c["chain"] == args.offset_ref), chains[0])
        for c in chains:
            c["offset_db"] = (
                round(c["lobe_db"] - ref["lobe_db"], 2)
                if c["lobe_db"] is not None and ref["lobe_db"] is not None
                else 0.0
            )
        day = meta["day"]
        # One .bin per chain, fetched by the page only when that chain is selected: with the
        # full channel axis a chain is tens of MB at nside 32.
        total = 0
        for c, arr in zip(chains, arrays):
            blob = b"".join(a.tobytes() for a in arr)
            c["bin"] = "cube_%s_%s.bin" % (day, c["chain"])
            c["bytes"] = len(blob)
            with open(os.path.join(args.outdir, c["bin"]), "wb") as fh:
                fh.write(blob)
            total += len(blob)
        man = {
            "day": day,
            "nside": nside,
            "source": meta["source"],
            "pointing": meta.get("pointing"),
            "units": meta.get("units", "power"),
            "array_epoch": meta.get("array_epoch"),
            "array_epoch_key": meta.get("array_epoch_key"),
            "bore_az_deg": bore[0],
            "bore_el_deg": bore[1],
            "veto_deg": meta["veto_deg"],
            "range_norm": meta["range_norm"],
            "offset_ref": ref["chain"],
            "offset_annulus_deg": list(args.offset_annulus),
            "chains": chains,
            # The .bin layout; the viewer slices one ArrayBuffer in this order.
            "layout": "per chain file: pix int32[n_pix], centre float32[n_pix*2] "
            "(az,el degrees), n uint32[n_sub*n_elem*n_pix], "
            "s1 float32[n_sub*n_elem*n_pix]",
        }
        with open(os.path.join(args.outdir, "cube_%s.json" % day), "w") as fh:
            json.dump(man, fh)
        # array_epoch goes in the index as well as the manifest: the page sums the days picked
        # from the index, and a sum across an epoch boundary co-adds two different arrays
        # without any error. The page skips a day whose array_epoch_key differs from the first
        # selected day's (sameCurrency in beamcube.js); the export does not refuse, since one
        # index legitimately lists days of several epochs. A null means a master built without
        # the field, which is no promise of a single epoch.
        index.append(
            {
                "day": day,
                "bytes": total,
                "array_epoch": meta.get("array_epoch"),
                "array_epoch_key": meta.get("array_epoch_key"),
                "chains": [c["chain"] for c in chains],
            }
        )
        print(
            "%s -> %s/cube_%s_<chain>.bin (%.1f MB, nside %d, %d chain(s), %s)"
            % (
                src,
                args.outdir,
                day,
                total / 1e6,
                nside,
                len(chains),
                "  ".join("%s %d ch" % (c["chain"], c["n_sub"]) for c in chains),
            )
        )
        print(
            "   lobe %.0f..%.0f deg vs %s: %s"
            % (
                args.offset_annulus[0],
                args.offset_annulus[1],
                ref["chain"],
                "  ".join("%s %+.1f dB" % (c["chain"], c["offset_db"]) for c in chains),
            )
        )
    index.sort(key=lambda d: d["day"])
    with open(os.path.join(args.outdir, "index.json"), "w") as fh:
        json.dump({"days": index}, fh)
    print("index.json: %d day(s)" % len(index))


def lobe_level_db(n, s1, az, el, annulus, bore, min_n=8):
    """Median per-pixel mean (all subbands, all elements) inside the annulus about `bore`
    (az, el), in dB. None if fewer than 10 pixels qualify (a chain that never saw the lobe has
    no offset)."""
    nn, ss = n.sum((0, 1)), s1.sum((0, 1))
    th = angsep_deg(az, el, *bore)
    ok = (nn >= min_n) & (th >= annulus[0]) & (th <= annulus[1])
    if ok.sum() < 10:
        return None
    m = np.median(ss[ok] / nn[ok])
    return float(10.0 * np.log10(m)) if m > 0 else None


def pix_centres(pix, nside):
    """(az, el) in degrees for each RING pixel centre, from healpy (export only).

    The build needs only ang2pix and does not import healpy; the export uses the reference
    pix2ang rather than a second hand-written pixelisation.
    """
    try:
        import healpy as hp
    except ImportError:
        sys.exit(
            "export needs healpy for pixel centres: run it with "
            "/home/kvand/gnss/venv/bin/python (not venv-ft). The build subcommand does "
            "not need it."
        )
    theta, phi = hp.pix2ang(int(nside), np.asarray(pix, np.int64), nest=False)
    return np.degrees(phi) % 360.0, 90.0 - np.degrees(theta)


def ring_downgrade(pix, nside_in, nside_out):
    """RING -> RING at a coarser nside, via NEST where the downgrade is a bit shift."""
    return nest2ring(
        ring2nest(np.asarray(pix, np.int64), nside_in)
        >> (2 * (int(np.log2(nside_in)) - int(np.log2(nside_out)))),
        nside_out,
    )


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


# --- RING <-> NEST, needed only for the downgrade. ----------------------------------------
# nest2ring is the closed form, checked against healpy by tests/test_beam_cube.py. ring2nest
# is its exact inverse permutation (npix entries, built once per nside and cached) rather than
# a second closed form, so it cannot be wrong in a way nest2ring is not.
_R2N_CACHE = {}


def _compress(v):
    v = np.asarray(v, np.int64) & 0x5555555555555555
    for s, m in (
        (1, 0x3333333333333333),
        (2, 0x0F0F0F0F0F0F0F0F),
        (4, 0x00FF00FF00FF00FF),
        (8, 0x0000FFFF0000FFFF),
        (16, 0x00000000FFFFFFFF),
    ):
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
        jp = np.where(
            jp > 4 * nside, jp - 4 * nside, np.where(jp < 1, jp + 4 * nside, jp)
        )
        out[eq] = ncap + (jr[eq] - nside) * 4 * nside + jp - 1
    return out


def ring2nest(pix, nside):
    key = int(nside)
    if key not in _R2N_CACHE:
        npix = 12 * key * key
        nest = np.arange(npix, dtype=np.int64)
        inv = np.empty(npix, np.int64)
        inv[nest2ring(nest, key)] = nest  # ring index -> nest index, exactly
        _R2N_CACHE[key] = inv
    return _R2N_CACHE[key][np.asarray(pix, np.int64)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="archives -> master cube per day")
    b.add_argument(
        "--days", nargs="+", required=True, help="UTC days to build, YYYYMMDD"
    )
    b.add_argument(
        "--chains",
        nargs="*",
        help="chains to build (default: every chain with an archive for the day)",
    )
    b.add_argument("--source", choices=["elem", "l0"], default="elem")
    b.add_argument(
        "--archive",
        default=None,
        help="input directory. elem: the element archive (default "
        "/home/kvand/gnss/fixtures/obs); l0: one pointing's directory of the "
        "cube archive (default /mnt/cs00/data/kvand/gnss_cube/l0/p0_dec40p73), "
        "or the same pointing under rung12/ or rung60/",
    )
    b.add_argument("--outdir", default="/home/kvand/gnss/fixtures/beamcube")
    b.add_argument("--nside", type=int, default=64)
    b.add_argument(
        "--veto-deg",
        type=float,
        default=5.0,
        help="cross-chain railing veto: drop every minute in which a satellite "
        "of any constellation is within this many degrees of boresight, for "
        "every chain. 0 disables it and keeps the main lobe, contaminated by "
        "the railing -- quote its level as a lower bound",
    )
    b.add_argument(
        "--veto-keep-own",
        action="store_true",
        help="in a vetoed minute keep the rows of the satellite closest to boresight "
        "(the railed main lobe; its core is an upper bound on the beam width) and drop "
        "only the other satellites' rows: the 10-28 element-archive method. Default: "
        "drop every row in a vetoed minute",
    )
    b.add_argument("--mask-deg", type=float, default=0.0)
    b.add_argument(
        "--health-mask",
        default="auto",
        help="chain-health mask from gnss_chain_health.py, whose masked time bins are "
        "dropped per chain (--source l0 only); 'auto' = %s/mask_<day>.json, "
        "'' = none" % HEALTH_DIR,
    )
    b.add_argument("--no-range-norm", action="store_true")
    b.add_argument(
        "--prn-norm",
        action="store_true",
        help="divide out each satellite's own level (EIRP + its antenna "
        "pattern), solved jointly with a coarse beam by median polish "
        "(--source l0 only). Without it each satellite paints its track at "
        "its own level, and the map grows arcs along the bright ones.",
    )
    b.add_argument(
        "--unbiased",
        action="store_true",
        help="l0: keep below-pedestal samples (signed debias) instead of dropping them",
    )
    b.add_argument(
        "--pedestal-ref",
        choices=["envelope", "bin"],
        default="envelope",
        help="l0: divide each sample by the low envelope of its pedestal (10th percentile "
        "over +-2 h of 5-min bins), or by the 5-min pedestal itself, which rises with "
        "every satellite near the beam (the pre-10-30 masters)",
    )
    b.set_defaults(func=cmd_build)

    e = sub.add_parser("export", help="master cubes -> browser cubes + index.json")
    e.add_argument("masters", nargs="+")
    e.add_argument("--outdir", default="/home/kvand/gnss/fixtures/beamcube/web")
    e.add_argument(
        "--nside",
        type=int,
        default=16,
        help="nside of the browser cubes (default %(default)s), at most the "
        "master's; 0 = keep the master's",
    )
    e.add_argument(
        "--subbands",
        type=int,
        default=0,
        help="group the channel axis into N bins (0 = every channel, the default -- "
        "the axis is absolute freq_id and a sum across it is the reader's call)",
    )
    e.add_argument(
        "--offset-ref",
        default="gps_l5",
        help="chain whose level in --offset-annulus defines 0 dB for the per-chain "
        "offsets (the first chain if it is absent)",
    )
    e.add_argument(
        "--offset-annulus",
        type=float,
        nargs=2,
        default=(2.0, 12.0),
        metavar=("LO", "HI"),
        help="off-boresight degrees where the offsets are measured",
    )
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    if getattr(args, "archive", 0) is None:
        args.archive = (
            "/mnt/cs00/data/kvand/gnss_cube/l0/p0_dec40p73"
            if args.source == "l0"
            else "/home/kvand/gnss/fixtures/obs"
        )
    args.func(args)


if __name__ == "__main__":
    main()
