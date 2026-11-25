#!/usr/bin/env python3
"""A position fix from the receiver's own measurements, with no model of where it is.

    python3 gnss_fix.py OBS.jsonl ... [--at UNIX] [--prior LAT,LON,ALT [--prior-km K]]
                       [--truth-yaml Y] [--json OUT]

Nothing here reads the broker's site, clock or model: the inputs are the fleet's records as the
observables rows carry them. The record Doppler (dop_rec_hz) is not one: it is the seed's, the
broker's model at whatever site it runs with (a site moved 99 km moved it 13-53 Hz, 11-24).

Without a prior the fix starts coarse. The fleet ADR's rate (fadr_dop_cycles over at least
DOP_BASELINE_S of fadr_hop), or a row's dop_meas_hz (gnss_bootstrap.py's: the record's Doppler
less the combiner's carrier residual, which agree to ~0.2 Hz), is each satellite's Doppler, a
range rate with no ambiguity; from
the visible satellites' mean direction it gives the position, with one common range-rate bias
for the reference oscillator. From that, or from --prior, the code ranges solve position and
one clock per chain.

A row's code phase is its record's replica phase (trkresid.phys_chips: the argument lifted
through the record's own Doppler, at the record's first sample), which the closed DLL holds on
the signal, so (t + clk_sat - phase/f_chip) mod T is the range over c up to the chain's clock,
T the tracker code's period (20 ms on L5 and B3I, 100 ms on E5a, B2a and E6, 1.5 s on L2C).
Each pass resolves the whole periods against the current position; a chain whose c*T/4 does
not cover the position's uncertainty waits. The record fields are float32 over a ~1e13-sample
lever, ~0.1-0.7 chips per record, so every row of the last CODE_SPAN_S goes in and the fit
averages them.

Row times are frame0 plus the hop count, the capture's labels; --time-offset-s takes the
anchor's error (gnss_skytime.py) off them for a host clock that was wrong.

Each stage drops rows over max(5 MADs, a floor) and solves again. Only offsets from
--truth-yaml's site are printed; --json gets the fix itself (gnss_set_site.py --fix-json).

Accuracy (11-24): from no prior the Doppler stage lands 130-290 m from the surveyed site and the
code stage 11-71 m. The rows' record phases sit +-1-2 chips per satellite off the broker's own
residual (which reads the telemetry records; same offset on B2a and B3I, cause not found), so
this is a bootstrap fix: inside the dead reckoning's 12-chip window, every chain relocks, and
the self-survey (gnss_pvt, the viewer's Position notice) takes it to metres. With the broker's
site moved 99 km (its model and every model-seeded chain lost, L5 tracking on its search) the
fix landed 11 m from the surveyed site, and applied, every chain relocked in ~2 minutes.
"""

import argparse
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np

import decoded_eph
import gnss_atmos
import gnss_brdc_supply
import gnss_ephemeris as ge
import gnss_pvt
from gnss_broker import signals
from gnss_broker.trkresid import phys_chips

C = ge.C_LIGHT
FS_HZ = 122880000
HOP_SAMPLES = 64
R_EARTH = 6371e3
HOPS_PER_S = Fraction(FS_HZ, HOP_SAMPLES)
CODE_SPAN_S = 30.0  # the code solve takes every row this close to the fix
# A converged code fix fits its ranges to metres or tens of metres; one that fits no better than
# this has resolved a wrong whole period somewhere, and the fix stays at the stage before
# (11-24: a 3.5 km rms code fix 3300 km off was applied over a 14.5 m Doppler fix).
CODE_RMS_MAX_M = 500.0
DOP_BASELINE_S = 5.0  # the shortest ADR difference taken as a Doppler
FRAME0_UTC = 1795132800.0  # config/bladerf_gnss.yaml gps_time frame0_nano
_KEY = r"\n    %s: ([-0-9.eE+]+)"


def site_of(yaml_path):
    s = open(yaml_path).read()
    return tuple(float(re.search(_KEY % k, s).group(1)) for k in ("lat", "lon", "alt"))


def gpst(t_unix):
    return ge.gpst_of_utc(datetime.fromtimestamp(t_unix, tz=timezone.utc))


def _seek(f, size, t0):
    """Move f to a line start at most a megabyte before the first row at or after t0 (rows
    are written in time order)."""
    lo, hi = 0, size
    while hi - lo > 1 << 20:
        mid = (lo + hi) // 2
        f.seek(mid)
        f.readline()
        line = f.readline()
        try:
            t = json.loads(line)["t"]
        except (ValueError, KeyError):
            t = None
        if t is not None and t < t0:
            lo = mid
        else:
            hi = mid
    f.seek(lo)
    if lo:
        f.readline()


def load(paths, t_at, span_s):
    """Present, non-probe rows in [t_at - span_s, t_at + 1]; with no t_at, the last line's."""
    rows = []
    for p in paths:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if t_at is None:
                f.seek(max(0, size - (1 << 16)))
                last = f.readlines()[-1]
                rows.append(json.loads(last))
                continue
            _seek(f, size, t_at - span_s)
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r["t"] > t_at + 1.0:
                    break
                if r["t"] < t_at - span_s:
                    continue
                if r.get("fleet_present") and not r.get("noise_probe"):
                    rows.append(r)
    return rows


class Orbits:
    """Satellite records from the BRDC cache, or with decoded_paths from the decoded ephemeris
    files (nav_decode.py and guowang_csk.py --eph-out): the satellites' own messages, which is
    all a receiver with no cache has. A record is the nearest-toe one within max_age_s, a
    decoded one the relay's only where no other source has the satellite."""

    def __init__(self, decoded_paths=None, max_age_s=14400.0):
        self.max_age_s = max_age_s
        self.dec = decoded_eph.load_files(decoded_paths) if decoded_paths else None
        self.brdc = (
            None
            if decoded_paths
            else ge.parse_rinex_nav(gnss_brdc_supply.cached_brdc())
        )

    def record(self, sys, prn, t):
        if self.brdc is not None:
            return ge.best_eph(self.brdc.get((sys, prn), []), t, self.max_age_s)
        best = None
        for signal, eph, toef in self.dec.get((sys, prn), []):
            toe = eph[toef]  # GLONASS's toe_gpst is GPS time already
            if toef != "toe_gpst":
                toe = decoded_eph.toe_gpst_near(sys, toe, t)
            age = abs(t - toe)
            key = (signal in decoded_eph.LAST_RESORT, age)
            max_age = min(self.max_age_s, decoded_eph.MAX_AGE_S.get(signal, age))
            if age <= max_age and (best is None or key < best[0]):
                best = (key, {"sys": sys, "prn": prn, "signal": signal, "eph": eph})
        return best[1] if best else None


def sat_state(e, t):
    """(ECEF position, velocity, clock s or None) of an Orbits record at GPS time t."""
    if "signal" in e:
        p, v, clk = decoded_eph.state(e["sys"], e["signal"], e["eph"], t)
        return np.array(p), np.array(v), clk
    return ge.sat_pos_clk(e, t)


def geometry(e, rx, t):
    """(range m, range rate m/s, unit line of sight, satellite velocity) of record e seen from
    ECEF rx at GPS time t, with light time and the Earth's rotation over it."""
    tau = 0.075
    for _ in range(3):
        pos, vel, _ = sat_state(e, t - tau)
        th = ge.OMEGA_E[e["sys"]] * tau
        c, s = math.cos(th), math.sin(th)
        p = np.array([pos[0] * c + pos[1] * s, -pos[0] * s + pos[1] * c, pos[2]])
        d = p - rx
        rng = float(np.linalg.norm(d))
        tau = rng / C
    v = np.array([vel[0] * c + vel[1] * s, -vel[0] * s + vel[1] * c, vel[2]])
    u = d / rng
    return rng, float(u @ v), u, v


def elev_tropo(rx, u):
    """(elevation deg, Saastamoinen slant delay m) of line of sight u from ECEF rx; no delay
    off the ground (a first step from a far prior can land anywhere)."""
    lat, lon, alt = gnss_pvt._ecef_to_llh(rx)
    el = math.degrees(
        math.asin(min(1.0, max(-1.0, float(u @ gnss_pvt._enu_axes(lat, lon)[2]))))
    )
    on_ground = -500.0 < alt < 9000.0 and el > 1.0
    return el, (gnss_atmos.tropo_delay_m(el, lat, alt) if on_ground else 0.0)


def keep_fit(res, floor):
    """Indices within max(5 MADs, floor) of the median residual."""
    med = np.median(res)
    lim = max(5.0 * 1.4826 * np.median(np.abs(res - med)), floor)
    return [i for i, r in enumerate(res) if abs(r - med) <= lim]


def grid_start(obs, centre, radius_m=4.0e6, step_m=1.0e5):
    """The point on the ground near `centre` whose model range rates fit `obs` best, a common
    bias taken out: a grid radius_m wide at step_m, then twice more, each ten times finer about
    the best. The start for doppler_fix: a LEO satellite's range rate swings by km/s within a
    pass, and a solve from the satellites' mean direction, ~800 km off, diverged (11-24). The
    fit is each satellite's median misfit, averaged over the satellites, so one with most of
    the rows does not choose alone (one pass fits a whole curve of points). Each row's satellite
    is placed once, from `centre`; the grid points differ only in the line of sight."""
    sub = obs[:: max(1, len(obs) // 300)]
    sat_p, sat_v, rr, sat_id = [], [], [], []
    for e, t, r in sub:
        rng, _, u, v = geometry(e, centre, t)
        sat_p.append(centre + rng * u)
        sat_v.append(v)
        rr.append(r)
        sat_id.append((e["sys"], e["prn"]))
    sat_p, sat_v, rr = np.array(sat_p), np.array(sat_v), np.array(rr)
    groups = [np.array([i == k for i in sat_id]) for k in set(sat_id)]
    best = centre
    for _ in range(3):
        lat, lon, _ = gnss_pvt._ecef_to_llh(best)
        enu = gnss_pvt._enu_axes(lat, lon)
        g = np.arange(-int(radius_m // step_m), int(radius_m // step_m) + 1) * step_m
        ge_, gn_ = np.meshgrid(g, g)
        pts = best + ge_.reshape(-1, 1) * enu[0] + gn_.reshape(-1, 1) * enu[1]
        pts = R_EARTH * pts / np.linalg.norm(pts, axis=1, keepdims=True)
        d = sat_p[None, :, :] - pts[:, None, :]
        u = d / np.linalg.norm(d, axis=2, keepdims=True)
        res = rr[None, :] - np.einsum("grk,rk->gr", u, sat_v)
        res -= np.median(res, axis=1, keepdims=True)
        cost = np.mean([np.median(np.abs(res[:, m]), axis=1) for m in groups], axis=0)
        best = pts[int(np.argmin(cost))]
        radius_m, step_m = 1.5 * step_m, step_m / 10.0
    return best


def sat_scatter(obs, half_s=5.0, floor=0.05):
    """{(sys, prn): robust scatter (m/s) of each satellite's range rates about a line through
    its neighbours within half_s}: each satellite's own measurement noise, with no position. The
    bootstrap's GuoWang rows scatter ~0.25 m/s against MEO's ~0.02, and a LEO row moves the
    position tens of times further per m/s, so they are weighted by it (11-24)."""
    by = defaultdict(list)
    for e, t, rr in obs:
        by[(e.get("sys"), e.get("prn", id(e)))].append((t, rr))
    out = {}
    for k, v in by.items():
        v.sort()
        tt = np.array([x[0] for x in v])
        rr = np.array([x[1] for x in v])
        res = []
        for i in range(len(v)):
            m = np.abs(tt - tt[i]) <= half_s
            if m.sum() >= 5:
                res.append(rr[i] - np.polyfit(tt[m] - tt[i], rr[m], 1)[1])
        out[k] = max(floor, 1.4826 * float(np.median(np.abs(res)))) if res else 1.0
    return out


def doppler_fix(obs, rx0, iters=30):
    """Position and common range-rate bias from range rates. obs: [(e, t, rr_meas)]. Each row
    is weighted by its satellite's own scatter (sat_scatter). Two rounds: fit, drop outliers
    (in those units), fit again."""
    sig = sat_scatter(obs)
    rx = rx0.copy()
    n_all = len(obs)
    for _ in range(2):
        w = np.array(
            [1.0 / sig[(e.get("sys"), e.get("prn", id(e)))] for e, _, _ in obs]
        )
        beta = 0.0
        for _ in range(iters):
            H, y = [], []
            for e, t, rr in obs:
                rng, rr_m, u, v = geometry(e, rx, t)
                # d(u.v)/d(rx) = -(v - (u.v) u) / range
                H.append(list(-(v - rr_m * u) / rng) + [1.0])
                y.append(rr - (rr_m + beta))
            H, y = np.array(H), np.array(y)
            dx, *_ = np.linalg.lstsq(H * w[:, None], y * w, rcond=None)
            rx, beta = rx + dx[:3], beta + dx[3]
            if np.linalg.norm(dx[:3]) < 0.1:
                break
        res = y - H @ dx
        kept = keep_fit(res * w, 3.0)
        obs = [obs[i] for i in kept]
    rms = float(np.sqrt(np.mean(res[kept] ** 2)))
    return rx, beta, rms, len(obs), n_all - len(obs)


def code_fix(meas, rx0, sigma0_m=1e4, mask_deg=10.0, iters=16):
    """Position and one clock per chain from code ranges. meas: [(chain, e, t, m_s, T)], m_s
    the measured range over c mod T, so range/c = m_s + clock + n*T. Each pass resolves n
    against the current position. A chain whose c*T/4 does not cover the position's
    uncertainty (sigma0_m, then three times the last step) waits until it does, unless too
    few would be left without it: a 20 ms chain resolves wrong whole periods from 1500 km.
    Once the position is within 10 km, satellites below mask_deg leave, and a final round
    drops outliers. None with fewer measurements than unknowns plus one, or fewer satellites
    than unknowns."""
    rx = rx0.copy()
    use = list(range(len(meas)))
    sigma = sigma0_m
    step = np.inf
    for it in range(iters):
        covered = [i for i in use if C * meas[i][4] / 4.0 > sigma]
        n_ch = len({meas[i][0] for i in covered})
        rows = covered if len(covered) >= 4 + n_ch else use
        geo = []
        for i in rows:
            _, e, t, _, _ = meas[i]
            rng, _, u, _ = geometry(e, rx, t)
            el, trop = elev_tropo(rx, u)
            if sigma >= 1e4 or el >= mask_deg:
                geo.append((i, rng + trop, u))
        chains = sorted({meas[g[0]][0] for g in geo})
        clk = {}
        for ch in chains:
            T = next(meas[g[0]][4] for g in geo if meas[g[0]][0] == ch)
            ang = [
                2 * math.pi * (g[1] / C - meas[g[0]][3]) / T
                for g in geo
                if meas[g[0]][0] == ch
            ]
            clk[ch] = (
                math.atan2(sum(map(math.sin, ang)), sum(map(math.cos, ang)))
                * T
                / (2 * math.pi)
            )
        H, y = [], []
        for i, rng, u in geo:
            ch, _, _, ms, T = meas[i]
            n = round((rng / C - ms - clk[ch]) / T)
            H.append(list(-u) + [1.0 if c == ch else 0.0 for c in chains])
            y.append(C * (ms + n * T + clk[ch]) - rng)
        # each satellite gives one range however many rows it has: count satellites
        recs = [meas[g[0]][1] for g in geo]
        n_sat = len({(e.get("sys"), e.get("prn", id(e))) for e in recs})
        if len(geo) < 3 + len(chains) + 1 or n_sat < 3 + len(chains):
            return None
        H, y = np.array(H), np.array(y)
        dx, *_ = np.linalg.lstsq(H, y, rcond=None)
        rx = rx + dx[:3]
        step = float(np.linalg.norm(dx[:3]))
        sigma = 3.0 * step
        if step < 0.01 and it > 1 and len(rows) == len(use):
            kept = keep_fit(y - H @ dx, 15.0)
            if len(kept) == len(geo):
                break
            use = [geo[k][0] for k in kept]
    res = y - H @ dx
    n_sats = len({(meas[g[0]][0], meas[g[0]][1].get("prn")) for g in geo})
    return rx, float(np.sqrt(np.mean(res**2))), len(geo), n_sats, len(chains)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("obs", nargs="+", help="gnss_observables.py rows (jsonl)")
    ap.add_argument(
        "--at", type=float, help="unix time of the fix (default: newest rows)"
    )
    ap.add_argument("--doppler-span-s", type=float, default=60.0)
    ap.add_argument("--prior", help="LAT,LON,ALT to start the code solve from")
    ap.add_argument(
        "--prior-km",
        type=float,
        default=100.0,
        help="how far the prior may be off: chains too short to cover it wait (default 100)",
    )
    ap.add_argument("--truth-yaml", help="print offsets from this broker yaml's site")
    ap.add_argument(
        "--decoded-eph",
        help="comma-separated decoded ephemeris files to take the orbits from, not the BRDC",
    )
    ap.add_argument(
        "--time-offset-s",
        type=float,
        default=0.0,
        help="the capture anchor's error (gnss_skytime.py), taken off every row's time",
    )
    ap.add_argument("--json", help="write the fix here")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    t_at = a.at
    if t_at is None:
        t_at = max(load([p], None, 0)[-1]["t"] for p in a.obs) - 2.0
    rows = load(a.obs, t_at, a.doppler_span_s)
    if not rows:
        print("no present rows in the %.0f s before %.0f" % (a.doppler_span_s, t_at))
        return 1
    orbits = Orbits(a.decoded_eph.split(",") if a.decoded_eph else None)
    sig, arcs, meas, obs_d = {}, defaultdict(list), [], []
    for r in rows:
        if r["band"] not in sig:
            sig[r["band"]] = signals.get(r["band"])
        s = sig[r["band"]]
        # GuoWang's rows count only with its own decoded orbits: Orbits has no TLE records
        if r.get("dop_meas_hz") is not None and r.get("rec_hop"):
            # a measured Doppler at the record (the bootstrap tracker's rows)
            t = gpst(FRAME0_UTC + r["rec_hop"] / float(HOPS_PER_S) - a.time_offset_s)
            e = orbits.record(r["sys"], r["prn"], t)
            if e is not None:
                obs_d.append((e, t, -r["dop_meas_hz"] * C / r["carrier_hz"]))
        elif r.get("fadr_hop") and r.get("fadr_dop_cycles") is not None:
            arcs[(r["band"], r["prn"], r.get("fadr_arc"))].append(r)
        if r["t"] < t_at - CODE_SPAN_S or not r.get("rec_hop"):
            continue
        if r.get("cp_rec_chips") is None or r.get("dop_rec_hz") is None:
            continue
        L = s.code_length * s.long_code_segments
        ph = phys_chips(
            r["cp_rec_chips"],
            r["dop_rec_hz"],
            r["rec_hop"],
            HOPS_PER_S,
            s.chip_rate_hz,
            s.carrier_hz,
            L,
        )
        t = gpst(FRAME0_UTC + r["rec_hop"] / float(HOPS_PER_S) - a.time_offset_s)
        e = orbits.record(r["sys"], r["prn"], t)
        if e is None:
            continue
        clk_sat = sat_state(e, t - 0.075)[2]
        if clk_sat is None:
            continue
        T = L / s.chip_rate_hz
        if s.long.endswith("_MC"):
            # A meander-tiled code is its first half and that half negated, and the data
            # flips the sign anyway: a blind track knows its phase only modulo the half.
            # GLONASS L2OF tracks sat on either 10 ms half (11-24 bootstrap: two of four).
            T /= 2.0
        meas.append((r["band"], e, t, (t + clk_sat - ph / s.chip_rate_hz) % T, T))
    # Doppler from the ADR: each row against the first of its arc at least DOP_BASELINE_S
    # earlier, at the pair's midpoint.
    for (band, prn, _), rs in arcs.items():
        rs.sort(key=lambda r: r["fadr_hop"])
        j = 0
        for r1 in rs:
            while (
                j < len(rs)
                and (r1["fadr_hop"] - rs[j]["fadr_hop"]) / float(HOPS_PER_S)
                > 2 * DOP_BASELINE_S
            ):
                j += 1
            r0 = rs[j]
            dt = (r1["fadr_hop"] - r0["fadr_hop"]) / float(HOPS_PER_S)
            if dt < DOP_BASELINE_S:
                continue
            dop = (r1["fadr_dop_cycles"] - r0["fadr_dop_cycles"]) / dt
            hop = 0.5 * (r0["fadr_hop"] + r1["fadr_hop"])
            t = gpst(FRAME0_UTC + hop / float(HOPS_PER_S) - a.time_offset_s)
            e = orbits.record(r1["sys"], prn, t)
            if e is not None:
                obs_d.append((e, t, -dop * C / r1["carrier_hz"]))
    truth = None
    if a.truth_yaml:
        lat, lon, alt = site_of(a.truth_yaml)
        truth = (
            np.array(gnss_pvt._llh_to_ecef(lat, lon, alt)),
            gnss_pvt._enu_axes(lat, lon),
        )

    def report(what, rx, extra):
        if truth is None:
            print("%s: %s" % (what, extra))
            return
        enu = truth[1] @ (rx - truth[0])
        print(
            "%s: E %+.1f N %+.1f U %+.1f m (|%.1f| m) %s"
            % (what, *enu, np.linalg.norm(enu), extra)
        )

    out = {"t": t_at}
    if a.prior:
        rx = np.array(gnss_pvt._llh_to_ecef(*(float(x) for x in a.prior.split(","))))
        report("prior", rx, "")
    else:
        # The satellites in view are above the horizon: their mean direction is near zenith,
        # and the range rates' best fit on a grid about it starts the solve.
        u = np.mean([sat_state(e, t)[0] for e, t, _ in obs_d], axis=0)
        rx = R_EARTH * u / np.linalg.norm(u)
        report("start (mean direction)", rx, "")
        rx = grid_start(obs_d, rx)
        report("start (grid)", rx, "")
        rx, beta, rms, n, drop = doppler_fix(obs_d, rx)
        report(
            "doppler",
            rx,
            "rr rms %.3f m/s, bias %+.3f m/s, %d rows (%d dropped)"
            % (rms, beta, n, drop),
        )
        out["doppler_rms_mps"] = rms
    fix = code_fix(meas, rx, a.prior_km * 1e3 if a.prior else 1e4)
    if fix is None:
        print(
            "code: too few satellites (%d rows); the fix stands at the stage above"
            % len(meas)
        )
    else:
        rx_c, rms, n, nsat, nch = fix
        bad = rms > CODE_RMS_MAX_M
        report(
            "code",
            rx_c,
            "rms %.2f m, %d rows of %d sats on %d chains%s"
            % (
                rms,
                n,
                nsat,
                nch,
                (
                    "; REJECTED (rms over %.0f m): the fix stands at the stage above"
                    % CODE_RMS_MAX_M
                    if bad
                    else ""
                ),
            ),
        )
        if not bad:
            rx = rx_c
            out.update(code_rms_m=rms, n_sats=n)
    out["ecef"] = list(rx)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
