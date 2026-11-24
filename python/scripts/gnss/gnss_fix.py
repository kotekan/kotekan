#!/usr/bin/env python3
"""A position fix from the receiver's measurements, with a prior or without one.

    python3 gnss_fix.py OBS.jsonl ... --model-yaml Y [--at UNIX] [--prior LAT,LON,ALT]
                       [--truth-yaml Y] [--gw-table PATH] [--json OUT]

Without a prior the fix starts coarse. The trackers' Dopplers (dop_rec_hz at rec_hop) are range
rates with no ambiguity; from the visible satellites' mean direction they give the position to
tens or hundreds of metres, with one common range-rate bias for the reference oscillator. From
that, or from --prior, the code ranges solve position and one clock per chain to metres. --prior-site starts from the broker's own site.

The code range of a row is the broker's residual (code_resid_m: measured minus model, receiver
clock left in) plus the broker's model at the site it ran with (--model-yaml): range plus
Saastamoinen troposphere, the satellite clock cancelling between the two. That site may be far
off; the residual is only known mod the tracker code's period T (20 ms on L5 and B3I, 100 ms on
E5a, B2a and E6, 1.5 s on L2C), and each pass resolves the whole periods against the current
position, a chain's clock first taken as the circular mean of its satellites. GuoWang stays
out of the code solve (its table clocks are good to microseconds) and in the Doppler one.

Each stage drops rows over max(5 MADs, a floor) from its fit and solves again. Only offsets from
--truth-yaml's site are printed; --json gets the fix itself.
"""

import argparse
import json
import math
import re
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

import gnss_atmos
import gnss_brdc_supply
import gnss_ephemeris as ge
import gnss_pvt
from gnss_broker import signals

C = ge.C_LIGHT
FS_HZ = 122880000
HOP_SAMPLES = 64
R_EARTH = 6371e3
CODE_SPAN_S = (
    5.0  # the code solve takes each satellite's newest row this close to the fix
)
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


def geometry(e, rx, t):
    """(range m, range rate m/s, unit line of sight, satellite velocity) of record e seen from
    ECEF rx at GPS time t, with light time and the Earth's rotation over it."""
    tau = 0.075
    for _ in range(3):
        pos, vel, _ = ge.sat_pos_clk(e, t - tau)
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
    el = math.degrees(math.asin(float(u @ gnss_pvt._enu_axes(lat, lon)[2])))
    on_ground = -500.0 < alt < 9000.0 and el > 1.0
    return el, (gnss_atmos.tropo_delay_m(el, lat, alt) if on_ground else 0.0)


def keep_fit(res, floor):
    """Indices within max(5 MADs, floor) of the median residual."""
    med = np.median(res)
    lim = max(5.0 * 1.4826 * np.median(np.abs(res - med)), floor)
    return [i for i, r in enumerate(res) if abs(r - med) <= lim]


def doppler_fix(obs, rx0, iters=30):
    """Position and common range-rate bias from range rates. obs: [(e, t, rr_meas)]. Two
    rounds: fit, drop outliers, fit again."""
    rx = rx0.copy()
    n_all = len(obs)
    for _ in range(2):
        beta = 0.0
        for _ in range(iters):
            H, y = [], []
            for e, t, rr in obs:
                rng, rr_m, u, v = geometry(e, rx, t)
                # d(u.v)/d(rx) = -(v - (u.v) u) / range
                H.append(list(-(v - rr_m * u) / rng) + [1.0])
                y.append(rr - (rr_m + beta))
            H, y = np.array(H), np.array(y)
            dx, *_ = np.linalg.lstsq(H, y, rcond=None)
            rx, beta = rx + dx[:3], beta + dx[3]
            if np.linalg.norm(dx[:3]) < 0.1:
                break
        res = y - H @ dx
        kept = keep_fit(res, 0.5)
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
    drops outliers. None with fewer measurements than unknowns plus one."""
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
        if len(geo) < 3 + len(chains) + 1:
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
    return rx, float(np.sqrt(np.mean(res**2))), len(geo), len(chains)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("obs", nargs="+", help="gnss_observables.py rows (jsonl)")
    ap.add_argument(
        "--model-yaml",
        required=True,
        help="the broker yaml the residuals were taken at",
    )
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
    ap.add_argument(
        "--prior-site",
        action="store_true",
        help="start the code solve from --model-yaml's site",
    )
    ap.add_argument("--truth-yaml", help="print offsets from this broker yaml's site")
    ap.add_argument("--gw-table", help="GuoWang table, for its Dopplers")
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
    eph = ge.parse_rinex_nav(gnss_brdc_supply.cached_brdc())
    if a.gw_table:
        eph.update(ge.load_gw_table(a.gw_table))
    site = np.array(gnss_pvt._llh_to_ecef(*site_of(a.model_yaml)))
    sig, obs_d, newest = {}, [], {}
    for r in rows:
        if r["band"] not in sig:
            sig[r["band"]] = signals.get(r["band"])
        if r.get("rec_hop") and r.get("dop_rec_hz") is not None:
            t = gpst(
                FRAME0_UTC + (r["rec_hop"] * HOP_SAMPLES + HOP_SAMPLES - 1) / FS_HZ
            )
            e = ge.best_eph(eph.get((r["sys"], r["prn"]), []), t)
            if e is not None:
                obs_d.append((e, t, -r["dop_rec_hz"] * C / r["carrier_hz"]))
        key = (r["band"], r["prn"])
        if (
            r.get("code_resid_m") is not None
            and r["t"] >= t_at - CODE_SPAN_S
            and (key not in newest or r["t"] > newest[key]["t"])
        ):
            newest[key] = r
    meas = []
    for r in newest.values():
        s = sig[r["band"]]
        if s.constellation == "W":
            continue
        t = gpst(r["t"])
        e = ge.best_eph(eph.get((r["sys"], r["prn"]), []), t)
        if e is None:
            continue
        rng, _, u, _ = geometry(e, site, t)
        P = r["code_resid_m"] + rng + elev_tropo(site, u)[1]
        T = s.code_length * s.long_code_segments / s.chip_rate_hz
        meas.append((r["band"], e, t, (P / C) % T, T))
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
    if a.prior or a.prior_site:
        rx = (
            site.copy()
            if a.prior_site
            else np.array(
                gnss_pvt._llh_to_ecef(*(float(x) for x in a.prior.split(",")))
            )
        )
        report("prior", rx, "")
    else:
        # The satellites in view are above the horizon: their mean direction is near zenith.
        u = np.mean([ge.sat_pos_clk(e, t)[0] for e, t, _ in obs_d], axis=0)
        rx = R_EARTH * u / np.linalg.norm(u)
        report("start (mean direction)", rx, "")
        rx, beta, rms, n, drop = doppler_fix(obs_d, rx)
        report(
            "doppler",
            rx,
            "rr rms %.3f m/s, bias %+.3f m/s, %d rows (%d dropped)"
            % (rms, beta, n, drop),
        )
        out["doppler_rms_mps"] = rms
    fix = code_fix(meas, rx, a.prior_km * 1e3 if (a.prior or a.prior_site) else 1e4)
    if fix is None:
        print(
            "code: too few satellites (%d rows); the fix stands at the stage above"
            % len(meas)
        )
    else:
        rx, rms, n, nch = fix
        report("code", rx, "rms %.2f m, %d sats on %d chains" % (rms, n, nch))
        out.update(code_rms_m=rms, n_sats=n)
    out["ecef"] = list(rx)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
