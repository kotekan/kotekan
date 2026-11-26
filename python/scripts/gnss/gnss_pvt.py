"""Single-point position (PVT self-survey) from the code observable.

The obs logs carry a per-satellite `code_resid_m` (measured code range minus the model range
predicted at the configured position; rows written before 2026-11-21 carry the opposite sign,
see code_resid_meas_m) and `az`/`el`. That residual is a linearized
single-point-position observation about the configured position:

    code_resid_i  ~=  -e_i . dx  +  c*dt_group  +  (iono_i + multipath_i + noise_i)

where e_i is the line-of-sight unit vector (rx -> sat, from az/el) and dx is the receiver-position
correction. The receiver clock is a free parameter per group (livebeam_server uses one group per
chain), since each band and constellation carries its own clock and inter-system bias. Least
squares over (dx, {dt_group}) recovers the position; the post-fit residual scales the covariance
into a 1-sigma error ellipse. Because the configured position is the true site, this is a
self-survey: it should return that position to within a few metres (single frequency, so the
uncorrected ionosphere sets the floor), and the offset and its error check that the whole chain
is self-consistent.

Pure geometry; no network, no broker state. Solve per group and a combined best-fit.
"""

import glob
import json
import math
import os
import statistics

import numpy as np

C_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
# The obs rows' code_resid_m has been measured minus model range since 2026-11-21, marked by
# code_resid_sign; rows without the marker were written as model minus measured.
CODE_RESID_SIGN = "meas-model"


def code_resid_meas_m(r):
    """An obs-log row's code residual as measured minus model range (m), or None."""
    c = r.get("code_resid_m")
    if c is None:
        return None
    return float(c) if r.get("code_resid_sign") == CODE_RESID_SIGN else -float(c)


def _llh_to_ecef(lat, lon, alt):
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(la) ** 2)
    return np.array(
        [
            (n + alt) * math.cos(la) * math.cos(lo),
            (n + alt) * math.cos(la) * math.sin(lo),
            (n * (1.0 - WGS84_E2) + alt) * math.sin(la),
        ]
    )


def _ecef_to_llh(p):
    x, y, z = p
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1.0 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        alt = r / math.cos(lat) - n
        lat = math.atan2(z, r * (1.0 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    alt = r / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def _enu_axes(lat, lon):
    """Rows E, N, U (ECEF components) at the given geodetic lat/lon -- to turn an ENU
    position correction into ECEF and to project the LOS unit vector into ENU."""
    la, lo = math.radians(lat), math.radians(lon)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    return np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


def _los_enu(az_deg, el_deg):
    """LOS unit vector (rx -> sat) in local ENU from az (from North, clockwise) and el."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])


def _fit(rows, npar):
    """One weighted LS pass. rows: (los[3], resid, group_idx, sigma). Returns
    (x, post_m, Ninv_weighted, Ninv_unit) or None; post is in metres, unweighted."""
    n = len(rows)
    if n < npar:
        return None
    H = np.zeros((n, npar))
    r = np.zeros(n)
    w = np.zeros(n)
    for i, (los, resid, g, sig) in enumerate(rows):
        H[i, 0:3] = -los
        H[i, 3 + g] = 1.0
        r[i] = resid
        w[i] = 1.0 / sig
    Hw, rw = H * w[:, None], r * w
    try:
        Ninv = np.linalg.inv(Hw.T @ Hw)
        Ninv_unit = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return None
    x = Ninv @ (Hw.T @ rw)
    return x, r - H @ x, Ninv, Ninv_unit


def _solve(rows, n_groups, reject_floor=60.0, reject_k=5.0):
    """Robust weighted LS with RAIM-style outlier rejection. The code residuals carry gross
    outliers (bad-tracking sats / wrong sub-code-period ambiguity, tens of km off a clean
    ~20 m cluster), so drop the single worst satellite while its post-fit residual exceeds
    max(reject_floor, reject_k * median|post|) and redundancy remains. Weights are 1/sigma^2
    from each measurement's own scatter (see solve); the rejection test stays in metres so
    a heavily down-weighted row can still be thrown out for being wrong rather than merely
    noisy. Returns (dx[3], clocks[n_groups], sigma0, sigma_diag, Ninv_unit, n_used, n_rej,
    rms_m) or None; sigma0 is the unitless a-posteriori factor (1.0 = the scatter matched the
    weights), sigma_diag the 1-sigma parameter errors in metres and rms_m the unweighted
    post-fit rms in metres."""
    npar = 3 + n_groups
    kept = list(rows)
    n_rej = 0
    while True:
        f = _fit(kept, npar)
        if f is None:
            return None
        x, post, Ninv, Ninv_unit = f
        ap = np.abs(post)
        w = int(np.argmax(ap))
        thr = max(reject_floor, reject_k * float(np.median(ap)))
        if ap[w] > thr and len(kept) > npar + 2:
            kept.pop(w)
            n_rej += 1
            continue
        break
    dof = max(1, len(kept) - npar)
    sig = np.array([k[3] for k in kept])
    sigma0 = math.sqrt(float(((post / sig) ** 2).sum()) / dof)
    # never report an error smaller than the weights alone imply: a lucky low chi^2 on few
    # degrees of freedom is not a tighter measurement
    cov = max(1.0, sigma0) ** 2 * Ninv
    return (
        x[0:3],
        x[3:],
        sigma0,
        np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        Ninv_unit,
        len(kept),
        n_rej,
        math.sqrt(float(post @ post) / dof),
    )


def post_fit_m(measurements, out):
    """Each measurement's post-fit residual (m), against the combined solution (combined_if
    for an "-IF" group) with its group's clock, else against its group's own solution; None
    where neither holds the group. The model is -los . d_enu + clock (_fit). resid_m keeps the
    receiver clock, tens of km on the tracker's range form, so it is not readable alone.
    """
    post = []
    for m in measurements:
        g = m.get("group")
        c = out.get("combined_if" if str(g).endswith("-IF") else "combined") or {}
        own = (out.get("groups") or {}).get(g)
        if g in (c.get("clocks_m") or {}):
            sol, clk = c, c["clocks_m"][g]
        elif own:
            sol, clk = own, own["clock_m"]
        else:
            sol = None
        if sol is None or None in (m.get("az"), m.get("el"), m.get("resid_m")):
            post.append(None)
            continue
        d = np.array([sol["d_e"], sol["d_n"], sol["d_u"]])
        post.append(float(m["resid_m"] + _los_enu(m["az"], m["el"]) @ d - clk))
    return post


def _dops(Ninv3):
    """PDOP/HDOP/VDOP from the position block of (H^T H)^-1 (unit-weighted geometry)."""
    d = np.clip(np.diag(Ninv3), 0.0, None)
    return dict(
        pdop=float(math.sqrt(d[0] + d[1] + d[2])),
        hdop=float(math.sqrt(d[0] + d[1])),
        vdop=float(math.sqrt(d[2])),
    )


def hatch_smooth(rows, window_s, wrap_m=None):
    """Carrier-smoothed code residual for one satellite-band: (resid_m, n, sd_m, method).

    Hatch filtering. Within one carrier arc the code residual and the carrier residual differ
    by a constant (the arc's ambiguity) plus what the carrier does not see: code noise, code
    multipath, twice the ionosphere (the code-carrier divergence, slow), and on a receiver
    whose LO is offset from its sample clock that offset times the wavelength, a steady drift
    (the bladeRF: ~0.3 Hz, ~0.08 m/s). So a line fitted to (code - carrier) over the window and
    evaluated at the newest row is a low-noise estimate of that difference now, and
    carrier_now + it is the code range with the carrier's noise and the code's absoluteness. A
    mean would lag the drift by half the window. Carrier phase must never be smoothed across an
    arc break (the constant changes there), so only rows on the newest row's carrier arc and
    source (`carr_arc` and `carr_resid_src`, else `adr_arc`) enter.

    `rows` are obs-log dicts (t, code_resid_m, carr_resid_m, carr_arc or adr_arc) for one
    (sys, prn, band); the newest row is the epoch reported. `wrap_m` is the code period in
    metres: the code residual is modular, so each row is first unwrapped to within half a
    period of the newest. Falls back to a plain boxcar of the code when no carrier is available
    or the boxcar is quieter, and to the single newest row when the window holds nothing else.
    """
    rows = sorted(
        (
            r
            for r in rows
            if code_resid_meas_m(r) is not None and r.get("t") is not None
        ),
        key=lambda r: r["t"],
    )
    if not rows:
        return None
    new = rows[-1]

    def _arc(r):
        # the arc the carrier residual lives on (carr_arc), falling back to the ADR's, and its
        # source: an adr and an fadr arc can share a number
        a = r.get("carr_arc")
        return (r.get("carr_resid_src"), a if a is not None else r.get("adr_arc"))

    t_new, arc = new["t"], _arc(new)
    sel = [r for r in rows if r["t"] >= t_new - window_s and _arc(r) == arc]
    c_new = code_resid_meas_m(new)

    def _code(r):
        c = code_resid_meas_m(r)
        return c - wrap_m * round((c - c_new) / wrap_m) if wrap_m else c

    if len(sel) < 2:
        return c_new, 1, None, "single"

    def _msd(xs):
        m = sum(xs) / len(xs)
        return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    c = [_code(r) for r in sel]
    m_box, sd_box = _msd(c)
    # Hatch helps only when code - carrier is quieter than the code alone; a carrier that jumps
    # between rows (a moved phase reference, an unmarked arc break, a phase stamped at a
    # different epoch than the row) makes it far worse. So both estimates are formed and the
    # quieter one is reported.
    carr_ok = new.get("carr_resid_m") is not None and all(
        r.get("carr_resid_m") is not None for r in sel
    )
    if carr_ok:
        d = [ci - float(r["carr_resid_m"]) for ci, r in zip(c, sel)]
        n = len(d)
        if n < 3:
            m_h, sd_h = _msd(d)
            res = sd_h
        else:
            # Least-squares line through (t, d), evaluated at the newest row. Whether the
            # carrier helps is judged on the line's residual scatter against the code's; the
            # sd returned is per row, as the boxcar's is (solve divides by sqrt(n)): the
            # endpoint's standard error times sqrt(n), about twice that scatter.
            ts = [r["t"] - t_new for r in sel]
            mt, md = sum(ts) / n, sum(d) / n
            sxx = sum((x - mt) ** 2 for x in ts)
            slope = (
                sum((x - mt) * (y - md) for x, y in zip(ts, d)) / sxx if sxx else 0.0
            )
            m_h = md - slope * mt
            res = math.sqrt(
                sum((y - md - slope * (x - mt)) ** 2 for x, y in zip(ts, d)) / (n - 2)
            )
            sd_h = res * math.sqrt(n * (1.0 / n + (mt * mt / sxx if sxx else 0.0)))
        if res < sd_box:
            return float(new["carr_resid_m"]) + m_h, n, sd_h, "hatch"
    return m_box, len(c), sd_box, "boxcar"


def solve(
    measurements,
    lat0,
    lon0,
    alt0,
    min_el_deg=10.0,
    sigma_floor_m=0.5,
    sigma_default_m=5.0,
):
    """PVT self-survey. `measurements`: iterable of dicts {group, az, el, resid_m} where `group`
    labels one receiver clock (a constellation and band), optionally with `sd_m` and `n` (the
    smoother's scatter and support): each row is weighted 1/sigma^2 with
    sigma = max(sigma_floor_m, sd_m/sqrt(n)), or sigma_default_m when it carries no scatter, so
    a chain in its post-restart transient (residuals jumping by chips) is down-weighted rather
    than steering the fit. A group whose label ends in "-IF" is an iono-free (dual-frequency)
    group; those are solved together into `combined_if` and the single-frequency groups into
    `combined`, so the two never mix (their clock biases differ and an IF satellite is
    correlated with its own single-frequency rows). Returns
    {groups: {group: result}, combined, combined_if}.

    Each result: n_sats, n_rejected, lat/lon/alt and ecef, d_e/d_n/d_u (m) from the a-priori,
    clock_m, resid_rms_m, sigma0, sigma_e/sigma_n/sigma_u (1-sigma m) and pdop/hdop/vdop.
    `combined`/`combined_if` solve their groups jointly with one clock per group (clocks_m;
    clock_m is None): the best-fit position and error. `combined_if` is the iono-free answer;
    `combined` the single-frequency, iono-limited one."""
    apr = _llh_to_ecef(lat0, lon0, alt0)
    R = _enu_axes(lat0, lon0)  # rows E,N,U in ECEF
    # bucket usable measurements by group
    by_group = {}
    for m in measurements:
        az, el, res = m.get("az"), m.get("el"), m.get("resid_m")
        g = m.get("group")
        if az is None or el is None or res is None or g is None or el < min_el_deg:
            continue
        sd, n = m.get("sd_m"), m.get("n")
        sig = (
            max(sigma_floor_m, float(sd) / math.sqrt(max(1, int(n or 1))))
            if sd is not None
            else sigma_default_m
        )
        by_group.setdefault(g, []).append((_los_enu(az, el), float(res), sig))
    # Gross pre-filter: good satellites cluster within tens of metres of the group median while
    # bad ones (wrong sub-code-period ambiguity, mislock) sit km away. Cutting at 1 km from the
    # median before the LS keeps a thin group (too few satellites to reject in the fit) from
    # being wrecked by an outlier the iterative RAIM cannot afford to drop.
    for g in list(by_group):
        obs = by_group[g]
        if len(obs) >= 4:
            med = float(np.median([o[1] for o in obs]))
            by_group[g] = [o for o in obs if abs(o[1] - med) < 1000.0]

    def _ok(Ninv, sig_diag, n_used, npar):
        # Reject a degenerate / ill-conditioned geometry (e.g. all sats at one elevation ->
        # the vertical is unconstrained) rather than publish a garbage position, and a fit
        # with no redundancy (n == npar fits anything exactly and reports +-0).
        pd = math.sqrt(max(0.0, sum(np.clip(np.diag(Ninv[0:3, 0:3]), 0.0, None))))
        return n_used > npar and pd < 30.0 and float(np.max(sig_diag[0:3])) < 1000.0

    def _pack(dx_enu, clock_m, sigma0, sig_diag, Ninv, n_used, n_rej, rms_m):
        pos = apr + R.T @ dx_enu  # ENU correction -> ECEF
        lat, lon, alt = _ecef_to_llh(pos)
        return dict(
            n_sats=n_used,
            n_rejected=n_rej,
            lat=lat,
            lon=lon,
            alt=alt,
            ecef=pos.tolist(),
            d_e=float(dx_enu[0]),
            d_n=float(dx_enu[1]),
            d_u=float(dx_enu[2]),
            clock_m=float(clock_m),
            resid_rms_m=float(rms_m),
            sigma0=float(sigma0),
            sigma_e=float(sig_diag[0]),
            sigma_n=float(sig_diag[1]),
            sigma_u=float(sig_diag[2]),
            **_dops(Ninv[0:3, 0:3])
        )

    out = {"groups": {}, "combined": None}
    for g, obs in sorted(by_group.items()):
        rows = [(los, res, 0, sig) for los, res, sig in obs]
        s = _solve(rows, 1)
        if s:
            dx, clk, sig0, sd, Ninv, nu, nr, rms = s
            if _ok(Ninv, sd, nu, 4):
                out["groups"][g] = _pack(dx, clk[0], sig0, sd, Ninv, nu, nr, rms)

    # combined: one clock column per group. Single-frequency and iono-free (-IF) groups are
    # solved separately: an IF group's clock folds two bands' clocks together and its
    # satellites are the same ones as the single-frequency rows, so mixing them double-counts
    # and cross-contaminates the two clock frames.
    def _combined(sel):
        if not sel:
            return None
        gi = {g: i for i, g in enumerate(sel)}
        rows = [(los, res, gi[g], sig) for g in sel for los, res, sig in by_group[g]]
        s = _solve(rows, len(sel))
        if not (s and _ok(s[4], s[3], s[5], 3 + len(sel))):
            return None
        dx, clks, sig0, sd, Ninv, nu, nr, rms = s
        c = _pack(dx, 0.0, sig0, sd, Ninv, nu, nr, rms)
        c["clock_m"] = None
        c["clocks_m"] = {g: float(clks[gi[g]]) for g in sel}
        c["n_groups"] = len(sel)
        return c

    all_groups = sorted(by_group)
    out["combined"] = _combined([g for g in all_groups if not g.endswith("-IF")])
    out["combined_if"] = _combined([g for g in all_groups if g.endswith("-IF")])
    return out


def measurements_from_obs(globs, max_age_s, t_now, smooth_s=90.0):
    """Per-(sys, prn, chain) code observation across the obs logs, for the PVT self-survey:
    the freshest row's epoch, carrier-smoothed over the last `smooth_s` seconds of its own
    arc (hatch_smooth). One row per satellite-chain. Residuals are wrapped,
    clock-dominated ranges, de-wrapped per group to the group median so the common clock is
    a single free parameter and no satellite straddles a code-period edge. Returns a list of
    {group, sys, prn, az, el, resid_m, t, f_hz} (the newest row's epoch and carrier), plus n,
    sd_m and method where smoothing succeeded.

    The group is the obs row's band, i.e. the chain: one chain, one clock.
    """
    t_floor = t_now - max(max_age_s, smooth_s) - 5.0
    hist = {}  # (sys, prn, band) -> rows inside the smoothing window
    latest = {}  # (sys, prn, band) -> (t, group, az, el, resid_m, L_m, carrier_hz)
    for pat in globs:
        for path in glob.glob(pat):
            try:
                # A file last written before the window holds no row inside it: the glob
                # matches every past day's log, and each tail-read costs up to 4 MB.
                if os.path.getmtime(path) < t_floor:
                    continue
                size = os.path.getsize(path)
                with open(path) as f:
                    # tail: the smoothing window of every satellite on this chain. Rows are
                    # ~1.2 kB and a chain logs ~13 rows/s, so 4 MB holds several minutes.
                    f.seek(max(0, size - 4 * 1024 * 1024))
                    f.readline()
                    for line in f:
                        # cheap epoch pre-filter before the JSON decode: rows start {"t":...
                        if line.startswith('{"t":'):
                            try:
                                if float(line[5 : line.index(",", 5)]) < t_floor:
                                    continue
                            except Exception:
                                pass
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        t = d.get("t")
                        sysid, prn, fr = d.get("sys"), d.get("prn"), d.get("band")
                        res, az, el = d.get("code_resid_m"), d.get("az"), d.get("el")
                        if (
                            t is None
                            or sysid is None
                            or prn is None
                            or fr is None
                            or res is None
                            or az is None
                            or el is None
                        ):
                            continue
                        if t < t_floor:
                            continue
                        # a replica that is not on its signal (a noise probe, or not
                        # fleet_present) has a code phase but no range
                        if d.get("noise_probe") or d.get("fleet_present") is False:
                            continue
                        key = (sysid, prn, fr)
                        hist.setdefault(key, []).append(d)
                        if t_now - t > max_age_s:
                            continue
                        if key in latest and latest[key][0] >= t:
                            continue
                        cl, ch = d.get("code_len"), d.get("chip_rate_hz")
                        L = (cl / ch * C_LIGHT) if (cl and ch) else None
                        latest[key] = (
                            t,
                            fr,
                            az,
                            el,
                            float(res),
                            L,
                            d.get("carrier_hz"),
                        )
            except Exception:
                continue
    # Carrier-smooth each satellite's code residual over its own arc (hatch_smooth):
    # the code observable is white at ~0.1-0.25 chips per record, while the carrier holds arcs
    # for hours.
    extra = {}
    for key, v in list(latest.items()):
        hs = hatch_smooth(hist.get(key, []), smooth_s, wrap_m=v[5])
        if hs is None:
            continue
        r_s, n, sd, method = hs
        latest[key] = (v[0], v[1], v[2], v[3], float(r_s), v[5], v[6])
        extra[key] = {"n": n, "sd_m": sd, "method": method}
        # The code discriminator over the window: a replica not yet on its signal (minutes
        # after a broker start) reads its range short.
        discs = [abs(r["dll_disc"]) for r in hist[key] if r.get("dll_disc") is not None]
        if discs:
            extra[key]["disc_chips"] = statistics.median(discs)

    # de-wrap each group's residuals to the group median (handles the clock straddling a
    # code-period boundary; L is the code period in metres).
    by_group = {}
    for (sysid, prn, fr), v in latest.items():
        by_group.setdefault(v[1], []).append((sysid, prn, fr, v))
    out = []
    for group, rows in by_group.items():
        med = statistics.median([r[3][4] for r in rows])
        for sysid, prn, fr, v in rows:
            t_m, g, az, el, res, L, f_hz = v
            if L:
                res = res - L * round(
                    (res - med) / L
                )  # unwrap to within +-L/2 of the median
            m = {
                "group": g,
                "sys": sysid,
                "prn": prn,
                "az": az,
                "el": el,
                "resid_m": res,
                "t": t_m,
                "f_hz": f_hz,
            }
            m.update(extra.get((sysid, prn, fr), {}))
            out.append(m)
    return out


def rows_from_obs(globs, t0, t1, sysid, every_s=10.0):
    """Every code observation of one constellation in [t0, t1] from the obs logs, one per
    satellite per every_s (the rows are smoothed and correlated over seconds): a list of
    {group, sys, prn, az, el, resid_m, t}, oldest first. The residuals are wrapped,
    clock-dominated ranges; each is de-wrapped to within half a code period of the median of
    the 20 before it, so the receiver clock stays continuous across the window."""
    rows, last = [], {}
    for pat in globs:
        for path in glob.glob(pat):
            try:
                if os.path.getmtime(path) < t0:
                    continue
                with open(path) as f:
                    for line in f:
                        if line.startswith('{"t":'):
                            try:
                                t = float(line[5 : line.index(",", 5)])
                                if t < t0 or t > t1:
                                    continue
                            except ValueError:
                                pass
                        try:
                            d = json.loads(line)
                        except ValueError:
                            continue
                        t, res = d.get("t"), d.get("code_resid_m")
                        if d.get("sys") != sysid or t is None or res is None:
                            continue
                        if (
                            not (t0 <= t <= t1)
                            or d.get("az") is None
                            or d.get("el") is None
                        ):
                            continue
                        if d.get("noise_probe") or d.get("fleet_present") is False:
                            continue
                        key = (d.get("prn"), d.get("band"))
                        if t - last.get(key, -1e18) < every_s:
                            continue
                        last[key] = t
                        cl, ch = d.get("code_len"), d.get("chip_rate_hz")
                        rows.append(
                            {
                                "group": d.get("band"),
                                "sys": sysid,
                                "prn": d.get("prn"),
                                "az": d["az"],
                                "el": d["el"],
                                "resid_m": float(res),
                                "t": t,
                                "L": (cl / ch * C_LIGHT) if (cl and ch) else None,
                            }
                        )
            except OSError:
                continue
    rows.sort(key=lambda r: r["t"])
    ref = []
    for r in rows:
        L = r.pop("L")
        if ref and L:
            med = statistics.median(ref[-20:])
            r["resid_m"] -= L * round((r["resid_m"] - med) / L)
        ref.append(r["resid_m"])
    return rows


def _batch_fit(H, y, keep, reject_k):
    """batch_solve's fit over the rows in `keep`, dropping rows whose post-fit residual exceeds
    max(60 m, reject_k x the median) until none does: (x, post over every row, Ninv, keep) or
    None when under 8 rows remain or the geometry is singular."""
    keep = keep.copy()
    for _ in range(10):
        if keep.sum() < 8:
            return None
        Hk, yk = H[keep], y[keep]
        try:
            Ninv = np.linalg.inv(Hk.T @ Hk)
        except np.linalg.LinAlgError:
            return None
        x = Ninv @ (Hk.T @ yk)
        post = y - H @ x
        thr = max(60.0, reject_k * float(np.median(np.abs(post[keep]))))
        bad = keep & (np.abs(post) > thr)
        if not bad.any():
            break
        keep &= ~bad
    return x, post, Ninv, keep


def batch_solve(rows, lat0, lon0, alt0, min_el_deg=10.0, sigma_m=5.0, reject_k=5.0):
    """A stationary receiver's position from code observations spread over a window, for a
    constellation too sparse to solve at one epoch (GuoWang). rows: {t, prn, az, el, resid_m}.
    The model is resid = -los . dx + c0 + c1 (t - t_mid): one receiver clock offset and rate over
    the window, which a disciplined oscillator holds, so the directions a low satellite sweeps
    through in minutes stand in for simultaneous satellites. A satellite whose broadcast clock is
    off sits at one offset through its pass, and a few satellites' fit absorbs much of it, which
    also inflates the median the row cut scales with. So with 3 or more satellites, each is first
    tested against the fit of the others: the one whose median residual there exceeds
    max(60 m, reject_k x that fit's median, reject_k x the error of its prediction) by the largest
    factor is excluded, and the test repeats. Then rows whose post-fit residual exceeds max(60 m, reject_k x the median) are
    dropped and the fit repeated. Valid only for a receiver that did not move over the window;
    the caller must check that. Returns solve's result fields plus n_rows, n_sats, span_s,
    clock_rate_m_s and excluded ({prn: median residual m against the others}), or None when the
    window cannot determine the position (under 2 satellites or 120 s, no redundancy, or a
    degenerate geometry)."""
    use = [r for r in rows if r.get("el") is not None and r["el"] >= min_el_deg]
    if len(use) < 8 or len({r.get("prn") for r in use}) < 2:
        return None
    ts = np.array([r["t"] for r in use], dtype=float)
    span = float(ts.max() - ts.min())
    if span < 120.0:
        return None
    t_mid = 0.5 * (ts.max() + ts.min())
    H = np.zeros((len(use), 5))
    y = np.zeros(len(use))
    for i, r in enumerate(use):
        H[i, 0:3] = -_los_enu(r["az"], r["el"])
        H[i, 3] = 1.0
        H[i, 4] = r["t"] - t_mid
        y[i] = r["resid_m"]
    prn = np.array([r.get("prn") for r in use])
    keep = np.ones(len(use), dtype=bool)
    excluded = {}
    while len(set(prn[keep].tolist())) >= 3:
        worst = None
        for s in sorted(set(prn[keep].tolist())):
            f = _batch_fit(H, y, keep & (prn != s), reject_k)
            if f is None or math.sqrt(max(0.0, float(np.trace(f[2][0:3, 0:3])))) > 30.0:
                continue
            x, post, Ninv, k = f
            med = float(np.median(np.abs(post[k])))
            # the others' fit predicts this satellite only as well as their geometry allows:
            # its prediction error, with rows counted as in the reported errors below
            Hs = H[prn == s]
            corr = math.sqrt(max(1.0, k.sum() / (3.0 * len(set(prn[k])))))
            pred = (
                math.sqrt(float(np.median(np.einsum("ij,jk,ik->i", Hs, Ninv, Hs))))
                * med
                * corr
            )
            off = float(np.median(post[prn == s]))
            ratio = abs(off) / max(60.0, reject_k * med, reject_k * pred)
            if ratio > 1.0 and (worst is None or ratio > worst[0]):
                worst = (ratio, s, off)
        if worst is None:
            break
        excluded[worst[1]] = round(worst[2], 1)
        keep &= prn != worst[1]
    f = _batch_fit(H, y, keep, reject_k)
    if f is None:
        return None
    x, post, Ninv, kept = f
    n_rejected = int((keep & ~kept).sum())
    keep = kept
    dof = max(1, int(keep.sum()) - 5)
    pk = post[keep]
    rms = math.sqrt(float(pk @ pk) / dof)
    sigma0 = rms / sigma_m
    # The rows are not independent (smoothed, and one satellite's error persists through its
    # pass), so the formal error is scaled by the rows per satellite: a pass counts as a few
    # measurements, not as many as it has rows.
    n_sats = len({use[i].get("prn") for i in np.nonzero(keep)[0]})
    corr = math.sqrt(max(1.0, keep.sum() / max(1, 3 * n_sats)))
    cov = (max(1.0, sigma0) * sigma_m * corr) ** 2 * Ninv
    pd = math.sqrt(max(0.0, float(np.trace(Ninv[0:3, 0:3]))))
    if pd > 30.0:
        return None
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    apr = _llh_to_ecef(lat0, lon0, alt0)
    pos = apr + _enu_axes(lat0, lon0).T @ x[0:3]
    lat, lon, alt = _ecef_to_llh(pos)
    return dict(
        n_sats=n_sats,
        n_rejected=n_rejected,
        excluded=excluded,
        n_rows=int(keep.sum()),
        span_s=span,
        lat=lat,
        lon=lon,
        alt=alt,
        ecef=pos.tolist(),
        d_e=float(x[0]),
        d_n=float(x[1]),
        d_u=float(x[2]),
        clock_m=float(x[3]),
        clock_rate_m_s=float(x[4]),
        resid_rms_m=rms,
        sigma0=sigma0,
        sigma_e=float(sd[0]),
        sigma_n=float(sd[1]),
        sigma_u=float(sd[2]),
        **_dops(Ninv[0:3, 0:3])
    )
