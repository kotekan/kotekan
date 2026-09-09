"""Single-point position (PVT self-survey) from the code observable.

Now that the code observable is metre-good (the REC_CP export-currency fix), the obs logs carry a
per-satellite `code_resid_m` = measured code range MINUS the model range predicted at the CONFIG
position, plus `az`/`el`. That residual is exactly a LINEARIZED single-point-position observation
about the config position:

    code_resid_i  ~=  -e_i . dx  +  c*dt_group  +  (iono_i + multipath_i + noise_i)

where e_i is the line-of-sight unit vector (rx -> sat, from az/el) and dx is the receiver-position
correction. The common receiver/dongle clock is a free parameter PER GROUP -- one group per
(constellation, band), since each band's dongle and each constellation carry a different clock/
inter-system bias. Least-squares over (dx, {dt_group}) recovers the position; the post-fit residual
scales the covariance -> a real 1-sigma error ellipse. Because the config position is the true site,
this is a SELF-SURVEY / validation: it should return the config position to within a few metres
(single-frequency, so iono is uncorrected and sets the floor), and the offset + error are the
"is the whole chain self-consistent" check.

Pure geometry; no network, no broker state. Solve per group and a combined best-fit.
"""

import math

import numpy as np

C_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def _llh_to_ecef(lat, lon, alt):
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo),
                     (n + alt) * math.cos(la) * math.sin(lo),
                     (n * (1.0 - WGS84_E2) + alt) * math.sin(la)])


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
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


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


def _solve(rows, n_groups, group_of, reject_floor=60.0, reject_k=5.0):
    """Robust weighted LS with RAIM-style outlier rejection. The code residuals carry gross
    outliers (bad-tracking sats / wrong sub-code-period ambiguity, tens of km off a clean
    ~20 m cluster), so drop the single worst satellite while its post-fit residual exceeds
    max(reject_floor, reject_k * median|post|) and redundancy remains. Weights are 1/sigma^2
    from each measurement's own scatter (see solve); the rejection test stays in metres so
    a heavily down-weighted row can still be thrown out for being wrong rather than merely
    noisy. Returns (dx[3], clocks[n_groups], sigma0, sigma_diag, Ninv_unit, n_used, n_rej)
    or None; sigma0 is the unitless a-posteriori factor (1.0 = the scatter matched the
    weights) and sigma_diag the 1-sigma parameter errors in metres."""
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
    return (x[0:3], x[3:], sigma0, np.sqrt(np.clip(np.diag(cov), 0.0, None)), Ninv_unit,
            len(kept), n_rej, math.sqrt(float(post @ post) / dof))


def _dops(Ninv3):
    """PDOP/HDOP/VDOP from the position block of (H^T H)^-1 (unit-weighted geometry)."""
    d = np.clip(np.diag(Ninv3), 0.0, None)
    return dict(pdop=float(math.sqrt(d[0] + d[1] + d[2])),
                hdop=float(math.sqrt(d[0] + d[1])),
                vdop=float(math.sqrt(d[2])))


def hatch_smooth(rows, window_s, wrap_m=None):
    """Carrier-smoothed code residual for one satellite-band: (resid_m, n, sd_m, method).

    Hatch filtering, in its simplest exact form. Within one carrier arc the code residual and
    the carrier residual differ by a constant (the arc's ambiguity) plus what the carrier does
    not see: code noise, code multipath, and twice the ionosphere (the code-carrier
    divergence, slow). So the mean of (code - carrier) over a window is a low-noise estimate
    of that constant, and carrier_now + that mean is the code range with the carrier's noise
    and the code's absoluteness. Carrier phase must never be averaged ACROSS an arc break --
    the constant changes there -- so only rows sharing the newest row's `adr_arc` enter.

    `rows` are obs-log dicts (t, code_resid_m, carr_resid_m, adr_arc) for ONE (sys, prn, band);
    the newest row is the epoch reported. `wrap_m` is the code period in metres: the code
    residual is modular, so each row is first unwrapped to within half a period of the newest.
    Falls back to a plain boxcar of the code when no carrier is available, and to the single
    newest row when the window holds nothing else.
    """
    rows = sorted((r for r in rows if r.get("code_resid_m") is not None and r.get("t") is not None),
                  key=lambda r: r["t"])
    if not rows:
        return None
    new = rows[-1]

    def _arc(r):
        # the arc the CARRIER residual lives on (carr_arc), falling back to the ADR's
        a = r.get("carr_arc")
        return a if a is not None else r.get("adr_arc")

    t_new, arc = new["t"], _arc(new)
    sel = [r for r in rows if r["t"] >= t_new - window_s and _arc(r) == arc]
    c_new = float(new["code_resid_m"])

    def _code(r):
        c = float(r["code_resid_m"])
        return c - wrap_m * round((c - c_new) / wrap_m) if wrap_m else c

    if len(sel) < 2:
        return c_new, 1, None, "single"

    def _msd(xs):
        m = sum(xs) / len(xs)
        return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    c = [_code(r) for r in sel]
    m_box, sd_box = _msd(c)
    # THE CARRIER MUST EARN ITS PLACE. Hatch is only a gain when code - carrier is quieter
    # than the code alone; a carrier that jumps between rows (a phase reference that moved, an
    # arc the logger did not mark, an accumulated phase stamped at a different epoch than the
    # row) makes it catastrophically worse, and the obs logs have carried exactly that. So
    # both estimates are formed and the quieter one is reported. When the carrier is healthy
    # this picks Hatch by orders of magnitude; when it is not, nothing is lost.
    carr_ok = (new.get("carr_resid_m") is not None
               and all(r.get("carr_resid_m") is not None for r in sel))
    if carr_ok:
        d = [ci - float(r["carr_resid_m"]) for ci, r in zip(c, sel)]
        m_h, sd_h = _msd(d)
        if sd_h < sd_box:
            return float(new["carr_resid_m"]) + m_h, len(d), sd_h, "hatch"
    return m_box, len(c), sd_box, "boxcar"


def solve(measurements, lat0, lon0, alt0, min_el_deg=10.0, sigma_floor_m=0.5,
          sigma_default_m=5.0):
    """PVT self-survey. `measurements`: iterable of dicts {group, az, el, resid_m} where `group`
    is a (constellation, band) label, optionally with `sd_m` and `n` (the smoother's scatter and
    support): each row is weighted 1/sigma^2 with sigma = max(sigma_floor_m, sd_m/sqrt(n)), or
    sigma_default_m when it carries no scatter. This is what keeps a chain in its post-restart
    transient (residuals jumping by chips, sd of 100 m) from steering the fit while seven quiet
    chains sit at 3 m. A group whose label ends in "-IF" is an IONO-FREE
    (dual-frequency) group; those are solved together into `combined_if`, and the single-frequency
    groups into `combined`, so the two never mix (their clock biases differ and an IF sat is
    correlated with its own single-freq rows). Returns {groups: {group: result}, combined,
    combined_if}.

    Each result: n_sats, position (lat/lon/alt + ECEF), offset dENU (m) from the config a-priori,
    clock (m), resid_rms_m, sigma_enu (1-sigma m), dops. `combined`/`combined_if` solve their
    groups jointly with one clock per group -- the best-fit position + error. `combined_if` is the
    few-metre iono-free answer; `combined` the single-frequency (iono-limited) one for comparison."""
    apr = _llh_to_ecef(lat0, lon0, alt0)
    R = _enu_axes(lat0, lon0)   # rows E,N,U in ECEF
    # bucket usable measurements by group
    by_group = {}
    for m in measurements:
        az, el, res = m.get("az"), m.get("el"), m.get("resid_m")
        g = m.get("group")
        if az is None or el is None or res is None or g is None or el < min_el_deg:
            continue
        sd, n = m.get("sd_m"), m.get("n")
        sig = (max(sigma_floor_m, float(sd) / math.sqrt(max(1, int(n or 1))))
               if sd is not None else sigma_default_m)
        by_group.setdefault(g, []).append((_los_enu(az, el), float(res), sig))
    # GROSS pre-filter: the good satellites cluster within ~tens of metres of the group median
    # while bad ones (wrong sub-code-period ambiguity / mislock) sit km away. Cut those against
    # the robust median BEFORE the LS, so even a thin group (too few sats to reject in-fit) is
    # not wrecked by an outlier the iterative RAIM cannot afford to drop.
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
        pos = apr + R.T @ dx_enu           # ENU correction -> ECEF
        lat, lon, alt = _ecef_to_llh(pos)
        return dict(n_sats=n_used, n_rejected=n_rej,
                    lat=lat, lon=lon, alt=alt, ecef=pos.tolist(),
                    d_e=float(dx_enu[0]), d_n=float(dx_enu[1]), d_u=float(dx_enu[2]),
                    clock_m=float(clock_m), resid_rms_m=float(rms_m), sigma0=float(sigma0),
                    sigma_e=float(sig_diag[0]), sigma_n=float(sig_diag[1]),
                    sigma_u=float(sig_diag[2]),
                    **_dops(Ninv[0:3, 0:3]))

    out = {"groups": {}, "combined": None}
    for g, obs in sorted(by_group.items()):
        rows = [(los, res, 0, sig) for los, res, sig in obs]
        s = _solve(rows, 1, {0: g})
        if s:
            dx, clk, sig0, sd, Ninv, nu, nr, rms = s
            if _ok(Ninv, sd, nu, 4):
                out["groups"][g] = _pack(dx, clk[0], sig0, sd, Ninv, nu, nr, rms)

    # combined: one clock column per group. Solve single-frequency groups and iono-free (-IF)
    # groups SEPARATELY -- an IF group's clock folds two dongle clocks together and its sats are
    # the same physical satellites as the single-freq rows, so mixing them double-counts and
    # cross-contaminates the two clock frames.
    def _combined(sel):
        if not sel:
            return None
        gi = {g: i for i, g in enumerate(sel)}
        rows = [(los, res, gi[g], sig) for g in sel for los, res, sig in by_group[g]]
        s = _solve(rows, len(sel), gi)
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


if __name__ == "__main__":
    # Synthetic self-test: a known site, sats at assorted az/el, and a per-sat ELEVATION-mapped
    # ionosphere injected on L1 and L5 with the physical 1/f^2 ratio. Elevation-dependent iono is
    # NOT absorbable by a constant per-group clock, so the single-frequency solve is biased (mostly
    # vertical); the iono-free combination must remove it and recover the a-priori position.
    import random
    random.seed(1)
    lat0, lon0, alt0 = 43.9687, -79.2521, 260.0
    fq = {"L1": 1575.42e6, "L5": 1176.45e6}
    dx_true = np.array([4.0, -3.0, 6.0])           # ENU offset (m) we must recover
    clk = {"L1": 12.0, "L5": -5.0}                 # independent per-band dongle clocks (m)
    m_l1, m_if = [], []
    for _ in range(14):
        az, el = random.uniform(0, 360), random.uniform(15, 85)
        los = _los_enu(az, el)
        iono_l1 = 3.0 / math.sin(math.radians(el))  # L1 slant iono delay (m), bigger at low el
        base = -float(los @ dx_true)
        r1 = base + clk["L1"] + iono_l1 + random.gauss(0, 0.3)
        r5 = base + clk["L5"] + iono_l1 * (fq["L1"] / fq["L5"]) ** 2 + random.gauss(0, 0.3)
        m_l1.append({"group": "G-L1", "az": az, "el": el, "resid_m": r1})
        rif = (fq["L1"] ** 2 * r1 - fq["L5"] ** 2 * r5) / (fq["L1"] ** 2 - fq["L5"] ** 2)
        m_if.append({"group": "G-IF", "az": az, "el": el, "resid_m": rif})
    r_l1 = solve(m_l1, lat0, lon0, alt0)["groups"]["G-L1"]
    r_if = solve(m_if, lat0, lon0, alt0)["groups"]["G-IF"]
    e_l1 = math.sqrt(sum((r_l1[k] - dx_true[i]) ** 2 for i, k in enumerate(("d_e", "d_n", "d_u"))))
    e_if = math.sqrt(sum((r_if[k] - dx_true[i]) ** 2 for i, k in enumerate(("d_e", "d_n", "d_u"))))
    print("single-freq L1 position error: %.2f m  (sigma_u %.2f, iono-biased)"
          % (e_l1, r_l1["sigma_u"]))
    print("iono-free    IF position error: %.2f m  (sigma_u %.2f)" % (e_if, r_if["sigma_u"]))
    assert e_if < e_l1, "iono-free did not beat single-frequency (%.2f vs %.2f)" % (e_if, e_l1)
    assert e_if < 2.0, "iono-free residual too large: %.2f m" % e_if
    print("PASS: dual-frequency iono-free removes the elevation-dependent iono bias")
