"""Decoding the broadcast navigation message, and cross-checking it against BRDC.

Eight decoders (LNAV, CNAV, CNAV-2, F/NAV, I/NAV, B-CNAV1/2/3). Every decoded ephemeris is
cross-checked against the downloaded BRDC ephemeris whenever the BRDC store is loaded
(--almanac-source brdc); CNAV-2 decodes none. The decoders are created lazily and persist
across cycles on `ctx.nav`: one that is rebuilt has forgotten the subframes it had collected
and can never complete an ephemeris.

Idle on CHORD: --nav-bits defaults on but self-gates on nav_obs, which no CHORD combiner
exports (bit_export), and no aux combiner is configured. No test reaches this stage, so a
change here is unverified until one of those is set.

The cross-check is the point, not the decode: a decoded ephemeris nobody compares with
broadcast is a second authority that can drift from the first without either side erroring.
ctx.dh_obs records each decoder's health and BRDC position residual on the 60 s health
cadence (the decode-health export, --decode-health-file), so a drift shows as a trend.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _now, _get, _log, _log_rl
from gnss_broker.sky import (
    _lnav_brdc_xcheck,
    _cnav_brdc_xcheck,
    _inav_brdc_xcheck,
    _fnav_brdc_xcheck,
    _bcnav1_brdc_xcheck,
    _bcnav2_brdc_xcheck,
    _bcnav3_brdc_xcheck,
)


def stage_nav_bits(ctx):
    """Decode the broadcast navigation message and cross-check it against BRDC.

    The decoders live on ctx.nav (loopstate.NavDecoders): created on the first row that needs
    them and kept across cycles. nav_obs from this broker's own combiner go to the decoder
    --nav-decoder names; each aux combiner's go to its own decoder.

    Idle on CHORD (see the module docstring); no test reaches it."""
    if ctx.args.nav_bits:
        for _p, _r in ctx.status.items():
            if "nav_obs" not in _r:
                continue
            # Route nav_obs to the decoder this band speaks. L1 C/A carries LNAV (periodic
            # subframes); L2C-CM and L5-I carry CNAV (FEC + CRC, no fixed schedule). CNAV
            # symbols sent to the LNAV frame sync never find a preamble.
            if ctx.args.nav_decoder == "cnav":
                if ctx.nav.cnav is None:
                    from cnav_predictor import CnavPredictor

                    ctx.nav.cnav = CnavPredictor(log=_log)
                    _log("CNAV decoder armed (combiner exports nav_obs)")
                ctx.nav.cnav.ingest(_p, _r["nav_obs"])
            elif ctx.args.nav_decoder == "bcnav3":
                # BeiDou B2b B-CNAV3, the B2b chain's own message: an NB-LDPC block code, not a
                # bit-prediction source, so this is decode and BRDC cross-check only.
                if ctx.nav.bcnav3 is None:
                    from bcnav3_predictor import Bcnav3Predictor

                    ctx.nav.bcnav3 = Bcnav3Predictor()
                    _log("B-CNAV3 decoder armed (combiner exports nav_obs)")
                ctx.nav.bcnav3.ingest(_p, _r["nav_obs"])
            else:
                if ctx.nav.navbits is None:
                    from navbit_predictor import LnavPredictor

                    ctx.nav.navbits = LnavPredictor(log=_log)
                    _log("nav-bit predictor armed (combiner exports nav_obs)")
                ctx.nav.navbits.ingest(_p, _r["nav_obs"])
        # Aux CNAV chain (--cnav-combiner): a second combiner polled only for its CNAV
        # symbols. At L5 this broker's own combiner is the Q pilot, whose nav_obs are
        # deterministic overlay signs handled above, while CNAV arrives on the derived L5-I
        # data sibling, which no broker otherwise reads. A failed read is logged
        # (rate-limited) and skipped: this is an observability path and must never disturb
        # the tracking loop that shares this cycle. The aux blocks below follow the same shape.
        if ctx.nav.cnav_combiner:
            try:
                _aux = {
                    int(r["prn"]): r
                    for r in _get("%s/get_status" % ctx.nav.cnav_combiner)
                    if r.get("prn")
                }
            except Exception as e:
                _aux = {}
                _log_rl(
                    "cnavaux",
                    "cnav aux combiner %s unreadable: %s" % (ctx.nav.cnav_combiner, e),
                )
            for _p, _r in _aux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.cnav is None:
                    from cnav_predictor import CnavPredictor

                    ctx.nav.cnav = CnavPredictor(log=_log)
                    _log("CNAV decoder armed on aux chain %s" % ctx.nav.cnav_combiner)
                ctx.nav.cnav.ingest(_p, _r["nav_obs"])
        # Galileo I/NAV from its aux data chain (--inav-combiner).
        if ctx.nav.inav_combiner:
            try:
                _iaux = {
                    int(r["prn"]): r
                    for r in _get("%s/get_status" % ctx.nav.inav_combiner)
                    if r.get("prn")
                }
            except Exception as e:
                _iaux = {}
                _log_rl(
                    "inavaux",
                    "inav aux combiner %s unreadable: %s" % (ctx.nav.inav_combiner, e),
                )
            for _p, _r in _iaux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.inav is None:
                    from inav_predictor import InavPredictor

                    ctx.nav.inav = InavPredictor()
                    _log("I/NAV decoder armed on aux chain %s" % ctx.nav.inav_combiner)
                ctx.nav.inav.ingest(_p, _r["nav_obs"])
            # 60 s health + BRDC cross-check (Kepler only; alm_sys is 'E' for a Galileo chain).
            # I/NAV rides E1B on L1 and E5b-I on the mid band with the same message and
            # decoder, so the decode-health label follows the aux combiner's name.
            if ctx.nav.inav is not None and _now() - ctx.nav.inav_log_t[0] > 60.0:
                ctx.nav.inav_log_t[0] = _now()
                _inav_sig = (
                    "GAL_E5BI_INAV"
                    if "e5b" in ctx.nav.inav_combiner
                    else "GAL_E1B_INAV"
                )
                for _p in sorted(ctx.nav.inav._p):
                    h = ctx.nav.inav.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.inav.ephemeris(_p)
                    xc = (
                        _inav_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                        if (eph is not None and ctx.brdc_alm is not None)
                        else ""
                    )
                    _log(
                        "inav PRN %d: %d pages, %d words, have %s, eph %s%s"
                        % (
                            _p,
                            h["pages"],
                            h["words"],
                            h["have"],
                            "YES" if eph is not None else "no",
                            xc,
                        )
                    )
                    ctx.dh_obs(_inav_sig, _p, h, eph, xc)
        # Galileo E5a-I F/NAV from its aux data chain (--fnav-combiner).
        if ctx.nav.fnav_combiner:
            try:
                _faux = {
                    int(r["prn"]): r
                    for r in _get("%s/get_status" % ctx.nav.fnav_combiner)
                    if r.get("prn")
                }
            except Exception as e:
                _faux = {}
                _log_rl(
                    "fnavaux",
                    "fnav aux combiner %s unreadable: %s" % (ctx.nav.fnav_combiner, e),
                )
            for _p, _r in _faux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.fnav is None:
                    from fnav_predictor import FnavPredictor

                    ctx.nav.fnav = FnavPredictor()
                    _log("F/NAV decoder armed on aux chain %s" % ctx.nav.fnav_combiner)
                ctx.nav.fnav.ingest(_p, _r["nav_obs"])
            # 60 s health + BRDC cross-check (Kepler only; alm_sys is 'E' for a Galileo chain)
            if ctx.nav.fnav is not None and _now() - ctx.nav.fnav_log_t[0] > 60.0:
                ctx.nav.fnav_log_t[0] = _now()
                for _p in sorted(ctx.nav.fnav._p):
                    h = ctx.nav.fnav.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.fnav.ephemeris(_p)
                    xc = (
                        _fnav_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                        if (eph is not None and ctx.brdc_alm is not None)
                        else ""
                    )
                    _log(
                        "fnav PRN %d: %d pages, %d words, have %s, eph %s%s"
                        % (
                            _p,
                            h["pages"],
                            h["words"],
                            h["have"],
                            "YES" if eph is not None else "no",
                            xc,
                        )
                    )
                    ctx.dh_obs("GAL_E5AI_FNAV", _p, h, eph, xc)
        # BeiDou B2a B-CNAV2 from its aux data chain (--bcnav2-combiner).
        if ctx.nav.bcnav2_combiner:
            try:
                _baux = {
                    int(r["prn"]): r
                    for r in _get("%s/get_status" % ctx.nav.bcnav2_combiner)
                    if r.get("prn")
                }
            except Exception as e:
                _baux = {}
                _log_rl(
                    "bcnav2aux",
                    "bcnav2 aux combiner %s unreadable: %s"
                    % (ctx.nav.bcnav2_combiner, e),
                )
            for _p, _r in _baux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.bcnav2 is None:
                    from bcnav2_predictor import Bcnav2Predictor

                    ctx.nav.bcnav2 = Bcnav2Predictor()
                    _log(
                        "B-CNAV2 decoder armed on aux chain %s"
                        % ctx.nav.bcnav2_combiner
                    )
                ctx.nav.bcnav2.ingest(_p, _r["nav_obs"])
            # 60 s health + BRDC cross-check (alm_sys is 'C' for a BeiDou chain)
            if ctx.nav.bcnav2 is not None and _now() - ctx.nav.bcnav2_log_t[0] > 60.0:
                ctx.nav.bcnav2_log_t[0] = _now()
                for _p in sorted(ctx.nav.bcnav2._p):
                    h = ctx.nav.bcnav2.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.bcnav2.ephemeris(_p)
                    xc = (
                        _bcnav2_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                        if (eph is not None and ctx.brdc_alm is not None)
                        else ""
                    )
                    _log(
                        "bcnav2 PRN %d: %d frames, %d crc, have %s, eph %s%s"
                        % (
                            _p,
                            h["pages"],
                            h["words"],
                            h["have"],
                            "YES" if eph is not None else "no",
                            xc,
                        )
                    )
                    ctx.dh_obs("BDS_B2A_BCNAV2", _p, h, eph, xc)
        # BeiDou B1C B-CNAV1 from its aux data chain (--bcnav1-combiner).
        if ctx.nav.bcnav1_combiner:
            try:
                _c1aux = {
                    int(r["prn"]): r
                    for r in _get("%s/get_status" % ctx.nav.bcnav1_combiner)
                    if r.get("prn")
                }
            except Exception as e:
                _c1aux = {}
                _log_rl(
                    "bcnav1aux",
                    "bcnav1 aux combiner %s unreadable: %s"
                    % (ctx.nav.bcnav1_combiner, e),
                )
            for _p, _r in _c1aux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.bcnav1 is None:
                    from bcnav1_predictor import Bcnav1Predictor

                    ctx.nav.bcnav1 = Bcnav1Predictor()
                    _log(
                        "B-CNAV1 decoder armed on aux chain %s"
                        % ctx.nav.bcnav1_combiner
                    )
                ctx.nav.bcnav1.ingest(_p, _r["nav_obs"])
            if ctx.nav.bcnav1 is not None and _now() - ctx.nav.bcnav1_log_t[0] > 60.0:
                ctx.nav.bcnav1_log_t[0] = _now()
                for _p in sorted(ctx.nav.bcnav1._p):
                    h = ctx.nav.bcnav1.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.bcnav1.ephemeris(_p)
                    xc = (
                        _bcnav1_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                        if (eph is not None and ctx.brdc_alm is not None)
                        else ""
                    )
                    _log(
                        "bcnav1 PRN %d: %d frames, %d crc, have %s, eph %s%s"
                        % (
                            _p,
                            h["pages"],
                            h["words"],
                            h["have"],
                            "YES" if eph is not None else "no",
                            xc,
                        )
                    )
                    ctx.dh_obs("BDS_B1C_BCNAV1", _p, h, eph, xc)
        # GPS L1C-D CNAV-2 from its aux data chain (--cnav2-combiner; alm_sys 'G'). The
        # broker's own --combiner is the L1C-P pilot; the data symbols come off the L1C-D sibling.
        if ctx.nav.cnav2_combiner:
            try:
                _c2aux = {
                    int(r["prn"]): r
                    for r in _get("%s/get_status" % ctx.nav.cnav2_combiner)
                    if r.get("prn")
                }
            except Exception as e:
                _c2aux = {}
                _log_rl(
                    "cnav2aux",
                    "cnav2 aux combiner %s unreadable: %s"
                    % (ctx.nav.cnav2_combiner, e),
                )
            for _p, _r in _c2aux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.cnav2 is None:
                    from cnav2_predictor import Cnav2Predictor

                    ctx.nav.cnav2 = Cnav2Predictor()
                    _log(
                        "CNAV-2 decoder armed on aux chain %s" % ctx.nav.cnav2_combiner
                    )
                ctx.nav.cnav2.ingest(_p, _r["nav_obs"])
            if ctx.nav.cnav2 is not None and _now() - ctx.nav.cnav2_log_t[0] > 60.0:
                ctx.nav.cnav2_log_t[0] = _now()
                for _p in sorted(ctx.nav.cnav2._p):
                    h = ctx.nav.cnav2.health(_p)
                    if not h or not h["words"]:
                        continue
                    # No ephemeris: gps_cnav2 has no field table for the SF2 orbit set.
                    _log(
                        "cnav2 PRN %d: %d frames CRC-OK, toi=%s"
                        % (_p, h["words"], h["toi"])
                    )
                    ctx.dh_obs("GPS_L1CD_CNAV2", _p, h, None, "")
        if ctx.nav.navbits is not None and _now() - ctx.nav.log_t > 60.0:
            ctx.nav.log_t = _now()
            for _p in sorted(ctx.nav.navbits._p):
                h = ctx.nav.navbits.health(_p)
                if not h:
                    continue
                # The decoded orbit set (subframes 1-3) with a BRDC position cross-check. BRDC's
                # GPS record is the LNAV message, so dpos is near zero unless an offset in
                # gps_nav_decode.LNAV_EPH_FIELDS is wrong; the same ephemeris feeds the
                # decoded-ephemeris BRDC fallback for L1.
                eph_s = ""
                e = None
                if h["synced"]:
                    e = ctx.nav.navbits.ephemeris(_p)
                    if e is not None:
                        eph_s = " eph toe=%.0f e=%.3e" % (e["toe"], e["e"])
                        if ctx.brdc_alm is not None:
                            eph_s += _lnav_brdc_xcheck(
                                ctx.brdc_alm, ctx.alm_sys, _p, e, _log
                            )
                    _log(
                        "navbit PRN %d: %d sf decoded, %d pages, predict-mismatch %s%s"
                        % (
                            _p,
                            h["decoded_sf"],
                            h["pages"],
                            ("%.4f" % h["mismatch"])
                            if h["mismatch"] is not None
                            else "n/a",
                            eph_s,
                        )
                    )
                else:
                    # Not synced: the contiguous run, the total history, and what sync needs.
                    _log(
                        "navbit PRN %d: NO SYNC (contig run %d/%d, hist %d)"
                        % (_p, h["run"], h["need"], h["hist"])
                    )
                ctx.dh_obs("GPS_L1_LNAV", _p, h, e, eph_s)
        # CNAV decode health and the decoded ephemeris (message types 10 + 11), with the BRDC
        # cross-check: the decoded CNAV ephemeris and the downloaded BRDC (LNAV) ephemeris are
        # propagated to the same instant (the CNAV toe) and the ECEF position residual is
        # reported. Two independent encodings of one orbit, so a small residual validates the
        # whole decode chain. Kepler propagation only, cheap enough for the 60 s cadence.
        if ctx.nav.cnav is not None and _now() - ctx.nav.log_t > 60.0:
            ctx.nav.log_t = _now()
            for _p in sorted(ctx.nav.cnav._p):
                h = ctx.nav.cnav.health(_p)
                if not h:
                    continue
                eph_s = ""
                e = None
                if h["eph"]:
                    e = ctx.nav.cnav.ephemeris(_p)
                    if e is not None:
                        eph_s = " eph toe=%.0f e=%.3e" % (e["toe"], e["e"])
                        if ctx.brdc_alm is not None:
                            eph_s += _cnav_brdc_xcheck(
                                ctx.brdc_alm, ctx.alm_sys, _p, e, _log
                            )
                if h["synced"]:
                    _log(
                        "cnav PRN %d: %d msgs decoded, %d stored, %d emits, g2=%s%s"
                        % (_p, h["decoded"], h["messages"], h["emits"], h["g2"], eph_s)
                    )
                else:
                    _log(
                        "cnav PRN %d: NO DECODE (%d emits accumulated, g2=%s)"
                        % (_p, h["emits"], h["g2"])
                    )
                ctx.dh_obs(ctx.nav.cnav_sig, _p, h, e, eph_s)
        # B-CNAV3 decode health (the B2b chain's own message): CRC-passing frames, the message
        # types held, the last SOW, and once a Type 10 is in hand its ephemeris and BRDC
        # cross-check.
        if ctx.nav.bcnav3 is not None and _now() - ctx.nav.bcnav3_log_t[0] > 60.0:
            ctx.nav.bcnav3_log_t[0] = _now()
            for _p in sorted(ctx.nav.bcnav3._p):
                h = ctx.nav.bcnav3.health(_p)
                if not h or not h["words"]:
                    continue
                eph = ctx.nav.bcnav3.ephemeris(_p)
                xc = (
                    _bcnav3_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                    if (eph is not None and ctx.brdc_alm is not None)
                    else ""
                )
                _log(
                    "bcnav3 PRN %d: %d frames CRC-OK, have %s, sow=%s, eph %s%s"
                    % (
                        _p,
                        h["words"],
                        h["have"],
                        h["sow"],
                        "YES" if eph is not None else "no",
                        xc,
                    )
                )
                ctx.dh_obs("BDS_B2B_BCNAV3", _p, h, eph, xc)
