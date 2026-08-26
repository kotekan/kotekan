"""Decoding the broadcast navigation message, and cross-checking it against BRDC.

Eight decoders (LNAV, CNAV, CNAV-2, F/NAV, I/NAV, B-CNAV1/2/3), a BRDC source to compare
against, and a continuous agreement monitor. The decoders are created lazily and PERSIST
across cycles on `ctx.nav` -- one that is rebuilt has forgotten the subframes it had
collected and can never complete an ephemeris.

⚠️ OFF IN PRODUCTION. `--nav-bits` is not set in the CHORD yaml, so nothing here runs on the
live instrument and no fixture reaches it. Changes are unverified until the flag is armed --
the same blind spot the carrier loop has, and the reason a latent bug can sit in either
indefinitely.

⚠️ THE CROSS-CHECK IS THE POINT, NOT THE DECODE. A decoder that produces an ephemeris nobody
compared against broadcast is a second authority that can drift from the first without either
side erroring. `_dh_obs` records the agreement continuously so a drift shows up as a trend
rather than as a surprise.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _now, _get, _log, _log_rl
from gnss_broker.sky import (
    _lnav_brdc_xcheck, _cnav_brdc_xcheck, _cnav2_brdc_xcheck, _inav_brdc_xcheck,
    _fnav_brdc_xcheck, _bcnav1_brdc_xcheck, _bcnav2_brdc_xcheck, _bcnav3_brdc_xcheck,
)


def stage_nav_bits(ctx):
    """NAV BITS: decode the broadcast navigation message and cross-check it against BRDC.
        
    Holds ELEVEN lazily-created decoder objects (LNAV, CNAV, CNAV2, FNAV, INAV, BCNAV1/2/3, the
    BRDC source and the agreement monitor). They are created on the first row that needs them and
    must persist across cycles -- hence the nonlocal list, which is the whole reason this block
    resisted extraction until its state was named.
        
    ⚠️ OFF IN PRODUCTION (`--nav-bits` is not set in the CHORD yaml), so the fixture gate is BLIND
    to every line below it. Changes here are unverified until the flag is armed."""
    if ctx.args.nav_bits:
        for _p, _r in ctx.status.items():
            if "nav_obs" not in _r:
                continue
            if ctx.nav.health is None:
                from navbit_health import BitAgreement
                ctx.nav.health = BitAgreement(log=_log)
            # Route nav_obs to the decoder this band actually speaks. L1CA carries LNAV
            # (periodic subframes -> future bits); L2C-CM / L5-I carry CNAV (FEC+CRC, no
            # fixed schedule -> decode + shadow-serve decoded spans). Sending CNAV symbols
            # to the LNAV frame-sync just spins forever finding no preamble.
            if ctx.args.nav_decoder == "cnav":
                if ctx.nav.cnav is None:
                    from cnav_predictor import CnavPredictor
                    ctx.nav.cnav = CnavPredictor(log=_log)
                    _log("CNAV decoder armed (combiner exports nav_obs)")
                ctx.nav.cnav.ingest(_p, _r["nav_obs"])
            elif ctx.args.nav_decoder == "bcnav3":
                # BeiDou B2b B-CNAV3: the PRIMARY chain's own nav decoder (NB-LDPC block, not a
                # bit-prediction scheme -> no peel role, pure decode + BRDC xcheck).
                if ctx.nav.bcnav3 is None:
                    from bcnav3_predictor import Bcnav3Predictor
                    ctx.nav.bcnav3 = Bcnav3Predictor(log=_log)
                    _log("B-CNAV3 decoder armed (combiner exports nav_obs)")
                ctx.nav.bcnav3.ingest(_p, _r["nav_obs"])
            else:
                if ctx.nav.navbits is None:
                    from navbit_predictor import LnavPredictor
                    ctx.nav.navbits = LnavPredictor(log=_log)
                    _log("nav-bit predictor armed (combiner exports nav_obs)")
                ctx.nav.navbits.ingest(_p, _r["nav_obs"])
            # ...and ask the question no single gate was asking continuously (decoder-
            # agnostic): did the bits we PUBLISHED last cycle actually match the ones coming
            # out of the sky? Scored only where the satellite is strong enough that a
            # disagreement means OUR bits are wrong rather than the air reading being wrong.
            ctx.nav.health.score(_p, _r["nav_obs"], _r.get("deep_snr"))
        # AUXILIARY CNAV CHAIN (--cnav-combiner, S4): a second combiner polled only for its
        # CNAV symbols. At L5 this broker's own combiner is the Q PILOT -- its nav_obs are
        # deterministic overlay predictions handled above -- while CNAV arrives on the
        # derived L5-I data sibling, which no broker otherwise reads. Failure here is
        # deliberately NON-FATAL and quiet-ish: this is an observability path, and a chain
        # that has not acquired yet simply returns nothing. It must never disturb the
        # tracking loop that shares this cycle.
        if ctx.nav.cnav_combiner:
            try:
                _aux = {int(r["prn"]): r for r in _get("%s/get_status" % ctx.nav.cnav_combiner)
                        if r.get("prn")}
            except Exception as e:
                _aux = {}
                _log_rl("cnavaux", "cnav aux combiner %s unreadable: %s"
                        % (ctx.nav.cnav_combiner, e))
            for _p, _r in _aux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.cnav is None:
                    from cnav_predictor import CnavPredictor
                    ctx.nav.cnav = CnavPredictor(log=_log)
                    _log("CNAV decoder armed on aux chain %s" % ctx.nav.cnav_combiner)
                ctx.nav.cnav.ingest(_p, _r["nav_obs"])
        # S5 D-component #1: Galileo E1B I/NAV, the exact CNAV-aux analogue.
        if ctx.nav.inav_combiner:
            try:
                _iaux = {int(r["prn"]): r for r in _get("%s/get_status" % ctx.nav.inav_combiner)
                         if r.get("prn")}
            except Exception as e:
                _iaux = {}
                _log_rl("inavaux", "inav aux combiner %s unreadable: %s"
                        % (ctx.nav.inav_combiner, e))
            for _p, _r in _iaux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.inav is None:
                    from inav_predictor import InavPredictor
                    ctx.nav.inav = InavPredictor(log=_log)
                    _log("I/NAV decoder armed on aux chain %s" % ctx.nav.inav_combiner)
                ctx.nav.inav.ingest(_p, _r["nav_obs"])
            # 60 s health + BRDC cross-check (Kepler only; alm_sys is 'E' for the GAL broker).
            # I/NAV rides E1B on L1 and E5b-I on the mid band -- SAME message + decoder; label
            # the decode-health obs by the actual carrier (from the aux combiner name).
            if ctx.nav.inav is not None and _now() - ctx.nav.inav_log_t[0] > 60.0:
                ctx.nav.inav_log_t[0] = _now()
                _inav_sig = "GAL_E5BI_INAV" if "e5b" in ctx.nav.inav_combiner else "GAL_E1B_INAV"
                for _p in sorted(ctx.nav.inav._p):
                    h = ctx.nav.inav.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.inav.ephemeris(_p)
                    xc = (_inav_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                          if (eph is not None and ctx.brdc_alm is not None) else "")
                    _log("inav PRN %d: %d pages, %d words, have %s, eph %s%s"
                         % (_p, h["pages"], h["words"], h["have"],
                            "YES" if eph is not None else "no", xc))
                    ctx.dh_obs(_inav_sig, _p, h, eph, xc)
        # S5 D-component #2: Galileo E5a-I F/NAV, the exact I/NAV-aux analogue on L5.
        if ctx.nav.fnav_combiner:
            try:
                _faux = {int(r["prn"]): r for r in _get("%s/get_status" % ctx.nav.fnav_combiner)
                         if r.get("prn")}
            except Exception as e:
                _faux = {}
                _log_rl("fnavaux", "fnav aux combiner %s unreadable: %s"
                        % (ctx.nav.fnav_combiner, e))
            for _p, _r in _faux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.fnav is None:
                    from fnav_predictor import FnavPredictor
                    ctx.nav.fnav = FnavPredictor(log=_log)
                    _log("F/NAV decoder armed on aux chain %s" % ctx.nav.fnav_combiner)
                ctx.nav.fnav.ingest(_p, _r["nav_obs"])
            # 60 s health + BRDC cross-check (Kepler only; alm_sys is 'E' for the GAL broker)
            if ctx.nav.fnav is not None and _now() - ctx.nav.fnav_log_t[0] > 60.0:
                ctx.nav.fnav_log_t[0] = _now()
                for _p in sorted(ctx.nav.fnav._p):
                    h = ctx.nav.fnav.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.fnav.ephemeris(_p)
                    xc = (_fnav_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                          if (eph is not None and ctx.brdc_alm is not None) else "")
                    _log("fnav PRN %d: %d pages, %d words, have %s, eph %s%s"
                         % (_p, h["pages"], h["words"], h["have"],
                            "YES" if eph is not None else "no", xc))
                    ctx.dh_obs("GAL_E5AI_FNAV", _p, h, eph, xc)
        # S5 D-component #3: BeiDou B2a B-CNAV2 (first LDPC), the F/NAV-aux analogue on BDS.
        if ctx.nav.bcnav2_combiner:
            try:
                _baux = {int(r["prn"]): r for r in _get("%s/get_status" % ctx.nav.bcnav2_combiner)
                         if r.get("prn")}
            except Exception as e:
                _baux = {}
                _log_rl("bcnav2aux", "bcnav2 aux combiner %s unreadable: %s"
                        % (ctx.nav.bcnav2_combiner, e))
            for _p, _r in _baux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.bcnav2 is None:
                    from bcnav2_predictor import Bcnav2Predictor
                    ctx.nav.bcnav2 = Bcnav2Predictor(log=_log)
                    _log("B-CNAV2 decoder armed on aux chain %s" % ctx.nav.bcnav2_combiner)
                ctx.nav.bcnav2.ingest(_p, _r["nav_obs"])
            # 60 s health + BRDC cross-check (alm_sys is 'C' for the BDS broker)
            if ctx.nav.bcnav2 is not None and _now() - ctx.nav.bcnav2_log_t[0] > 60.0:
                ctx.nav.bcnav2_log_t[0] = _now()
                for _p in sorted(ctx.nav.bcnav2._p):
                    h = ctx.nav.bcnav2.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.bcnav2.ephemeris(_p)
                    xc = (_bcnav2_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                          if (eph is not None and ctx.brdc_alm is not None) else "")
                    _log("bcnav2 PRN %d: %d frames, %d crc, have %s, eph %s%s"
                         % (_p, h["pages"], h["words"], h["have"],
                            "YES" if eph is not None else "no", xc))
                    ctx.dh_obs("BDS_B2A_BCNAV2", _p, h, eph, xc)
        # S5 D-component #4 (LAST): BeiDou B1C B-CNAV1, the B-CNAV2-aux analogue on L1 BDS.
        if ctx.nav.bcnav1_combiner:
            try:
                _c1aux = {int(r["prn"]): r for r in _get("%s/get_status" % ctx.nav.bcnav1_combiner)
                          if r.get("prn")}
            except Exception as e:
                _c1aux = {}
                _log_rl("bcnav1aux", "bcnav1 aux combiner %s unreadable: %s"
                        % (ctx.nav.bcnav1_combiner, e))
            for _p, _r in _c1aux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.bcnav1 is None:
                    from bcnav1_predictor import Bcnav1Predictor
                    ctx.nav.bcnav1 = Bcnav1Predictor(log=_log)
                    _log("B-CNAV1 decoder armed on aux chain %s" % ctx.nav.bcnav1_combiner)
                ctx.nav.bcnav1.ingest(_p, _r["nav_obs"])
            if ctx.nav.bcnav1 is not None and _now() - ctx.nav.bcnav1_log_t[0] > 60.0:
                ctx.nav.bcnav1_log_t[0] = _now()
                for _p in sorted(ctx.nav.bcnav1._p):
                    h = ctx.nav.bcnav1.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.bcnav1.ephemeris(_p)
                    xc = (_bcnav1_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                          if (eph is not None and ctx.brdc_alm is not None) else "")
                    _log("bcnav1 PRN %d: %d frames, %d crc, have %s, eph %s%s"
                         % (_p, h["pages"], h["words"], h["have"],
                            "YES" if eph is not None else "no", xc))
                    ctx.dh_obs("BDS_B1C_BCNAV1", _p, h, eph, xc)
        # GPS L1C-D CNAV-2: the bcnav1-aux analogue on the L1C broker (alm_sys 'G'). The broker's
        # own --combiner is the L1C-P pilot; the CNAV-2 data symbols come off the derived L1C-D.
        if ctx.nav.cnav2_combiner:
            try:
                _c2aux = {int(r["prn"]): r for r in _get("%s/get_status" % ctx.nav.cnav2_combiner)
                          if r.get("prn")}
            except Exception as e:
                _c2aux = {}
                _log_rl("cnav2aux", "cnav2 aux combiner %s unreadable: %s"
                        % (ctx.nav.cnav2_combiner, e))
            for _p, _r in _c2aux.items():
                if "nav_obs" not in _r:
                    continue
                if ctx.nav.cnav2 is None:
                    from cnav2_predictor import Cnav2Predictor
                    ctx.nav.cnav2 = Cnav2Predictor(log=_log)
                    _log("CNAV-2 decoder armed on aux chain %s" % ctx.nav.cnav2_combiner)
                ctx.nav.cnav2.ingest(_p, _r["nav_obs"])
            if ctx.nav.cnav2 is not None and _now() - ctx.nav.cnav2_log_t[0] > 60.0:
                ctx.nav.cnav2_log_t[0] = _now()
                for _p in sorted(ctx.nav.cnav2._p):
                    h = ctx.nav.cnav2.health(_p)
                    if not h or not h["words"]:
                        continue
                    eph = ctx.nav.cnav2.ephemeris(_p)
                    xc = (_cnav2_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                          if (eph is not None and ctx.brdc_alm is not None) else "")
                    _log("cnav2 PRN %d: %d frames CRC-OK, toi=%s, eph %s%s"
                         % (_p, h["words"], h["toi"], "YES" if eph is not None else "no", xc))
                    ctx.dh_obs("GPS_L1CD_CNAV2", _p, h, eph, xc)
        # Recalibrate the constructed source: it needs the ephemeris, this cycle's geometry
        # (range + sat clock per PRN), and at least one SYNCED satellite to pin the common
        # capture-clock -> GPS offset. GPS LNAV only; other constellations get their own
        # source when their encoders exist.
        if (ctx.args.nav_bits_brdc and ctx.nav.navbits is not None and ctx.alm_sys == "G"
                and ctx.brdc_alm is not None and ctx.pred):
            if ctx.nav.brdc is None:
                from navbit_brdc import BrdcLnavSource
                ctx.nav.brdc = BrdcLnavSource(log=_log)
                _log("constructed-bit source armed (BRDC LNAV, un-synced PRNs)")
            try:
                ctx.nav.brdc.update(ctx.brdc_alm["eph"], ctx.pred, ctx.nav.navbits)
            except Exception as e:
                _log("navbrdc update failed: %s" % e)
        if ctx.nav.navbits is not None and _now() - ctx.nav.log_t > 60.0:
            ctx.nav.log_t = _now()
            for _p in sorted(ctx.nav.navbits._p):
                h = ctx.nav.navbits.health(_p)
                if not h:
                    continue
                # Now that LNAV extracts the orbit set (subframes 1-3), surface the live
                # ephemeris + a BRDC position cross-check exactly as CNAV/I-NAV do -- this is
                # the live validator of the LNAV_EPH_FIELDS bit offsets (near-zero dpos, since
                # BRDC IS the LNAV message) and the on-node BRDC-fallback source for L1.
                eph_s = ""
                e = None
                if h["synced"]:
                    e = ctx.nav.navbits.ephemeris(_p)
                    if e is not None:
                        eph_s = " eph toe=%.0f e=%.3e" % (e["toe"], e["e"])
                        if ctx.brdc_alm is not None:
                            eph_s += _lnav_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, e, _log)
                    _log("navbit PRN %d: %d sf decoded, %d pages, predict-mismatch %s%s"
                         % (_p, h["decoded_sf"], h["pages"],
                            ("%.4f" % h["mismatch"]) if h["mismatch"] is not None else "n/a",
                            eph_s))
                else:
                    # NOT synced == this PRN is NOT peeled (peel_require_bits). Say so, with
                    # the reason: contiguous run vs total history vs what sync needs.
                    _log("navbit PRN %d: NO SYNC (contig run %d/%d, hist %d)%s"
                         % (_p, h["run"], h["need"], h["hist"],
                            " -> CONSTRUCTED from BRDC"
                            if (ctx.nav.brdc is not None and ctx.nav.brdc.ready())
                            else " -> not peeled"))
                ctx.dh_obs("GPS_L1_LNAV", _p, h, e, eph_s)
            if ctx.nav.brdc is not None:
                # The calibration IS the trust boundary: a bad offset makes every
                # constructed bit confidently wrong, so state it every cycle. `verify`
                # scores constructed bits against a SYNCED satellite's own received bits
                # -- the live form of the offline 113820/113820 test.
                if ctx.nav.brdc.ready():
                    chk = []
                    for _p in sorted(ctx.nav.navbits._p):
                        r = ctx.nav.brdc.verify(_p, ctx.nav.navbits)
                        if r and r[0] >= 200:
                            chk.append("%d:%.1f%%" % (_p, 100.0 * r[1] / r[0]))
                    _log("navbrdc: offset %.6f s, spread %.2f ms, %d cal sats "
                         "(%d outliers dropped); verify %s"
                         % (ctx.nav.brdc.offset, (ctx.nav.brdc.spread or 0.0) * 1e3,
                            ctx.nav.brdc.n_cal, ctx.nav.brdc.n_rej, " ".join(chk) or "n/a"))
                else:
                    _log("navbrdc: NOT ready (%s)" % ctx.nav.brdc.why_not())
        # CNAV decode health + the live ephemeris (types 10+11). eph toe/e prove a decoded
        # ephemeris set. S4 EPHEMERIS CROSS-CHECK: propagate the live-decoded CNAV ephemeris
        # and the independently-downloaded BRDC (LNAV) ephemeris to the SAME absolute instant
        # (the CNAV toe) and report the ECEF position residual. Two independent nav messages,
        # two encodings (CNAV FEC+CRC vs LNAV parity), one truth -- a small residual VALIDATES
        # the whole decode chain against an outside reference, and is the foundation for
        # eventually trusting live CNAV ephemeris over the 2 h-latency download. Cheap (Kepler
        # propagation, no Viterbi), so it rides the existing 60 s health cadence.
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
                            eph_s += _cnav_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, e, _log)
                if h["synced"]:
                    _log("cnav PRN %d: %d msgs decoded, %d stored, %d emits, g2=%s%s"
                         % (_p, h["decoded"], h["messages"], h["emits"],
                            h["g2"], eph_s))
                else:
                    _log("cnav PRN %d: NO DECODE (%d emits accumulated, g2=%s)"
                         % (_p, h["emits"], h["g2"]))
                ctx.dh_obs(ctx.nav.cnav_sig, _p, h, e, eph_s)
        # B-CNAV3 decode health (BeiDou B2b PRIMARY chain). Frame decode is convention-complete
        # (LDPC + CRC); the message-type histogram + SOW logged here is exactly what maps the
        # ephemeris field table (Phase 2), after which eph + a BRDC dpos xcheck join (like cnav).
        if ctx.nav.bcnav3 is not None and _now() - ctx.nav.bcnav3_log_t[0] > 60.0:
            ctx.nav.bcnav3_log_t[0] = _now()
            for _p in sorted(ctx.nav.bcnav3._p):
                h = ctx.nav.bcnav3.health(_p)
                if not h or not h["words"]:
                    continue
                eph = ctx.nav.bcnav3.ephemeris(_p)
                xc = (_bcnav3_brdc_xcheck(ctx.brdc_alm, ctx.alm_sys, _p, eph, _log)
                      if (eph is not None and ctx.brdc_alm is not None) else "")
                _log("bcnav3 PRN %d: %d frames CRC-OK, have %s, sow=%s, eph %s%s"
                     % (_p, h["words"], h["have"], h["sow"],
                        "YES" if eph is not None else "no", xc))
                ctx.dh_obs("BDS_B2B_BCNAV3", _p, h, eph, xc)
