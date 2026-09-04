"""S4: the CM/CL sibling chain -- seed a long-code tracker from this chain's solution.

The sibling despreads the SAME satellite on a different code (GPS L2 CM/CL), so it needs no
search of its own: the visible set, the predicted Doppler and the receiver clock have all been
solved here. It consumes and never feeds back, which is why this stage has no outputs into the
rest of the cycle.

⚠️ THE ANCHOR EPOCH IS EVALUATED SEPARATELY, NOT EXTRAPOLATED. Linear extrapolation back to
utc0 is no cure for orbit curvature (tens of ms over hours), so this runs a SECOND model
evaluation at the fixed anchor epoch, cached per ephemeris refresh -- the anchor never moves,
only the ephemeris does.

@author Keith Vanderlinde
"""

from datetime import datetime, timezone

from gnss_broker.sky import C_LIGHT, brdc_predict
from gnss_broker.transport import _now, _get, _post, _log, _log_rl


def stage_cl_sibling(ctx):
    """S4: the CM/CL SIBLING CHAIN -- seed the long-code tracker from this chain's solution.
        
    The sibling despreads the SAME satellite on a different code (GPS L2 CM/CL), so it needs no
    search of its own: everything it wants -- the visible set, the predicted Doppler, the receiver
    clock -- has already been solved here. It consumes; it never feeds back. That is why this
    whole stage has ZERO outputs into the rest of the cycle, and why it is the first block that
    could be lifted out of the loop body unchanged.
        
    ⚠️ THE ANCHOR EPOCH IS EVALUATED SEPARATELY, NOT EXTRAPOLATED. Linear extrapolation back to
    utc0 is no cure for orbit curvature (tens of ms over hours), so this runs a SECOND model
    evaluation at the fixed anchor epoch, cached per ephemeris refresh -- the anchor never moves,
    only the ephemeris does."""
    if ctx.cls.tracker and ctx.utc0_sample0 and ctx.args.almanac and ctx.pred:
        # tau AND the SV clock must be evaluated AT THE ANCHOR EPOCH (utc0_sample0),
        # because that is where cp is referenced. The first deploy evaluated them at
        # "now": invisible at launch, but the per-sat error grows at range_rate/c (up to
        # +-2.7 us/s) -- +-10 ms/h, a guaranteed universal mis-pin by hour 2-3 -- and
        # LINEAR extrapolation back is no cure (orbit curvature ~tens of ms over hours).
        # So: a second model evaluation at the FIXED anchor epoch, cached per ephemeris
        # refresh (the anchor never moves; only the ephemeris does).
        if ctx.brdc_alm is not None:
            _k0 = (round(ctx.utc0_sample0, 3), ctx.brdc_alm.get("eph_t"))
            if ctx.cls.pred0.get("key") != _k0:
                try:
                    ctx.cls.pred0["val"] = brdc_predict(
                        ctx.brdc_alm, ctx.args.lat, ctx.args.lon, ctx.args.alt, ctx.alm_sys, ctx.alm_min_prn,
                        datetime.fromtimestamp(ctx.utc0_sample0, tz=timezone.utc),
                        ctx.args.carrier_hz)
                    ctx.cls.pred0["key"] = _k0
                    _log("CL: anchor-epoch geometry rebuilt (%d sats)"
                         % len(ctx.cls.pred0["val"]))
                except Exception as e:
                    _log("CL: anchor-epoch predict failed (%s); now-epoch fallback "
                         "(fine will drift ~ms/10min)" % e)
        _pred0 = ctx.cls.pred0.get("val") or {}
        cl_payload = []
        _fines = []
        for d in ctx.payload:
            pv = ctx.pred.get(d["prn"])
            # No geometry -> no k -> no CL row (fail closed; CM unaffected). Below the
            # elevation mask -> ALSO no row: those seeds are the below-horizon NOISE
            # PROBES, whose cp is deliberately noise -- deriving CL from them wastes a
            # tracker slot and their "fine" poisoned the first margin analysis (the
            # same el<0 trap the obs-aggregate rule exists for).
            if pv is None or pv[2] < ctx.args.mask_deg:
                continue
            _g = _pred0.get(d["prn"])
            tau0 = (_g[3] if _g is not None else pv[3]) / C_LIGHT
            clk0 = _g[4] if _g is not None else pv[4]
            # Their segment-search correction (cl_segsearch) on OUR parameterised epoch:
            # LC_EPOCH/LC_SEG replaced the hardcoded 1.5 s / 75 segments so the CL assist
            # is not L2C-CL-only. Defaults are 1.5/75, so this is a no-op on the prototype.
            t_sv = (ctx.utc0_sample0 - tau0 + clk0 + ctx.args.cl_time_adjust - ctx.cls.toff[0]
                    + ctx.cls.segsearch["corr"] * ctx.cls.seg_s)
            cl_chips = (t_sv % ctx.lc_epoch) * ctx.args.chip_rate_hz
            cp_cm = d["code_phase_chips"] % ctx.code_len
            k = int(round((cl_chips - cp_cm) / ctx.code_len))
            fine_ms = (cl_chips - cp_cm - k * ctx.code_len) / ctx.args.chip_rate_hz * 1e3
            k %= ctx.lc_seg
            _fines.append(fine_ms)
            if abs(fine_ms) > 5.0:
                # Half the +-10 ms budget gone AFTER centering: the seed still goes out
                # (a wrong k reads as CL noise, which the verify below names;
                # withholding would silently dark the chain instead of showing it).
                _log_rl("clthin-%d" % d["prn"],
                        "CL PIN MARGIN THIN: PRN %d fine %+.2f ms of +-10 (post-center; "
                        "clock-offset est %+.2f ms)"
                        % (d["prn"], fine_ms, ctx.cls.toff[0] * 1e3))
            # K-SCAN: for the one probe PRN, offset the seed by the current step --
            # whole segments (segment mode) or fractional chips (comb mode) -- so its
            # CL row despreads at the shifted position. Everything else (fine, k
            # report, auto-center) uses the true k untouched; only the probe's seed is
            # shifted, so the scan cannot perturb the fleet's pin.
            _cp_extra = 0.0
            k_seed = k
            if ctx.args.cl_kscan_prn and d["prn"] == ctx.args.cl_kscan_prn:
                _off = ctx.cls.kscan_seq[(ctx.cls.kscan[0] // max(ctx.args.cl_kscan_dwell, 1))
                                  % len(ctx.cls.kscan_seq)]
                if ctx.cls.kscan_frac:
                    _cp_extra = _off
                else:
                    k_seed = (k + int(_off)) % ctx.lc_seg
            dcl = {kk: d[kk] for kk in ("prn", "doppler_hz", "code_phase_rate", "ref_hop",
                                        "doppler_rate_hz_s", "carrier_trim_hz") if kk in d}
            # Their k-scan probe (k_seed/_cp_extra) on OUR parameterised segment count.
            dcl["code_phase_chips"] = ((cp_cm + k_seed * ctx.code_len + _cp_extra)
                                       % (float(ctx.lc_seg) * ctx.code_len))
            cl_payload.append(dcl)
            kp = ctx.cls.k.get(d["prn"])
            if kp is not None and kp != k:
                msg = ("CL k-step PRN %d: %d -> %d (fine %+.2f ms)"
                       % (d["prn"], kp, k, fine_ms))
                if (k - kp) % ctx.lc_seg in (1, ctx.lc_seg - 1):
                    _log_rl("clk-%d" % d["prn"], msg)  # geometry advancing: routine
                else:
                    _log("CL K-JUMP (not +-1 -- clock/anchor fault?): " + msg)
            ctx.cls.k[d["prn"]] = k
            ctx.cl_report.append("PRN %d k=%d fine %+.1f ms" % (d["prn"], k, fine_ms))
        # AUTO-CENTER: the across-sat MEDIAN fine is the common receiver-clock/anchor
        # offset (class-1 continuous state -- measured +4.5 ms on first light, i.e. half
        # the +-10 ms pin budget spent on a knowable constant). A slow EMA of the median
        # (tau ~10 s at 5 Hz) folds it back into the next cycle's t_sv, re-centering
        # every sat's margin; the +-8 ms clamp keeps a broken clock from walking the pin
        # off a segment. Median (not mean): one sat mid k-step must not drag the fleet.
        # The k pins themselves stay integer and per-cycle -- this only recenters the
        # window they are rounded in.
        if _fines:
            _med = sorted(_fines)[len(_fines) // 2] * 1e-3
            ctx.cls.toff[0] = max(-8e-3, min(8e-3, ctx.cls.toff[0] + 0.02 * _med))
            _log_rl("cltoff", "CL clock-offset est %+.2f ms (median fine %+.2f ms, "
                    "%d sats)" % (ctx.cls.toff[0] * 1e3, _med * 1e3, len(_fines)))
        if cl_payload:
            try:
                _post("%s/set_seeds" % ctx.cls.tracker, cl_payload)
            except Exception as e:
                _log("CL set_seeds %s failed: %s" % (ctx.cls.tracker, e))
        # VERIFY (the other half of the class-2 pin): CL deep_snr vs CM per PRN. Equal
        # power split -> a right k reads ~CM's deep; a wrong k despreads noise. Read
        # beside the k it verifies, in this log, so the pin and its evidence never
        # separate.
        if ctx.cls.combiner:
            try:
                cls_ = {int(r["prn"]): r for r in _get("%s/get_status" % ctx.cls.combiner)}
                pairs = []
                for prn in sorted(ctx.cls.k):
                    cm_d = (ctx.status.get(prn) or {}).get("deep_snr") or 0.0
                    cl_d = (cls_.get(prn) or {}).get("deep_snr") or 0.0
                    pairs.append("%d:%.0f/%.0f" % (prn, cm_d, cl_d))
                if pairs:
                    _log_rl("clverify", "CL verify (PRN:cm/cl deep): " + " ".join(pairs))
                # SEGMENT AUTO-SEARCH, judged on this same verify data. Step only when
                # the fleet is unambiguously dead (>=2 strong CM sats, ZERO green CL) --
                # a partly-green fleet must never be stepped away from -- and latch the
                # moment >=2 strong sats read green. Disabled while a k-scan diagnostic
                # is shifting the probe's seed.
                if (ctx.args.cl_autoseg and not ctx.args.cl_kscan_prn
                        and not ctx.cls.segsearch["latched"]):
                    _strong = [prn for prn in ctx.cls.k
                               if ((ctx.status.get(prn) or {}).get("deep_snr") or 0.0) > 50.0]
                    _green = [prn for prn in _strong
                              if ((cls_.get(prn) or {}).get("deep_snr") or 0.0)
                              > ((ctx.status.get(prn) or {}).get("deep_snr") or 0.0) / 3.0]
                    _nowv = _now()
                    if ctx.cls.segsearch["t_step"] == 0.0:
                        ctx.cls.segsearch["t_step"] = _nowv
                    elif len(_strong) >= 2 and len(_green) >= 2:
                        ctx.cls.segsearch["latched"] = True
                        _log("CL SEG-SEARCH LATCHED: correction %+d segment(s) "
                             "(compensating a %+.0f ms utc0_sample0 anchor error); "
                             "%d/%d strong sats green"
                             % (ctx.cls.segsearch["corr"], -ctx.cls.segsearch["corr"] * ctx.cls.seg_s * 1e3,
                                len(_green), len(_strong)))
                    elif (len(_strong) >= 2 and not _green
                          and _nowv - ctx.cls.segsearch["t_step"] > ctx.args.cl_autoseg_dwell):
                        ctx.cls.segsearch["idx"] = ((ctx.cls.segsearch["idx"] + 1)
                                               % len(ctx.cls.spiral))
                        ctx.cls.segsearch["corr"] = ctx.cls.spiral[ctx.cls.segsearch["idx"]]
                        ctx.cls.segsearch["t_step"] = _nowv
                        _log("CL SEG-SEARCH: fleet dead under strong CM (%d strong, 0 "
                             "green) -- trying correction %+d segment(s)"
                             % (len(_strong), ctx.cls.segsearch["corr"]))
                # K-SCAN readout. The seeded offset for THIS cycle is
                # seq[(cycle//dwell) % n]; the combiner is integrating that same offset
                # (the seed goes out just above, then we read deep next). CL deep takes
                # tens of seconds to build, and the tracker needs a few cycles to re-lock
                # after a segment jump -- so only RECORD in the back half of each dwell,
                # and only DECLARE a result after a full sweep has completed, requiring a
                # winner that clears noise by a real margin.
                if ctx.args.cl_kscan_prn:
                    _p = ctx.args.cl_kscan_prn
                    _dw = max(ctx.args.cl_kscan_dwell, 1)
                    _idx = (ctx.cls.kscan[0] // _dw) % len(ctx.cls.kscan_seq)
                    _cur = ctx.cls.kscan_seq[_idx]
                    _pos = ctx.cls.kscan[0] % _dw            # position within this dwell
                    _cl = (cls_.get(_p) or {}).get("deep_snr") or 0.0
                    _cm = (ctx.status.get(_p) or {}).get("deep_snr") or 0.0
                    if _pos >= _dw // 2:              # settled back half only
                        ctx.cls.kscan_deep[_cur] = max(ctx.cls.kscan_deep.get(_cur, 0.0), _cl)
                    _log_rl("kscan-%d" % _p,
                            "CL KSCAN PRN %d: %s (dwell %d/%d) cl_deep %.0f cm %.0f"
                            % (_p, ctx.cls.kfmt(_cur), _pos, _dw, _cl, _cm), every_s=5.0)
                    # a full sweep completes when we return to seq index 0 having filled
                    # every offset; declare once per sweep.
                    if (_idx == 0 and _pos == 0 and ctx.cls.kscan[0] > 0
                            and len(ctx.cls.kscan_deep) >= len(ctx.cls.kscan_seq)):
                        _best = max(ctx.cls.kscan_deep, key=ctx.cls.kscan_deep.get)
                        _bd = ctx.cls.kscan_deep[_best]
                        _2nd = sorted(ctx.cls.kscan_deep.values())[-2]
                        _cmp = (ctx.status.get(_p) or {}).get("deep_snr") or 0.0
                        _clear = _bd > 20.0 and _bd > 3.0 * max(_2nd, 1.0)
                        _win_says = (
                            ("SUB-CHIP/COMB FAULT, offset %s chips" % ctx.cls.kfmt(_best)
                             if _best != 0 else
                             "true cp CORRECT -- fault is not a sub-chip seed offset")
                            if ctx.cls.kscan_frac else
                            ("WHOLE-SEGMENT ANCHOR BUG, magnitude %s" % ctx.cls.kfmt(_best)
                             if _best != 0 else
                             "true k CORRECT -- fault is NOT the segment pin"))
                        _log("CL KSCAN PRN %d SWEEP: %s -> %s"
                             % (_p, " ".join("%s:%.0f" % (ctx.cls.kfmt(o), ctx.cls.kscan_deep[o])
                                             for o in sorted(ctx.cls.kscan_deep)),
                                ("best %s clears noise %.0fx: %s" % (
                                    ctx.cls.kfmt(_best), _bd / max(_2nd, 1.0), _win_says))
                                if _clear else
                                "NO offset in this range despreads (best %s only %.0f "
                                "vs cm %.0f) -- %s" % (
                                    ctx.cls.kfmt(_best), _bd, _cmp,
                                    "not a sub-chip seed offset either; the fault is "
                                    "past the seed (replica/carrier/comb in the C++)"
                                    if ctx.cls.kscan_frac else
                                    "not a small whole-segment error; widen the range "
                                    "or look past the pin")))
                        ctx.cls.kscan_deep.clear()          # fresh accumulation next sweep
                ctx.cls.kscan[0] += 1
            except Exception as e:
                _log_rl("clverify", "CL verify poll failed: %s" % e)
