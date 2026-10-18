"""Narrowing the search: push per-satellite predicted Doppler, and the NH overlay alignment,
to the detectors.

Each search then scans a window around the prediction instead of the blind grid, which is what
makes acquisition affordable at all. Pure output: it POSTs hints and returns nothing.

A hint is only as good as the clock it carries. When the receiver clock bias is stale the
margin widens rather than the hint narrowing. A confidently wrong narrow window is worse than
no hint: the search then cannot find the very signal whose detection would correct the bias,
so the error sustains itself. See `clockbias.py`.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _post, _log, _log_rl, axis_now
from gnss_broker.clockhealth import dr_solve_healthy


def stage_narrow_search(ctx):
    """Push per-PRN predicted Doppler (--narrow-search) and NH alignment hints (--nh-hint) to
    the detectors.

    Doppler hints cover every up, capable PRN, with the narrow margin once the clock bias is
    solved and not stale and the wide margin otherwise (see the module docstring)."""
    if ctx.args.narrow_search and ctx.args.almanac and ctx.pred:
        margin = (
            ctx.args.search_margin_hz
            if ctx.cb.ema is not None and not ctx.cb.stale
            else ctx.args.search_margin_wide_hz
        )
        hints = [
            dict(prn=p, doppler_hz=ctx.pred[p][0] + ctx.cb.value, margin_hz=margin)
            for p in sorted(ctx.pred)
            if (ctx.up is None or p in ctx.up)
            and (ctx.capable is None or p in ctx.capable)
        ]
        # NH alignment hint (--nh-hint). The acquire builds a full surface per overlay
        # alignment, so scanning all of them is most of a pass. nh is the overlay chip index
        # at transmit, so the ephemeris gives it outright (pred[3] is the range in m, pred[4]
        # the broadcast satellite clock in s):
        #   nh = round((gpst(t) - range/c + clk_sv) / period) mod overlay_len
        # That leaves one global constant, the receiver clock reference, shared by every
        # satellite and measured from any detection. So a satellite never detected is hinted
        # too, where a first acquisition would otherwise pay the full scan. The search scans
        # nh_hint_span alignments either side of the hint, which absorbs the measured offset's
        # scatter of a couple of alignments (its cause is not established).
        nh_hints = []
        if ctx.args.nh_hint and ctx.pred and ctx.utc0_sample0:
            try:
                import gnss_ephemeris as _nh2

                _per = ctx.args.code_length / ctx.args.chip_rate_hz

                def _pred_nh(_p, _t):
                    _v = ctx.pred[_p]
                    return (
                        int(
                            round(
                                (_nh2.gpst_of_utc(_t) - _v[3] / _nh2.C_LIGHT + _v[4])
                                / _per
                            )
                        )
                        % ctx.args.nh_overlay_len
                    )

                # (a) Re-measure the constant from each new detection. A PRN counts again
                # only when its ref_hop moves: re-appending an unchanged detection every
                # cycle would fill the window with copies, and nh_hint_min_samples would then
                # count uptime rather than evidence.
                #
                # Samples age out after nh_hint_max_age_s, and with fewer than
                # nh_hint_min_samples fresh the offset is dropped (below). Without that
                # expiry a wrong offset sustains itself: the search, narrowed onto the wrong
                # overlay phase, detects nothing that could correct it.
                #
                # Nothing is learned while the clock solve is not established
                # (dr_solve_healthy False): detections it does not vouch for (noise, a
                # collapsed sky) would teach a wrong offset that the narrowed search then
                # confirms. The expiry widens the search on its own meanwhile.
                _learn = (
                    dr_solve_healthy(getattr(ctx, "dr_state", None), ctx.t0)
                    is not False
                )
                if not _learn:
                    _log_rl(
                        "nhhint-unhealthy",
                        "nh hint: search solve not established -- not learning an offset "
                        "from these detections",
                        every_s=120.0,
                    )
                for _p, (_nh, _rh) in ctx.nho.seen.items() if _learn else ():
                    if _p not in ctx.pred or ctx.nho.last_rh.get(_p) == _rh:
                        continue
                    ctx.nho.last_rh[_p] = _rh
                    _t = ctx.utc0_sample0 + _rh / ctx.args.hops_per_sec
                    ctx.nho.off_hist.append(
                        (ctx.t0, (_pred_nh(_p, _t) - _nh) % ctx.args.nh_overlay_len)
                    )
                del ctx.nho.off_hist[:-64]
                _fresh = [
                    o
                    for (_ts, o) in ctx.nho.off_hist
                    if ctx.t0 - _ts <= ctx.args.nh_hint_max_age_s
                ]
                if (
                    len(_fresh) < ctx.args.nh_hint_min_samples
                    and ctx.nho.offset[0] is not None
                ):
                    _log(
                        "nh hint EXPIRED: %d sample(s) inside %.0f s (need %d) -- "
                        "dropping the offset so the search widens instead of staying "
                        "narrowed on a stale one"
                        % (
                            len(_fresh),
                            ctx.args.nh_hint_max_age_s,
                            ctx.args.nh_hint_min_samples,
                        )
                    )
                    ctx.nho.offset[0] = None
                # (b) Circular median: the offsets cluster, so rotate to the mode before
                # taking it, or a cluster straddling the 0/overlay_len wrap averages to
                # nonsense.
                if len(_fresh) >= ctx.args.nh_hint_min_samples:
                    _mode = max(set(_fresh), key=_fresh.count)
                    _rot = [
                        (
                            (o - _mode + ctx.args.nh_overlay_len // 2)
                            % ctx.args.nh_overlay_len
                        )
                        - ctx.args.nh_overlay_len // 2
                        for o in _fresh
                    ]
                    _rot.sort()
                    ctx.nho.offset[0] = (
                        _mode + _rot[len(_rot) // 2]
                    ) % ctx.args.nh_overlay_len
                # (c) Hint every visible satellite, detected or not, referenced to this
                # cycle's hop (the hop-axis now, transport.axis_now, and its UTC for the
                # predicted alignment) so the search propagates it over seconds rather than a
                # minute.
                _ax = axis_now(ctx)
                if ctx.nho.offset[0] is not None and _ax is not None:
                    _rh_now = int(round(_ax * ctx.args.hops_per_sec))
                    _t_now = ctx.utc0_sample0 + _ax
                    nh_hints = [
                        dict(
                            prn=int(_p),
                            nh=(_pred_nh(_p, _t_now) - ctx.nho.offset[0])
                            % ctx.args.nh_overlay_len,
                            ref_hop=_rh_now,
                        )
                        for _p in ctx.pred
                        if ctx.pred[_p][2] >= ctx.args.mask_deg
                    ]
                    _log_rl(
                        "nhhint",
                        "nh hint: offset %d (%d samples) -> %d sat(s)"
                        % (ctx.nho.offset[0], len(_fresh), len(nh_hints)),
                        every_s=60.0,
                    )
            except Exception as e:
                _log_rl("nhhint-err", "nh hint failed: %s" % e, every_s=60.0)
        pushed = 0
        for d_ep in ctx.detectors:
            try:
                _post("%s/set_doppler_hints" % d_ep, hints)
                pushed += 1
            except Exception as e:
                _log("set_doppler_hints %s failed: %s" % (d_ep, e))
            # A separate try: a detector whose binary lacks /set_nh_hint fails here, and
            # sharing the block above would report it as the Doppler hint failing.
            # Rate-limited because such a binary fails every cycle.
            if nh_hints:
                try:
                    _post("%s/set_nh_hint" % d_ep, nh_hints)
                except Exception as e:
                    _log_rl(
                        "nhpost-%s" % d_ep,
                        "set_nh_hint %s failed (old binary?): %s" % (d_ep, e),
                        every_s=120.0,
                    )
        _log_rl(
            "narrow",
            "narrowed search: %d hints +-%d Hz (%s) -> %d/%d detectors"
            % (
                len(hints),
                int(margin),
                (
                    "bias solved"
                    if ctx.cb.ema is not None and not ctx.cb.stale
                    else "bias STALE, wide re-solve"
                    if ctx.cb.ema is not None
                    else "pre-solve wide"
                ),
                pushed,
                len(ctx.detectors),
            ),
        )
