"""Narrowing the search: push per-satellite predicted Doppler to the detectors.

Each search then scans a window around the prediction instead of the blind grid, which is what
makes acquisition affordable at all. Pure output -- it POSTs hints and returns nothing.

⚠️ A HINT IS ONLY AS GOOD AS THE CLOCK IT CARRIES. When the receiver clock bias is stale the
margin must WIDEN rather than the hint narrow. A confidently wrong narrow window is worse than
no hint: the search then cannot find the very signal whose detection would correct the bias, so
the error sustains itself. See `clockbias.py`.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _post, _log, _log_rl


def stage_narrow_search(ctx):
    """2b: ALMANAC-NARROW THE SEARCH -- push per-PRN predicted Doppler to the detectors.
        
    Each search then scans a window around the prediction instead of the blind grid, which is what
    makes acquisition affordable. Pure output: it POSTs hints and returns nothing to the cycle.
        
    ⚠️ THE HINT IS ONLY AS GOOD AS THE CLOCK IT CARRIES. When the receiver clock bias is stale the
    margin must widen rather than the hint narrow -- a confidently wrong narrow window is worse
    than no hint at all, because the search then cannot find what it was told to look near."""
    if (ctx.args.narrow_search and ctx.args.almanac and ctx.pred) or (ctx.xb_pred and ctx.args.xband_seed):
        margin = (ctx.args.search_margin_hz
                  if ctx.cb.ema is not None and not ctx.cb.stale
                  else ctx.args.search_margin_wide_hz)
        hints = [dict(prn=p, doppler_hz=ctx.pred[p][0] + ctx.cb.value, margin_hz=margin)
                 for p in sorted(ctx.pred) if (ctx.up is None or p in ctx.up)
                 and (ctx.capable is None or p in ctx.capable)] if (ctx.args.almanac and ctx.pred) else []
        # RESCUE: for a sat the sibling band tracks but BRDC did NOT just hint (no pred /
        # no almanac), add a cross-band hint so the search narrows instead of going blind.
        # Wider margin than a BRDC hint -- the cross-band seed accuracy is the inter-band
        # MAD (~10 Hz) plus this band's own unsolved-LO width -- but far better than the
        # blind grid. Provably rescue-only: a sat BRDC covered is already in `hints`.
        if ctx.args.xband_seed and ctx.xb_pred:
            _hinted = {h["prn"] for h in hints}
            _xb_margin = max(margin, ctx.args.xband_hint_margin_hz)
            for _p, _xd in sorted(ctx.xb_pred.items()):
                if _p in _hinted or (ctx.capable is not None and _p not in ctx.capable):
                    continue
                if ctx.up is not None and _p not in ctx.up:
                    continue
                hints.append(dict(prn=_p, doppler_hz=_xd, margin_hz=_xb_margin))
                _log_rl("xbandseed-%d" % _p,
                        "XBAND RESCUE HINT PRN %d: %+.0f Hz (sibling tracks it, BRDC does "
                        "not) -> search narrows instead of blind" % (_p, _xd),
                        every_s=30.0)
        # SECONDARY-CODE ALIGNMENT HINT, the Doppler hint's twin and the bigger saving:
        # the acquire builds a FULL surface per alignment, so 20 of them are ~92% of a pass.
        # We echo back the stage's OWN last reported nh with the hop it was measured at, and
        # it propagates by counter arithmetic (one index per code period, an exact hop
        # count). Self-referential on purpose: no GPS time, no range model, nothing to
        # bootstrap -- a PRN with no hint simply scans all 20 as before, and one detection
        # establishes it. Only worth anything because the revisit is now short: over the
        # 1276 s revisit this fleet had before 2026-08-04 the prediction would drift 4.3
        # periods and be worse than useless.
        # OFF BY DEFAULT, and it must stay off until the reported nh is reproducible.
        # Measured 2026-08-04: propagating the search's own reported alignment forward gets
        # 4 of 26 right, whether or not the code phase is carried. PRN 3 reported 11->11
        # over 78.2 s and 11->10 over 77.7 s -- gaps differing by exactly 0 mod 20 periods,
        # so no propagation law fits. Posting a wrong hint NARROWS the scan away from the
        # truth, which is worse than not hinting: self-healing (the TTL restores the full
        # scan) but at the cost of a revisit each time.
        # ALIGNMENT HINT from the EPHEMERIS, not from echoing our own report back.
        #
        # nh is the overlay chip index at TRANSMIT, so BRDC gives it outright:
        #   nh = round((gpst(t) - range/c + clk_sv) / period) mod overlay_len
        # (the convention --nh-assist already uses for the combiner, proven to 0.01 chip).
        # What that leaves is ONE global constant -- the receiver clock reference -- shared
        # by every satellite and measured from any detection at all. Measured on sky
        # 2026-08-04: PRN 3 gave offset 16 on nine consecutive detections, and pooled across
        # six satellites 100% of samples landed within +-2 of 16.
        #
        # Strictly better than echoing our own nh back: no propagation law (mine got 4 of 26
        # right), no dependence on the previous detection, and it works for a satellite
        # NEVER detected -- which is the real prize, since a first acquisition otherwise
        # pays the full 20-way scan.
        #
        # OPEN: the +-2 jitter should not exist. nh is deterministic given ephemeris and
        # time, so a spread means something upstream is not -- likely the same cause that
        # broke propagation (suspect the 16-period acquire window against a 20-chip
        # overlay, 16 !== 0 mod 20). The span absorbs it; it does not explain it.
        nh_hints = []
        if ctx.args.nh_hint and ctx.pred and ctx.utc0_sample0:
            try:
                import gnss_ephemeris as _nh2
                _per = ctx.args.code_length / ctx.args.chip_rate_hz
                def _pred_nh(_p, _t):
                    _v = ctx.pred[_p]
                    return int(round((_nh2.gpst_of_utc(_t) - _v[3] / _nh2.C_LIGHT
                                      + (_v[4] if len(_v) > 4 else 0.0)) / _per)) % ctx.args.nh_overlay_len
                # (a) re-measure the constant from every fresh detection we have
                #
                # ⚠️ "FRESH" MEANS A NEW DETECTION, NOT A NEW CYCLE. This loop used to
                # append every nh_seen entry every cycle, so an unchanged detection was
                # re-appended every --interval seconds and the 64-sample window filled
                # with the SAME value repeated. nh_hint_min_samples then read that as 64
                # independent confirmations of an offset that might rest on one stale
                # detection -- a sample count that measures uptime rather than evidence.
                # ref_hop identifies the detection, so admit a PRN only when its ref_hop
                # has actually moved.
                #
                # AND THE HISTORY AGES. The comment at nh_seen's refresh says a wrong hint
                # is "recoverable -- the hint expires and the full scan returns", but
                # nothing implemented that expiry: nh_offset kept its last value forever
                # and the ±2-of-20 narrowing kept being pushed. That closes a loop with no
                # restoring force -- bad clock -> bad hint -> search narrowed onto the
                # wrong overlay phase -> no detections -> the clock cannot be re-solved --
                # and the only escape is the code clock random-walking back onto truth by
                # chance. Observed 2026-08-10 17:06-17:21: the clock wandered
                # 6996 -> 5095 -> 1797 -> 1555 -> 9210 -> 134 chips and only then snapped
                # to 150.8 and locked, ~15 min of a self-reinforcing outage that read as a
                # frontend sensitivity loss (docs 11.33).
                for _p, (_nh, _rh) in ctx.nho.seen.items():
                    if _p not in ctx.pred or ctx.nho.last_rh.get(_p) == _rh:
                        continue
                    ctx.nho.last_rh[_p] = _rh
                    _t = ctx.utc0_sample0 + _rh / ctx.args.hops_per_sec
                    ctx.nho.off_hist.append((ctx.t0, (_pred_nh(_p, _t) - _nh) % ctx.args.nh_overlay_len))
                del ctx.nho.off_hist[:-64]
                _fresh = [o for (_ts, o) in ctx.nho.off_hist
                          if ctx.t0 - _ts <= ctx.args.nh_hint_max_age_s]
                if len(_fresh) < ctx.args.nh_hint_min_samples and ctx.nho.offset[0] is not None:
                    _log("nh hint EXPIRED: %d sample(s) inside %.0f s (need %d) -- "
                         "dropping the offset so the search widens instead of staying "
                         "narrowed on a stale one"
                         % (len(_fresh), ctx.args.nh_hint_max_age_s, ctx.args.nh_hint_min_samples))
                    ctx.nho.offset[0] = None
                # (b) circular median: the offsets cluster, so rotate to the mode before
                # taking it, or a cluster straddling the 0/20 wrap averages to nonsense.
                if len(_fresh) >= ctx.args.nh_hint_min_samples:
                    _mode = max(set(_fresh), key=_fresh.count)
                    _rot = [((o - _mode + ctx.args.nh_overlay_len // 2) % ctx.args.nh_overlay_len)
                            - ctx.args.nh_overlay_len // 2 for o in _fresh]
                    _rot.sort()
                    ctx.nho.offset[0] = (_mode + _rot[len(_rot) // 2]) % ctx.args.nh_overlay_len
                # (c) hint EVERY visible sat, detected or not, at a hop the stage can
                # propagate over a few seconds rather than a minute
                if ctx.nho.offset[0] is not None:
                    _rh_now = int(round((ctx.t0 - ctx.utc0_sample0) * ctx.args.hops_per_sec))
                    nh_hints = [dict(prn=int(_p),
                                     nh=(_pred_nh(_p, ctx.t0) - ctx.nho.offset[0]) % ctx.args.nh_overlay_len,
                                     ref_hop=_rh_now)
                                for _p in ctx.pred if ctx.pred[_p][2] >= ctx.args.mask_deg]
                    _log_rl("nhhint", "nh hint: offset %d (%d samples) -> %d sat(s), span %d"
                            % (ctx.nho.offset[0], len(_fresh), len(nh_hints),
                               ctx.args.nh_hint_span), every_s=60.0)
            except Exception as e:
                _log_rl("nhhint-err", "nh hint failed: %s" % e, every_s=60.0)
        pushed = 0
        for d_ep in ctx.detectors:
            try:
                _post("%s/set_doppler_hints" % d_ep, hints)
                pushed += 1
            except Exception as e:
                _log("set_doppler_hints %s failed: %s" % (d_ep, e))
            # SEPARATE try: a detector running a binary without /set_nh_hint 404s here, and
            # sharing the block above would make that failure look like the DOPPLER hint
            # failing -- silently un-narrowing the search on every cycle during a rolling
            # upgrade. Rate-limited because a stale binary fails every single cycle.
            if nh_hints:
                try:
                    _post("%s/set_nh_hint" % d_ep, nh_hints)
                except Exception as e:
                    _log_rl("nhpost-%s" % d_ep,
                            "set_nh_hint %s failed (old binary?): %s" % (d_ep, e),
                            every_s=120.0)
        _log_rl("narrow",
                "narrowed search: %d hints +-%d Hz (%s) -> %d/%d detectors"
                % (len(hints), int(margin),
                   ("bias solved" if ctx.cb.ema is not None and not ctx.cb.stale
                    else "bias STALE, wide re-solve" if ctx.cb.ema is not None
                    else "pre-solve wide"),
                   pushed, len(ctx.detectors)))
