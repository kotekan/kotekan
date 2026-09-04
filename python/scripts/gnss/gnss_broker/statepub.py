"""Publishing this chain's state record for the other chains and the operator.

⚠️ CHIPS ARE NOT COMPARABLE ACROSS CHAINS. A chip is a different duration per signal (1.023
Mcps at L1 C/A, 10.23 at L5) and the value is modulo a different code length, so anything
published in chips alone is the count-where-a-time-was-meant trap that has bitten this
instrument three times. Publish the time beside it.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _now, _log_rl


def stage_publish_state(ctx):
    """S2 OBSERVER: publish the chain's state record.
        
    ⚠️ CHIPS ARE NOT COMPARABLE ACROSS CHAINS. A chip is a different duration per signal (1.023
    Mcps at L1 C/A, 10.23 at L5) and the value is modulo a different CODE_LEN, so anything
    published in chips alone is the count-where-a-time-was-meant trap that has bitten this
    instrument three times. Publish the time too."""
    if ctx.state_w is not None:
        try:
            if ctx.dr_state is not None:
                _iw = _now()
                _ir = [v[0] for v in ctx.dr_state.get("integ", {}).values()
                       if isinstance(v, (list, tuple)) and _iw - v[1] < 30.0]
                # ⚠️ CHIPS ARE NOT COMPARABLE ACROSS CHAINS -- a chip is a different
                # duration per signal (1.023 Mcps at L1 C/A, 10.23 at L5), and this
                # value is mod a different CODE_LEN too. Publishing only chips is the
                # count-where-a-time-was-meant trap that has bitten this node three
                # times; carry the rate and the derived TIME/FRACTIONAL forms so a
                # consumer never has to guess. drift_ppm is the important one: it is
                # directly comparable to l-a and to the carrier bias in ppm, and that
                # comparison is what shows dr drift to be a RESIDUAL (measured 0.000
                # -0.11 of l-a on all 8 chains) rather than a third estimator.
                _cr = float(ctx.args.chip_rate_hz) if ctx.args.chip_rate_hz else None
                _dr = ctx.dr_state.get("drift")
                _im = ctx.receiver_state.mad(_ir)
                ctx.state_w.observe(
                    "rxclock",
                    chips=ctx.dr_state.get("clk"),
                    chip_rate_hz=_cr,
                    us=((ctx.dr_state["clk"] / _cr * 1e6)
                        if (_cr and ctx.dr_state.get("clk") is not None) else None),
                    drift_chips_s=_dr,
                    drift_ppm=((_dr / _cr * 1e6) if (_cr and _dr is not None) else None),
                    n=len(_ir),
                    integ_mad_chips=_im,
                    integ_mad_us=((_im / _cr * 1e6) if (_cr and _im is not None) else None),
                    untrusted=len(ctx.dr_untrusted),
                    age_s=(round(ctx.t0 - ctx.dr_state["clk_t"], 2)
                           if ctx.dr_state.get("clk_t") else None))
            # ---- S2c: fuse this dongle's chains, PUBLISH, consume NOTHING ----
            # Sources are read from the state files, self included (self's record is up
            # to one flush old -- irrelevant for a quantity measured flat within noise
            # over 15 min, and it keeps the ordering trivial). No feedback loop exists:
            # sources_from() reads only `carrier.raw_hz` / `code.raw_ppm`, never the
            # `fused` group this writes back.
            if ctx.args.state_fuse and ctx.state_dir:
                # Reuse the cycle's cached fusion -- the same object the consumption
                # path above was handed, so the published record and the value actually
                # used can never disagree.
                _fus = ctx.fuse_cached(ctx.t0)
                if _fus:
                    ctx.state_w.observe(
                        "fused",
                        lo_ppm=_fus["lo_ppm"], se_ppm=_fus["se_ppm"],
                        lo_ppm_norej=_fus["lo_ppm_norej"],
                        n_src=_fus["n_src"], n_carrier=_fus["n_carrier"],
                        n_code=_fus["n_code"], n_rejected=_fus["n_rejected"],
                        all_outliers=_fus["all_outliers"],
                        worst_sigma=_fus["worst_sigma"], chains=_fus["chains"],
                        hz_here=_fus["lo_ppm"] * 1e-6 * ctx.args.carrier_hz)
                    # SHADOW LINE: what the fused prior says, beside what this chain is
                    # actually using. The delta is the whole S2d decision, logged every
                    # minute so a soak can be read straight out of the broker log.
                    _fhz = _fus["lo_ppm"] * 1e-6 * ctx.args.carrier_hz
                    _own = ctx.cb.ema
                    _log_rl("shadowfuse",
                            "SHADOW fused LO %+.5f ppm +-%.5f (%d src: %dc/%dd over %s"
                            "%s%s) -> %+.2f Hz here; chain uses %s; delta %s [%s]"
                            % (_fus["lo_ppm"], _fus["se_ppm"] or 0.0, _fus["n_src"],
                               _fus["n_carrier"], _fus["n_code"],
                               ",".join(_fus["chains"]),
                               "; REJECTED %d" % _fus["n_rejected"]
                               if _fus["n_rejected"] else "",
                               "; ALL-OUTLIERS (no majority -- do not trust)"
                               if _fus["all_outliers"] else "",
                               _fhz,
                               ("%+.2f Hz" % _own) if _own is not None else "UNSOLVED",
                               ("%+.2f Hz" % (_fhz - _own)) if _own is not None
                               else "n/a (this is exactly the case fusion rescues)",
                               # three honest modes: actively rescuing an unsolved
                               # chain / armed but idle (steady state, proven no-op) /
                               # pure shadow. "CONSUMED" when the chain is solved would
                               # be a lie under rescue-only semantics.
                               ("RESCUING (own unsolved)"
                                if ctx.args.state_consume and ctx.cb.ema is None
                                else "RESCUE-ARMED, idle"
                                if ctx.args.state_consume else "SHADOW")),
                            every_s=60.0)
            ctx.state_w.flush(ctx.t0)
        except Exception as e:
            # NOT a bare pass. A silent except here once swallowed a broken format
            # string in the shadow line: the line simply stopped appearing and nothing
            # said why -- and a soak read from that log would have looked like "the
            # fuser stopped running" or, worse, like clean data. Rate-limited so a
            # persistent fault names itself once a minute instead of flooding.
            _log_rl("stateobs", "receiver-state observe/flush failed: %r" % (e,),
                    every_s=60.0)
