"""3c: the fleet delay-lock stage -- poll the discriminator, then run everything hung off it.

This is the shell. It fills `ctx.dllp.fleet` (the cycle's central per-satellite state), calls
the eight `instruments` diagnostics, and finally `codeloop.stage_dll_control`, the only part
that actuates.

⚠️ THE q FLOOR IS MEASURED FROM THIS CYCLE'S OWN POPULATION, NEVER A CONSTANT. Summing across
instances tightens the noise distribution instead of raising q, so the correct bar FALLS as
instances are added: any fixed constant is right for exactly one fleet size.

⚠️ JUDGE LOCK ON q, NEVER ON sig/deep/cn0_coh. Those duty-cycle with the known-rate fold and
will read as lock loss when the tracking is fine.

@author Keith Vanderlinde
"""

import math

from gnss_broker.sky import nearest_boresight
from gnss_broker.transport import _now, _log, _log_rl, log_tag
from gnss_broker.fleet import fleet_dll, fleet_coherent
from gnss_broker.fits import q_stall_verdict, instance_stall_verdict
from gnss_broker import instruments
from gnss_broker import codeloop


def stage_fleet_dll(ctx):
    """3c: THE FLEET DELAY-LOCK LOOP -- the instrument suite, then the control loop.
        
    Polls the fleet discriminator into `_dllp.fleet` (the cycle's central per-satellite state),
    runs the eight _instr_* diagnostics hung off that same poll, and finally calls the one routine
    that actuates: _stage_dll_control.
        
    ⚠️ THE q FLOOR IS MEASURED FROM THIS CYCLE'S OWN POPULATION, NEVER A CONSTANT. Summing
    tightens the noise distribution instead of raising q, so the correct bar FALLS as instances are
    added: any fixed constant is right for exactly one fleet size."""
    if ctx.args.dll_gain > 0.0:
        # FLEET COMBINE (docs/CHORD_GNSS_SHARED_DLL.md). Sum the RAW powers across every
        # instance that reports the same window, then form ONE discriminator. Ratios do not
        # sum -- (SUM E - SUM L)/(SUM E + SUM L) is not any function of the per-instance
        # dll_disc values -- which is exactly why the combiner publishes e_pow/l_pow/p_pow.
        # #79: the effective deep-gate set = hand-listed PRNs UNION the ones the search
        # is currently detecting. `True` (--dll-deep-gate all) stays absorbing.
        ctx.dllp.deep_gate_eff = ctx.deep_gate
        if ctx.args.dll_deep_gate_from_search > 0.0 and ctx.deep_gate is not True:
            _fresh_dg = {p for p, _ts in ctx.dls.deep_gate_seen.items()
                         if ctx.t0 - _ts <= ctx.args.dll_deep_gate_search_hold_s}
            if _fresh_dg:
                ctx.dllp.deep_gate_eff = set(ctx.deep_gate or ()) | _fresh_dg
                if _fresh_dg != ctx.dg_auto_last[0]:
                    _log("DEEP GATE (auto, #79): search-admitted PRN %s at snr >= %.0f "
                         "(hold %.0f s)%s"
                         % (",".join(str(p) for p in sorted(_fresh_dg)),
                            ctx.args.dll_deep_gate_from_search,
                            ctx.args.dll_deep_gate_search_hold_s,
                            "" if not ctx.deep_gate else
                            " + hand-listed %s"
                            % ",".join(str(p) for p in sorted(ctx.deep_gate))))
                    ctx.dg_auto_last[0] = set(_fresh_dg)
        ctx.dllp.inst_hops = {}
        # THE E3 ADMISSION (--presence-admit-displaced): thresholds bundled once so the
        # polled arm, the comb arm and its shadow all judge with the SAME numbers -- two
        # copies of an admission policy is how an A/B stops measuring the powers.
        ctx.dllp.admit_disp = ({"pedestal_max": ctx.args.presence_disp_pedestal_max,
                                "off_max_chips": ctx.args.presence_disp_off_max,
                                # the OFFSET-BLIND evidence bar (see apply_presence): a
                                # detector that re-searches code phase, so it does not
                                # inherit the displacement it is being asked to judge
                                "deep_margin": ctx.args.presence_disp_deep_margin}
                               if ctx.args.presence_admit_displaced else None)
        ctx.dllp.fleet = fleet_dll(ctx.dll_combiners, ctx.dll_hop_window, ctx.args.dll_min_instances,
                          ctx.args.dll_quality_sigma, ctx.args.dll_quality_min,
                          deep_gate_prns=ctx.dllp.deep_gate_eff,
                          deep_gate_margin=ctx.args.dll_deep_gate_margin,
                          # the noise ANCHOR for the presence floor (#49): without it
                          # the bar is built from the tracked population and becomes a
                          # peer competition. See the note in fleet_dll.
                          probe_prns=ctx.probe_set,
                          # #70: collect the per-instance newest hop from the poll we are
                          # already making. No extra HTTP -- fleet_dll parses pow_hop for
                          # the currency check and then aggregates the axis away.
                          src_hops=ctx.dllp.inst_hops,
                          admit_displaced=ctx.dllp.admit_disp)
        # TASK #63: THE SAME DISCRIMINATOR, FORMED HERE FROM THE UN-SUMMED COMB. The powers
        # above were built by each tracker summing across its own channels -- "the one
        # combine the broker can never undo" -- and everything derived from them inherits
        # that. combdll rebuilds them from the per-channel Early/Prompt/Late the transport
        # ships, in the one place that can see the whole band.
        #
        # THE POLLED ARM IS STILL COMPUTED ABOVE, ON PURPOSE, AND FOR TWO REASONS: the deep
        # statistics are not in the comb (they come from the combiner's fold, so they are
        # handed across as coh_rows), and running both makes the swap self-monitoring --
        # the log line below is the same paired comparison the offline A/B makes, on the
        # loop's own cycles. Any cycle where the gather has nothing for this chain simply
        # keeps the polled fleet.
        # ⚠️ AN UNANCHORED CHAIN MUST BE LOUD. --presence-require-probes refuses the verdict
        # rather than guessing it, and a refusal that only shows up as "0 present" is
        # indistinguishable from a dead sky -- which is exactly the ambiguity this whole
        # afternoon was spent removing. Say WHY, and say what to do about it.
        _fl0 = next(iter((ctx.dllp.fleet or {}).values()), None)
        if _fl0 is not None and _fl0.get("present_gate") == "UNANCHORED":
            _log_rl("unanchored",
                    "⚠️ PRESENCE UNANCHORED: only %d probe(s) reporting (need 3). NOBODY is "
                    "admitted and nothing is trimmed -- deliberately, because the alternative "
                    "is a floor built from the SATELLITES, which passes about half of them by "
                    "construction. Fix the PROBE SUPPLY: check that the selected probe PRNs "
                    "have slots on the nodes (--probe-require-slot), not this gate."
                    % _fl0.get("n_probe_q", 0), every_s=60.0)
        instruments.instr_tap_walk(ctx)
        # PROMPT HOLD for the NEXT cycle's lock gate (fold-independent, see
        # --lock-prompt-hold). Mutated in place, never rebound: the gate closes over it.
        ctx.hold.prev.clear()
        ctx.hold.q.clear()
        for _p, _fl in (ctx.dllp.fleet or {}).items():
            _pm = _fl.get("p_med")
            if _pm:
                ctx.hold.prev[_p] = _fl["p_pow"] / _pm
            # ...and the FLEET DISCRIMINATOR QUALITY, from the same dict, for the same
            # next-cycle gate. See --lock-q: q is the metric this project judges lock on
            # everywhere EXCEPT, until now, the one decision that can destroy a lock.
            if _fl.get("q") is not None:
                ctx.hold.q[_p] = float(_fl["q"])
        # CROSS-NODE COHERENT COMBINE. Separate from the DLL on purpose: the DLL sums
        # POWERS (phase-free, which is what makes it cheap) while this removes the common
        # per-record PHASE, and the two answer different questions off the same endpoints.
        # It touches NO loop -- purely an observable, published for the viewer and the beam
        # map -- so a fault here can degrade what is displayed but can never move the code
        # or carrier loops.
        ctx.dllp.fcoh = {}
        # TASK #59. The gather feed, when it is asked for AND actually has this chain's
        # windows. `source=None` falls straight through to the REST poll below, so a gather
        # that is down, restarting, or simply not carrying this chain costs a rate-limited
        # log line and nothing else -- the two feeds run side by side until the new one has
        # been shown ON SKY to be at least as good, and neither is load-bearing for the
        # other.
        _tsrc = None
        if ctx.telem_client is not None:
            # THE ALIGNMENT CHECK, PUBLISHED. `spread` is max-min of the instances' newest
            # window index: 0 or 1 is the transport working. Logged whether or not anything
            # consumes the feed, because the entire argument for this transport is that
            # misalignment becomes visible immediately instead of surfacing weeks later as
            # a physics anomaly (#46, #52, #53 all did).
            _st = ctx.telem_client.stats()
            _cs = _st["chains"].get(ctx.telem_chain)
            if not _cs:
                _msg = "chain %s: nothing yet" % ctx.telem_chain
            elif not _cs.get("live"):
                _msg = ("chain %s: ALL %d instances stale (%s)"
                        % (ctx.telem_chain, _cs["instances"], ",".join(_cs["stale"])))
            else:
                # SPREAD IS OVER LIVE INSTANCES ONLY; the stale ones are NAMED. A stopped
                # instance keeps its last window forever, so folding it into the spread
                # turns every instance death into a four-digit alarm about alignment --
                # which is the one number here that must stay trustworthy.
                _msg = ("chain %s: %d live, win %d..%d spread %d%s"
                        % (ctx.telem_chain, _cs["live"], _cs["win_min"], _cs["win_max"],
                           _cs["spread"],
                           (" | STALE %s" % ",".join(_cs["stale"])) if _cs["stale"] else ""))
            _log_rl("telem-stat",
                    "TELEM %s frames %d gaps %d bad %d | %s"
                    % ("up" if _st["connected"] else "DOWN", _st["frames"], _st["gaps"],
                       _st["bad"], _msg),
                    every_s=30.0)
        if ctx.args.telem_coherent and ctx.telem_client is not None:
            try:
                _tsrc = ctx.telem_client.coherent_source(
                    ctx.telem_chain, prns=set(ctx.seeds) or None, n_win=ctx.args.telem_windows, lag=1)
                if not _tsrc[0]:
                    _tsrc = None
                    _log_rl("telem-empty",
                            "telem: no windows for chain %r yet -- falling back to "
                            "/get_records (gather stats: %s)"
                            % (ctx.telem_chain, ctx.telem_client.stats()))
            except Exception as e:
                _tsrc = None
                _log_rl("telem-src", "telem: source failed (%s); using /get_records" % e)
        if ctx.args.fleet_coherent:
            try:
                ctx.dllp.fcoh = fleet_coherent(ctx.dll_combiners, ctx.args.coh_min_instances,
                                      ctx.args.coh_min_records, prns=set(ctx.seeds) or None,
                                      log=None, floor_margin=ctx.args.coh_floor_margin,
                                      seed=int(_now()),
                                      # lets it fit the record-stream carrier rate off
                                      # the records it already fetched (#33 coarse feed).
                                      # ⚠️ HOPS, NOT RECORDS. get_records' first tuple
                                      # element is a HOP COUNT -- phaseslope.py divides
                                      # it by 195312.5, not by the record rate. Passing
                                      # hops_per_sec/2048 made the time axis 2048x too
                                      # long, so every fitted rate came out 2048x too
                                      # SMALL: +-0.005 Hz where the fold read +-10,
                                      # ratio 1907-2140 across satellites.
                                      hop_rate_hz=ctx.args.hops_per_sec,
                                      # #59: when set, the poll is skipped entirely and
                                      # this identical estimator runs on the gathered
                                      # records instead.
                                      source=_tsrc)
            except Exception as e:
                _log_rl("fleet-coh", "fleet coherent: skipped this cycle (%s)" % e)
        # PATH B, same estimator, separate population. Reported side by side rather than
        # merged: blending the two streams would make the very comparison this exists to
        # support impossible, and their per-record noise is NOT independent (both despread
        # the same antenna voltages), so a merged sum would not buy the sqrt(2) it appears
        # to. Publishes nothing and touches no loop.
        ctx.dllp.fcoh_n2 = {}
        if ctx.args.fleet_coherent and ctx.n2_combiners:
            try:
                ctx.dllp.fcoh_n2 = fleet_coherent(ctx.n2_combiners, ctx.args.coh_min_instances,
                                         ctx.args.coh_min_records, prns=set(ctx.seeds) or None,
                                         log=None, floor_margin=ctx.args.coh_floor_margin,
                                         seed=int(_now()))
            except Exception as e:
                _log_rl("fleet-coh-n2", "fleet coherent (path B): skipped (%s)" % e)
        instruments.instr_coherent_rows(ctx)
        # FLEET PHASE-SLOPE DELAY FIT (task #32, docs/CHORD_JOINT_TRACKING.md P1).
        # MEASUREMENT ONLY: touches no seed and no loop -- it is logged and published so
        # it can be judged against the disc, the E/L asymmetry and GPS's search-measured
        # code phase BEFORE anything consumes it (the #30 rule: measure the statistic
        # before the loop). Gated on its OWN flag, which is what keeps every recorded
        # transcript replaying byte-identically: replay is strict-ordered, an
        # unrecorded GET is a TRANSCRIPT DIVERGENCE, and old transcripts' argv does not
        # carry --spectrum-endpoints, so replays never issue the new polls.
        ctx.dllp.spec_fit = {}
        instruments.instr_spectrum_fit(ctx)
        # THE SERVED C/N0 (task #57): per-record prompt power off the gather feed,
        # q-gated, debiased against the noise probes. Fits NOTHING -- the deep fold's
        # per-integration rate re-search is a fit on something the tracking loop already
        # fixed, and its ~20 dB of paired self-scatter is why cn0_coh_db cannot be the
        # radiometry. Measurement-only: touches no loop, issues no polls (the telemetry
        # client is a push stream, so recorded transcripts replay byte-identically).
        # Needs the probes as its noise anchor; without them it publishes nothing rather
        # than falling back to the peer competition (#49's lesson).
        # Throttle (--estimator-every-s): both telemetry-walk estimators run together,
        # at most this often; the last values keep being served in between. Defined
        # HERE because this is the FIRST of the two blocks in cycle order.
        ctx.dllp.run_est = (ctx.telem_client is not None and ctx.probe_set
                    and _now() >= ctx.est_next[0])
        if ctx.dllp.run_est:
            ctx.est_next[0] = _now() + ctx.args.estimator_every_s
        # ⚠️ THE THROTTLE EXISTS FOR THE *WALK*, NOT FOR THE ESTIMATOR. Its whole
        # justification (see --estimator-every-s) is that these are "pure-Python walks over
        # ~1500 record decodes each", which at every-cycle cadence across five chains ate
        # ~75% of the interpreter and starved the telemetry reader. With --comb-taps-cpp
        # armed, PROMPT-CN0 no longer walks anything: it fetches an already-reduced series
        # from the gather. So the reason to throttle IT is gone, and throttling it now only
        # costs the served C/N0 its freshness -- the rows keep serving a value up to
        # --estimator-every-s old for no saving at all.
        #
        # KCOH still walks and stays throttled. They were gated together because they had
        # the same cost; they no longer do, so they no longer share a gate.
        ctx.dllp.run_pcn0 = ctx.dllp.run_est or (ctx.args.comb_taps_cpp >= 2 and ctx.args.fleet_trim_url
                                 and ctx.telem_client is not None and ctx.probe_set)
        ctx.dllp.pcn0 = ctx.est_last["pcn0"]
        instruments.instr_prompt_cn0(ctx)
        # THE KNOWN-RATE COHERENT C/N0 (task #57 step 3): the ~1 s fold with the rate
        # INJECTED from the PREVIOUS cycle's record-stream fit (_kcoh_rates, updated
        # below AFTER the fold so this integration never consumes a rate estimated
        # from itself). No search, no q gate -- a fixed-rate fold over noise is noise,
        # which is why this cannot fire on noise the way the deep fold did, and why it
        # reaches the deep-sidelobe satellites a per-record gate never passes.
        # Measurement-only; rides the same telemetry the comb DLL uses.
        # Same throttle gate as PROMPT-CN0 above (_run_est, set there -- the first of
        # the two telemetry-walk blocks in cycle order).
        ctx.dllp.kcoh = ctx.est_last["kcoh"]
        instruments.instr_kcoh(ctx)
        # NOW the rates for the NEXT cycle, from THIS cycle's record-stream fit.
        for _p, _fc2 in (ctx.dllp.fcoh or {}).items():
            _r2 = _fc2.get("rate_hz")
            if _r2 is not None:
                ctx.kcoh_rates[_p] = float(_r2)
        # THE PER-ELEMENT COMPLEX GAIN (task #57 step 2): amplitude AND phase per
        # antenna, per instance -- the beam/peel coefficients. Assembled from the
        # combiners' /get_elements (raw leave-one-out cross-products accumulated per
        # record node-side, where the element axis lives), significance-anchored on the
        # noise probes per (instance, element). Measurement-only; touches no loop.
        # ── #8: RF-PATH HEALTH, POLLED ──────────────────────────────────────────────
        # Clip fraction and per-band power from each GPU's voltage tap. Measurement
        # only: it steers nothing and gates nothing, exactly like the element poll below.
        #
        # ⚠️ THIS IS THE ONE NUMBER THAT SEPARATES "LOUD" FROM "RAILED", and we have
        # never had it. The fleet's amplitudes swing 5-10x an hour (#56) with the root
        # upstream of both tracking and the combine, and on 08-18 something lit up the
        # band hard enough to take chains down for hours -- with no way to tell whether
        # the front end saturated or merely saw a large linear signal. Those two have
        # different fixes and we could not distinguish them.
        #
        # Grouped into LOBES by channel contiguity (rf_lobes), because the tap's channel
        # list is the union of every chain's covering set and therefore arrives as one
        # run per band. A band-selective source is only diagnosable against a band that
        # was quiet in the SAME sample.
        instruments.instr_rf_stats(ctx)

        # ⚠️ NEVER THROTTLE IN REPLAY. A transcript is an ordered recording of every GET,
        # checked by URL as it is consumed, so making FEWER calls than the recording
        # desynchronises the stream: the next endpoint reads the previous one's response
        # and `_get` catches the divergence per call, so the replay limps on producing
        # garbage instead of stopping. Found the hard way -- this moved the holds digest
        # and it looked like the (unrelated) _solve refactor until the two arms were
        # diffed and BOTH gave the same moved digest. "TRANSCRIPT DIVERGENCE at get #116"
        # was in the replay's own stderr the whole time.
        #
        # The general rule: broker_equiv can gate what a call COMPUTES, never how OFTEN it
        # is made. Any cadence change is live-only and needs a live measurement.
        instruments.instr_element_poll(ctx)
        # #83 2(d): the served innovation -- freshest value + 10-minute p95 per PRN.
        # The statistic is cut by TIME here at read, not by count at write: a PRN
        # detected once in ten minutes reports that one sample, and a PRN that set
        # 20 minutes ago stops being served instead of fossilizing its last window.
        ctx.dllp.innov_pub = {}
        for _p, _ih in list(ctx.innov_hist.items()):
            if _now() - _ih[-1][0] > 1200.0:
                del ctx.innov_hist[_p]
                continue
            _win = [(tt, vv) for tt, vv in _ih if _now() - tt <= 600.0]
            if not _win:
                continue
            _av = sorted(abs(vv) for _, vv in _win)
            ctx.dllp.innov_pub[_p] = {
                "innov_chips": _win[-1][1],
                "innov_age_s": _now() - _win[-1][0],
                "innov_p95_10m": _av[max(0, math.ceil(0.95 * len(_av)) - 1)],
                "innov_n_10m": len(_win),
            }
        # #83 P3-3a: the MODEL innovation rides the same rows -- minnov_* keys. A PRN
        # can carry either or both (INNOV needs a standing seed, MINNOV an established
        # joint row); absent keys mean "not measurable", never zero.
        for _p, _mh in list(ctx.minnov_hist.items()):
            if _now() - _mh[-1][0] > 1200.0:
                del ctx.minnov_hist[_p]
                continue
            _win = [(tt, vv) for tt, vv in _mh if _now() - tt <= 600.0]
            if not _win:
                continue
            _av = sorted(abs(vv) for _, vv in _win)
            ctx.dllp.innov_pub.setdefault(_p, {}).update({
                "minnov_chips": _win[-1][1],
                "minnov_p95_10m": _av[max(0, math.ceil(0.95 * len(_av)) - 1)],
                "minnov_n_10m": len(_win)})
        # ── #83 P3-3b: THE FLIP DECISION (see --model-primacy-max) ──
        # One writer for mp_flipped, once per cycle, from the MEASURED p95s built
        # above. ENTER: p95 < gate with enough samples; the best-p95 eligible PRNs
        # fill the cap, the rest are the in-poll controls. EXIT: p95 beyond the
        # hysteresis bound, or the referee starved (no detection while flipped for
        # --model-primacy-starve-s). Every transition is one loud line -- a flip
        # whose firing cannot be seen is how gates fail here.
        instruments.instr_model_primacy(ctx)
        if ctx.dllp.innov_pub:
            _log_rl("innov",
                    "INNOV %s: %s"
                    % (log_tag() or ctx.args.signal,
                       " ".join("%d:%+.2f(p95 %.2f, n%d)"
                                % (_p, v["innov_chips"], v["innov_p95_10m"],
                                   v["innov_n_10m"])
                                for _p, v in sorted(ctx.dllp.innov_pub.items())
                                if "innov_chips" in v)),
                    every_s=60.0)
            _mv = ["%d:%+.2f(p95 %.2f, n%d)"
                   % (_p, v["minnov_chips"], v["minnov_p95_10m"], v["minnov_n_10m"])
                   for _p, v in sorted(ctx.dllp.innov_pub.items()) if "minnov_chips" in v]
            if _mv:
                _log_rl("minnov",
                        "MINNOV %s (model vs sky, flip-gate statistic): %s"
                        % (log_tag() or ctx.args.signal, " ".join(_mv)),
                        every_s=60.0)
        if ctx.publisher is not None:
            # Published BEFORE the trim update so the row shows the state the loop acted
            # on, not the state after it acted -- otherwise a reader can never see the
            # input that produced a given correction.
            ctx.publisher.update(ctx.dllp.fleet, ctx.seeds, ctx.dls.trim, len(ctx.dll_combiners), ctx.last_dets, ctx.dllp.fcoh,
                             pcn0=ctx.dllp.pcn0, kcoh=ctx.dllp.kcoh, innov=ctx.dllp.innov_pub,
                             cpp_trim={_p: (_r.get("trim_chips") or 0.0)
                                       for _p, _r in ctx.dls.readback.items()})
        ctx.dllp.report = []
        codeloop.stage_dll_control(ctx)
        if ctx.dllp.report:
            _log("DLL: " + "; ".join(ctx.dllp.report))
        # ── D0: THE POPULATION-HONEST q SERIES ────────────────────────────────────────
        # Recorded for EVERY seeded satellite, present or not -- the line above lists only
        # the ones that passed the presence gate, so a satellite whose q craters leaves it.
        # A statistic over that line measures survivors, and survivors always look healthy.
        ctx.qpop.note_cycle(ctx.t0, set(ctx.seeds), ctx.dllp.fleet)
        _tag = log_tag() or ctx.args.signal
        _qline = ctx.qpop.line(_tag)
        if _qline:
            _log_rl("qpop", _qline, every_s=120.0)

        # ── D1: the chain-wide brownout, as a labelled episode ────────────────────────
        # Promoted out of #90's admission gate, where it silently suppressed fires and
        # nothing downstream could tell a window had contained one.
        _npres = sum(1 for _f in (ctx.dllp.fleet or {}).values()
                     if isinstance(_f, dict) and _f.get("present"))
        _bmsg = ctx.brown.note_cycle(ctx.t0, _npres)
        if _bmsg:
            _log("%s: %s" % (_tag, _bmsg))

        # ── THE BORESIGHT-TRANSIT VETO (shared by D2 and D3) ──────────────────────────
        # Pooled over EVERY constellation -- ctx.dr_pd carries all three, and the quantiser is
        # shared, so a per-chain answer would be worse than none (it would clean the chain
        # carrying the bright satellite and leave the rest looking like clean controls).
        # Transits recur on the SIDEREAL day, so this is predictable rather than bad luck.
        _near = nearest_boresight(ctx.dr_pd) if ctx.args.detector_transit_veto_deg > 0.0 else None
        _in_transit = bool(_near and _near[0] < ctx.args.detector_transit_veto_deg)
        if _in_transit:
            _log_rl("transit-veto",
                    "%s: BORESIGHT TRANSIT -- %s%d is %.1f deg off boresight; D2/D3 suppressed "
                    "(the quantiser rails and satellites drop across every chain at once)"
                    % (_tag, _near[1][0], _near[1][1], _near[0]),
                    every_s=120.0)

        # ── D2: the deep latch, UNARMED ───────────────────────────────────────────────
        # #90's four armed flights produced zero genuine targets, so the base rate is the
        # missing number. This measures it at no risk; it actuates nothing.
        for _lp, _labs, _lq in ctx.latch.scan(
                ctx.t0, ctx.qpop, ctx.brown.active(),
                uptime_s=ctx.t0 - ctx.broker_t0,
                # The plant's own convergence window, not the process's -- a broker that has
                # been up for hours is still looking at a sky that just came back.
                recovering=ctx.brown.recovering(ctx.t0, ctx.latch.startup_hold_s),
                in_transit=_in_transit):
            _log("%s: LATCH PRN %d absent %.0f s after q %.2f -- #90 v3 would have fired "
                 "here (detector only, nothing armed)" % (_tag, _lp, _labs, _lq))

        # ── D3: the handover sawtooth ─────────────────────────────────────────────────
        # Fed the HANDOVER-CORRECTED trim: without that subtraction a successful #92
        # handover -- the cure -- reads as the disease it was applied to.
        for _sp, _sr in (ctx.dls.readback or {}).items():
            if not isinstance(_sr, dict) or _sr.get("trim_chips") is None:
                continue
            if not _sr.get("armed"):
                ctx.saw.drop(_sp)     # a released trim decays by the LEAK, not a wipe
                continue
            _sq = ctx.qpop.summary(_sp)
            _sbt = ctx.birth_steps.get(_sp)
            _smsg = ctx.saw.note(ctx.t0, _sp,
                                 ctx.handover.corrected(_sp, float(_sr["trim_chips"])),
                                 browned_out=ctx.brown.active(),
                                 uptime_s=ctx.t0 - ctx.broker_t0,
                                 rebase_age_s=(ctx.t0 - _sbt) if _sbt is not None else None,
                                 present_frac=_sq[4] if _sq else None,
                                 q_mean=_sq[2] if _sq else None,
                                 in_transit=_in_transit)
            if _smsg:
                _log("%s: %s" % (_tag, _smsg))
        # CODE-DERIVED CARRIER ERROR, logged only -- not applied yet.
        #
        # Carrier and code are locked in ratio: a Doppler error dF drifts the code at
        # dF * f_chip/f_carrier (validated offline, STATE 7.2: measured 0.01476 chips/s
        # against 0.01476 predicted). Inverting, the fitted cp slope gives the carrier error
        # directly, at f_carrier/f_chip = 115.03 Hz per chip/s.
        #
        # This matters because carrier_hz_resid is signal-free (2026-08-04: |resid| median
        # 0.519 Hz on satellites with signal, 0.492 Hz on satellites without), so the carrier
        # loop was integrating noise and is now off. The code side, by contrast, is strong --
        # sustained q ~ 3.2 and an 8-point cp fit.
        #
        # LOGGED, NOT APPLIED. Three loops today were found eating a statistic that did not
        # measure what its name said; this one gets compared against the estimator it would
        # replace before it is allowed to move anything.
        if ctx.args.carrier_from_code:
            _k = ctx.args.carrier_hz / ctx.args.chip_rate_hz
            _rows = []
            for _p in sorted(ctx.cpt.fit_slope):
                _rec = ctx.status.get(_p, {})
                _meas = float(_rec.get("carrier_hz_resid", 0.0))
                _sig = ctx.sig_of(_rec)
                if _sig < ctx.args.lock_snr:
                    continue
                _rows.append("PRN %d code->%+.2f Hz meas %+.2f Hz (sig %.1f)"
                             % (_p, ctx.cpt.fit_slope[_p] * _k, _meas, _sig))
            if _rows:
                _log_rl("carfromcode", "CARRIER-FROM-CODE (shadow): " + "; ".join(_rows[:6]),
                        every_s=30.0)
        if ctx.dop_rate_rejected:
            _log("dop-rate: %d fit(s) REJECTED against the model (kept the model): %s"
                 % (len(ctx.dop_rate_rejected),
                    ", ".join("PRN %d fit %+.3f vs model %+.3f" % (k, v[0], v[1])
                              for k, v in sorted(ctx.dop_rate_rejected.items())[:5])))
        # The same report for the CODE rate (#96). Separate line, not folded into the one
        # above: these reject against the POOLED CLOCK rather than an orbit model, and a
        # reader who cannot tell which reference rejected a fit cannot act on it.
        if ctx.cp_rate_rejected:
            _log("cp-rate: %d fit(s) REJECTED against the pooled clock (kept the clock "
                 "rate, kept the fitted position): %s"
                 % (len(ctx.cp_rate_rejected),
                    ", ".join("PRN %d fit %+.3f vs clock %+.3f chips/s" % (k, v[0], v[1])
                              for k, v in sorted(ctx.cp_rate_rejected.items())[:5])))
        _absent = sorted(p for p in ctx.seeds
                         if ctx.seeds[p].get("doppler_rate_hz_s") is None)
        if _absent:
            # No carrier extrapolation AND no quadratic code term for these sats.
            _log("dop-rate: %d seeded PRN(s) have NO doppler rate at all: %s"
                 % (len(_absent), _absent[:8]))
        if ctx.dop_rate_fitted:
            _log_rl("doprate", "doppler-rate FIT seeded on %d sat(s): %s"
                    % (len(ctx.dop_rate_fitted),
                       "; ".join("PRN %d %+.4f Hz/s" % (k, v)
                                 for k, v in sorted(ctx.dop_rate_fitted.items())[:5])),
                    every_s=60.0)
        # The BAR, every cycle it is measured. A threshold on a noisy statistic that is
        # never printed is a threshold nobody can audit -- and this one legitimately moves
        # with the fleet size, so a reader has to be able to see where it went.
        if ctx.dllp.fleet:
            any_fl = next(iter(ctx.dllp.fleet.values()))
            _log_rl("dll-floor",
                    # WHICH ARM produced these numbers. Since #63 this line can describe
                    # either the polled powers or the ones formed here from the comb, and
                    # a floor with no provenance is exactly the kind of number that gets
                    # compared across a switch without anyone noticing it changed source.
                    "fleet DLL [%s]: %d PRN(s) over %d combiner(s), %d present, "
                    "q floor %.2f%s"
                    % (any_fl.get("src") or "polled", len(ctx.dllp.fleet), len(ctx.dll_combiners),
                       sum(1 for v in ctx.dllp.fleet.values() if v["present"]), any_fl["q_floor"],
                       "" if any_fl["q_med"] is None
                       else " (noise median %.2f, sigma %.3f)"
                            % (any_fl["q_med"], any_fl["q_sigma"])))
        # ── THE INSTANCE LIVENESS GUARD (#70, 2026-08-18) ──────────────────────────
        # WHY THIS EXISTS, and why the guard we already had could not do it. On 08-18's
        # full-fleet restart FOUR instances came up wedged -- cx42/gnss0, cx43/gnss0,
        # cx44/gnss1, cx51/gnss0 -- each with its DPDK capture window frozen and the
        # ENTIRE 195,313 pkt/s stream being dropped, and each serving plausible,
        # well-formed rows to every poll for as long as it was left alone. An earlier
        # instance of the same fault ran on cx19 for 25 HOURS and 18.7 billion dropped
        # packets before a human noticed.
        #
        # ⚠️ "ALL 12 RESPOND" IS NOT "ALL 12 ARE ALIVE". That is the whole trap, and it
        # is why this keys on a COUNTER and never on reachability: healthy is ~5.9M
        # hops per 30 s, wedged is exactly 0, and both answer 200.
        #
        # ⚠️ AND IT IS NOT --fe-axis-stale-s, which watches the MAXIMUM hop over
        # instances. That question ("has the time base frozen?") is real and that guard
        # caught the cx19 collapse -- but eleven healthy instances keep the maximum
        # climbing, so it was correctly silent through all four wedges. The axis it
        # cannot resolve is per-instance, and this is that axis. Both stay.
        #
        # FREE: the hop comes from the fleet DLL poll above, which already parses
        # pow_hop for its currency check and then aggregates the per-instance axis away.
        # The decision is a pure function in fits.py (test_instance_stall.py) with a
        # CONTROL CLAUSE -- if most of the fleet is also standing still this is global
        # (a paused F-engine, a replay, a clock step) and accusing an instance would
        # point the next hour in the wrong direction, so it says nothing instead.
        # ── LINK 1 OF THE WALKOFF CHAIN, MEASURED PER INSTANCE ────────────────────
        # ⚠️ THE POPULATION IS THE POINT, AND I GOT IT WRONG ONCE. The obvious place for
        # this is beside `_fh = max(pow_hop)` in the status block -- but that `status` is
        # `{prn: row}` from ONE combiner, and every PRN in a poll carries the SAME
        # pow_hop, so it reports spread 0.00 s across 32 rows and says nothing about the
        # fleet ([[identical-numbers-are-not-agreement]]). The axis lag that link 1 is
        # about is ACROSS INSTANCES: on 2026-08-23, gps_l5/gal_e5a read -18.0..-19.3 s
        # while bds_b2a read -7.8 s at the same instant. `_inst_hops_now` is the right
        # population -- the newest hop per combiner, from the poll fleet_dll already makes.
        #
        # Sign convention matches _dax below: NEGATIVE = that instance lags the wall.
        # POSITIVE = a hop the F-engine has not reached, i.e. link 2's FUTURE HOP, which
        # is impossible for a processed record and names the instance serving it.
        if ctx.utc0_sample0 and ctx.dllp.inst_hops:
            _w = _now()
            _ia = sorted(((ctx.utc0_sample0 + float(h) / ctx.args.hops_per_sec) - _w, str(k))
                         for k, h in ctx.dllp.inst_hops.items() if h)
            if _ia:
                _fut = [k for d, k in _ia if d > 0.5]
                _log_rl("axis-inst",
                        "AXIS INST: n=%d  lag median %+.2f s  worst %+.2f s (%s)  "
                        "freshest %+.2f s (%s)  spread %.2f s%s"
                        % (len(_ia), _ia[len(_ia) // 2][0], _ia[0][0], _ia[0][1],
                           _ia[-1][0], _ia[-1][1], _ia[-1][0] - _ia[0][0],
                           "" if not _fut else
                           "  *** %d FUTURE instance(s) >0.5 s AHEAD: %s"
                           % (len(_fut), ",".join(sorted(_fut)[:4]))),
                        every_s=30.0)
        # ⚠️⚠️ dr_state CAN BE None, AND DEREFERENCING IT HERE KILLED ALL FIVE CHAINS
        # (2026-08-28 00:11). It starts None and is only assigned when the dead-reckon block
        # succeeds -- which is inside a try/except that DISABLES dead reckoning and logs
        # "dead-reckon unavailable", leaving None behind. This line then raised
        # AttributeError inside the cycle, which is fatal to a chain rather than to a stage.
        #
        # A chain that has lost dead reckoning should run DEGRADED, not die: everything else
        # in this stage -- the trims, the presence verdicts, the axis checks -- is still
        # valid. Skipping one stall check is a smaller loss than the whole chain, and the
        # difference decided a fleet-wide outage.
        #
        # ⚠️ SAY SO RATHER THAN SKIPPING QUIETLY. A silent skip here would make a chain with
        # no dead reckoning indistinguishable from one whose instances are simply healthy.
        if ctx.args.instance_stall_s > 0 and ctx.dllp.inst_hops and ctx.dr_state is None:
            _log_rl("inststall-nodr",
                    "instance-stall check SKIPPED: dr_state is None, so dead reckoning never "
                    "initialised on this chain (look for 'dead-reckon unavailable' at "
                    "startup). The chain is running DEGRADED -- seeds are not being "
                    "dead-reckoned -- which is a bigger problem than the skipped check.",
                    every_s=120.0)
        elif ctx.args.instance_stall_s > 0 and ctx.dllp.inst_hops:
            _ih, _stalled = instance_stall_verdict(
                ctx.dr_state.get("inst_hops", {}), ctx.dllp.inst_hops, ctx.t0,
                ctx.args.instance_stall_s)
            ctx.dr_state["inst_hops"] = _ih
            if _stalled:
                _log_rl("inststall",
                        "⚠️ INSTANCE STALLED: %d of %d serving but NOT ADVANCING -- %s. "
                        "These answer 200 with plausible rows; their pow_hop has not "
                        "moved (healthy is ~5.9M hops/30 s). The usual cause is a frozen "
                        "DPDK capture window dropping the whole stream, which no amount "
                        "of waiting clears -- check the node log for RESYNC lines, then "
                        "restart that node. Bandwidth is degraded fleet-wide until then."
                        % (len(_stalled), len(_ih),
                           ", ".join("%s stuck at hop %d for %.0f s" % (u, h, dt_)
                                     for u, h, dt_ in _stalled[:4])),
                        every_s=300.0)

        # ── THE q STALL GUARD (#70/#87, 2026-08-18) ────────────────────────────────
        # WHY THIS EXISTS: on 08-18 three chains sat in a degraded steady state for
        # 3.5 h after the RFI outage railed their C++ trims -- gal_e5b at q_med 1.09
        # against its own 2.09 baseline -- and NOTHING said so. The numbers were in
        # this very log the whole time; it took a same-time-of-day comparison, run by
        # hand because someone asked for a status check, to see it. A degradation
        # that only a human comparison can find is a degradation that runs for hours.
        #
        # WHAT IT WATCHES: this chain's own q duty (fraction of judged sats at
        # q >= --q-stall-bar) over a trailing window, against the BEST duty this chain
        # has reached in this process's lifetime. Self-referential on purpose: chains
        # differ 4x in duty by construction (#49), so a fleet-common bar would either
        # cry wolf on bds_b2b or never fire on gps_l5. The baseline only RISES, so a
        # chain that degrades cannot quietly redefine "normal" downward -- which is
        # exactly how the 3.5 h went unnoticed.
        #
        # ⚠️ IT IS A NOTICE, NOT A CONTROL. It changes nothing and gates nothing: a
        # guard that acts on a statistic this noisy would be a new failure mode, and
        # the honest recovery (a NODE restart, which clears the C++ trim state a
        # broker restart cannot) is not the broker's to perform.
        if ctx.args.q_stall_window > 0 and ctx.dllp.fleet:
            _qs = [v["q"] for v in ctx.dllp.fleet.values() if v.get("q") is not None]
            if _qs:
                _duty = sum(1 for q in _qs if q >= ctx.args.q_stall_bar) / float(len(_qs))
                _qh = ctx.dr_state.setdefault("q_hist", [])
                _qh.append((ctx.t0, _duty))
                del _qh[:max(0, len(_qh) - 4000)]
                # The decision itself is a PURE function in fits.py so it can be
                # tested against a constructed collapse (test_q_stall.py) -- the
                # on-sky fixtures run ~11 cycles at a duty that never falls, so a
                # replay cannot distinguish "did not fire" from "cannot fire".
                _qb, _qv = q_stall_verdict(_qh, ctx.t0, ctx.args.q_stall_window,
                                           ctx.args.q_stall_frac, ctx.args.q_stall_min_best,
                                           ctx.dr_state.get("q_duty_best"))
                ctx.dr_state["q_duty_best"] = _qb
                if _qv is not None:
                    _log_rl("qstall",
                            "⚠️ q STALL: duty %.2f over the last %.0f s vs %.2f best "
                            "this session (%.0f%% of it, bar q>=%.1f, %d sat(s)). "
                            "This chain has been degraded, not absent -- the usual "
                            "cause is railed C++ trim state, which a BROKER restart "
                            "does NOT clear (#87). Judge against the same time of day, "
                            "then a NODE restart."
                            % (_qv[0], ctx.args.q_stall_window, _qv[1], 100.0 * _qv[2],
                               ctx.args.q_stall_bar, len(_qs)),
                            every_s=ctx.args.q_stall_notice_s)
        for k in list(ctx.dls.trim):
            if k not in ctx.seeds:
                del ctx.dls.trim[k]
                ctx.dls.last_hop.pop(k, None)
