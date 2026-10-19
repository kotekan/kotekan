"""3c: the fleet delay-lock stage -- poll the discriminator, then run everything hung off it.

This is the shell. It fills `ctx.dllp.fleet` (the cycle's central per-satellite state), calls
the nine `instruments` diagnostics, then `codeloop.stage_dll_control`, the only part that
actuates, and after it the D0/D1/D3 detectors and the stall checks.

The q floor is measured from this cycle's noise probes, never a constant. Summing across
instances tightens the noise distribution instead of raising q, so the correct bar falls as
instances are added: any fixed constant is right for exactly one fleet size.

Judge lock on q, never on sig/deep/cn0_coh. Those duty-cycle with the known-rate fold and
read as lock loss while the tracking is fine.

@author Keith Vanderlinde
"""

import math

from gnss_broker.sky import nearest_boresight
from gnss_broker.transport import _now, _log, _log_rl, log_tag
from gnss_broker.fleet import fleet_dll, fleet_coherent
from gnss_broker.fits import q_stall_verdict, instance_stall_verdict
from gnss_broker import instruments
from gnss_broker import codeloop
from gnss_broker import trkresid
from gnss_broker import fleetadr


def deep_gate_auto(ctx):
    """The effective deep-gate set, into `ctx.dllp.deep_gate_eff`: the hand-listed PRNs
    (--dll-deep-gate) plus those the search detected at snr >= --dll-deep-gate-from-search
    within --dll-deep-gate-search-hold-s. 'all' (True) already covers every PRN.

    The DEEP GATE (auto) line prints when the search-admitted set changes, emptying included,
    so a set that empties and comes back is logged again."""
    ctx.dllp.deep_gate_eff = ctx.deep_gate
    if ctx.args.dll_deep_gate_from_search > 0.0 and ctx.deep_gate is not True:
        _fresh_dg = {
            p
            for p, _ts in ctx.dls.deep_gate_seen.items()
            if ctx.t0 - _ts <= ctx.args.dll_deep_gate_search_hold_s
        }
        if _fresh_dg:
            ctx.dllp.deep_gate_eff = set(ctx.deep_gate or ()) | _fresh_dg
            if _fresh_dg != ctx.dg_auto_last[0]:
                _log(
                    "DEEP GATE (auto): search-admitted PRN %s at snr >= %.0f "
                    "(hold %.0f s)%s"
                    % (
                        ",".join(str(p) for p in sorted(_fresh_dg)),
                        ctx.args.dll_deep_gate_from_search,
                        ctx.args.dll_deep_gate_search_hold_s,
                        ""
                        if not ctx.deep_gate
                        else " + hand-listed %s"
                        % ",".join(str(p) for p in sorted(ctx.deep_gate)),
                    )
                )
                ctx.dg_auto_last[0] = set(_fresh_dg)
        elif ctx.dg_auto_last[0]:
            _log(
                "DEEP GATE (auto): search-admitted set empty, no PRN at snr >= %.0f "
                "within the %.0f s hold"
                % (
                    ctx.args.dll_deep_gate_from_search,
                    ctx.args.dll_deep_gate_search_hold_s,
                )
            )
            ctx.dg_auto_last[0] = set()


def q_stall_notice(ctx):
    """The q stall guard: a notice when this chain's q duty falls well below the best it has
    reached in this process (fits.q_stall_verdict).

    Its history lives in dr_state, so without dead reckoning (dr_state None) it does not run;
    the instance-stall check's SKIPPED line says why."""
    # This chain's q duty (fraction of judged satellites at q >= q_stall_bar) over a
    # trailing window, against the best duty it has reached in this process. Without it a
    # chain can sit in a degraded steady state for hours, visible only to a same-time-of-day
    # comparison. Self-referential because chains differ ~4x in duty by construction: a
    # fleet-common bar would cry wolf on bds_b2b or never fire on gps_l5. The baseline only
    # rises, so a chain that degrades cannot redefine normal downward.
    #
    # A notice, not a control: it changes and gates nothing. Acting on a statistic this
    # noisy would be a new failure mode, and the recovery (a node restart, which clears the
    # C++ trim state a broker restart cannot) is not the broker's to perform.
    if ctx.dllp.fleet and ctx.dr_state is not None:
        _qs = [v["q"] for v in ctx.dllp.fleet.values() if v.get("q") is not None]
        if _qs:
            _duty = sum(1 for q in _qs if q >= ctx.args.q_stall_bar) / float(len(_qs))
            _qh = ctx.dr_state.setdefault("q_hist", [])
            _qh.append((ctx.t0, _duty))
            del _qh[: max(0, len(_qh) - 4000)]
            # The decision is a pure function (fits.q_stall_verdict) so a constructed
            # collapse can test it (test_q_stall.py): the on-sky replay fixtures never see
            # the duty fall, so a replay cannot tell "did not fire" from "cannot fire".
            _qb, _qv = q_stall_verdict(
                _qh,
                ctx.t0,
                ctx.args.q_stall_window,
                ctx.args.q_stall_frac,
                ctx.args.q_stall_min_best,
                ctx.dr_state.get("q_duty_best"),
            )
            ctx.dr_state["q_duty_best"] = _qb
            if _qv is not None:
                _log_rl(
                    "qstall",
                    "q STALL: duty %.2f over the last %.0f s vs %.2f best "
                    "this session (%.0f%% of it, bar q>=%.1f, %d sat(s)). "
                    "This chain has been degraded, not absent -- the usual "
                    "cause is railed C++ trim state, which a broker restart "
                    "does not clear. Judge against the same time of day, "
                    "then a node restart."
                    % (
                        _qv[0],
                        ctx.args.q_stall_window,
                        _qv[1],
                        100.0 * _qv[2],
                        ctx.args.q_stall_bar,
                        len(_qs),
                    ),
                    every_s=ctx.args.q_stall_notice_s,
                )


def stage_fleet_dll(ctx):
    """3c: the fleet delay-lock loop -- the instrument suite, then the control loop.

    Polls the fleet discriminator into `ctx.dllp.fleet` (the cycle's central per-satellite
    state), runs the `instruments.instr_*` diagnostics and measurements hung off that same
    poll, then the one routine that actuates, `codeloop.stage_dll_control`, then the detectors.
    Does nothing when dll_gain is 0."""
    if ctx.args.dll_gain > 0.0:
        # Fleet combine (docs/CHORD_GNSS_SHARED_DLL.md): sum the raw powers across every
        # instance that reports the same window, then form one discriminator. Ratios do not
        # sum -- (SUM E - SUM L)/(SUM E + SUM L) is not any function of the per-instance
        # dll_disc values -- which is why the combiner publishes e_pow/l_pow/p_pow.
        deep_gate_auto(ctx)
        ctx.dllp.inst_hops = {}
        # Displaced admission (--presence-admit-displaced): the thresholds are bundled once so
        # the polled arm (fleet_dll) and the comb arm (instr_tap_walk) judge with the same
        # numbers; with two copies of the policy, comparing the arms would compare the copies.
        ctx.dllp.admit_disp = (
            {
                "pedestal_max": ctx.args.presence_disp_pedestal_max,
                "off_max_chips": ctx.args.presence_disp_off_max,
                # the deep-fold evidence bar (fleet.apply_presence): the fold's
                # ~10x integration gain keeps a satellite a fraction of a chip
                # off-peak detectable where prompt power fails
                "deep_margin": ctx.args.presence_disp_deep_margin,
                # no displaced admission while the chain is browned out. D1 is fed
                # after the poll below, so this is the previous cycle's verdict.
                "browned_out": ctx.brown.active(),
            }
            if ctx.args.presence_admit_displaced
            else None
        )
        ctx.dllp.fleet = fleet_dll(
            ctx.dll_combiners,
            ctx.dll_hop_window,
            ctx.args.dll_min_instances,
            ctx.args.dll_quality_sigma,
            ctx.args.dll_quality_min,
            deep_gate_prns=ctx.dllp.deep_gate_eff,
            deep_gate_margin=ctx.args.dll_deep_gate_margin,
            # the noise probes anchor the presence floor; with too few
            # reporting, the chain is UNANCHORED and admits nobody
            # (fleet.apply_presence)
            probe_prns=ctx.probe_set,
            # the newest hop per instance, from the poll fleet_dll already
            # makes (no extra HTTP), for the axis and stall checks below; the
            # aggregate throws that per-instance axis away
            src_hops=ctx.dllp.inst_hops,
            admit_displaced=ctx.dllp.admit_disp,
            # the displaced admission fits the offset at this chain's tap spacing
            dll_spacing=ctx.args.dll_spacing,
            # judge each instance before its rows count: a pre-re-base row
            # (ahead of the anchor) or one that stopped advancing must not set
            # the fleet hop (fleet.live_instances)
            hop_hist=ctx.fleet_hop_hist,
            anchor_utc=ctx.utc0_sample0 or None,
            hops_per_sec=ctx.args.hops_per_sec,
        )
        # A chain whose presence floor has no noise anchor refuses every verdict (UNANCHORED):
        # nobody is admitted and nothing is trimmed. Log why and what to fix, so the refusal
        # is not mistaken for an empty sky.
        _fl0 = next(iter((ctx.dllp.fleet or {}).values()), None)
        if _fl0 is not None and _fl0.get("present_gate") == "UNANCHORED":
            _log_rl(
                "unanchored",
                "PRESENCE UNANCHORED: only %d probe(s) reporting (need 3, and 2 of them "
                "with prompt power). Nobody is admitted and nothing is trimmed -- deliberately, "
                "because the alternative is a floor built from the satellites, which passes "
                "about half of them by construction. Fix the probe supply: check that the "
                "selected probe PRNs have slots on the nodes (--probe-require-slot), not "
                "this gate." % _fl0.get("n_probe_q", 0),
                every_s=60.0,
            )
        # The comb arm (--telem-dll): the same discriminator rebuilt from the per-channel
        # Early/Prompt/Late the transport ships, rather than from powers each tracker already
        # summed across its own channels. When the gather has windows for this chain it
        # replaces ctx.dllp.fleet. The polled arm above still runs: the deep statistics come only
        # from the combiner's fold (handed across as coh_from), and the COMB-DLL line compares
        # the two arms.
        instruments.instr_tap_walk(ctx)
        # Prompt hold for the next cycle's lock gates: the fleet prompt ratio p_pow/p_med
        # (lock_prompt_hold, read by the coast/drop gate and the dead-reckon re-pin) and the
        # fleet q (lock_q, the dead-reckon re-pin). q is the metric lock is judged on.
        ctx.hold.prev.clear()
        ctx.hold.q.clear()
        for _p, _fl in (ctx.dllp.fleet or {}).items():
            _pm = _fl.get("p_med")
            if _pm:
                ctx.hold.prev[_p] = _fl["p_pow"] / _pm
            if _fl.get("q") is not None:
                ctx.hold.q[_p] = float(_fl["q"])
        # Cross-node coherent combine. Separate from the DLL: the DLL sums powers (phase-free,
        # which is what makes it cheap), while this removes the common per-record phase, and the
        # two answer different questions off the same endpoints. It actuates nothing: it is
        # published for the viewer and the beam map, and its record-stream rate (rate_hz) is
        # the rate the KCOH fold derotates with next cycle. A failure costs this cycle's rows.
        ctx.dllp.fcoh = {}
        # With --telem-coherent, fleet_coherent reads this chain's windows from the gather
        # (telem coherent_source) instead of polling /get_records. source=None, when the
        # gather is down, restarting or not carrying this chain, falls back to the REST poll at
        # the cost of a rate-limited log line.
        _tsrc = None
        if ctx.telem_client is not None:
            # The telemetry alignment check. `spread` is max-min of the live instances' newest
            # window index: 0 or 1 is the transport working. Logged whether or not anything
            # consumes the feed, so a misalignment shows at once rather than later as a physics
            # anomaly.
            _st = ctx.telem_client.stats()
            _cs = _st["chains"].get(ctx.telem_chain)
            if not _cs:
                _msg = "chain %s: nothing yet" % ctx.telem_chain
            elif not _cs.get("live"):
                _msg = "chain %s: ALL %d instances stale (%s)" % (
                    ctx.telem_chain,
                    _cs["instances"],
                    ",".join(_cs["stale"]),
                )
            else:
                # Spread is over live instances only; the stale ones are named. A stopped
                # instance keeps its last window forever, so folding it into the spread would
                # turn every instance death into a large alignment alarm.
                _msg = "chain %s: %d live, win %d..%d spread %d%s" % (
                    ctx.telem_chain,
                    _cs["live"],
                    _cs["win_min"],
                    _cs["win_max"],
                    _cs["spread"],
                    (" | STALE %s" % ",".join(_cs["stale"])) if _cs["stale"] else "",
                )
            _log_rl(
                "telem-stat",
                "TELEM %s frames %d gaps %d bad %d | %s"
                % (
                    "up" if _st["connected"] else "DOWN",
                    _st["frames"],
                    _st["gaps"],
                    _st["bad"],
                    _msg,
                ),
                every_s=30.0,
            )
        if ctx.args.telem_coherent and ctx.telem_client is not None:
            try:
                _tsrc = ctx.telem_client.coherent_source(
                    ctx.telem_chain,
                    prns=set(ctx.seeds) or None,
                    n_win=ctx.args.telem_windows,
                    lag=1,
                )
                if not _tsrc[0]:
                    _tsrc = None
                    _log_rl(
                        "telem-empty",
                        "telem: no windows for chain %r yet -- falling back to "
                        "/get_records (gather stats: %s)"
                        % (ctx.telem_chain, ctx.telem_client.stats()),
                    )
            except Exception as e:
                _tsrc = None
                _log_rl(
                    "telem-src", "telem: source failed (%s); using /get_records" % e
                )
        try:
            ctx.dllp.fcoh = fleet_coherent(
                ctx.dll_combiners,
                ctx.args.coh_min_instances,
                ctx.args.coh_min_records,
                prns=set(ctx.seeds) or None,
                floor_margin=ctx.args.coh_floor_margin,
                seed=int(_now()),
                # lets it fit the record-stream carrier rate (rate_hz) off
                # the records it already fetched. Hops per second, not records:
                # a record's first tuple element is a hop count.
                hop_rate_hz=ctx.args.hops_per_sec,
                # when set, the poll is skipped and the same estimator runs on
                # the gathered records
                source=_tsrc,
            )
        except Exception as e:
            _log_rl("fleet-coh", "fleet coherent: skipped this cycle (%s)" % e)
        # Path B (--n2-combiners): the same estimator on a separate population, reported beside
        # path A in the FLEET-COH line rather than merged. Merging would hide the comparison, and
        # both despread the same antenna voltages, so their noise is not independent and a merged
        # sum would not buy the sqrt(2) it appears to. Publishes nothing and touches no loop.
        ctx.dllp.fcoh_n2 = {}
        if ctx.n2_combiners:
            try:
                ctx.dllp.fcoh_n2 = fleet_coherent(
                    ctx.n2_combiners,
                    ctx.args.coh_min_instances,
                    ctx.args.coh_min_records,
                    prns=set(ctx.seeds) or None,
                    floor_margin=ctx.args.coh_floor_margin,
                    seed=int(_now()),
                )
            except Exception as e:
                _log_rl("fleet-coh-n2", "fleet coherent (path B): skipped (%s)" % e)
        instruments.instr_coherent_rows(ctx)
        # Fleet phase-slope delay fit (--spectrum-endpoints, docs/CHORD_JOINT_TRACKING.md). Its
        # tau feeds b_sat (SatBiasFilter, bsat_gain), which the dead-reckon seeds add to the
        # model's code phase. On its own flag because replay is strict-ordered: an unrecorded GET
        # is a TRANSCRIPT DIVERGENCE, and a replay of a transcript recorded without the flag
        # never issues these polls.
        ctx.dllp.spec_fit = {}
        instruments.instr_spectrum_fit(ctx)
        # The served C/N0 (instr_prompt_cn0, combdll.prompt_cn0): per-record prompt power off
        # the gather feed, debiased against the noise probes. It fits nothing, unlike the deep
        # fold, whose per-integration rate re-search on something the tracking loop already
        # fixed carries ~20 dB of paired self-scatter; that is why cn0_coh_db cannot be the
        # radiometry. Measurement only. It needs the probes as its noise anchor; without them it
        # publishes nothing rather than falling back to a peer comparison.
        # estimator_every_s throttles the telemetry-walk estimators; the last values are served
        # in between. run_est is set here because this is the first of the two blocks in cycle
        # order.
        ctx.dllp.run_est = (
            ctx.telem_client is not None and ctx.probe_set and _now() >= ctx.est_next[0]
        )
        if ctx.dllp.run_est:
            ctx.est_next[0] = _now() + ctx.args.estimator_every_s
        # The throttle exists for the walk, not for the estimator: each walk is pure Python over
        # ~1500 record decodes. With --comb-taps-cpp >= 2 and --fleet-trim-url, prompt C/N0
        # fetches the gather's already-reduced series instead of walking, so it runs every cycle
        # (throttling it would only age the served value) and only KCOH stays throttled.
        ctx.dllp.run_pcn0 = ctx.dllp.run_est or (
            ctx.args.comb_taps_cpp >= 2
            and ctx.args.fleet_trim_url
            and ctx.telem_client is not None
            and ctx.probe_set
        )
        ctx.dllp.pcn0 = ctx.est_last["pcn0"]
        instruments.instr_prompt_cn0(ctx)
        # The known-rate coherent C/N0 (instr_kcoh, combdll.coh_cn0): the ~1 s fold centred on
        # the previous cycle's record-stream rate (ctx.kcoh_rates, updated below after the fold,
        # so this integration never uses a rate fitted to its own records) or on zero, plus a
        # one-parameter residual-rate fit that the probes ride identically, so the probe floor
        # absorbs its gain on noise. No q gate, so it reaches the deep-sidelobe satellites a
        # per-record gate never passes. Rides the same telemetry as the comb DLL, throttled with
        # run_est; its rate feeds the joint rrate state only under --rrate-kcoh-feed.
        ctx.dllp.kcoh = ctx.est_last["kcoh"]
        instruments.instr_kcoh(ctx)
        # The rates for the next cycle, from this cycle's record-stream fit.
        for _p, _fc2 in (ctx.dllp.fcoh or {}).items():
            _r2 = _fc2.get("rate_hz")
            if _r2 is not None:
                ctx.kcoh_rates[_p] = float(_r2)
        # RF-path health (instr_rf_stats): clip fraction and per-band power from each GPU's
        # voltage tap. Measurement only: it steers and gates nothing. The clip fraction is what
        # separates a saturated front end from a large linear signal, which need different
        # fixes. Channels are grouped into bands by contiguity and band identity
        # (fits.rf_lobes), because a band-selective source is diagnosable only against a band
        # that was quiet in the same sample.
        instruments.instr_rf_stats(ctx)
        # Dark instances (instr_dark_inst): this cycle's per-instance deep snr (fcoh) against
        # each instance's tap. An instance that despreads nothing while its tap is fed is a
        # channel map or wiring fault that the fleet counts still list as present. Logged and
        # published; it changes no sum.
        instruments.instr_dark_inst(ctx)

        # The per-element complex gain (instr_element_poll): amplitude and phase per antenna,
        # per instance, the beam/peel coefficients. Assembled from the combiners'
        # /get_elements (leave-one-out cross-products accumulated per record on the node, where
        # the element axis lives), significance-anchored on the noise probes per (instance,
        # element). Measurement only; touches no loop.
        #
        # Never throttle in replay: the element poll runs every cycle while a transcript is
        # read. A transcript is an ordered recording of every GET, checked by URL as it is
        # consumed, so making fewer calls than the recording desynchronises the stream: each
        # later call consumes another call's entry, _get raises TRANSCRIPT DIVERGENCE on the URL
        # mismatch, and the callers catch it, so the replay continues on wrong data instead of
        # stopping. broker_equiv can gate what a call computes, never how often it is made; a
        # cadence change is live-only and needs a live measurement.
        instruments.instr_element_poll(ctx)
        # The served innovation: freshest value and 10-minute p95 per PRN. Cut by time here at
        # read, not by count at write. The history holds one sample per detection, however many
        # cycles the search re-serves it, so a PRN detected once in ten minutes reports that one
        # sample; one with no sample in ten minutes stops being served, and its history is
        # dropped after twenty.
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
        # The model innovation rides the same rows as minnov_* keys. A PRN can carry either or
        # both (INNOV needs a standing seed, MINNOV an established joint row); absent keys mean
        # "not measurable", never zero.
        for _p, _mh in list(ctx.minnov_hist.items()):
            if _now() - _mh[-1][0] > 1200.0:
                del ctx.minnov_hist[_p]
                continue
            _win = [(tt, vv) for tt, vv in _mh if _now() - tt <= 600.0]
            if not _win:
                continue
            _av = sorted(abs(vv) for _, vv in _win)
            ctx.dllp.innov_pub.setdefault(_p, {}).update(
                {
                    "minnov_chips": _win[-1][1],
                    "minnov_p95_10m": _av[max(0, math.ceil(0.95 * len(_av)) - 1)],
                    "minnov_n_10m": len(_win),
                }
            )
        # ── The model-primacy flip decision (--model-primacy-max) ──
        # The one writer of mp_flipped, once per cycle, from the measured p95s built above.
        # Enter: p95 below model_primacy_p95 over at least model_primacy_min_n samples; the
        # best-p95 eligible PRNs fill the cap and the rest are logged as the controls. Exit: p95
        # above model_primacy_exit_p95, or the referee starved (no MINNOV in the 10-minute
        # window, or no detection while flipped for model_primacy_starve_s). Every transition
        # logs one line.
        instruments.instr_model_primacy(ctx)
        if ctx.dllp.innov_pub:
            _log_rl(
                "innov",
                "INNOV %s: %s"
                % (
                    log_tag() or ctx.args.signal,
                    " ".join(
                        "%d:%+.2f(p95 %.2f, n%d)"
                        % (_p, v["innov_chips"], v["innov_p95_10m"], v["innov_n_10m"])
                        for _p, v in sorted(ctx.dllp.innov_pub.items())
                        if "innov_chips" in v
                    ),
                ),
                every_s=60.0,
            )
            _mv = [
                "%d:%+.2f(p95 %.2f, n%d)"
                % (_p, v["minnov_chips"], v["minnov_p95_10m"], v["minnov_n_10m"])
                for _p, v in sorted(ctx.dllp.innov_pub.items())
                if "minnov_chips" in v
            ]
            if _mv:
                _log_rl(
                    "minnov",
                    "MINNOV %s (model vs sky, flip-gate statistic): %s"
                    % (log_tag() or ctx.args.signal, " ".join(_mv)),
                    every_s=60.0,
                )
        # The tracker's code residual, from this cycle's records against this cycle's model
        # and clock. Failure here must cost the loop nothing: it is a published
        # measurement, not an input to any actuator.
        try:
            ctx.dllp.trk = trkresid.tracker_residuals(ctx)
        except Exception as _te:
            ctx.dllp.trk = {}
            _log_rl("trkresid", "tracker residual failed (%s)" % _te, every_s=60.0)
        if ctx.dllp.trk:
            _log_rl(
                "trkres",
                "TRKRES %s (tracker code residual, chips, clock removed): %s"
                % (
                    log_tag() or ctx.args.signal,
                    " ".join(
                        "%d:%+.3f(n%d sd%.2f)" % (_p, _v["chips"], _v["n"], _v["sd"])
                        for _p, _v in sorted(ctx.dllp.trk.items())
                    ),
                ),
                every_s=60.0,
            )
        # The fleet ADR: this cycle's records folded into each satellite's carrier arc, at
        # exact hops. A measurement, not an actuator input: failure costs the loop nothing.
        try:
            ctx.dllp.fadr = fleetadr.stage_fleet_adr(ctx)
        except Exception as _fe:
            ctx.dllp.fadr = {}
            _log_rl("fleetadr", "fleet ADR failed (%s)" % _fe, every_s=60.0)
        if ctx.dllp.fadr:
            _log_rl(
                "fadr",
                "FADR %s (fleet ADR: arc/records/instances, Doppler-only cycles): %s"
                % (
                    log_tag() or ctx.args.signal,
                    " ".join(
                        "%d:a%d/n%d/i%d %+.1f"
                        % (_p, _v["arc"], _v["n_rec"], _v["n_inst"], _v["dop_cycles"])
                        for _p, _v in sorted(ctx.dllp.fadr.items())
                    ),
                ),
                every_s=60.0,
            )
        if ctx.publisher is not None:
            # Published before the trim update so the row shows the state the loop acted
            # on, not the state after it acted -- otherwise a reader can never see the
            # input that produced a given correction.
            ctx.publisher.update(
                ctx.dllp.fleet,
                ctx.seeds,
                ctx.dls.trim,
                ctx.last_dets,
                ctx.dllp.fcoh,
                pcn0=ctx.dllp.pcn0,
                kcoh=ctx.dllp.kcoh,
                innov=ctx.dllp.innov_pub,
                cpp_trim={
                    _p: (_r.get("trim_chips") or 0.0)
                    for _p, _r in ctx.dls.readback.items()
                },
                integ=(ctx.dr_state or {}).get("integ"),
                # Both ages' stamps are ctx.drp.now_w, the frozen clock of the cycle whose
                # dead-reckon pass set them; against this cycle's, a pass that did not run
                # shows as a growing age.
                integ_now=ctx.t0,
                trk=ctx.dllp.trk,
                fadr=ctx.dllp.fadr,
            )
        ctx.dllp.report = []
        codeloop.stage_dll_control(ctx)
        if ctx.dllp.report:
            _log("DLL: " + "; ".join(ctx.dllp.report))
        # ── D0: the population-honest q series ─────────────────────────────────────────────
        # Recorded for every seeded satellite, present or not. The DLL line above lists only the
        # ones that passed the presence gate, so a satellite whose q craters leaves it, and a
        # statistic over that line measures survivors.
        ctx.qpop.note_cycle(ctx.t0, set(ctx.seeds), ctx.dllp.fleet)
        _tag = log_tag() or ctx.args.signal
        _qline = ctx.qpop.line(_tag)
        if _qline:
            _log_rl("qpop", _qline, every_s=120.0)

        # ── D1: the chain-wide brownout, as a labelled episode ────────────────────────
        # Labelled so a window that contained one can be excluded or stratified. Its active()
        # state also blocks displaced admission (above) and is passed to D3 (below).
        _npres = sum(
            1
            for _f in (ctx.dllp.fleet or {}).values()
            if isinstance(_f, dict) and _f.get("present")
        )
        _bmsg = ctx.brown.note_cycle(ctx.t0, _npres)
        if _bmsg:
            _log("%s: %s" % (_tag, _bmsg))

        # ── The boresight-transit veto (D3) ───────────────────────────────────────────────
        # Pooled over every constellation: ctx.transit_pd is the geometry-only sky
        # (sky.transit_sky, health gate off, since an unhealthy satellite still rails; the
        # strict sky dr_pd until the first dead-reckon predict), and the quantiser is shared, so
        # a per-chain answer would be worse than none: it would clear the chain carrying the
        # bright satellite and leave the rest looking like clean controls.
        _near = (
            nearest_boresight(ctx.transit_pd)
            if ctx.args.detector_transit_veto_deg > 0.0
            else None
        )
        _in_transit = bool(_near and _near[0] < ctx.args.detector_transit_veto_deg)
        if _in_transit:
            _log_rl(
                "transit-veto",
                "%s: BORESIGHT TRANSIT -- %s%d is %.1f deg off boresight (health %s); D3 "
                "suppressed (the quantiser rails and satellites drop across every chain at once)"
                % (
                    _tag,
                    _near[1][0],
                    _near[1][1],
                    _near[0],
                    (ctx.transit_pd or {}).get(_near[1], {}).get("health", "?"),
                ),
                every_s=120.0,
            )

        # ── D3: the handover sawtooth ─────────────────────────────────────────────────
        _uptime = ctx.uptime_s  # the startup hold-off, on the cycle clock (context.py)
        for _sp, _sr in (ctx.dls.readback or {}).items():
            if not isinstance(_sr, dict) or _sr.get("trim_chips") is None:
                continue
            if not _sr.get("armed"):
                ctx.saw.drop(_sp)  # a released trim decays by the leak, not a wipe
                continue
            _sq = ctx.qpop.summary(_sp)
            _sbt = ctx.birth_steps.get(_sp)
            _smsg = ctx.saw.note(
                ctx.t0,
                _sp,
                float(_sr["trim_chips"]),
                browned_out=ctx.brown.active(),
                uptime_s=_uptime,
                rebase_age_s=(ctx.t0 - _sbt) if _sbt is not None else None,
                present_frac=_sq[4] if _sq else None,
                q_mean=_sq[2] if _sq else None,
                in_transit=_in_transit,
            )
            if _smsg:
                _log("%s: %s" % (_tag, _smsg))
        if ctx.dop_rate_rejected:
            _log(
                "dop-rate: %d fit(s) REJECTED against the model (kept the model): %s"
                % (
                    len(ctx.dop_rate_rejected),
                    ", ".join(
                        "PRN %d fit %+.3f vs model %+.3f" % (k, v[0], v[1])
                        for k, v in sorted(ctx.dop_rate_rejected.items())[:5]
                    ),
                )
            )
            # Cleared after logging: an uncleared dict repeats every PRN ever rejected with
            # identical stale values, which reads as a live fault. cp_rate_rejected likewise.
            ctx.dop_rate_rejected.clear()
        # The same report for the code rate, on its own line: these reject against the pooled
        # clock rather than an orbit model, and a reader who cannot tell which reference
        # rejected a fit cannot act on it.
        if ctx.cp_rate_rejected:
            _log(
                "cp-rate: %d fit(s) REJECTED against the pooled clock (rate seeded per "
                "--cp-rate-model-primary, kept the fitted position): %s"
                % (
                    len(ctx.cp_rate_rejected),
                    ", ".join(
                        "PRN %d fit %+.3f vs clock %+.3f chips/s" % (k, v[0], v[1])
                        for k, v in sorted(ctx.cp_rate_rejected.items())[:5]
                    ),
                )
            )
            # Cleared after logging, for the same reason as dop_rate_rejected above.
            ctx.cp_rate_rejected.clear()
        _absent = sorted(
            p for p in ctx.seeds if ctx.seeds[p].get("doppler_rate_hz_s") is None
        )
        if _absent:
            # No carrier extrapolation and no quadratic code term for these satellites.
            _log(
                "dop-rate: %d seeded PRN(s) have no doppler rate at all: %s"
                % (len(_absent), _absent[:8])
            )
        if ctx.dop_rate_fitted:
            _log_rl(
                "doprate",
                "doppler-rate FIT seeded on %d sat(s): %s"
                % (
                    len(ctx.dop_rate_fitted),
                    "; ".join(
                        "PRN %d %+.4f Hz/s" % (k, v)
                        for k, v in sorted(ctx.dop_rate_fitted.items())[:5]
                    ),
                ),
                every_s=60.0,
            )
        # The presence bar, logged (rate-limited) whenever it is measured. A threshold on a
        # noisy statistic that is never printed cannot be audited, and this one moves with the
        # fleet size.
        if ctx.dllp.fleet:
            any_fl = next(iter(ctx.dllp.fleet.values()))
            _log_rl(
                "dll-floor",
                # Which arm produced these numbers: the polled powers, or the
                # comb arm's ("comb"). A floor with no provenance gets compared
                # across a switch without anyone noticing it changed source.
                "fleet DLL [%s]: %d PRN(s) over %d combiner(s), %d present, "
                "q floor %.2f%s"
                % (
                    any_fl.get("src") or "polled",
                    len(ctx.dllp.fleet),
                    len(ctx.dll_combiners),
                    sum(1 for v in ctx.dllp.fleet.values() if v["present"]),
                    any_fl["q_floor"],
                    ""
                    if any_fl["q_med"] is None
                    else " (noise median %.2f, sigma %.3f)"
                    % (any_fl["q_med"], any_fl["q_sigma"]),
                ),
            )
        # ── Per-instance axis lag ─────────────────────────────────────────────────────────
        # Each instance's newest hop (ctx.dllp.inst_hops, from the fleet DLL poll) against the
        # wall clock. The --combiner status cannot show this: it is one instance, and every PRN
        # in a poll carries the same pow_hop. Negative = the instance lags the wall. Positive =
        # a hop the F-engine has not reached, impossible for a processed record, so instances
        # more than 0.5 s ahead are named.
        if ctx.utc0_sample0 and ctx.dllp.inst_hops:
            _w = _now()
            _ia = sorted(
                ((ctx.utc0_sample0 + float(h) / ctx.args.hops_per_sec) - _w, str(k))
                for k, h in ctx.dllp.inst_hops.items()
                if h
            )
            if _ia:
                _fut = [k for d, k in _ia if d > 0.5]
                _log_rl(
                    "axis-inst",
                    "AXIS INST: n=%d  lag median %+.2f s  worst %+.2f s (%s)  "
                    "freshest %+.2f s (%s)  spread %.2f s%s"
                    % (
                        len(_ia),
                        _ia[len(_ia) // 2][0],
                        _ia[0][0],
                        _ia[0][1],
                        _ia[-1][0],
                        _ia[-1][1],
                        _ia[-1][0] - _ia[0][0],
                        ""
                        if not _fut
                        else "  *** %d FUTURE instance(s) >0.5 s AHEAD: %s"
                        % (len(_fut), ",".join(sorted(_fut)[:4])),
                    ),
                    every_s=30.0,
                )
        # ── Instance liveness ─────────────────────────────────────────────────────────────
        # A wedged instance (its stream stalled, e.g. a starved ring) still answers every poll
        # with plausible, well-formed rows, so this keys on the hop counter, never on
        # reachability: healthy advances ~5.9M hops per 30 s, wedged exactly 0.
        # fe_axis_stale_s cannot see it: it watches only the --combiner instance's hop. The
        # verdict is fits.instance_stall_verdict (test_instance_stall.py), which stays silent
        # when most of the fleet is also standing still (a global cause: a paused F-engine, a
        # replay, a clock step).
        #
        # dr_state is None when dead reckoning is off (no --dead-reckon or no --almanac) or failed
        # to initialise ('dead-reckon unavailable' at startup). Skip the check rather than
        # dereference it: an exception here kills the chain, not the stage, while the trims,
        # presence verdicts and axis checks are still valid. Say so, so a chain without dead
        # reckoning is not mistaken for one whose instances are healthy.
        if ctx.dllp.inst_hops and ctx.dr_state is None:
            if ctx.args.dead_reckon and ctx.args.almanac:
                _why = (
                    "so dead reckoning never initialised on this chain (look for "
                    "'dead-reckon unavailable' at startup); the q stall guard is skipped "
                    "too. The chain is running DEGRADED -- seeds are not being "
                    "dead-reckoned -- which is a bigger problem than the skipped checks."
                )
            else:
                _why = (
                    "because dead reckoning is off on this chain (--dead-reckon and "
                    "--almanac are not both set); the q stall guard is skipped too."
                )
            _log_rl(
                "inststall-nodr",
                "instance-stall check SKIPPED: dr_state is None, " + _why,
                every_s=120.0,
            )
        elif ctx.dllp.inst_hops:
            _ih, _stalled = instance_stall_verdict(
                ctx.dr_state.get("inst_hops", {}),
                ctx.dllp.inst_hops,
                ctx.t0,
                ctx.args.instance_stall_s,
            )
            ctx.dr_state["inst_hops"] = _ih
            if _stalled:
                _log_rl(
                    "inststall",
                    "INSTANCE STALLED: %d of %d serving but not advancing -- %s. "
                    "These answer 200 with plausible rows; their pow_hop has not "
                    "moved (healthy is ~5.9M hops/30 s) while at least half the "
                    "fleet advanced, so the fault is on those nodes -- check the node log, "
                    "then restart that node. Bandwidth is degraded fleet-wide until "
                    "then."
                    % (
                        len(_stalled),
                        len(_ih),
                        ", ".join(
                            "%s stuck at hop %d for %.0f s" % (u, h, dt_)
                            for u, h, dt_ in _stalled[:4]
                        ),
                    ),
                    every_s=300.0,
                )

        # ── The q stall guard ─────────────────────────────────────────────────────────────
        q_stall_notice(ctx)
        for k in list(ctx.dls.trim):
            if k not in ctx.seeds:
                del ctx.dls.trim[k]
                ctx.dls.last_hop.pop(k, None)
