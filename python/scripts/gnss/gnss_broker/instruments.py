"""The DLL stage's instrument suite: everything hung off the fleet poll that measures.

The fleet delay-lock stage (fleetdll.stage_fleet_dll) does two jobs in one pass: it closes the
code loop (codeloop.stage_dll_control) and it runs diagnostics off the same poll of the same
endpoints. This module is the second job. Its functions measure, log, publish and feed
estimators (b_sat); none writes a trim or a command, and only instr_model_primacy changes who
owns a seed (it decides which satellites the dead-reckon model seeds). Every function returns
None and communicates through ctx, mostly `ctx.dllp`.

A measurement bug does not crash; it produces a wrong verdict. Each docstring below names its
measurement's known trap: read it before changing the function.

Every function takes the `ChainContext` and nothing else -- see `context.py` for what that
holds and, more importantly, for the stable/per-cycle split that makes holding it sound.

@author Keith Vanderlinde
"""

import json
import os
import time

from gnss_broker.transport import _TR, _now, _get, _log, _log_rl, parse_endpoints
from gnss_broker import combdll
from gnss_broker import elemgain
from gnss_broker.fits import rf_lobes
from gnss_broker import signals
from gnss_broker.fleet import (
    fleet_spectrum_aligned,
    fit_spectrum_delay,
    poll_rf_stats,
)


def instr_coherent_rows(ctx):
    """FLEET-COH: one log line for both coherent-combine paths (A: --dll-combiners or the
    gather, B: --n2-combiners), plus the FLEET-INST and FLEET-DROP lines.
        
    The best single instance's snr is printed beside the fleet number on purpose: the fleet
    total alone cannot distinguish "the combine worked" from "one instance was already
    strong", and those have opposite implications for any conclusion drawn from a coherent
    gain."""
    if ctx.dllp.fcoh or ctx.dllp.fcoh_n2:
        # One line, both paths, only PRNs one of them detected.
        rows = []
        for prn in sorted(set(ctx.dllp.fcoh) | set(ctx.dllp.fcoh_n2)):
            a, b = ctx.dllp.fcoh.get(prn), ctx.dllp.fcoh_n2.get(prn)
            if not ((a and a.get("present")) or (b and b.get("present"))):
                continue

            def _f(v):
                # per_inst is url -> deep_snr (a float, not a row); best_inst_snr is its
                # max. '*' marks a value that did not clear the shuffled-null floor,
                # printed anyway: a number just under the bar is the useful one when
                # judging whether a fleet is big enough.
                if not v:
                    return "--"
                return "%.0f/%d%s (best inst %.0f, floor %.1f)" % (
                    v.get("deep_snr", 0.0),
                    v.get("n_src", 0),
                    "" if v.get("present") else "*",
                    v.get("best_inst_snr", 0.0),
                    v.get("floor", 0.0),
                )

            rows.append("PRN %d A %s | B %s" % (prn, _f(a), _f(b)))
        if rows:
            _log("FLEET-COH: " + "; ".join(rows))
        # FLEET-INST: FLEET-COH reduces the instances to a max, which cannot separate a
        # common fluctuation (sky, or the shared broker seed) from independent
        # per-instance noise. Log every instance's own deep snr and the alignment, so
        # the split is computable per emit offline. Path A only. Instances are sorted
        # by URL, stable across polls.
        for prn in sorted(ctx.dllp.fcoh):
            v = ctx.dllp.fcoh[prn]
            pi = v.get("per_inst") or {}
            if len(pi) < 2:
                continue
            # Values are tagged by instance (elemgain._tag) rather than positional, so a
            # missing instance cannot shift the columns of every instance after it.
            _log(
                "FLEET-INST: PRN %d align %.3f n_rec %d %s"
                % (
                    prn,
                    v.get("align", 0.0),
                    v.get("n_rec", 0),
                    " ".join("%s=%.1f" % (elemgain._tag(u), pi[u]) for u in sorted(pi)),
                )
            )
        # FLEET-DROP: fleet_coherent anchors on the freshest window rather than demanding
        # unanimity, so a stalled instance degrades the combine instead of killing it, and
        # a fleet quietly running short can hide a frozen GPU. Name the dropped instances
        # and how many hops they shared with the window.
        _dropped = {}
        for prn, v in sorted(ctx.dllp.fcoh.items()):
            for u, n_h in v.get("dropped") or []:
                _dropped.setdefault(u, []).append(n_h)
        if _dropped:
            _log_rl(
                "fleet-drop",
                "FLEET-DROP: %d instance(s) outside the fleet's current window "
                "-- %s (a frozen or lagging tracker; the combine continued on "
                "the rest)"
                % (
                    len(_dropped),
                    ", ".join(
                        "%s (%d hops shared, %d PRN)"
                        % (elemgain._tag(u), max(v), len(v))
                        for u, v in sorted(_dropped.items())
                    ),
                ),
                every_s=120.0,
            )


def instr_rf_stats(ctx):
    """RF/SK statistics poll (--rf-stats-endpoints), published for the band-power and RFI watch.
        
    Throttled on its own clock (rf_stats_poll_s) rather than the cycle, because it is a
    diagnostic and must never pace the control loop."""
    if ctx.args.rf_stats_endpoints and ctx.publisher is not None:
        _rf_ep = parse_endpoints(ctx.args.rf_stats_endpoints, ctx.base)
        if _rf_ep and ctx.t0 - ctx.rf_last[0] >= ctx.args.rf_stats_poll_s:
            ctx.rf_last[0] = ctx.t0
            # The declared band set (--rf-bands), resolved once per poll from chain names
            # to carriers via the signal table. An empty list leaves every row unnamed,
            # the correct degradation: a missing declaration must cost a label, never
            # produce a wrong one.
            _carr = signals.carriers_for_chains(
                [c for c in (ctx.args.rf_bands or "").split(",") if c.strip()]
            )
            _rf = poll_rf_stats(
                _rf_ep,
                lambda ch, pw, lo, hi, fids=None: rf_lobes(ch, pw, lo, hi, fids, _carr),
                fetch_sk=ctx.args.rfi_stats,
                fetch_drops=ctx.args.drop_stats,
            )
            ctx.publisher.set_rf(_rf, ctx.t0)
            _on = [v for v in _rf.values() if v.get("state") == "on"]
            if _on:
                _clip = max(
                    (max(l["clip"] for l in v["lobes"]) for v in _on if v.get("lobes")),
                    default=0.0,
                )
                _log_rl(
                    "rfstats",
                    "RF PATH: %d/%d instance(s) armed, worst clip %.4f of "
                    "nibbles, %d lobe(s)/instance, pass cost %.2f ms"
                    % (
                        len(_on),
                        len(_rf_ep),
                        _clip,
                        max((len(v.get("lobes") or []) for v in _on), default=0),
                        max((v.get("cost_ms") or 0.0) for v in _on),
                    ),
                    every_s=300.0,
                )
                # A rail is damage, not a level: past ~1% of nibbles the quantiser is
                # discarding the signal it is meant to carry, and everything downstream
                # reads it as a coherence loss.
                if _clip > 0.01:
                    _log_rl(
                        "rfclip",
                        "RF CLIPPING: %.2f%% of nibbles at a rail. Above ~1%% "
                        "the 4+4b quantiser is losing signal, not just headroom, "
                        "and every C/N0 and coherence below this point is "
                        "understated for a reason that is not the sky. Check "
                        "which lobe (get_rf) before blaming a chain." % (100.0 * _clip),
                        every_s=120.0,
                    )


def instr_tap_walk(ctx):
    """The comb DLL (--telem-dll): the fleet discriminator rebuilt from the gathered taps.
        
    Walks the gathered telemetry frames into the per-PRN aggregate the DLL then acts on. With
    --comb-taps-cpp 2 and --fleet-trim-url the taps come from the gather's C++ reduction
    instead; everything after (the aggregate, the per-channel merge, the presence policy) is
    the same code, which is what makes the two arms comparable.
        
    When it returns PRNs it replaces `ctx.dllp.fleet`, the per-PRN state the rest of the cycle
    reads; otherwise the polled discriminator stands — the COMB-DLL line compares the two."""
    if ctx.args.telem_dll and ctx.telem_client is not None:
        # The C++ taps arm: `taps_src` replaces the Python walk over the gathered frames.
        _tsrc_cpp = None
        if ctx.args.comb_taps_cpp >= 2 and ctx.args.fleet_trim_url:
            _tsrc_cpp = lambda _ch, _pr: combdll.taps_from_rest(
                _get,
                ctx.args.fleet_trim_url,
                _ch,
                prns=_pr,
                want_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
            )
        try:
            _cf = combdll.fleet_dll_comb(
                ctx.telem_client,
                ctx.telem_chain,
                taps_src=_tsrc_cpp,
                n_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
                min_instances=ctx.args.dll_min_instances,
                k_sigma=ctx.args.dll_quality_sigma,
                q_fallback=ctx.args.dll_quality_min,
                prns=set(ctx.seeds) or None,
                probe_prns=ctx.probe_set,
                # The same effective deep-gate set as the polled arm: this arm replaces
                # `fleet` below, so gating it on the hand-listed set alone would compute
                # the auto-admission and then throw it away.
                deep_gate_prns=ctx.dllp.deep_gate_eff,
                deep_gate_margin=ctx.args.dll_deep_gate_margin,
                # deep_snr / deep_floor / coherence_s and the quadrature fallback, from
                # the arm that has them
                coh_from=ctx.dllp.fleet,
                admit_displaced=ctx.dllp.admit_disp,
            )
        except Exception as e:
            _cf = None
            _log_rl(
                "comb-dll-err",
                "COMB-DLL: failed (%s) -- the polled discriminator is unchanged" % e,
            )
        if _cf:
            _shared = sorted(set(_cf) & set(ctx.dllp.fleet or {}))
            _dd = sorted(_cf[p]["disc"] - ctx.dllp.fleet[p]["disc"] for p in _shared)
            # xcoh per present PRN: the lobe sum's own check that the senders' phi0 put them
            # on one reference. A present PRN reading ~0 is a cross-sender reference loss --
            # its prompt is then below what per-sender summing would give -- and this is
            # the only place that loss is visible from (q and disc both stay plausible).
            _xc = " ".join(
                "%d:%.2f" % (p, v["xcoh"])
                for p, v in sorted(_cf.items())
                if v.get("present") and v.get("xcoh") is not None
            )
            _log_rl(
                "comb-dll",
                "COMB-DLL %s: %d PRNs from %d instances / %d channels; vs polled on "
                "%d shared: median ddisc %+.4f, max %.4f%s | xcoh %s"
                % (
                    ctx.telem_chain,
                    len(_cf),
                    max(v["n_src"] for v in _cf.values()),
                    int(max(v["n_chan"] for v in _cf.values())),
                    len(_shared),
                    _dd[len(_dd) // 2] if _dd else 0.0,
                    max((abs(x) for x in _dd), default=0.0),
                    "" if _shared else "  (no overlap -- check the chain key)",
                    _xc or "-",
                ),
                every_s=30.0,
            )
            ctx.dllp.fleet = _cf
        else:
            _log_rl(
                "comb-dll-empty",
                "COMB-DLL %s: no windows yet -- closing the loop on the polled "
                "discriminator this cycle" % ctx.telem_chain,
                every_s=60.0,
            )


def instr_prompt_cn0(ctx):
    """Served prompt C/N0 (combdll.prompt_cn0), debiased against the noise probes.
        
    Under the same flags as the comb taps' C++ arm (--comb-taps-cpp 2, --fleet-trim-url) it
    reads the gather's reduced records; otherwise it walks the gathered frames a second time
    for the same channel tuples, the cost of an independent estimate.
        
    Served C/N0 is blind to code error: it comes from the prompt correlator, so a satellite
    sitting off the peak can read healthy. Never judge lock on it; judge on q."""
    if ctx.dllp.run_pcn0:
        try:
            ctx.dllp.pcn0 = combdll.prompt_cn0(
                ctx.telem_client,
                ctx.telem_chain,
                # The same C++ arm as the comb taps, under the same flags; without it
                # the gathered frames are walked a second time.
                recs_src=(
                    (
                        lambda _ch, _pr: combdll.recs_from_rest(
                            _get,
                            ctx.args.fleet_trim_url,
                            _ch,
                            prns=_pr,
                            want_win=(
                                ctx.args.telem_dll_windows or ctx.args.telem_windows
                            ),
                        )
                    )
                    if (ctx.args.comb_taps_cpp >= 2 and ctx.args.fleet_trim_url)
                    else None
                ),
                n_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
                min_instances=ctx.args.dll_min_instances,
                min_sig=ctx.args.cn0_prompt_min_sig,
                prns=set(ctx.seeds) or None,
                probe_prns=ctx.probe_set,
                hop_s=1.0 / ctx.args.hops_per_sec,
            )
        except Exception as e:
            ctx.dllp.pcn0 = None
            _log_rl("pcn0-err", "PROMPT-CN0: failed (%s) -- rows served without it" % e)
        ctx.est_last["pcn0"] = ctx.dllp.pcn0
        if ctx.dllp.pcn0:
            _lv = [
                "PRN %d %.1f dB-Hz (duty %.2f%s)"
                % (
                    p,
                    v["cn0_db"],
                    v["duty"],
                    "" if v["split_db"] is None else ", split %+.1f" % v["split_db"],
                )
                for p, v in sorted(ctx.dllp.pcn0.items())
                if v["cn0_db"] is not None and not v["probe"]
            ]
            _any = next(iter(ctx.dllp.pcn0.values()))
            # sigma2 is in the line on purpose: a live, probe-anchored noise power at cycle
            # cadence, the series that shows a fleet-wide level swing carrying every
            # satellite's served C/N0 with it, common-mode.
            _log_rl(
                "pcn0",
                "PROMPT-CN0 %s: %s | present at t>=%.0f, q_noise %.2f, sigma2 %.3e "
                "from %d probe records"
                % (
                    ctx.telem_chain,
                    "; ".join(_lv) if _lv else "no PRN above the noise",
                    _any["min_sig"],
                    _any["q_noise"],
                    _any["sigma2"],
                    _any["n_probe_rec"],
                ),
                every_s=30.0,
            )
        elif ctx.telem_client is not None:
            _log_rl(
                "pcn0-empty",
                "PROMPT-CN0 %s: no estimate this cycle (no windows, or fewer "
                "than 16 probe records for the noise anchor)" % ctx.telem_chain,
                every_s=120.0,
            )


def instr_element_poll(ctx):
    """Per-element complex gain poll (--element-poll, elemgain) from the combiners.
        
    Publishes the gain table, stale instances excluded, and under --element-archive-dir
    appends the raw per-instance values for present satellites and probes to
    elem_<chain>_<YYYYMMDD>.jsonl. Polled every element_poll_every_s, and every cycle in a
    transcript replay, which must make the same calls as the recording."""
    if (
        ctx.args.element_poll
        and ctx.dll_combiners
        and (
            _TR.mode == "read"
            or _now() - ctx.elem_poll_t[0] >= ctx.args.element_poll_every_s
        )
    ):
        ctx.elem_poll_t[0] = _now()
        try:
            _pe, _srv = elemgain.poll_elements(ctx.dll_combiners)
            # Wedged instances are excluded and named (elemgain.drop_stale): a frozen
            # combiner keeps serving identical gains from a sky minutes gone, and a median
            # over instances would blend them in.
            _pe, _stale = elemgain.drop_stale(_pe) if _pe else ({}, [])
            _etab = elemgain.gain_table(_pe, ctx.probe_set) if _pe else {}
            if _etab and ctx.publisher is not None:
                ctx.publisher.set_elements(_etab)
            _log_rl(
                "elemgain",
                "ELEM-GAIN: %d/%d instance(s) served, %d used, %d PRN(s)%s"
                % (
                    _srv,
                    len(ctx.dll_combiners),
                    len(_pe),
                    len(_etab),
                    (
                        "  STALE (excluded): "
                        + ", ".join(
                            "%s %s" % (t, "%.0f s behind" % l if l else "no hop")
                            for t, l in _stale
                        )
                    )
                    if _stale
                    else "",
                ),
                every_s=120.0,
            )
            # Raw archive, present satellites and probes, every element_archive_every_s.
            # Reopened per write: at one append a minute a persistent handle buys nothing,
            # and a reopened one survives log rotation and NFS hiccups.
            if (
                _etab
                and ctx.args.element_archive_dir
                and _now() - ctx.elem_arch_t[0] >= ctx.args.element_archive_every_s
            ):
                ctx.elem_arch_t[0] = _now()
                _keep = {
                    p for p, v in (ctx.dllp.fleet or {}).items() if v.get("present")
                } | set(ctx.probe_set)
                _fn = os.path.join(
                    ctx.args.element_archive_dir,
                    "elem_%s_%s.jsonl"
                    % (ctx.chain_id, time.strftime("%Y%m%d", time.gmtime())),
                )
                with open(_fn, "a") as _fh:
                    for _tag2, _d2 in _pe.items():
                        for _p2, _v2 in _d2.items():
                            if _p2 not in _keep:
                                continue
                            _fh.write(
                                json.dumps(
                                    {
                                        "t": round(_now(), 2),
                                        "chain": ctx.chain_id,
                                        "prn": _p2,
                                        "inst": _tag2,
                                        "probe": _p2 in ctx.probe_set,
                                        "keff": round(_v2["keff"], 1),
                                        "hop": _v2["hop"],
                                        "u": [
                                            [float("%.5g" % r), float("%.5g" % i)]
                                            for r, i in _v2["u"]
                                        ],
                                        "p2": [float("%.5g" % x) for x in _v2["p2"]],
                                        "q": [float("%.5g" % x) for x in _v2["q"]],
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
        except Exception as e:
            _log_rl(
                "elemgain-err", "ELEM-GAIN: cycle failed (%s) -- table unchanged" % e
            )


def instr_model_primacy(ctx):
    """Model primacy (--model-primacy-max): which satellites the model, not the search, seeds.
        
    Flips in the eligible satellites with the lowest MINNOV p95 up to the cap, and flips one
    back when its p95 rises past model_primacy_exit_p95 or its referee starves. The p95
    population is what separates a satellite that is genuinely model-primary from one merely
    detection-starved, which look identical in any single-cycle view."""
    if ctx.args.model_primacy_max > 0:
        _mp_p95 = {
            int(_p): (v["minnov_p95_10m"], v["minnov_n_10m"])
            for _p, v in ctx.dllp.innov_pub.items()
            if "minnov_chips" in v
        }
        for _p in sorted(ctx.mp_flipped):
            _pv = _mp_p95.get(_p)
            _starved = (
                _now() - ctx.mp_last_det.get(_p, _now())
                > ctx.args.model_primacy_starve_s
            )
            if _p in ctx.dr_untrusted:
                # dr_untrusted judges the solved-clock model without b_sat, which a
                # flipped seed does not run: the flipped seed is the joint model's
                # (clk + b_sat, via the slew), and MINNOV referees it through the exits
                # below. The dead-reckon seeding guard exempts flipped satellites for the
                # same reason, or they would be left seedless. Log the disagreement.
                _log_rl(
                    "mp-legacy-%d" % _p,
                    "MODEL-PRIMACY NOTE PRN %d: the dead-reckon integrity check "
                    "flags the solved-clock model (%s) -- overridden while flipped; "
                    "MINNOV referees this sat" % (_p, ctx.dr_untrusted[_p]),
                    every_s=300.0,
                )
            if _pv is not None and _pv[0] > ctx.args.model_primacy_exit_p95:
                ctx.mp_flipped.discard(_p)
                ctx.mp_cooldown[_p] = _now()
                ctx.dr_state["seeded"].discard(_p)
                ctx.dr_state["pin"].pop(_p, None)
                _log(
                    "MODEL-PRIMACY EXIT PRN %d: minnov p95 %.2f > %.2f -- the "
                    "search re-anchors on its next detection"
                    % (_p, _pv[0], ctx.args.model_primacy_exit_p95)
                )
            elif _pv is None or _starved:
                ctx.mp_flipped.discard(_p)
                ctx.mp_cooldown[_p] = _now()
                ctx.dr_state["seeded"].discard(_p)
                ctx.dr_state["pin"].pop(_p, None)
                _log(
                    "MODEL-PRIMACY EXIT PRN %d: referee starved (no fresh "
                    "MINNOV/detection in %.0f s) -- nothing holds a seed "
                    "unrefereed" % (_p, ctx.args.model_primacy_starve_s)
                )
        _elig = sorted(
            (_pv[0], _p)
            for _p, _pv in _mp_p95.items()
            if _p not in ctx.mp_flipped
            and _p not in ctx.probe_set
            and _p not in ctx.dr_untrusted
            and _now() - ctx.mp_cooldown.get(_p, -1e9) > 300.0
            and _pv[1] >= ctx.args.model_primacy_min_n
            and _pv[0] < ctx.args.model_primacy_p95
        )
        for _v95, _p in _elig:
            if len(ctx.mp_flipped) >= ctx.args.model_primacy_max:
                break
            ctx.mp_flipped.add(_p)
            ctx.dr_state["seeded"].add(_p)
            # A standing hold is a search-currency freeze; the model owns the seed now,
            # so the hold is released (the dead-reckon guard relies on this).
            ctx.cp_held.discard(_p)
            ctx.mp_last_det[_p] = _now()
            _log(
                "MODEL-PRIMACY ENTER PRN %d: minnov p95 %.2f (n>=%d) -- seed is "
                "now the model's (dr slew, clk+b_sat); detections feed the "
                "filter and the referee only. Controls (eligible, unflipped): %s"
                % (
                    _p,
                    _v95,
                    ctx.args.model_primacy_min_n,
                    ",".join(str(q) for _, q in _elig if q not in ctx.mp_flipped)
                    or "none",
                )
            )


def instr_kcoh(ctx):
    """KCOH: the known-rate coherent C/N0 estimator (combdll.coh_cn0).
        
    The rate it derotates with must be causal. `ctx.kcoh_rates` is updated after each cycle's
    fold from that cycle's record-stream fit (fleetdll), so the fold only uses a rate estimated
    from earlier records. A fold derotated by a rate fitted to its own records, as the deep
    fold's per-integration rate search is, measures the fit, not the sky."""
    if ctx.dllp.run_est:
        try:
            ctx.dllp.kcoh = combdll.coh_cn0(
                ctx.telem_client,
                ctx.telem_chain,
                rates=dict(ctx.kcoh_rates),
                n_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
                min_instances=ctx.args.dll_min_instances,
                prns=set(ctx.seeds) or None,
                probe_prns=ctx.probe_set,
                hop_s=1.0 / ctx.args.hops_per_sec,
            )
        except Exception as e:
            ctx.dllp.kcoh = None
            _log_rl("kcoh-err", "KCOH: failed (%s) -- rows served without it" % e)
        ctx.est_last["kcoh"] = ctx.dllp.kcoh
        if ctx.dllp.kcoh:
            _kv = [
                "PRN %d %.1f dB-Hz (sig %.0f, eta %s, f %+.2f%+.2f)"
                % (
                    p,
                    v["cn0_db"],
                    v["sig"],
                    "%.0f" % v["eta"] if v["eta"] is not None else "--",
                    v["rate_hz"],
                    v.get("rate_resid_hz", 0.0),
                )
                for p, v in sorted(ctx.dllp.kcoh.items())
                if v["cn0_db"] is not None and not v["probe"] and v["sig"] > 3.0
            ]
            _log_rl(
                "kcoh",
                "KCOH %s: %s | %d PRNs folded, floor from %d probe folds"
                % (
                    ctx.telem_chain,
                    "; ".join(_kv) if _kv else "no fold above the probe floor",
                    len(ctx.dllp.kcoh),
                    next(iter(ctx.dllp.kcoh.values()))["n_probe"],
                ),
                every_s=30.0,
            )


def instr_spectrum_fit(ctx):
    """The fleet phase-slope delay fit (--spectrum-endpoints): per-PRN tau, which feeds b_sat.
        
    The peak ratio (p/f) is a shuffled-null significance, built from the same points with the
    phases permuted within each instance. Passing the bar means the fold beat its own null; it
    is not a tuned constant. It is also not a significance at poll cadence in the latched
    regime: high ratios can attach to contradictory taus seconds apart."""
    if ctx.spectrum_endpoints:
        try:
            # Aligned gather: every instance must sum the same records on one phase
            # reference, or each carries a free phase. fleet_spectrum_aligned names one
            # window index, asks every instance for exactly that window, rotates each
            # reply onto one phase reference (phi0), drops PRNs whose accumulator
            # re-anchored mid-window, and refuses the window unless the index meant the
            # same samples everywhere. An instance that cannot serve the window is dropped
            # rather than mixed in: one unaligned member puts the free phase back.
            _spec, _smeta = fleet_spectrum_aligned(
                ctx.spectrum_endpoints,
                prns=set(ctx.seeds) or None,
                log=_log,
                stale_margin=ctx.args.spectrum_stale_margin,
            )
            _log_rl(
                "specwin",
                "SPEC-WINDOW %s: %d/%d instance(s) served%s%s"
                % (
                    _smeta.get("window"),
                    len(_smeta.get("served") or {}),
                    len(ctx.spectrum_endpoints),
                    (", %d dropped" % len(_smeta["dropped"]))
                    if _smeta.get("dropped")
                    else "",
                    (", %d re-anchored" % len(_smeta["reanchored"]))
                    if _smeta.get("reanchored")
                    else "",
                ),
                every_s=120.0,
            )
            for prn, pts in _spec.items():
                r = fit_spectrum_delay(
                    pts, ctx.args.chip_rate_hz, ctx.args.hops_per_sec
                )
                if r is not None:
                    ctx.dllp.spec_fit[prn] = {
                        "tau_chips": r[0],
                        "peak": r[1],
                        "floor": r[2],
                        "n_pts": r[3],
                        "n_inst": r[4],
                    }
        except Exception as e:
            _log_rl("fleet-spec", "fleet spectrum: skipped this cycle (%s)" % e)
        if ctx.dllp.spec_fit:
            _log_rl(
                "specfit",
                "SPEC-FIT: "
                + "; ".join(
                    "PRN %d tau %+.3f chips (p/f %.1fx, %d ch/%d inst)"
                    % (
                        prn,
                        v["tau_chips"],
                        v["peak"] / max(v["floor"], 1e-12),
                        v["n_pts"],
                        v["n_inst"],
                    )
                    for prn, v in sorted(ctx.dllp.spec_fit.items())
                ),
                every_s=30.0,
            )
            # Feed b_sat, gated on the lock metric the dead-reckon re-pin trusts
            # (sig_of >= lock_snr): a weak satellite's tau is biased toward zero by
            # self-reference, so a weak tau ~ 0 is not a measurement. SatBiasFilter adds
            # its own thin-fleet and lobe gates. Deliberately not given the
            # fold-independent prompt path the re-pin gate has: this admits a
            # measurement, not a retention decision, and a prompt on the signal does not
            # make this poll's tau trustworthy.
            for prn, v in ctx.dllp.spec_fit.items():
                if ctx.sig_of(ctx.status.get(prn, {})) >= ctx.args.lock_snr:
                    ctx.bsat.update(prn, v["tau_chips"], v["n_inst"], ctx.t0)
            _log_rl("bsat", "BSAT: " + ctx.bsat.summary(ctx.t0), every_s=60.0)
