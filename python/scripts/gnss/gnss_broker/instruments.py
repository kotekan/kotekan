"""The DLL stage's instrument suite: everything hung off the fleet poll that MEASURES.

WHAT IS AND IS NOT HERE. The fleet delay-lock stage does two unrelated jobs in one pass: it
closes the code loop (that is `_stage_dll_control`, and it is the only part that actuates),
and it runs a suite of diagnostics off the same poll of the same endpoints. This module is the
second job. Nothing here writes a seed, a trim or a command; every function returns None and
communicates by filling in `ctx.dllp`.

⚠️ THAT PROPERTY IS WHY THESE MOVED FIRST. A pure measurement has no ordering hazard against
the control path, so relocating it cannot change what the instrument does to the sky -- only
what it reports. The stages that actuate are a harder problem and stay nested for now.

⚠️ MEASUREMENT BUGS DO NOT CRASH, THEY PRODUCE WRONG VERDICTS. That is the worst failure mode
this project has, and each function below carries the specific trap that has already bitten
its own measurement. Read those before changing one.

Every function takes the `ChainContext` and nothing else -- see `context.py` for what that
holds and, more importantly, for the stable/per-cycle split that makes holding it sound.

@author Keith Vanderlinde
"""

import json
import os
import re
import time

from gnss_broker.transport import _TR, _now, _get, _log, _log_rl, parse_endpoints
from gnss_broker import combdll
from gnss_broker import elemgain
from gnss_broker.fits import rf_lobes
from gnss_broker.fleet import (
    fleet_spectrum, fleet_spectrum_aligned, fit_spectrum_delay, poll_rf_stats,
)


def instr_coherent_rows(ctx):
    """INSTRUMENT: one log line for both coherent-combine paths (fleet and N2).
        
    ⚠️ per_inst IS PRINTED BESIDE THE FLEET NUMBER ON PURPOSE. The fleet total alone cannot
    distinguish "the combine worked" from "one instance was already strong" -- and those two have
    opposite implications for every conclusion drawn from a coherent gain."""
    if ctx.dllp.fcoh or ctx.dllp.fcoh_n2:
        # One line, both paths, only PRNs one of them actually detected. per_inst is
        # printed alongside the fleet number because the fleet total alone cannot
        # distinguish "the combine worked" from "one instance was already strong".
        rows = []
        for prn in sorted(set(ctx.dllp.fcoh) | set(ctx.dllp.fcoh_n2)):
            a, b = ctx.dllp.fcoh.get(prn), ctx.dllp.fcoh_n2.get(prn)
            if not ((a and a.get("present")) or (b and b.get("present"))):
                continue

            def _f(v):
                # per_inst is url -> deep_snr (a FLOAT, not a row); best_inst_snr is
                # already reduced for us. '*' marks a value that did NOT clear the
                # shuffled-null floor -- printed anyway, because a number just under
                # the bar is the useful one when judging whether a fleet is big enough.
                if not v:
                    return "--"
                return "%.0f/%d%s (best inst %.0f, floor %.1f)" % (
                    v.get("deep_snr", 0.0), v.get("n_src", 0),
                    "" if v.get("present") else "*", v.get("best_inst_snr", 0.0),
                    v.get("floor", 0.0))

            rows.append("PRN %d A %s | B %s" % (prn, _f(a), _f(b)))
        if rows:
            _log("FLEET-COH: " + "; ".join(rows))
        # FLEET-INST (task #40). The FLEET-COH line reduces 12 instances to a max,
        # and a max cannot separate a COMMON fluctuation (sky, or the shared broker
        # seed -- the two surviving suspects for the level-proportional scatter)
        # from independent per-instance noise. Log every instance's own deep snr
        # plus the alignment, so the common/independent split is computable per
        # emit offline. Path A only -- path B's per_inst rides the same dict if
        # ever needed. Instance order is SORTED BY URL, stable across polls, so a
        # column in the log is a node/GPU throughout.
        for prn in sorted(ctx.dllp.fcoh):
            v = ctx.dllp.fcoh[prn]
            pi = v.get("per_inst") or {}
            if len(pi) < 2:
                continue

            def _tag(u):
                # http://cx19:12048/gnss1_n2combine -> cx19/1. Values are TAGGED
                # rather than positional so a missing instance cannot shift the
                # columns of every instance after it.
                m = re.search(r"//(\w+):\d+/\w*?(\d)[^/]*$", u)
                return "%s/%s" % (m.group(1), m.group(2)) if m else u[-12:]

            _log("FLEET-INST: PRN %d align %.3f n_rec %d %s"
                 % (prn, v.get("align", 0.0), v.get("n_rec", 0),
                    " ".join("%s=%.1f" % (_tag(u), pi[u]) for u in sorted(pi))))
        # WHO WAS LEFT OUT, AND WHY (2026-08-12). fleet_coherent now anchors on the
        # freshest window instead of demanding unanimity, so a stalled instance
        # degrades the combine instead of killing it -- but a fleet quietly running
        # on 7 of 8 is exactly the state that hid a frozen GPU for an hour. Name the
        # dropped instances and how many hops they had in the window, so "we are
        # combining fewer nodes than we own" is visible without polling by hand.
        _dropped = {}
        for prn, v in sorted(ctx.dllp.fcoh.items()):
            for u, n_h in (v.get("dropped") or []):
                _dropped.setdefault(u, []).append(n_h)
        if _dropped:
            _log_rl("fleet-drop",
                    "FLEET-DROP: %d instance(s) outside the fleet's current window "
                    "-- %s (a frozen or lagging tracker; the combine continued on "
                    "the rest)"
                    % (len(_dropped),
                       ", ".join("%s (%d hops shared, %d PRN)"
                                 % (_tag(u), max(v), len(v))
                                 for u, v in sorted(_dropped.items()))),
                    every_s=120.0)


def instr_rf_stats(ctx):
    """INSTRUMENT: RF/SK statistics poll, published for the band-power and RFI watch.
        
    Throttled on its own clock (--rf-stats-poll-s) rather than the cycle, because it is a
    diagnostic and must never pace the control loop."""
    if ctx.args.rf_stats_endpoints and ctx.publisher is not None:
        _rf_ep = parse_endpoints(ctx.args.rf_stats_endpoints, ctx.base)
        if _rf_ep and ctx.t0 - ctx.rf_last[0] >= ctx.args.rf_stats_poll_s:
            ctx.rf_last[0] = ctx.t0
            _rf = poll_rf_stats(_rf_ep, rf_lobes,
                                fetch_sk=ctx.args.rfi_stats,
                                fetch_drops=ctx.args.drop_stats)
            ctx.publisher.set_rf(_rf, ctx.t0)
            _on = [v for v in _rf.values() if v.get("state") == "on"]
            if _on:
                _clip = max((max((l["clip_lo"] + l["clip_hi"]) for l in v["lobes"])
                             for v in _on if v.get("lobes")), default=0.0)
                _log_rl("rfstats",
                        "RF PATH: %d/%d instance(s) armed, worst clip %.4f of "
                        "nibbles, %d lobe(s)/instance, pass cost %.2f ms"
                        % (len(_on), len(_rf_ep), _clip,
                           max((len(v.get("lobes") or []) for v in _on), default=0),
                           max((v.get("cost_ms") or 0.0) for v in _on)),
                        every_s=300.0)
                # A rail is not a level, it is DAMAGE: past a few percent the
                # quantiser is discarding the signal it is meant to carry, and
                # everything downstream reads it as a coherence loss.
                if _clip > 0.01:
                    _log_rl("rfclip",
                            "⚠️ RF CLIPPING: %.2f%% of nibbles at a rail. Above ~1%% "
                            "the 4+4b quantiser is losing signal, not just headroom, "
                            "and every C/N0 and coherence below this point is "
                            "understated for a reason that is NOT the sky. Check "
                            "which lobe (get_rf) before blaming a chain."
                            % (100.0 * _clip), every_s=120.0)


def instr_tap_walk(ctx):
    """INSTRUMENT: the fleet tap walk (--telem-dll), and the C++ taps arm.
        
    Walks the gathered telemetry frames into the per-PRN aggregate the DLL then acts on.
    `--comb-taps-cpp` replaces the Python walk with the C++ one; everything after it -- the
    aggregate, the per-channel merge, the presence policy -- is unchanged and stays here, which is
    what makes the two arms comparable at all.
        
    ⚠️ IT PRODUCES `_dllp.fleet`, the per-PRN state dict the whole cycle reads."""
    if ctx.args.telem_dll and ctx.telem_client is not None:
        # THE C++ TAPS ARM (--comb-taps-cpp). `taps_src` replaces the Python walk over
        # the gathered frames; everything after it -- the aggregate, the per-channel
        # merge, the presence policy -- is unchanged and stays here.
        _tsrc_cpp = None
        if ctx.args.comb_taps_cpp and ctx.args.fleet_trim_url:
            _tsrc_cpp = (lambda _ch, _pr: combdll.taps_from_rest(
                _get, ctx.args.fleet_trim_url, _ch, prns=_pr))
        try:
            _cf = combdll.fleet_dll_comb(
                ctx.telem_client, ctx.telem_chain,
                taps_src=_tsrc_cpp if ctx.args.comb_taps_cpp >= 2 else None,
                n_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
                min_instances=ctx.args.dll_min_instances,
                k_sigma=ctx.args.dll_quality_sigma, q_fallback=ctx.args.dll_quality_min,
                prns=set(ctx.seeds) or None, probe_prns=ctx.probe_set,
                # #79: the SAME effective set as the polled arm above. This is the
                # arm that ships on gps_l5 (COMB-DLL replaces `fleet` below), so
                # gating it on the hand-listed set alone would compute the
                # auto-admission and then throw it away.
                deep_gate_prns=ctx.dllp.deep_gate_eff,
                deep_gate_margin=ctx.args.dll_deep_gate_margin,
                # deep_snr / deep_floor / coherence_s AND the quadrature fallback,
                # from the arm that has them
                coh_from=ctx.dllp.fleet,
                admit_displaced=ctx.dllp.admit_disp,
)
        except Exception as e:
            _cf = None
            _log_rl("comb-dll-err",
                    "COMB-DLL: failed (%s) -- the polled discriminator is unchanged" % e)
        # SHADOW: fetch the C++ taps, build the SAME product from them, and report the
        # paired difference on the loop's own cycles. Costs one GET and one rebuild and
        # changes nothing -- which is the point: the offline gate proves the arithmetic
        # on identical bytes, and this proves the two arms are looking at the same sky
        # through the same window depth with the same instance tags, which no offline
        # fixture can.
        if ctx.args.comb_taps_cpp == 1 and _tsrc_cpp is not None and _cf:
            try:
                _cx = combdll.fleet_dll_comb(
                    ctx.telem_client, ctx.telem_chain, taps_src=_tsrc_cpp,
                    n_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
                    min_instances=ctx.args.dll_min_instances,
                    k_sigma=ctx.args.dll_quality_sigma, q_fallback=ctx.args.dll_quality_min,
                    prns=set(ctx.seeds) or None, probe_prns=ctx.probe_set,
                    deep_gate_prns=ctx.dllp.deep_gate_eff,
                    deep_gate_margin=ctx.args.dll_deep_gate_margin, coh_from=ctx.dllp.fleet,
                    admit_displaced=ctx.dllp.admit_disp,
    )
                _sh = sorted(set(_cf) & set(_cx))
                if _sh:
                    _dd = sorted(abs(_cf[p]["disc"] - _cx[p]["disc"]) for p in _sh)
                    _dq = sorted(abs(_cf[p]["q"] - _cx[p]["q"]) for p in _sh)
                    _log("COMB-TAPS shadow %s: %d shared PRN(s) (py %d, cpp %d); "
                         "|ddisc| med %.2e max %.2e; |dq| med %.2e max %.2e"
                         % (ctx.telem_chain, len(_sh), len(_cf), len(_cx),
                            _dd[len(_dd) // 2], _dd[-1], _dq[len(_dq) // 2], _dq[-1]))
                else:
                    _log_rl("comb-taps-empty",
                            "COMB-TAPS shadow %s: NO SHARED PRNs (py %d, cpp %d) -- "
                            "that is a chain-name, window-depth or instance-tag "
                            "mismatch, not a rounding difference"
                            % (ctx.telem_chain, len(_cf), len(_cx)), every_s=60.0)
            except Exception as e:
                _log_rl("comb-taps-sh",
                        "COMB-TAPS shadow failed (%s) -- nothing changed" % e,
                        every_s=60.0)
        if _cf:
            _shared = sorted(set(_cf) & set(ctx.dllp.fleet or {}))
            _dd = sorted(_cf[p]["disc"] - ctx.dllp.fleet[p]["disc"] for p in _shared)
            _log_rl("comb-dll",
                    "COMB-DLL %s: %d PRNs from %d instances / %d channels; vs polled on "
                    "%d shared: median ddisc %+.4f, max %.4f%s"
                    % (ctx.telem_chain, len(_cf),
                       max(v["n_src"] for v in _cf.values()),
                       int(max(v["n_chan"] for v in _cf.values())), len(_shared),
                       _dd[len(_dd) // 2] if _dd else 0.0,
                       max((abs(x) for x in _dd), default=0.0),
                       "" if _shared else "  (NO OVERLAP -- check the chain key)"),
                    every_s=30.0)
            ctx.dllp.fleet = _cf
        else:
            _log_rl("comb-dll-empty",
                    "COMB-DLL %s: no windows yet -- closing the loop on the polled "
                    "discriminator this cycle" % ctx.telem_chain, every_s=60.0)


def instr_prompt_cn0(ctx):
    """INSTRUMENT: served prompt C/N0 (#57 step 1) from the gathered frames.
        
    The same C++ arm as the comb taps and under the same flag; this walks the gathered frames a
    SECOND time for the same channel tuples, which is the cost being traded for an independent
    estimate.
        
    ⚠️ SERVED C/N0 IS BLIND TO CODE ERROR (#47) -- it is computed from the prompt correlator, so a
    satellite sitting off the peak can read healthy. Never judge lock on it; judge on q."""
    if ctx.dllp.run_pcn0:
        try:
            ctx.dllp.pcn0 = combdll.prompt_cn0(
                ctx.telem_client, ctx.telem_chain,
                # The same C++ arm as the comb taps, under the same flag: this walks
                # the gathered frames a SECOND time for the same channel-tuples.
                recs_src=((lambda _ch, _pr: combdll.recs_from_rest(
                    _get, ctx.args.fleet_trim_url, _ch, prns=_pr))
                    if (ctx.args.comb_taps_cpp >= 2 and ctx.args.fleet_trim_url) else None),
                n_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
                min_instances=ctx.args.dll_min_instances,
                k_sigma=ctx.args.dll_quality_sigma,
                prns=set(ctx.seeds) or None, probe_prns=ctx.probe_set,
                hop_s=1.0 / ctx.args.hops_per_sec)
        except Exception as e:
            ctx.dllp.pcn0 = None
            _log_rl("pcn0-err",
                    "PROMPT-CN0: failed (%s) -- rows served without it" % e)
        ctx.est_last["pcn0"] = ctx.dllp.pcn0
        if ctx.dllp.pcn0:
            _lv = ["PRN %d %.1f dB-Hz (duty %.2f%s)"
                   % (p, v["cn0_db"], v["duty"],
                      "" if v["split_db"] is None
                      else ", split %+.1f" % v["split_db"])
                   for p, v in sorted(ctx.dllp.pcn0.items())
                   if v["cn0_db"] is not None and not v["probe"]]
            _any = next(iter(ctx.dllp.pcn0.values()))
            # sigma2 is IN the line on purpose: it is a live, probe-anchored noise
            # power at cycle cadence -- the first greppable series for #56's
            # fleet-wide level swings (measured moving 3 dB in 2 min on 2026-08-15,
            # carrying every satellite's served C/N0 with it, common-mode).
            _log_rl("pcn0",
                    "PROMPT-CN0 %s: %s | q_gate %.2f, sigma2 %.3e from %d probe "
                    "records"
                    % (ctx.telem_chain,
                       "; ".join(_lv) if _lv else "no PRN above the noise",
                       _any["q_gate"], _any["sigma2"], _any["n_probe_rec"]),
                    every_s=30.0)
        elif ctx.telem_client is not None:
            _log_rl("pcn0-empty",
                    "PROMPT-CN0 %s: no estimate this cycle (no windows, or fewer "
                    "than 16 probe records for the noise anchor)" % ctx.telem_chain,
                    every_s=120.0)


def instr_element_poll(ctx):
    """INSTRUMENT: per-element complex gain poll (#57 step 2, elemgain).
        
    ⚠️ ELEMCAL MUST ACTUALLY WARM. The weak-lock hunt of 2026-08-20 ended here: trackers re-anchor
    roughly every record, and BOTH reset() and the warmth-growing update() were tied to that, so
    the calibrator never accumulated. The array was already phase-coherent (equal-weight coherence
    1.00) -- the estimator was the patient."""
    if (ctx.args.element_poll and ctx.dll_combiners
            and (_TR.mode == "read"
                 or _now() - ctx.elem_poll_t[0] >= ctx.args.element_poll_every_s)):
        ctx.elem_poll_t[0] = _now()
        try:
            _pe, _srv = elemgain.poll_elements(ctx.dll_combiners)
            # WEDGED INSTANCES ARE EXCLUDED AND NAMED (see elemgain.drop_stale):
            # a frozen combiner keeps serving byte-identical gains from a sky ten
            # minutes gone, and a median over instances would blend them in.
            _pe, _stale = elemgain.drop_stale(_pe) if _pe else ({}, [])
            _etab = elemgain.gain_table(_pe, ctx.probe_set) if _pe else {}
            if _etab and ctx.publisher is not None:
                ctx.publisher.set_elements(_etab)
            _log_rl("elemgain",
                    "ELEM-GAIN: %d/%d instance(s) served, %d used, %d PRN(s)%s"
                    % (_srv, len(ctx.dll_combiners), len(_pe), len(_etab),
                       ("  ⚠️ STALE (excluded): "
                        + ", ".join("%s %s" % (t, "%.0f s behind" % l if l else "no hop")
                                    for t, l in _stale)) if _stale else ""),
                    every_s=120.0)
            # RAW archive, present sats + probes, throttled. Reopened per tick --
            # at one append a minute a persistent handle buys nothing and a
            # reopened one survives log rotation and NFS hiccups.
            if (_etab and ctx.args.element_archive_dir
                    and _now() - ctx.elem_arch_t[0] >= ctx.args.element_archive_every_s):
                ctx.elem_arch_t[0] = _now()
                _keep = {p for p, v in (ctx.dllp.fleet or {}).items()
                         if v.get("present")} | set(ctx.probe_set)
                _fn = os.path.join(
                    ctx.args.element_archive_dir, "elem_%s_%s.jsonl"
                    % (ctx.chain_id, time.strftime("%Y%m%d", time.gmtime())))
                with open(_fn, "a") as _fh:
                    for _tag2, _d2 in _pe.items():
                        for _p2, _v2 in _d2.items():
                            if _p2 not in _keep:
                                continue
                            _fh.write(json.dumps(
                                {"t": round(_now(), 2), "chain": ctx.chain_id,
                                 "prn": _p2, "inst": _tag2,
                                 "probe": _p2 in ctx.probe_set,
                                 "keff": round(_v2["keff"], 1),
                                 "hop": _v2["hop"],
                                 "u": [[float("%.5g" % r), float("%.5g" % i)]
                                       for r, i in _v2["u"]],
                                 "p2": [float("%.5g" % x) for x in _v2["p2"]],
                                 "q": [float("%.5g" % x) for x in _v2["q"]]},
                                separators=(",", ":")) + "\n")
        except Exception as e:
            _log_rl("elemgain-err",
                    "ELEM-GAIN: cycle failed (%s) -- table unchanged" % e)


def instr_model_primacy(ctx):
    """INSTRUMENT: model-primacy census -- which satellites the model, not the search, is driving.
        
    Reads the innovation p95 population to decide whether a flipped satellite is genuinely
    model-primary or merely DETECTION-STARVED, which look identical in any single-cycle view."""
    if ctx.args.model_primacy_max > 0:
        _mp_p95 = {int(_p): (v["minnov_p95_10m"], v["minnov_n_10m"])
                   for _p, v in ctx.dllp.innov_pub.items() if "minnov_chips" in v}
        for _p in sorted(ctx.mp_flipped):
            _pv = _mp_p95.get(_p)
            _starved = (_now() - ctx.mp_last_det.get(_p, _now())
                        > ctx.args.model_primacy_starve_s)
            if _p in ctx.dr_untrusted:
                # The legacy a0 integrity (EMA clock, no b_sat, ~1-chip bar)
                # judges a model the flip does not run: the flipped seed is the
                # JOINT model's (clk+b_sat via the slew). MINNOV referees flipped
                # sats -- the p95/starve exits below. The dr loop's seeding guard
                # exempts flipped sats for the same reason (BOTH sites, or the
                # sat is orphaned seedless). Keep the disagreement visible:
                _log_rl("mp-legacy-%d" % _p,
                        "MODEL-PRIMACY NOTE PRN %d: legacy integrity flags the "
                        "EMA model (%s) -- overridden while flipped; MINNOV "
                        "referees this sat" % (_p, ctx.dr_untrusted[_p]),
                        every_s=300.0)
            if _pv is not None and _pv[0] > ctx.args.model_primacy_exit_p95:
                ctx.mp_flipped.discard(_p)
                ctx.mp_cooldown[_p] = _now()
                ctx.dr_state["seeded"].discard(_p)
                ctx.dr_state["pin"].pop(_p, None)
                _log("MODEL-PRIMACY EXIT PRN %d: minnov p95 %.2f > %.2f -- the "
                     "search re-anchors on its next detection" %
                     (_p, _pv[0], ctx.args.model_primacy_exit_p95))
            elif _pv is None or _starved:
                ctx.mp_flipped.discard(_p)
                ctx.mp_cooldown[_p] = _now()
                ctx.dr_state["seeded"].discard(_p)
                ctx.dr_state["pin"].pop(_p, None)
                _log("MODEL-PRIMACY EXIT PRN %d: referee starved (no fresh "
                     "MINNOV/detection in %.0f s) -- nothing holds a seed "
                     "unrefereed" % (_p, ctx.args.model_primacy_starve_s))
        _elig = sorted(
            (_pv[0], _p) for _p, _pv in _mp_p95.items()
            if _p not in ctx.mp_flipped and _p not in ctx.probe_set
            and _p not in ctx.dr_untrusted
            and _now() - ctx.mp_cooldown.get(_p, -1e9) > 300.0
            and _pv[1] >= ctx.args.model_primacy_min_n
            and _pv[0] < ctx.args.model_primacy_p95)
        for _v95, _p in _elig:
            if len(ctx.mp_flipped) >= ctx.args.model_primacy_max:
                break
            ctx.mp_flipped.add(_p)
            ctx.dr_state["seeded"].add(_p)
            # A standing hold is a SEARCH-currency freeze; the model owns the
            # seed now, so the freeze is released (the dr guard relies on this).
            ctx.cp_held.discard(_p)
            ctx.mp_last_det[_p] = _now()
            _log("MODEL-PRIMACY ENTER PRN %d: minnov p95 %.2f (n>=%d) -- seed is "
                 "now the MODEL's (dr slew, clk+b_sat); detections feed the "
                 "filter and the referee only. Controls (eligible, unflipped): %s"
                 % (_p, _v95, ctx.args.model_primacy_min_n,
                    ",".join(str(q) for _, q in _elig
                             if q not in ctx.mp_flipped) or "none"))


def instr_kcoh(ctx):
    """INSTRUMENT: the known-rate coherent estimator (#57 step 3, cn0_kcoh).
        
    ⚠️ THE RATE IT DEROTATES WITH MUST BE CAUSAL. `_kcoh_rates` is updated AFTER each cycle's fold
    from that cycle's record-stream fit, so the fold only ever uses a rate estimated from EARLIER
    records. That is the entire difference between this estimator and the deep fold it replaces --
    a fold derotated by a rate fitted to its own records measures the fit, not the sky.
        
    ⚠️ ARM 17 (row-injected rates) WAS RETIRED FOR CAUSE on 2026-08-25: it was a carrier mirror,
    convicted by an attribution toggle (q SD 2x on both bands). The fallback path -- fit-fed rates,
    and probes always -- is the one that stands."""
    if ctx.dllp.run_est:
        # ARM 17: overlay converged rows' predictions onto the fit-fed rates.
        # See --kcoh-rate-from-row. Fallback for any sat without a converged
        # row (and every probe) is exactly the old path.
        _rates_in = dict(ctx.kcoh_rates)
        _row_inj = 0
        if ctx.args.kcoh_rate_from_row and ctx.args.rrate_state:
            try:
                _jri = ctx.rx.joint_receiver(ctx.band_id, ctx.code_len, rereference=ctx.args.joint_rereference)
                for _pi in (set(ctx.seeds) - ctx.probe_set):
                    _ki = (ctx.args.dr_constellation, int(_pi))
                    _sy = ((ctx.args.carrier_hz / _jri.C_LIGHT)
                           * _jri.rrate_sigma(_ki) + _jri.f_carrier_sigma())
                    if _sy <= ctx.args.kcoh_row_max_sigma:
                        _rates_in[_pi] = _jri.carrier_correction_hz(
                            _ki, ctx.args.carrier_hz)
                        _row_inj += 1
            except Exception as e:
                _log_rl("kcoh-row-err",
                        "KCOH row-injection skipped: %s" % e, every_s=300.0)
        try:
            ctx.dllp.kcoh = combdll.coh_cn0(
                ctx.telem_client, ctx.telem_chain, rates=_rates_in,
                n_win=(ctx.args.telem_dll_windows or ctx.args.telem_windows),
                min_instances=ctx.args.dll_min_instances,
                prns=set(ctx.seeds) or None, probe_prns=ctx.probe_set,
                hop_s=1.0 / ctx.args.hops_per_sec)
        except Exception as e:
            ctx.dllp.kcoh = None
            _log_rl("kcoh-err",
                    "KCOH: failed (%s) -- rows served without it" % e)
        ctx.est_last["kcoh"] = ctx.dllp.kcoh
        if ctx.dllp.kcoh:
            _kv = ["PRN %d %.1f dB-Hz (sig %.0f, eta %s, f %+.2f%+.2f)"
                   % (p, v["cn0_db"], v["sig"],
                      "%.0f" % v["eta"] if v["eta"] is not None else "--",
                      v["rate_hz"], v.get("rate_resid_hz", 0.0))
                   for p, v in sorted(ctx.dllp.kcoh.items())
                   if v["cn0_db"] is not None and not v["probe"] and v["sig"] > 3.0]
            _log_rl("kcoh",
                    "KCOH %s: %s | %d PRNs folded, floor from %d probe folds"
                    "%s"
                    % (ctx.telem_chain,
                       "; ".join(_kv) if _kv else "no fold above the probe floor",
                       len(ctx.dllp.kcoh),
                       next(iter(ctx.dllp.kcoh.values()))["n_probe"],
                       (", %d row-injected" % _row_inj) if _row_inj else ""),
                    every_s=30.0)


def instr_spectrum_fit(ctx):
    """INSTRUMENT: the spectrum delay fit (#32/#53), which feeds the #50 re-seed.
        
    ⚠️ ALIGNED GATHER, NOT THE LEGACY POLL. The legacy fleet_spectrum takes whatever each instance
    accumulated since ITS OWN last poll, so the instances are not summing the same records and
    each carries a free phase. That is the defect of #52 -- a transport artifact, not a modelling
    one -- and it is why the aligned path exists.
        
    ⚠️ THE PEAK RATIO IS A SHUFFLED-NULL SIGNIFICANCE, built from the same points with values
    reassigned within each instance. Passing the bar means the fold beat its own null; it is not a
    tuned constant. It is ALSO not significance at poll cadence in the latched regime -- ratios up
    to 3.2 attached to contradictory taus seconds apart is what tripped #90's flight 2."""
    if ctx.spectrum_endpoints:
        try:
            # ALIGNED GATHER (task #53). The legacy fleet_spectrum takes
            # whatever each instance accumulated since ITS OWN last poll, so the
            # instances are not summing the same records and each one carries a
            # free phase -- which is the defect of #52, not a modelling
            # convenience. The addressable path names ONE window index, asks every
            # instance for exactly that window, rotates each reply onto a single
            # phase reference (phi0), drops PRNs whose accumulator was re-anchored
            # mid-window, and ASSERTS that the same index meant the same samples
            # everywhere. Instances that cannot address windows (a node still on a
            # pre-#53 config) are EXCLUDED rather than mixed in -- one unaligned
            # member puts the free phase straight back.
            #
            # Behind a flag because replay is strict-ordered: the aligned path
            # issues TWO GETs per instance (availability, then the window) where
            # the legacy path issues one, so an old transcript replayed with it on
            # would diverge. Same pattern as --spectrum-endpoints itself.
            if ctx.args.spectrum_aligned:
                _spec, _smeta = fleet_spectrum_aligned(
                    ctx.spectrum_endpoints, prns=set(ctx.seeds) or None, log=_log,
                    stale_margin=ctx.args.spectrum_stale_margin)
                _log_rl("specwin",
                        "SPEC-WINDOW %s: %d/%d instance(s) served%s%s"
                        % (_smeta.get("window"), len(_smeta.get("served") or {}),
                           len(ctx.spectrum_endpoints),
                           (", %d dropped" % len(_smeta["dropped"]))
                           if _smeta.get("dropped") else "",
                           (", %d re-anchored" % len(_smeta["reanchored"]))
                           if _smeta.get("reanchored") else ""),
                        every_s=120.0)
            else:
                _spec = fleet_spectrum(ctx.spectrum_endpoints,
                                       prns=set(ctx.seeds) or None)
            # PER-SUBBAND ARCHIVE (task #25, 2026-08-11). fleet_spectrum ALREADY
            # returns (freq_id, amplitude, energy, instance) per PRN -- the
            # per-frequency x per-element product the science side wants -- and
            # until now every one of those points was collapsed to a single tau by
            # fit_spectrum_delay and then discarded. This writes the RAW points,
            # before any collapse. Nothing is recomputed: the cost is one line per
            # (prn, channel, instance) per poll, a few kB, and no node-side change.
            #
            # Deliberately RAW, matching the observables record's contract: no
            # clock removed, no combine, no normalisation. A per-subband beam map,
            # a per-element gain solve and the band-health summary are all
            # functions of these rows, computed offline where they can be redone
            # when the model improves -- and a COMBINED number can always be
            # rebuilt from the parts, while parts can never be recovered from a
            # combined number. That asymmetry is the whole argument for storing
            # this axis rather than a central estimator over it.
            if ctx.spec_writer is not None and ctx.drp.t_now_abs is not None:
                try:
                    # WALL CLOCK for the archive, not t_now_abs -- the latter is
                    # seconds since the F-engine's sample-0 anchor (now_w -
                    # utc0_sample0), so using it named the first file
                    # gps_l5_19700102.jsonl and would have made every row
                    # unjoinable with the observables record. The receiver-relative
                    # time is kept alongside, since it is the one the code phases
                    # are referenced to.
                    _n = ctx.spec_writer(ctx.drp.now_w, ctx.band_id, _spec, t_rx=ctx.drp.t_now_abs)
                    _log_rl("specarch",
                            "SPEC-ARCHIVE: %d point(s) this poll -> %s"
                            % (_n, ctx.args.spectrum_archive), every_s=300.0)
                except Exception as e:
                    _log_rl("specarch-err",
                            "spectrum archive write failed: %s" % e, every_s=120.0)
            for prn, pts in _spec.items():
                r = fit_spectrum_delay(pts, ctx.args.chip_rate_hz, ctx.args.hops_per_sec)
                if r is not None:
                    ctx.dllp.spec_fit[prn] = {"tau_chips": r[0], "peak": r[1],
                                     "floor": r[2], "n_pts": r[3], "n_inst": r[4]}
        except Exception as e:
            _log_rl("fleet-spec", "fleet spectrum: skipped this cycle (%s)" % e)
        if ctx.dllp.spec_fit:
            _log_rl("specfit", "SPEC-FIT: " + "; ".join(
                "PRN %d tau %+.3f chips (p/f %.1fx, %d ch/%d inst)"
                % (prn, v["tau_chips"], v["peak"] / max(v["floor"], 1e-12),
                   v["n_pts"], v["n_inst"])
                for prn, v in sorted(ctx.dllp.spec_fit.items())), every_s=30.0)
            # FEED b_sat (task #33). Presence-gated by the same lock metric the
            # reseed logic trusts (deep-certified sig), because P1 measured weak-sat
            # tau to be self-reference-biased toward zero -- a weak "tau ~ 0" is not
            # a measurement. The filter adds its own thin-fleet and lobe gates.
            # ⚠️ DELIBERATELY NOT given the fold-independent prompt path that the
            # re-pin gate got (#58): this one admits a MEASUREMENT, not a retention
            # decision, and P1 measured weak-sat tau biased toward zero by
            # self-reference. "The prompt is on the signal" does not establish that
            # this poll's tau is trustworthy. Separate question, separate evidence.
            for prn, v in ctx.dllp.spec_fit.items():
                if ctx.sig_of(ctx.status.get(prn, {})) >= ctx.args.lock_snr:
                    ctx.bsat.update(prn, v["tau_chips"], v["n_inst"], ctx.t0)
            _log_rl("bsat", "BSAT: " + ctx.bsat.summary(ctx.t0), every_s=60.0)
            # Ride the fleet dict into the publisher: the viewer/status row grows
            # spec_tau_chips + spec_peak_ratio next to the disc it will one day
            # replace. EXISTING rows only -- publisher.update() indexes fleet rows'
            # p_pow/disc/q directly, so a bare fit-only row would crash it; a PRN
            # the DLL poll missed this cycle is still in the SPEC-FIT log line.
            for prn, v in ctx.dllp.spec_fit.items():
                if prn in ctx.dllp.fleet:
                    ctx.dllp.fleet[prn]["spec_tau"] = v["tau_chips"]
                    ctx.dllp.fleet[prn]["spec_ratio"] = v["peak"] / max(v["floor"], 1e-12)
                    ctx.dllp.fleet[prn]["bsat"] = ctx.bsat.get(prn, ctx.t0)
            # #85: stash (tau, ratio, t) for the model-primary joint feed, which
            # runs EARLIER in cycle order and so reads last cycle's fit -- one poll
            # stale, same causality as the trim readback it sits beside.
            if ctx.dr_state is not None and ctx.drp.t_now_abs is not None:
                _sy = ctx.dr_state.setdefault("spec_y", {})
                for prn, v in ctx.dllp.spec_fit.items():
                    _sy[prn] = (v["tau_chips"],
                                v["peak"] / max(v["floor"], 1e-12), ctx.drp.t_now_abs)
