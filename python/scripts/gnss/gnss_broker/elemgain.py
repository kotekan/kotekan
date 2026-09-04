"""THE PER-ELEMENT COMPLEX GAIN, from the combiners' /get_elements (task #57 step 2).

What the node accumulates (GnssCoherentCombiner): per (PRN, element), an EMA of

    u_e = A_e * conj(ref_e),   ref_e = the sum of the OTHER elements' prompts, SAME record

so the sky phase -- common across elements, near-white in time (gnssElemCal.hpp) -- cancels
per record before any averaging. The mean of u_e is unbiased for the element's complex gain
times <|ref|^2>: the element's noise is uncorrelated with its leave-one-out reference, so
noise averages down coherently instead of rectifying. The reference is built from the RAW
per-antenna prompts, upstream of the element cal's weights, so #62's non-causal weighting
cannot enter. This module turns those raw parts into the served table:

    amp_e   = |u_e| / q_e          gain magnitude, in array-mean units
    phase_e = arg(u_e)             gain phase RELATIVE TO THE ARRAY MEAN -- the peel/beam
                                   coefficient. Absolute phase does not exist; this one does.
    sig_e   = |u_e| / n_floor_e    significance against the PER-ELEMENT noise anchor:
                                   n_floor_e = median over the NOISE PROBES' |u_e| at the
                                   same (instance, element). Same probe discipline as
                                   cn0_prompt ([[chord-cn0-prompt-estimator]]): the probes
                                   ride the identical pipeline, so every normalisation
                                   cancels in the ratio. NO peer fallback -- without probe
                                   rows the sig column is simply absent, never guessed.

Each INSTANCE keeps its own row set on purpose: an instance is ~7 frequency channels, so the
per-instance axis IS the frequency axis -- a per-element delay is a phase ramp across it,
and reducing instances away would destroy exactly what a peel solution needs. A combined
number can be rebuilt from the parts; parts can never be recovered from a combined number.
"""
import cmath
import math
import re

from .transport import _get, _log_rl


def _tag(url):
    """http://cx19:12049/gnss1_e5b_n2combine -> cx19/1 (stable, sortable, column-safe)."""
    m = re.search(r"//(\w+):\d+/\w*?(\d)[^/]*$", url)
    return "%s/%s" % (m.group(1), m.group(2)) if m else url[-12:]


def poll_elements(endpoints):
    """{tag: {prn: {u:[(re,im)..], p2:[..], q:[..], keff, hop}}} plus the served-count.

    Unreachable or pre-#57 instances (404: the endpoint ships with the next node restart)
    are skipped and counted, never fatal -- the table degrades, the caller logs the count.
    """
    out, served = {}, 0
    for url in endpoints:
        try:
            rows = _get("%s/get_elements" % url)
        except Exception as e:
            _log_rl("elemgain-%s" % url,
                    "ELEM-GAIN: %s unreachable (%s) -- a node predating /get_elements, "
                    "or down" % (url, e), every_s=600.0)
            continue
        served += 1
        d = {}
        for r in rows or []:
            els = r.get("elems") or []
            d[int(r["prn"])] = {
                "u": [(float(e[0]), float(e[1])) for e in els],
                "p2": [float(e[2]) for e in els],
                "q": [float(e[3]) for e in els],
                "keff": float(r.get("keff") or 0.0),
                "hop": int(r.get("pow_hop") or -1),
            }
        if d:
            out[_tag(url)] = d
    return out, served


def drop_stale(per_inst, max_lag_s=10.0, hop_rate_hz=195312.5):
    """Drop instances whose element snapshot is far behind the fleet's newest. -> (kept, dropped)

    ⚠️ THIS IS NOT A REFINEMENT, IT IS THE DIFFERENCE BETWEEN A BEAM MAP AND A MIXTURE.
    Measured 2026-08-15 17:55: cx19/0, cx42/0 and cx43/0 served e5a element blocks a
    CONSTANT ~115 million hops (~10 minutes) behind the other nine, which agreed to 0.36 s
    -- byte-identical gains poll after poll (#60's wedged chain, visible here because the
    element export carries pow_hop). A median over instances then mixes a 10-minute-old sky
    with the live one, and the summary jumps whenever the ordering flips: that is a
    candidate mechanism for beam-amplitude scatter that has nothing to do with the array.
    Same discipline as fleet_coherent's window anchor -- drop and NAME, never silently sum.

    ⚠️ The TELEMETRY-fed estimators (comb DLL, cn0_prompt, kcoh) are immune by construction:
    they key on an absolute window index, so a frozen instance simply stops appearing in the
    recent windows. Only this REST-polled table needed the guard. Worth remembering when
    judging which numbers a wedged instance can and cannot corrupt.
    """
    newest = {}
    for tag, d in per_inst.items():
        hops = [v.get("hop", -1) for v in d.values() if v.get("hop", -1) >= 0]
        if hops:
            newest[tag] = max(hops)
    if not newest:
        return per_inst, []
    fleet_newest = max(newest.values())
    lag_hops = max_lag_s * hop_rate_hz
    kept, dropped = {}, []
    for tag, d in per_inst.items():
        lag = fleet_newest - newest.get(tag, -1)
        if tag in newest and lag <= lag_hops:
            kept[tag] = d
        else:
            dropped.append((tag, lag / hop_rate_hz if tag in newest else None))
    return kept, dropped


def gain_table(per_inst, probe_prns, min_keff=8.0):
    """The served table: {prn: {probe, inst: {tag: {keff, amp[], ph[], sig[]}}}}.

    sig is present only where the probes provided a per-(instance, element) noise anchor;
    a table with amp/ph but no sig says exactly what it knows and no more.

    ⚠️ Feed this the OUTPUT OF drop_stale(), not a raw poll -- see there.
    """
    probe_prns = set(int(p) for p in (probe_prns or ()))
    # Per-element noise floor per instance: median over probe PRNs of |u_e|. Median of
    # (typically) 3 probes is coarse per element, but |u| under noise is one-sided and the
    # anchor is only a SIGNIFICANCE scale -- amp/ph are served undivided by it.
    floors = {}
    for tag, d in per_inst.items():
        rows = [v for p, v in d.items() if p in probe_prns and v["keff"] >= min_keff]
        if not rows:
            continue
        n_el = len(rows[0]["u"])
        fl = []
        for e in range(n_el):
            mags = sorted(abs(complex(*r["u"][e])) for r in rows if e < len(r["u"]))
            fl.append(mags[len(mags) // 2] if mags else 0.0)
        floors[tag] = fl
    out = {}
    for tag, d in per_inst.items():
        fl = floors.get(tag)
        for prn, v in d.items():
            if v["keff"] < min_keff:
                continue
            amp, ph, sig = [], [], []
            for e, (re_, im_) in enumerate(v["u"]):
                u = complex(re_, im_)
                q = v["q"][e] if e < len(v["q"]) else 0.0
                amp.append(abs(u) / q if q > 0.0 else 0.0)
                ph.append(cmath.phase(u))
                if fl is not None and e < len(fl) and fl[e] > 0.0:
                    sig.append(abs(u) / fl[e])
            row = out.setdefault(prn, {"probe": prn in probe_prns, "inst": {}})
            ir = {"keff": round(v["keff"], 1),
                  "amp": [float("%.4g" % x) for x in amp],
                  "ph": [round(x, 4) for x in ph]}
            if sig:
                ir["sig"] = [float("%.3g" % x) for x in sig]
            row["inst"][tag] = ir
    return out
