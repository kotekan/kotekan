"""Fleet combines: the shared code discriminator and the cross-node coherent sum.

fleet_dll sums every instance's raw Early/Prompt/Late powers into one code discriminator per PRN
(docs/CHORD_GNSS_SHARED_DLL.md), and apply_presence decides which PRNs it may trim.
fleet_coherent removes the common per-record phase across instances for the deep statistics and
the record-stream carrier rate. fleet_spectrum_aligned and fit_spectrum_delay fit a code delay
from the per-channel prompts, and poll_rf_stats collects receiver-wide stream health.

The combines are per chain (each signal has its own combiners). No function here keeps state
between calls: fleet_dll's hop_hist, the one thing carried across cycles, is owned by the caller.
"""
import cmath
import math

import numpy as _np
import random
import re

from .transport import _get, _log_rl, _now


def sk_url_for(tap_url):
    """Derive the SK-metrics endpoint for a search-tap url:
    .../gnssG_srch_tap -> .../rfi_sk_metrics/sk_metrics_G/sk.
    Returns None if it does not match."""
    m = re.search(r"/gnss(\d)_srch_tap/?$", tap_url)
    if not m:
        return None
    return tap_url[: m.start()] + "/rfi_sk_metrics/sk_metrics_" + m.group(1) + "/sk"


def sk_summary(r):
    """Reduce the per-element spectral-kurtosis array to what a viewer renders. SK ~ 1 is clean;
    RFI drives it away in either direction. The flag band scales with the estimator's
    accumulation (ema_frames): the generalized-SK variance is ~4/M, so sigma ~ sqrt(4/M) and we
    flag |SK-1| > 3 sigma. Elements with no valid measurement report null and are dropped (a
    dead slot is not a jammed element). Keeps the per-element array on the node, as
    poll_rf_stats does."""
    sk = r.get("sk") or []
    valid = [(i, v) for i, v in enumerate(sk) if isinstance(v, (int, float))]
    if not valid:
        return None
    m = r.get("ema_frames") or 256
    sig = math.sqrt(4.0 / max(m, 1))
    lo, hi = 1.0 - 3.0 * sig, 1.0 + 3.0 * sig
    flagged = sum(1 for _, v in valid if v < lo or v > hi)
    worst_i, _ = max(valid, key=lambda iv: abs(iv[1] - 1.0))
    vals = [v for _, v in valid]
    return {
        "sk_n": len(valid),
        "sk_flagged": flagged,
        "sk_max": max(vals),
        "sk_worst": worst_i,
        "sk_med": sorted(vals)[len(vals) // 2],
        "ema_frames": m,
        "sk_lo": round(lo, 3),
        "sk_hi": round(hi, 3),
    }


def parse_drops(metrics_text, gpu):
    """Pull this instance's drop counters out of a node's /metrics dump. buffer-send drops are
    per gnss{gpu} stage; dpdk misses/ring-full are node-level (shared by both GPUs, reported so
    a single congested NIC is visible on every instance rather than nowhere)."""
    out = {}
    srch = re.search(
        r'kotekan_buffer_send_dropped_frame_count\{stage_name="/gnss%d_srch_send"\}\s+([0-9.eE+-]+)'
        % gpu,
        metrics_text,
    )
    telem = re.search(
        r'kotekan_buffer_send_dropped_frame_count\{stage_name="/gnss%d_telem_send"\}\s+([0-9.eE+-]+)'
        % gpu,
        metrics_text,
    )
    if srch:
        out["srch_send_drops"] = int(float(srch.group(1)))
    if telem:
        out["telem_send_drops"] = int(float(telem.group(1)))
    miss = [
        int(float(x))
        for x in re.findall(
            r"kotekan_dpdk_nic_rx_missed_total\{[^}]*\}\s+([0-9.eE+-]+)", metrics_text
        )
    ]
    ring = [
        int(float(x))
        for x in re.findall(
            r"kotekan_dpdk_distributor_ring_full_dropped_packets_total\{[^}]*\}\s+([0-9.eE+-]+)",
            metrics_text,
        )
    ]
    if miss:
        out["dpdk_missed"] = max(miss)
    if ring:
        out["dpdk_ring_full"] = sum(ring)
    return out


def poll_rf_stats(endpoints, lobes_fn, fetch_sk=False, fetch_drops=False):
    """GET <tap>/rf_stats from each endpoint -> {url: per-instance stream health}: power and
    clip lobes, plus spectral-kurtosis RFI (fetch_sk) and drop counters (fetch_drops). All of it
    is per-instance, receiver-wide health, not per satellite, so it goes to the publisher's
    get_rf (the viewer's stream-health panel), not through the gather.

    Each tap serves clip fraction and power per monitored channel and per absolute element; this
    reduces them to what a viewer renders and a log carries, and leaves the per-element arrays
    on the node.

    An instance whose monitor is not armed reports enabled=false. That is recorded as state
    "off", never published as zeros, so a dark panel reads as "not armed" rather than "no RFI";
    an unreachable instance is a third state, also not zero. SK and drops are best-effort: a
    failed fetch leaves the field absent and never changes the rf state or drops the instance.
    """
    metrics_cache = {}  # host_base -> /metrics text, fetched once per host per poll

    def _metrics(host_base):
        if host_base not in metrics_cache:
            try:
                import urllib.request

                with urllib.request.urlopen(host_base + "/metrics", timeout=6) as resp:
                    metrics_cache[host_base] = resp.read().decode("utf-8", "replace")
            except Exception:
                metrics_cache[host_base] = ""
        return metrics_cache[host_base]

    out = {}
    for url in endpoints:
        try:
            r = _get("%s/rf_stats" % url)
        except Exception as e:
            _log_rl("rf-stats-%s" % url, "rf stats: %s unreachable (%s)" % (url, e))
            out[url] = {"state": "unreachable"}
            continue
        if not r.get("enabled"):
            out[url] = {"state": "off"}
            continue
        ec = r.get("elem_clip") or []
        ep = r.get("elem_power") or []
        ep_live = [
            p for p in ep if p > 0.0
        ]  # unpopulated element slots read exactly 0, so a
        # median that included them would read 0 and hide the signal
        summ = {
            "state": "on",
            # freq_ids: absolute channel identity from the frame's chord metadata, parallel to
            # chans. The tap serves [] until it has measured a chord-metadata frame, and
            # rf_lobes then leaves the rows unnamed rather than guessing a band.
            "lobes": lobes_fn(
                r.get("chans") or [],
                r.get("power") or [],
                r.get("clip_lo") or [],
                r.get("clip_hi") or [],
                r.get("freq_ids") or [],
            ),
            "elem_clip_max": max(ec) if ec else None,
            "elem_clip_worst": (
                max(range(len(ec)), key=lambda i: ec[i]) if ec else None
            ),
            "elem_power_med": (sorted(ep_live)[len(ep_live) // 2] if ep_live else None),
            "n_live_elem": len(ep_live) if ep else None,
            "passes": r.get("passes"),
            "cost_ms": r.get("cost_ms"),
            "age_s": r.get("age_s"),
            "fpga_seq": r.get("fpga_seq"),
        }
        if fetch_sk:
            sk_url = sk_url_for(url)
            if sk_url:
                try:
                    summ["sk"] = sk_summary(_get(sk_url))
                except Exception as e:
                    _log_rl("sk-%s" % url, "rfi SK: %s unreachable (%s)" % (sk_url, e))
        if fetch_drops:
            m = re.search(r"^(https?://[^/]+)/gnss(\d)_srch_tap", url)
            if m:
                d = parse_drops(_metrics(m.group(1)), int(m.group(2)))
                if d:
                    summ["drops"] = d
        out[url] = summ
    return out


def live_instances(
    served,
    hop_hist,
    now,
    anchor_utc=None,
    hops_per_sec=None,
    ahead_s=60.0,
    frozen_s=10.0,
):
    """Which instances' rows may enter the fleet sum. -> (excluded {url: reason}).

    `served` is {url: newest pow_hop this poll} (-1 = none yet). `hop_hist` is {url: (hop,
    first_seen_at_that_hop)}, carried across calls and updated here.

    The fleet hop is the newest instance's, which is right for a laggard (only ever behind) but
    not across an F-engine re-base: the counter restarts at 0, and a node not yet relaunched
    still serves its pre-re-base rows, far ahead of the live ones. Those would win the max, put
    the live instances outside hop_window, and make the published fleet hop the old session's.

    Two exclusions, each refusing to blind the fleet:
      * ahead: with the anchor known, a hop past what the F-engine has counted by `now` plus
        `ahead_s` is from a previous session. Applied only while at least one instance is not
        ahead; if every instance is, the anchor is what is wrong.
      * frozen: a served hop unchanged for `frozen_s` while another instance advanced inside
        that time. Any change counts as advancing, so a relaunched node's backwards jump to a
        small hop clears it at once. When nobody advances (an outage, a paused F-engine) nobody
        is excluded.
    """
    excluded = {}
    for url, hop in served.items():
        if hop < 0:
            continue
        old = hop_hist.get(url)
        hop_hist[url] = (hop, old[1] if (old is not None and old[0] == hop) else now)
    live = {u: h for u, h in served.items() if h >= 0}
    if anchor_utc and hops_per_sec:
        bound = (now - anchor_utc + ahead_s) * hops_per_sec
        ahead = {u for u, h in live.items() if h > bound}
        if ahead and len(ahead) < len(live):
            for u in ahead:
                excluded[u] = "ahead"
    advancing = {u for u in live if now - hop_hist[u][1] < frozen_s}
    if advancing:
        for u in live:
            if u not in advancing and u not in excluded:
                excluded[u] = "frozen"
    return excluded


def fleet_dll(
    endpoints,
    hop_window,
    min_instances,
    k_sigma,
    q_fallback,
    deep_gate_prns=None,
    deep_gate_margin=3.0,
    probe_prns=None,
    src_hops=None,
    admit_displaced=None,
    *,
    hop_hist,
    anchor_utc=None,
    hops_per_sec=None,
    now=None,
):
    """Sum the fleet's raw Early/Prompt/Late powers per PRN -> one full-bandwidth discriminator.

    The F-engine comb spreads each band across the nodes and each GPU holds seven 195.3 kHz
    channels, so one tracker instance correlates 1.37 MHz of the 20.46 MHz L5 lobe (6.7%,
    -11.8 dB). No instance can do better, because the rest of the bandwidth is not on its
    machine; the sum across instances recovers it.

    Summing is legitimate because the DLL discriminator (E-L)/(E+L) is non-coherent: it is built
    from powers, and powers add. It needs no phase alignment and no sample-level sync between
    nodes, only the same integration window, which pow_hop identifies exactly (an absolute
    F-engine hop index shared by every node, so equality means the same sky). A few numbers per
    PRN per window ride the existing REST poll.

    It must sum the raw powers, not dll_disc: (SUM E - SUM L)/(SUM E + SUM L) is not any
    function of the per-instance (E-L)/(E+L).

    What the sum buys is a lower bar, not a higher statistic. Summing K instances does not raise
    q = 2P/(E+L): every tap's mean scales by K, so the ratio is unchanged. Each summed power has
    K times the mean and K times the variance, so q's spread falls as 1/sqrt(K) and a
    signal-free PRN's q tightens onto 1.0. The presence bar must therefore be a noise floor
    measured at the current fleet size (apply_presence), never a fixed q: the same q can sit
    inside the noise for one instance and clear it for eight.

    Instances that live_instances excludes still report their newest hop into src_hops but
    contribute no rows.

    Returns {prn: {disc, q, p_pow, hop, n_src, n_chan, coh_row, coh_src, coh_quad}}, plus the
    presence fields apply_presence adds (q_floor, q_med, q_sigma, p_floor, p_med, p_floor_src,
    present, present_gate, and on the paths that set them n_probe_q, off_chips, pedestal,
    deep_gate_snr, deep_gate_floor), for PRNs with >= min_instances instances inside hop_window
    of the newest. PRNs below that are absent, so the caller falls back to the single-combiner
    discriminator and a partly-down fleet degrades rather than stalls.
    """
    if not endpoints:
        return {}
    # prn -> list of (hop, e, p, l, n_chan). Unreachable instances are skipped, not fatal:
    # a node down for maintenance must cost sensitivity, never the loop.
    rows = {}
    best_coh = (
        {}
    )  # prn -> ((deep_snr, amp_snr), row, url): strongest instance's coherent view
    coh_cleared = (
        {}
    )  # prn -> floor-cleared per-instance deep_snrs, for the quadrature fallback
    polled = {}
    for url in endpoints:
        try:
            polled[url] = _get("%s/get_status" % url) or []
        except Exception as e:
            _log_rl("fleet-dll-%s" % url, "fleet DLL: %s unreachable (%s)" % (url, e))
    # Judge each instance before any of its rows count (live_instances). Excluded instances still
    # report their hop to src_hops, so the axis and stall instruments keep seeing them.
    served = {
        u: max([int(r.get("pow_hop", -1)) for r in g if isinstance(r, dict)] or [-1])
        for u, g in polled.items()
    }
    # the cycle's frozen clock (_now): wall time live, the recorded instant in replay
    excluded = live_instances(
        served,
        hop_hist,
        _now() if now is None else now,
        anchor_utc=anchor_utc,
        hops_per_sec=hops_per_sec,
    )
    for u, why in sorted(excluded.items()):
        _log_rl(
            "fleet-dll-excl-%s-%s" % (why, u),
            "fleet DLL: %s EXCLUDED (%s) at hop %d -- %s"
            % (
                u,
                why,
                served[u],
                "past what the F-engine has counted: a previous session's row"
                if why == "ahead"
                else "not advancing while its peers do: pre-outage state or a wedged chain",
            ),
            every_s=60.0,
        )
    for url, got in polled.items():
        for r in got:
            hop = int(r.get("pow_hop", -1))
            e = float(r.get("e_pow", 0.0))
            l = float(r.get("l_pow", 0.0))
            if hop < 0 or (e + l) <= 0.0:
                continue  # no metadata on the record frames, or this PRN is not being despread
            prn = int(r["prn"])
            # The newest hop this instance served, from the parse already done here: the
            # per-PRN aggregate below discards the per-instance axis, which is the one a wedged
            # instance shows on.
            if src_hops is not None and hop > src_hops.get(url, -1):
                src_hops[url] = hop
            if url in excluded:
                continue
            rows.setdefault(prn, []).append(
                (hop, e, float(r.get("p_pow", 0.0)), l, float(r.get("n_chan", 0.0)))
            )
            # Best-of for the coherent statistics. deep_amplitude, deep_snr and coherence_s come
            # from each instance's own deep integration and cannot be merged here: this combine
            # sums powers, which is what makes it phase-blind and cheap. So carry the strongest
            # instance's coherent row rather than a fleet number that would need cross-node
            # phase alignment. Ranked on (deep_snr, amp_snr), so a chain that never certifies a
            # deep still reports its best view.
            key = (float(r.get("deep_snr", 0.0)), float(r.get("amp_snr", 0.0)))
            if prn not in best_coh or key > best_coh[prn][0]:
                best_coh[prn] = (key, r, url)
            # Quadrature fallback input. sqrt(sum snr_i^2) over the floor-cleared instances is
            # the significance of their incoherent power sum; the publisher serves it as
            # deep_snr when fleet_coherent has no floor-cleared result for the PRN, so the series
            # stays near the fleet level instead of dropping to the best single instance each
            # time the fleet gate flickers. It buys continuity, not a quieter estimate.
            # The floor-cleared gate is load-bearing: a floored deep reads ~7 on pure noise, so
            # an ungated quadrature over 12 instances would read sqrt(12)*7 ~ 24 and manufacture
            # a detection. coherence_s > 0 is the rule the broker's sig_of() applies.
            if (
                float(r.get("coherence_s", 0.0) or 0.0) > 0.0
                and float(r.get("deep_snr", 0.0)) > 0.0
            ):
                coh_cleared.setdefault(prn, []).append(float(r["deep_snr"]))
    out = {}
    for prn, rs in rows.items():
        # Instances free-run, so their windows do not end on the same hop: take the newest
        # window and admit everything within hop_window of it (integer arithmetic on the
        # F-engine's own counter). The code offset against the replica moves ~0.003 chips/s, so
        # rows a few seconds apart differ by about 0.01 chips; older rows are a fault and stay
        # out of the sum.
        newest = max(r[0] for r in rs)
        use = [r for r in rs if newest - r[0] <= hop_window]
        if len(use) < min_instances:
            continue
        E = sum(r[1] for r in use)
        P = sum(r[2] for r in use)
        L = sum(r[3] for r in use)
        if E + L <= 0.0:
            continue
        bc = best_coh.get(prn)
        _cl = coh_cleared.get(prn) or []
        out[prn] = {
            "disc": (E - L) / (E + L),
            # the strongest instance's coherent row and which instance (url) it came from
            "coh_row": bc[1] if bc else None,
            "coh_src": bc[2] if bc else None,
            # quadrature over floor-cleared instances (see coh_cleared above); None when
            # fewer than 2 cleared, and the publisher then keeps the single-instance value
            "coh_quad": (sum(x * x for x in _cl) ** 0.5, len(_cl))
            if len(_cl) >= 2
            else None,
            # q = 2P/(E+L): 1.0 with no peak (all three taps at the noise power), 4.0 at a
            # clean lock with 0.5-chip spacing. The combiner builds the three powers the same
            # way, which makes q comparable across instances and meaningful once summed.
            "q": 2.0 * P / (E + L),
            # kept raw: presence also needs the summed prompt power, because a ratio answers
            # "am I on the peak", never "is there signal here"
            "p_pow": P,
            "hop": newest,
            "n_src": len(use),
            "n_chan": sum(r[4] for r in use),
        }

    # The floors, the presence verdict and the deep gate live in apply_presence() so the comb
    # arm (combdll.fleet_dll_comb) reaches the same verdict from the same numbers; with two
    # copies of the policy, comparing the arms would compare the copies.
    return apply_presence(
        out,
        k_sigma,
        q_fallback,
        probe_prns=probe_prns,
        deep_gate_prns=deep_gate_prns,
        deep_gate_margin=deep_gate_margin,
        admit_displaced=admit_displaced,
    )


def epl_decompose(q, disc, spacing=0.5):
    """(offset_chips, pedestal) from the two numbers on every DLL row.

    q and disc determine the tap powers: E/P = (1+disc)/q, L/P = (1-disc)/q. A fade scales all
    three taps together (ratios hold, the noise pedestal rises); a code offset moves power
    between taps (E/P falls, L/P climbs past 1, pedestal flat). The two land in different
    corners of the (E/P, L/P) plane, so this closed-form solve separates what q alone hides:
    q ~ 1 means "off-peak" or "weak", and a gate that reads it as weak throws away the
    displaced-but-strong satellite the DLL exists to rescue, whose offset suppresses q by
    construction.

    Only the offset d is scanned (+-0.9 chips in 0.001 steps); the pedestal follows from the
    E-L difference exactly:
        n = (R(d-s) - R(d+s)) / (l - e) - R(d)
    and the E+L equation supplies the residual that picks d. R is the squared triangular
    correlation and s the tap spacing. offset > 0 = the peak is late of the prompt (same sign
    as the trim that corrects it).

    The fit does not certify the row. Two observables, two parameters: the solve is near-exact
    for any input including pure noise (a probe at the floor returns a confident large offset).
    Only evidence of signal admits a row (apply_presence: the offset-blind deep detection or
    prompt power), never this fit. E == P == L returns (0.0, inf), the no-information answer.
    """
    if q <= 0.0:
        return 0.0, float("inf")
    e, l = (1.0 + disc) / q, (1.0 - disc) / q

    def acf(x):
        a = max(0.0, 1.0 - abs(x))
        return a * a

    dl = l - e
    if abs(dl) < 1e-6:
        if e >= 1.0 - 1e-9:
            return 0.0, float("inf")
        return 0.0, max(0.0, (e - acf(spacing)) / (1.0 - e))
    best = None
    for i in range(-900, 901):
        d = i / 1000.0
        n = (acf(d - spacing) - acf(d + spacing)) / dl - acf(d)
        if n < 0.0:
            continue
        P = acf(d) + n
        if P <= 0.0:
            continue
        err = ((acf(d + spacing) + n) / P - e) ** 2 + (
            (acf(d - spacing) + n) / P - l
        ) ** 2
        if best is None or err < best[0]:
            best = (err, d, n)
    if best is None:
        return 0.0, float("inf")
    return best[1], best[2]


def displaced_chain_event(admitted, browned_out, same_chips=0.1, n_same=3):
    """The displaced admissions of one cycle that are a chain event, with the reason.

    `admitted` is [(prn, off_chips, ...)], what the displaced gate would admit this cycle.
    Returns (refused, reason). N satellites do not step by one offset in one cycle: when they
    all report the same displacement, the chain moved its clock or lost its power (a transit's
    null crossing can do this to every PRN on a chain, and admitting them re-seeds each one off
    its peak, where the leak-bounded DLL then holds it). Likewise, while the chain is browned
    out a displacement reports the chain's lost power, not a satellite that moved. Either way
    the admission is refused and the q/prompt verdict stands; `n_same` PRNs within
    `same_chips` of one another is the chain bar.

    A row within `same_chips` of its peak is not displaced and has nothing to attribute to the
    chain: the offset-blind fold admits dim on-peak satellites at off ~0 in numbers, and
    re-seeding those is harmless, so they are never refused.
    """
    displaced = [a for a in admitted if abs(a[1]) > same_chips]
    if not displaced:
        return [], None
    if browned_out:
        return displaced, "BROWNOUT open"
    same = [
        a
        for a in displaced
        if sum(1 for b in displaced if abs(b[1] - a[1]) <= same_chips) >= n_same
    ]
    if same:
        return same, "same offset"
    return [], None


def apply_presence(
    out,
    k_sigma,
    q_fallback,
    probe_prns=None,
    deep_gate_prns=None,
    deep_gate_margin=3.0,
    admit_displaced=None,
):
    """Floors, the presence verdict and the deep gate, in place, on a fleet_dll-shaped dict.

    Shared by fleet_dll and the comb arm (combdll.fleet_dll_comb), so both judge with one
    policy. It reads only the row fields (q, disc, p_pow, coh_row) and the probe population,
    never how a row's powers were formed.

    The noise reference is the probe PRNs (probe_prns, signal-free by construction), never the
    tracked satellites, whose median on CHORD is a signal level. With fewer than 3 probes
    reporting q, or fewer than 2 reporting p_pow, every row is refused: present False,
    present_gate UNANCHORED.
    """
    # The presence bars are measured every cycle from the noise probes, never assumed: q's noise
    # value and spread move with the number of contributing instances and with the EMA length,
    # so any constant bar suits one fleet size only. Both q and prompt power are on-peak
    # statistics, so on their own they refuse a satellite whose tap has slid off the peak; the
    # displaced admission and the deep gate below are the offset-blind routes in.
    # _floor: median and MAD-sigma of a population, and the bar max(median + k*sigma,
    # median + lo_margin); None with fewer than 8 values.
    def _floor(vals, k, lo_margin):
        s = sorted(vals)
        if len(s) < 8:  # too few rows to characterise a population
            return None, None, None
        m = s[len(s) // 2]
        mad = sorted(abs(x - m) for x in s)[len(s) // 2]
        sg = 1.4826 * mad
        # If MAD collapses (a degenerate population -- every instance reporting the same window
        # of zeros) the bar would collapse with it, so keep a small absolute margin too.
        return m, sg, max(m + k * sg, m + lo_margin)

    # q floor from the probes only. Probes are signal-free, so their q is the no-peak value.
    # The tracked satellites' median reads ~1.0 only while most of them are off-peak; with
    # most rows locked it would sit near the lock value and shut out everything working.
    _probe_q = [v["q"] for k, v in out.items() if probe_prns and k in probe_prns]
    if len(_probe_q) >= 3:
        q_med, q_sigma, q_floor = _floor(_probe_q, k_sigma, 0.05)
        if q_med is None:  # _floor needs 8; do it directly for a few
            _s = sorted(_probe_q)
            q_med = _s[len(_s) // 2]
            _mad = sorted(abs(x - q_med) for x in _probe_q)[len(_probe_q) // 2]
            q_sigma = 1.4826 * _mad
            q_floor = max(q_med + k_sigma * q_sigma, q_med + 0.05)
    else:
        # Fewer than 3 probes: the noise q is unmeasured and the chain cannot judge (refused
        # below). There is deliberately no fallback to the tracked population.
        q_med = q_sigma = q_floor = None
    # Prompt-power floor. p_pow spans orders of magnitude between satellites, so the bar is
    # multiplicative against N, the probes' median p_pow: present needs p_pow >= max(k_sigma,
    # 1) x N. A ratio to a measured median holds up with only a handful of probes, where a MAD
    # would not. The tracked population cannot supply N: its p_pow has no bimodality to locate
    # the noise, and on CHORD most tracked rows are signal, so its median is a signal level.
    _probe_p = [
        v["p_pow"]
        for k, v in out.items()
        if probe_prns and k in probe_prns and v.get("p_pow")
    ]
    if len(_probe_p) >= 2:
        _pm = sorted(_probe_p)[len(_probe_p) // 2]
        p_med, p_floor = _pm, _pm * max(k_sigma, 1.0)
        for v in out.values():
            v["p_floor_src"] = "probes:%d" % len(_probe_p)
    else:
        # Fewer than 2 probes: no noise level and no p bar, and no fallback to the tracked
        # population; the chain refuses below.
        p_med, p_floor = None, None
        for v in out.values():
            v["p_floor_src"] = "UNANCHORED"
    # Refusal: a chain that cannot measure its noise floor admits nobody, so nothing is armed
    # or trimmed. That is the conservative direction (a trim not applied is recoverable), and
    # the fleet DLL stage logs PRESENCE UNANCHORED so the refusal is not read as an empty sky.
    # It is unconditional; there is no tracked-population bar to fall back to.
    if len(_probe_q) < 3 or p_floor is None:
        for v in out.values():
            v["q_med"], v["q_sigma"] = q_med, q_sigma
            # q_floor stays numeric (q_fallback when the noise q is unmeasured): consumers
            # format it with %.2f, so a refusal must still publish a well-formed row.
            v["q_floor"] = q_fallback if q_med is None else q_floor
            v["p_med"], v["p_floor"] = p_med, p_floor
            v["present"] = False
            v["present_gate"] = "UNANCHORED"
            v["n_probe_q"] = len(_probe_q)
        return out
    # Rows the displaced admission let in this pass, for displaced_chain_event and the log.
    _admitted = []
    # items(): `out` is keyed by PRN and the rows do not carry their PRN.
    for _prn, v in out.items():
        v["q_med"], v["q_sigma"] = q_med, q_sigma
        v["q_floor"] = q_floor
        v["p_med"], v["p_floor"] = p_med, p_floor
        # Present needs both bars. q is a ratio and asks "is there a peak under the tap"; it
        # is scale-free, so a faint satellite with a clean peak passes while a bright one whose
        # tap has slid does not (brightness varies widely between satellites for reasons
        # unrelated to tap placement). q alone cannot tell a faint real peak from a noise
        # realisation that happens to be centred, so p_pow must also clear the
        # probe-referenced floor: "is there anything there at all".
        v["present"] = v["q"] >= v["q_floor"] and v["p_pow"] >= p_floor
        v["present_gate"] = "q+p:probes"
        # Displaced admission (--presence-admit-displaced). q ~ 1 means "off-peak" or "weak",
        # and both bars read it as weak, so the displaced-but-strong satellite the DLL exists
        # to rescue is refused. That latches: the offset suppresses q and prompt power by
        # construction, so the fault disables its own cure. epl_decompose separates the two:
        # a displaced row has a low pedestal (its power went to a neighbouring tap, not to
        # noise), a weak row a large one. A row that failed the bars is re-admitted when
        #   * there is evidence it is really there: deep_snr >= deep_margin x deep_floor from
        #     the combiner's offset-blind fold (it re-searches rate and phase, so it sees the
        #     satellite wherever the tap sits; deep_floor is the combiner's own rectification
        #     level), or p_pow >= p_floor. Prompt power is accepted but not required, since
        #     the offset being rescued suppresses it; and
        #   * pedestal <= pedestal_max and |offset| <= off_max_chips, which refuse a centred
        #     noise realisation (large pedestal) and a satellite beyond the DLL's pull-in
        #     (large offset; that one needs the search, not the loop).
        # Reached only under the probe-anchored gate. Admissions are logged (PRESENCE
        # +displaced) with the evidence that carried each.
        if admit_displaced is not None and not v["present"]:
            _c = v.get("coh_row") or {}
            _ds = float(_c.get("deep_snr", 0.0) or 0.0)
            _dfl = float(_c.get("deep_floor", 0.0) or 0.0)
            _blind = (
                _dfl > 0.0 and _ds >= admit_displaced.get("deep_margin", 3.0) * _dfl
            )
            _bright = v["p_pow"] >= p_floor
            if _blind or _bright:
                _off, _ped = epl_decompose(v["q"], v["disc"])
                v["off_chips"], v["pedestal"] = _off, _ped
                if (
                    _ped <= admit_displaced["pedestal_max"]
                    and abs(_off) <= admit_displaced["off_max_chips"]
                ):
                    v["present"] = True
                    # which evidence admitted it: prompt power, or the deep fold alone
                    v["present_gate"] = (
                        "q+p:probes+disp" if _bright else "q+deep:probes+disp"
                    )
                    _admitted.append((_prn, _off, _ped, v["p_pow"] / p_floor, _bright))

        # Deep gate (deep_gate_prns: the --dll-deep-gate PRNs plus those the search recently
        # detected, or True for all). Prompt power is suppressed when the tap is off-peak, so a
        # prompt-gated satellite that drifts off the peak fails the gate, is never trimmed, and
        # stays off-peak: the loop cannot pull in from the shoulder. Another tap ratio would not fix
        # that; every tap statistic is on-peak-biased. The deep fold re-searches rate and
        # phase, so deep_snr detects the satellite wherever the tap sits (the same property
        # makes it wrong for a C/N0). For a PRN in the set with a combiner floor, present =
        # deep_snr >= deep_gate_margin x deep_floor, overriding the q/prompt and displaced
        # verdicts. Opt-in per PRN because deep_snr can fire on noise and this gate has no
        # pedestal or offset test.
        if deep_gate_prns:
            for prn, v in out.items():
                if deep_gate_prns is not True and prn not in deep_gate_prns:
                    continue
                c = v.get("coh_row") or {}
                ds = float(c.get("deep_snr", 0.0) or 0.0)
                fl = float(c.get("deep_floor", 0.0) or 0.0)
                # Needs a real floor: with none there is no bar, and defaulting to present
                # would trim on noise. The q/prompt verdict stands instead.
                if fl <= 0.0 or ds <= 0.0:
                    continue
                v["present"] = ds >= deep_gate_margin * fl
                v["present_gate"] = "deep"
                v["deep_gate_snr"], v["deep_gate_floor"] = ds, fl
    _refused, _why = displaced_chain_event(
        _admitted, admit_displaced is not None and admit_displaced.get("browned_out")
    )
    if _refused:
        # Revert only rows the displaced gate still owns: the deep gate above may have
        # re-judged a PRN in its set since, and its verdict stands.
        for _prn, _off, _ped, _r, _bright in _refused:
            if out[_prn]["present_gate"].endswith("+disp"):
                out[_prn]["present"] = False
                out[_prn]["present_gate"] = "q+p:probes"
        _admitted = [a for a in _admitted if a not in _refused]
        _offs = sorted(a[1] for a in _refused)
        # One line per cycle, not per PRN: the refusal is the statement that this was one
        # event. Keyed by reason so a brownout's repeat does not hide a same-offset event.
        _log_rl(
            "presence-disp-refused-" + _why,
            "PRESENCE -displaced: %s -- %d PRN(s) at off %+.3f..%+.3f REFUSED, a chain "
            "event rather than N detections: %s"
            % (
                _why,
                len(_refused),
                _offs[0],
                _offs[-1],
                " ".join("PRN %s %+.3f" % (a[0], a[1]) for a in _refused),
            ),
            every_s=30.0,
        )
    if _admitted:
        _log_rl(
            "presence-disp",
            "PRESENCE +displaced: %s -- admitted by the displaced gate, which the q/prompt "
            "gate had refused. (off = tap offset in chips, ped = noise pedestal; 'p' means "
            "prompt power carried it, 'deep' means the offset-blind fold did.)"
            % "; ".join(
                "PRN %s off %+.3f ped %.3f p/floor %.1f via %s"
                % (p2, o, pe, r, "p" if b else "deep")
                for p2, o, pe, r, b in sorted(_admitted, key=lambda x: -(x[3] or 0))[:8]
            ),
            every_s=120.0,
        )
    return out


def _coherent_sum(a):
    """Coherent sum SNR / amplitude / coherence fraction of a per-record complex series.

    snr and amplitude are the statistic of gnss::coherent_sum (gnssChannelizedDespread.cpp):
    rotate the sum onto the real axis and take the orthogonal component as the noise estimate.
    Keep the two identical: the broker's fleet number and a combiner's local number are compared
    against each other and against the same floors, and two SNR conventions would make the
    floors meaningless.

    Returns (snr, amplitude, coh_frac) where coh_frac = |sum|/sum|.|, a coherence measure that
    does not depend on how the series is chopped; snr scales with sqrt(N) and coh_frac does not.
    """
    n = len(a)
    if n < 2:
        return 0.0, 0.0, 0.0
    # Vectorised on purpose: callers pass numpy rows, and iterating a numpy array element by
    # element is slower than iterating a list of complex, because each access boxes a new
    # scalar object.
    _a = _np.asarray(a, dtype=complex)
    sr = float(_a.real.sum())
    si = float(_a.imag.sum())
    mag = math.hypot(sr, si)
    if mag <= 0.0:
        return 0.0, 0.0, 0.0
    cr, ci = sr / mag, -si / mag  # e^{-i arg(sum)}
    _im = _a.real * ci + _a.imag * cr  # Im(x * rot)
    denom = math.sqrt(float((_im * _im).sum()))
    tot_abs = float(_np.abs(_a).sum())
    # Degenerate-residual guard, mirroring gnss::residual_snr in gnssChannelizedDespread.cpp
    # (keep the two identical). `denom > 0.0` alone catches only an exactly-zero residual; a
    # series aligned to within double precision leaves ~1e-17 and would return ~1e17. Both
    # callers derotate by a unit-modulus reference (s * conj(r)/|r|, and x * conj(rest)/|rest|
    # per instance), so their products are nearly real by construction and the orthogonal
    # component is what the derotation removed. The floor is far above machine epsilon and
    # far below anything physical (a 60 dB record has an orthogonal residual ~1e-3 of mag).
    # Fail closed (0.0), matching n < 2.
    degen = mag * math.sqrt(n) * 1e-12
    return (
        (mag / denom if denom > degen else 0.0),
        mag / n,
        (mag / tot_abs if tot_abs > 0 else 0.0),
    )


def _fit_record_rate(hops, tot, hop_rate_hz):
    """Record-stream carrier rate: the phase slope of the fleet-summed per-record series.

    `hops` are the records' hop indices and `tot` the aligned fleet sum over them. This fits
    the frequency of the series itself, unlike the deep fold's rate argmax (deep_rate_full_hz),
    which maximises the fold's amplitude over a coarse grid. fleet_coherent returns it as
    rate_hz: the next cycle's KCOH fold centres on it (ctx.kcoh_rates), and the publisher
    serves it as rec_rate_hz beside deep_rate_full_hz, which remains the rrate state's coarse
    feed. It reuses the records the caller already fetched, so it adds no GET and replay stays
    ordered.

    The records lie on a lattice of hop indices, so the periodogram is an FFT of the
    zero-filled lattice: a missing hop contributes z = 0, exactly as omitting its term from a
    direct DFT would. Keep it an FFT: a dense matched-filter bank over the frequency grid is
    O(N*M) where this is O(N log N).

    Returns (rate_hz, rate_sigma_hz), or (None, None) without hop_rate_hz or when the series is
    too short, not on one lattice, or has an empty band. The sigma is half the split-half
    disagreement.
    """
    n = len(hops)
    if n < 2 or not hop_rate_hz:
        return None, None
    _h0 = int(hops[0])
    _hi = [int(h) - _h0 for h in hops]
    _span_s = _hi[-1] / float(hop_rate_hz)
    if _span_s <= 0.0:
        return None, None
    # Search band: +-Nyquist of the average record spacing, span/(n-1). Without gaps that is
    # the lattice Nyquist and the band mask is a no-op; with gaps it is narrower than the
    # lattice allows, which is the conservative choice.
    _fmax = 0.5 / (_span_s / (n - 1))
    # The lattice. Decline rather than mis-grid if the hops are not on one.
    _step = min(_hi[i + 1] - _hi[i] for i in range(n - 1))
    if _step <= 0 or any(x % _step for x in _hi):
        return None, None
    _idx = [x // _step for x in _hi]
    _L = _idx[-1] + 1
    _dts = _step / float(hop_rate_hz)
    # FFT length: a power of two, at least 4096 (~4000 bins across the band when there are no
    # gaps) and at least 4x the lattice length, zero-padded.
    _N = 4096
    while _N < 4 * _L:
        _N <<= 1
    _g = _np.fft.fftshift(_np.fft.fftfreq(_N, _dts))
    _band = _np.abs(_g) <= _fmax * (1.0 + 1e-12)
    if not _band.any():
        return None, None
    _zf = _np.zeros(_L, dtype=complex)
    _zf[_idx] = _np.asarray(tot, dtype=complex)

    def _peak(lo, hi, refine):
        # |DFT| is invariant to the time origin, so a slice needs no re-referencing.
        _p = _np.abs(_np.fft.fftshift(_np.fft.fft(_zf[lo:hi], _N)))
        _p = _np.where(_band, _p, 0.0)
        _k = int(_np.argmax(_p))
        _f = float(_g[_k])
        if refine and 0 < _k < _N - 1:
            _a, _b, _c = _p[_k - 1], _p[_k], _p[_k + 1]
            _d = _a - 2 * _b + _c
            if _d != 0.0:
                _f += 0.5 * (_a - _c) / _d * (_g[1] - _g[0])
        return _f

    # Split-half sigma: fit each half independently on the same grid; the sigma is half their
    # disagreement.
    _m = n // 2
    return (
        _peak(0, _L, True),
        abs(_peak(0, _idx[_m - 1] + 1, False) - _peak(_idx[_m], _L, False)) / 2.0,
    )


def fleet_coherent(
    endpoints,
    min_instances,
    min_records,
    prns=None,
    null_trials=1,
    floor_margin=3.0,
    seed=0,
    max_age_hops=1 << 20,
    hop_rate_hz=None,
    source=None,
):
    """Cross-node coherent combine: the per-record sky phase, and the deep folds it unlocks.

    Every instance's records carry a common per-record phase error: coherent across instances
    (same sky, independent thermal noise) but nearly white in time. A within-node temporal
    estimator cannot remove it (a record's neighbours do not predict it), while the other
    instances measure it at once; correcting each instance against the others brings its deep
    fold toward the thermal bound.

    It must run here, not on a node. Each instance has the SNR to measure the phase, but
    derotating a record by a phase computed from that same record subtracts its own noise,
    rectifies it, and inflates every statistic downstream. It needs an independent estimate of
    a common quantity, and the only independent view of the same sky is on the other nodes.
    (The element axis is the other independent axis; see gnssElemCal.)

    The two estimators, both leave-one-out by construction:
      * per instance: derotate instance i's records by the phase of the sum of the others. The
        rotation is then independent of instance i's noise, so noise cannot align with itself.
      * fleet total, a one-way split: sum the instances into two disjoint halves S and R and
        derotate S by R's phase; S is integrated, R is only the reference. Not the symmetric
        S*e^{-i arg R} + R*e^{-i arg S}, and not a derotation by the fleet's own sum: both are
        unbiased in the mean and both destroy the variance estimate every significance is
        built on. Fleet-sum derotation puts each record on the positive real axis outright;
        the symmetric pair expands to (r_S + r_R) cos d + i (r_S - r_R) sin d, whose imaginary
        part vanishes for balanced halves. coherent_sum then finds no orthogonal component and
        the SNR explodes. One-way keeps arg(S) - arg(R) random under noise. The price is that
        half the aperture is spent on the reference: ~sqrt(2) below a full-aperture coherent
        sum, still ~sqrt(K/2) above a single instance.

    No rate search is needed: every instance despreads against the same broker seed, so their
    residual carrier rates are identical. The alignment cross-product cancels the common rate,
    and the leave-one-out derotation removes the common ramp with the sky phase. The combine
    is O(records x instances) per PRN.

    Alignment: instances have arbitrary constant phase offsets (different combs, different NCO
    history), so each is rotated onto the most energetic instance by arg(<A_i conj(A_ref)>)
    before anything is summed; an unaligned fleet sum is meaningless. Each instance is weighted
    by its mean replica energy, the MRC weight (A = G/E has variance ~1/E).

    The floor is measured, not assumed (as for fleet_dll's q_floor). Each call re-runs the same
    math null_trials times on a null built by shuffling each instance's records independently
    in time: real amplitudes, energies and instance count, but no common per-record phase,
    which is what this estimator locks onto. A detection must clear floor_margin x the largest
    null value. A shuffle, not a roll: a roll shifts the shared linear carrier ramp by a
    constant and leaves it intact, so a rolled null would measure the rate-search floor
    instead of this estimator's.

    `source` = (got, fleet_now) from the telemetry gather replaces the /get_records poll, and
    `endpoints` is then ignored. A PRN whose newest record trails the fleet's newest by more
    than max_age_hops is dropped.

    Returns {prn: {deep_snr, deep_amplitude, coh_frac, n_src, n_rec, align, per_inst,
    best_inst_snr, rate_hz, rate_sigma_hz, dropped, floor, present}} for PRNs seen by >=
    min_instances instances over >= min_records common hops. `present` is the floor-cleared
    flag: the caller publishes the coherent numbers only for those and keeps the
    single-instance value otherwise, so a partly-down fleet degrades rather than stalls and a
    noise-only PRN cannot manufacture a fleet detection.
    """
    # url -> prn -> {hop: (A, energy)}. Unreachable instances are skipped, never fatal.
    got = {}
    # The fleet's clock, taken across every satellite an instance serves, including ones this
    # call was not asked about: deriving it from the requested PRNs alone would make "now"
    # depend on one of them being current, which fails when a set of satellites goes down
    # together.
    fleet_now_all = 0

    # `source` (--telem-coherent) is (got, fleet_now) already in this function's shape, from
    # the frame-synced telemetry gather (telem.py coherent_source, which forms
    # A = (P_re + i P_im)/P_energy from the record header as the combiner forms ar = gr/energy).
    # One estimator for both transports, so a difference between them measures the transport.
    if source is not None:
        got, fleet_now_all = source
        endpoints = []
    if not endpoints and not got:
        return {}
    for url in endpoints:
        try:
            recs = _get("%s/get_records" % url)
        except Exception as e:
            _log_rl(
                "fleet-coh-%s" % url, "fleet coherent: %s unreachable (%s)" % (url, e)
            )
            continue
        per = {}
        for r in recs or []:
            prn = int(r.get("prn", -1))
            if prn <= 0:
                continue
            for _x in r.get("records") or []:
                try:
                    fleet_now_all = max(fleet_now_all, int(_x[0]))
                except (TypeError, ValueError, IndexError):
                    pass
            if prns is not None and prn not in prns:
                continue
            d = {}
            for x in r.get("records") or []:
                try:
                    hop, re_, im_, en = int(x[0]), float(x[1]), float(x[2]), float(x[3])
                except (TypeError, ValueError, IndexError):
                    continue
                if en > 0.0:
                    d[hop] = (complex(re_, im_), en)
            if d:
                per[prn] = d
        if per:
            got[url] = per

    def _solve(store, want_rate=True):
        out = {}
        all_prns = set()
        for per in store.values():
            all_prns.update(per)
        # the fleet's own "now": the newest record anywhere, over every satellite
        fleet_now = max(
            fleet_now_all,
            max(
                (max(d) for per in store.values() for d in per.values() if d), default=0
            ),
        )
        for prn in sorted(all_prns):
            src = {u: per[prn] for u, per in store.items() if prn in per}
            if len(src) < min_instances:
                continue
            # ---- absolute staleness: a set satellite must not keep reporting ----
            # Combiner buffers never expire a PRN's records: once a satellite stops being
            # tracked its last window stays there indefinitely. The relative test below cannot
            # see that, because when every instance is equally stale they agree, and the combine
            # would publish an old detection as current. `fleet_now` is the newest record any
            # instance holds for any satellite, the fleet's own clock (no wall time, no
            # F-engine epoch); a PRN whose newest record trails it by more than max_age_hops is
            # not being tracked and is dropped.
            if max_age_hops and fleet_now:
                prn_now = max(max(d) for d in src.values())
                if fleet_now - prn_now > max_age_hops:
                    continue
            # ---- anchor on the freshest window, not on unanimity or the largest group ----
            # Intersecting every contributor's hops would let one instance outside the others'
            # window empty the common set for every satellite, which reads as an empty sky; an
            # estimator must not get more fragile as the fleet grows. The largest group is wrong
            # too: several instances frozen together would outvote the live ones. Records are
            # stamped by hop, so anchor on the instance reaching the newest hop, keep everyone
            # who overlaps it by min_records (shedding the least-overlapping while the common
            # set is short), and decline an anchor more than its own window span behind the
            # fleet's newest.
            fresh_max = max(max(d) for d in src.values())
            keep, hops, dropped = None, set(), []
            for anchor in sorted(src, key=lambda u: max(src[u]), reverse=True):
                span = max(src[anchor]) - min(src[anchor])
                if max(src[anchor]) < fresh_max - span:
                    break  # stale anchor, and later ones are no fresher: decline
                aset = set(src[anchor])
                cohort = {
                    u: d for u, d in src.items() if len(set(d) & aset) >= min_records
                }
                if len(cohort) < min_instances:
                    continue
                h = set.intersection(*(set(d) & aset for d in cohort.values()))
                while len(h) < min_records and len(cohort) > min_instances:
                    worst = min(cohort, key=lambda u: len(set(cohort[u]) & aset))
                    del cohort[worst]
                    h = set.intersection(*(set(d) & aset for d in cohort.values()))
                if len(h) >= min_records and len(cohort) >= min_instances:
                    keep, hops = cohort, h
                    dropped = [
                        (u, len(set(src[u]) & aset)) for u in src if u not in cohort
                    ]
                    break
            if keep is None:
                continue
            src = keep
            hops = sorted(hops)
            drop_note = dropped
            # ---- the combine, as matrices ----
            # The alignment, the fleet sum, the leave-one-out per instance and the S/R split are
            # the same (n_inst x n_rec) complex array seen four ways. The order of the sums is
            # part of the answer: `tot`, the S/R halves and the leave-one-out all sum over
            # instances in sorted instance-key order (another order moves the results in the
            # last bits), and the S/R split assigns halves by position in that order.
            _U = list(src)  # src order: an argmax tie picks the first instance
            _n = len(hops)
            _A = _np.empty((len(_U), _n), dtype=complex)
            _E = _np.empty((len(_U), _n), dtype=float)
            for _i, _u in enumerate(_U):
                _d = src[_u]
                _A[_i] = [_d[h][0] for h in hops]
                _E[_i] = [_d[h][1] for h in hops]
            _esum = _E.sum(axis=1)
            _ref = _A[int(_np.argmax(_esum))]  # most energetic view
            _cr = _A @ _np.conj(_ref)
            _acr = _np.abs(_cr)
            _keep = _acr > 0.0  # a zero cross-product cannot be aligned
            # Alignment coherence |<A conj(A_ref)>| / sum|A||A_ref|, reported not assumed.
            _den = _np.abs(_A) @ _np.abs(_ref)
            _refi = int(_np.argmax(_esum))
            align_q = [
                float(_acr[i] / _den[i])
                for i in range(len(_U))
                if _keep[i] and _den[i] > 0.0 and i != _refi
            ]
            _idx = [i for i in range(len(_U)) if _keep[i]]
            if len(_idx) < min_instances:
                continue
            _order = sorted(
                _idx, key=lambda i: _U[i]
            )  # sorted by instance key (URL), the sum order
            urls = [_U[i] for i in _order]
            nrec = _n
            # e^{-i arg(<A conj(A_ref)>)} x MRC weight (mean energy), applied per instance
            _rot = _np.conj(_cr[_order]) / _acr[_order]
            _w = _esum[_order] / float(_n)
            _AL = _A[_order] * (_rot * _w)[:, None]
            tot = _AL.sum(axis=0)
            # ---- record-stream carrier rate (_fit_record_rate) ----
            # Skipped on the null pass: _solve runs once for real and once per null trial on
            # shuffled data, and the null reads only deep_snr and per_inst.
            rate_hz = rate_sig = None
            if want_rate and hop_rate_hz and nrec >= 16:
                try:
                    rate_hz, rate_sig = _fit_record_rate(hops, tot, hop_rate_hz)
                except Exception:
                    rate_hz = rate_sig = None
            # ---- per instance: leave that instance out of its own reference ----
            _rest = tot[None, :] - _AL
            _m = _np.abs(_rest)
            _safe = _np.where(_m > 0.0, _m, 1.0)
            _corr = _np.where(_m > 0.0, _AL * _np.conj(_rest) / _safe, _AL)
            per_inst = {u: _coherent_sum(_corr[j])[0] for j, u in enumerate(urls)}
            # ---- fleet total: one-way split (S integrated, R only referenced) ----
            # Halves alternate by sorted instance key, a pure function of the instance set: no
            # measurement enters it. The estimator integrates S and spends R on the phase
            # reference, so a split balanced in sum|w|^2 would be ideal, but any rule that reads
            # per-poll amplitudes or energies re-decides membership on noise, and integrating a
            # different aperture each poll steps deep_snr for reasons unrelated to the sky. A
            # fixed split's energy imbalance is second-order on the variance (the one-way split
            # already concedes sqrt(2)). Sorted keys (gnss0/gnss1 URLs, or <node>.<gpu> telemetry
            # tags) alternate the two GPUs of each node, so with every instance present each half
            # gets one GPU from every node: equal channel count, equal hardware, balanced in
            # expectation.
            half = {u: (i % 2) for i, u in enumerate(sorted(urls))}
            _h = _np.array([half[u] for u in urls])
            _s = _AL[_h == 0].sum(axis=0)  # signal half
            _r = _AL[_h == 1].sum(axis=0)  # reference half
            _mr = _np.abs(_r)
            fleet = _np.where(
                _mr > 0.0, _s * _np.conj(_r) / _np.where(_mr > 0.0, _mr, 1.0), 0j
            )
            snr, amp, cf = _coherent_sum(fleet)
            out[prn] = {
                "deep_snr": snr,
                "deep_amplitude": amp,
                "coh_frac": cf,
                "n_src": len(urls),
                "n_rec": nrec,
                "align": (sum(align_q) / len(align_q)) if align_q else 0.0,
                "per_inst": per_inst,
                "best_inst_snr": max(per_inst.values()) if per_inst else 0.0,
                # record-stream carrier rate + its split-half sigma (Hz). None when the caller
                # did not supply hop_rate_hz, the arc has fewer than 16 records, or the fit
                # declined (see _fit_record_rate).
                "rate_hz": rate_hz,
                "rate_sigma_hz": rate_sig,
                # Instances excluded from this PRN's combine, as (instance key, hops it shared
                # with the anchor's window). Non-empty means the fleet degraded rather than
                # failed; the caller logs it (FLEET-DROP).
                "dropped": drop_note,
            }
        return out

    out = _solve(got)
    if not out:
        return {}
    # ---- measured null floor (see the docstring): same math, common phase destroyed ----
    # The shuffle is per instance and per PRN, so each instance's records keep their amplitudes
    # and energies but no longer line up in time with anyone else's. The floor is the max over
    # every null statistic produced this call, fleet totals and per-instance values alike,
    # since both are reported and both must be defensible.
    rng = random.Random(seed)
    floor = 0.0
    for _ in range(max(1, null_trials)):
        shuf = {}
        for u, per in got.items():
            sp = {}
            for prn, d in per.items():
                hops = list(d)
                vals = [d[h] for h in hops]
                rng.shuffle(vals)
                sp[prn] = {hops[i]: vals[i] for i in range(len(hops))}
            shuf[u] = sp
        nres = _solve(shuf, want_rate=False)
        for v in nres.values():
            floor = max(floor, v["deep_snr"], *(v["per_inst"].values() or [0.0]))
    for v in out.values():
        v["floor"] = floor
        v["present"] = v["deep_snr"] >= floor_margin * floor
    return out


# --- fleet phase-slope delay fit (docs/CHORD_JOINT_TRACKING.md section 2) ------------------
#
# A delay is a phase ramp across frequency: dphi/df = -2*pi*tau. The per-channel prompts
# (GnssGpuRecordAssemble /get_spectrum: NCO-derotated, element-combined, window-summed)
# contain the full-band delay at matched-filter precision, sigma_tau ~ 1/(2*pi*beta_rms*SNR).
# E/P/L keeps only three lag samples of this spectrum, and per narrow channel E ~ P ~ L, which
# rails the disc (E5a); a slope fit is immune to that.
#
# Why a fleet fit: one instance's 7 channels at 3.125 MHz stride alias at 3.27 chips at L5
# (the tracker grating lobes). The defence is the union of the instances' combs, which have
# different per-node offsets, as in the search. Two consequences in the code below:
#   * instances carry unknown constant phases phi_i (different NCO histories), solved jointly
#     with tau by alternation: given tau, phi_i = arg sum_{c in i} E_c A_c e^{+i 2 pi f_c tau};
#     given phi_i, one coherent fold over all channels. A few rounds from tau = 0 converge,
#     because a locked seed is already within ~1 chip.
#   * summing magnitudes per instance does not work: |.| erases the cross-instance phase that
#     tells the true peak from the comb lobes.
#
# The floor is measured, never assumed (as in fleet_coherent): the same fold runs on a null in
# which each instance's channel phases are permuted, amplitudes and energies staying with their
# channels, which destroys the frequency-phase association the fit locks onto. A tau is
# reported with that floor, and the caller sees both.


def fit_spectrum_delay(
    points,
    chip_rate_hz,
    chan_width_hz,
    span_chips=2.0,
    coarse_step_chips=0.02,
    rounds=4,
    rng=None,
):
    """Joint (tau, phi_i) fit. `points` = [(freq_id, complex A, energy, inst_key)].

    Returns (tau_chips, peak, floor, n_pts, n_inst): tau of the correlation peak relative to
    the replica placement (positive = the sky's code arrives later than the replica models),
    the fold magnitude at the peak, the shuffled-null floor from the same points, and the
    sizes. Returns None unless at least 3 usable points sit in instances holding at least 2
    each. Deterministic given `rng` (a random.Random; default seed 0xC0DE).
    """
    if len(points) < 3:
        return None
    insts = {}
    for fid, a, en, key in points:
        if en > 0.0 and (a.real != 0.0 or a.imag != 0.0):
            insts.setdefault(key, []).append((fid, a, en))
    insts = {k: v for k, v in insts.items() if len(v) >= 2}
    if not insts or sum(len(v) for v in insts.values()) < 3:
        return None
    # Weights: energy is the ML combining weight (A = G/E has variance ~1/E), so each point
    # contributes E*A. Frequencies are offsets from the lowest channel, f0.
    f0 = min(fid for v in insts.values() for fid, _, _ in v)

    def _prep(store):
        """Flatten a store to (dfreq, weights, group-matrix, n_inst).

        `f0` and the channel set are shared with the caller's `insts`, so the null store
        preps to the same shapes -- only the values differ.
        """
        keys = sorted(store)
        cols = {k: i for i, k in enumerate(keys)}
        fid = _np.array([f for k in keys for f, _, _ in store[k]], dtype=float)
        w = _np.array([en * a for k in keys for _, a, en in store[k]], dtype=complex)
        g = _np.array([cols[k] for k in keys for _ in store[k]], dtype=int)
        # one-hot (n_points x n_inst): a matmul is the group-sum, far cheaper than
        # np.add.at's fancy-index path for this few instances.
        G = _np.zeros((len(g), len(keys)), dtype=float)
        G[_np.arange(len(g)), g] = 1.0
        return (fid - f0) * chan_width_hz, w, G, len(keys)

    def scan(prep, taus, phi):
        df, w, G, _ = prep
        t = _np.asarray(taus, dtype=float)
        # (n_tau x n_points) phase ramp -> weight -> sum per instance -> derotate -> fold
        E = _np.exp(2j * _np.pi * _np.outer(t, df))
        total = ((E * w) @ G) @ _np.exp(-1j * phi)
        m = _np.abs(total)
        k = int(_np.argmax(m))
        return float(t[k]), float(m[k])

    def solve_phi(prep, tau_s):
        df, w, G, _ = prep
        s = (_np.exp(2j * _np.pi * tau_s * df) * w) @ G
        # an instance whose sum is zero has no phase and takes 0.0
        return _np.where(_np.abs(s) > 0.0, _np.angle(s), 0.0)

    def run(store):
        prep = _prep(store)
        span_s = span_chips / chip_rate_hz
        step_s = coarse_step_chips / chip_rate_hz
        n = max(3, int(round(2.0 * span_s / step_s)) + 1)
        taus = [-span_s + i * (2.0 * span_s / (n - 1)) for i in range(n)]
        tau, mag = 0.0, 0.0
        for _ in range(rounds):
            phi = solve_phi(prep, tau)
            tau, mag = scan(prep, taus, phi)
            # zoom: next round scans +-3 coarse steps around the peak at 1/5 step
            taus = [tau + (i - 15) * step_s / 5.0 for i in range(31)]
        return tau, mag

    tau_s, peak = run(insts)
    # Shuffled null: same instances, same freqs, same energy weights; only arg(A) is permuted.
    # Moving (A, en) together would move the weights, and the null would then fold as well as
    # the data whenever one channel dominates.
    rng = rng or random.Random(0xC0DE)
    null_store = {}
    for k, v in insts.items():
        ph = [cmath.phase(a) for _, a, _ in v]
        rng.shuffle(ph)
        null_store[k] = [
            (fid, abs(a) * cmath.exp(1j * ph[i]), en)
            for i, (fid, a, en) in enumerate(v)
        ]
    _, floor = run(null_store)
    return (
        tau_s * chip_rate_hz,
        peak,
        floor,
        sum(len(v) for v in insts.values()),
        len(insts),
    )


def fleet_spectrum_aligned(endpoints, prns=None, log=None, stale_margin=0):
    """Poll every instance for the same hop-quantised window and gather the per-channel points
    across the whole fleet.

    Returns (points_by_prn, meta) where meta = {window, w0, w1, served, dropped, reanchored};
    meta is {window: None, served: {}, dropped} when no reachable instance has a complete
    window, and {window, served: {}, dropped, misaligned: True} when the instances disagree on
    the window's samples. `points_by_prn` is {prn: [(freq_id, A, energy, inst_key)]}, what
    fit_spectrum_delay consumes; every point comes from the same window (the same samples on
    every instance, checked below), with each instance's phase accumulator (phi0) rotated out.

    Choosing the common index: every reply carries available:[lo,hi], success or refusal, and
    the index is min(hi), the newest window every instance has finished. Instances lag each
    other by a few records, so the fastest is typically a window ahead; taking its index would
    make the laggards answer not_yet forever.

    Never block on unanimity: an instance that cannot serve the common window is dropped by
    name and the rest combine without it. Requiring all of them would let one frozen instance
    empty the common window for every satellite, which reads as an empty sky.
    """
    avail, served, dropped, reanchored = {}, {}, [], []
    for url in endpoints:
        try:
            r = _get("%s/get_spectrum" % url)
        except Exception as e:
            _log_rl(
                "fleet-spec-%s" % url, "fleet spectrum: %s unreachable (%s)" % (url, e)
            )
            dropped.append((url, "unreachable"))
            continue
        a = r.get("available") or [-1, -1]
        if a[1] is None or a[1] < 0:
            dropped.append((url, "no complete window yet"))
            continue
        avail[url] = (int(a[0]), int(a[1]))
    if not avail:
        return {}, {"window": None, "served": {}, "dropped": dropped}

    # min(hi) over all instances hands the common index to the slowest member, so a frozen
    # instance (hi never advancing) would pin the window: the healthy instances refuse a window
    # they have rotated out, and the fit runs on the frozen one alone. stale_margin > 0
    # (--spectrum-stale-margin) first drops, by name, every instance whose hi trails the
    # fleet's newest by more than that many windows (healthy lag is a few records).
    if stale_margin and avail:
        _mx = max(hi for _, hi in avail.values())
        for url in [u for u, (_, hi) in avail.items() if hi < _mx - stale_margin]:
            dropped.append(
                (
                    url,
                    "stale (hi %d, fleet newest %d, margin %d)"
                    % (avail[url][1], _mx, stale_margin),
                )
            )
            del avail[url]
    idx = min(hi for _, hi in avail.values())
    out, w0s, w1s = {}, set(), set()
    for url, (lo, hi) in avail.items():
        if idx < lo or idx > hi:
            # Refused by its own bookkeeping before we even ask -- name it and move on.
            dropped.append((url, "window %d outside [%d,%d]" % (idx, lo, hi)))
            continue
        try:
            r = _get("%s/get_spectrum?window=%d" % (url, idx))
        except Exception as e:
            dropped.append((url, "unreachable on second poll (%s)" % e))
            continue
        if r.get("status") != "ok":
            dropped.append((url, r.get("status") or "no status"))
            continue
        w0s.add(r.get("wstart0"))
        w1s.add(r.get("wstart1"))
        served[url] = r
        fids = r.get("freq_ids") or []
        for row in r.get("prns") or []:
            prn = int(row.get("prn", -1))
            if prn <= 0 or (prns is not None and prn not in prns):
                continue
            # Phase currency. The export rotated this window by exp(-i*phi0), phi0 being the
            # instance's phase accumulator at the window's first record (arbitrary origin, and
            # it takes each re-pin step); multiplying by exp(+i*phi0) rotates it out, which puts
            # every window and every instance on one reference. n_reanchor > 0 means a fresh
            # acquisition reset the accumulator mid-window: no single constant can undo that,
            # so the PRN is dropped for this window rather than folding a discontinuity in.
            if int(row.get("n_reanchor", 0) or 0) > 0:
                reanchored.append((url, prn))
                continue
            derot = cmath.exp(1j * float(row.get("phi0", 0.0) or 0.0))
            ch = row.get("chan") or []
            pts = out.setdefault(prn, [])
            for i, fid in enumerate(fids):
                if i >= len(ch):
                    break
                re_, im_, en = ch[i]
                if en > 0.0:
                    pts.append((int(fid), complex(re_, im_) * derot, float(en), url))
    # The invariant, asserted rather than assumed: the same index must mean the same samples.
    # If two instances disagree the quantisation is broken (a node config with a different
    # spectrum_window_samples), and every phase downstream would be wrong, so refuse loudly
    # instead of combining.
    if len(w0s) > 1 or len(w1s) > 1:
        if log:
            log(
                "fleet spectrum: window %d has different sample spans across instances "
                "(wstart0 %s, wstart1 %s) -- the quantisation is broken, check that every node "
                "config has the same spectrum_window_samples"
                % (idx, sorted(w0s), sorted(w1s))
            )
        return (
            {},
            {"window": idx, "served": {}, "dropped": dropped, "misaligned": True,},
        )
    return (
        out,
        {
            "window": idx,
            "w0": next(iter(w0s), None),
            "w1": next(iter(w1s), None),
            "served": served,
            "dropped": dropped,
            "reanchored": reanchored,
        },
    )
