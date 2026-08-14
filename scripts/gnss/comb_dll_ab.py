#!/usr/bin/env python3
"""A/B the COMB-derived fleet DLL against the polled one, on the same satellites, live.

Task #63 step 2. The broker closes its code loop on fleet_dll()'s discriminator, which is
built from powers the TRACKERS formed by summing across each instance's channels. combdll's
fleet_dll_comb() forms the same three powers in the broker from the un-summed comb. Before
the loop is switched, the two have to be shown to agree -- and "agree" has to be measured
against how much the polled arm disagrees WITH ITSELF, because a couple of dB of churn cycle
to cycle would otherwise read as a defect in the new path.

WHAT IS COMPARED, AND WHY IT IS THE RATIOS
  * disc = (E-L)/(E+L) and q = 2P/(E+L) -- the two numbers the loop and the presence gate
    actually consume. Both are RATIOS, so they survive the two known offsets between the arms
    (#62's non-causal element cal, and EMA-vs-mean averaging) that a raw power comparison
    would trip over.
  * p_pow is reported as a dB offset, NOT as an error. An offset here is expected and is
    attributable; it is logged so it can be watched, not gated on.

THE CONTROL IS MANDATORY (the lesson from #61's ab_fleetcoh). Each cycle also compares the
POLLED arm against the polled arm of the PREVIOUS cycle. That is the same estimator, the same
endpoints, a few seconds apart -- pure churn. If comb-vs-poll scatter is at or below
poll-vs-poll scatter, the two paths are the same measurement and the difference is the
instrument moving, not the arithmetic.

usage:
  comb_dll_ab.py [--chain gps_l5] [--cycles 20] [--interval 4] [--gather 127.0.0.1:11061]
                 [--windows 32] [--config config/gnss_chains_chord.yaml]

Read-only: it polls the combiners' /get_status (which the broker does anyway) and reads the
gather's broadcast. It never posts, and it never touches the broker.
"""
import argparse
import importlib.util
import math
import os
import statistics
import sys
import time

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))

from gnss_broker import combdll, telem  # noqa: E402
from gnss_broker.fleet import fleet_dll  # noqa: E402
from gnss_broker.transport import expand_token  # noqa: E402
import gnss_broker.transport as _tr  # noqa: E402

_tr._log_rl = lambda k, m: None   # a probe does not narrate the broker's polling


def chain_endpoints(cfg_path, chain):
    """The chain's dll-combiners, expanded -- read from the SAME file the broker runs on.

    Hand-listing endpoints here is how a probe ends up measuring a different fleet than the
    loop does (fleetq.py carries a comment about exactly that asymmetry).
    """
    import yaml
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    chains = cfg.get("chains") or {}
    # The chain KEY is the broker's log tag, which is also the chain key the telemetry frames
    # carry (broker: `telem_chain = log_tag() or signal`) -- so one name selects both arms.
    entry = chains.get(chain)
    if entry is None:
        raise SystemExit("chain %r not in %s (have: %s)"
                         % (chain, cfg_path, ", ".join(sorted(chains))))
    raw = entry.get("dll-combiners") or entry.get("n2-combiners") or ""
    eps = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            eps.extend(expand_token(tok))
    return eps, entry


def paired(a, b, key):
    """[(prn, a_val, b_val)] over PRNs both dicts hold."""
    return [(p, a[p][key], b[p][key]) for p in sorted(set(a) & set(b))]


def spread(pairs):
    """(n, median difference, robust sigma of the difference, max |difference|)."""
    d = [y - x for _p, x, y in pairs]
    if not d:
        return 0, None, None, None
    med = statistics.median(d)
    mad = statistics.median([abs(x - med) for x in d])
    return len(d), med, 1.4826 * mad, max(abs(x) for x in d)


def fmt(v, f="%+.4f"):
    return "  n/a " if v is None else f % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--config", default=os.path.join(K, "config", "gnss_chains_chord.yaml"))
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--windows", type=int, default=32,
                    help="telemetry windows meaned; 32 x 4 records = the combiners' 128")
    ap.add_argument("--hop-window", type=int, default=976562, help="5 s, as the broker runs it")
    ap.add_argument("--min-instances", type=int, default=2)
    ap.add_argument("--k-sigma", type=float, default=3.0)
    ap.add_argument("--profile", type=int, default=-1,
                    help="print the per-channel q profile for this PRN each cycle; "
                         "0 = whichever PRN has the highest fleet q this cycle")
    args = ap.parse_args()

    eps, entry = chain_endpoints(args.config, args.chain)
    probes = None  # the probe PRNs are chosen by the broker at runtime, not in the config
    host, port = telem.parse_endpoint(args.gather)
    cl = telem.TelemClient(host, port, depth=max(64, args.windows * 2), chains=[args.chain])
    cl.start()
    print("chain %s: %d combiner endpoints, gather %s:%d, %d windows"
          % (args.chain, len(eps), host, port, args.windows))
    print("waiting for the gather to fill %d windows..." % args.windows)
    t0 = time.time()
    while time.time() - t0 < 60.0:
        if len(cl.windows(args.chain, lag=1)) >= args.windows:
            break
        time.sleep(1.0)
    have = len(cl.windows(args.chain, lag=1))
    if have < 2:
        raise SystemExit("gather has %d windows for chain %r after 60 s: %s"
                         % (have, args.chain, cl.stats()))
    if have < args.windows:
        print("⚠️  only %d windows -- the comb arm integrates less than the polled one" % have)

    prev_poll = None
    disc_ab, disc_cc, q_ab, q_cc, pdb = [], [], [], [], []
    agree = {"both": 0, "poll_only": 0, "comb_only": 0, "neither": 0}
    for c in range(args.cycles):
        poll = fleet_dll(eps, args.hop_window, args.min_instances, args.k_sigma, 2.2,
                         probe_prns=probes)
        comb = combdll.fleet_dll_comb(cl, args.chain, n_win=args.windows,
                                      min_instances=args.min_instances,
                                      k_sigma=args.k_sigma, probe_prns=probes)
        if not poll or not comb:
            print("cycle %d: poll %d rows, comb %d rows -- skipped" % (c, len(poll), len(comb)))
            time.sleep(args.interval)
            continue
        pd = paired(poll, comb, "disc")
        pq = paired(poll, comb, "q")
        pp = paired(poll, comb, "p_pow")
        disc_ab += [y - x for _p, x, y in pd]
        q_ab += [y - x for _p, x, y in pq]
        pdb += [10.0 * math.log10(y / x) for _p, x, y in pp if x > 0 and y > 0]
        if prev_poll:
            disc_cc += [y - x for _p, x, y in paired(prev_poll, poll, "disc")]
            q_cc += [y - x for _p, x, y in paired(prev_poll, poll, "q")]
        for p in set(poll) & set(comb):
            a, b = bool(poll[p].get("present")), bool(comb[p].get("present"))
            agree["both" if (a and b) else "neither" if not (a or b)
                  else "poll_only" if a else "comb_only"] += 1
        n, med, sg, mx = spread(pd)
        nq, qmed, qsg, _ = spread(pq)
        print("%s cycle %2d  poll %2d rows / comb %2d rows / %2d shared | "
              "disc %s +- %s (max %s) | q %s +- %s | inst %d vs %d"
              % (time.strftime("%H:%M:%S"), c, len(poll), len(comb), n,
                 fmt(med), fmt(sg), fmt(mx), fmt(qmed, "%+.3f"), fmt(qsg, "%.3f"),
                 max(v["n_src"] for v in poll.values()),
                 max(v["n_src"] for v in comb.values())))
        if args.profile >= 0:
            prn = args.profile
            if prn == 0:
                prn = max(comb, key=lambda p: comb[p]["q"])
            row = comb.get(prn)
            if row:
                prof = combdll.chan_profile(row)
                print("    PRN %d comb (fleet q %.2f): %d channels  " % (prn, row["q"], len(prof))
                      + " ".join("%d:%.2f" % (f, q) for f, q, _d in prof))
        prev_poll = poll
        time.sleep(args.interval)

    def report(name, v, unit=""):
        if not v:
            print("  %-28s no data" % name)
            return
        med = statistics.median(v)
        mad = 1.4826 * statistics.median([abs(x - med) for x in v])
        print("  %-28s n %4d  median %+.4f%s  sigma %.4f%s" % (name, len(v), med, unit, mad, unit))

    print("\n=== %s, %d cycles ===" % (args.chain, args.cycles))
    report("disc  comb - poll", disc_ab)
    report("disc  poll - poll (CONTROL)", disc_cc)
    report("q     comb - poll", q_ab)
    report("q     poll - poll (CONTROL)", q_cc)
    report("p_pow comb/poll", pdb, " dB")
    print("  presence: both %d, poll-only %d, comb-only %d, neither %d"
          % (agree["both"], agree["poll_only"], agree["comb_only"], agree["neither"]))
    # ⚠️ WITHOUT PROBES THE ABSOLUTE PRESENCE RATE HERE MEANS NOTHING. The noise anchor is a
    # set of below-horizon PRNs the BROKER picks at runtime (--noise-probes), which this tool
    # has no way to know, so apply_presence falls back to the peer bar -- the one #49 measured
    # excluding 72% of well-detected satellites. Both arms get the same bar, so the DISAGREEMENT
    # count below is still a fair comparison; a low "both" count is the gate, not the combine.
    print("    (no probe anchor in this tool: both arms fall back to the peer bar, so read the "
          "DISAGREEMENT, never the absolute rate -- see #49)")
    # McNemar on the discordant pairs only: the concordant ones carry no information about
    # which arm is different, and counting them inflates any "agreement rate" toward 1.
    nd = agree["poll_only"] + agree["comb_only"]
    if nd:
        b_, c_ = agree["poll_only"], agree["comb_only"]
        chi = (abs(b_ - c_) - 1) ** 2 / float(nd) if nd > 1 else 0.0
        print("  McNemar: %d discordant (%d poll-only, %d comb-only), chi2 %.2f%s"
              % (nd, b_, c_, chi, "  <== ASYMMETRIC" if chi > 3.84 else ""))
    if disc_ab and disc_cc:
        s_ab = 1.4826 * statistics.median(
            [abs(x - statistics.median(disc_ab)) for x in disc_ab])
        s_cc = 1.4826 * statistics.median(
            [abs(x - statistics.median(disc_cc)) for x in disc_cc])
        print("  VERDICT: comb-vs-poll disc scatter %.4f against a same-estimator control of "
              "%.4f -- %s" % (s_ab, s_cc,
                              "within the churn" if s_ab <= 1.5 * s_cc
                              else "LARGER than the churn, investigate"))
    cl.stop()


if __name__ == "__main__":
    main()
