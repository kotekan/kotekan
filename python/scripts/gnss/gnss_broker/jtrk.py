"""The tracker's code feed into the joint receiver state, as a shadow (#191 step 2).

docs/CHORD_JOINT_TRACKING.md section 6.3 step 2: every admitted chain feeds trkresid's
observable, where the closed DLL holds the replica minus the model, into a joint state of its
own: a second JointReceiverState (receiver.TRK_STATE), beside the one gps_l5's search
detections feed. Nothing reads it: no seed, clock or rate comes from it. It is logged
(JOINT[trk], JTRK) and served (/get_joint), so the step-2 gates can be judged on the sky
before step 3 lets a chain consume it.

The measurement, per satellite, in this chain's chips then converted to reference chips:

    y_i = wrap(cp_phys - cp_model) = clk + b_i + noise

trkresid's `raw`, which keeps the receiver clock in. b_i is keyed by (constellation, PRN), so
every chain on one satellite updates the same row: E5a, E5b and E6 of one Galileo satellite
are one range error seen three times, which is the point of the joint state and the reason a
chain is admitted only once its signal reads within ~1 m of the others on the same satellites
(--joint-trk-feed, set per chain in config/gnss_chains_chord.yaml).

The observable is the sky only while the loop is on the peak (on noise the replica sits where
the seed put it, and the residual is the seed read back: the mirror). So every gate below is
load-bearing, and a satellite that fails one contributes nothing: there is no fallback value
(the #85 feed's fallback to seed + trim - model was the #152 latch pathway). Per satellite:
present (trkresid admits only those, probes excluded), fleet q at or above the floor,
not admitted displaced (the displaced admission rescues an off-peak tap), prompt
power at least joint_trk_p_min times the noise probes' median, the spectrum fit's |tau| under
joint_trk_spec_tau_max whenever the fit is valid, enough records behind the residual, records
newer than the last fed, and the chain's own track gate. The power bar is stricter than
presence's (k_sigma x the median, 3): a satellite fading at low elevation stays present at
3-5x while its replica wanders on noise (10-26 02:23-04:50Z, G1/G3/G28 at el 9-13 deg, steps
of +-3 chips fed), and 97% of rows overnight sat above 50x. And the node's own trim within
joint_trk_rail_frac of its clamp: while the trim follows the signal the residual is the sky,
but at the rail the replica is pinned and the signal moves on (10-26 03:21-03:23Z: G3 at C/N0
~50, trim walking -0.55 -> -1.47, the L5 clamp, a minute before its power fell); off gps_l5,
<= 0.06% of present rows overnight reached 70% of the clamp. Per
chain and cycle: no feed during the warm-up or a boresight transit, and none unless at least
joint_trk_min_sats satellites pass (a thin set cannot separate clk from b).
"""

from gnss_broker.transport import _log, _log_rl
from gnss_broker.trimarm import LOBE_MIN_RECORDS, LOBE_XCOH

#: Gate names, in the order a satellite is tested; the first that fails is the one counted.
GATES = ("track", "q", "onpeak", "xcoh", "power", "rail", "tau", "nrec", "stale")
#: The presence verdicts that admit a satellite whose tap is OFF the peak by construction: the
#: displaced admission ("q+p:probes+disp", "q+deep:probes+disp") rescues exactly that row, and
#: there this residual is the seed read back (the mirror), not the sky. The deep gate ("deep")
#: is not one of them: on gps_l5 it re-judges every recently detected satellite, strong ones
#: included, and the q and power bars below judge those rows like any other.
OFFPEAK_SUFFIX = "+disp"


def feed_rows(
    trk,
    fleet,
    spec_fit,
    track_ok,
    last_hop,
    tag,
    k,
    sigma_ref,
    tau_max,
    min_rec,
    p_min,
    trims=None,
    rail=None,
):
    """The rows this chain may feed, and why the rest may not. Pure.

    trk: trkresid.tracker_residuals' {prn: {"raw", "n", "hop", ...}} (present, non-probe);
    fleet: the fleet DLL's {prn: {"q", "q_floor", "present_gate", "p_pow", "p_med", ...}};
    p_min: the prompt-power bar in units of p_med; trims: {prn: the node's trim readback,
    chips}, a PRN absent reading 0; rail: |trim| at or beyond which a PRN is refused (None:
    no rail gate); spec_fit: {prn: {"tau_chips"}} for
    the PRNs whose fit is valid this cycle; track_ok(prn) the chain's track gate; last_hop:
    {prn: newest hop already fed}, not modified; k: reference chips per chip of this chain.

    Returns (rows, failed, hops): rows [((tag, prn), y_ref, sigma_ref)], failed {gate: n},
    hops {prn: hop} for the rows (to record once fed)."""
    rows, hops = [], {}
    failed = dict.fromkeys(GATES, 0)
    for prn in sorted(trk):
        v = trk[prn]
        f = fleet.get(prn) or {}
        if not track_ok(prn):
            failed["track"] += 1
        elif f.get("q") is None or f.get("q_floor") is None or f["q"] < f["q_floor"]:
            failed["q"] += 1
        elif str(f.get("present_gate") or "").endswith(OFFPEAK_SUFFIX):
            failed["onpeak"] += 1
        elif (
            f.get("xcoh") is not None
            and int(f.get("n_xcoh") or 0) >= LOBE_MIN_RECORDS
            and f["xcoh"] < LOBE_XCOH
        ):
            # The instances' prompts do not agree: the tracker is off the signal, whatever q and
            # the power say (10-27 G20 21:47, G27 22:44, 10-28 G03 03:12: xcoh ~0 for minutes,
            # residuals -10..+23 chips, every other gate passed). trimarm's lobe test reads it.
            failed["xcoh"] += 1
        elif (
            not f.get("p_med")
            or f.get("p_pow") is None
            or f["p_pow"] < p_min * f["p_med"]
        ):
            failed["power"] += 1
        elif rail is not None and abs((trims or {}).get(prn) or 0.0) >= rail:
            failed["rail"] += 1
        elif prn in spec_fit and abs(spec_fit[prn]["tau_chips"]) >= tau_max:
            failed["tau"] += 1
        elif int(v.get("n", 0)) < min_rec:
            failed["nrec"] += 1
        elif last_hop.get(prn) is not None and int(v["hop"]) <= last_hop[prn]:
            failed["stale"] += 1
        else:
            rows.append(((tag, prn), float(v["raw"]) * k, sigma_ref))
            hops[prn] = int(v["hop"])
    return rows, failed, hops


def stage_joint_trk(ctx):
    """Feed this chain's gated tracker residuals to the shadow state (receiver.TRK_STATE).

    Runs on a chain with --joint-trk-feed once trkresid has produced this cycle's residuals.
    Changes nothing any actuator reads; failure costs the loop nothing."""
    from gnss_broker.deadreckon import _transit_pd
    from gnss_broker.receiver import JOINT_REF_RATE_HZ, TRK_STATE
    from gnss_broker.sky import nearest_boresight
    from gnss_broker.trimarm import trim_clamp

    a = ctx.args
    st = ctx.dr_state.setdefault("jtrk", {"last_hop": {}, "cycles": 0, "thin": 0})
    st["cycles"] += 1
    why = None
    if ctx.uptime_s < a.joint_feed_warmup_s:
        why = "warm-up (%.0f s of %.0f)" % (ctx.uptime_s, a.joint_feed_warmup_s)
    else:
        near = nearest_boresight(_transit_pd(ctx))
        if near is not None and near[0] < a.dr_clock_transit_freeze_deg:
            why = "transit (%s %d at %.1f deg)" % (near[1][0], near[1][1], near[0])
    k = JOINT_REF_RATE_HZ / a.chip_rate_hz
    rows, failed, hops = feed_rows(
        ctx.dllp.trk or {},
        ctx.dllp.fleet or {},
        ctx.dllp.spec_fit or {},
        ctx.track_ok,
        st["last_hop"],
        ctx.drp.tag,
        k,
        a.joint_trk_sigma,
        a.joint_trk_spec_tau_max,
        a.joint_trk_min_records,
        a.joint_trk_p_min,
        trims={p: r.get("trim_chips") for p, r in (ctx.dls.readback or {}).items()},
        rail=a.joint_trk_rail_frac * trim_clamp(a.chip_rate_hz),
    )
    if why is None and len(rows) < a.joint_trk_min_sats:
        why = "thin set (%d < %d)" % (len(rows), a.joint_trk_min_sats)
        st["thin"] += 1
    n_ok = 0
    # escape_clock_min_sats: a tracker's step on one satellite moves that satellite's b, and
    # the clock only when as many satellites as a clock solve needs agree (state_filter).
    js = ctx.rx.joint_receiver(
        ctx.band_id,
        ctx.code_len,
        a.chip_rate_hz,
        which=TRK_STATE,
        gauge_mode="prior",
        escape_clock_min_sats=a.joint_trk_min_sats,
    )
    if why is None:
        n_ok = js.cycle(rows, ctx.drp.t_now_abs)
        st["last_hop"].update(hops)
    else:
        # Held (a transit, a thin set, the warm-up): still a time update, so the clock moves
        # on clk_rate and P grows. Without it the state stood at its last epoch, and a chain
        # adopting it took a clock frozen for the whole hold (10-28 G32 transit: 13 min,
        # +0.6 chips by the end, at the +0.0008 chips/s the rate row held).
        js.predict(ctx.drp.t_now_abs)
    for n in js.drain_notes():
        _log_rl("jtrk-note", "JOINT[trk]: %s" % n, every_s=10.0)
    st["last"] = {
        "t": ctx.t0,
        "offered": len(ctx.dllp.trk or {}),
        "passed": len(rows),
        "accepted": n_ok,
        "failed": failed,
        "held": why,
    }
    _log_rl(
        "jtrk-chain",
        "JTRK %s: %s; %d of %d residuals passed (failed %s); %d accepted; thin-set cycles %d "
        "of %d"
        % (
            ctx.args.signal,
            ("held: " + why) if why else "fed",
            len(rows),
            len(ctx.dllp.trk or {}),
            " ".join("%s %d" % (g, n) for g, n in failed.items() if n) or "none",
            n_ok,
            st["thin"],
            st["cycles"],
        ),
        every_s=60.0,
    )
    snap = ctx.rx.joint_trk_snapshot(ctx.drp.t_now_abs)
    if ctx.publisher is not None and snap is not None:
        ctx.publisher.set_joint(snap, st["last"], ctx.t0)
    if snap is not None:
        live = snap.get("live_clk")
        _log_rl(
            "jtrk-summary",
            "JOINT[trk] %s | clk %+.3f sigma %.3f vs search-fed joint %s"
            % (
                js.summary(ctx.drp.t_now_abs),
                snap["clk"],
                snap["clk_sigma"],
                "%+.3f (trk - joint %+.3f, wrapped)" % (live, snap["clk_delta"])
                if live is not None
                else "absent",
            ),
            every_s=30.0,
        )
