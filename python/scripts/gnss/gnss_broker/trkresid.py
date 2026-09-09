"""The TRACKER's code observable: the replica's physical code phase minus the model.

Every record the fleet emits carries, in slots 1 and 2, the Doppler the despread ran at and
the sample-0-anchored code ARGUMENT in that Doppler's currency (gnssRecord.hpp, the slot-2
currency contract). propagate_seed puts the standing fleet trim inside that argument, so

    cp_phys(hop) = REC_CP + t_abs * f_chip * (1 + sgn * REC_DOPPLER / f_c)      (mod L)

IS the tracker's code phase: the replica placement the closed DLL holds on the correlation
peak. wrap(cp_phys - cp_model - clk) is then a code residual in this chain's chips with the
receiver clock removed -- the same quantity the dead-reckon integrity residual forms from the
SEARCH's detections, but measured by the stage that resolves ~1e-2 chips rather than the one
that resolves ~0.5. Unlike `offs`, it exists on every chain, detectors or not: a model-primary
chain has no detections, but it has records.

WHY RECORDS ARE AVERAGED. Slot 1 is float32, so its ulp at a few kHz is ~1e-4 Hz, and the
lift above multiplies it by t_abs * f_chip / f_c -- thousands of chips per Hz at any real
uptime. One record's reconstruction therefore carries ~0.1-0.3 chips of quantization noise,
white from record to record because the propagated Doppler crosses many ulps per record.
Averaging every instance's records over the newest windows takes it to ~1e-2 chips. The
instances agree to one ulp (one seed, one fleet trim: nothing here is per-node), so the
fleet's records are one population, not twelve.

WHY THE NOMINAL RAMP IS INTEGER ARITHMETIC. t_abs * f_chip is ~1e12 chips; taken as a double
and reduced mod L it keeps ~1e-3 chips, which is fine, but the exact form costs one bigint
multiply and removes the question. Only the Doppler term (1e6-1e7 chips) is a float.
"""
import math
from fractions import Fraction


def phys_chips(cp_arg, dop_hz, hop, hps, chip_rate_hz, carrier_hz, code_len, sgn=1.0):
    """Physical code phase (chips, mod code_len) of a record's replica at its first sample.

    `hps` is the hop rate as a Fraction (hops per second); `cp_arg`/`dop_hz` are the record's
    own slot 2 / slot 1 -- they MUST come from the same record (the argument is only meaningful
    against the Doppler it was expressed in).
    """
    n_num = hop * hps.denominator * int(chip_rate_hz)     # t_abs*f_chip = n_num / hps.numerator
    nominal = (n_num % (int(code_len) * hps.numerator)) / float(hps.numerator)
    t_abs = hop / float(hps)
    dopp = t_abs * chip_rate_hz * (sgn * dop_hz / carrier_hz)
    return (float(cp_arg) + nominal + dopp) % code_len


def wrap(x, code_len):
    return (x + code_len / 2.0) % code_len - code_len / 2.0


def telem_records(client, chain, prns, n_win=2, lag=1):
    """{prn: [(hop, cp_arg, dop_hz), ...]} over every instance's records in the newest windows.

    Records with zero prompt energy are silence, not measurements, and are skipped exactly as
    coherent_source skips them.
    """
    from gnss_broker.telem import REC_CP, REC_DOPPLER, REC_P_ENERGY
    out = {}
    want = set(int(p) for p in prns)
    for w in client.windows(chain, lag=lag)[-int(n_win):]:
        for _inst, f in client.frame_set(chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                hop = f.hop(r)
                for prn in f.prns():
                    if prn not in want:
                        continue
                    row = f.row(r, prn)
                    if row is None or row[REC_P_ENERGY] <= 0.0:
                        continue
                    out.setdefault(prn, []).append((hop, float(row[REC_CP]),
                                                    float(row[REC_DOPPLER])))
    return out


def residuals(records, pd, tag, cp_predicted, clk, drift, t_now_abs, hps, chip_rate_hz,
              carrier_hz, code_len, sgn=1.0):
    """Per-PRN tracker residual, chips, clock removed. Pure: every input is a number or a dict.

    records: {prn: [(hop, cp_arg, dop_hz)]}; pd: {(tag, prn): predict_all row}; cp_predicted(v,
    t_abs) the model's physical code phase EXCLUDING the receiver clock (deadreckon's); clk the
    solved clock (chips), drift its rate (chips/s, may be None); t_now_abs the capture age the
    clock is referenced to. Each record's offset is normalised to t_now_abs by the drift, as
    the integrity residual is, then wrapped about the clock; the per-PRN value is the mean of
    the wrapped records and `sd` their scatter (the float32 quantization, mostly).

    Returns {prn: {"chips", "sd", "n", "hop"}}; a PRN with no model row or no records is absent.
    """
    out = {}
    if clk is None:
        return out
    drift = drift or 0.0
    hps = Fraction(hps).limit_denominator(10 ** 6)
    for prn, recs in records.items():
        v = pd.get((tag, prn))
        if v is None or not recs:
            continue
        rs = []
        for hop, cp, dop in recs:
            t_abs = hop / float(hps)
            d = (phys_chips(cp, dop, hop, hps, chip_rate_hz, carrier_hz, code_len, sgn)
                 - cp_predicted(v, t_abs) + drift * (t_now_abs - t_abs))
            rs.append(wrap(d - clk, code_len))
        # circular mean about the first record: the wrap is only safe once per record
        r0 = rs[0]
        ds = [wrap(r - r0, code_len) for r in rs]
        m = sum(ds) / len(ds)
        sd = math.sqrt(sum((x - m) ** 2 for x in ds) / len(ds)) if len(ds) > 1 else 0.0
        r_clk = wrap(r0 + m, code_len)
        # `raw` keeps the receiver clock IN (the code RANGE residual, what a carrier residual
        # also carries, so the two can be differenced); `chips` has the solved clock removed
        out[prn] = {"chips": r_clk, "raw": wrap(r_clk + clk, code_len), "sd": sd, "n": len(rs),
                    "hop": max(h for h, _c, _d in recs)}
    return out


def tracker_residuals(ctx, n_win=2):
    """The stage entry point: records from the telem client (falling back to the fleet's
    coherent row -- one record, ~0.2 chips of quantization), model and clock from the
    dead-reckon products. {} until a clock exists, which is the honest state to publish."""
    st = ctx.dr_state or {}
    pd = st.get("pd")
    clk = st.get("clk")
    if not pd or clk is None or ctx.drp.t_now_abs is None or ctx.cp_predicted is None:
        return {}
    # ONLY SATELLITES THE FLEET IS ON. A replica that is not on its signal still has a code
    # phase, and the number this module would make from it is a wrong answer with a small
    # formal error (a replica parked on noise sits still: chips off, sd ~0). The presence
    # verdict is the fleet DLL's own; the below-horizon noise probes are excluded by name.
    probes = ctx.probe_set or set()
    fleet = {p: v for p, v in (ctx.dllp.fleet or {}).items()
             if v.get("present") and p not in probes}
    recs = {}
    if ctx.telem_client is not None and ctx.telem_chain:
        try:
            recs = telem_records(ctx.telem_client, ctx.telem_chain, fleet.keys(), n_win=n_win)
        except Exception:
            recs = {}
    for prn, v in fleet.items():
        if prn in recs:
            continue
        c = v.get("coh_row") or {}
        if c.get("code_phase_chips") is None or c.get("doppler_hz") is None \
                or int(c.get("pow_hop", -1)) < 0:
            continue
        recs[prn] = [(int(c["pow_hop"]), float(c["code_phase_chips"]), float(c["doppler_hz"]))]
    res = residuals(recs, pd, ctx.drp.tag, ctx.cp_predicted, clk, st.get("drift"),
                    ctx.drp.t_now_abs, ctx.args.hops_per_sec, ctx.args.chip_rate_hz,
                    ctx.args.carrier_hz, ctx.code_len, ctx.args.code_doppler_sign)
    now = ctx.drp.now_w
    for r in res.values():
        r["t"] = now
        r["s"] = r["chips"] / ctx.args.chip_rate_hz
        r["raw_s"] = r["raw"] / ctx.args.chip_rate_hz
    return res
