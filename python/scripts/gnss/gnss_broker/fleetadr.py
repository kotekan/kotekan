"""FLEET ADR: the accumulated carrier phase at EXACT hops, from the telemetry records.

THE OBSERVABLE. The tracker removes a commanded carrier phase Phi_cmd from the data and the
prompt carries what is left, so the received carrier phase is Phi_rx = Phi_cmd - arg(A)/2pi
(gnssRecord.hpp). Both halves are ambiguous on their own and both ambiguities die in the
per-record INCREMENT: the commanded advance over a record is known from the Doppler the
replica ran at, and the measured half is arg(A_k conj A_{k-1}) -- ~1e-3 cycles per record
while the carrier loop holds, against an unambiguous range of +-0.25.

WHY NOT SLOT 15. The record's own commanded increment (REC_CPHASE) is float32, and on CHORD
the commanded phase is anchored at RF, so the increment is ~1.2e7 cycles per record and the
slot quantizes it to WHOLE CYCLES (every live value is an exact integer). An ADR folded from
it -- the C++ combiner's `adr_cycles` -- is a +-0.5 cycle/record random walk plus the
re-anchor steps of each window's first record, not a carrier phase. The replica runs at a
constant f_c + dop within a record (propagate_seed per record, the re-pin step folded so the
prompt stays continuous), so the commanded advance over [k-1, k] is exactly
    (f_c + dop_{k-1}) * dt + trim_inc_k
with dop from slot 1 (float32, ~20 cycles of Doppler per record: ulp 2e-6), trim_inc from
slot 19, and dt from the integer hops. The nominal f_c*dt is exact in rationals and kept OUT
of the accumulator, which therefore holds only the Doppler-integrated phase and never loses
a double's precision.

WHY IT LIVES HERE. Every record in the telemetry carries its absolute hop; every chain shares
the one F-engine hop axis; so an ADR folded here is exact where it matters -- pairing two bands
into the geometry-free combination needs equal epochs to ~1e-12 s, i.e. the same integer hop --
and common across all bands at once. Stamped with the arc's first hop too.

THE FLEET COMBINE IS A SUM OF CROSS-PRODUCTS, NOT OF PHASORS. Every instance despreads the
same seed with the same fleet trim, so their prompts are one phasor seen through twelve
channel subsets -- but each carries a different constant (the assembler's NCO origin is set
per instance) AND that NCO is where the tracker folds the re-pin phase steps (Delta_dop*t_abs,
~0.3 cycles per record at this uptime) to keep the exported prompt continuous. REC_PHI0 is
that accumulator, and it jumps by radians every record. The comb's cross-instance rule
(multiply by exp(+i*phi0)) puts instances on one reference WITHIN a record; applied across
records it re-injects the folded steps, and an ADR built that way walks by hundreds of cycles
(measured before this was understood). So the exported prompt is used AS EXPORTED, and the
per-record increment is
    dres = arg( sum_inst S_i(k) * conj(S_i(k-1)) ) / 4pi,    S_i = H_i^2 + T_i^2,
where H is the head amplitude and T = A - H: the per-instance constants cancel in each
product, the fold stays folded, and the twelve products add coherently (a single instance's
step is ~0.1 cycles rms on a weak satellite -- at the alias edge; the sum is ~0.03). Squaring
makes S sign-free (overlay pilots flip sign record to record) and straddle-immune (both
segments of a record that spans a secondary-chip transition add in power). A tracker that
does not segment writes head == A, tail 0, and S reduces to A^2.

THE FRAME-BOUNDARY FOLD IS REPAIRED HERE. The assembler folds the replica's per-record
re-pin step into its NCO (REC_PHI0 is that accumulator) and the exported prompt is
continuous WITHIN a frame -- but at the first record of every frame the increment it applies
is wrong by an amount that is uniform modulo a cycle, per satellite, identical on every
instance, while the kernel's own anchor (REC_ANG0) steps regularly through the boundary. The
true Doppler step across a boundary is the same as inside the frame (the records are
contiguous and the seed does not change), so the fold the boundary SHOULD have received is
the frame's in-frame step; the difference is applied to the boundary cross-product before
its phase is read. Measured on sky before/after on a strong satellite: boundary step 0.145 ->
0.073 cycles rms, the in-frame level. The root (why the tracker's dcyc differs at record 0)
is not located yet; this makes the ADR walk-free of it without a node change.

THE COMMANDED INCREMENT MUST COVER OUR STEP. The advance over (prev -> hop) needs the Doppler
the replica ran at from prev, so only instances that were present at prev can vouch for it;
if none can, the arc ends -- an increment we cannot account for is a break, not a guess.
"""
import array
import cmath
import collections
import math
from fractions import Fraction

from gnss_broker.telem import (_HDR_BYTES, REC_DOPPLER, REC_P_RE, REC_P_IM, REC_P_ENERGY,
                               REC_PH_RE, REC_PH_IM, REC_TRIM_INC, REC_PHI0)

HPS = Fraction(390625, 2)   # F-engine hops per second (3.2e9 / 16384), exact
GRID_HOPS = 96 * 2048       # ~1.0066 s: a record hop common to every chain (see SatAdr.grid)


class SatAdr(object):
    """One satellite's running arc."""
    __slots__ = ("hop", "hop0", "s_prev", "adr", "trim", "res", "arc", "n", "n_inst",
                 "inst_prev", "breaks", "t", "grid", "dphi_intra", "bfix", "n_bfix")

    def __init__(self):
        self.hop = None        # hop of the last record folded
        self.hop0 = None       # first hop of the current arc
        self.s_prev = None     # sum of the previous record's S_i (diagnostic amplitude only)
        self.adr = 0.0         # DOPPLER-ONLY cycles since hop0 (commanded minus residual, minus
                               # the nominal f_c*dt, which is added back exactly at publish)
        self.trim = 0.0        # commanded carrier-trim cycles over the same arc
        self.res = 0.0         # residual cycles over the same arc
        self.arc = 0           # increments at every break
        self.n = 0             # records folded into this arc
        self.n_inst = 0        # instances behind the last record
        self.inst_prev = {}    # inst -> (hop, dop_hz, S_i) of that instance's last record seen
        self.breaks = 0        # arcs ended by a gap or an unaccountable increment
        self.t = 0.0           # wall time of the last fold
        # THE GRID SNAPSHOT (hop, adr, arc, n): the state at the newest record whose hop is a
        # multiple of GRID_HOPS. Every chain's records sit on hops that are multiples of the
        # record length, so grid hops are the SAME hops on every chain -- the epochs at which
        # two bands pair exactly. The live value above is whatever hop this cycle ended on.
        self.grid = None
        # the assembler's in-frame fold increment (median REC_PHI0 step, rad), kept as a
        # running median of the last in-frame records: what a frame boundary should have got
        self.dphi_intra = collections.deque(maxlen=24)
        self.bfix = 0.0      # cumulative boundary correction applied this arc, cycles
        self.n_bfix = 0


def fold_record(st, hop, per_inst, hpr, hps=HPS, max_gap_rec=3, min_inst=2):
    """Fold one record hop into `st`.

    per_inst: {inst: (dop_hz, trim_inc, S_i, phi0, r)} for the instances that despread this
    PRN at `hop`, S_i the instance's squared prompt AS EXPORTED, phi0 the assembler's NCO
    accumulator (REC_PHI0) and r the record's index in its frame. Returns True if the record
    extended the arc, False if it started a new one (or was dropped for having too few
    instances).
    """
    usable = {i: v for i, v in per_inst.items() if v[2] != 0}
    if len(usable) < min_inst:
        for i, v in per_inst.items():
            st.inst_prev[i] = (hop, v[0], v[2], v[3])
        return False
    prev = st.hop
    contiguous = (prev is not None and 0 < hop - prev <= max_gap_rec * hpr)
    dt = (hop - prev) / float(hps) if contiguous else 0.0
    # instances present at prev vouch for the step: their Doppler ran the replica from prev,
    # and their cross-product carries the received increment with the constant cancelled
    vouch = [(i, v) for i, v in usable.items()
             if contiguous and i in st.inst_prev and st.inst_prev[i][0] == prev
             and st.inst_prev[i][2] != 0]
    if len(vouch) >= min_inst:
        dcmd = sorted(st.inst_prev[i][1] * dt + v[1] for i, v in vouch)[len(vouch) // 2]
        trim = sorted(v[1] for _i, v in vouch)[len(vouch) // 2]
        P = sum(v[2] * st.inst_prev[i][2].conjugate() for i, v in vouch)
        # the assembler's fold increment this step (radians, wrapped), median over instances
        dphi = sorted(math.remainder(v[3] - st.inst_prev[i][3], 2.0 * math.pi)
                      for i, v in vouch)[len(vouch) // 2]
        r_idx = next(iter(usable.values()))[4]
        if r_idx == 0 and len(st.dphi_intra) >= 3:
            # a frame boundary: rotate the (squared) product by the fold it should have had
            eps = sorted(st.dphi_intra)[len(st.dphi_intra) // 2] - dphi
            P *= cmath.exp(-2j * eps)
            st.bfix += eps / (2.0 * math.pi)
            st.n_bfix += 1
        elif r_idx != 0:
            st.dphi_intra.append(dphi)
        dres = cmath.phase(P) / (4.0 * math.pi)
        st.adr += dcmd - dres
        st.res += dres
        st.trim += trim
        st.n += 1
        ok = True
    else:
        if st.hop is not None:
            st.breaks += 1
        st.arc += 1
        st.hop0 = hop
        st.adr = st.trim = st.res = 0.0
        st.bfix = 0.0
        st.n_bfix = 0
        st.n = 1
        ok = False
    st.s_prev = sum(v[2] for v in usable.values())
    st.hop = hop
    st.n_inst = len(usable)
    for i, v in per_inst.items():
        st.inst_prev[i] = (hop, v[0], v[2], v[3])
    if hop % GRID_HOPS == 0:
        st.grid = (hop, st.adr, st.arc, st.n)
    return ok


def records_of_frame(f, want):
    """{prn: {r: (dop_hz, trim_inc, S_i, phi0, r)}} for one sender's frame, decoded in one pass.

    S_i = H^2 + T^2 with H the head amplitude and T = A - H, AS EXPORTED (no phi0 rotation --
    see the module note); 0 when the record carries no prompt energy (silence, not a measurement).
    """
    idx = f._index()
    rows = {prn: p for prn, p in idx.items() if prn in want}
    if not rows:
        return {}
    stride = f.row_total
    a = array.array("f")
    a.frombytes(f._buf[_HDR_BYTES:_HDR_BYTES + f.n_rec * f.n_prn * stride * 4])
    out = {}
    for r in range(f.n_rec):
        if not f.has_record(r):
            continue
        base_r = r * f.n_prn * stride
        for prn, p in rows.items():
            b = base_r + p * stride
            e = a[b + REC_P_ENERGY]
            if e <= 0.0:
                continue
            A = complex(a[b + REC_P_RE] / e, a[b + REC_P_IM] / e)
            H = complex(a[b + REC_PH_RE] / e, a[b + REC_PH_IM] / e)
            if H == 0:
                H = A          # an unsegmented tracker: head == A, tail 0
            T = A - H
            S = H * H + T * T
            out.setdefault(prn, {})[r] = (float(a[b + REC_DOPPLER]), float(a[b + REC_TRIM_INC]), S,
                                          float(a[b + REC_PHI0]), r)
    return out


class FleetAdr(object):
    """Per-chain state: {prn: SatAdr} plus the newest window already folded."""

    def __init__(self, hpr=2048, max_gap_rec=3, min_inst=2):
        self.sats = {}
        self.last_win = None
        self.hpr = int(hpr)
        self.max_gap_rec = int(max_gap_rec)
        self.min_inst = int(min_inst)
        self.windows_lost = 0

    def fold_windows(self, client, chain, prns, now, lag=1):
        """Fold every window newer than the last one folded. Returns the number of windows."""
        want = set(int(p) for p in prns)
        wins = [w for w in client.windows(chain, lag=lag)
                if self.last_win is None or w > self.last_win]
        if self.last_win is not None and wins and wins[0] > self.last_win + 1:
            # windows the ring already dropped: the arcs will break honestly on the gap
            self.windows_lost += wins[0] - self.last_win - 1
        for w in wins:
            frames = client.frame_set(chain, w)
            by_hop = {}       # hop -> prn -> inst -> (dcmd, trim, S)
            for inst, f in frames.items():
                for prn, recs in records_of_frame(f, want).items():
                    for r, v in recs.items():
                        by_hop.setdefault(f.hop(r), {}).setdefault(prn, {})[inst] = v
            for hop in sorted(by_hop):
                for prn, per_inst in by_hop[hop].items():
                    st = self.sats.get(prn)
                    if st is None:
                        st = self.sats[prn] = SatAdr()
                    fold_record(st, hop, per_inst, self.hpr, max_gap_rec=self.max_gap_rec,
                                min_inst=self.min_inst)
                    st.t = now
            self.last_win = w
        # satellites that stopped arriving: forget them after a while so a return is a new arc
        for prn in [p for p, s in self.sats.items() if now - s.t > 30.0]:
            del self.sats[prn]
        return len(wins)

    def publish(self, carrier_hz, now):
        """{prn: fields} for the publisher: the Doppler-only ADR (what the fold accumulates),
        the full ADR with the exact nominal f_c*(hop - hop0)/hps added back, and the hops."""
        out = {}
        fc = Fraction(carrier_hz).limit_denominator(1)
        for prn, s in self.sats.items():
            if s.hop is None or s.n < 2:
                continue
            nominal = Fraction(s.hop - s.hop0) / HPS * fc
            out[prn] = {"dop_cycles": s.adr, "hop": s.hop, "hop0": s.hop0, "arc": s.arc,
                        "n_rec": s.n, "n_inst": s.n_inst, "trim_cycles": s.trim,
                        "res_cycles": s.res, "breaks": s.breaks,
                        "bfix_cycles": s.bfix, "n_bfix": s.n_bfix,
                        # the full received phase, the exact nominal added back to the
                        # Doppler-only accumulator (a double holds ~3e13 cycles to 4e-3)
                        "cycles": float(Fraction(s.adr) + nominal),
                        "age_s": round(now - s.t, 2)}
            if s.grid is not None and s.grid[2] == s.arc:
                gh, ga, _garc, gn = s.grid
                out[prn].update({"g_hop": gh, "g_dop_cycles": ga, "g_n_rec": gn,
                                 "g_cycles": float(Fraction(ga) + Fraction(gh - s.hop0) / HPS * fc)})
        return out


_STATE = {}


def stage_fleet_adr(ctx):
    """Fold this cycle's telemetry into the chain's fleet ADRs; {} when telemetry is absent."""
    if ctx.telem_client is None or not ctx.telem_chain:
        return {}
    fa = _STATE.get(ctx.chain_id)
    if fa is None:
        fa = _STATE[ctx.chain_id] = FleetAdr(hpr=int(getattr(ctx.args, "hops_per_record", 2048) or 2048))
    now = ctx.drp.now_w or 0.0
    fa.fold_windows(ctx.telem_client, ctx.telem_chain, set(ctx.seeds) | set(ctx.dllp.fleet or {}), now)
    return fa.publish(ctx.args.carrier_hz, now)
