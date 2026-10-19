"""Fleet ADR: the accumulated carrier phase at exact hops, from the telemetry records.

The observable. The tracker removes a commanded carrier phase Phi_cmd from the data and the
prompt carries what is left, so the received carrier phase is Phi_rx = Phi_cmd - arg(A)/2pi
(gnssRecord.hpp). Both halves are ambiguous on their own and both ambiguities die in the
per-record increment: the commanded advance over a record is known from the Doppler the
replica ran at, and the measured half is arg(A_k conj A_{k-1}), ~1e-3 cycles per record
while the carrier loop holds, against an unambiguous range of +-0.25.

Why not slot 15. The record's own commanded increment (REC_CPHASE) is float32, and on CHORD
the commanded phase is anchored at RF, so the increment is ~1.2e7 cycles per record and the
slot rounds it to whole cycles. An ADR folded from it (the C++ combiner's adr_cycles) is a
+-0.5 cycle per record random walk, not a carrier phase. The replica runs at a constant
f_c + dop within a record (propagate_seed per record, the re-pin step folded so the prompt
stays continuous), so the commanded advance over [k-1, k] is exactly
    (f_c + dop_{k-1}) * dt
with dop from slot 1 (float32, ~20 cycles of Doppler per record: ulp 2e-6) and dt from the
integer hops. The nominal f_c*dt is exact in rationals and kept out of the accumulator, which
therefore holds only the Doppler-integrated phase and never loses a double's precision.

The carrier trim is not in the export. The despread runs its replica at dop + ctrim, and the
assembler then rotates the exported prompt by the NCO phase it accumulates from f_nco = ctrim,
which takes the trim back out: what ships is the prompt relative to the model Doppler alone.
So the commanded advance the export is measured against is dop*dt, and slot 19 (the applied
trim's phase increment over the record) is carried here only as a record of what the carrier
loop commanded (`trim_cycles`). Adding it to the advance would put the trim into the ADR: a
geometry-free drift of lambda * d(trim)/dt between a trimmed band and an untrimmed one,
which no ionosphere makes.

Why it lives here. Every record in the telemetry carries its absolute hop and every chain
shares the one F-engine hop axis, so an ADR folded here is exact where it matters (pairing two
bands into the geometry-free combination needs equal epochs to ~1e-12 s, i.e. the same integer
hop) and common across all bands at once. Stamped with the arc's first hop too.

The increments are cross-products, not phasors. Every instance despreads the same seed with
the same fleet trim, so their prompts are one phasor seen through each instance's channel
subset, but each carries a different constant (the assembler's NCO origin is set per
instance), and that NCO is where the tracker folds the re-pin phase steps (Delta_dop*t_abs)
to keep the exported prompt continuous. REC_PHI0 is that accumulator, and it jumps by radians
every record. The comb's cross-instance rule (multiply by exp(+i*phi0)) puts instances on one
reference within a record; applied across records it re-injects the folded steps, and an ADR
built that way walks by hundreds of cycles. So the exported prompt is used as exported, and
each instance's per-record increment is
    arg( S_i(k) * conj(S_i(k-1)) ) / 4pi,    S_i = H_i^2 + T_i^2,
where H is the head amplitude and T = A - H: the instance's constant cancels in the product
and the fold stays folded. Squaring makes S sign-free (overlay pilots flip sign record to
record) and straddle-immune (both segments of a record that spans a secondary-chip transition
add in power). A record with no head (PH zero: a tracker that does not segment) is taken as
head == A, tail 0, and S reduces to A^2.

The fleet combine is a median of per-instance phases, not of increments. Consecutive
increments of one instance telescope: sum_k arg(S_i(k) conj S_i(k-1)) is the phase of S_i(k)
minus that of S_i(0), and the per-record noise does not accumulate. A fleet increment formed
first (the phase of sum_i S_i(k) conj S_i(k-1), or a median of the instances' increments)
does not telescope: its weights (|S_i|^2, which swing with where the secondary chip flips
inside each record) and its membership change every record, so each fleet increment carries
fresh noise and their sum is a random walk. So each vouching instance keeps its own
continuous residual phase, an instance that (re)joins is placed at the fleet's value, an
instance that slips by a half cycle (the squared phasor's period) is repaired by whole
half-cycles before the median is taken (see the note in fold_record), and the fleet residual
is the median of the instances' phases.

The broker does not re-fold the producer's phase. The assembler folds the replica's
per-record re-pin step into its NCO from a history shared across the pipeline's command
instances, so the exported prompt is continuous across frame boundaries as well as inside a
frame. A repair here keyed on REC_PHI0's per-record step cannot tell a wrong boundary fold
from a seed re-pin: at a re-pin the step is thousands of cycles modulo one, i.e. a random
angle, and a "repair" of a boundary that happens to be a re-pin record injects a random
half-cycle into the arc, a per-satellite rate error the loop's own estimators never see. A
wrong producer fold is a producer bug, fixed at the producer.

The commanded increment must cover our step. The advance over (prev -> hop) needs the Doppler
the replica ran at from prev, so only instances that were present at prev can vouch for it;
if fewer than min_inst can, the arc ends: an increment we cannot account for is a break, not
a guess.
"""
import array
import cmath
import math
from fractions import Fraction

from gnss_broker.telem import (
    _HDR_BYTES,
    REC_DOPPLER,
    REC_P_RE,
    REC_P_IM,
    REC_P_ENERGY,
    REC_PH_RE,
    REC_PH_IM,
    REC_TRIM_INC,
    REC_PROJ_COST,
    REC_PROJ_K,
    REC_PROJ_SRC,
)

HPS = Fraction(390625, 2)  # F-engine hops per second (3.2e9 / 16384), exact
GRID_HOPS = 96 * 2048  # ~1.0066 s: a record hop common to every chain (see SatAdr.grid)
GRID_KEEP = 4  # grid snapshots retained per satellite (see SatAdr.grid)
EPOCH_MARGIN_WIN = 64  # windows: TelemClient's epoch_margin (see FleetAdr.fold_windows)


class SatAdr(object):
    """One satellite's running arc."""

    __slots__ = (
        "hop",
        "hop0",
        "adr",
        "trim",
        "res",
        "arc",
        "n",
        "n_inst",
        "inst_prev",
        "inst_x",
        "t",
        "grid",
        "dark",
        "dark_silent",
        "noenergy",
        "lvl",
        "n_silent",
        "silent",
        "run",
        "proj",
        "last_break",
    )

    def __init__(self):
        self.hop = None  # hop of the last record folded
        self.hop0 = None  # first hop of the current arc
        self.adr = (
            0.0  # Doppler-only cycles since hop0 (commanded minus residual, minus
        )
        # the nominal f_c*dt, which is added back exactly at publish)
        self.trim = 0.0  # commanded carrier-trim cycles over the same arc
        self.res = 0.0  # residual cycles over the same arc: model minus received
        self.arc = 0  # increments at every break
        self.n = 0  # records folded into this arc
        self.n_inst = 0  # instances behind the last record
        self.inst_prev = (
            {}
        )  # inst -> (hop, dop_hz, S_i) of that instance's last record seen
        self.inst_x = (
            {}
        )  # inst -> that instance's own continuous residual phase, cycles
        self.t = 0.0  # wall time of the last fold
        # The grid snapshots [(hop, adr, arc, n), ...], oldest first, at most GRID_KEEP of
        # them: the state at each recent record whose hop is a multiple of GRID_HOPS. Every
        # chain's records sit on hops that are multiples of the record length, so grid hops
        # are the same hops on every chain: the epochs at which two bands pair exactly. The
        # live value above is whatever hop this cycle ended on.
        #
        # A history, not the newest only. The fold and the publish run once per broker cycle
        # (--interval, 2 s in production) against this ~1.0066 s grid, so a single slot would
        # publish every other grid hop, each chain its own alternate, and bands would pair on
        # few of them. Four snapshots span ~4 s, so consecutive 2 s cycles overlap by two and
        # no hop is lost unless a cycle is more than GRID_KEEP grid hops late.
        self.grid = []
        self.dark = 0  # consecutive records with too few usable instances since the last good fold
        self.dark_silent = 0  # of those, records that silence (not absence) made dark
        # records since the last good fold whose telemetry arrived with no prompt energy on any
        # instance (counted in FleetAdr.fold_windows: they never reach the fold)
        self.noenergy = 0
        self.lvl = {}  # inst -> [level dB, records learned] of its prompt power |S_i|
        self.n_silent = 0  # instances in the last record when it was silent
        self.silent = (
            False  # the satellite's prompts have lost the signal (see fold_record)
        )
        self.run = 0  # records in a row towards entering or leaving silence
        # this cycle's projection basis over the instances' records: [sum of b over records
        # with a basis in force, their count, the largest rank, the OR of the sources]
        self.proj = [0.0, 0, 0, 0]
        self.last_break = None  # why the last arc ended (see fold_record)


DARK_DB = 10.0  # the instances' prompt power this far below their level is down
R_MIN = 0.5  # agreement of the instances' phase steps below this is noise (signal ~1, noise ~0.25)
R_MIN_INST = 6  # instances needed for an agreement or level verdict
SILENT_MIN_REC = 5  # records in a row to enter (or, still down, leave) silence (~50 ms)
LVL_DECAY_DB_H = 20.0  # the level follows a fade slower than this (dB per hour)
LVL_EMA_REC = 10000  # records (~100 s) the level averages over, once warm
LVL_WARM_REC = 100  # records (~1 s) an instance must have been seen before it counts
BREAK_LOG_MIN_REC = 100  # an arc shorter than this (~1 s) ends without a log line


def fold_record(
    st,
    hop,
    per_inst,
    hpr,
    hps=HPS,
    max_gap_rec=3,
    min_inst=2,
    dark_db=DARK_DB,
    lvl_decay_db_h=LVL_DECAY_DB_H,
):
    """Fold one record hop into `st`.

    per_inst: {inst: (dop_hz, trim_inc, S_i, ...)} for the instances that despread this PRN at
    `hop`, S_i the instance's squared prompt as exported; any further slots are not read here.
    Returns True if the record extended the arc, False if it started a new one (or was dropped
    for having too few instances).
    """
    # The prompt must still carry the signal. A record whose prompts have lost the signal is
    # silent, not measuring: their phases are noise, and an increment folded from them walks
    # the arc by up to a quarter cycle per record while the arc key holds (a bright satellite
    # crossing the beam blinds every arc it captures this way). A silent record is treated
    # like an energy-less one, so the arc ends and a new one starts when the signal returns.
    # Silence needs both of: the instances' prompt power down dark_db against their own level
    # (median over instances), and their phase steps disagreeing (R, the mean resultant of the
    # unit steps S_i(k) conj S_i(k-1), below R_MIN; ~1 while the phase is measured, ~0.25 for
    # twelve noise prompts), for SILENT_MIN_REC records in a row, each verdict needing
    # R_MIN_INST instances. Power alone is not enough: a strong satellite's power can swing by
    # more than dark_db with its phase still measured, and a seed re-pin dips every instance
    # for one record. Silence ends when the power is back, or when the steps agree again for
    # SILENT_MIN_REC records. The level is a dB mean over ~LVL_EMA_REC records, learning from
    # every record whose phase is measured or whose power is within the margin, and decaying
    # lvl_decay_db_h under a dip it does not learn from. |S_i| is the prompt normalised by
    # its replica energy (records_of_frame).
    decay_db = lvl_decay_db_h * (hpr / float(hps)) / 3600.0
    z = 0j
    n_r = 0
    drops = []
    for i, v in per_inst.items():
        if v[2] == 0:
            continue
        pv = st.inst_prev.get(i)
        if pv is not None and pv[0] == hop - hpr and pv[2] != 0:
            d = v[2] * pv[2].conjugate()
            if d != 0:
                z += d / abs(d)
                n_r += 1
        h = st.lvl.get(i)
        if h is not None and h[1] >= LVL_WARM_REC:
            drops.append(10.0 * math.log10(abs(v[2])) - h[0])
    coherent = n_r >= R_MIN_INST and abs(z) >= R_MIN * n_r
    incoherent = n_r >= R_MIN_INST and not coherent
    down = len(drops) >= R_MIN_INST and sorted(drops)[len(drops) // 2] < -dark_db
    if not st.silent:
        st.run = st.run + 1 if (down and incoherent) else 0
        if st.run >= SILENT_MIN_REC:
            st.silent, st.run = True, 0
    elif not down:
        st.silent, st.run = False, 0
    else:
        st.run = st.run + 1 if coherent else 0
        if st.run >= SILENT_MIN_REC:
            st.silent, st.run = False, 0
    for i, v in per_inst.items():
        if v[2] == 0:
            continue
        p_db = 10.0 * math.log10(abs(v[2]))
        h = st.lvl.get(i)  # [level dB, records learned]
        if h is None:
            st.lvl[i] = [p_db, 1]
        elif coherent or h[1] < LVL_WARM_REC or p_db >= h[0] - dark_db:
            h[0] += (p_db - h[0]) / min(h[1] + 1, LVL_EMA_REC)
            h[1] += 1
        else:
            h[0] -= decay_db
    usable = {} if st.silent else {i: v for i, v in per_inst.items() if v[2] != 0}
    st.n_silent = sum(1 for v in per_inst.values() if v[2] != 0) if st.silent else 0
    if len(usable) < min_inst:
        # too few usable instances (no prompt, or silent): remember them; the gap is judged
        # when enough return
        st.dark += 1
        if st.silent:
            st.dark_silent += 1
        for i, v in per_inst.items():
            st.inst_prev[i] = (hop, v[0], v[2])
        return False
    prev = st.hop
    contiguous = prev is not None and 0 < hop - prev <= max_gap_rec * hpr
    dt = (hop - prev) / float(hps) if contiguous else 0.0
    # instances present at prev vouch for the step: their Doppler ran the replica from prev,
    # and their cross-product carries the received increment with the constant cancelled
    vouch = [
        (i, v)
        for i, v in usable.items()
        if contiguous
        and i in st.inst_prev
        and st.inst_prev[i][0] == prev
        and st.inst_prev[i][2] != 0
    ]
    if len(vouch) >= min_inst:
        dcmd = sorted(st.inst_prev[i][1] * dt for i, _v in vouch)[len(vouch) // 2]
        trim = sorted(v[1] for _i, v in vouch)[len(vouch) // 2]
        # each voucher advances its own phase; the fleet value is the median of the phases
        for i, v in vouch:
            x = st.inst_x.get(i, st.res)
            st.inst_x[i] = x + cmath.phase(v[2] * st.inst_prev[i][2].conjugate()) / (
                4.0 * math.pi
            )
        xs = sorted(st.inst_x[i] for i, _v in vouch)
        res = xs[len(xs) // 2]
        # A slip is repaired before the final median is taken, not re-seated after. The
        # squared phasor's increment is ambiguous by half a cycle, so a weak instance whose
        # prompt phase moves more than a quarter cycle in one record steps its own sum by
        # exactly +-0.5. Re-seating it to a median taken with the slipped values inside the
        # sorted list moves that median (one rank per slipper, several when a weak satellite's
        # instances slip together), and the re-seated copies then hold those ranks, so the
        # step stays. Undoing each slip by whole half-cycles puts the instance back at its own
        # rank first, so the median never sees it move; re-seating remains only for an
        # excursion that is not a multiple of a half cycle.
        repaired = False
        for i, _v in vouch:
            d = st.inst_x[i] - res
            if abs(d) > 0.3:
                st.inst_x[i] -= round(d * 2.0) / 2.0
                repaired = True
        if repaired:
            xs = sorted(st.inst_x[i] for i, _v in vouch)
            res = xs[len(xs) // 2]
        for i, _v in vouch:
            if abs(st.inst_x[i] - res) > 0.3:  # not a slip: an excursion, re-seat it
                st.inst_x[i] = res
        for i in usable:
            if i not in st.inst_x:  # (re)joining: adopt the fleet's phase
                st.inst_x[i] = res
        st.dark = 0
        st.dark_silent = 0
        st.noenergy = 0
        dres = res - st.res
        st.adr += dcmd - dres
        st.res += dres
        st.trim += trim
        st.n += 1
        ok = True
    else:
        if st.hop is not None:
            # Say why. Every arc break costs the carrier level of this satellite, and a break
            # on every satellite of every chain in the same second is a broker or node event,
            # not sky. The break records the gap, the records folded with too few usable
            # instances (dark, silent ones counted apart) and the instances' last-seen hops,
            # which say whether the telemetry stopped or the trackers did. A record with no
            # prompt energy on any instance never reaches the fold (records_of_frame leaves it
            # out): it is in the gap, and counted apart as noenergy when its telemetry arrived.
            _gap = (hop - prev) / float(hpr) if prev is not None else float("nan")
            _seen = sorted(st.inst_prev.get(i, (None,))[0] or 0 for i in per_inst)
            st.last_break = {
                "hop": hop,
                "n_rec": st.n,
                "gap_rec": _gap,
                "dark_rec": st.dark,
                "silent_rec": st.dark_silent,
                "noenergy_rec": st.noenergy,
                "n_usable": len(usable),
                "n_vouch": len(vouch),
                "inst_lag_rec": [(hop - h) / float(hpr) for h in _seen[-3:]],
            }
        st.dark = 0
        st.dark_silent = 0
        st.noenergy = 0
        st.arc += 1
        st.hop0 = hop
        st.adr = st.trim = st.res = 0.0
        st.inst_x = {i: 0.0 for i in usable}
        st.n = 1
        ok = False
    st.hop = hop
    st.n_inst = len(usable)
    for i, v in per_inst.items():
        st.inst_prev[i] = (hop, v[0], v[2])
    for i in [i for i in st.inst_x if i not in usable]:
        del st.inst_x[i]  # gone: it rejoins at the fleet's phase
    if hop % GRID_HOPS == 0:
        st.grid.append((hop, st.adr, st.arc, st.n))
        del st.grid[:-GRID_KEEP]
    return ok


def records_of_frame(f, want):
    """{prn: {r: (dop_hz, trim_inc, S_i, proj_cost, proj_k, proj_src)}} for one sender's frame,
    decoded in one pass. The last three are the projection slots (gnssRecord.hpp), carried for
    the published row only; the fold reads the first three.

    S_i = H^2 + T^2 with H the head amplitude and T = A - H, as exported (no phi0 rotation; see
    the module note). A record with no prompt energy (P_energy <= 0) is left out.
    """
    idx = f._index()
    rows = {prn: p for prn, p in idx.items() if prn in want}
    if not rows:
        return {}
    stride = f.row_total
    a = array.array("f")
    a.frombytes(f._buf[_HDR_BYTES : _HDR_BYTES + f.n_rec * f.n_prn * stride * 4])
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
                H = A  # an unsegmented tracker: head == A, tail 0
            T = A - H
            S = H * H + T * T
            out.setdefault(prn, {})[r] = (
                float(a[b + REC_DOPPLER]),
                float(a[b + REC_TRIM_INC]),
                S,
                float(a[b + REC_PROJ_COST]),
                int(a[b + REC_PROJ_K]),
                int(a[b + REC_PROJ_SRC]),
            )
    return out


class FleetAdr(object):
    """Per-chain state: {prn: SatAdr} plus the newest window already folded."""

    def __init__(self, hpr=2048, max_gap_rec=3, min_inst=2, log=None):
        self.log = log
        self.sats = {}
        self.last_win = None
        self.hpr = int(hpr)
        self.max_gap_rec = int(max_gap_rec)
        self.min_inst = int(min_inst)

    def fold_windows(self, client, chain, prns, now, lag=1):
        """Fold every window newer than the last one folded. Returns the number of windows."""
        want = set(int(p) for p in prns)
        ws = client.windows(chain, lag=lag)
        # An F-engine re-base restarts the window counter and the client re-anchors its ring on
        # the new epoch, so the newest window held drops far below the last one folded and no
        # window would pass the filter below until the counter climbed back. Every arc ends
        # there and the fold starts over, as after a broker restart.
        if (
            self.last_win is not None
            and ws
            and ws[-1] < self.last_win - EPOCH_MARGIN_WIN
        ):
            if self.log is not None:
                self.log(
                    "FADR EPOCH RESET %s: newest window %d is far below the last folded %d "
                    "(an F-engine re-base) -- %d arc(s) ended, folding restarts"
                    % (chain, ws[-1], self.last_win, len(self.sats))
                )
            self.last_win = None
            self.sats.clear()
        wins = [w for w in ws if self.last_win is None or w > self.last_win]
        for st in self.sats.values():
            st.proj = [0.0, 0, 0, 0]
        # windows the ring already dropped leave a hop gap: the arcs break honestly on it
        for w in wins:
            frames = client.frame_set(chain, w)
            by_hop = {}  # hop -> prn -> inst -> the records_of_frame tuple
            for inst, f in frames.items():
                for r in range(f.n_rec):
                    if f.has_record(r):
                        by_hop.setdefault(f.hop(r), {})
                for prn, recs in records_of_frame(f, want).items():
                    for r, v in recs.items():
                        by_hop.setdefault(f.hop(r), {}).setdefault(prn, {})[inst] = v
            for hop in sorted(by_hop):
                # Telemetry for this hop arrived without a record of the satellite carrying
                # prompt energy: its rows had none, or GnssTelemPack shipped no row for it (a
                # PRN not despread at the window's first record). The break line counts these
                # apart from the gap, which also covers records that never arrived.
                for prn, st in self.sats.items():
                    if prn in want and prn not in by_hop[hop] and st.hop is not None:
                        st.noenergy += 1
                for prn, per_inst in by_hop[hop].items():
                    st = self.sats.get(prn)
                    if st is None:
                        st = self.sats[prn] = SatAdr()
                    for v in per_inst.values():
                        if v[3] >= 0.0:
                            st.proj[0] += v[3]
                            st.proj[1] += 1
                        st.proj[2] = max(st.proj[2], v[4])
                        st.proj[3] |= v[5]
                    fold_record(
                        st,
                        hop,
                        per_inst,
                        self.hpr,
                        max_gap_rec=self.max_gap_rec,
                        min_inst=self.min_inst,
                    )
                    st.t = now
            self.last_win = w
        # satellites that stopped arriving: forget them after 30 s so a return is a new arc
        for prn in [p for p, s in self.sats.items() if now - s.t > 30.0]:
            del self.sats[prn]
        # One FADR BREAK line per cycle naming every arc that broke and why: a fleet-wide break
        # reads as one line with every PRN and the same gap, which is the signature to look
        # for. A seeded satellite nobody tracks folds noise in short arcs, so only an arc that
        # lived BREAK_LOG_MIN_REC records gets a line.
        broke = [
            (p, s.last_break)
            for p, s in self.sats.items()
            if s.last_break is not None
            and s.last_break.get("logged") is None
            and s.last_break.get("n_rec", BREAK_LOG_MIN_REC) >= BREAK_LOG_MIN_REC
        ]
        if broke and self.log is not None:
            parts = []
            for p, b in sorted(broke):
                b["logged"] = True
                parts.append(
                    "%d:gap%.0f/dark%d/silent%d/noenergy%d/usable%d/vouch%d/lag%s"
                    % (
                        p,
                        b["gap_rec"],
                        b["dark_rec"],
                        b.get("silent_rec", 0),
                        b.get("noenergy_rec", 0),
                        b["n_usable"],
                        b["n_vouch"],
                        ",".join("%.0f" % x for x in b["inst_lag_rec"]),
                    )
                )
            self.log(
                "FADR BREAK %s: %d arc(s) ended -- prn:gap(records)/dark(records with too few "
                "usable instances)/silent(of those, the prompts had lost the signal)/noenergy("
                "records that arrived with no prompt energy)/usable/vouchers/"
                "instance-lag(records) %s" % (chain, len(broke), " ".join(parts))
            )
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
            out[prn] = {
                "dop_cycles": s.adr,
                "hop": s.hop,
                "hop0": s.hop0,
                "arc": s.arc,
                "n_rec": s.n,
                "n_inst": s.n_inst,
                "n_silent": s.n_silent,
                # the bright-satellite projection over this cycle's records: b the mean cost
                # on this satellite where a basis was in force (-1 none), k the largest rank,
                # src the OR of the sources (1 own row, 2 sibling chain, 4 probe stack)
                "proj_b": s.proj[0] / s.proj[1] if s.proj[1] else -1.0,
                "proj_k": s.proj[2],
                "proj_src": s.proj[3],
                "trim_cycles": s.trim,
                "res_cycles": s.res,
                # the full received phase, the exact nominal added back to the
                # Doppler-only accumulator (a double holds ~3e13 cycles to 4e-3)
                "cycles": float(Fraction(s.adr) + nominal),
                "age_s": round(now - s.t, 2),
            }
            # Only this arc's snapshots: a break resets the accumulator, so an older arc's
            # phase is not continuous with the current one and must never pair against it.
            g = [t for t in s.grid if t[2] == s.arc]
            if g:
                gh, ga, _garc, gn = g[-1]
                out[prn].update(
                    {
                        "g_hop": gh,
                        "g_dop_cycles": ga,
                        "g_n_rec": gn,
                        "g_cycles": float(
                            Fraction(ga) + Fraction(gh - s.hop0) / HPS * fc
                        ),
                    }
                )
                # g_hist: [hop, dop_cycles, cycles, n_rec] for each retained grid hop of this
                # arc, newest last; the g_* scalars above are its last entry.
                out[prn]["g_hist"] = [
                    [h, a, float(Fraction(a) + Fraction(h - s.hop0) / HPS * fc), n]
                    for h, a, _arc, n in g
                ]
        return out


_STATE = {}


def stage_fleet_adr(ctx):
    """Fold this cycle's telemetry into the chain's fleet ADRs; {} when telemetry is absent."""
    if ctx.telem_client is None or not ctx.telem_chain:
        return {}
    fa = _STATE.get(ctx.chain_id)
    if fa is None:
        from gnss_broker.transport import _log

        fa = _STATE[ctx.chain_id] = FleetAdr(log=_log)
    now = ctx.drp.now_w or 0.0
    fa.fold_windows(
        ctx.telem_client,
        ctx.telem_chain,
        set(ctx.seeds) | set(ctx.dllp.fleet or {}),
        now,
    )
    return fa.publish(ctx.args.carrier_hz, now)
