"""Streaming GPS L1C-D CNAV-2 decoder for the broker (the L1C-D data sibling's nav decoder).

Mirrors bcnav3_predictor.py: ingest the combiner's per-PRN nav_obs symbol emits, stitch them into
an absolute-time symbol timeline and decode CRC-valid frames. gps_cnav2.py is the codec (SF1-TOI
framesync + block deinterleave + systematic LDPC extract + CRC-24Q); it has no SF2 ephemeris field
table, so there is no ephemeris to serve.

Like B-CNAV1 the frame is 18 s (1800 symbols) -- long, and find_frames needs 1852 contiguous
symbols of one sign (frame + the next frame's SF1). The L1C-D combiner emits one symbol per
10 ms (navwipe_bit_records 1 -> br=1, rec_dt=0.010); integration_length 100 gives each emit a
1 s window, longer than the broker's --interval, so the windows of successive polls overlap and
no symbol is lost between them. Each emit stores only the symbols past the newest one held, and
the cache is bounded in symbols (SYM_MAX), since a run of overlapping emits holds far fewer
symbols than emits times their length. br=1 has no slot jitter; the stitcher still anchors slots
to a per-PRN grid and salvages gappy emits, as inav_predictor does.

Polarity. Each emit resolves its +-1 ambiguity on its own, so the emits are chained as
navbit_predictor chains LNAV: rot = e^{-i*theta0} follows the carrier, and an odd number of
pi-steps in theta0 from the last emit flips this emit's sign relative to it. One wrong link
flips everything after it; the frame that straddles it fails, and a boundary-flip repair
(_repair) finds the link.
"""

import math
import time

import numpy as np

import gps_cnav2 as C2

SYM_S = C2.SYM_S  # 0.010
FRAME = C2.FRAME_SYMS  # 1800
NEED = FRAME + C2.N_SF1  # 1852 (frame + next SF1) for a sync+decode
# Symbols held per PRN: two frames with their next SF1s, so a frame stays whole in the cache for
# about a frame time after its next SF1 arrives, and the find scan stays bounded.
SYM_MAX = 2 * NEED


class _PrnState:
    def __init__(self):
        self.emits = {}  # slot0 -> np.array(0/1), the new symbols of one nav_obs emit
        self.grid0 = None
        self.last_obs = None
        self.theta = None  # last emit's theta0 (rad, (-pi/2, pi/2])
        self.pol = 1  # chained polarity applied to incoming emits
        self.sf2 = None  # (bits[600], t_decoded)
        self.toi = None
        self.n_frames = 0
        # frame start slots the repair has searched: a complete frame's stored symbols do not
        # change, so each is searched once
        self.tried = set()
        self.last_decode = 0.0


class Cnav2Predictor:
    def __init__(self):
        self._p = {}
        self._min_gap = 1.0

    # ----------------------------------------------------------------- ingest
    def ingest(self, prn, obs):
        st = self._p.setdefault(prn, _PrnState())
        try:
            utc_ref = float(obs["utc_ref"])
            rec_dt = float(obs["rec_dt"])
            phase = int(obs["phase"])
            br = int(obs["br"])
            rot = complex(float(obs.get("rot_re", 1.0)), float(obs.get("rot_im", 0.0)))
            pairs = obs["bits"]
        except (KeyError, TypeError, ValueError):
            return
        if rec_dt <= 0 or br <= 0 or not pairs:
            return
        key = (utc_ref, phase, br)
        if key == st.last_obs:
            return
        st.last_obs = key
        # theta0 = -arg(rot), reduced mod pi: an odd pi-step from the last emit flips the chain
        theta = -math.atan2(rot.imag, rot.real)
        if st.theta is not None and round((theta - st.theta) / math.pi) % 2:
            st.pol = -st.pol
        st.theta = theta
        sym_s = br * rec_dt
        t0 = utc_ref - phase * rec_dt
        if st.grid0 is None:
            st.grid0 = t0
        base = int(round((t0 - st.grid0) / sym_s))
        held = max((s0 + len(a) for s0, a in st.emits.items()), default=None)
        # nav_obs signs are +-1; CNAV-2 symbols are 0/1 (SF1 patterns + LDPC), so map -1->1, +1->0
        # after the chain's polarity. Successive emits overlap: only the slots past the newest one
        # held are stored, so each symbol is held once.
        m = {}
        for bi, sg in pairs:
            bi, sg = int(bi), int(sg)
            if sg not in (1, -1) or (held is not None and base + bi < held):
                continue
            m[base + bi] = 0 if sg * st.pol > 0 else 1
        if len(m) < 2:
            return
        slots = sorted(m)
        seg = 0
        for i in range(len(slots)):
            if i + 1 == len(slots) or slots[i + 1] != slots[i] + 1:
                s0, s1 = slots[seg], slots[i]
                if s1 - s0 + 1 >= 2:
                    st.emits[s0] = np.array(
                        [m[s] for s in range(s0, s1 + 1)], dtype=np.int8
                    )
                seg = i + 1
        n = sum(len(a) for a in st.emits.values())
        if n > SYM_MAX:
            while n > SYM_MAX and len(st.emits) > 1:
                n -= len(st.emits.pop(min(st.emits)))
            lo = min(st.emits)
            st.tried = {s for s in st.tried if s >= lo}
        now = time.time()
        if self._min_gap and now - st.last_decode < self._min_gap:
            return
        st.last_decode = now
        self._decode(prn, st)

    # ----------------------------------------------------------------- decode
    def _runs(self, st):
        """Contiguous runs of the cached emits: (start slot, symbols, cuts), cuts holding
        (index into symbols, emit slot0) where each later emit's first new symbol lands."""
        if not st.emits:
            return []
        runs = []
        cur0 = None
        cur = None
        cuts = []
        for s0 in sorted(st.emits):
            a = st.emits[s0]
            if cur is None:
                cur0, cur = s0, a.copy()
                continue
            end = cur0 + len(cur)
            if s0 <= end:
                ov = end - s0
                if ov < len(a):
                    cuts.append((len(cur), s0))
                    cur = np.concatenate([cur, a[ov:]])
            else:
                runs.append((cur0, cur, cuts))
                cur0, cur, cuts = s0, a.copy(), []
        runs.append((cur0, cur, cuts))
        return runs

    def _decode(self, prn, st):
        for run0, syms, cuts in self._runs(st):
            if len(syms) < NEED:
                continue
            found = set()
            for i, toi, sf2, _sf3 in C2.find_frames(syms):
                found.add(i)
                self._store(st, toi, sf2)
            self._repair(st, run0, syms, cuts, found)

    def _store(self, st, toi, sf2):
        st.n_frames += 1
        st.toi = toi
        st.sf2 = (sf2, time.time())

    def _repair(self, st, run0, syms, cuts, found):
        """CRC-guided boundary-flip repair of the run's frames that find_frames did not decode.

        A complete frame whose SF1 is in the run but which fails sync or CRC may straddle one
        wrong chain link: an emit whose sign came out flipped, and every later emit with it.
        Flip the frame and its next SF1 from each emit boundary inside them and take the first
        flip under which find_frames decodes it. That emit and every later one are flipped in
        the cache, the chain with them. syms is updated in place for the run's later frames.
        """
        lut = C2._sf1_lut()
        for i in range(len(syms) - NEED + 1):
            if i in found or run0 + i in st.tried:
                continue
            if syms[i : i + C2.N_SF1].tobytes() not in lut:
                continue
            st.tried.add(run0 + i)
            for b, s0 in cuts:
                if not i < b < i + NEED:
                    continue
                w = syms[i : i + NEED].copy()
                w[b - i :] ^= 1
                got = next(C2.find_frames(w), None)
                if got is None:
                    continue
                for s in st.emits:
                    if s >= s0:
                        st.emits[s] = st.emits[s] ^ 1
                st.pol = -st.pol
                syms[b:] ^= 1
                self._store(st, got[1], got[2])
                break

    # ----------------------------------------------------------------- serve
    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_frames,
            "words": st.n_frames,
            "toi": st.toi,
            "have": ([2, 3] if st.sf2 else []),
        }


# ------------------------------------------------------------------- self-test
def _selftest():
    """Build CNAV-2 frames (systematic SF2/SF3 + CRC + arbitrary parity, SF1, interleave), feed as
    overlapping 1 s emits of br=1 10 ms symbols, recover the SF2 + TOI through the full
    predictor."""
    import gps_cnav2 as C
    from galileo_inav import crc24q

    rng = np.random.default_rng(4)
    prn = 23

    def crc_append(k):
        m = rng.integers(0, 2, k - 24).tolist()
        c = crc24q(m)
        return m + [(c >> (23 - j)) & 1 for j in range(24)]

    def build(toi):
        sf2 = crc_append(C.K_SF2)
        sf3 = crc_append(C.K_SF3)
        deint = np.array(
            sf2
            + rng.integers(0, 2, C.N_SF2 - C.K_SF2).tolist()
            + sf3
            + rng.integers(0, 2, C.N_SF3 - C.K_SF3).tolist(),
            dtype=np.int8,
        )
        il = deint.reshape(C.IL_COLS, C.IL_ROWS).T.ravel()
        return np.hstack([C._sf1(toi), il]).astype(np.int8), sf2, sf3

    # two consecutive frames so find_frames sees frame + next SF1
    f0, sf2_0, sf3_0 = build(100)
    f1, _, _ = build(101)
    stream = np.hstack([f0, f1]).astype(np.int8)  # 3600 symbols

    # 1 s emits (100 symbols) every 25 symbols, as a 0.25 s broker poll reads the rolling
    # combiner, made as the combiner makes them: the carrier phase phi drifts 0.1 rad per emit,
    # each emit reports theta0 = phi reduced to [-pi/2, pi/2) as rot = e^{-i*theta0} and takes
    # its signs against theta0, so its sign flips whenever phi crosses an odd multiple of pi/2.
    # A 2 rad phase step at the emit starting on symbol 1000 is a link the chain gets wrong; the
    # boundary-flip repair must find it.
    pred = Cnav2Predictor()
    pred._min_gap = 0.0
    e = 0
    ELEN, STEP = 100, 25
    phi = 2 * math.pi * rng.random()
    while e + ELEN <= len(stream):
        n = math.floor(phi / math.pi + 0.5)
        theta0, sgn = phi - math.pi * n, 1 - 2 * (n % 2)
        obs = {
            "utc_ref": e * SYM_S,
            "rec_dt": 0.010,
            "phase": 0,
            "br": 1,
            "rot_re": math.cos(theta0),
            "rot_im": -math.sin(theta0),
            "bits": [[i, int((1 - 2 * int(stream[e + i])) * sgn)] for i in range(ELEN)],
        }
        pred.ingest(prn, obs)
        e += STEP
        phi += 2.0 if e == 1000 else 0.1

    h = pred.health(prn)
    print("health:", h)
    ok = bool(h and h["words"] >= 1 and h["toi"] == 100 and st_sf2(pred, prn) == sf2_0)
    print("PASS" if ok else "FAIL")
    return ok


def st_sf2(pred, prn):
    st = pred._p.get(prn)
    return st.sf2[0] if st and st.sf2 else None


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)
