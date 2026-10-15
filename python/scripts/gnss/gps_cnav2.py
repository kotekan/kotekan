"""GPS L1C-D CNAV-2 codec: frame symbols <-> nav bits (IS-GPS-800).

CNAV-2 is protected by a binary LDPC code. A CNAV-2 frame is 1800 symbols = 18 s at 100 sps (a
10 ms L1C-D record = one symbol):

  Subframe 1 (52 syms): TOI (Time Of Interval, 9 bits) as a fixed per-TOI pattern -- 1 explicit
      bit9 + a 51-symbol 8-stage-LFSR sequence XOR bit9. Used only for frame sync + polarity.
  Subframes 2+3 (1748 syms): block-interleaved (write 46x38 by rows, read by columns), then
      Subframe 2 = LDPC(1200,600) -> 600 info bits (ephemeris + clock + CRC-24Q);
      Subframe 3 = LDPC(548,274)  -> 274 info bits (page-typed: UTC/iono/EOP/... + CRC-24Q).

The LDPC is systematic (info = the first k bits of the codeword, as PocketSDR's decode_B_LDPC
returns dblk[:m]). The decode relies on the deep-integration SNR giving an error-free codeword:
deinterleave, take the systematic prefix, CRC-check, with no LDPC iteration, so a frame with an
error in its info bits fails the CRC and is dropped; a binary min-sum pass would extend the
decode to weaker satellites. Frame geometry + the SF1 LFSR are transcribed from PocketSDR
(sdr_nav.py decode_L1CD/decode_CNV2/sync_CNV2_frame, T.Takasu).

The frame decode (systematic extract + CRC) is self-validating: the CRC is the check. The SF2
ephemeris (the CNAV Keplerian set) has no field table, so this codec yields no ephemeris.
"""

import numpy as np

from galileo_inav import crc24q, _uint  # CRC-24Q + MSB-first bit->uint (shared)

SYM_S = 0.010  # L1C-D symbol = 10 ms (100 sps), one 10 ms BOC(1,1) code period
FRAME_SYMS = 1800  # 18 s frame
N_SF1 = 52  # subframe 1 (TOI) symbols
IL_ROWS, IL_COLS = 46, 38  # block interleaver: syms[52:].reshape(46,38).T.ravel()
N_SF2, K_SF2 = 1200, 600  # SF2 LDPC(1200,600)
N_SF3, K_SF3 = 548, 274  # SF3 LDPC(548,274)
N_TOI = 400  # TOI counts 18 s frames, 0..399 within each two-hour interval


# ------------------------------------------------------------------ SF1 (TOI) sync pattern
def _rev_reg(R, n):
    rr = 0
    for i in range(n):
        rr = (rr << 1) | ((R >> i) & 1)
    return rr


def _lfsr_bits(N, R, tap, n):
    """PocketSDR LFSR (n-stage): returns N outputs, bit = R&1 (0/1)."""
    out = np.zeros(N, dtype=np.int8)
    for i in range(N):
        out[i] = R & 1
        fb = bin(R & tap).count("1") & 1
        R = (fb << (n - 1)) | (R >> 1)
    return out


_CNV2_SF1 = {}


def _sf1(toi):
    """52-symbol (0/1) subframe-1 pattern for TOI `toi` (== PocketSDR CNV2_SF1[toi])."""
    if toi not in _CNV2_SF1:
        # PocketSDR: code = LFSR(51, rev_reg(t&0xFF,8), 0b10011111, 8) as CHIP(+1/-1); the SF1
        # data symbols are ((code+1)//2) ^ bit9, prefixed by the explicit bit9. _lfsr_bits returns
        # the bit R&1 directly; CHIP[b]=+1 if b==0 else -1, so (CHIP+1)//2 = 1-b -> the SF1 bit is
        # (1-b)^bit9.
        b = _lfsr_bits(51, _rev_reg(toi & 0xFF, 8), 0b10011111, 8)
        bit9 = (toi >> 8) & 1
        data = ((1 - b) ^ bit9).astype(np.int8)
        _CNV2_SF1[toi] = np.hstack([np.int8(bit9), data])
    return _CNV2_SF1[toi]


def sync_cnv2(syms1852):
    """Given 1852 hard symbols (0/1) [1800 frame + next-frame's 52-sym SF1], find the TOI +
    polarity. Returns (toi, rev) or (None, None). rev=0 normal, 1 inverted (carrier ambiguity)."""
    s = np.asarray(syms1852, dtype=np.int8)
    if len(s) < 1852:
        return None, None
    head, tail = s[:N_SF1], s[-N_SF1:]
    for toi in range(N_TOI):
        sf1, sfn = _sf1(toi), _sf1((toi + 1) % N_TOI)
        if np.array_equal(head, sf1) and np.array_equal(tail, sfn):
            return toi, 0
        if np.array_equal(head, sf1 ^ 1) and np.array_equal(tail, sfn ^ 1):
            return toi, 1
    return None, None


# ------------------------------------------------------------------ deinterleave + CRC + decode
def deinterleave(syms1748):
    """1748 SF2+SF3 symbols -> deinterleaved (write 46x38 by rows, read by columns)."""
    return np.asarray(syms1748, dtype=np.int8).reshape(IL_ROWS, IL_COLS).T.ravel()


def _crc_ok(bits):
    """CNAV-2 CRC-24Q: the trailing 24 bits are the CRC over the preceding message bits. The
    IS-GPS-800 byte-left-pad is a no-op for the init-0 CRC-24Q, so this is the plain check."""
    b = [int(x) & 1 for x in bits]
    return crc24q(b[:-24]) == _uint(b[-24:])


def decode_frame(syms1852, toi=None, rev=None):
    """1852 hard symbols -> (toi, SF2_bits[600], SF3_bits[274]) if both CRCs pass, else None.
    Systematic LDPC extract (error-free-codeword assumption); CRC-24Q is the filter."""
    s = np.asarray(syms1852, dtype=np.int8)
    if toi is None:
        toi, rev = sync_cnv2(s)
        if toi is None:
            return None
    frame = s[:FRAME_SYMS] ^ (rev & 1)  # de-rotate to normal polarity
    d = deinterleave(frame[N_SF1:])  # 1748 -> SF2(1200) + SF3(548)
    sf2 = d[:K_SF2].tolist()  # systematic prefix = info bits
    sf3 = d[N_SF2 : N_SF2 + K_SF3].tolist()
    if not (_crc_ok(sf2) and _crc_ok(sf3)):
        return None
    return toi, sf2, sf3


_SF1_LUT = None


def _sf1_lut():
    """{52-sym SF1 pattern bytes -> (toi, rev)} for O(1) frame-start detection over a long run."""
    global _SF1_LUT
    if _SF1_LUT is None:
        _SF1_LUT = {}
        for toi in range(N_TOI):
            p = _sf1(toi)
            _SF1_LUT[p.tobytes()] = (toi, 0)
            _SF1_LUT[(p ^ 1).tobytes()] = (toi, 1)
    return _SF1_LUT


def find_frames(syms):
    """Slide over a hard-symbol run; at each offset use the SF1 dict as a cheap membership filter
    ("could this be a frame start?"), then sync_cnv2 for the actual (toi, rev) and decode. Yields
    (offset, toi, SF2, SF3) per CRC-OK frame. The dict alone can't name the TOI -- SF1 patterns
    collide under inversion (_sf1(a) == _sf1(b)^1), so head-only lookup is ambiguous; sync_cnv2's
    head+tail check over all TOIs disambiguates. Head-hits are ~0 at random offsets, so the
    O(400) sync runs only near true frame starts."""
    s = np.asarray(syms, dtype=np.int8)
    n = len(s)
    lut = _sf1_lut()
    i = 0
    while i + FRAME_SYMS + N_SF1 <= n:
        if s[i : i + N_SF1].tobytes() in lut:
            toi, rev = sync_cnv2(s[i : i + FRAME_SYMS + N_SF1])
            if toi is not None:
                r = decode_frame(s[i : i + FRAME_SYMS + N_SF1], toi, rev)
                if r is not None:
                    yield i, r[0], r[1], r[2]
        i += 1


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(11)
    fails = 0

    # 1. SF1 patterns: distinct per TOI, reversible bit9, exact length
    okp = all(len(_sf1(t)) == N_SF1 for t in range(N_TOI))
    seen = {tuple(_sf1(t).tolist()) for t in range(0, N_TOI, 7)}
    okp = okp and len(seen) == len(range(0, N_TOI, 7))
    print("1. SF1 TOI patterns (length + distinct): %s" % ("OK" if okp else "FAIL"))
    fails += 0 if okp else 1

    # 2. frame roundtrip: build a frame from chosen SF2/SF3 info (systematic prefix + CRC + random
    #    parity), SF1 for a TOI, interleave -> decode recovers the info + TOI, both polarities.
    def crc_append(msg_bits, total):
        m = list(msg_bits[: total - 24])
        c = crc24q(m)
        return m + [(c >> (23 - k)) & 1 for k in range(24)]

    p2 = 0
    for toi in (0, 123, 399):
        for rev in (0, 1):
            sf2 = crc_append(rng.integers(0, 2, K_SF2).tolist(), K_SF2)
            sf3 = crc_append(rng.integers(0, 2, K_SF3).tolist(), K_SF3)
            cw2 = (
                sf2 + rng.integers(0, 2, N_SF2 - K_SF2).tolist()
            )  # systematic + arbitrary parity
            cw3 = sf3 + rng.integers(0, 2, N_SF3 - K_SF3).tolist()
            deint = np.array(cw2 + cw3, dtype=np.int8)  # 1748
            il = deint.reshape(IL_COLS, IL_ROWS).T.ravel()  # inverse of deinterleave
            frame = np.hstack([_sf1(toi), il]).astype(np.int8)
            stream = np.hstack([frame, _sf1((toi + 1) % N_TOI)]) ^ (rev & 1)
            r = decode_frame(stream)
            if r is None or r[0] != toi or r[1] != sf2 or r[2] != sf3:
                p2 += 1
    print(
        "2. frame sync + deinterleave + systematic extract + CRC (both pol): %s"
        % ("OK" if p2 == 0 else "FAIL %d/6" % p2)
    )
    fails += p2

    # 3. corruption caught: flip an info bit -> CRC fails -> frame dropped
    sf2 = crc_append(rng.integers(0, 2, K_SF2).tolist(), K_SF2)
    sf3 = crc_append(rng.integers(0, 2, K_SF3).tolist(), K_SF3)
    deint = np.array(
        sf2
        + rng.integers(0, 2, N_SF2 - K_SF2).tolist()
        + sf3
        + rng.integers(0, 2, N_SF3 - K_SF3).tolist(),
        dtype=np.int8,
    )
    deint[10] ^= 1
    il = deint.reshape(IL_COLS, IL_ROWS).T.ravel()
    stream = np.hstack([_sf1(7), il, _sf1(8)]).astype(np.int8)
    p3 = 0 if decode_frame(stream) is None else 1
    print("3. corrupted info bit -> CRC rejects: %s" % ("OK" if p3 == 0 else "FAIL"))
    fails += p3

    print(
        "\n%s"
        % (
            "CNAV-2 frame codec SELF-CONSISTENT"
            if not fails
            else "SELF-TEST FAILURES: %d" % fails
        )
    )
    sys.exit(1 if fails else 0)
