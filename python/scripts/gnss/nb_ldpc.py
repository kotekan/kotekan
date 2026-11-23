"""Non-binary LDPC decoder over GF(64) -- shared by the BeiDou B-CNAV decoders.

Ported from PocketSDR sdr_nb_ldpc.py (T.Takasu): GF(64) tables + Tanner-graph QSPA with the
extended-min-sum (EMS) check-node update, plus per-node edge groupings so an errored-frame
decode stays O(sum degree^2) not O(edges^2) (fast enough for the live broker loop).

Parameterized by (H_idx, H_ele, m, n): B-CNAV2 uses (48,96); B-CNAV1 uses SF2 (100,200) (SF3
is not decoded); B-CNAV3 uses (81,162). The H matrices are ICD-defined, transcribed by
PocketSDR from the BDS ICDs. The code is systematic (info = first m GF symbols), so at
deep-integration SNR the received codeword is error-free -> the syndrome passes on iteration 0
and decode is a parity check.
"""

import math

import numpy as np

N_GF = 6
Q_GF = 1 << N_GF  # 64
MAX_ITER = 15
NM_EMS = 4
ERR_PROB = 1e-5

GF_VEC = (
    1,
    2,
    4,
    8,
    16,
    32,
    3,
    6,
    12,
    24,
    48,
    35,
    5,
    10,
    20,
    40,
    19,
    38,
    15,
    30,
    60,
    59,
    53,
    41,
    17,
    34,
    7,
    14,
    28,
    56,
    51,
    37,
    9,
    18,
    36,
    11,
    22,
    44,
    27,
    54,
    47,
    29,
    58,
    55,
    45,
    25,
    50,
    39,
    13,
    26,
    52,
    43,
    21,
    42,
    23,
    46,
    31,
    62,
    63,
    61,
    57,
    49,
    33,
)
GF_POW = (
    0,
    0,
    1,
    6,
    2,
    12,
    7,
    26,
    3,
    32,
    13,
    35,
    8,
    48,
    27,
    18,
    4,
    24,
    33,
    16,
    14,
    52,
    36,
    54,
    9,
    45,
    49,
    38,
    28,
    41,
    19,
    56,
    5,
    62,
    25,
    11,
    34,
    31,
    17,
    47,
    15,
    23,
    53,
    51,
    37,
    44,
    55,
    40,
    10,
    61,
    46,
    30,
    50,
    22,
    39,
    43,
    29,
    60,
    42,
    21,
    20,
    59,
    57,
    58,
)

_GF_MUL = None


def init_gf():
    global _GF_MUL
    if _GF_MUL is not None:
        return _GF_MUL
    m = np.zeros((Q_GF, Q_GF), dtype=np.uint8)
    for i in range(1, Q_GF):
        for j in range(1, Q_GF):
            m[i][j] = GF_VEC[(GF_POW[i] + GF_POW[j]) % (Q_GF - 1)]
    _GF_MUL = m
    return _GF_MUL


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return GF_VEC[(GF_POW[a] + GF_POW[b]) % (Q_GF - 1)]


def gf_inv(a):
    return GF_VEC[(Q_GF - 1 - GF_POW[a]) % (Q_GF - 1)]


# _QBITS[q][j]: bit j of GF symbol q, most significant first (bin2gf's order)
_QBITS = np.array(
    [[(q >> (N_GF - 1 - j)) & 1 for j in range(N_GF)] for q in range(Q_GF)],
    dtype=np.uint8,
)


def bin2gf(syms):
    n = len(syms) // N_GF
    code = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        v = 0
        for j in range(N_GF):
            v = (v << 1) + int(syms[i * N_GF + j])
        code[i] = v
    return code


def gf2bin(code):
    syms = np.zeros(len(code) * N_GF, dtype=np.uint8)
    for i in range(len(code)):
        for j in range(N_GF):
            syms[i * N_GF + j] = (int(code[i]) >> (N_GF - 1 - j)) & 1
    return syms


_EDGE_CACHE = {}


def _edges(H_idx, H_ele, n_check, n_var):
    # Keyed by identity; holding both matrices in the entry keeps their ids from being reused.
    key = (id(H_idx), id(H_ele))
    hit = _EDGE_CACHE.get(key)
    if hit is not None:
        return hit[2]
    ie, je, he = [], [], []
    for i in range(len(H_idx)):
        for j in range(len(H_idx[i])):
            ie.append(i)
            je.append(H_idx[i][j])
            he.append(H_ele[i][j])
    ne = len(he)
    check_edges = [[] for _ in range(n_check)]
    var_edges = [[] for _ in range(n_var)]
    for e in range(ne):
        check_edges[ie[e]].append(e)
        var_edges[je[e]].append(e)
    c = (ie, je, he, ne, check_edges, var_edges)
    _EDGE_CACHE[key] = (H_idx, H_ele, c)
    return c


def _permute_V2C(GM, h, V2C):
    out = np.zeros(Q_GF, dtype=np.float32)
    row = GM[h]
    for i in range(Q_GF):
        out[row[i]] = V2C[i]
    return out


def _permute_C2V(GM, h, C2V):
    out = np.zeros(Q_GF, dtype=np.float32)
    row = GM[h]
    for i in range(Q_GF):
        out[i] = C2V[row[i]]
    return out


def _ext_min_sum(L1, L2):
    if len(L1) == 0:
        return L2
    idx1 = np.argsort(L1)[:NM_EMS]
    idx2 = np.argsort(L2)[:NM_EMS]
    maxL = L1[idx1[-1]] + L2[idx2[-1]]
    Ls = np.full(Q_GF, maxL, dtype=np.float32)
    for i in idx1:
        for j in idx2:
            v = L1[i] + L2[j]
            if v < Ls[i ^ j]:
                Ls[i ^ j] = v
    return Ls


def _check_parity(GM, ie, je, he, n_check, code):
    s = np.zeros(n_check, dtype=np.uint8)
    for i in range(len(ie)):
        s[ie[i]] ^= GM[he[i]][code[je[i]]]
    return bool(np.all(s == 0))


def decode(H_idx, H_ele, m, n, syms, rel=None, cost=None):
    """syms: n*6 hard bits (0/1). rel: optional n*6 reliabilities (|soft value|), each bit's
    cost of being flipped; without them every bit costs the same. cost: optional n x 64 costs of
    each GF symbol taking each value, for a channel that sends whole symbols (GuoWang's CSK);
    it replaces syms and rel. Returns (info_bits[m*6], nerr). nerr>=0 decoded (0=clean), -1 =
    failed to converge."""
    GM = init_gf()
    ie, je, he, ne, check_edges, var_edges = _edges(H_idx, H_ele, m, n)
    V2C = np.zeros((ne, Q_GF), dtype=np.float32)
    C2V = np.zeros((ne, Q_GF), dtype=np.float32)
    if cost is not None:
        L = np.asarray(cost, dtype=np.float32)
        code = np.argmin(L, axis=1).astype(np.uint8)
        s = gf2bin(code)
    else:
        s = np.asarray(syms, dtype=np.uint8)
        code = bin2gf(s)
        c0 = -math.log(ERR_PROB)
        # L[i][q]: the cost of GF symbol i being q, the summed reliability of the bits q flips
        r = np.ones(n * N_GF) if rel is None else np.asarray(rel, dtype=float)
        flips = _QBITS[None, :, :] != s.reshape(n, N_GF)[:, None, :]
        L = (c0 * (flips * r.reshape(n, N_GF)[:, None, :]).sum(axis=2)).astype(
            np.float32
        )
    for e in range(ne):
        V2C[e] = _permute_V2C(GM, he[e], L[je[e]])
    for _ in range(MAX_ITER):
        if _check_parity(GM, ie, je, he, m, code):
            dec = gf2bin(code)
            return dec[: m * N_GF], int(np.count_nonzero(dec ^ s))
        for cn in range(m):
            es = check_edges[cn]
            for e in es:
                Ls = []
                for f in es:
                    if f != e:
                        Ls = _ext_min_sum(Ls, V2C[f])
                Ls = Ls - np.min(Ls)
                C2V[e] = _permute_C2V(GM, he[e], Ls)
        for vn in range(n):
            es = var_edges[vn]
            for e in es:
                Ls = L[vn].copy()
                for f in es:
                    if f != e:
                        Ls = Ls + C2V[f]
                Ls = Ls - np.min(Ls)
                V2C[e] = _permute_V2C(GM, he[e], Ls)
        for vn in range(n):
            Lp = L[vn].copy()
            for e in var_edges[vn]:
                Lp = Lp + C2V[e]
            code[vn] = int(np.argmin(Lp))
    # the last iteration's update has not been checked yet
    if _check_parity(GM, ie, je, he, m, code):
        dec = gf2bin(code)
        return dec[: m * N_GF], int(np.count_nonzero(dec ^ s))
    return gf2bin(code)[: m * N_GF], -1


def _wht(x):
    """Walsh-Hadamard transform over the last axis (64), unnormalized: the XOR group's
    Fourier transform, which turns the sum of GF(64) symbols into a product."""
    shape = x.shape
    h = 1
    while h < Q_GF:
        x = x.reshape(shape[:-1] + (Q_GF // (2 * h), 2, h))
        a, b = x[..., 0, :], x[..., 1, :]
        x = np.stack((a + b, a - b), axis=-2)
        h *= 2
    return x.reshape(shape)


def decode_bp(H_idx, H_ele, m, n, logp, max_iter=50):
    """Belief propagation over GF(64) (the FFT/WHT sum-product): each check node combines its
    other edges' symbol distributions through the Walsh-Hadamard transform, so every message is
    a whole distribution rather than EMS's best NM_EMS values. A missing symbol enters as a flat
    distribution and is recovered from the code, which EMS cannot rank. logp: n x 64 channel
    log-probabilities of each symbol's values (a flat row for an erasure). Returns
    (info_bits[m*6], symbols changed from the channel's best), or (bits, -1) when the parity
    checks never hold."""
    GM = init_gf()
    hi = np.asarray(H_idx, dtype=np.int64)
    he = np.asarray(H_ele, dtype=np.int64)
    d_c = hi.shape[1]
    ev, eh = hi.ravel(), he.ravel()
    mul = GM[eh].astype(np.int64)  # mul[e][a] = h_e * a
    lp = np.asarray(logp, dtype=float)
    lp = lp - lp.max(axis=1, keepdims=True)
    q = np.exp(lp[ev])
    q /= q.sum(axis=1, keepdims=True)
    first = lp.argmax(axis=1)
    hard = first
    for _ in range(max_iter):
        # the permuted message h*x: qp[e][h a] = q[e][a]
        qp = np.empty_like(q)
        np.put_along_axis(qp, mul, q, axis=1)
        w = _wht(qp).reshape(m, d_c, Q_GF)
        # each edge's product over the check's other edges, from prefix and suffix products
        pre = np.ones_like(w)
        suf = np.ones_like(w)
        for k in range(1, d_c):
            pre[:, k] = pre[:, k - 1] * w[:, k - 1]
            suf[:, d_c - 1 - k] = suf[:, d_c - k] * w[:, d_c - k]
        rp = _wht((pre * suf).reshape(m * d_c, Q_GF)) / Q_GF
        rp = np.maximum(rp, 1e-30)
        r = np.take_along_axis(rp, mul, axis=1)  # back from h*x to x
        r /= r.sum(axis=1, keepdims=True)
        lr = np.log(r)
        tot = lp.copy()
        np.add.at(tot, ev, lr)
        hard = tot.argmax(axis=1)
        syn = np.bitwise_xor.reduce(GM[eh, hard[ev]].reshape(m, d_c), axis=1)
        if not syn.any():
            return gf2bin(hard)[: m * N_GF], int(np.count_nonzero(hard != first))
        lq = tot[ev] - lr
        lq -= lq.max(axis=1, keepdims=True)
        q = np.exp(lq)
        q /= q.sum(axis=1, keepdims=True)
    return gf2bin(hard)[: m * N_GF], -1


def encode_systematic(H_idx, H_ele, m, n, info_syms):
    """For self-tests: given the first m GF(64) info symbols, solve H c = 0 for the parity
    symbols (info-systematic) and return the full n*6-bit codeword. Requires the parity
    submatrix (cols m..n-1) to be invertible over GF(64) -- true for these ICD codes."""
    H = [[0] * n for _ in range(m)]
    for i in range(m):
        for k in range(len(H_idx[i])):
            H[i][H_idx[i][k]] = H_ele[i][k]
    rhs = [0] * m
    for i in range(m):
        for col in range(m):
            if H[i][col]:
                rhs[i] ^= gf_mul(H[i][col], int(info_syms[col]))
    npar = n - m
    Hp = [[H[i][m + j] for j in range(npar)] for i in range(m)]
    piv_col = [-1] * m
    r = 0
    for col in range(npar):
        pr = next((rr for rr in range(r, m) if Hp[rr][col] != 0), None)
        if pr is None:
            continue
        Hp[r], Hp[pr] = Hp[pr], Hp[r]
        rhs[r], rhs[pr] = rhs[pr], rhs[r]
        inv = gf_inv(Hp[r][col])
        Hp[r] = [gf_mul(inv, x) for x in Hp[r]]
        rhs[r] = gf_mul(inv, rhs[r])
        for rr in range(m):
            if rr != r and Hp[rr][col]:
                f = Hp[rr][col]
                Hp[rr] = [Hp[rr][cc] ^ gf_mul(f, Hp[r][cc]) for cc in range(npar)]
                rhs[rr] ^= gf_mul(f, rhs[r])
        piv_col[r] = col
        r += 1
    p = [0] * npar
    for rr in range(r):
        p[piv_col[rr]] = rhs[rr]
    code = list(int(x) for x in info_syms[:m]) + p
    return gf2bin(np.array(code, dtype=np.uint8))
