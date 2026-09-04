"""Non-binary LDPC decoder over GF(64) -- shared by the BeiDou B-CNAV decoders.

Ported from PocketSDR sdr_nb_ldpc.py (T.Takasu): GF(64) tables + Tanner-graph QSPA with the
extended-min-sum (EMS) check-node update, plus per-node edge groupings so an errored-frame
decode stays O(sum degree^2) not O(edges^2) (fast enough for the live broker loop).

Parameterized by (H_idx, H_ele, m, n): B-CNAV2 uses (48,96); B-CNAV1 uses SF2 (100,200) and
SF3 (44,88); the H matrices are ICD-defined, transcribed by PocketSDR from the BDS ICDs. The
code is SYSTEMATIC (info = first m GF symbols), so at deep-integration SNR the received codeword
is error-free -> the syndrome passes on iteration 0 and decode is a parity check.
"""

import math

import numpy as np

N_GF = 6
Q_GF = 1 << N_GF               # 64
MAX_ITER = 15
NM_EMS = 4
ERR_PROB = 1e-5

GF_VEC = (
    1, 2, 4, 8, 16, 32, 3, 6, 12, 24, 48, 35, 5, 10, 20, 40,
    19, 38, 15, 30, 60, 59, 53, 41, 17, 34, 7, 14, 28, 56, 51, 37,
    9, 18, 36, 11, 22, 44, 27, 54, 47, 29, 58, 55, 45, 25, 50, 39,
    13, 26, 52, 43, 21, 42, 23, 46, 31, 62, 63, 61, 57, 49, 33)
GF_POW = (
    0, 0, 1, 6, 2, 12, 7, 26, 3, 32, 13, 35, 8, 48, 27, 18,
    4, 24, 33, 16, 14, 52, 36, 54, 9, 45, 49, 38, 28, 41, 19, 56,
    5, 62, 25, 11, 34, 31, 17, 47, 15, 23, 53, 51, 37, 44, 55, 40,
    10, 61, 46, 30, 50, 22, 39, 43, 29, 60, 42, 21, 20, 59, 57, 58)

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
    key = id(H_idx)
    c = _EDGE_CACHE.get(key)
    if c is not None:
        return c
    ie, je, he = [], [], []
    for i in range(len(H_idx)):
        for j in range(len(H_idx[i])):
            ie.append(i); je.append(H_idx[i][j]); he.append(H_ele[i][j])
    ne = len(he)
    check_edges = [[] for _ in range(n_check)]
    var_edges = [[] for _ in range(n_var)]
    for e in range(ne):
        check_edges[ie[e]].append(e)
        var_edges[je[e]].append(e)
    c = (ie, je, he, ne, check_edges, var_edges)
    _EDGE_CACHE[key] = c
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


def decode(H_idx, H_ele, m, n, syms):
    """syms: n*6 hard bits (0/1). Returns (info_bits[m*6], nerr). nerr>=0 decoded (0=clean),
    -1 = failed to converge."""
    GM = init_gf()
    s = np.asarray(syms, dtype=np.uint8)
    code = bin2gf(s)
    ie, je, he, ne, check_edges, var_edges = _edges(H_idx, H_ele, m, n)
    V2C = np.zeros((ne, Q_GF), dtype=np.float32)
    C2V = np.zeros((ne, Q_GF), dtype=np.float32)
    L = np.zeros((n, Q_GF), dtype=np.float32)
    c0 = -math.log(ERR_PROB)
    for i in range(n):
        for j in range(Q_GF):
            L[i][j] = c0 * bin(int(code[i]) ^ j).count("1")
    for e in range(ne):
        V2C[e] = _permute_V2C(GM, he[e], L[je[e]])
    for _ in range(MAX_ITER):
        if _check_parity(GM, ie, je, he, m, code):
            dec = gf2bin(code)
            return dec[:m * N_GF], int(np.count_nonzero(dec ^ s))
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
    return gf2bin(code)[:m * N_GF], -1


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
        Hp[r], Hp[pr] = Hp[pr], Hp[r]; rhs[r], rhs[pr] = rhs[pr], rhs[r]
        inv = gf_inv(Hp[r][col])
        Hp[r] = [gf_mul(inv, x) for x in Hp[r]]; rhs[r] = gf_mul(inv, rhs[r])
        for rr in range(m):
            if rr != r and Hp[rr][col]:
                f = Hp[rr][col]
                Hp[rr] = [Hp[rr][cc] ^ gf_mul(f, Hp[r][cc]) for cc in range(npar)]
                rhs[rr] ^= gf_mul(f, rhs[r])
        piv_col[r] = col; r += 1
    p = [0] * npar
    for rr in range(r):
        p[piv_col[rr]] = rhs[rr]
    code = list(int(x) for x in info_syms[:m]) + p
    return gf2bin(np.array(code, dtype=np.uint8))
