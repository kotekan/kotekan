/**
 * @file
 * @brief The arithmetic of GnssN2Project, header-only and independent of the stage: the n2k
 *        blocked lower-triangular layout, the live-block extraction and write-back, and the
 *        rank-k projection V' = (I - Q Q^H) V (I - Q Q^H).
 */
#ifndef GNSS_N2_PROJECT_MATH_HPP
#define GNSS_N2_PROJECT_MATH_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace gnss_n2proj {

using cd = std::complex<double>;

/// cudaCorrelator / n2k output for one frequency: num_blocks blocks of [16][16][2] int32,
/// block b = ihi (ihi + 1) / 2 + jhi over jhi <= ihi, entry (ilo, jlo) = station i = 16 ihi + ilo
/// (row) against j = 16 jhi + jlo (column), valid where j <= i (N2Accumulate reads only that
/// half; the other half of a diagonal block is never written). The Hermitian completion
/// V(j, i) = conj V(i, j) is implied, and whichever of the two the kernel conjugates, reading
/// and writing with the same rule keeps the projection exact.
struct Layout {
    int n_elem = 128;
    int bs = 16;
    int lin() const {
        return n_elem / bs;
    }
    int num_blocks() const {
        return lin() * (lin() + 1) / 2;
    }
    /// int32 words per frequency.
    size_t per_freq() const {
        return (size_t)num_blocks() * bs * bs * 2;
    }
    /// Word offset of entry (i, j), i >= j, within one frequency.
    size_t idx(int i, int j) const {
        const int ihi = i / bs, jhi = j / bs, ilo = i % bs, jlo = j % bs;
        const size_t b = (size_t)ihi * (ihi + 1) / 2 + jhi;
        return ((b * bs + ilo) * bs + jlo) * 2;
    }
};

/// Hermitian live block V[a][b] (n x n, row-major) of the stations @p st (ascending).
inline void extract(const Layout& L, const int32_t* f, const std::vector<int>& st, cd* V) {
    const int n = (int)st.size();
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {
            const size_t k = L.idx(st[a], st[b]);
            const cd v((double)f[k], (double)f[k + 1]);
            V[(size_t)a * n + b] = v;
            V[(size_t)b * n + a] = std::conj(v);
        }
}

inline int32_t to_i32(double x) {
    const double r = std::nearbyint(x);
    if (r > 2147483647.0)
        return 2147483647;
    if (r < -2147483648.0)
        return -2147483647 - 1;
    return (int32_t)r;
}

/// Write the lower triangle of V back into the frame (the diagonal's imaginary part is 0).
inline void writeback(const Layout& L, int32_t* f, const std::vector<int>& st, const cd* V) {
    const int n = (int)st.size();
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {
            const size_t k = L.idx(st[a], st[b]);
            const cd v = V[(size_t)a * n + b];
            f[k] = to_i32(v.real());
            f[k + 1] = (a == b) ? 0 : to_i32(v.imag());
        }
}

/// V <- (I - Q Q^H) V (I - Q Q^H) in place for Hermitian V, Q = k orthonormal columns given
/// row-wise as q[j * n + i] (ProjSubspace's layout). With W = V Q, M = Q^H W and Q^H V = W^H
/// (V Hermitian): V' = V - Q W^H - W Q^H + Q M Q^H = V - Q W^H - (W - Q M) Q^H. Taking
/// U = W - Q M first makes the update two rank-k products, linear in k. The update is computed
/// on the lower triangle and mirrored, ~2 k n^2 complex multiply-adds in all; every [n][k]
/// operand (Q transposed, W, U) is row-major so the inner loops run over k contiguously.
inline void project(int n, int k, const cd* q, cd* V, std::vector<cd>& W, std::vector<cd>& M) {
    if (k <= 0)
        return;
    W.assign((size_t)3 * n * k, cd(0.0, 0.0));
    M.assign((size_t)k * k, cd(0.0, 0.0));
    cd* qt = W.data();
    cd* WQ = qt + (size_t)n * k;
    cd* U = WQ + (size_t)n * k;
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i)
            qt[(size_t)i * k + j] = q[(size_t)j * n + i];
    for (int i = 0; i < n; ++i) {
        const cd* Vi = V + (size_t)i * n;
        for (int j = 0; j < k; ++j) {
            const cd* qj = q + (size_t)j * n;
            cd s(0.0, 0.0);
            for (int c = 0; c < n; ++c)
                s += Vi[c] * qj[c];
            WQ[(size_t)i * k + j] = s;
        }
    }
    // M = Q^H W, U = W - Q M
    for (int i = 0; i < n; ++i) {
        const cd* qi = qt + (size_t)i * k;
        const cd* wi = WQ + (size_t)i * k;
        for (int a = 0; a < k; ++a) {
            const cd ca = std::conj(qi[a]);
            cd* Ma = M.data() + (size_t)a * k;
            for (int b = 0; b < k; ++b)
                Ma[b] += ca * wi[b];
        }
    }
    for (int i = 0; i < n; ++i) {
        const cd* qi = qt + (size_t)i * k;
        cd* Ui = U + (size_t)i * k;
        std::copy(WQ + (size_t)i * k, WQ + (size_t)(i + 1) * k, Ui);
        for (int l = 0; l < k; ++l) {
            const cd ql = qi[l];
            const cd* Ml = M.data() + (size_t)l * k;
            for (int b = 0; b < k; ++b)
                Ui[b] -= ql * Ml[b];
        }
    }
    // V'_ic = V_ic - sum_j q_ji conj(W_cj) - sum_j U_ij conj(q_jc), for c <= i
    for (int i = 0; i < n; ++i) {
        cd* Vi = V + (size_t)i * n;
        const cd* qi = qt + (size_t)i * k;
        const cd* Ui = U + (size_t)i * k;
        for (int c = 0; c <= i; ++c) {
            const cd* Wc = WQ + (size_t)c * k;
            const cd* qc = qt + (size_t)c * k;
            cd s(0.0, 0.0);
            for (int j = 0; j < k; ++j)
                s += qi[j] * std::conj(Wc[j]) + Ui[j] * std::conj(qc[j]);
            Vi[c] -= s;
        }
    }
    for (int i = 0; i < n; ++i)
        for (int c = i + 1; c < n; ++c)
            V[(size_t)i * n + c] = std::conj(V[(size_t)c * n + i]);
}

/// The products between the stations @p st and the inputs @p ot outside them, as X[a][o] =
/// V(st[a], ot[o]) (n x n_o, row-major).
inline void extract_cross(const Layout& L, const int32_t* f, const std::vector<int>& st,
                          const std::vector<int>& ot, cd* X) {
    const int n = (int)st.size(), n_o = (int)ot.size();
    for (int a = 0; a < n; ++a)
        for (int o = 0; o < n_o; ++o) {
            const bool lower = st[a] > ot[o];
            const size_t k = lower ? L.idx(st[a], ot[o]) : L.idx(ot[o], st[a]);
            const cd v((double)f[k], (double)f[k + 1]);
            X[(size_t)a * n_o + o] = lower ? v : std::conj(v);
        }
}

inline void writeback_cross(const Layout& L, int32_t* f, const std::vector<int>& st,
                            const std::vector<int>& ot, const cd* X) {
    const int n = (int)st.size(), n_o = (int)ot.size();
    for (int a = 0; a < n; ++a)
        for (int o = 0; o < n_o; ++o) {
            const bool lower = st[a] > ot[o];
            const size_t k = lower ? L.idx(st[a], ot[o]) : L.idx(ot[o], st[a]);
            const cd v = lower ? X[(size_t)a * n_o + o] : std::conj(X[(size_t)a * n_o + o]);
            f[k] = to_i32(v.real());
            f[k + 1] = to_i32(v.imag());
        }
}

/// X <- (I - Q Q^H) X for the station side of the cross products, so that with project() on the
/// station block the frame is M V M^H, M = (I - Q Q^H) on the stations and the identity on the
/// rest: the projection of the station voltages alone.
inline void project_cross(int n, int n_o, int k, const cd* q, cd* X, std::vector<cd>& Y) {
    if (k <= 0 || n_o <= 0)
        return;
    Y.assign((size_t)k * n_o, cd(0.0, 0.0));
    for (int j = 0; j < k; ++j) {
        const cd* qj = q + (size_t)j * n;
        cd* Yj = Y.data() + (size_t)j * n_o;
        for (int a = 0; a < n; ++a) {
            const cd c = std::conj(qj[a]);
            const cd* Xa = X + (size_t)a * n_o;
            for (int o = 0; o < n_o; ++o)
                Yj[o] += c * Xa[o];
        }
    }
    for (int j = 0; j < k; ++j) {
        const cd* qj = q + (size_t)j * n;
        const cd* Yj = Y.data() + (size_t)j * n_o;
        for (int a = 0; a < n; ++a) {
            cd* Xa = X + (size_t)a * n_o;
            for (int o = 0; o < n_o; ++o)
                Xa[o] -= qj[a] * Yj[o];
        }
    }
}
/// Mean of the diagonal (the live autos), the stage's power unit for its thresholds.
inline double mean_auto(int n, const cd* V) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += V[(size_t)i * n + i].real();
    return n > 0 ? s / n : 0.0;
}

} // namespace gnss_n2proj

#endif
