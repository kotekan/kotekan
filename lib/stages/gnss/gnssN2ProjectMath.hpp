/**
 * @file
 * @brief The arithmetic of GnssN2Project, header-only so a standalone test can run it:
 *        the n2k blocked lower-triangular layout, the live-block extraction/write-back and
 *        the rank-k projection V' = (I - Q Q^H) V (I - Q Q^H).
 */
#ifndef GNSS_N2_PROJECT_MATH_HPP
#define GNSS_N2_PROJECT_MATH_HPP

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

/// V <- (I - Q Q^H) V (I - Q Q^H) in place, Q = k orthonormal columns given row-wise as
/// q[j * n + i] (ProjSubspace's layout). With W = V Q and M = Q^H W:
/// V' = V - Q W^H - W Q^H + Q M Q^H. Cost ~4 k n^2.
inline void project(int n, int k, const cd* q, cd* V, std::vector<cd>& W, std::vector<cd>& M) {
    if (k <= 0)
        return;
    W.assign((size_t)n * k, cd(0.0, 0.0));
    M.assign((size_t)k * k, cd(0.0, 0.0));
    for (int i = 0; i < n; ++i) {
        const cd* Vi = V + (size_t)i * n;
        for (int j = 0; j < k; ++j) {
            const cd* qj = q + (size_t)j * n;
            cd s(0.0, 0.0);
            for (int c = 0; c < n; ++c)
                s += Vi[c] * qj[c];
            W[(size_t)i * k + j] = s;
        }
    }
    for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b) {
            cd s(0.0, 0.0);
            for (int i = 0; i < n; ++i)
                s += std::conj(q[(size_t)a * n + i]) * W[(size_t)i * k + b];
            M[(size_t)a * k + b] = s;
        }
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < n; ++c) {
            cd s(0.0, 0.0);
            for (int j = 0; j < k; ++j) {
                const cd qij = q[(size_t)j * n + i], qcj = q[(size_t)j * n + c];
                s -= qij * std::conj(W[(size_t)c * k + j]) + W[(size_t)i * k + j] * std::conj(qcj);
                for (int l = 0; l < k; ++l)
                    s += qij * M[(size_t)j * k + l] * std::conj(q[(size_t)l * n + c]);
            }
            V[(size_t)i * n + c] += s;
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
