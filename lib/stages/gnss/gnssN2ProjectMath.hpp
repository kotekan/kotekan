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

/// Eight doubles; the projection's blocks are written in these so they compile to whole vector
/// operations (GCC and clang). Read and written through casts of any double* (unaligned,
/// aliasing double), never passed to or returned from a function, so no ABI depends on them.
typedef double v8d __attribute__((vector_size(64), aligned(8), __may_alias__));

/// V <- (I - Q Q^H) V (I - Q Q^H) in place for Hermitian V, Q = k orthonormal columns given
/// row-wise as q[j * n + i] (ProjSubspace's layout). With W = V Q, M = Q^H W and Q^H V = W^H
/// (V Hermitian): V' = V - Q W^H - W Q^H + Q M Q^H = V - Q W^H - (W - Q M) Q^H. Taking
/// U = W - Q M first makes the update two rank-k products, linear in k, ~2 k n^2 complex
/// multiply-adds with the update on the lower triangle (mirrored after).
///
/// The work is done on real and imaginary parts held apart in @p W, padded to whole blocks of
/// eight, each inner step a block of eight outputs kept in registers while a k-loop runs: no
/// reduction in an inner loop, so the compiler has no sums to reorder. The row length is
/// padded one block past n so the rows do not sit a power of two apart (at n = 127 the 1 KB
/// stride put the update's four operands in the same few L1 sets).
inline void project(int n, int k, const cd* q, cd* V, std::vector<cd>& W, std::vector<cd>& M) {
    if (k <= 0)
        return;
    constexpr int B = 8;
    constexpr int G = 5; // blocks of a row of W held at once
    const int np = (n + B - 1) / B * B + B, kp = (k + B - 1) / B * B, nb = kp / B;
    const size_t nk = (size_t)np * kp, nn = (size_t)np * np, kk = (size_t)kp * kp;
    W.resize(5 * nk + nn);
    M.assign(kk, cd(0.0, 0.0));
    double* s = reinterpret_cast<double*>(W.data());
    double *qr = s, *qi = qr + nk;       // Q, [kp][np]
    double *tr = qi + nk, *ti = tr + nk; // Q transposed, [np][kp]
    double *wr = ti + nk, *wi = wr + nk; // W, [np][kp]
    double *xr = wi + nk, *xi = xr + nk; // W transposed, [kp][np]
    double *ur = xi + nk, *ui = ur + nk; // U, [np][kp]
    double *vr = ui + nk, *vi = vr + nn; // V, [np][np]
    double* mr = reinterpret_cast<double*>(M.data());
    double* mi = mr + kk;
    // The blocks read past n and k: those pads must be finite, so they are zeroed.
    for (int i = 0; i < n; ++i)
        for (int j = k; j < kp; ++j)
            tr[(size_t)i * kp + j] = ti[(size_t)i * kp + j] = 0.0;
    for (int j = 0; j < k; ++j)
        for (int c = n; c < np; ++c)
            qr[(size_t)j * np + c] = qi[(size_t)j * np + c] = xr[(size_t)j * np + c] =
                xi[(size_t)j * np + c] = 0.0;
    for (int i = 0; i < n; ++i)
        for (int c = n; c < np; ++c)
            vr[(size_t)i * np + c] = vi[(size_t)i * np + c] = 0.0;
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i) {
            const cd x = q[(size_t)j * n + i];
            qr[(size_t)j * np + i] = tr[(size_t)i * kp + j] = x.real();
            qi[(size_t)j * np + i] = ti[(size_t)i * kp + j] = x.imag();
        }
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < n; ++c) {
            vr[(size_t)i * np + c] = V[(size_t)i * n + c].real();
            vi[(size_t)i * np + c] = V[(size_t)i * n + c].imag();
        }
    // W = V Q, up to G blocks of a row of W in registers while the row of V streams past
    for (int i = 0; i < n; ++i)
        for (int g0 = 0; g0 < nb; g0 += G) {
            const int ng = std::min(G, nb - g0);
            v8d ar[G] = {}, ai[G] = {};
            for (int c = 0; c < n; ++c) {
                const double a = vr[(size_t)i * np + c], b = vi[(size_t)i * np + c];
                const double* tc = tr + (size_t)c * kp + g0 * B;
                const double* sc = ti + (size_t)c * kp + g0 * B;
                for (int t = 0; t < ng; ++t) {
                    const v8d x = *(const v8d*)(tc + t * B), y = *(const v8d*)(sc + t * B);
                    ar[t] += a * x - b * y;
                    ai[t] += a * y + b * x;
                }
            }
            for (int t = 0; t < ng; ++t) {
                *(v8d*)(wr + (size_t)i * kp + (g0 + t) * B) = ar[t];
                *(v8d*)(wi + (size_t)i * kp + (g0 + t) * B) = ai[t];
            }
        }
    // M = Q^H W, row by row
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < k; ++a) {
            const double qa = tr[(size_t)i * kp + a], qb = ti[(size_t)i * kp + a];
            for (int b0 = 0; b0 < kp; b0 += B) {
                const v8d w1 = *(const v8d*)(wr + (size_t)i * kp + b0),
                          w2 = *(const v8d*)(wi + (size_t)i * kp + b0);
                *(v8d*)(mr + (size_t)a * kp + b0) =
                    *(const v8d*)(mr + (size_t)a * kp + b0) + qa * w1 + qb * w2;
                *(v8d*)(mi + (size_t)a * kp + b0) =
                    *(const v8d*)(mi + (size_t)a * kp + b0) + qa * w2 - qb * w1;
            }
        }
    // U = W - Q M
    for (int i = 0; i < n; ++i)
        for (int b0 = 0; b0 < kp; b0 += B) {
            v8d sr = *(const v8d*)(wr + (size_t)i * kp + b0),
                si = *(const v8d*)(wi + (size_t)i * kp + b0);
            for (int l = 0; l < k; ++l) {
                const double a = tr[(size_t)i * kp + l], c = ti[(size_t)i * kp + l];
                const v8d m1 = *(const v8d*)(mr + (size_t)l * kp + b0),
                          m2 = *(const v8d*)(mi + (size_t)l * kp + b0);
                sr -= a * m1 - c * m2;
                si -= a * m2 + c * m1;
            }
            *(v8d*)(ur + (size_t)i * kp + b0) = sr;
            *(v8d*)(ui + (size_t)i * kp + b0) = si;
        }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) {
            xr[(size_t)j * np + i] = wr[(size_t)i * kp + j];
            xi[(size_t)j * np + i] = wi[(size_t)i * kp + j];
        }
    // V'_ic = V_ic - sum_j q_ji conj(W_cj) - sum_j U_ij conj(q_jc), blocks of c reaching c <= i
    for (int c0 = 0; c0 < n; c0 += B)
        for (int i = c0; i < n; ++i) {
            v8d sr = *(const v8d*)(vr + (size_t)i * np + c0),
                si = *(const v8d*)(vi + (size_t)i * np + c0);
            for (int j = 0; j < k; ++j) {
                const double ar = tr[(size_t)i * kp + j], ai = ti[(size_t)i * kp + j];
                const double br = ur[(size_t)i * kp + j], bi = ui[(size_t)i * kp + j];
                const v8d x1 = *(const v8d*)(xr + (size_t)j * np + c0),
                          x2 = *(const v8d*)(xi + (size_t)j * np + c0);
                const v8d q1 = *(const v8d*)(qr + (size_t)j * np + c0),
                          q2 = *(const v8d*)(qi + (size_t)j * np + c0);
                sr -= ar * x1 + ai * x2 + br * q1 + bi * q2;
                si -= ai * x1 - ar * x2 + bi * q1 - br * q2;
            }
            *(v8d*)(vr + (size_t)i * np + c0) = sr;
            *(v8d*)(vi + (size_t)i * np + c0) = si;
        }
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c <= i; ++c)
            V[(size_t)i * n + c] = cd(vr[(size_t)i * np + c], vi[(size_t)i * np + c]);
        for (int c = 0; c < i; ++c)
            V[(size_t)c * n + i] = std::conj(V[(size_t)i * n + c]);
    }
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
