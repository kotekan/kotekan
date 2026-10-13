#include "gnssChannelizedDespread.hpp"

#include <algorithm> // for nth_element
#include <cmath>     // for arg, llround, polar, sqrt
#include <stdexcept> // for invalid_argument

namespace gnss {

DespreadResult channelized_despread(const std::vector<std::vector<std::complex<float>>>& data_ch,
                                    const std::vector<std::vector<std::complex<float>>>& repl_ch) {
    if (data_ch.size() != repl_ch.size())
        throw std::invalid_argument("channelized_despread: channel count mismatch");

    DespreadResult r;
    std::complex<double> G(0.0, 0.0);
    double energy = 0.0;

    for (size_t c = 0; c < data_ch.size(); ++c) {
        const auto& x = data_ch[c];
        const auto& rep = repl_ch[c];
        if (x.size() != rep.size())
            throw std::invalid_argument("channelized_despread: channel length mismatch");

        std::complex<double> g(0.0, 0.0);
        double e = 0.0;
        for (size_t m = 0; m < x.size(); ++m) {
            g += std::complex<double>(x[m]) * std::conj(std::complex<double>(rep[m]));
            e += std::norm(std::complex<double>(rep[m]));
        }
        G += g;
        energy += e;
    }

    r.correlation = G;
    r.replica_energy = energy;
    r.amplitude = (energy > 0.0) ? G / energy : std::complex<double>(0.0, 0.0);
    return r;
}

/// Coherent SNR mag / noise_sum, where noise_sum is the residual orthogonal to the rotated sum;
/// 0 when that residual is degenerate.
///
/// Records phase-aligned to double precision, as a derotation (e.g. phase_track_loo) leaves
/// them, have a residual of ~1e-17 rather than exactly 0, which would read as ~1e17 sigma. Even
/// a 60 dB record has an orthogonal residual ~1e-3 of its amplitude, so a residual below
/// 1e-12 * mag * sqrt(nrec) was removed by construction, not measured. Return 0, the same
/// "cannot compute" value as nrec < 2, so a candidate selection cannot pick it.
///
/// This guards the division only: scoring a derotated candidate by the residual its
/// derotation removed is still self-referential, and the best of several such candidates
/// rectifies noise.
static double residual_snr(double mag, double noise_sum, int nrec) {
    const double degen = mag * std::sqrt((double)std::max(nrec, 1)) * 1e-12;
    return (noise_sum > degen) ? mag / noise_sum : 0.0;
}

bool overlay_apply(const std::vector<std::complex<double>>& a, const std::vector<double>& utc,
                   const std::vector<int8_t>& overlay, int phase,
                   const std::vector<std::complex<double>>* head,
                   std::vector<std::complex<double>>& out_a,
                   std::vector<std::complex<double>>* out_head) {
    // chip k = round((utc[r]-utc[0])/rec_dt) + phase, mod L, on the record grid.
    using cd = std::complex<double>;
    const int nrec = (int)a.size();
    const int L = (int)overlay.size();
    if (L <= 0 || nrec < 2 || (int)utc.size() != nrec)
        return false;
    if (head && (int)head->size() != nrec)
        head = nullptr; // size mismatch: fall back to the unsegmented wipe
    std::vector<double> dt;
    dt.reserve(nrec - 1);
    for (int r = 1; r < nrec; ++r)
        dt.push_back(utc[r] - utc[r - 1]);
    std::nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
    const double rec_dt = dt[dt.size() / 2];
    if (!(rec_dt > 0.0))
        return false;
    out_a.assign((size_t)nrec, cd(0.0, 0.0));
    if (out_head)
        out_head->assign((size_t)nrec, cd(0.0, 0.0));
    for (int r = 0; r < nrec; ++r) {
        long long k = ((long long)std::llround((utc[r] - utc[0]) / rec_dt) + phase) % L;
        if (k < 0)
            k += L;
        const double c0 = (double)overlay[(size_t)k];
        if (head) {
            const double c1 = (double)overlay[(size_t)((k + 1) % L)];
            const cd h = (*head)[r];
            // Head takes chip k, tail (a-h) chip k+1: a record straddling a chip transition
            // summed whole would cancel to |2f-1|.
            const cd hw = h * c0;
            out_a[(size_t)r] = hw + (a[(size_t)r] - h) * c1;
            if (out_head)
                (*out_head)[(size_t)r] = hw; // wiped head, so a downstream segmented wipe
                                             // (nav-bit) keeps a consistent head/tail split
        } else {
            out_a[(size_t)r] = a[(size_t)r] * c0;
            if (out_head)
                (*out_head)[(size_t)r] = out_a[(size_t)r]; // head == a, tail 0
        }
    }
    return true;
}

OverlayWipeResult coherent_sum(const std::vector<std::complex<double>>& a) {
    // Rotate the sum onto the real axis and take the orthogonal component as the noise.
    using cd = std::complex<double>;
    OverlayWipeResult out;
    const int nrec = (int)a.size();
    if (nrec < 2)
        return out;
    cd s(0.0, 0.0);
    for (int r = 0; r < nrec; ++r)
        s += a[(size_t)r];
    const double mag = std::abs(s);
    const cd rot = mag > 0.0 ? std::polar(1.0, -std::arg(s)) : cd(1.0, 0.0);
    double noise2 = 0.0;
    for (int r = 0; r < nrec; ++r) {
        const cd vr = a[(size_t)r] * rot;
        noise2 += std::imag(vr) * std::imag(vr);
    }
    const double noise_sum = std::sqrt(noise2);
    out.amplitude = mag / (double)nrec;
    out.snr = residual_snr(mag, noise_sum, nrec);
    return out;
}

bool phase_track_loo(const std::vector<std::complex<double>>& a, int half_width,
                     std::vector<std::complex<double>>& out_a) {
    // Sliding leave-one-out boxcar: window sum minus self, O(n) via a running sum.
    // Zero-amplitude records contribute nothing to any estimate.
    using cd = std::complex<double>;
    const int n = (int)a.size();
    if (n < 4 || half_width < 1)
        return false;
    out_a.assign((size_t)n, cd(0.0, 0.0));
    cd win(0.0, 0.0); // sum of a[j] over the current window [lo, hi]
    int lo = 0, hi = -1;
    int n_used = 0; // nonzero records inside the window
    for (int k = 0; k < n; ++k) {
        // Advance the window to [k - half_width, k + half_width].
        while (hi < std::min(n - 1, k + half_width)) {
            ++hi;
            if (std::norm(a[(size_t)hi]) > 0.0) {
                win += a[(size_t)hi];
                ++n_used;
            }
        }
        while (lo < k - half_width) {
            if (std::norm(a[(size_t)lo]) > 0.0) {
                win -= a[(size_t)lo];
                --n_used;
            }
            ++lo;
        }
        const bool self_nz = std::norm(a[(size_t)k]) > 0.0;
        const cd s = win - (self_nz ? a[(size_t)k] : cd(0.0, 0.0)); // leave-one-out
        const int nb = n_used - (self_nz ? 1 : 0);
        // < 2 neighbours: the estimate would be a single record's noise; pass through.
        out_a[(size_t)k] = (nb >= 2 && std::norm(s) > 0.0)
                               ? a[(size_t)k] * std::polar(1.0, -std::arg(s))
                               : a[(size_t)k];
    }
    return true;
}

} // namespace gnss
