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
    r.per_channel.resize(data_ch.size());
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
        r.per_channel[c] = g;
        G += g;
        energy += e;
    }

    r.correlation = G;
    r.replica_energy = energy;
    r.amplitude = (energy > 0.0) ? G / energy : std::complex<double>(0.0, 0.0);
    return r;
}

OverlayWipeResult overlay_wipe(const std::vector<std::complex<double>>& a,
                               const std::vector<double>& utc, const std::vector<int8_t>& overlay,
                               const std::vector<std::complex<double>>* head) {
    using cd = std::complex<double>;
    OverlayWipeResult out;
    const int nrec = (int)a.size();
    const int L = (int)overlay.size();
    if (L <= 0 || nrec < 2 || (int)utc.size() != nrec)
        return out;
    if (head && (int)head->size() != nrec)
        head = nullptr; // size mismatch (mixed-schema window): fall back to unsegmented

    // Absolute primary-period index per record from capture-UTC: a dropped record just skips an
    // index, so the overlay stays aligned (vs binning by buffer position, which a gap shifts).
    std::vector<double> dt;
    dt.reserve(nrec - 1);
    for (int r = 1; r < nrec; ++r)
        dt.push_back(utc[r] - utc[r - 1]);
    std::nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
    const double rec_dt = dt[dt.size() / 2]; // median step = the no-drop record period
    if (!(rec_dt > 0.0))
        return out;
    std::vector<long long> cpi(nrec);
    for (int r = 0; r < nrec; ++r)
        cpi[r] = (long long)std::llround((utc[r] - utc[0]) / rec_dt);

    auto chip = [&](int r, int phase) -> double {
        long long k = (cpi[r] + phase) % L;
        if (k < 0)
            k += L;
        return (double)overlay[(size_t)k];
    };
    // Overlay-corrected record: segmented (head gets its own chip, tail the NEXT chip --
    // the period boundary inside the record is where the overlay flips) or whole-record.
    auto corrected = [&](int r, int phase) -> cd {
        if (!head)
            return a[r] * chip(r, phase);
        const cd h = (*head)[r];
        return h * chip(r, phase) + (a[r] - h) * chip(r, phase + 1);
    };

    // Search the L overlay alignments for the one maximising the coherent sum of the
    // overlay-corrected records (the overlay is KNOWN, so this is a phase search, not a
    // per-bit +-1 estimate). The pilot has no nav bit, so the correct alignment integrates
    // coherently across primary periods.
    int best_phase = 0;
    double best_mag = -1.0;
    cd best_sum(0.0, 0.0);
    for (int phase = 0; phase < L; ++phase) {
        cd s(0.0, 0.0);
        for (int r = 0; r < nrec; ++r)
            s += corrected(r, phase);
        if (std::abs(s) > best_mag) {
            best_mag = std::abs(s);
            best_phase = phase;
            best_sum = s;
        }
    }

    // Significance: align the coherent sum onto the real axis; the component of each
    // overlay-corrected record ORTHOGONAL to it carries no signal -> the deep's uncertainty.
    const cd rot = best_mag > 0.0 ? std::polar(1.0, -std::arg(best_sum)) : cd(1.0, 0.0);
    double noise2 = 0.0;
    for (int r = 0; r < nrec; ++r) {
        const cd vr = corrected(r, best_phase) * rot;
        noise2 += std::imag(vr) * std::imag(vr);
    }
    const double noise_sum = std::sqrt(noise2); // noise std of the coherent sum
    out.amplitude = best_mag / (double)nrec;     // coherent mean of the overlay-corrected A
    out.snr = noise_sum > 0.0 ? best_mag / noise_sum : 0.0;
    out.phase = best_phase;
    return out;
}

OverlayWipeResult overlay_wipe_at(const std::vector<std::complex<double>>& a,
                                  const std::vector<double>& utc,
                                  const std::vector<int8_t>& overlay, int phase,
                                  const std::vector<std::complex<double>>* head) {
    // SELECTION-FREE variant of overlay_wipe: the alignment is GIVEN (dead-reckoned by the
    // caller from a previously-locked emit -- the overlay advances one chip per primary
    // period, deterministically in capture-UTC), so this is one straight coherent sum: no
    // max-over-alignments, hence none of the extreme-value bias/variance that inflates the
    // searched estimator's noise floor. E[snr^2] under pure noise = 2 exactly (complex
    // 2-dof), which is what makes the debiased coherent power unbiased per emit.
    using cd = std::complex<double>;
    OverlayWipeResult out;
    const int nrec = (int)a.size();
    const int L = (int)overlay.size();
    if (L <= 0 || nrec < 2 || (int)utc.size() != nrec)
        return out;
    if (head && (int)head->size() != nrec)
        head = nullptr; // size mismatch (mixed-schema window): fall back to unsegmented
    std::vector<double> dt;
    dt.reserve(nrec - 1);
    for (int r = 1; r < nrec; ++r)
        dt.push_back(utc[r] - utc[r - 1]);
    std::nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
    const double rec_dt = dt[dt.size() / 2];
    if (!(rec_dt > 0.0))
        return out;
    cd s(0.0, 0.0);
    std::vector<cd> v(nrec); // overlay-corrected records (segmented: head chip k, tail chip k+1)
    for (int r = 0; r < nrec; ++r) {
        long long k = ((long long)std::llround((utc[r] - utc[0]) / rec_dt) + phase) % L;
        if (k < 0)
            k += L;
        const double c0 = (double)overlay[(size_t)k];
        if (head) {
            const double c1 = (double)overlay[(size_t)((k + 1) % L)];
            const cd h = (*head)[r];
            v[r] = h * c0 + (a[r] - h) * c1;
        } else {
            v[r] = a[r] * c0;
        }
        s += v[r];
    }
    const double mag = std::abs(s);
    const cd rot = mag > 0.0 ? std::polar(1.0, -std::arg(s)) : cd(1.0, 0.0);
    double noise2 = 0.0;
    for (int r = 0; r < nrec; ++r) {
        const cd vr = v[r] * rot;
        noise2 += std::imag(vr) * std::imag(vr);
    }
    const double noise_sum = std::sqrt(noise2);
    out.amplitude = mag / (double)nrec;
    out.snr = noise_sum > 0.0 ? mag / noise_sum : 0.0;
    out.phase = ((phase % L) + L) % L;
    return out;
}

} // namespace gnss
