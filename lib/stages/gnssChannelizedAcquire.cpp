#include "gnssChannelizedAcquire.hpp"

#include "fftwPlannerLock.hpp" // for fftw_planner_mutex

#include <cmath>  // for cos, sin, fmod, M_PI
#include <mutex>  // for lock_guard

namespace gnss {

using cd = std::complex<double>;
using cf = std::complex<float>;

void AcquireWorkspace::ensure(int n) {
    if (n == Mp)
        return;
    std::lock_guard<std::mutex> lk(fftw_planner_mutex());
    if (fwd)
        fftwf_destroy_plan(fwd);
    if (inv)
        fftwf_destroy_plan(inv);
    if (in)
        fftwf_free(in);
    if (out)
        fftwf_free(out);
    Mp = n;
    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Mp);
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Mp);
    fwd = fftwf_plan_dft_1d(Mp, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    inv = fftwf_plan_dft_1d(Mp, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
}

AcquireWorkspace::~AcquireWorkspace() {
    std::lock_guard<std::mutex> lk(fftw_planner_mutex());
    if (fwd)
        fftwf_destroy_plan(fwd);
    if (inv)
        fftwf_destroy_plan(inv);
    if (in)
        fftwf_free(in);
    if (out)
        fftwf_free(out);
}

std::vector<std::vector<cf>>
channel_correlate(const std::vector<cf>& data, const std::vector<cf>& repl0,
                  const std::vector<double>& doppler_grid, double sample_rate, int samples_per_hop,
                  AcquireWorkspace& ws) {
    const int M = (int)data.size();   // data window length, hops
    const int Mp = (int)repl0.size(); // replica hop-period (>= M)
    const int nd = (int)doppler_grid.size();
    const int sph = (samples_per_hop > 0) ? samples_per_hop : Mp;

    ws.ensure(Mp);
    cf* IN = reinterpret_cast<cf*>(ws.in);
    cf* OUT = reinterpret_cast<cf*>(ws.out);

    // conj(FFT(repl0)) -- Doppler-independent, computed once.
    std::vector<cf> conjB(Mp);
    for (int k = 0; k < Mp; ++k)
        IN[k] = repl0[k];
    fftwf_execute(ws.fwd);
    for (int k = 0; k < Mp; ++k)
        conjB[k] = std::conj(OUT[k]);

    const float inv_mp = 1.0f / (float)Mp; // FFTW backward is unnormalized
    std::vector<std::vector<cf>> P(nd, std::vector<cf>(Mp));
    std::vector<cf> wipe(M);
    for (int d = 0; d < nd; ++d) {
        // Per-hop Doppler wipe e^{-i 2pi fd (m*sph)/Fs} via a phasor recurrence.
        const double dphi = -2.0 * M_PI * doppler_grid[d] * (double)sph / sample_rate;
        cd rot(std::cos(dphi), std::sin(dphi));
        cd ph(1.0, 0.0);
        for (int m = 0; m < M; ++m) {
            wipe[m] = cf((float)ph.real(), (float)ph.imag());
            ph *= rot;
        }
        for (int m = 0; m < M; ++m)
            IN[m] = data[m] * wipe[m];
        for (int m = M; m < Mp; ++m)
            IN[m] = cf(0.0f, 0.0f);
        fftwf_execute(ws.fwd); // OUT = FFT(wiped0) = A
        for (int k = 0; k < Mp; ++k)
            IN[k] = OUT[k] * conjB[k]; // A * conj(B)
        fftwf_execute(ws.inv);         // OUT = Mp * P_c(q)
        for (int q = 0; q < Mp; ++q)
            P[d][q] = OUT[q] * inv_mp;
    }
    return P;
}

AcquisitionSurface
aggregate_accumulate(const std::vector<std::vector<std::vector<cf>>>& P,
                     const std::vector<int>& chan_freq, int samples_per_hop,
                     std::vector<double>& surf) {
    const int nc = (int)P.size();
    const int nd = nc ? (int)P[0].size() : 0;
    const int Mp = (nc && nd) ? (int)P[0][0].size() : 0;
    const int sph = samples_per_hop;
    const AcquisitionSurface dims{nd, Mp, sph};
    if ((long)surf.size() != dims.size())
        surf.assign(dims.size(), 0.0);
    if (nc == 0)
        return dims;

    // Cross-channel fine-lag phase ramp e^{+i 2pi f_c s / sph}; SoA float so the
    // inner s-loop vectorizes (std::complex<float> does not auto-vectorize).
    std::vector<std::vector<float>> rampRe(nc, std::vector<float>(sph));
    std::vector<std::vector<float>> rampIm(nc, std::vector<float>(sph));
    for (int ci = 0; ci < nc; ++ci) {
        const int fc = chan_freq[ci];
        for (int s = 0; s < sph; ++s) {
            const double a = 2.0 * M_PI * fc * s / sph;
            rampRe[ci][s] = (float)std::cos(a);
            rampIm[ci][s] = (float)std::sin(a);
        }
    }

    std::vector<float> Dre(sph), Dim(sph);
    for (int d = 0; d < nd; ++d) {
        double* base = surf.data() + (long)d * Mp * sph;
        for (int q = 0; q < Mp; ++q) {
            for (int s = 0; s < sph; ++s) {
                Dre[s] = 0.0f;
                Dim[s] = 0.0f;
            }
            for (int ci = 0; ci < nc; ++ci) {
                const float pre = P[ci][d][q].real(), pim = P[ci][d][q].imag();
                const float* rr = rampRe[ci].data();
                const float* ri = rampIm[ci].data();
                for (int s = 0; s < sph; ++s) {
                    Dre[s] += pre * rr[s] - pim * ri[s];
                    Dim[s] += pre * ri[s] + pim * rr[s];
                }
            }
            double* row = base + (long)q * sph;
            for (int s = 0; s < sph; ++s)
                row[s] += (double)(Dre[s] * Dre[s] + Dim[s] * Dim[s]);
        }
    }
    return dims;
}

AcquisitionSurface
channelized_accumulate(const std::vector<std::vector<std::complex<float>>>& data_ch,
                       const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                       const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                       double sample_rate, int num_chan, std::vector<double>& surf,
                       AcquireWorkspace& ws, const std::vector<int>& chan_freq, int samples_per_hop) {
    // The decimation (samples per hop): N for a critically-sampled complex bank,
    // 2N for an r2c real-FFT bank. This is now just per-channel correlate (the
    // distributable part) + cross-channel aggregate (see channel_correlate /
    // aggregate_accumulate); the distributed search runs the two halves as
    // separate stages.
    const int sph = (samples_per_hop > 0) ? samples_per_hop : num_chan;
    const int nc = (int)covering.size();
    std::vector<std::vector<std::vector<cf>>> P(nc);
    for (int ci = 0; ci < nc; ++ci)
        P[ci] = channel_correlate(data_ch[covering[ci]], repl0_ch[covering[ci]], doppler_grid,
                                  sample_rate, sph, ws);
    std::vector<int> fc(nc);
    for (int ci = 0; ci < nc; ++ci)
        fc[ci] = chan_freq.empty() ? covering[ci] : chan_freq[ci];
    return aggregate_accumulate(P, fc, sph, surf);
}

AcquisitionResult channelized_peak(const std::vector<double>& surf, const AcquisitionSurface& dims,
                                   const std::vector<double>& doppler_grid, double sample_rate,
                                   double chip_rate, long code_length) {
    AcquisitionResult best{0.0, 0.0, 0, -1.0, 0.0};
    double surf_sum = 0.0;
    long surf_n = 0;
    int best_d = 0;
    long best_tau = 0;

    for (int d = 0; d < dims.n_dop; ++d) {
        const double* base = surf.data() + (long)d * dims.Mp * dims.sph;
        for (int q = 0; q < dims.Mp; ++q) {
            const double* row = base + (long)q * dims.sph;
            for (int s = 0; s < dims.sph; ++s) {
                const double pw = row[s];
                surf_sum += pw;
                surf_n++;
                if (pw > best.peak) {
                    best.peak = pw;
                    best_d = d;
                    best_tau = (long)q * dims.sph + s;
                }
            }
        }
    }

    best.doppler_hz = doppler_grid.empty() ? 0.0 : doppler_grid[best_d];
    const double mean = (surf_n > 0) ? surf_sum / (double)surf_n : 0.0;
    best.snr = (mean > 0.0) ? best.peak / mean : 0.0;

    // The correlation peaks at the negative of the replica's code-phase advance
    // (data = repl0 advanced by the delay). Report the delay so it can seed the
    // measurement replica generator directly (same "code[floor(cp + n*r)]"
    // convention).
    const long Ns = (long)dims.Mp * dims.sph;
    best.peak_tau_samples = ((Ns - best_tau) % Ns + Ns) % Ns;
    const double chips = (double)best.peak_tau_samples * chip_rate / sample_rate;
    best.code_phase_chips = std::fmod(chips, (double)code_length);
    if (best.code_phase_chips < 0.0)
        best.code_phase_chips += code_length;
    return best;
}

AcquisitionResult
channelized_acquire(const std::vector<std::vector<std::complex<float>>>& data_ch,
                    const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                    const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                    double sample_rate, double chip_rate, int num_chan, long code_length,
                    const std::vector<int>& chan_freq, int samples_per_hop) {
    // Single-window acquire = one accumulation + reduce (local FFT workspace).
    std::vector<double> surf;
    AcquireWorkspace ws;
    const auto dims = channelized_accumulate(data_ch, repl0_ch, covering, doppler_grid, sample_rate,
                                             num_chan, surf, ws, chan_freq, samples_per_hop);
    return channelized_peak(surf, dims, doppler_grid, sample_rate, chip_rate, code_length);
}

} // namespace gnss
