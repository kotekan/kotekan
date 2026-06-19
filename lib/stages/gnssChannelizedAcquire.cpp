#include "gnssChannelizedAcquire.hpp"

#include <cmath> // for cos, sin, fmod, M_PI

namespace gnss {

using cd = std::complex<double>;

AcquisitionResult
channelized_acquire(const std::vector<std::vector<std::complex<float>>>& data_ch,
                    const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                    const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                    double sample_rate, double chip_rate, int num_chan, long code_length) {
    const int N = num_chan;
    const int M = (int)data_ch[0].size();
    const int nc = (int)covering.size();

    // Pre-tabulate the cross-channel fine-lag phase ramp e^{+i 2pi c s / N} for
    // the covering channels and all sub-hop offsets s.
    std::vector<std::vector<cd>> ramp(nc, std::vector<cd>(N));
    for (int ci = 0; ci < nc; ++ci)
        for (int s = 0; s < N; ++s) {
            const double a = 2.0 * M_PI * covering[ci] * s / N;
            ramp[ci][s] = cd(std::cos(a), std::sin(a));
        }

    AcquisitionResult best{0.0, 0.0, 0, -1.0, 0.0};
    double surf_sum = 0.0;
    long surf_n = 0;

    std::vector<cd> wiped(M); // Doppler-wiped data for one channel

    for (double fd : doppler_grid) {
        // Per covering channel: P_c(q) = sum_m X'_c[m] conj(R0_c[m-q]).
        std::vector<std::vector<cd>> P(nc, std::vector<cd>(M));
        for (int ci = 0; ci < nc; ++ci) {
            const int c = covering[ci];
            for (int m = 0; m < M; ++m) {
                const double ph = -2.0 * M_PI * fd * (double)(m * N) / sample_rate;
                wiped[m] = cd(data_ch[c][m]) * cd(std::cos(ph), std::sin(ph));
            }
            for (int q = 0; q < M; ++q) {
                cd acc(0.0, 0.0);
                for (int m = 0; m < M; ++m)
                    acc += wiped[m] * std::conj(cd(repl0_ch[c][(m - q % M + M) % M]));
                P[ci][q] = acc;
            }
        }

        // Fine lag: D(qN+s) = sum_c P_c(q) e^{+i 2pi c s / N}.
        for (int q = 0; q < M; ++q) {
            for (int s = 0; s < N; ++s) {
                cd D(0.0, 0.0);
                for (int ci = 0; ci < nc; ++ci)
                    D += P[ci][q] * ramp[ci][s];
                const double pw = std::norm(D);
                surf_sum += pw;
                surf_n++;
                if (pw > best.peak) {
                    best.peak = pw;
                    best.doppler_hz = fd;
                    best.peak_tau_samples = (long)q * N + s;
                }
            }
        }
    }

    const double mean = (surf_n > 0) ? surf_sum / (double)surf_n : 0.0;
    best.snr = (mean > 0.0) ? best.peak / mean : 0.0;

    // The correlation peaks at the negative of the replica's code-phase advance
    // (data = repl0 advanced by the delay). Report the delay so it can seed the
    // measurement replica generator directly (same "code[floor(cp + n*r)]"
    // convention).
    const long Ns = (long)M * N;
    best.peak_tau_samples = ((Ns - best.peak_tau_samples) % Ns + Ns) % Ns;
    const double chips = (double)best.peak_tau_samples * chip_rate / sample_rate;
    best.code_phase_chips = std::fmod(chips, (double)code_length);
    if (best.code_phase_chips < 0.0)
        best.code_phase_chips += code_length;
    return best;
}

} // namespace gnss
