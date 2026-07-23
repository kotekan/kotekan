// f-sweep harness: run the ACTUAL overlay_wipe / overlay_wipe_at over a synthetic
// B1C-like window with the code-period boundary at fraction f of each record.
// If segmentation works, recovered SNR is flat in f; the straddle bug dips at f=0.5.
#include "gnssChannelizedDespread.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <complex>

using cd = std::complex<double>;

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const int L = 1800, NREC = 100, K0 = 137;
    const double REC_DT = 0.01, AMP = 1.0, NOISE = 0.05;

    std::vector<int8_t> overlay(L);
    for (auto& c : overlay) c = (rng() & 1) ? 1 : -1;

    printf("#  f    seg_snr  seg_amp   unseg_snr unseg_amp   at_snr  at_amp  (search phase ok?)\n");
    for (double f = 0.02; f < 0.99; f += 0.04) {
        std::vector<cd> a(NREC), head(NREC);
        std::vector<double> utc(NREC);
        for (int r = 0; r < NREC; ++r) {
            utc[r] = r * REC_DT;
            const int k = (K0 + r) % L, k1 = (K0 + r + 1) % L;
            const cd h = AMP * f * (double)overlay[k]
                         + cd(gauss(rng), gauss(rng)) * NOISE * std::sqrt(f);
            const cd t = AMP * (1.0 - f) * (double)overlay[k1]
                         + cd(gauss(rng), gauss(rng)) * NOISE * std::sqrt(1.0 - f);
            head[r] = h;
            a[r] = h + t;
        }
        auto seg   = gnss::overlay_wipe(a, utc, overlay, &head);
        auto unseg = gnss::overlay_wipe(a, utc, overlay, nullptr);
        auto at    = gnss::overlay_wipe_at(a, utc, overlay, K0, &head);
        printf("  %.2f  %7.1f  %7.3f   %7.1f  %7.3f   %7.1f %7.3f   search_phase=%d (want %d)\n",
               f, seg.snr, seg.amplitude, unseg.snr, unseg.amplitude,
               at.snr, at.amplitude, seg.phase, K0);
    }
    return 0;
}
