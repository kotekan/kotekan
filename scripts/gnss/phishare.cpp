/**
 * @file
 * @brief Does ONE Doppler-free (Phi, Psi) pair reproduce the per-PRN replica? (GPU TODO item 2)
 *
 * phibits [2c] validates the ALGEBRA -- the reconstruction of a single chip-window difference.
 * This validates the GENERATOR: it streams a full record of the channelized replica twice,
 *
 *   TRUTH  hoprate_filter(want, dop)      -- today's per-PRN table, Doppler baked in
 *   SHARED hoprate_filter(want, 0, -1, true) + the ddw reconstruction inside hoprate_stream
 *
 * and compares sample by sample. That is the step #71 is a monument to: a formula that is
 * right in isolation can still fail the contract of the code that consumes it, and only
 * driving the SHIPPED generator finds that out.
 *
 * Usage: ./phishare [prn] [n_hops]
 */
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <vector>

int main(int argc, char** argv) {
    const int prn = (argc > 1) ? atoi(argv[1]) : 1;
    const int n_hops = (argc > 2) ? atoi(argv[2]) : 64;
    const double FS = 3.2e9, F_OFF = 1176.45e6;
    const int NFFT = 16384, NTAPS = 4;

    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    if (!sig) {
        printf("no GPS_L5_Q\n");
        return 2;
    }
    gnss::ChannelizedReplicaBank bank(*sig, FS, F_OFF, NFFT, NTAPS, dsp::Window::Hamming, {prn});
    std::vector<int> want{5972, 5988, 6004, 6020, 6036, 6052, 6068};

    printf("phishare: PRN %d, %d hops, %zu channels, fs %.1f GHz, f_off %.2f MHz\n",
           prn, n_hops, want.size(), FS / 1e9, F_OFF / 1e6);
    printf("  the SHARED pair is built ONCE at doppler 0 and reused for every Doppler below\n\n");
    // ONE shared, Doppler-free filter -- built once, exactly as production would hold it.
    const auto shared = bank.hoprate_filter(want, 0.0, -1, /*want_psi=*/true);

    printf("   doppler      max|err|     max|ref|    rel(worst)   rel(rms)\n");
    int bad = 0;
    for (double dop : {0.0, 250.0, 1000.0, 2500.0, 5000.0, -5000.0, 10000.0}) {
        const auto truth_f = bank.hoprate_filter(want, dop);
        const auto A = bank.hoprate_stream(truth_f, 0, 0, 123.456, dop, n_hops, nullptr, -1);
        const auto B = bank.hoprate_stream(shared, 0, 0, 123.456, dop, n_hops, nullptr, -1);
        double emax = 0.0, rmax = 0.0, se = 0.0, sr = 0.0;
        size_t n = 0;
        for (size_t c = 0; c < A.size(); ++c)
            for (size_t m = 0; m < A[c].size(); ++m) {
                const double e = std::abs(std::complex<double>(A[c][m]) - std::complex<double>(B[c][m]));
                const double r = std::abs(std::complex<double>(A[c][m]));
                emax = std::max(emax, e);
                rmax = std::max(rmax, r);
                se += e * e;
                sr += r * r;
                n++;
            }
        const double relw = (rmax > 0.0) ? emax / rmax : 0.0;
        const double relr = (sr > 0.0) ? std::sqrt(se / sr) : 0.0;
        printf("  %+8.0f Hz   %.4e   %.4e   %.3e   %.3e%s\n", dop, emax, rmax, relw, relr,
               (relw > 1e-3) ? "   <-- FAIL" : "");
        if (relw > 1e-3)
            bad++;
    }
    printf("\n  bar: 1e-3 relative (fp16 storage alone costs 3.3e-4)\n");
    printf("  %s\n", bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}
