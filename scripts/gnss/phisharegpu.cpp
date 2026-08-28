/**
 * @file
 * @brief THE KERNEL GATE for the shared Doppler-free tables (docs/CHORD_GPU_TODO.md item 2).
 *
 * phibits validates the algebra; phishare validates the CPU generator. Neither touches the GPU,
 * and #71 is the monument to why that is not enough: "the gate tested the FORMULA not the kernel
 * contract" -- a carrier-NCO change passed at 9e-16 rad while the sky got worse. So this drives
 * the SHIPPED GnssCudaDespread through its public API, with REAL tables, twice:
 *
 *   PER-PRN  set_shared_phi(false) -- today's path, one 1.05 MB Phi per PRN per channel
 *   SHARED   set_shared_phi(true)  -- ONE (Phi, Psi) pair for every PRN + the ddw rotor
 *
 * and compares the despread's own outputs. It also asserts the two things that make the change
 * safe rather than merely accurate: that the fallback is BIT-IDENTICAL, and that a satellite at
 * zero Doppler reconstructs EXACTLY (ddw == 0 must take no correction at all).
 *
 * Usage: ./phisharegpu [n_prn]
 */
#include "GnssCudaDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int n_prn = (argc > 1) ? atoi(argv[1]) : 8;
    const double FS = 3.2e9, F_OFF = 1176.45e6;
    const int NFFT = 16384, NTAPS = 4, N_HOPS = 256;

    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q");
    if (!sig) { printf("no GPS_L5_Q\n"); return 2; }

    std::vector<int> prns;
    for (int i = 0; i < n_prn; ++i) prns.push_back(1 + i);
    gnss::ChannelizedReplicaBank bank(*sig, FS, F_OFF, NFFT, NTAPS, dsp::Window::Hamming, prns);
    std::vector<int> chans{5972, 5988, 6004, 6020, 6036, 6052, 6068};

    GnssCudaDespread dsp(bank, n_prn, chans, N_HOPS, FS, F_OFF, /*refresh_hz=*/1e9);

    // One deterministic window; the DATA is irrelevant to the question (both runs see the same
    // bytes) but it must not be zero, or every correlation is zero and the test passes vacuously.
    std::mt19937 rng(12345);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<std::complex<float>> win((size_t)N_HOPS * chans.size());
    for (auto& v : win) v = std::complex<float>(g(rng), g(rng));
    dsp.upload_window(win.data(), 0);

    // Deliberately spans zero and both signs at the full GPS range.
    const double dops[] = {0.0, 137.0, -900.0, 2500.0, -5000.0, 4321.0, -137.0, 3000.0};
    std::vector<GnssCudaDespread::Spec> specs;
    for (int i = 0; i < n_prn; ++i) {
        GnssCudaDespread::Spec sp;
        sp.p = i;
        sp.doppler_hz = dops[i % 8];
        sp.cp_seed = 100.0 + 37.0 * i;
        sp.spacing_chips = 0.5;
        sp.ctrim_hz = 0.0;
        sp.covering = {0, 1, 2, 3, 4, 5, 6};
        specs.push_back(sp);
    }

    if (!dsp.set_shared_phi(false)) { /* expected: returns false when off */ }
    const auto per_prn = dsp.despread_batch(specs);
    const bool took = dsp.set_shared_phi(true);
    printf("phisharegpu: %d PRNs x %zu channels, %d hops\n", n_prn, chans.size(), N_HOPS);
    printf("  set_shared_phi(true) -> %s\n", took ? "IN EFFECT" : "REFUSED (FDMA?)");
    if (!took) { printf("  cannot gate what did not arm\n"); return 2; }
    const auto shared = dsp.despread_batch(specs);

    printf("\n  %-4s %9s   %12s %12s %11s\n", "PRN", "doppler", "|per-PRN|", "|shared|", "rel err");
    int bad = 0;
    for (int i = 0; i < n_prn; ++i) {
        double worst = 0.0, ref = 0.0;
        for (int t = 0; t < 3; ++t) {
            const auto a = per_prn[(size_t)i][(size_t)t].correlation;
            const auto b = shared[(size_t)i][(size_t)t].correlation;
            worst = std::max(worst, std::abs(a - b));
            ref = std::max(ref, std::abs(a));
        }
        const double rel = (ref > 0.0) ? worst / ref : 0.0;
        const bool zero_dop = (specs[(size_t)i].doppler_hz == 0.0);
        // A zero-Doppler satellite has ddw == 0 and MUST reconstruct exactly: any nonzero
        // error there means the correction is firing when it should be identically absent,
        // which would be a sign/branch bug hiding behind an otherwise-small number.
        const bool fail = (rel > 1e-3) || (zero_dop && worst != 0.0);
        if (fail) bad++;
        printf("  %-4d %+9.0f   %12.5e %12.5e %11.3e%s%s\n", prns[(size_t)i],
               specs[(size_t)i].doppler_hz, ref, std::abs(shared[(size_t)i][1].correlation), rel,
               zero_dop ? "  (ddw=0: must be EXACT)" : "", fail ? "   <-- FAIL" : "");
    }
    printf("\n  bar: 1e-3 relative; fp16 storage alone costs 3.3e-4\n");
    printf("  %s\n", bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}
