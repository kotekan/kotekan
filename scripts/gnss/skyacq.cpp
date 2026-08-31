/**
 * @file
 * @brief OFFLINE acquisition on CAPTURED SKY -- the E6-C bring-up's decisive instrument.
 *
 * Every synthetic gate is self-consistent by construction (e2e generates its own sky), so a
 * replica-vs-sky mismatch is exactly what none of them can see. This closes that hole: it takes
 * a REAL voltage window (a /buffer/<voltage>/frame prefix, unpacked offline to [M][nc] cf32 for
 * ONE element), builds the replica bank the production stages build, and runs the SHIPPED CPU
 * acquisition (channelized_accumulate + channelized_peak) over a full Doppler grid x every
 * secondary-code alignment -- applying per-PRN secondaries (E6-C CS100) DIRECTLY from the code
 * tables, since the bank's single-sequence overlay slot rightly refuses them.
 *
 * The experiment is only an experiment WITH ITS CONTROL: run the same frame's B3I channels with
 * BDS_B3I first -- that chain tracks on sky, so the control must detect, or the tool (not the
 * signal) is at fault. [[experiment-that-cannot-succeed]]
 *
 * usage: skyacq <data.bin> <SIGNAL> <f_offset_hz> <prn_list> <global_ids...>
 *               [--dop-span HZ] [--elems N] [--dop-center hz1,hz2,...]
 *   data.bin: [n_elems][M=2048 hops][n_chan] cf32 planes; the surface is summed
 *   INCOHERENTLY over elements (they are not phased at this stage -- coherent summing
 *   would need the element cal; incoherent stacking buys sqrt(N) in detection
 *   significance, which is what the failed single-element control was short of).
 *   --dop-center: per-PRN Doppler window centre (the model is verified; a narrow window
 *   spends the compute on the axis under test, the CODE/CS phase).
 */
#include "gnssChannelizedReplica.hpp"
#include "gnssChannelizedAcquire.hpp"
#include "gnssSignal.hpp"
#include "galileoE6Code.hpp"
#include "beidouB3ICode.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <string>
#include <vector>

using cf = std::complex<float>;

int main(int argc, char** argv) {
    if (argc < 6) { printf("usage: see header\n"); return 2; }
    const char* binpath = argv[1];
    const char* signame = argv[2];
    const double f_off = atof(argv[3]);
    std::vector<int> prns;
    for (char* t = strtok(argv[4], ","); t; t = strtok(nullptr, ","))
        prns.push_back(atoi(t));
    std::vector<int> gids;
    double dop_span = 5000.0;
    int n_el = 1;
    std::vector<double> dopc;
    for (int i = 5; i < argc; ++i) {
        if (!strcmp(argv[i], "--dop-span")) { dop_span = atof(argv[++i]); continue; }
        if (!strcmp(argv[i], "--elems")) { n_el = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--dop-center")) {
            for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ","))
                dopc.push_back(atof(t));
            continue;
        }
        gids.push_back(atoi(argv[i]));
    }
    const int nc = (int)gids.size();
    const int M = 2048, Mp = 3125, fft_len = 16384, fine_step = 128;
    const double fs = 3.2e9;

    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signame);
    if (!sig) { printf("no signal %s\n", signame); return 2; }

    // data: n_el planes of [M][nc]
    std::vector<std::vector<std::vector<cf>>> dche(
        (size_t)n_el, std::vector<std::vector<cf>>((size_t)nc, std::vector<cf>((size_t)M)));
    {
        FILE* fp = fopen(binpath, "rb");
        if (!fp) { printf("cannot open %s\n", binpath); return 2; }
        std::vector<float> row((size_t)2 * nc);
        for (int e = 0; e < n_el; ++e)
            for (int m = 0; m < M; ++m) {
                if (fread(row.data(), sizeof(float), (size_t)2 * nc, fp) != (size_t)2 * nc) {
                    printf("short read at elem %d hop %d\n", e, m); return 2;
                }
                for (int c = 0; c < nc; ++c)
                    dche[(size_t)e][(size_t)c][(size_t)m] =
                        cf(row[(size_t)2 * c], row[(size_t)2 * c + 1]);
            }
        fclose(fp);
    }

    gnss::ChannelizedReplicaBank bank(*sig, fs, f_off, 8192, 4, dsp::Window::Hamming, prns);
    const long Lc = bank.eff_code_length();
    const long anchor = (long)Mp * fft_len;
    std::vector<long long> k0((size_t)Mp);
    for (int m = 0; m < Mp; ++m) {
        const double chip = ((double)anchor + (double)m * fft_len) * bank.chip_rate_hz() / fs;
        k0[(size_t)m] = (long long)std::floor(chip / (double)Lc);
    }
    // secondary sequence, per PRN where applicable (returned per call below)
    const std::string sn(signame);
    auto secondary = [&](int prn) -> std::vector<int8_t> {
        if (sn == "GAL_E6_C") {
            auto a = galileo::generate_e6c_secondary(prn);
            return std::vector<int8_t>(a.begin(), a.end());
        }
        if (sn == "BDS_B3I")
            return std::vector<int8_t>(beidou::B3I_NH20.begin(), beidou::B3I_NH20.end());
        return {1}; // no overlay: one trivial alignment
    };

    const int nd = 2 * (int)std::floor(dop_span / 62.5) + 1;
    std::vector<int> cov_local(nc);
    for (int c = 0; c < nc; ++c)
        cov_local[(size_t)c] = c;

    printf("skyacq: %s at %.2f MHz, %d chan (%d..%d), %d PRNs, %d elem planes, "
           "dop +-%.0f Hz (%d bins)%s\n",
           signame, f_off / 1e6, nc, gids.front(), gids.back(), (int)prns.size(), n_el, dop_span,
           nd, dopc.empty() ? "" : " per-PRN centred");

    gnss::AcquireWorkspace ws;
    std::vector<double> surf;
    for (size_t pi = 0; pi < prns.size(); ++pi) {
        const auto sec = secondary(prns[pi]);
        const int n_nh = (int)sec.size();
        // head/tail exactly as GnssChannelizedSearch builds them
        const auto A = bank.channels_hoprate((int)pi, anchor, 0.0, 0.0, Mp, gids, {}, -1);
        const auto B = bank.channels_hoprate(
            (int)pi, anchor, 0.0, 0.0, Mp, gids,
            [Lc](long long chip) {
                const long long k = (long long)std::floor((double)chip / (double)Lc);
                return ((k % 2 + 2) % 2 == 0) ? 1.0f : -1.0f;
            },
            -1);
        std::vector<std::vector<cf>> head((size_t)nc), tail((size_t)nc), repl0((size_t)nc);
        for (int c = 0; c < nc; ++c) {
            head[(size_t)c].assign((size_t)Mp, cf(0, 0));
            tail[(size_t)c].assign((size_t)Mp, cf(0, 0));
            repl0[(size_t)c].assign((size_t)Mp, cf(0, 0));
            for (int m = 0; m < Mp; ++m) {
                const float sgn = ((k0[(size_t)m] % 2 + 2) % 2 == 0) ? 1.0f : -1.0f;
                const cf dv = B[(size_t)c][(size_t)m] * sgn;
                head[(size_t)c][(size_t)m] = 0.5f * (A[(size_t)c][(size_t)m] + dv);
                tail[(size_t)c][(size_t)m] = 0.5f * (A[(size_t)c][(size_t)m] - dv);
            }
        }
        // per-PRN bin-aligned Doppler grid
        const double ctr = (pi < dopc.size()) ? 62.5 * std::llround(dopc[pi] / 62.5) : 0.0;
        std::vector<double> grid((size_t)nd);
        for (int d = 0; d < nd; ++d)
            grid[(size_t)d] = ctr + (d - nd / 2) * 62.5;
        double best_snr = 0, best_dop = 0;
        long best_tau = 0;
        int best_nh = -1;
        for (int nh = 0; nh < n_nh; ++nh) {
            for (int c = 0; c < nc; ++c)
                for (int m = 0; m < Mp; ++m) {
                    const float s0 = (float)sec[(size_t)(((k0[(size_t)m] + nh) % n_nh + n_nh) % n_nh)];
                    const float s1 = (float)sec[(size_t)(((k0[(size_t)m] + 1 + nh) % n_nh + n_nh) % n_nh)];
                    repl0[(size_t)c][(size_t)m] =
                        head[(size_t)c][(size_t)m] * s0 + tail[(size_t)c][(size_t)m] * s1;
                }
            surf.assign(surf.size(), 0.0);
            gnss::AcquisitionSurface dims{};
            for (int e = 0; e < n_el; ++e)
                dims = gnss::channelized_accumulate(dche[(size_t)e], repl0, cov_local, grid, fs,
                                                    nc, surf, ws, gids, fft_len, 16, fine_step);
            const auto pk = gnss::channelized_peak(surf, dims, grid, fs, bank.chip_rate_hz(),
                                                   (long)sig->code_length,
                                                   gnss::FINE_LAG_SIGN_PFB, false, M);
            if (pk.snr > best_snr) {
                best_snr = pk.snr; best_dop = pk.doppler_hz; best_tau = pk.peak_tau_samples;
                best_nh = nh;
            }
        }
        printf("  PRN %2d: best snr %8.2f  dop %+8.1f Hz  nh %3d/%d  tau %ld\n", prns[pi],
               best_snr, best_dop, best_nh, n_nh, best_tau);
        fflush(stdout);
    }
    return 0;
}
