// acqbench -- validate and time the WHOLE device-resident acquisition chain (A2) against the
// shipped CPU path, gnss::channelized_accumulate + gnss::channelized_peak.
//
// aggbench validated the aggregate in isolation against a synthetic P. This drives the real
// thing end to end: snapshot -> gather/pad -> FFT(data) -> materialise replica -> FFT -> Doppler
// wipe -> inverse correlate -> aggregate -> peak, and compares the resulting SURFACE, cell for
// cell, against the CPU computing the same quantity from the same inputs.
//
// It does NOT use gnss::ChannelizedReplicaBank: head/tail are synthetic. That is deliberate --
// this is testing the correlate/aggregate arithmetic and the cuFFT layout, and a synthetic
// replica with a PLANTED delay makes the answer knowable in advance. Whether the replica itself
// is right is what the on-sky A/B tests, and it is already tested by the CPU path.
//
// build:  KBUILD=build_nodpdk ./build_tool.sh acqbench
// usage:  ./acqbench [--nd 321] [--mp 3125] [--nc 79] [--fine-step 128] [--m 2048]
//                    [--reps 3] [--tau 12345] [--dop-bin 7] [--no-cpu]

#include "GnssCudaAcquire.hpp"
#include "gnssChannelizedAcquire.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using cf = std::complex<float>;

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<int> chord_channels(int nc) {
    std::vector<int> ids;
    for (int off = 0; off < 16 && (int)ids.size() < nc; ++off)
        for (int b = 5972 + off; b <= 6076 && (int)ids.size() < nc; b += 16)
            ids.push_back(b);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    ids.resize(std::min<size_t>(ids.size(), (size_t)nc));
    return ids;
}

int main(int argc, char** argv) {
    int nd = 321, Mp = 3125, nc = 79, fine_step = 128, M = 2048, reps = 3;
    int tau_q = 1234;   // planted COARSE lag (hops)
    int dop_bin = 7;    // planted Doppler, in whole transform bins
    bool do_cpu = true;
    const int sph = 16384;
    const double fs = 3.2e9;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::atoi(argv[++i]); };
        if (a == "--nd") nd = next();
        else if (a == "--mp") Mp = next();
        else if (a == "--nc") nc = next();
        else if (a == "--fine-step") fine_step = next();
        else if (a == "--m") M = next();
        else if (a == "--reps") reps = next();
        else if (a == "--tau") tau_q = next();
        else if (a == "--dop-bin") dop_bin = next();
        else if (a == "--no-cpu") do_cpu = false;
    }

    const std::vector<int> ids = chord_channels(nc);
    nc = (int)ids.size();
    const int s_cols = sph / fine_step;
    const double bin_hz = fs / ((double)Mp * sph);

    printf("acqbench: nc %d  n_dop %d  Mp %d  M %d  sph %d  s_cols %d  bin %.4f Hz\n", nc, nd, Mp,
           M, sph, s_cols, bin_hz);

    // ---------------------------------------------------------------------------------------
    // Synthetic inputs. head/tail are random per (channel, hop); the "signal" is the replica
    // itself delayed by tau_q hops and shifted by dop_bin bins, so the correlation peak lands
    // at a cell we can name in advance. Everything is float32 on both sides, so CPU and GPU are
    // being asked for bit-comparable work.
    // ---------------------------------------------------------------------------------------
    std::mt19937 rng(9876);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<std::vector<cf>> head(nc, std::vector<cf>(Mp)), tail(nc, std::vector<cf>(Mp));
    for (int c = 0; c < nc; ++c)
        for (int m = 0; m < Mp; ++m) {
            head[c][m] = cf(g(rng), g(rng));
            tail[c][m] = cf(g(rng), g(rng));
        }
    // One secondary-code alignment's sign pair, hoisted per hop exactly as the search stage does.
    std::vector<float> sgn0(Mp), sgn1(Mp);
    for (int m = 0; m < Mp; ++m) {
        sgn0[m] = (m % 7 < 4) ? 1.0f : -1.0f;
        sgn1[m] = (m % 5 < 2) ? 1.0f : -1.0f;
    }
    // Materialise the replica the same way the GPU will, so the CPU reference gets the identical
    // repl0 and any difference is the CORRELATE, not the replica.
    std::vector<std::vector<cf>> repl0(nc, std::vector<cf>(Mp));
    for (int c = 0; c < nc; ++c)
        for (int m = 0; m < Mp; ++m)
            repl0[c][m] = head[c][m] * sgn0[m] + tail[c][m] * sgn1[m];

    // Snapshot: noise + the replica delayed by tau_q and Doppler-shifted by dop_bin bins.
    // A cyclic shift of the SPECTRUM by b bins is a multiply by e^{+2 pi i b m / Mp} in hops,
    // which is exactly what a bin-aligned Doppler wipe undoes.
    const int n_chan_total = nc; // the tool's snapshot holds only the covering channels
    std::vector<cf> snap((size_t)M * n_chan_total);
    for (int m = 0; m < M; ++m)
        for (int c = 0; c < nc; ++c) {
            const int src = ((m - tau_q) % Mp + Mp) % Mp;
            const double ph = 2.0 * M_PI * dop_bin * (double)m / (double)Mp;
            const cf sig = repl0[c][src] * cf((float)std::cos(ph), (float)std::sin(ph));
            snap[(size_t)m * n_chan_total + c] = 0.30f * cf(g(rng), g(rng)) + sig;
        }

    // Doppler grid, BIN-ALIGNED and centred so the planted bin is inside it.
    std::vector<double> grid((size_t)nd);
    for (int d = 0; d < nd; ++d)
        grid[(size_t)d] = (d - nd / 2 + dop_bin) * bin_hz;

    std::vector<int> local(nc);
    std::iota(local.begin(), local.end(), 0);

    // ---------------------------------------------------------------------------------------
    // CPU reference
    // ---------------------------------------------------------------------------------------
    std::vector<double> surf_cpu;
    gnss::AcquisitionSurface dims{};
    double t_cpu = 0.0;
    if (do_cpu) {
        std::vector<std::vector<cf>> dch(nc, std::vector<cf>(M));
        for (int c = 0; c < nc; ++c)
            for (int m = 0; m < M; ++m)
                dch[c][m] = snap[(size_t)m * n_chan_total + c];
        gnss::AcquireWorkspace ws;
        const double t0 = now_s();
        dims = gnss::channelized_accumulate(dch, repl0, local, grid, fs, nc, surf_cpu, ws, ids,
                                            sph, 16, fine_step);
        t_cpu = now_s() - t0;
        const auto pk = gnss::channelized_peak(surf_cpu, dims, grid, fs, 10.23e6, 10230);
        printf("\nCPU  : %8.4f s   snr %.3f  peak %.6g  tau %ld  dop %+.2f Hz\n", t_cpu, pk.snr,
               pk.peak, pk.peak_tau_samples, pk.doppler_hz);
        printf("       planted: coarse lag %d hops (tau %ld), doppler bin %+d (%+.2f Hz)\n", tau_q,
               (long)tau_q * sph, dop_bin, dop_bin * bin_hz);
    }

    // ---------------------------------------------------------------------------------------
    // GPU chain
    // ---------------------------------------------------------------------------------------
    try {
        GnssCudaAcquire eng(nc, Mp, M, n_chan_total, nd, s_cols);
        eng.set_channels(local, ids, sph, fine_step);
        if (!eng.set_doppler_grid(grid, fs, sph)) {
            printf("GPU  : REJECTED the Doppler grid (not bin-aligned) -- that is the intended\n"
                   "       behaviour for a grid this engine cannot represent exactly.\n");
            return 1;
        }
        printf("GPU  : %.2f GB device\n", 1e-9 * (double)eng.device_bytes());

        eng.set_snapshot(snap.data());
        eng.set_replica(head, tail);

        // warm up (module load + cuFFT plan work happens on first exec)
        eng.clear_surface();
        eng.accumulate(sgn0.data(), sgn1.data());
        (void)eng.peak();

        double t_gpu = 0.0;
        GnssCudaAcquire::Peak pk{};
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            eng.clear_surface();
            eng.accumulate(sgn0.data(), sgn1.data());
            pk = eng.peak();
            t_gpu += now_s() - t0;
        }
        t_gpu /= reps;

        const double snr = pk.peak / pk.mean;
        printf("GPU  : %8.4f s   snr %.3f  peak %.6g  (d %d, q %d, s %d)\n", t_gpu, snr, pk.peak,
               pk.d, pk.q, pk.s);
        if (do_cpu)
            printf("       speedup %.1fx\n", t_cpu / t_gpu);

        // The planted answer, independent of both implementations.
        //
        // The SURFACE CELL is at q = +tau_q. The negation that 5q warns about ("the lag axis is
        // the negative of physical delay") lives in channelized_peak's cell -> delay conversion,
        // NOT in the surface indexing: with tau_q = 1234 and Mp = 3125 the peak cell is q = 1234
        // while the reported peak_tau_samples is 1891*sph. Both are correct; they are different
        // quantities, and conflating them makes correct code look off by (Mp - 2*tau_q) hops.
        const int want_d = nd / 2;
        const int want_q = tau_q;
        printf("check: planted (d %d, q %d)   got (d %d, q %d)   %s\n", want_d, want_q, pk.d, pk.q,
               (pk.d == want_d && pk.q == want_q) ? "OK" : "MISMATCH");

        if (do_cpu) {
            std::vector<double> surf_gpu;
            eng.download_surface(surf_gpu);
            if (surf_gpu.size() != surf_cpu.size()) {
                printf("check: SIZE MISMATCH cpu %zu gpu %zu\n", surf_cpu.size(), surf_gpu.size());
                return 1;
            }
            double maxabs = 0, maxdiff = 0, sc = 0, sg = 0;
            long am_c = 0, am_g = 0;
            double pc = -1, pg = -1;
            for (size_t i = 0; i < surf_cpu.size(); ++i) {
                maxabs = std::max(maxabs, surf_cpu[i]);
                maxdiff = std::max(maxdiff, std::fabs(surf_cpu[i] - surf_gpu[i]));
                sc += surf_cpu[i];
                sg += surf_gpu[i];
                if (surf_cpu[i] > pc) { pc = surf_cpu[i]; am_c = (long)i; }
                if (surf_gpu[i] > pg) { pg = surf_gpu[i]; am_g = (long)i; }
            }
            printf("check: surface rel %.3g   mean cpu %.6g gpu %.6g (rel %.3g)\n",
                   maxdiff / maxabs, sc / surf_cpu.size(), sg / surf_gpu.size(),
                   std::fabs(sc - sg) / std::fabs(sc));
            printf("check: argmax cpu %ld gpu %ld   snr cpu %.3f gpu %.3f   %s\n", am_c, am_g,
                   pc / (sc / surf_cpu.size()), pg / (sg / surf_gpu.size()),
                   am_c == am_g ? "OK" : "MISMATCH");
        }
    } catch (const std::exception& e) {
        printf("GPU  : %s\n", e.what());
        return 1;
    }
    return 0;
}
