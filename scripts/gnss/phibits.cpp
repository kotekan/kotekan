/**
 * @file
 * @brief How many bits does the Phi table actually need, and how far apart are two Dopplers?
 *
 * Two questions the synthesis kernel's cost hangs on, both answerable offline:
 *
 *  1. PRECISION. Phi is a PREFIX SUM, and the gather uses only DIFFERENCES of it over ~inv_cps
 *     taps. If |Phi| is large against |dPhi| the difference cancels catastrophically and the
 *     storage format needs the headroom; if the modulation keeps |Phi| bounded near |dPhi|, it
 *     does not, and the table could be stored in half the bytes -- which is a direct 2x on a
 *     kernel that is memory-bound by ~60x.
 *
 *  2. DOPPLER FACTORING. Phi_A[k] = sum_{j<k} proto[j] exp(-i(off_c + wc) j), and the Doppler
 *     is inside wc, multiplied by j. Across PRNs the Doppler spread is only a few kHz, so the
 *     RATIO of two PRNs' tables is exp(-i d(wc) j) -- a slow tilt across the span. If it is slow
 *     enough, a low-order expansion about a bucket centre lets ALL PRNs on a channel share one
 *     set of tables instead of each owning 1.06 MB.
 *
 * Usage: ./phibits [doppler_hz] [delta_hz]
 */

#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using cd = std::complex<double>;

int main(int argc, char** argv) {
    const double dop = argc > 1 ? atof(argv[1]) : 0.0;
    const double dhz = argc > 2 ? atof(argv[2]) : 5000.0;

    // CHORD L5, as deployed.
    const double FS = 3.2e9, F_OFF = 1176450000.0;
    const int NCH = 8192, TAPS = 4; // spectrum_length = N; fft_len = 2N = 16384 as deployed
    const auto* sig = gnss::signal_by_name("GPS_L5_Q_NH");
    if (!sig) {
        printf("no signal descriptor\n");
        return 1;
    }
    std::vector<int> prns{3};
    gnss::ChannelizedReplicaBank bank(*sig, FS, F_OFF, NCH, TAPS, dsp::Window::Hamming, prns);
    const std::vector<int> want{6023};

    const auto f0 = bank.hoprate_filter(want, dop);
    const int Lf = (int)f0.PhiA[0].size() - 1;
    const double inv_cps = FS / sig->chip_rate_hz;
    const int ks = (int)inv_cps;
    printf("CHORD L5: Lf %d  inv_cps %.3f  n_chips %d  channel %d\n", Lf, inv_cps, f0.n_chips,
           want[0]);

    // ---- 1. dynamic range of Phi vs the differences the gather actually consumes -----------
    const auto& P = f0.PhiA[0];
    double pmax = 0.0;
    for (int k = 0; k <= Lf; ++k)
        pmax = std::max(pmax, std::abs(P[(size_t)k]));
    std::vector<double> dmag;
    for (int d = 1; d < f0.n_chips; ++d) {
        const int t1 = std::min(Lf, d * ks), t0 = std::min(Lf, (d - 1) * ks);
        dmag.push_back(std::abs(P[(size_t)t1] - P[(size_t)t0]));
    }
    std::sort(dmag.begin(), dmag.end());
    const double dmed = dmag[dmag.size() / 2], dmin = dmag.front(), dmaxv = dmag.back();
    printf("\n[1] PRECISION\n");
    printf("    max|Phi| over the table      %12.6g\n", pmax);
    printf("    |dPhi| over one chip step:   min %.4g  median %.4g  max %.4g\n", dmin, dmed,
           dmaxv);
    printf("    cancellation: max|Phi|/median|dPhi| = %.1f (%.1f bits), "
           "max|Phi|/min|dPhi| = %.0f (%.1f bits)\n",
           pmax / dmed, std::log2(pmax / dmed), pmax / dmin, std::log2(pmax / dmin));
    printf("    mantissa bits: fp32 24, fp16 11, bf16 8\n");
    // What does rounding Phi to fp16/bf16 actually cost the DIFFERENCE?
    auto quant = [&](int mant) {
        // TRUE float behaviour: every value keeps its OWN exponent and gets `mant` mantissa
        // bits, which is what fp16/bf16 do -- not a shared-exponent fixed point.
        auto rnd = [&](double x) {
            if (x == 0.0)
                return 0.0;
            int ex = 0;
            std::frexp(x, &ex);
            const double u = std::ldexp(1.0, ex - mant - 1);
            return std::round(x / u) * u;
        };
        double worst = 0.0, med = 0.0;
        std::vector<double> rel;
        for (int d = 1; d < f0.n_chips; ++d) {
            const int t1 = std::min(Lf, d * ks), t0 = std::min(Lf, (d - 1) * ks);
            auto q = [&](cd z) { return cd(rnd(z.real()), rnd(z.imag())); };
            const cd ex = P[(size_t)t1] - P[(size_t)t0];
            const cd ap = q(P[(size_t)t1]) - q(P[(size_t)t0]);
            if (std::abs(ex) > 0.0)
                rel.push_back(std::abs(ap - ex) / std::abs(ex));
        }
        std::sort(rel.begin(), rel.end());
        worst = rel.back();
        med = rel[rel.size() / 2];
        printf("        %2d bits -> median %.3e   worst %.3e\n", mant, med, worst);
        return worst;
    };
    printf("    worst RELATIVE error on dPhi if Phi is stored with N mantissa bits (shared exp):\n");
    for (int m : {23, 15, 11, 8})
        (void)quant(m);

    // ---- 2. how far apart are two Dopplers' tables? ----------------------------------------
    const auto f1 = bank.hoprate_filter(want, dop + dhz);
    const auto& Q = f1.PhiA[0];
    double relmax = 0.0, tiltmax = 0.0;
    for (int d = 1; d < f0.n_chips; ++d) {
        const int t1 = std::min(Lf, d * ks), t0 = std::min(Lf, (d - 1) * ks);
        const cd a = P[(size_t)t1] - P[(size_t)t0], b = Q[(size_t)t1] - Q[(size_t)t0];
        if (std::abs(a) > 0.0)
            relmax = std::max(relmax, std::abs(b - a) / std::abs(a));
    }
    tiltmax = 2.0 * M_PI * dhz / FS * (double)Lf;
    printf("\n[2] DOPPLER SEPARATION  (%.0f Hz apart)\n", dhz);
    printf("    phase tilt across the whole span 2*pi*ddop/fs*Lf = %.3f rad\n", tiltmax);
    printf("    worst relative change in a chip-step dPhi        = %.3e\n", relmax);
    printf("    -> a shared table + low-order correction must beat THIS to be worth it\n");

    // ---- 3. CAN ONE SHARED TABLE SET COVER A DOPPLER BUCKET? ---------------------------------
    // Phi(delta) is analytic in the Doppler offset, so build the expansion NUMERICALLY from a
    // few exact tables at the bucket nodes -- which is how it would be built for real, and it
    // avoids needing the prototype taps. Linear = 2 nodes at the edges; quadratic = 3 nodes
    // (edges + centre). Then evaluate INSIDE the bucket, where interpolation is worst, and
    // compare the chip-step differences the gather actually consumes.
    printf("\n[3] SHARED-TABLE RECONSTRUCTION over a Doppler bucket\n");
    printf("%10s %8s %14s %14s\n", "half-width", "order", "max rel err", "median rel err");
    for (double W : {250.0, 500.0, 1000.0, 2000.0, 5000.0}) {
        const auto fm = bank.hoprate_filter(want, dop - W);
        const auto fc = bank.hoprate_filter(want, dop);
        const auto fp = bank.hoprate_filter(want, dop + W);
        const auto &Pm = fm.PhiA[0], &Pc = fc.PhiA[0], &Pp = fp.PhiA[0];
        for (int order : {1, 2}) {
            double worst = 0.0;
            std::vector<double> rel;
            // Probe inside the bucket -- the nodes themselves are exact by construction.
            for (double frac : {-0.75, -0.5, -0.25, 0.25, 0.5, 0.75}) {
                const double d = frac * W;
                const auto fe = bank.hoprate_filter(want, dop + d);
                const auto& Pe = fe.PhiA[0];
                const double u = d / W; // in [-1,1]
                for (int dd = 1; dd < f0.n_chips; ++dd) {
                    const int t1 = std::min(Lf, dd * ks), t0 = std::min(Lf, (dd - 1) * ks);
                    auto dphi = [&](const std::vector<cd>& T) {
                        return T[(size_t)t1] - T[(size_t)t0];
                    };
                    cd approx;
                    if (order == 1) // linear through the two edges
                        approx = 0.5 * (1.0 - u) * dphi(Pm) + 0.5 * (1.0 + u) * dphi(Pp);
                    else // quadratic Lagrange through -W, 0, +W
                        approx = 0.5 * u * (u - 1.0) * dphi(Pm) + (1.0 - u * u) * dphi(Pc)
                                 + 0.5 * u * (u + 1.0) * dphi(Pp);
                    const cd ex = dphi(Pe);
                    if (std::abs(ex) > 0.0)
                        rel.push_back(std::abs(approx - ex) / std::abs(ex));
                }
            }
            std::sort(rel.begin(), rel.end());
            worst = rel.back();
            printf("%9.0f %8d %14.3e %14.3e\n", W, order, worst, rel[rel.size() / 2]);
        }
    }
    printf("    (fp16 storage costs 3.3e-04 -- a reconstruction much better than that is wasted)\n");
    return 0;
}
