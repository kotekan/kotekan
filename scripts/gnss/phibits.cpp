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

    // ---- 2b. THE MULTIPLICATIVE FORM: one table for the WHOLE SKY? --------------------------
    //
    // [2] and [3] both measure ADDITIVE reconstructions of Phi itself, and [2]'s 0.63 is not a
    // property of the table at all -- it is |exp(i * 0.643) - 1|, i.e. the pure ENDPOINT PHASE
    // TILT across the span. The gather never uses Phi; it uses the DIFFERENCE over one chip
    // window (~ks taps), and over that window the extra Doppler advances only
    // ddop/fs*ks ~ 3e-3 rad. So write the difference as
    //
    //     dPhi(w0 + ddw)  =  exp(-i*ddw*kbar) * dPhi(w0) * [1 + O((ddw*spread)^2)]
    //
    // with kbar the PROTOTYPE-WEIGHTED CENTROID of the window (the weighting is what makes the
    // first-order term vanish; using k_lo or the midpoint instead leaves a first-order residual
    // and is measured below for comparison). The correction is ONE complex multiply per chip,
    // and since kbar advances by ~ks per chip it is a CONSTANT ROTOR -- no table, no
    // transcendental in the loop.
    //
    // If the residual here is below the fp16 storage error (3.3e-04 from [1]), then the Doppler
    // does not need a table at all: ONE Phi per channel serves every PRN and every Doppler, and
    // the per-PRN 1.05 MB collapses. That is a different claim from [3]'s buckets, and a much
    // stronger one, so it gets measured rather than argued.
    printf("\n[2b] MULTIPLICATIVE (rotor) RECONSTRUCTION -- one table for the whole sky\n");
    printf("      dop offset    COMPLEX centroid    |w|-centroid    midpoint      no rotor\n");
    for (double dd : {250.0, 1000.0, 5000.0, 10000.0}) {
        const auto fx = bank.hoprate_filter(want, dop + dd);
        const auto& X = fx.PhiA[0];
        double w_cen = 0.0, w_mid = 0.0, w_non = 0.0, w_real = 0.0;
        const double ddw = 2.0 * M_PI * dd / FS;
        for (int d = 1; d < f0.n_chips; ++d) {
            const int t1 = std::min(Lf, d * ks), t0 = std::min(Lf, (d - 1) * ks);
            if (t1 <= t0)
                continue;
            const cd ex = X[(size_t)t1] - X[(size_t)t0];   // the exact Doppler-shifted step
            const cd base = P[(size_t)t1] - P[(size_t)t0]; // the SHARED (dop) table's step
            if (std::abs(ex) == 0.0)
                continue;
            // prototype-weighted centroid of THIS window, from the shared table's own
            // increments -- available wherever the table is, no extra state.
            // THE CENTROID MUST BE COMPLEX. dPhi = sum_j w_j with w_j a ROTATING phasor, so
            // the first-order term of sum_j w_j exp(-i*ddw*j) is -i*ddw*sum_j j*w_j, and the
            // multiplier that cancels it is exp(-i*ddw*c) with c = sum_j j*w_j / sum_j w_j --
            // complex. Weighting by |w_j| instead gives a REAL centroid, leaves the
            // first-order term alive, and the error then scales LINEARLY in the Doppler
            // (measured: 2.3e-4 / 9.3e-4 / 4.7e-3 / 9.4e-3 at 250/1k/5k/10k Hz -- exactly
            // proportional). Both are reported so the difference is visible.
            cd cnum(0.0, 0.0), cden(0.0, 0.0);
            double num = 0.0, den = 0.0;
            for (int j = t0; j < t1; ++j) {
                const cd w = P[(size_t)(j + 1)] - P[(size_t)j];
                cnum += w * (double)j;
                cden += w;
                const double wgt = std::abs(w);
                num += wgt * (double)j;
                den += wgt;
            }
            const cd cbar = (std::abs(cden) > 0.0) ? cnum / cden : cd(0.5 * (t0 + t1), 0.0);
            const double kbar = (den > 0.0) ? num / den : 0.5 * (t0 + t1);
            const double kmid = 0.5 * (t0 + t1);
            const cd rc = std::exp(cd(0.0, -ddw) * cbar);
            const cd rr = std::exp(cd(0.0, -ddw * kbar));
            const cd rm = std::exp(cd(0.0, -ddw * kmid));
            w_cen = std::max(w_cen, std::abs(rc * base - ex) / std::abs(ex));
            w_real = std::max(w_real, std::abs(rr * base - ex) / std::abs(ex));
            w_mid = std::max(w_mid, std::abs(rm * base - ex) / std::abs(ex));
            w_non = std::max(w_non, std::abs(base - ex) / std::abs(ex));
        }
        printf("      %7.0f Hz     %.3e        %.3e     %.3e   %.3e\n",
               dd, w_cen, w_real, w_mid, w_non);
    }
    printf("      (fp16 storage costs 3.3e-04 -- anything below that is free accuracy)\n");

    // ---- 2c. THE FORM THAT WOULD SHIP -------------------------------------------------------
    // 2b's complex centroid c = sum_j j*w_j / sum_j w_j is exact to second order but needs a
    // per-window quantity, i.e. a SECOND prefix table Psi[k] = sum_{j<k} j*proto[j]*e^{-i(off+w0)j}.
    // Two shared tables still beats one table PER PRN by 12-16x, but the reconstruction must
    // avoid a complex divide and a transcendental per chip. Split c into the part we know from
    // the INDICES and the part that needs Psi:
    //
    //   exp(-i*ddw*c) = exp(-i*ddw*mid) * exp(-i*ddw*(c - mid)),   mid = (t0+t1)/2
    //
    // The first factor is a CONSTANT ROTOR (mid advances by ~ks per chip: one complex multiply,
    // no transcendental). The second has a SMALL argument -- |c - mid| ~ 470 taps, so
    // ddw*(c-mid) ~ 5e-3 rad at 5 kHz -- and linearizes:
    //
    //   dPhi(w0+ddw) ~ exp(-i*ddw*mid) * [ dPhi_0 - i*ddw*( dPsi - mid*dPhi_0 ) ]
    //
    // Per chip that is: the Psi telescope (2 loads, carried like Phi's), one complex FMA, one
    // rotor multiply. No divide, no exp in the loop. This is the candidate; 2b's complex
    // centroid is its unreachable ceiling and is kept above as the bound.
    printf("\n[2c] SHIPPABLE FORM: rotor(midpoint) + linear Psi correction, TWO shared tables\n");
    printf("      ANCHOR: midpoint (algebra) vs t_prev (what the kernel can advance)\n");
    {
        // Psi over the SHARED table's Doppler -- built once, Doppler-free, like Phi.
        std::vector<cd> Psi((size_t)Lf + 1, cd(0.0, 0.0));
        for (int j = 0; j < Lf; ++j)
            Psi[(size_t)(j + 1)] = Psi[(size_t)j] + (double)j * (P[(size_t)(j + 1)] - P[(size_t)j]);
        for (double dd : {250.0, 1000.0, 5000.0, 10000.0}) {
            const auto fx = bank.hoprate_filter(want, dop + dd);
            const auto& X = fx.PhiA[0];
            const double ddw = 2.0 * M_PI * dd / FS;
            double worst = 0.0, worst_p = 0.0;
            for (int d = 1; d < f0.n_chips; ++d) {
                const int t1 = std::min(Lf, d * ks), t0 = std::min(Lf, (d - 1) * ks);
                if (t1 <= t0)
                    continue;
                const cd ex = X[(size_t)t1] - X[(size_t)t0];
                if (std::abs(ex) == 0.0)
                    continue;
                const cd base = P[(size_t)t1] - P[(size_t)t0];
                const cd dpsi = Psi[(size_t)t1] - Psi[(size_t)t0];
                const double mid = 0.5 * ((double)t0 + (double)t1);
                const cd corr = base - cd(0.0, ddw) * (dpsi - mid * base);
                const cd got = std::exp(cd(0.0, -ddw * mid)) * corr;
                worst = std::max(worst, std::abs(got - ex) / std::abs(ex));
                // ANCHOR ON t_prev INSTEAD OF THE MIDPOINT. On-device the rotor must
                // ADVANCE, and t_prev steps by exactly ks or ks+1 (f accumulates kf in
                // [0,1)) -- two precomputed rotors and an exact select, no transcendental in
                // the loop. The midpoint steps by ks, ks+1/2 or ks+1: three cases and a half
                // step, so it is the wrong anchor for the kernel however natural it is for
                // the algebra. t_prev sits ~ks/2 further from the centroid, so the linear
                // term is larger and this is a measured question, not an obvious one.
                const double tp = (double)t0;
                const cd corr_p = base - cd(0.0, ddw) * (dpsi - tp * base);
                const cd got_p = std::exp(cd(0.0, -ddw * tp)) * corr_p;
                worst_p = std::max(worst_p, std::abs(got_p - ex) / std::abs(ex));
            }
            printf("      %7.0f Hz   mid %.3e  t_prev %.3e   (fp16/ %.0fx, %.0fx)\n",
                   dd, worst, worst_p, 3.3e-4 / worst, 3.3e-4 / worst_p);
        }
    }



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
