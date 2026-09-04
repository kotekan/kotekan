// Synthetic validation of the two phase-floor processing fixes (CHORD_GNSS_STATE 8.21.5):
//
//   [A] gnss::phase_track_loo -- the leave-one-out common-phase tracker. Injects the measured
//       on-sky wander (AR(1), sigma 0.9 rad, ~42 ms correlation) into a synthetic record
//       stream and compares the deep coherent_sum straight / tracked / genie (true phase
//       removed), across per-record SNRs bracketing the fleet -- PLUS a noise-only control:
//       the self-excluded estimate must NOT inflate the noise-only deep snr (fail-closed).
//
//   [B] gnss::ElemCal -- the self-calibrated weighted element sum. 32 elements with random
//       static complex gains, several dead, common wander on all of them; checks the sum's
//       per-record SNR gain vs the reference element against the MRC bound, that slow fringe
//       rotation is tracked, and that a dead reference falls over to a healthy anchor.
//
// Build: ./build_tool.sh sumtrack_test    Run: ./sumtrack_test
// Everything is deterministic (fixed seeds) so a regression is a diff, not a maybe.

#include "gnssChannelizedDespread.hpp"
#include "gnssElemCal.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>

using cd = std::complex<double>;

// AR(1) phase wander: stationary sigma `sig` rad, correlation time `tau_rec` records.
static std::vector<double> make_wander(int n, double sig, double tau_rec, std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    const double rho = std::exp(-1.0 / tau_rec);
    const double se = sig * std::sqrt(1.0 - rho * rho);
    std::vector<double> phi((size_t)n);
    phi[0] = sig * g(rng);
    for (int k = 1; k < n; ++k)
        phi[(size_t)k] = rho * phi[(size_t)k - 1] + se * g(rng);
    return phi;
}

int main() {
    printf("== [A] phase_track_loo: AR(1) wander sigma 0.9 rad, corr 4 records, N=190 "
           "(2 s of CHORD records), 40 trials ==\n");
    printf("%8s %10s | tracked at half_width:%26s | %10s\n", "snr/rec", "straight", "", "genie");
    printf("%8s %10s %10s %10s %10s %10s %10s\n", "", "", "hw=1", "hw=2", "hw=4", "hw=8", "");
    const int N = 190, TRIALS = 40;
    const int HWS[] = {1, 2, 4, 8};
    for (double s : {0.0, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        double m_st = 0.0, m_ge = 0.0, m_tr[4] = {0, 0, 0, 0}, mx_tr = 0.0;
        std::mt19937 rng(12345);
        std::normal_distribution<double> g(0.0, 1.0);
        for (int t = 0; t < TRIALS; ++t) {
            const std::vector<double> phi = make_wander(N, 0.9, 4.0, rng);
            std::vector<cd> a((size_t)N), genie((size_t)N);
            for (int k = 0; k < N; ++k) {
                const cd n(g(rng), g(rng));
                a[(size_t)k] = s * std::polar(1.0, phi[(size_t)k]) + n;
                genie[(size_t)k] = a[(size_t)k] * std::polar(1.0, -phi[(size_t)k]);
            }
            m_st += gnss::coherent_sum(a).snr;
            m_ge += gnss::coherent_sum(genie).snr;
            for (int h = 0; h < 4; ++h) {
                std::vector<cd> tr;
                gnss::phase_track_loo(a, HWS[h], tr);
                const double tk = gnss::coherent_sum(tr).snr;
                m_tr[h] += tk;
                if (tk > mx_tr)
                    mx_tr = tk;
            }
        }
        m_st /= TRIALS;
        m_ge /= TRIALS;
        for (int h = 0; h < 4; ++h)
            m_tr[h] /= TRIALS;
        if (s == 0.0)
            printf("%8.1f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f   NOISE CONTROL: max "
                   "tracked over %d trials x 4 hw = %.2f (fail-closed: no inflation)\n",
                   s, m_st, m_tr[0], m_tr[1], m_tr[2], m_tr[3], m_ge, TRIALS, mx_tr);
        else
            printf("%8.1f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f   best %.2fx\n", s, m_st,
                   m_tr[0], m_tr[1], m_tr[2], m_tr[3], m_ge,
                   std::max({m_tr[0], m_tr[1], m_tr[2], m_tr[3]}) / m_st);
    }

    printf("\n== [B] ElemCal: 32 elements (4 dead), static gains amp 0.5-1.5, common wander, "
           "tau 5 s, rec 10.5 ms ==\n");
    const int NE = 32, NDEAD = 4, NREC = 20000; // 42 s: warm-up plus a long measure window
    const double DT = 0.0105, TAU = 5.0;
    auto run_elemcal = [&](double s_elem, double fringe_hz, bool kill_ref, const char* label) {
        std::mt19937 rng(777);
        std::normal_distribution<double> g(0.0, 1.0);
        std::uniform_real_distribution<double> ua(0.5, 1.5), up(0.0, 2.0 * M_PI);
        std::vector<cd> gains((size_t)NE);
        for (int e = 0; e < NE; ++e)
            gains[(size_t)e] = std::polar(ua(rng), up(rng));
        for (int e = NE - NDEAD; e < NE; ++e)
            gains[(size_t)e] = 0.0; // absent/unpowered
        if (kill_ref)
            gains[0] = 0.0;
        gnss::ElemCal cal(NE, /*ref*/ 0, TAU, /*min_w_frac*/ 0.02);
        const std::vector<double> phi = make_wander(NREC, 0.9, 4.0, rng);
        std::vector<cd> gp((size_t)NE);
        // Measure per-record SNR of the emitted header vs the reference element: demodulate by
        // the KNOWN signal phase (so the wander counts against neither), then SNR = |complex
        // mean| / std of the component ORTHOGONAL to it -- phase-convention-free, so the ref
        // element's arbitrary gain phase (which both carry) cannot alias into the number.
        std::vector<cd> dm_h, dm_r;
        bool moved = false;
        for (int k = 0; k < NREC; ++k) {
            const cd sig = s_elem * std::polar(1.0, phi[(size_t)k]);
            const cd fr = std::polar(1.0, 2.0 * M_PI * fringe_hz * k * DT);
            for (int e = 0; e < NE; ++e) {
                const cd ge = (e >= NE / 2) ? gains[(size_t)e] * fr : gains[(size_t)e];
                gp[(size_t)e] =
                    (std::norm(ge) > 0.0) ? ge * sig + cd(g(rng), g(rng)) : cd(g(rng), g(rng));
            }
            const cd href = gp[0];
            const cd h = cal.warm() ? cal.combine(gp.data()) : href;
            cal.update(gp.data(), DT);
            if (k >= NREC / 2 && cal.warm()) { // measure on the settled second half
                moved = moved || cal.anchor_moved();
                const cd dem = std::polar(1.0, -phi[(size_t)k]);
                dm_h.push_back(h * dem);
                dm_r.push_back(href * dem);
            }
        }
        auto snr_of = [](const std::vector<cd>& x) {
            if (x.size() < 2)
                return 0.0;
            cd m(0.0, 0.0);
            for (const cd& v : x)
                m += v;
            m /= (double)x.size();
            if (std::norm(m) <= 0.0)
                return 0.0;
            const cd rot = std::conj(m) / std::abs(m);
            double o2 = 0.0;
            for (const cd& v : x) {
                const double o = (v * rot).imag();
                o2 += o * o;
            }
            return std::abs(m) / std::sqrt(o2 / (double)x.size());
        };
        const double snr_h = snr_of(dm_h), snr_r = snr_of(dm_r);
        // MRC bound vs the reference element (amp 1 by construction of gains[0] draw? no --
        // measured): sqrt(sum |g_e|^2)/|g_ref| for live elements.
        double p2 = 0.0;
        for (int e = 0; e < NE; ++e)
            p2 += std::norm(gains[(size_t)e]);
        std::mt19937 rng2(777); // re-draw ref amplitude for the bound label
        std::uniform_real_distribution<double> ua2(0.5, 1.5), up2(0.0, 2.0 * M_PI);
        const double ref_amp = kill_ref ? 1.0 : std::abs(gains[0]);
        (void)ua2;
        (void)up2;
        (void)rng2;
        printf("  %-34s per-rec SNR: header %6.2f  ref-elem %6.2f  gain %5.2fx  "
               "(MRC bound %.2fx)%s\n",
               label, snr_h, snr_r, snr_r > 0.0 ? snr_h / snr_r : 0.0,
               std::sqrt(p2) / ref_amp, moved ? "  [anchor moved]" : "");
    };
    run_elemcal(0.6, 0.0, false, "static gains, s_elem 0.6:");
    run_elemcal(0.3, 0.0, false, "static gains, s_elem 0.3:");
    run_elemcal(0.6, 0.1, false, "0.1 Hz fringe on half the array:");
    run_elemcal(0.6, 0.0, true, "DEAD reference element:");

    // ---------------------------------------------------------------------------------------
    // [C] THE REAL CHORD CASE, as measured on sky 2026-08-05: a per-record common phase that is
    // coherent across ELEMENTS but ~white in TIME (lag-1 autocorr 0.57), on an array of ~8
    // effective elements with unequal gains and a couple of oscillating (excess-noise) feeds.
    // Temporal LOO cannot touch this; leave-one-ELEMENT-out should recover nearly all of it.
    // ---------------------------------------------------------------------------------------
    printf("\n== [C] sky phase (common across elements, white in time) + noisy feeds: "
           "deep fold over 128 records ==\n");
    printf("%22s %9s %9s %9s %9s %9s\n", "case", "straight", "temporal", "split-ap", "genie",
           "thermal");
    auto run_sky = [&](double sky_sig, int n_osc, double osc_excess, double s_elem,
                       const char* label, bool diag_on = false, int seed = 4242) {
        const int NE2 = 32, NLIVE = 8, NREC2 = 128 + 20000; // 2000 records of cal warm-up
        const double DT2 = 0.0105;
        std::mt19937 rng((unsigned)seed);
        std::normal_distribution<double> g(0.0, 1.0);
        std::uniform_real_distribution<double> ua(0.4, 1.4), up(0.0, 2.0 * M_PI);
        std::vector<cd> gains((size_t)NE2, cd(0.0, 0.0));
        for (int e = 0; e < NLIVE; ++e) // only NLIVE elements instrumented
            gains[(size_t)e] = std::polar(ua(rng), up(rng));
        std::vector<double> exn((size_t)NE2, 1.0);
        for (int e = 0; e < n_osc; ++e) // oscillating feeds: real signal, excess noise
            exn[(size_t)e + 1] = osc_excess;
        gnss::ElemCal cal(NE2, 0, 5.0, 0.02);
        std::vector<cd> gp((size_t)NE2);
        std::vector<cd> raw, sky, genie;
        std::vector<std::vector<cd>> hist; // per-record element vectors, for the shuffled null
        std::vector<cd> halfg;             // the INTEGRATING half's genie -- the real bound
        for (int k = 0; k < NREC2; ++k) {
            const double phi_sky = sky_sig * g(rng); // WHITE in time, common to all elements
            const cd sig = s_elem * std::polar(1.0, phi_sky);
            for (int e = 0; e < NE2; ++e)
                gp[(size_t)e] = gains[(size_t)e] * sig
                                + exn[(size_t)e] * cd(g(rng), g(rng)) * 0.70710678;
            const cd h = cal.warm() ? cal.combine(gp.data()) : gp[0];
            if (cal.warm() && k >= NREC2 - 128) {
                raw.push_back(h);
                sky.push_back(cal.combine_split(gp.data()));
                genie.push_back(h * std::polar(1.0, -phi_sky));
                hist.push_back(std::vector<cd>(gp.begin(), gp.end()));
                halfg.push_back(cal.combine_half_genie(gp.data(), phi_sky));
            }
            cal.update(gp.data(), DT2);
        }
        if (raw.size() < 64) {
            printf("  %-20s (cal never warmed)\n", label);
            return;
        }
        // SHUFFLED NULL (docs/gnss_phase_estimator_traps.md): re-run combine_split on records
        // whose per-element histories have been permuted INDEPENDENTLY per element. Amplitudes
        // and gains are untouched; the per-record COMMON phase -- the only thing there is to
        // recover -- is destroyed. A sound estimator must collapse to the incoherent floor here.
        // Shuffle, never roll: a roll leaves a shared ramp intact.
        std::vector<cd> nullv;
        {
            std::mt19937 sh(999);
            const int NR = (int)hist.size();
            std::vector<std::vector<int>> perm((size_t)NE2, std::vector<int>((size_t)NR));
            for (int e = 0; e < NE2; ++e) {
                for (int k = 0; k < NR; ++k) perm[(size_t)e][(size_t)k] = k;
                std::shuffle(perm[(size_t)e].begin(), perm[(size_t)e].end(), sh);
            }
            std::vector<cd> tmp((size_t)NE2);
            for (int k = 0; k < NR; ++k) {
                for (int e = 0; e < NE2; ++e)
                    tmp[(size_t)e] = hist[(size_t)perm[(size_t)e][(size_t)k]][(size_t)e];
                nullv.push_back(cal.combine_split(tmp.data()));
            }
        }
        std::vector<cd> tr;
        gnss::phase_track_loo(raw, 2, tr);
        printf("%22s %9.1f %9.1f %9.1f %9.1f %9s\n", label, gnss::coherent_sum(raw).snr,
               gnss::coherent_sum(tr).snr, gnss::coherent_sum(sky).snr,
               gnss::coherent_sum(genie).snr, "--");
        printf("      half-genie %6.1f (THE bound) | split/half %.3f | null %4.1f | "
               "wfrac(A) |w| %.3f |w|^2 %.3f\n",
               gnss::coherent_sum(halfg).snr,
               gnss::coherent_sum(sky).snr / gnss::coherent_sum(halfg).snr,
               gnss::coherent_sum(nullv).snr, cal.half_weight_frac(false),
               cal.half_weight_frac(true));
        if (diag_on) {
            // A TRUE genie leaves every record on a common phase. If it does not, it is not a
            // genie -- and then the "you may not beat it" rule is being applied to the wrong bar.
            auto spread = [](const std::vector<cd>& v, const char* nm) {
                cd M(0.0, 0.0);
                for (const cd& x : v) M += x;
                const double a0 = std::arg(M);
                double s1 = 0.0, s2 = 0.0, amp = 0.0;
                for (const cd& x : v) {
                    double d = std::arg(x) - a0;
                    while (d > M_PI) d -= 2.0 * M_PI;
                    while (d < -M_PI) d += 2.0 * M_PI;
                    s1 += d; s2 += d * d; amp += std::abs(x);
                }
                const double n = (double)v.size();
                printf("      %-6s |mean|=%9.3f  <|x|>=%9.3f  arg spread=%6.3f rad  bias=%+6.3f\n",
                       nm, std::abs(M) / n, amp / n, std::sqrt(s2 / n - (s1 / n) * (s1 / n)),
                       s1 / n);
            };
            spread(raw, "raw"); spread(sky, "split"); spread(genie, "genie");
            double rho2, th, mag; int live = 0;
            for (int e = 0; e < NE2; ++e) { cal.diag(e, rho2, th, mag); if (mag > 0.0) ++live; }
            printf("      cal: %d live elements\n", live);
        }
    };
    // s_elem sweep: the split-aperture correction only helps when the REFERENCE half has a
    // per-record SNR comfortably above 1, since arg(B) is the phase estimate. Live CHORD runs at
    // ~9.25 per record per instance over ~8 effective elements, i.e. s_elem ~ 3.
    run_sky(0.0,  0, 1.0, 0.55, "weak, no sky:");
    run_sky(0.75, 0, 1.0, 0.55, "weak, 0.75 rad:");
    run_sky(0.75, 2, 3.0, 0.55, "weak, 0.75 + 2 osc:");
    printf("%22s %9s %9s %9s %9s\n", "-- live CHORD level --", "", "", "", "");
    run_sky(0.0,  0, 1.0, 3.00, "live, no sky:");
    run_sky(0.75, 0, 1.0, 3.00, "live, 0.75 rad:");
    run_sky(0.75, 2, 3.0, 3.00, "live, 0.75 + 2 osc:", true);
    run_sky(1.20, 2, 3.0, 3.00, "live, 1.20 + 2 osc:", true);
    // Seed sweep: the array GAIN realisation changes with the seed, and it moves the genie by a
    // factor of 3 -- far more than anything being compared. One seed cannot support a conclusion
    // here, which is how "the genie moves with the injected amplitude" got recorded from seed
    // 4242 alone (7 of 8 other seeds show <7% between 0.75 and 1.20 rad).
    printf("%22s\n", "-- seed sweep: split vs its OWN bound --");
    for (int sd = 1; sd <= 8; ++sd) {
        char lb[64];
        snprintf(lb, sizeof lb, "1.20 seed %d:", sd);
        run_sky(1.20, 2, 3.0, 3.00, lb, false, sd);
    }
    return 0;
}
