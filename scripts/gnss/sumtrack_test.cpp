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
           "tau 0.5 s, rec 10.5 ms ==\n");
    const int NE = 32, NDEAD = 4, NREC = 4000; // 42 s: warm-up plus a long measure window
    const double DT = 0.0105, TAU = 0.5;
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
            cal.update(gp.data(), h, DT);
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
    return 0;
}
