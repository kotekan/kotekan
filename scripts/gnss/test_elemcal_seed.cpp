// Offline validation of ElemCal::seed() (path B warm-start). Synthetic: each element sees the
// SAME satellite signal through a fixed instrumental gain a_e = A_e e^{i phi_e} plus thermal
// noise. A cold cal must bootstrap ~3 tau before combine() coheres; a SEEDED cal (given the
// a_e) must cohere on the FIRST record, and must stay coherent across a reset() (re-anchor).
//
//   g++ -std=c++17 -O2 -I lib/stages/gnss scripts/gnss/test_elemcal_seed.cpp -o /tmp/tes && /tmp/tes
#include "gnssElemCal.hpp"
#include <cstdio>
#include <random>
#include <complex>
#include <vector>

using cd = std::complex<double>;
static std::mt19937 rng(12345);
static std::normal_distribution<double> gn(0.0, 1.0);

int main() {
    const int N = 32;
    const double NOISE = 0.30;           // per-element thermal amplitude (signal amp = 1)
    std::vector<cd> a(N);                 // fixed instrumental gains (random phase, unit-ish mag)
    for (int e = 0; e < N; ++e) {
        double ph = 2.0 * M_PI * (double)(e * 2654435761u % 1000) / 1000.0;  // deterministic "random"
        a[e] = std::polar(0.8 + 0.4 * ((e * 7 % 5) / 4.0), ph);
    }
    auto draw = [&](std::vector<cd>& g) {                 // one record: a_e * S + noise
        cd S(1.0, 0.0);
        for (int e = 0; e < N; ++e)
            g[e] = a[e] * S + cd(NOISE * gn(rng), NOISE * gn(rng));
    };
    auto coh_gain = [&](const gnss::ElemCal& c, const std::vector<cd>& g) {
        // |combine| vs mean per-element |g|. Coherent N-element sum ~ sqrt(N)-ish in SNR; here
        // we check the combine MAGNITUDE relative to a single element -- a phased sum lands near
        // |S| with the noise beaten down, an UNphased sum is ~1/sqrt(N) of that.
        cd cm = c.combine(g.data());
        double me = 0; for (auto& v : g) me += std::abs(v); me /= N;
        return std::abs(cm) / me;
    };

    int fails = 0;
    std::vector<cd> g(N);

    // 1. COLD cal: not warm on record 1, combine ~0 (no weights yet).
    gnss::ElemCal cold(N, 0, 5.0, 0.02);
    draw(g); cold.update(g.data(), 0.1);
    if (cold.warm()) { printf("FAIL: cold cal warm after 1 record\n"); fails++; }

    // 2. SEEDED cal: warm immediately, and combine COHERES on record 1.
    gnss::ElemCal seeded(N, 0, 5.0, 0.02);
    seeded.seed(a.data(), N);
    if (!seeded.warm()) { printf("FAIL: seeded cal not warm\n"); fails++; }
    draw(g);
    double gseed = coh_gain(seeded, g);
    // a coherent phased combine keeps ~|S|/mean|g| ~ 1/(1+noise) ~ 0.7-1.0; an unphased one is
    // ~1/sqrt(N) ~ 0.18. Assert clearly on the coherent side.
    if (gseed < 0.6) { printf("FAIL: seeded combine did not cohere: gain %.3f\n", gseed); fails++; }
    else printf("ok  seeded combine coheres on record 1: gain %.3f (unphased ~%.2f)\n", gseed, 1.0/std::sqrt((double)N));

    // 3. WRONG (random) prior does NOT cohere -- proves the phases are doing the work.
    std::vector<cd> bad(N);
    for (int e = 0; e < N; ++e) bad[e] = std::polar(1.0, 2.0*M_PI*((e*40503u%997)/997.0));
    gnss::ElemCal wrong(N, 0, 5.0, 0.02);
    wrong.seed(bad.data(), N);
    draw(g);
    double gbad = coh_gain(wrong, g);
    if (gbad > 0.45) { printf("FAIL: a WRONG prior still cohered (%.3f) -- test not discriminating\n", gbad); fails++; }
    else printf("ok  wrong prior stays incoherent: gain %.3f\n", gbad);

    // 4. reset() RE-APPLIES the prior: still warm, still coherent (the deadlock-breaker).
    seeded.reset();
    if (!seeded.warm()) { printf("FAIL: seeded cal cold after reset (prior not re-applied)\n"); fails++; }
    draw(g);
    double gre = coh_gain(seeded, g);
    if (gre < 0.6) { printf("FAIL: combine incoherent after reset: gain %.3f\n", gre); fails++; }
    else printf("ok  reset re-applies prior: warm + coherent (gain %.3f)\n", gre);

    // 5. clearing the prior (all-zero) returns to cold self-bootstrap on the next reset.
    std::vector<cd> zero(N, cd(0,0));
    seeded.seed(zero.data(), N);
    seeded.reset();
    if (seeded.warm()) { printf("FAIL: cleared prior still warms on reset\n"); fails++; }
    else printf("ok  all-zero prior clears -> reset is cold again\n");

    printf(fails ? "\nFAILED (%d)\n" : "\nALL GOOD\n", fails);
    return fails ? 1 : 0;
}
