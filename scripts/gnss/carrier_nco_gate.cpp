// THE GATE FOR #71: does the replica carrier phase survive a Doppler update?
//
// Drives gnss::carrier_nco_advance and gnss::carrier_phase_absolute -- the SAME expressions
// GnssCudaDespread::Impl calls -- rather than a second copy of the arithmetic, so a pass here
// cannot be a pass against a formula production does not use.
//
// ⚠️ WHAT A GATE MUST VARY. The axis is THE DOPPLER CHANGING BETWEEN RECORDS, because that is
// the only thing that distinguishes the two arms: at a frozen Doppler they agree exactly, and
// a test that forgot to move it would pass against the bug. Leg 0 checks precisely that they
// agree when nothing moves, so the later legs' disagreement is attributable.
//
//   build:  g++ -O2 -I lib/stages/gnss -o scripts/gnss/carrier_nco_gate \
//                  scripts/gnss/carrier_nco_gate.cpp
#include "gnssCarrierNco.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double FS = 3.2e9;              // combined-stream sample rate
constexpr long long REC = 2048LL * 16384; // samples per record
constexpr double F_CARRIER = 1.17645e9;   // L5/E5a
constexpr long long GAP_MAX = 64LL * REC;
constexpr double TWO_PI = 6.283185307179586;

double wrap(double x) {
    while (x > M_PI)
        x -= TWO_PI;
    while (x < -M_PI)
        x += TWO_PI;
    return x;
}

/// Physical phase advanced over one record at the mean frequency -- what the replica SHOULD do.
double truth_step(double f0, double f1, long long dn) {
    const long double fbar = 0.5L * ((long double)f0 + (long double)f1);
    long double cyc = fmodl(fbar * (long double)dn / (long double)FS, 1.0L);
    return (double)(TWO_PI * cyc);
}

int fails = 0;
void check(bool ok, const char* what, double got, double want) {
    if (!ok) {
        fails++;
        std::printf("  FAIL %-52s got %+.6g want %+.6g\n", what, got, want);
    } else {
        std::printf("  ok   %-52s %+.6g\n", what, got);
    }
}

} // namespace

int main() {
    // t0 = the absolute-sample lever the old arm multiplies the frequency by: a week of uptime.
    const long long n_start = (long long)(5.85e5 * FS);

    // ── LEG 0: a FROZEN Doppler. Both arms must track the truth; if this fails, nothing
    //    later is attributable to the Doppler moving. ────────────────────────────────────
    std::printf("LEG 0  frozen Doppler -- the arms must AGREE (else no later leg is "
                "attributable)\n");
    {
        gnss::CarrierNco a;
        unsigned long long re = 0;
        const double f = F_CARRIER + 1234.0;
        double prev_acc = 0, prev_abs = 0;
        double worst_acc = 0, worst_abs = 0;
        for (int k = 0; k < 8; ++k) {
            const long long n = n_start + (long long)k * REC;
            const double acc = gnss::carrier_nco_advance(a, f, n, FS,
                                                         gnss::carrier_phase_absolute(f, n, FS),
                                                         GAP_MAX, &re);
            const double abs_ = gnss::carrier_phase_absolute(f, n, FS);
            if (k) {
                const double t = truth_step(f, f, REC);
                worst_acc = std::fmax(worst_acc, std::fabs(wrap(acc - prev_acc - t)));
                worst_abs = std::fmax(worst_abs, std::fabs(wrap(abs_ - prev_abs - t)));
            }
            prev_acc = acc;
            prev_abs = abs_;
        }
        check(worst_acc < 1e-6, "accumulated step == physical (frozen f)", worst_acc, 0.0);
        // ⚠️ THE ABSOLUTE ARM CANNOT REACH ZERO EVEN HERE, and the bar says so rather than
        // pretending: frac(f*n0/fs) at n0 ~ 1.9e15 discards ~6.9e14 whole cycles, leaving the
        // ~2.3e-4 rad long-double floor that ang0_for's own comment quotes. Measured 2.9e-4.
        // That residue is why arm 1 exists at all -- but it is a PRECISION floor, three orders
        // below the RADIANS that arm 2 removes, so the two must not be conflated.
        check(worst_abs < 1e-3, "absolute    step == physical to its long-double floor",
              worst_abs, 0.0);
        check(re == 1, "exactly one re-anchor (the first sight)", (double)re, 1.0);
    }

    // ── LEG 1: a Doppler RAMP, the live case. dop_rate * 10.5 ms per record. ────────────
    std::printf("\nLEG 1  Doppler ramp (the live case): does a frequency update TELEPORT the "
                "phase?\n");
    for (double dop_rate : {0.01, 0.1, 0.5}) {
        gnss::CarrierNco a;
        unsigned long long re = 0;
        const double dt = (double)REC / FS;
        double prev_acc = 0, prev_abs = 0, f_prev = 0;
        double worst_acc = 0, worst_abs = 0;
        for (int k = 0; k < 32; ++k) {
            const long long n = n_start + (long long)k * REC;
            const double f = F_CARRIER + 1234.0 + dop_rate * dt * k;
            const double acc = gnss::carrier_nco_advance(a, f, n, FS,
                                                         gnss::carrier_phase_absolute(f, n, FS),
                                                         GAP_MAX, &re);
            const double abs_ = gnss::carrier_phase_absolute(f, n, FS);
            if (k) {
                const double t = truth_step(f_prev, f, REC);
                worst_acc = std::fmax(worst_acc, std::fabs(wrap(acc - prev_acc - t)));
                worst_abs = std::fmax(worst_abs, std::fabs(wrap(abs_ - prev_abs - t)));
            }
            prev_acc = acc;
            prev_abs = abs_;
            f_prev = f;
        }
        char msg[128];
        std::snprintf(msg, sizeof msg, "accumulated: step error, dop_rate %.2f Hz/s", dop_rate);
        check(worst_acc < 1e-6, msg, worst_acc, 0.0);
        // The old arm must FAIL here, and be SEEN to fail -- a gate whose bad arm quietly
        // passes is not measuring the axis it claims to.
        std::snprintf(msg, sizeof msg, "absolute:    step error (EXPECTED LARGE) %.2f Hz/s",
                      dop_rate);
        check(worst_abs > 1.0, msg, worst_abs, 3.14);
    }

    // ── LEG 2: a single Doppler STEP -- the re-seed case, isolated. ─────────────────────
    std::printf("\nLEG 2  one Doppler step of 1e-6 Hz (a re-seed): the lever, in one number\n");
    {
        const double f0 = F_CARRIER + 1000.0, f1 = f0 + 1e-6;
        const long long n1 = n_start, n2 = n_start + REC;
        gnss::CarrierNco a;
        unsigned long long re = 0;
        const double a1 = gnss::carrier_nco_advance(a, f0, n1, FS,
                                                    gnss::carrier_phase_absolute(f0, n1, FS),
                                                    GAP_MAX, &re);
        const double a2 = gnss::carrier_nco_advance(a, f1, n2, FS, 0.0, GAP_MAX, &re);
        const double t = truth_step(f0, f1, REC);
        const double err_acc = std::fabs(wrap(a2 - a1 - t));
        const double b1 = gnss::carrier_phase_absolute(f0, n1, FS);
        const double b2 = gnss::carrier_phase_absolute(f1, n2, FS);
        const double err_abs = std::fabs(wrap(b2 - b1 - t));
        check(err_acc < 1e-6, "accumulated: 1e-6 Hz step costs nothing", err_acc, 0.0);
        check(err_abs > 1.0, "absolute:    1e-6 Hz step TELEPORTS (expected)", err_abs, 3.14);
        std::printf("       (lever: 2*pi*df*n0/fs = %.3g rad for df = 1e-6 Hz)\n",
                    TWO_PI * 1e-6 * (double)n_start / FS);
    }

    // ── LEG 3: gaps. A dropped record must integrate through; a long hole must re-anchor
    //    and be COUNTED, because that is a genuine discontinuity a consumer has to see. ──
    std::printf("\nLEG 3  gaps: integrate through a short one, re-anchor and COUNT a long one\n");
    {
        gnss::CarrierNco a;
        unsigned long long re = 0;
        const double f = F_CARRIER + 500.0;
        const double p1 = gnss::carrier_nco_advance(a, f, n_start, FS,
                                                    gnss::carrier_phase_absolute(f, n_start, FS),
                                                    GAP_MAX, &re);
        const unsigned long long re_after_first = re;
        // 3-record hole: still integrable.
        const double p2 = gnss::carrier_nco_advance(a, f, n_start + 4 * REC, FS, 0.0, GAP_MAX,
                                                    &re);
        const double t = truth_step(f, f, 4 * REC);
        check(std::fabs(wrap(p2 - p1 - t)) < 1e-6, "short gap integrates through",
              std::fabs(wrap(p2 - p1 - t)), 0.0);
        check(re == re_after_first, "short gap does NOT re-anchor", (double)(re - re_after_first),
              0.0);
        // Beyond GAP_MAX: must re-anchor and count.
        gnss::carrier_nco_advance(a, f, n_start + 4 * REC + GAP_MAX + REC, FS, 0.0, GAP_MAX, &re);
        check(re == re_after_first + 1, "long gap re-anchors and is counted",
              (double)(re - re_after_first), 1.0);
        // Idempotence: asking twice for the same record must not advance it.
        const double q1 = gnss::carrier_nco_advance(a, f, a.n0, FS, 0.0, GAP_MAX, &re);
        const double q2 = gnss::carrier_nco_advance(a, f, a.n0, FS, 0.0, GAP_MAX, &re);
        check(q1 == q2, "idempotent for a repeated n0 (build_jobs has 3 callers)",
              std::fabs(q1 - q2), 0.0);
    }

    std::printf("\ncarrier_nco_gate: %s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
