#include "gnssSeedTransport.hpp"

#include <cmath>

namespace gnss {

DetectionPhase detection_phase(const ChannelizedReplicaBank& bank, double best_cp, int best_nh,
                               int n_nh, double dop, long long snap_start_hop, long long anchor,
                               double sample_rate) {
    DetectionPhase out;
    const int fft_len = bank.fft_len();
    const int Mp = bank.repl_period_hops();
    const double cps = bank.chip_rate_hz() / sample_rate;
    const double L = (double)bank.code_length();

    // best_cp is the satellite code phase at the snapshot's absolute start
    // (S_snap = snap_start_hop * fft_len). Reference it to absolute sample 0 so a consumer on
    // the same hop*2N grid can seed directly: cp0 = best_cp - S_snap*chip_per_sample (mod L).
    // Reduce the hop offset modulo the replica period first to keep float precision.
    const double off = std::fmod((double)(snap_start_hop % Mp) * (double)fft_len * cps, L);
    // Code-Doppler drift of the reference over the FULL absolute snapshot start (NOT
    // Mp-periodic, so use the full snap_start_hop): makes cp0 the true sample-0 phase, matching
    // the feed-forward in ChannelizedReplicaBank, so a stale seed no longer drifts the cp.
    const double drift = std::fmod((double)snap_start_hop * (double)fft_len * cps
                                       * (bank.code_doppler_sign * dop / bank.carrier_hz()),
                                   L);
    out.cp0 = std::fmod(best_cp - off - drift, L);
    if (out.cp0 < 0.0)
        out.cp0 += L;

    if (n_nh <= 1 || best_nh < 0)
        return out;

    // LONG-CODE PHASE, mod n_nh*L -- the phase a tracker despreading the overlaid code actually
    // needs. Everything above is reduced mod L, which is right for the primary and DESTROYS the
    // overlay period: a phase known only mod L leaves the tracker a 1-in-n_nh guess, and no
    // constant correction can supply the missing period because it is not constant. Two
    // separate reductions have to change, not one:
    //   * best_cp lifts into the long space by the alignment the acquire MEASURED;
    //   * the `% Mp` shortcut is valid ONLY mod L. One replica period is Mp*fft_len*cps = 16 L
    //     exactly at CHORD -- 0 mod L, but NOT 0 mod 20 L -- so the long form needs the FULL
    //     absolute advance. That product reaches ~7e12 chips and must stay good to a chip,
    //     hence long double (double's ulp there is ~1e-3 chips, fine, but the margin is free).
    const long double LL = (long double)L * (long double)n_nh;
    const long double adv =
        (long double)snap_start_hop * (long double)fft_len * (long double)cps;
    const long double off_l = adv - std::floor(adv / LL) * LL;
    const double drift_l = std::fmod((double)snap_start_hop * (double)fft_len * cps
                                         * (bank.code_doppler_sign * dop / bank.carrier_hz()),
                                     (double)LL);
    double cpl = std::fmod(best_cp + (double)best_nh * L - (double)off_l - drift_l, (double)LL);
    if (cpl < 0.0)
        cpl += (double)LL;
    out.cp_long = cpl;

    // ...and the same phase referenced to the SNAPSHOT, which is what should actually be
    // transported. best_cp is the argument of a replica at `anchor` that matches the data at
    // the snapshot, so the data's phase there is just best_cp lifted plus the ANCHOR's own
    // advance -- a fixed ~163732 chips whose Doppler sensitivity is 1e-4 chips/Hz, against
    // 5903 chips/Hz for the sample-0 route above. Same information, seven orders better
    // conditioned. (Done by hand rather than through bank.phase_from_arg() because THIS bank
    // reduces mod L and would throw the overlay period away.)
    const long double n_anc = (long double)anchor + (long double)fft_len - 1.0L;
    const long double cps_d =
        (long double)cps * (1.0L + (long double)(bank.code_doppler_sign * dop / bank.carrier_hz()));
    const long double a_adv = n_anc * cps_d;
    double phr = std::fmod(best_cp + (double)best_nh * L
                               + (double)(a_adv - std::floor(a_adv / LL) * LL),
                           (double)LL);
    if (phr < 0.0)
        phr += (double)LL;
    out.cp_at_ref = phr;
    return out;
}

SeedPropagation propagate_seed(const ChannelizedReplicaBank& bank, const SeedState& sd,
                               long long hop0, double sample_rate, double f_offset_hz,
                               double trim_chips) {
    SeedPropagation out;
    const int fft_len = bank.fft_len();
    const double dh = (double)(hop0 - sd.ref_hop);
    const double dt = dh * (double)fft_len / sample_rate;
    out.doppler_hz = sd.doppler_hz + sd.dop_rate * dt;

    // The phase advances at the TRUE code rate: nominal scaled by the code Doppler. Dropping
    // the scaling loses chip_rate*dop/(carrier*hops_per_sec) chips per hop -- 1.05e-4 at dop
    // 2350, so 41 chips on a seed only 2 s old and thousands on a minute-old one. It is
    // invisible if the seed's `code_phase_rate` happens to carry that same term (which is what
    // hand seeding did, and why hand seeds locked), but the broker's convention -- airspy's,
    // and the right one -- is that code_phase_rate is a RESIDUAL and the geometry is fed
    // forward here, exactly as the replica generator feeds it forward through cps(dop).
    const double chips_per_hop =
        bank.chip_rate_hz() * (double)fft_len / sample_rate
        * (1.0 + bank.code_doppler_sign * sd.doppler_hz / bank.carrier_hz());
    const double quad = 0.5 * (bank.chip_rate_hz() / f_offset_hz) * sd.dop_rate * dt * dt;

    out.phase_ref = (sd.phase_ref_chips >= 0.0)
                        ? sd.phase_ref_chips // transported as a phase: no back-reference at all
                        : bank.phase_from_arg(sd.cp_chips, sd.ref_hop * (long long)fft_len,
                                              sd.doppler_hz);
    out.phase_now = out.phase_ref + (chips_per_hop + sd.cp_rate) * dh + quad + trim_chips;
    out.cp = bank.arg_from_phase(out.phase_now, hop0 * (long long)fft_len, out.doppler_hz);
    return out;
}

} // namespace gnss
