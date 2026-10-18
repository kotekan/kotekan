// Checks that a live PRN slot swap reaches the code table the GPU correlates against.
//
//   scripts/gnss/site/build_tool.sh prnswap && scripts/gnss/prnswap
//
// docs/CHORD_LIVE_PRN_RECONFIG.md lets the broker repoint a PRN slot at a different satellite
// while the pipeline runs. A swap that updates the host's map but not the device's code table
// raises no error: the slot keeps correlating the old satellite's code and never locks again,
// and every host-side check agrees with itself. So the test synthesizes the new satellite's
// signal, despreads it in the swapped slot, and requires the correlation to appear there and
// the old satellite's to vanish.
//
// It also covers two engines sharing one bank, as the search does (one GnssCudaDespread per
// <=64-channel refine group over a single ChannelizedReplicaBank). The bank is shared but the
// device tables are not, so an engine that compared against the bank's PRN would skip its own
// upload.
//
// Run it on cf06, not on a tracking node: it allocates a despread engine and launches kernels
// on a GPU the live chains are using.
//
// @author Keith Vanderlinde
#include "GnssCudaDespread.hpp"
#include "dspPfbPrototype.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using cf = std::complex<float>;

static int g_fail = 0;

static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++g_fail;
}

int main() {
    // L1 C/A at a small geometry (5 MHz, 20 channels): fast, and the code tables differ
    // strongly between PRNs, which is the property under test.
    const double fs = 5.0e6, f_off = 1.25e6;
    const int N = 20, taps = 4;
    const double dop = -2900.0, cp_true = 517.25;
    const int PRN_A = 19, PRN_B = 7, PRN_C = 11;

    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L1CA");
    if (!sig) {
        fprintf(stderr, "no GPS_L1CA descriptor\n");
        return 2;
    }
    const auto win = dsp::window_from_string("hamming");

    // ---- 1. The bank: a swapped slot is byte-identical to a slot built that way ----
    // The constructor and set_prn share one code expansion (build_full_code); if they diverged,
    // the only symptom would be one satellite that cannot be tracked.
    gnss::ChannelizedReplicaBank bank(*sig, fs, f_off, N, taps, win, {PRN_A, PRN_B});
    gnss::ChannelizedReplicaBank ref(*sig, fs, f_off, N, taps, win, {PRN_C, PRN_B});
    check(bank.prns() == std::vector<int>({PRN_A, PRN_B}), "the bank remembers its PRN list");
    check(bank.prn_at(1) == PRN_B && bank.prn_at(9) == -1, "prn_at reads slots, -1 out of range");

    const auto code_a_before = bank.full_code(0);
    check(!bank.set_prn(0, 999), "set_prn REFUSES a PRN with no code for this signal");
    check(bank.full_code(0) == code_a_before && bank.prn_at(0) == PRN_A,
          "... and a refused swap changes NOTHING (no half-applied slot)");
    check(!bank.set_prn(7, PRN_C), "set_prn refuses an out-of-range slot");

    check(bank.set_prn(0, PRN_C), "set_prn accepts a valid PRN");
    check(bank.full_code(0) == ref.full_code(0),
          "the swapped slot's code table is byte-identical to one built that way");
    check(bank.full_code(1) == ref.full_code(1), "... and the untouched slot is untouched");
    check(bank.prns() == std::vector<int>({PRN_C, PRN_B}), "the bank's map follows the swap");
    check(bank.set_prn(0, PRN_C), "a no-op swap succeeds and costs nothing");

    // ---- 2. The device: the swap reaches the bytes the kernel reads ----
    // Fresh bank, so the engines below start from the same place the pipeline does.
    gnss::ChannelizedReplicaBank live(*sig, fs, f_off, N, taps, win, {PRN_A, PRN_B});
    const int n_hops = live.repl_period_hops();
    const auto cover = live.covering_bins(dop, 250000.0);
    const int n_chan = (int)cover.size();
    std::vector<int> local(n_chan);
    for (int c = 0; c < n_chan; ++c)
        local[c] = c;
    const long long w0 = 114436200145LL * (long long)(2 * N); // CHORD-scale absolute anchor

    // "Sky" for a slot of a bank: the replica itself, without noise. This tests which code is
    // loaded, not sensitivity, so a clean signal makes the answer unambiguous.
    auto sky_of = [&](gnss::ChannelizedReplicaBank& b, int slot) {
        const auto ch = b.channels_hoprate(slot, w0, cp_true, dop, n_hops, cover);
        std::vector<cf> w((size_t)n_hops * n_chan);
        for (int m = 0; m < n_hops; ++m)
            for (int c = 0; c < n_chan; ++c)
                w[(size_t)m * n_chan + c] = ch[c][m];
        return w;
    };
    // A one-PRN bank per satellite, purely to synthesize that satellite's sky.
    gnss::ChannelizedReplicaBank gen_a(*sig, fs, f_off, N, taps, win, {PRN_A});
    gnss::ChannelizedReplicaBank gen_c(*sig, fs, f_off, N, taps, win, {PRN_C});
    const auto sky_a = sky_of(gen_a, 0);
    const auto sky_c = sky_of(gen_c, 0);

    GnssCudaDespread eng1(live, 2, cover, n_hops, fs);
    GnssCudaDespread eng2(live, 2, cover, n_hops, fs); // same bank, own device table

    // Normalised prompt amplitude for slot 0 against a given sky: |corr| / replica energy. ~1
    // when the loaded code matches the sky, ~0 when it does not (C/A cross-correlation is at
    // most ~-24 dB).
    auto amp = [&](GnssCudaDespread& e, const std::vector<cf>& sky) {
        e.upload_window(sky.data(), w0);
        GnssCudaDespread::Spec sp;
        sp.p = 0;
        sp.cp_seed = cp_true;
        sp.spacing_chips = 0.5;
        sp.doppler_hz = dop;
        sp.covering = local;
        const auto out = e.despread_batch({sp});
        if (out.empty())
            return 0.0;
        const auto& P = out[0][1]; // prompt
        return (P.replica_energy > 0.0) ? std::abs(P.correlation) / P.replica_energy : 0.0;
    };

    const double a_before = amp(eng1, sky_a);
    const double c_before = amp(eng1, sky_c);
    check(a_before > 0.5, "before the swap, slot 0 correlates PRN A's sky");
    check(c_before < 0.1 * a_before, "... and does NOT correlate PRN C's");

    check(eng1.set_prn(0, PRN_C, nullptr), "engine 1 accepts the swap");
    const double c_after1 = amp(eng1, sky_c);
    const double a_after1 = amp(eng1, sky_a);
    check(c_after1 > 0.5,
          "AFTER the swap slot 0 correlates PRN C's sky -- the DEVICE code table moved");
    check(a_after1 < 0.1 * c_after1, "... and PRN A's sky no longer correlates there");

    // ---- 3. Two engines, one bank ----
    // The bank says slot 0 holds PRN C because engine 1 changed it, while engine 2's device
    // table still holds PRN A. An engine that compared against the bank would do nothing and
    // leave engine 2 correlating the wrong code, with the host map reporting the swap done.
    const double c_eng2_stale = amp(eng2, sky_c);
    check(c_eng2_stale < 0.1, "engine 2 has NOT silently followed the shared bank");
    check(eng2.set_prn(0, PRN_C, nullptr), "engine 2 accepts the same swap");
    const double c_eng2 = amp(eng2, sky_c);
    check(c_eng2 > 0.5,
          "... and its OWN device table updates (the shared-bank short-circuit trap)");

    printf("\n  amplitudes: A/before %.3f  C/before %.3f | C/after %.3f  A/after %.3f "
           "| eng2 stale %.3f -> %.3f\n",
           a_before, c_before, c_after1, a_after1, c_eng2_stale, c_eng2);
    printf("\n%s (%d check(s) failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
