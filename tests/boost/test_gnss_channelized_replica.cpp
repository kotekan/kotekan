#define BOOST_TEST_MODULE "test_gnss_channelized_replica"

#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank
#include "gnssSignal.hpp"              // for signal_by_name
#include "pfbPrototype.hpp"            // for Window

#include <boost/test/included/unit_test.hpp>
#include <complex>
#include <vector>

using cf = std::complex<float>;
using gnss::ChannelizedReplicaBank;

namespace {

// L5/L1 both have a 1 ms primary period, so at this Fs both give a clean replica
// period; 2 samples/chip on L5, 20 on L1.
constexpr double FS = 20.46e6;
constexpr double FOFF = 5.115e6; // carrier a few channels up from DC
constexpr int N = 10;            // 1.023 MHz channels
constexpr int P = 4;

// Matched-filter amplitude of replica `data` against replica `repl`, over all
// channels (the bank's replica energy normalizes it, so self -> 1).
double despread_mag(const std::vector<std::vector<cf>>& data,
                    const std::vector<std::vector<cf>>& repl) {
    return std::abs(gnss::channelized_despread(data, repl).amplitude);
}

ChannelizedReplicaBank make_bank(const char* signal, const std::vector<int>& prns) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    BOOST_REQUIRE_MESSAGE(sig != nullptr, "unknown signal");
    return ChannelizedReplicaBank(*sig, FS, FOFF, N, P, dsp::Window::Hamming, prns);
}

} // namespace

// The bank dispatches the L5 Q5 code: self-despread is a perfect matched filter (1),
// and a different PRN's replica decorrelates -- so it really uses PRN-distinct L5
// codes (a broken dispatch would give all-PRN-identical or empty codes -> no rejection).
BOOST_AUTO_TEST_CASE(l5q_self_and_cross_prn) {
    std::vector<int> prns = {1, 2};
    auto bank = make_bank("GPS_L5_Q", prns);
    const int H = bank.repl_period_hops();
    auto rA = bank.channels(0, 0, 0.0, 0.0, H); // PRN 1
    auto rB = bank.channels(1, 0, 0.0, 0.0, H); // PRN 2

    BOOST_CHECK_CLOSE(despread_mag(rA, rA), 1.0, 1e-2); // self == matched filter
    BOOST_CHECK_LT(despread_mag(rA, rB), 0.3);          // different PRN decorrelates
}

// I5 and Q5 are distinct codes on the same carrier -> they decorrelate.
BOOST_AUTO_TEST_CASE(l5i_distinct_from_l5q) {
    std::vector<int> prns = {1};
    auto bq = make_bank("GPS_L5_Q", prns);
    auto bi = make_bank("GPS_L5_I", prns);
    const int H = bq.repl_period_hops();
    auto rQ = bq.channels(0, 0, 0.0, 0.0, H);
    auto rI = bi.channels(0, 0, 0.0, 0.0, H);

    BOOST_CHECK_CLOSE(despread_mag(rQ, rQ), 1.0, 1e-2);
    BOOST_CHECK_LT(despread_mag(rQ, rI), 0.3); // I vs Q decorrelate
}

// The dispatch picks a genuinely different code per signal: an L1 C/A replica does
// not correlate with an L5 Q5 replica (same Fs/channelization, different code).
BOOST_AUTO_TEST_CASE(l1ca_distinct_from_l5q) {
    std::vector<int> prns = {1};
    auto bl1 = make_bank("GPS_L1CA", prns);
    auto bl5 = make_bank("GPS_L5_Q", prns);
    const int H = bl5.repl_period_hops();
    auto r1 = bl1.channels(0, 0, 0.0, 0.0, H);
    auto r5 = bl5.channels(0, 0, 0.0, 0.0, H);

    BOOST_CHECK_CLOSE(despread_mag(r1, r1), 1.0, 1e-2);
    BOOST_CHECK_LT(despread_mag(r1, r5), 0.3); // L1 code vs L5 code decorrelate
}
