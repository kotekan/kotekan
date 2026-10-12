#define BOOST_TEST_MODULE "test_gnss_chan_power"

#include "gnssChanPower.hpp" // for ChanPower, chan_norm_weights

#include <boost/test/included/unit_test.hpp>
#include <cmath> // for sqrt, abs
#include <vector>

using gnss::chan_norm_weights;
using gnss::ChanPower;

// Weights go as 1/sqrt(power) and average to one, so only the RELATIVE channel weights change.
BOOST_AUTO_TEST_CASE(weights_are_inverse_rms_with_mean_one) {
    const double pw[4] = {1.0, 4.0, 9.0, 16.0};
    double w[4] = {0, 0, 0, 0};
    BOOST_CHECK(chan_norm_weights(pw, 4, w));
    double mean = 0.0;
    for (double x : w)
        mean += x / 4.0;
    BOOST_CHECK_CLOSE(mean, 1.0, 1e-9);
    // w_i * sqrt(pw_i) is the same constant for every channel.
    const double k = w[0] * std::sqrt(pw[0]);
    for (int i = 1; i < 4; ++i)
        BOOST_CHECK_CLOSE(w[i] * std::sqrt(pw[i]), k, 1e-9);
    BOOST_CHECK(w[0] > w[1] && w[1] > w[2] && w[2] > w[3]);
}

// A flat power spectrum is the identity: no channel is re-weighted.
BOOST_AUTO_TEST_CASE(flat_power_is_unit_weights) {
    const double pw[3] = {2.5, 2.5, 2.5};
    double w[3] = {0, 0, 0};
    BOOST_CHECK(chan_norm_weights(pw, 3, w));
    for (double x : w)
        BOOST_CHECK_CLOSE(x, 1.0, 1e-9);
}

// A missing or dead channel refuses the whole set and leaves the caller's weights alone, so
// normalised and raw channels are never mixed in one sum.
BOOST_AUTO_TEST_CASE(missing_power_refuses_and_leaves_weights) {
    const double pw[3] = {1.0, 0.0, 4.0};
    double w[3] = {7.0, 7.0, 7.0};
    BOOST_CHECK(!chan_norm_weights(pw, 3, w));
    for (double x : w)
        BOOST_CHECK_EQUAL(x, 7.0);
    const double neg[2] = {1.0, -1.0};
    BOOST_CHECK(!chan_norm_weights(neg, 2, w));
    BOOST_CHECK(!chan_norm_weights(pw, 0, w));
}

// The table is per element: a reader averages over ITS element range, skipping elements with
// no power (dead slots), so inputs outside the beam cannot set the weights. First value taken
// as is, later ones folded in by the EMA; unknown ids read 0.
BOOST_AUTO_TEST_CASE(table_publish_lookup_and_ema) {
    ChanPower& t = ChanPower::instance();
    // Two channels, four elements each: elements 0-1 are the beam, 2 is dead, 3 is an RFI
    // antenna with a flat, higher power.
    t.publish({6532, 6548}, {1.0, 3.0, 0.0, 9.0, /* 6548: */ 4.0, 4.0, 0.0, 9.0}, 4);
    int ids[3] = {6532, 6548, 6564};
    double pw[3] = {-1, -1, -1};
    const int beam[2] = {0, 1}, all[4] = {0, 1, 2, 3}, dead[1] = {2}, sparse[3] = {1, 3, 99};
    BOOST_CHECK_EQUAL(t.lookup(ids, 3, beam, 2, pw), 2);
    BOOST_CHECK_CLOSE(pw[0], 2.0, 1e-9); // mean of 1 and 3
    BOOST_CHECK_CLOSE(pw[1], 4.0, 1e-9);
    BOOST_CHECK_EQUAL(pw[2], 0.0);
    // The whole frame: the dead element is skipped, the RFI antenna is not.
    BOOST_CHECK_EQUAL(t.lookup(ids, 1, all, 4, pw), 1);
    BOOST_CHECK_CLOSE(pw[0], 13.0 / 3.0, 1e-9);
    // A list with no powered element reads 0 (the caller then keeps unit weights); an element
    // beyond the frame is ignored, not an error.
    BOOST_CHECK_EQUAL(t.lookup(ids, 1, dead, 1, pw), 1);
    BOOST_CHECK_EQUAL(pw[0], 0.0);
    BOOST_CHECK_EQUAL(t.lookup(ids, 1, sparse, 3, pw), 1);
    BOOST_CHECK_CLOSE(pw[0], 6.0, 1e-9); // mean of 3 and 9
    // 6532's element 0 steps 1 -> 9: EMA alpha 0.5 lands halfway, element 1 unchanged.
    t.publish({6532}, {9.0, 3.0, 0.0, 9.0}, 4, 0.5);
    BOOST_CHECK_EQUAL(t.lookup(ids, 1, beam, 2, pw), 1);
    BOOST_CHECK_CLOSE(pw[0], 4.0, 1e-9); // mean of 5 and 3
    // A short power vector publishes only the channels it covers.
    t.publish({6564, 6580}, {1.0, 1.0, 1.0, 1.0}, 4);
    BOOST_CHECK_EQUAL(t.lookup(ids, 3, beam, 2, pw), 3);
    BOOST_CHECK_CLOSE(pw[2], 1.0, 1e-9);
    double w[3];
    BOOST_CHECK(chan_norm_weights(pw, 3, w));
    BOOST_CHECK_CLOSE(w[0], w[1], 1e-9); // equal power (4 and 4), equal weight
    BOOST_CHECK(w[2] > w[0]);            // the quietest channel gets the largest weight
}
