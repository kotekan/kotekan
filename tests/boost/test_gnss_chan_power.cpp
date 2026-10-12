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

// The table: first value taken as is, later ones folded in by the EMA; unknown ids read 0.
BOOST_AUTO_TEST_CASE(table_publish_lookup_and_ema) {
    ChanPower& t = ChanPower::instance();
    t.publish({6532, 6548}, {4.0, 16.0});
    int ids[3] = {6532, 6548, 6564};
    double pw[3] = {-1, -1, -1};
    BOOST_CHECK_EQUAL(t.lookup(ids, 3, pw), 2);
    BOOST_CHECK_CLOSE(pw[0], 4.0, 1e-9);
    BOOST_CHECK_CLOSE(pw[1], 16.0, 1e-9);
    BOOST_CHECK_EQUAL(pw[2], 0.0);
    // 6532 steps to 9: EMA alpha 0.5 lands halfway.
    t.publish({6532}, {9.0}, 0.5);
    BOOST_CHECK_EQUAL(t.lookup(ids, 1, pw), 1);
    BOOST_CHECK_CLOSE(pw[0], 6.5, 1e-9);
    // Mismatched lengths publish only the pairs both vectors have.
    t.publish({6564, 6580}, {1.0});
    BOOST_CHECK_EQUAL(t.lookup(ids, 3, pw), 3);
    BOOST_CHECK_CLOSE(pw[2], 1.0, 1e-9);
    double w[3];
    BOOST_CHECK(chan_norm_weights(pw, 3, w));
}
