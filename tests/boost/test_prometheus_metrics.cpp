#define BOOST_TEST_MODULE "test_updateQueue"

#include <boost/test/included/unit_test.hpp>

#include "prometheusMetrics.hpp"
using kotekan::prometheusMetrics;


// BOOST_AUTO_TEST_CASE(example1) {
//     const int i = 1;
//     BOOST_CHECK(i);
//     BOOST_CHECK(i <= 2);
// }

BOOST_AUTO_TEST_CASE(simple_metrics) {
    prometheusMetrics& metrics = prometheusMetrics::instance();
    BOOST_CHECK(metrics.serialize() == "");

    metrics.add_stage_metric("foo_metric", "foo", 1);
    BOOST_CHECK(metrics.serialize().find("foo_metric{stage_name=\"foo\"} 1 ") != std::string::npos);

    metrics.add_stage_metric("foo_metric", "foo", 2); // update existing value
    metrics.add_stage_metric("foo_metric", "foos", 10); // a new stage of the same metric
    metrics.add_stage_metric("bar_metric", "foos", 100); // a metric for the same stage

    auto multi_metrics = metrics.serialize();
    // new value
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 2 ") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 1 ") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foos\"} 10 ") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_metric{stage_name=\"foos\"} 100 ") != std::string::npos);
}


BOOST_AUTO_TEST_CASE(metrics_with_labels) {
    prometheusMetrics& metrics = prometheusMetrics::instance();

    metrics.add_stage_metric("foo_metric", "foo", 1, "quux=\"fred\"");
    BOOST_CHECK(metrics.serialize().find("foo_metric{stage_name=\"foo\",quux=\"fred\"} 1 ") != std::string::npos);

    metrics.add_stage_metric("foo_metric", "foo", 2, "quux=\"fred\""); // update existing metric value
    metrics.add_stage_metric("foo_metric", "foo", 10, "quux=\"baz\""); // a different label value of the same metric
    metrics.add_stage_metric("bar_metric", "foo", 42, "quux=\"baz\""); // a different metric with the same label combination

    auto multi_metrics = metrics.serialize();
    // new value
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\",quux=\"fred\"} 2 ") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\",quux=\"fred\"} 1 ") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\",quux=\"baz\"} 10 ") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_metric{stage_name=\"foo\",quux=\"baz\"} 42 ") != std::string::npos);
}
