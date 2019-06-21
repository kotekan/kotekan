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
    std::cout << metrics.serialize() << "\n";
    BOOST_CHECK(metrics.serialize().find("foo_metric{stage_name=\"foo\"} 1 ") != std::string::npos);

    metrics.add_stage_metric("foo_metric", "foo", 2); // update existing value
    metrics.add_stage_metric("foo_metric", "foos", 10); // a new stage of the same metric
    metrics.add_stage_metric("bar_metric", "foos", 100); // a metric for the same stage

    auto multi_metrics = metrics.serialize();
    std::cout << multi_metrics << "\n";

    // new value
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 2 ") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 1 ") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foos\"} 10 ") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_metric{stage_name=\"foos\"} 100 ") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(new_metric_api) {
    prometheusMetrics& metrics = prometheusMetrics::instance();

    auto* m = metrics.add_stage_metric("foo_metric", "foo", 41);
    std::cout << metrics.serialize() << "\n";
    BOOST_CHECK(metrics.serialize().find("foo_metric{stage_name=\"foo\"} 41 ") != std::string::npos);

    m->set(2);
    metrics.add_stage_metric("foo_metric", "foos", 10); // a new stage of the same metric
    metrics.add_stage_metric("bar_metric", "foos", 100); // a metric for the same stage

    auto multi_metrics = metrics.serialize();
    std::cout << multi_metrics << "\n";

    // new value
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 2 ") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 41 ") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foos\"} 10 ") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_metric{stage_name=\"foos\"} 100 ") != std::string::npos);
}


BOOST_AUTO_TEST_CASE(metrics_with_labels) {
    prometheusMetrics& metrics = prometheusMetrics::instance();

    auto* m1 = metrics.add_stage_metric("foo_metric", "foo", 1, "quux=\"fred\"");
    BOOST_CHECK(metrics.serialize().find("foo_metric{stage_name=\"foo\",quux=\"fred\"} 1 ") != std::string::npos);

    m1->set(2);

    auto* m2 = metrics.add_stage_metric("foo_metric", "foo", 0, "quux=\"baz\""); // a different label value of the same metric
    m2->set(10);

    metrics.add_stage_metric("foo_metric", "foo", 10, "quux=\"baz\""); // a different label value of the same metric
    metrics.add_stage_metric("bar_metric", "foo", 42, "quux=\"baz\""); // a different metric with the same label combination

    auto multi_metrics = metrics.serialize();
    std::cout << multi_metrics << "\n";

    // new value
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\",quux=\"fred\"} 2 ") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\",quux=\"fred\"} 1 ") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\",quux=\"baz\"} 10 ") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_metric{stage_name=\"foo\",quux=\"baz\"} 42 ") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(counters) {
    prometheusMetrics& metrics = prometheusMetrics::instance();

    auto* c = metrics.add_stage_counter("foo_counter", "foo");
    auto ms = metrics.serialize();
    std::cout << ms << "\n";
    BOOST_CHECK(ms.find("foo_counter{stage_name=\"foo\"} 0 ") != std::string::npos);

    c->inc();
    // metrics.add_stage_counter("foo_counter", "foos")->inc(1.0/10); // a new stage of the same metric
    // metrics.add_stage_counter("bar_counter", "foos")->inc(1/3.0); // a metric for the same stage

    ms = metrics.serialize();
    // new value
    BOOST_CHECK(ms.find("foo_counter{stage_name=\"foo\"} 1 ") != std::string::npos);
    // old value is not present
    BOOST_CHECK(ms.find("foo_counter{stage_name=\"foo\"} 0 ") == std::string::npos);
    // // new time series
    // BOOST_CHECK(ms.find("foo_counter{stage_name=\"foos\"} 0.1 ") != std::string::npos);
    // BOOST_CHECK(ms.find("bar_counter{stage_name=\"foos\"} 0.333333 ") != std::string::npos);

    // c->inc(3);
    // ms = metrics.serialize();
    // BOOST_CHECK(ms.find("foo_counter{stage_name=\"foo\"} 4 ") != std::string::npos);

    BOOST_CHECK_THROW(metrics.add_stage_counter("foo_counter", "foo"), std::runtime_error);
}

