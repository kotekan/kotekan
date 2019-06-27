#define BOOST_TEST_MODULE "test_updateQueue"

#include <boost/test/included/unit_test.hpp>

#include "prometheusMetrics.hpp"
using kotekan::prometheusMetrics;


BOOST_AUTO_TEST_CASE(simple_metrics) {
    prometheusMetrics& metrics = prometheusMetrics::instance();
    BOOST_CHECK(metrics.serialize() == "");

    auto& foo = metrics.AddCounter("foo_metric", "foo");
    foo.inc();
    std::cout << metrics.Serialize();
    BOOST_CHECK(metrics.Serialize().find("# HELP foo_metric\n# TYPE foo_metric counter\nfoo_metric{stage_name=\"foo\"} 1") != std::string::npos);

    auto& foos = metrics.AddGauge("foo_metric", "foos"); // a new stage of the same metric
    foos.set(10);
    auto& bar = metrics.AddGauge("bar_metric", "foos"); // a metric for the same stage
    bar.set(100);

    foo.inc();
    auto multi_metrics = metrics.Serialize();

    // new value
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 2") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 1") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foos\"} 10") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_metric{stage_name=\"foos\"} 100") != std::string::npos);
}


BOOST_AUTO_TEST_CASE(counters_with_labels) {
    prometheusMetrics& metrics = prometheusMetrics::instance();

    auto& m1 = metrics.AddCounter("http_requests_total", "main", {"method", "handler"});
    m1.Labels({"POST", "/messages"}).inc();
    m1.Labels({"GET", "/messages"}).inc();
    m1.Labels({"GET", "/messages"}).inc();
    std::cout << m1.Serialize();
    BOOST_CHECK(m1.Serialize().find("# HELP http_requests_total\n# TYPE http_requests_total counter\nhttp_requests_total{stage_name=\"main\",method=\"POST\",handler=\"/messages\"} 1") != std::string::npos);
    BOOST_CHECK(m1.Serialize().find("http_requests_total{stage_name=\"main\",method=\"GET\",handler=\"/messages\"} 2"));

    auto& m2 = metrics.AddCounter("total_count", "sidecar");
    m2.inc();
    m2.inc();
    m2.inc();
    BOOST_CHECK(metrics.Serialize().find("# HELP total_count\n#TYPE total_count counter\ntotal_count{stage_name=\"sidecar\"} 3"));
}


BOOST_AUTO_TEST_CASE(gauges_with_labels) {
    prometheusMetrics& metrics = prometheusMetrics::instance();

    auto& m1 = metrics.AddGauge("foo_with_labels", "foo", {"quux"});
    m1.Labels({"fred"}).set(1);
    BOOST_CHECK(metrics.Serialize().find("foo_with_labels{stage_name=\"foo\",quux=\"fred\"} 1") != std::string::npos);

    m1.Labels({"fred"}).set(2);

    m1.Labels({"baz"}).set(10); // a different label value of the same metric

    auto& m2 = metrics.AddGauge("bar_with_labels", "foo", {"quux"}); // a different label value of the same metric
    m2.Labels({"baz"}).set(42);

    auto multi_metrics = metrics.Serialize();
    // std::cout << multi_metrics << "\n";

    // new value
    BOOST_CHECK(multi_metrics.find("foo_with_labels{stage_name=\"foo\",quux=\"fred\"} 2") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_with_labels{stage_name=\"foo\",quux=\"fred\"} 1") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_with_labels{stage_name=\"foo\",quux=\"baz\"} 10") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_with_labels{stage_name=\"foo\",quux=\"baz\"} 42") != std::string::npos);
}
