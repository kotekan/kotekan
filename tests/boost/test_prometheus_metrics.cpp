#define BOOST_TEST_MODULE "test_updateQueue"

#include "prometheusMetrics.hpp" // for Metrics, MetricFamily, Counter, Gauge

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_CHECK, BOOST_PP_BOOL_2
#include <cmath>                             // for sqrt, log
#include <iostream>                          // for cout, ostream
#include <string>                            // for string, allocator, basic_string, operator==

using kotekan::prometheus::Metrics;


BOOST_AUTO_TEST_CASE(simple_metrics) {
    Metrics& metrics = Metrics::instance();
    BOOST_CHECK(metrics.serialize() == "");

    auto foo = metrics.add_counter("foo_metric", "foo", {});
    BOOST_CHECK(
        metrics.serialize().find(
            "# HELP foo_metric\n# TYPE foo_metric counter\nfoo_metric{stage_name=\"foo\"} 0")
        != std::string::npos);

    foo->labels({}).inc();
    BOOST_CHECK(
        metrics.serialize().find(
            "# HELP foo_metric\n# TYPE foo_metric counter\nfoo_metric{stage_name=\"foo\"} 1")
        != std::string::npos);

    auto foos = metrics.add_gauge("foo_metric", "foos", {}); // a new stage of the same metric
    foos->labels({}).set(10);
    auto bar = metrics.add_gauge("bar_metric", "foos", {}); // a metric for the same stage
    bar->labels({}).set(100);

    auto baznan = metrics.add_gauge("baznan_metric", "foos", {}); // a metric with NaNs
    baznan->labels({}).set(sqrt(-1));
    auto bazinf = metrics.add_gauge("bazinf_metric", "foos", {}); // a metric with inf
    bazinf->labels({}).set(log(0));

    foo->labels({}).inc();
    auto multi_metrics = metrics.serialize();

    // new value
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 2") != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 1") == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foos\"} 10.0") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_metric{stage_name=\"foos\"} 100.0") != std::string::npos);

    // proper formatting of NaN
    BOOST_CHECK(multi_metrics.find("baznan_metric{stage_name=\"foos\"} NaN") != std::string::npos);

    // proper formatting of inf
    BOOST_CHECK(multi_metrics.find("bazinf_metric{stage_name=\"foos\"} -Inf") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(remove_stage_metrics) {
    Metrics& metrics = Metrics::instance();
    metrics.remove_stage_metrics("foo");
    metrics.remove_stage_metrics("foos");
    BOOST_CHECK(metrics.serialize() == "");

    metrics.add_counter("foo_metric", "foo", {});
    metrics.add_counter("foo_metric", "foos", {});
    auto multi_metrics = metrics.serialize();
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 0") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foos\"} 0") != std::string::npos);

    // remove the metrics we just added
    metrics.remove_stage_metrics("foo");
    // all metrics from this stage will be missing
    BOOST_CHECK(metrics.serialize().find("stage_name=\"foo\"") == std::string::npos);
    // while other stages are not affected
    BOOST_CHECK(metrics.serialize().find("foo_metric{stage_name=\"foos\"} 0") != std::string::npos);

    // now remove the rest
    metrics.remove_stage_metrics("foos");
    BOOST_CHECK(metrics.serialize() == "");

    // removing already-deleted stages is OK
    metrics.remove_stage_metrics("foo");

    // now remove the other stage
    metrics.remove_stage_metrics("foos");
    BOOST_CHECK(metrics.serialize() == "");

    // re-adding metrics from stages that were deleted once is also OK
    metrics.add_counter("foo_metric", "foo", {});
    metrics.add_counter("foo_metric", "foos", {});
    multi_metrics = metrics.serialize();
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foo\"} 0") != std::string::npos);
    BOOST_CHECK(multi_metrics.find("foo_metric{stage_name=\"foos\"} 0") != std::string::npos);
}


BOOST_AUTO_TEST_CASE(counters_with_labels) {
    Metrics& metrics = Metrics::instance();

    auto m1 = metrics.add_counter("http_requests_total", "main", {"method", "handler"});
    m1->labels({"POST", "/messages"}).inc();
    m1->labels({"GET", "/messages"}).inc();
    m1->labels({"GET", "/messages"}).inc();
    std::cout << m1.serialize();
    BOOST_CHECK(m1.serialize().find("# HELP http_requests_total\n# TYPE http_requests_total "
                                    "counter\nhttp_requests_total{stage_name=\"main\",method="
                                    "\"POST\",handler=\"/messages\"} 1")
                != std::string::npos);
    BOOST_CHECK(m1.serialize().find(
        "http_requests_total{stage_name=\"main\",method=\"GET\",handler=\"/messages\"} 2"));

    auto m2 = metrics.add_counter("total_count", "sidecar", {});
    m2->labels({}).inc();
    m2->labels({}).inc();
    m2->labels({}).inc();
    BOOST_CHECK(metrics.serialize().find(
        "# HELP total_count\n#TYPE total_count counter\ntotal_count{stage_name=\"sidecar\"} 3"));
}


BOOST_AUTO_TEST_CASE(gauges_with_labels) {
    Metrics& metrics = Metrics::instance();

    auto m1 = metrics.add_gauge("foo_with_labels", "foo", {"quux"});
    m1->labels({"fred"}).set(1);
    BOOST_CHECK(metrics.serialize().find("foo_with_labels{stage_name=\"foo\",quux=\"fred\"} 1.0")
                != std::string::npos);

    m1->labels({"fred"}).set(2);

    m1->labels({"baz"}).set(10); // a different label value of the same metric

    auto m2 = metrics.add_gauge("bar_with_labels", "foo",
                                {"quux"}); // a different label value of the same metric
    m2->labels({"baz"}).set(42);

    auto multi_metrics = metrics.serialize();
    // std::cout << multi_metrics << "\n";

    // new value
    BOOST_CHECK(multi_metrics.find("foo_with_labels{stage_name=\"foo\",quux=\"fred\"} 2.0")
                != std::string::npos);
    // old value is not present
    BOOST_CHECK(multi_metrics.find("foo_with_labels{stage_name=\"foo\",quux=\"fred\"} 1")
                == std::string::npos);

    // new time series
    BOOST_CHECK(multi_metrics.find("foo_with_labels{stage_name=\"foo\",quux=\"baz\"} 10.0")
                != std::string::npos);
    BOOST_CHECK(multi_metrics.find("bar_with_labels{stage_name=\"foo\",quux=\"baz\"} 42.0")
                != std::string::npos);
}
