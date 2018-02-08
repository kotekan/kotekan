#include "prometheusMetrics.hpp"
#include "errors.h"

prometheusMetrics::prometheusMetrics() {
}

prometheusMetrics::~prometheusMetrics() {
    for (auto &process_metric : process_metrics)
        delete process_metric;
}

prometheusMetrics::metric::~metric() {
}

template<class T>
string prometheusMetrics::processMetric<T>::to_string() {
    return to_string(value);
}

template<class T>
void prometheusMetrics::add_process_metric(const string& name,
                                           const string& process_name,
                                           const T& value,
                                           const string& tags = "") {

    std::tuple<string, string, string> key = {name, process_name, tags};

    if (process_metrics.count(key) == 0) {
        processMetric<T> * new_metric = new processMetric<T>;
        process_metrics[key] = (prometheusMetrics::metric *)new_metric;
    }

    ((processMetric<T> *)process_metrics[key])->value = value;
    ((processMetric<T> *)process_metrics[key])->value = get_time_in_milliseconds();
}

void prometheusMetrics::remove_metric(const string& name,
                                      const string& process_name,
                                      const string& tags) {
    std::tuple<string, string, string> key = {name, process_name, tags};

    if (process_metrics.count(key) == 1) {
        delete process_metrics[key];
        process_metrics.erase(key);
    } else {
        WARN("Tried to remove metric (%s, %s, %s), which does not exist",
                name.c_str(), process_name.c_str(), tags.c_str());
    }
}

void prometheusMetrics::metrics_callback(connectionInstance& conn) {

}

void prometheusMetrics::register_with_server(restServer* rest_server) {

}

uint64_t prometheusMetrics::get_time_in_milliseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
}
