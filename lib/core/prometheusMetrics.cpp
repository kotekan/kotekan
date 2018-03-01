#include "prometheusMetrics.hpp"
#include "errors.h"
#include "metadata.h"

prometheusMetrics::prometheusMetrics() {
    char local_host_name[128];
    gethostname(local_host_name, sizeof(local_host_name));
    hostname = string(local_host_name);
}

prometheusMetrics &prometheusMetrics::instance() {
    static prometheusMetrics _instance;
    return _instance;
}


prometheusMetrics::~prometheusMetrics() {
    for (auto &process_metric : process_metrics)
        delete process_metric.second;
}

prometheusMetrics::metric::~metric() {
}

void prometheusMetrics::remove_metric(const string& name,
                                      const string& process_name,
                                      const string& labels) {
    std::lock_guard<std::mutex> lock(metrics_lock);
    std::tuple<string, string, string> key {name, process_name, labels};

    if (process_metrics.count(key) == 1) {
        delete process_metrics[key];
        process_metrics.erase(key);
    } else {
        WARN("Tried to remove metric (%s, %s, %s), which does not exist",
                name.c_str(), process_name.c_str(), labels.c_str());
    }
}

void prometheusMetrics::metrics_callback(connectionInstance& conn) {
    string output;

    {
        std::lock_guard<std::mutex> lock(metrics_lock);

        for (auto &element : process_metrics) {
            string metric_name = std::get<0>(element.first);
            string process_name = std::get<1>(element.first);
            string extra_labels = std::get<2>(element.first);

            output += metric_name + "{instance='" + hostname + "',process_name='" + process_name
                      + "'";
            if (extra_labels != "")
                output += "," + extra_labels;
            output += "} " + element.second->to_string() + " "
                      + std::to_string(element.second->last_update_time_stamp) + "\n";
        }
    }

    // Sending the reply doesn't need to be locked.
    // Just accessing the metrics array.
    conn.send_text_reply(output);
}

void prometheusMetrics::register_with_server(restServer* rest_server) {
    using namespace std::placeholders;
    rest_server->register_get_callback("/metrics",
            std::bind(&prometheusMetrics::metrics_callback, this, _1));
}

uint64_t prometheusMetrics::get_time_in_milliseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
}
