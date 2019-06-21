#include "prometheusMetrics.hpp"

#include "errors.h"
#include "metadata.h"

namespace kotekan {

prometheusMetrics::prometheusMetrics() {}

prometheusMetrics& prometheusMetrics::instance() {
    static prometheusMetrics _instance;
    return _instance;
}


prometheusMetrics::~prometheusMetrics() {
    restServer::instance().remove_get_callback("/metrics");
}

Gauge* prometheusMetrics::add_stage_metric(const string& name, const string& stage_name, const double value,
                                            const string& labels) {
    std::lock_guard<std::mutex> lock(metrics_lock);
    std::tuple<string, string, string> key{name, stage_name, labels};

    if (stage_metrics.count(key) == 0) {
        stage_metrics[key] = std::unique_ptr<metric>(new Gauge);;
    }

    Gauge* gauge = ((Gauge*)stage_metrics[key].get());
    gauge->set(value);
    return gauge;
}

Counter* prometheusMetrics::add_stage_counter(const string& name, const string& stage_name,
                                              const string& labels) {
    std::lock_guard<std::mutex> lock(metrics_lock);
    std::tuple<string, string, string> key{name, stage_name, labels};

    if (stage_metrics.count(key) == 0) {
        stage_metrics[key] = std::unique_ptr<metric>(new Counter);;
    }
    else {
        throw std::runtime_error("Cannot define an already existing metric: " + name + "/" + stage_name + "/" + labels);
    }

    Counter* counter = ((Counter*)stage_metrics[key].get());
    return counter;
}

void prometheusMetrics::remove_metric(const string& name, const string& stage_name,
                                      const string& labels) {
    std::lock_guard<std::mutex> lock(metrics_lock);
    std::tuple<string, string, string> key{name, stage_name, labels};

    if (stage_metrics.count(key) == 1) {
        stage_metrics.erase(key);
    } else {
        WARN("Tried to remove metric (%s, %s, %s), which does not exist", name.c_str(),
             stage_name.c_str(), labels.c_str());
    }
}


string prometheusMetrics::serialize() {
    std::ostringstream output;

    std::lock_guard<std::mutex> lock(metrics_lock);

    for (auto& element : stage_metrics) {
        string metric_name = std::get<0>(element.first);
        string stage_name = std::get<1>(element.first);
        string extra_labels = std::get<2>(element.first);

        output << metric_name << "{stage_name=\"" << stage_name << "\"";
        if (extra_labels != "")
            output << "," << extra_labels;
        output << "} " << element.second->to_string() << "\n";
    }

    return output.str();
}


void prometheusMetrics::metrics_callback(connectionInstance& conn) {
    string output = serialize();

    // Sending the reply doesn't need to be locked.
    // Just accessing the metrics array.
    conn.send_text_reply(output);
}

void prometheusMetrics::register_with_server(restServer* rest_server) {
    using namespace std::placeholders;
    rest_server->register_get_callback("/metrics",
                                       std::bind(&prometheusMetrics::metrics_callback, this, _1));
}

void Gauge::set(const double value) {
    this->value = value;
    this->last_update_time_stamp = get_time_in_milliseconds();
}

string Gauge::to_string() {
    std::ostringstream output;
    output << value << " " << std::to_string(last_update_time_stamp);
    return output.str();
}

void Counter::inc() {
    this->value += 1;
}

string Counter::to_string() {
    std::ostringstream output;
    output << value << " " << get_time_in_milliseconds();
    return output.str();
}

/* static */
uint64_t metric::get_time_in_milliseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
}

} // namespace kotekan
