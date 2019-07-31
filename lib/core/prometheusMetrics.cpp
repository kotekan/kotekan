#include "prometheusMetrics.hpp"

#include "errors.h"
#include "metadata.h"

#include "fmt.hpp"

namespace kotekan {
namespace prometheus {

Metric::Metric(const std::vector<string>& label_values) : label_values(label_values) {}


Counter::Counter(const std::vector<string>& label_values) : Metric(label_values) {}

void Counter::inc() {
    std::lock_guard<std::mutex> lock(metric_lock);

    ++value;
}

string Counter::to_string() {
    return std::to_string(value);
}

std::ostringstream& Counter::to_string(std::ostringstream& out) {
    std::lock_guard<std::mutex> lock(metric_lock);

    out << value;
    return out;
}


Gauge::Gauge(const std::vector<string>& label_values) : Metric(label_values) {}

void Gauge::set(const double value) {
    std::lock_guard<std::mutex> lock(metric_lock);

    this->value = value;
    this->last_update_time_stamp = get_time_in_milliseconds();
}

string Gauge::to_string() {
    return fmt::format("{:f} {}", value, last_update_time_stamp);
}

std::ostringstream& Gauge::to_string(std::ostringstream& out) {
    std::lock_guard<std::mutex> lock(metric_lock);

    fmt::print(out, "{:f} {}", value, last_update_time_stamp);
    return out;
}

/* static */
uint64_t Gauge::get_time_in_milliseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
}


template<typename T>
MetricFamily<T>::MetricFamily(const string& name, const string& stage_name,
                              const std::vector<string>& label_names,
                              const MetricFamily<T>::MetricType metric_type) :
    name(name),
    stage_name(stage_name),
    label_names(label_names),
    metric_type(metric_type) {}

template<typename T>
string MetricFamily<T>::serialize() {
    std::lock_guard<std::mutex> lock(metrics_lock);

    if (metrics.empty())
        return "";

    std::ostringstream out;
    out << "# HELP " << name << "\n";
    switch (metric_type) {
        case MetricFamily<T>::MetricType::Counter:
            out << "# TYPE " << name << " counter\n";
            break;
        case MetricFamily<T>::MetricType::Gauge:
            out << "# TYPE " << name << " gauge\n";
            break;
        default:
            out << "# TYPE " << name << " untyped\n";
    }
    for (auto& m : metrics) {
        out << name;
        out << "{"
            << "stage_name=\"" << stage_name << "\"";
        if (!label_names.empty()) {
            auto value = m.label_values.begin();
            for (auto label : label_names) {
                out << ",";
                out << label << "=\"" << *value++ << "\"";
            }
        }
        out << "}"
            << " ";
        m.to_string(out);
        out << "\n";
    }
    return out.str();
}


Metrics::Metrics() {}

Metrics& Metrics::instance() {
    static Metrics _instance;
    return _instance;
}


Metrics::~Metrics() {
    restServer::instance().remove_get_callback("/metrics");
}

string Metrics::serialize() {
    std::ostringstream out;

    std::lock_guard<std::mutex> lock(metrics_lock);

    for (auto& f : families) {
        // out << f.first << ": " << f.second.label_names.size() << "\n";
        out << f.second->serialize();
    }

    return out.str();
}

void Metrics::add(const string name, const string stage_name,
                  std::shared_ptr<Serializable> metric) {
    if (name.empty()) {
        ERROR("Empty metric name. Exiting.");
        throw std::runtime_error("Empty metric name.");
    }
    if (stage_name.empty()) {
        ERROR("Empty stage for metric %s. Exiting.", name.c_str());
        throw std::runtime_error("Empty stage name: " + name);
    }

    std::lock_guard<std::mutex> lock(metrics_lock);

    auto key = std::make_tuple(name, stage_name);
    if (families.count(key)) {
        ERROR("Duplicate metric name: %s. Exiting.", (name + ":" + stage_name).c_str());
        throw std::runtime_error("Duplicate metric name: " + name + ":" + stage_name);
    }
    families[key] = metric;
}

Gauge& Metrics::add_gauge(const string& name, const string& stage_name) {
    const std::vector<string> empty_labels;
    auto f = std::make_shared<MetricFamily<Gauge>>(name, stage_name, empty_labels,
                                                   MetricFamily<Gauge>::MetricType::Gauge);
    add(name, stage_name, f);
    return f->labels({});
}

MetricFamily<Gauge>& Metrics::add_gauge(const string& name, const string& stage_name,
                                        const std::vector<string>& label_names) {
    auto f = std::make_shared<MetricFamily<Gauge>>(name, stage_name, label_names,
                                                   MetricFamily<Gauge>::MetricType::Gauge);
    add(name, stage_name, f);
    return *f;
}

Counter& Metrics::add_counter(const string& name, const string& stage_name) {
    const std::vector<string> empty_labels;
    auto f = std::shared_ptr<MetricFamily<Counter>>(new MetricFamily<Counter>(
        name, stage_name, empty_labels, MetricFamily<Counter>::MetricType::Counter));
    add(name, stage_name, f);
    return f->labels({});
}

MetricFamily<Counter>& Metrics::add_counter(const string& name, const string& stage_name,
                                            const std::vector<string>& label_names) {
    auto f = std::shared_ptr<MetricFamily<Counter>>(new MetricFamily<Counter>(
        name, stage_name, label_names, MetricFamily<Counter>::MetricType::Counter));
    add(name, stage_name, f);
    return *f;
}


void Metrics::remove_stage_metrics(const string& stage_name) {
    std::lock_guard<std::mutex> lock(metrics_lock);

    for (auto it = std::begin(families), last = std::end(families); it != last;) {
        if (std::get<1>(it->first) == stage_name) {
            it = families.erase(it);
        } else {
            ++it;
        }
    }
}


void Metrics::metrics_callback(connectionInstance& conn) {
    string output = serialize();

    // Sending the reply doesn't need to be locked.
    // Just accessing the metrics array.
    conn.send_text_reply(output);
}

void Metrics::register_with_server(restServer* rest_server) {
    using namespace std::placeholders;
    rest_server->register_get_callback("/metrics", std::bind(&Metrics::metrics_callback, this, _1));
}

} // namespace prometheus
} // namespace kotekan
