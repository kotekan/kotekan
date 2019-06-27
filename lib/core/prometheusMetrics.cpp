#include "prometheusMetrics.hpp"

#include "errors.h"
#include "metadata.h"

namespace kotekan {

Metric::Metric(const std::vector<string>& label_values) :
    label_values(label_values)
{}


Counter::Counter(const std::vector<string>& label_values) :
    Metric(label_values)
{}

void Counter::inc() {
    ++value;
}

string Counter::to_string() {
    return std::to_string(value);
}

std::ostringstream& Counter::to_string(std::ostringstream& out) {
    out << value;
    return out;
}

Gauge::Gauge(const std::vector<string>& label_values) :
    Metric(label_values)
{}

void Gauge::set(const double value) {
    this->value = value;
}

string Gauge::to_string() {
    std::ostringstream out;
    to_string(out);
    return out.str();
}

std::ostringstream& Gauge::to_string(std::ostringstream& out) {
    out << value;
    return out;
}

template<typename T>
Family<T>::Family(const string& name,
                  const string& stage_name,
                  const std::vector<string>& label_names,
                  const MetricType metric_type) :
    name(name),
    stage_name(stage_name),
    label_names(label_names),
    metric_type(metric_type)
{}

template<typename T>
string Family<T>::Serialize() {
    if (metrics.empty()) return "";

    std::ostringstream out;
    out << "# HELP " << name << "\n";
    switch (metric_type) {
    case MetricType::Counter:
        out << "# TYPE " << name << " counter\n";
        break;
    case MetricType::Gauge:
        out << "# TYPE " << name << " gauge\n";
        break;
    default:
        out << "# TYPE " << name << " untyped\n";
    }
    for (auto m : metrics) {
        out << name;
        out << "{" << "stage_name=\"" << stage_name << "\"";
        if (!label_names.empty()) {
            auto value = m.label_values.begin();
            for (auto label : label_names) {
                out << ",";
                out << label << "=\"" << *value++ << "\"";
            }
        }
        out << "}" << " ";
        m.to_string(out);
        out << "\n";
    }
    return out.str();
}

prometheusMetrics::prometheusMetrics() {}

prometheusMetrics& prometheusMetrics::instance() {
    static prometheusMetrics _instance;
    return _instance;
}


prometheusMetrics::~prometheusMetrics() {
    restServer::instance().remove_get_callback("/metrics");
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

void prometheusMetrics::Add(const string name,
                            const string stage_name,
                            std::shared_ptr<Serializable> s) {
    if (stage_name.empty()) {
        throw std::runtime_error("Empty stage name: " + stage_name);
    }
    auto key = std::make_tuple(name, stage_name);
    if (families.count(key)) {
        throw std::runtime_error("Duplicate metric name: " + name);
    }
    families[key] = s;
}

string prometheusMetrics::Serialize() {
    std::ostringstream out;

    for (auto& f : families) {
        // out << f.first << ": " << f.second.label_names.size() << "\n";
        out << f.second->Serialize();
    }

    return out.str();
}

Gauge& prometheusMetrics::AddGauge(const string& name,
                                   const string& stage_name) {
    const std::vector<string> empty_labels;
    auto f = std::make_shared<Family<Gauge>>(name, stage_name, empty_labels, MetricType::Gauge);
    Add(name, stage_name, f);
    return f->Labels({});
}

Family<Gauge>& prometheusMetrics::AddGauge(const string& name,
                                           const string& stage_name,
                                           const std::vector<string>& label_names) {
    auto f = std::make_shared<Family<Gauge>>(name, stage_name, label_names, MetricType::Gauge);
    Add(name, stage_name, f);
    return *f;
}

Counter& prometheusMetrics::AddCounter(const string& name,
                                       const string& stage_name) {
    const std::vector<string> empty_labels;
    auto f = std::make_shared<Family<Counter>>(name, stage_name, empty_labels, MetricType::Counter);
    Add(name, stage_name, f);
    return f->Labels({});
}

Family<Counter>& prometheusMetrics::AddCounter(const string& name,
                                               const string& stage_name,
                                               const std::vector<string>& label_names) {
    auto f = std::make_shared<Family<Counter>>(name, stage_name, label_names, MetricType::Counter);
    Add(name, stage_name, f);
    return *f;
}


void prometheusMetrics::metrics_callback(connectionInstance& conn) {
    string output = Serialize();

    // Sending the reply doesn't need to be locked.
    // Just accessing the metrics array.
    conn.send_text_reply(output);
}

void prometheusMetrics::register_with_server(restServer* rest_server) {
    using namespace std::placeholders;
    rest_server->register_get_callback("/metrics",
                                       std::bind(&prometheusMetrics::metrics_callback, this, _1));
}

void gauge::set(const double value) {
    this->value = value;
    this->last_update_time_stamp = get_time_in_milliseconds();
}

string gauge::to_string() {
    std::ostringstream output;
    output << value << " " << std::to_string(last_update_time_stamp);
    return output.str();
}

/* static */
uint64_t metric::get_time_in_milliseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
}

} // namespace kotekan
