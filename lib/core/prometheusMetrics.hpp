#ifndef PROMETHEUS_METRICS_HPP
#define PROMETHEUS_METRICS_HPP

#include "restServer.hpp"

#include "json.hpp"

#include <map>
#include <mutex>
#include <string>
#include <tuple>

using nlohmann::json;
using std::map;
using std::string;

namespace kotekan {
namespace prometheus {

/**
 * @class metric
 * @brief An internal base class for storing metric values
 */
class metric {
public:
    /// @brief Returns the stored value as a string.
    virtual string to_string() = 0;
    virtual ~metric() = default;

protected:
    /// The actual value to be returned
    double value = 0;

    /// Internal function to get the time in
    static uint64_t get_time_in_milliseconds();
};

class gauge : public metric {
public:
    /// @brief Sets the current metric value
    void set(double value);
    string to_string() override;

private:
    /// Time stamp in milliseconds.
    uint64_t last_update_time_stamp;
};

/**
 * @class Metric
 * @brief An internal base class for storing metric values
 */
class Metric {
public:
    Metric(const std::vector<string>&);
    virtual ~Metric() = default;

    /// @brief Returns the stored value as a string.
    virtual string to_string() = 0;
    /// @brief Formats the stored value as a string into the given output stream.
    virtual std::ostringstream& to_string(std::ostringstream& out) = 0;
    const std::vector<string> label_values;
};

/**
 * @class Counter
 * @brief Represents a metric whose value can only go up
 * @see https://prometheus.io/docs/concepts/metric_types/#counter
 */
class Counter : public Metric {
public:
    Counter(const std::vector<string>&);
    void inc();
    string to_string() override;
    std::ostringstream& to_string(std::ostringstream& out) override;

private:
    /// The actual value to be returned
    int value = 0;
};

/**
 * @class Gauge
 * @brief Represents a metric whose value can go up and down
 * @see https://prometheus.io/docs/concepts/metric_types/#gauge
 */
class Gauge : public Metric {
public:
    Gauge(const std::vector<string>&);
    void set(const double);
    string to_string() override;
    std::ostringstream& to_string(std::ostringstream& out) override;

private:
    /// The actual value to be returned
    double value = 0;
};

enum class MetricType {
    Counter,
    Gauge,
    Untyped,
};

/**
 * @class Serializable
 * @brief Interface for types that can be represented in Prometheus text format.
 */
struct Serializable {
    virtual ~Serializable() = default;

    /**
     * @brief Returns a string representation of metrics in Prometheus text format.
     *
     * @remark See [Prometheus
     * documentation](https://prometheus.io/docs/instrumenting/exposition_formats/) for the precise
     * format specification.
     */
    virtual string Serialize() = 0;
};

template<typename T>
class Family : public Serializable {
public:
    Family(const string& name, const string& stage, const std::vector<string>& label_names,
           const MetricType = MetricType::Untyped);

    // T& Labels(const std::vector<string>& label_values);

    T& Labels(const std::vector<string>& label_values) {
        if (label_names.size() != label_values.size()) {
            throw std::runtime_error("Label values don't match the names");
        }

        for (auto& m : metrics) {
            if (m.label_values == label_values) {
                return m;
            }
        }
        metrics.emplace_back(label_values);
        return metrics.back();
    }


    string Serialize() override;

    const string name;
    const string stage_name;
    const std::vector<string> label_names;

private:
    std::vector<T> metrics;
    const MetricType metric_type;
};

/**
 * @class Metrics
 * @brief Class for exporting system metrics to a prometheus server
 *
 * This class must be registered with a kotekan REST server instance.=,
 * using the @c register_with_server() function.
 *
 * The most common function to call will be @c add_stage_metric, which
 * adds or updates a given metric.
 *
 * This class is a singleton, and can be accessed with @c instance()
 *
 * @todo Make this class auto register with the REST server.
 *
 * @author Andre Renard
 */
class Metrics {
public:
    /// Destructor
    ~Metrics();

    /**
     * @brief Returns the singleton instance of the prometheusMetrics object.
     * @return A pointer to the prometheusMetrics object
     */
    static Metrics& instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        /metrics end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer* rest_server);

    /**
     * @brief The call back function for the REST server to use.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results too.
     */
    void metrics_callback(connectionInstance& conn);

    /**
     * @brief Converts the registered metrics to Prometheus text exposition format
     *
     * Metrics are serialized one per line, ending with a newline, with each line
     * in the following format:
     * ```
     * metric_name [
     * "{" label_name "=" `"` label_value `"` { "," label_name "=" `"` label_value `"` } [ "," ] "}"
     * ] value timestamp
     * ```
     *
     * @remark See [Prometheus
     * documentation](https://prometheus.io/docs/instrumenting/exposition_formats/)
     * for the precise format specification.
     *
     * @return A string representation of the metrics
     */
    string serialize();

    /**
     * @brief Adds a new metric or updates an existing one.
     *
     * The value given must be a number (float, double, int, etc.) and be
     * convertible to a string with the standard @c std::to_string function.
     *
     * Metrics are stored based on the unique tuple: (name, stage_name, labels).
     *
     * Any new tuple will be added to the list of metrics, and any existing
     * tuple will be updates with the value give.
     *
     * Note the time this function was last called is also stored with the value
     * so even if the value might not update, it can still be useful in some cases
     * to call this function to update the time stamp associated with the metric.
     *
     * Note metrics should follow the prometheus metric name and label
     * conventions which can be found here: https://prometheus.io/docs/practices/naming/
     *
     * In particular, metrics must be named to follow certain conventions. They
     * should be prefixed by `kotekan_<stagetype>_` where `<stage_type>` is
     * the class name of the stage in lower case, followed by the rest of the
     * metric name, which should satisfy the standard prometheus guidelines.
     * For example `kotekan_viswriter_write_time_seconds`.
     *
     * Also note that label values must be surrounded by double quotes.
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @param value The value associated with this metric.
     * @param labels (optional) The metric labels.
     */

    /**
     * @brief Removes a given metric.
     *
     * Removed the metric based on the unique tuple: (name, stage_name, labels).
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name.
     * @param labels (optional) Any metric labels.
     */
    void remove_metric(const string& name, const string& stage_name, const string& labels = "");

    string Serialize();

    Gauge& AddGauge(const string&, const string&);
    Family<Gauge>& AddGauge(const string&, const string&, const std::vector<string>&);
    Counter& AddCounter(const string&, const string&);
    Family<Counter>& AddCounter(const string&, const string&, const std::vector<string>&);

private:
    /// Constructor, not used directly
    Metrics();

    void Add(const string, const string, std::shared_ptr<Serializable>);

    /**
     * The metric storage object with the format:
     * <<metric_name, stage_name, tags>, metric>
     */
    map<std::tuple<string, string, string>, std::unique_ptr<metric>> stage_metrics;

    /**
     * The metric storage object with the format:
     * <<metric_name, stage_name>, Family>
     */
    map<std::tuple<string, string>, std::shared_ptr<Serializable>> families;

    /// Metric updating lock
    std::mutex metrics_lock;
};

} // namespace prometheus
} // namespace kotekan

#endif /* PROMETHEUS_METRICS_HPP */
