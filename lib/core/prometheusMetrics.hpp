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

class Gauge : public metric {
public:
    /// @brief Sets the current metric value
    void set(double value);
    string to_string() override;
private:
    /// Time stamp in milliseconds.
    uint64_t last_update_time_stamp;
};

class Counter : public metric {
public:
    void inc();
    string to_string() override;
};
/**
 * @class prometheusMetrics
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
class prometheusMetrics {
public:
    /// Destructor
    ~prometheusMetrics();

    /**
     * @brief Returns the singleton instance of the prometheusMetrics object.
     * @return A pointer to the prometheusMetrics object
     */
    static prometheusMetrics& instance();

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
    Gauge* add_stage_metric(const string& name,
                            const string& stage_name,
                            const double value,
                            const string& labels = "");

    Counter* add_stage_counter(const string& name,
                               const string& stage_name,
                               const string& labels = "");

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

private:

    /// Constructor, not used directly
    prometheusMetrics();

    /**
     * The metric storage object with the format:
     * <<metric_name, stage_name, tags>, metric>
     */
    map<std::tuple<string, string, string>, std::unique_ptr<metric>> stage_metrics;

    /// Metric updating lock
    std::mutex metrics_lock;
};

} // namespace kotekan

#endif /* PROMETHEUS_METRICS_HPP */
