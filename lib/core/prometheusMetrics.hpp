#ifndef PROMETHEUS_METRICS_HPP
#define PROMETHEUS_METRICS_HPP

#include <map>
#include <string>
#include <tuple>
#include <mutex>

#include "json.hpp"
#include "restServer.hpp"

using std::map;
using std::string;
using nlohmann::json;

/**
 * @class prometheusMetrics
 * @brief Class for exporting system metrics to a prometheus server
 *
 * This class must be registered with a kotekan REST server instance.=,
 * using the @c register_with_server() function.
 *
 * The most common function to call will be @c add_process_metric, which
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
    static prometheusMetrics &instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        /metrics end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer * rest_server);

    /**
     * @brief The call back function for the REST server to use.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results too.
     */
    void metrics_callback(connectionInstance& conn);

    /**
     * @brief Adds a new metric or updates an existing one.
     *
     * The value given must be a number (float, double, int, etc.) and be
     * convertible to a string with the standard @c std::to_string function.
     *
     * Metrics are stored based on the unique tuple: (name, process_name, labels).
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
     * should be prefixed by `kotekan_<processtype>_` where `<process_type>` is
     * the class name of the process in lower case, followed by the rest of the
     * metric name, which should satisfy the standard prometheus guidelines.
     * For example `kotekan_viswriter_write_time_seconds`.
     *
     * Also note that label values must be surrounded by double quotes.
     * 
     * @param name The name of the metric.
     * @param process_name The unique process name, normally @c unique_name.
     * @param value The value associated with this metric.
     * @param labels (optional) The metric labels.
     */
    template<class T>
    void add_process_metric(const string& name,
                            const string& process_name,
                            const T& value,
                            const string& labels = "") {
        std::lock_guard<std::mutex> lock(metrics_lock);
        std::tuple<string, string, string> key {name, process_name, labels};

        if (process_metrics.count(key) == 0) {
            processMetric<T> * new_metric = new processMetric<T>;
            process_metrics[key] = (prometheusMetrics::metric *)new_metric;
        }

        ((processMetric<T> *)process_metrics[key])->value = value;
        ((processMetric<T> *)process_metrics[key])->last_update_time_stamp = get_time_in_milliseconds();
    }

    /**
     * @brief Removes a given metric.
     *
     * Removed the metric based on the unique tuple: (name, process_name, labels).
     *
     * @param name The name of the metric.
     * @param process_name The unique process name.
     * @param labels (optional) Any metric labels.
     */
    void remove_metric(const string &name,
                       const string &process_name,
                       const string &labels = "");

private:

    /**
     * @class metric
     * @brief An internal base class for storing metric values
     */
    struct metric {
        virtual ~metric();
        /// Time stamp in milliseconds.
        uint64_t last_update_time_stamp;
        /// The pure virtual function for converting the stored value to a string
        virtual string to_string() = 0;
    };

    /**
     * @class processMetric
     * @brief A template class for storing a numeric value which can be converted
     *        to a string with @c std::to_string()
     */
    template <typename T,
              typename = std::enable_if<std::is_arithmetic<T>::value>>
    struct processMetric : public metric {
        /// The actual value to be returned
        T value;
        /**
         * @brief Returns the stored value as a string.
         */
        string to_string() override {
            return std::to_string(value);
        }
    };

    /// Constructor, not used directly
    prometheusMetrics();

    /**
     * The metric storage object with the format:
     * <<metric_name, process_name, tags>, metric>
     */
    map<std::tuple<string, string, string>, metric*> process_metrics;

    /// Internal function to get the time in
    uint64_t get_time_in_milliseconds();

    /// Metric updating lock
    std::mutex metrics_lock;
};

#endif /* PROMETHEUS_METRICS_HPP */