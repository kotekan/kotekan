#ifndef PROMETHEUS_METRICS_HPP
#define PROMETHEUS_METRICS_HPP

#include "visUtil.hpp"

#include <deque>     // for deque
#include <iosfwd>    // for ostringstream
#include <map>       // for map
#include <memory>    // for shared_ptr
#include <mutex>     // for mutex, lock_guard
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint64_t
#include <string>    // for string
#include <tuple>     // for tuple
#include <vector>    // for vector

namespace kotekan {

class restServer;
class connectionInstance;

namespace prometheus {

/**
 * @class Metric
 * @brief An internal base class for storing metric value for a given combination of label values
 */
class Metric {
public:
    Metric(const std::vector<std::string>& label_values);
    virtual ~Metric() = default;

    /// @brief Returns the stored value as a string.
    virtual std::string to_string() = 0;
    /// @brief Formats the stored value as a string into the given output stream.
    virtual std::ostringstream& to_string(std::ostringstream& out) = 0;
    const std::vector<std::string> label_values;

protected:
    /// Metric updating lock
    std::mutex metric_lock;
};

/**
 * @class Counter
 * @brief Represents a metric whose value can only go up
 *
 * @remark See [Prometheus
 * documentation](https://prometheus.io/docs/instrumenting/exposition_formats/) for the precise
 * format specification.
 */
class Counter : public Metric {
public:
    Counter(const std::vector<std::string>&);
    void inc();
    void inc(const uint64_t increment);
    std::string to_string() override;
    std::ostringstream& to_string(std::ostringstream& out) override;

private:
    /// The actual value to be returned
    uint64_t value = 0;
};

/**
 * @class Gauge
 * @brief Represents a metric whose value can go up and down
 *
 * @remark See [Prometheus
 * documentation](https://prometheus.io/docs/instrumenting/exposition_formats/) for the precise
 * format specification.
 */
class Gauge : public Metric {
public:
    Gauge(const std::vector<std::string>&);
    void set(const double);
    std::string to_string() override;
    std::ostringstream& to_string(std::ostringstream& out) override;

private:
    /// Internal function to get the time in
    static uint64_t get_time_in_milliseconds();

    /// The actual value to be returned
    double value = 0;

    /// Time stamp in milliseconds.
    uint64_t last_update_time_stamp;
};

/**
 * @class EndpointTimer
 * @brief Represents a metric that can store timer stats and output avg and max reply time
 */
class EndpointTimer : public Metric {
public:
    EndpointTimer(const std::vector<std::string>&);
    void update(const double);
    std::string to_string() override;
    std::ostringstream& to_string(std::ostringstream& out) override;
    bool is_slow();

private:
    /// Structure to store values and compute max and avg.
    StatTracker stat_tracker;
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
    virtual std::string serialize() = 0;
};

/**
 * @class MetricFamily
 * @brief Groups together a set of metrics with the same name, type, and label names, but different
 * label values.
 */
template<typename T>
class MetricFamily : public Serializable {
public:
    enum class MetricType {
        Counter,
        Gauge,
        EndpointTimer,
        Untyped,
    };

    MetricFamily(const std::string& name, const std::string& stage,
                 const std::vector<std::string>& label_names,
                 const MetricType metric_type = MetricType::Untyped);

    /**
     * @brief Returns the ``Metric`` instance for the given combination of label values
     *
     * If the combination of values is seen for the first time, a new Metric
     * instance will be created and added to the family.
     *
     * @param label_values
     * @return reference to the Metric
     * @ @throw std::runtime_error if the number of label values doesn't match the length of the
     * family's label names
     */
    T& labels(const std::vector<std::string>& label_values) {
        if (label_names.size() != label_values.size()) {
            throw std::runtime_error("Label values don't match the names");
        }

        std::lock_guard<std::mutex> lock(metrics_lock);

        for (auto& m : metrics) {
            if (m.label_values == label_values) {
                return m;
            }
        }
        metrics.emplace_back(label_values);
        return metrics.back();
    }


    std::string serialize() override;

    /// metric name
    const std::string name;

    /// stage name
    const std::string stage_name;

    /// label names
    const std::vector<std::string> label_names;

private:
    /// metric instances for label combinations observed so far
    std::deque<T> metrics;

    /// metric type
    const MetricType metric_type;

    /// Metric list updating lock
    std::mutex metrics_lock;
};

/**
 * @class Metrics
 * @brief Class for exporting system metrics to a prometheus server
 *
 * This class must be registered with a kotekan REST server instance.=,
 * using the @c register_with_server() function.
 *
 * The typical usage is to declare the metric with @c add_gauge or @c
 * add_counter methods, and then use the returned Metric instance to set the
 * metric's value (for a particular combination of label values, if applicable).
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
    std::string serialize();

    /**
     * @brief Adds a new metric of type gauge and no labels
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @return a reference to the newly created @c Gauge instance
     * @throw std::runtime_error if the metric with that name is already registered.
     */
    Gauge& add_gauge(const std::string& name, const std::string& stage_name);

    /**
     * @brief Adds a new metric family of type gauge
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @param label_names The names of the labels used
     * @return a reference to the newly created @c MetricFamily<Gauge> instance
     * @throw std::runtime_error if the metric with that name is already registered.
     */
    MetricFamily<Gauge>& add_gauge(const std::string& name, const std::string& stage_name,
                                   const std::vector<std::string>& label_names);

    /**
     * @brief Adds a new metric of type endpoint_timer and no labels
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @return a reference to the newly created @c EndpointTimer instance
     * @throw std::runtime_error if the metric with that name is already registered.
     */
    EndpointTimer& add_endpoint(const std::string& name, const std::string& stage_name);

    /**
     * @brief Adds a new metric family of type endpoint_timer
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @param label_names The names of the labels used
     * @return a reference to the newly created @c MetricFamily<EndpointTimer> instance
     * @throw std::runtime_error if the metric with that name is already registered.
     */
    MetricFamily<EndpointTimer>& add_endpoint(const std::string& name,
                                              const std::string& stage_name,
                                              const std::vector<std::string>& label_names);

    /**
     * @brief Adds a new metric of type counter and no labels
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @return a reference to the newly created @c Counter instance
     * @throw std::runtime_error if the metric with that name is already registered.
     */
    Counter& add_counter(const std::string& name, const std::string& stage_name);

    /**
     * @brief Adds a new metric family of type counter
     *
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @param label_names The names of the labels used
     * @return a reference to the newly created @c MetricFamily<Counter> instance
     * @throw std::runtime_error if the metric with that name is already registered.
     */
    MetricFamily<Counter>& add_counter(const std::string& name, const std::string& stage_name,
                                       const std::vector<std::string>& label_names);

    /**
     * @brief Remove all registered stage metrics
     *
     * After the method completes, these metrics can be re-declared with the same metric and stage
     * name without ``add`` throwing an error.
     *
     * It is not an error to call this method with an unknown stage name, or a stage_name for which
     * metrics were already deleted.
     *
     * Note: after this method completes, references to those ``Metric`` instances that were
     * returned by the ``Metric::add_gauge`` and ``Metric::add_counter`` are invalid.
     *
     * @param stage_name The stage name used in metric declaration, normally @c unique_name.
     */
    void remove_stage_metrics(const std::string& stage_name);

private:
    /// Constructor, not used directly
    Metrics();

    /**
     * @brief Adds a new metric
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
     * @param name The name of the metric.
     * @param stage_name The unique stage name, normally @c unique_name.
     * @param metric the metric family to registered under this name/stage
     * @throw std::runtime_error if the metric with that name is already registered or if the
     * ``name`` or ``stage_name`` are left empty.
     */
    void add(const std::string name, const std::string stage_name,
             std::shared_ptr<Serializable> metric);

    /**
     * The metric storage object with the format:
     * <<metric_name, stage_name>, MetricFamily>
     */
    std::map<std::tuple<std::string, std::string>, std::shared_ptr<Serializable>> families;

    /// Metric updating lock
    std::mutex metrics_lock;
};

} // namespace prometheus
} // namespace kotekan

#endif /* PROMETHEUS_METRICS_HPP */
