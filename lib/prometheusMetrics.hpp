#ifndef PROMETHEUS_METRICS_HPP
#define PROMETHEUS_METRICS_HPP

#include <map>
#include <string>
#include <tuple>

#include "json.hpp"
#include "restServer.hpp"

using std::map;
using std::string;
using nlohmann::json;

/**
 *
 * TODO: Support double not just int64_t as a data type
 */
class prometheusMetrics {
public:

    struct metric {
        virtual ~metric();
        /// Time stamp in milliseconds.
        uint64_t last_update_time_stamp;
        virtual string to_string() = 0;
    };

    template <class T>
    struct processMetric : public metric {
        /// The actual value to be returned
        T value;
        /**
         * @brief Returns the prometheous formated value.
         */
        string to_string();
    };

    prometheusMetrics();
    ~prometheusMetrics();

    void register_with_server(restServer * rest_server);

    void metrics_callback(connectionInstance& conn);

    template <class T>
    void add_process_metric(const string &name,
                            const string &process_name,
                            const T &value,
                            const string &tags = "");

    void remove_metric(const string &name,
                       const string &process_name,
                       const string &tags = "");

private:

    // <<metric_name, process_name, tags>, metric>
    map<std::tuple<string, string, string>, metric*> process_metrics;

    /// Internal function to get the time in
    uint64_t get_time_in_milliseconds();
};

#endif /* PROMETHEUS_METRICS_HPP */