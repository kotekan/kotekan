#include "gnssHostRows.hpp"

#include "kotekanLogging.hpp"    // for FATAL_ERROR_NON_OO, ERROR_NON_OO, FatalError
#include "prometheusMetrics.hpp" // for Metrics

#include <mutex>     // for mutex, lock_guard
#include <pthread.h> // for pthread_setaffinity_np
#include <set>       // for set
#include <thread>    // for thread

using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::restServer;
using nlohmann::json;

namespace gnss {

namespace {
/// Serialises the check and the registration of rest_register_unique, and guards row_names.
std::mutex host_lock;
/// The row names host_rows has handed out and release_host has not taken back.
std::set<std::string> row_names;

void run_row(const std::function<void(size_t)>& run, size_t row) {
    try {
        run(row);
    } catch (const FatalError& e) {
        // FATAL_ERROR already called exit_kotekan, as for a stage's own thread.
        ERROR_NON_OO("GNSS host row {:d} caught FatalError: {:s}", row, e.what());
    }
}
} // namespace

std::vector<HostRow> host_rows(Config& config, const std::string& unique_name,
                               const std::string& suffix, const std::vector<std::string>& row_keys,
                               const std::vector<std::string>& row_only_keys) {
    std::vector<HostRow> rows;
    if (!config.exists(unique_name, "chains")) {
        rows.push_back({"", unique_name, unique_name});
    } else {
        const json chains = config.get_value(unique_name, "chains");
        if (!chains.is_object() || chains.empty())
            FATAL_ERROR_NON_OO("{:s}: chains must be a non-empty map of rows", unique_name);
        for (const std::string& k : row_only_keys)
            if (config.exists(unique_name, k))
                FATAL_ERROR_NON_OO("{:s}: {:s} is set per row in chains, not on the host",
                                   unique_name, k);
        for (auto it = chains.begin(); it != chains.end(); ++it) {
            HostRow row = {it.key(), unique_name + "/chains/" + it.key(), "/" + it.key() + suffix};
            for (const std::string& k : row_keys)
                if (!config.exists(row.cfg_path, k))
                    FATAL_ERROR_NON_OO("{:s}: row {:s} does not set {:s}", unique_name, row.key, k);
            rows.push_back(row);
        }
    }

    std::lock_guard<std::mutex> lock(host_lock);
    for (const HostRow& row : rows)
        if (!row_names.insert(row.name).second)
            FATAL_ERROR_NON_OO("{:s}: row name {:s} is already in use", unique_name, row.name);
    return rows;
}

void rest_register_unique(const std::string& path,
                          std::function<void(connectionInstance&)> callback,
                          std::vector<std::string>& paths) {
    restServer& rest = restServer::instance();
    std::lock_guard<std::mutex> lock(host_lock);
    if (rest.has_callback(path))
        FATAL_ERROR_NON_OO("REST endpoint {:s} is already registered", path);
    rest.register_get_callback(path, callback);
    paths.push_back(path);
}

void rest_register_unique(const std::string& path,
                          std::function<void(connectionInstance&, json&)> callback,
                          std::vector<std::string>& paths) {
    restServer& rest = restServer::instance();
    std::lock_guard<std::mutex> lock(host_lock);
    if (rest.has_callback(path))
        FATAL_ERROR_NON_OO("REST endpoint {:s} is already registered", path);
    rest.register_post_callback(path, callback);
    paths.push_back(path);
}

void rest_register_chains(const std::string& unique_name, const std::vector<HostRow>& rows,
                          std::vector<std::string>& paths) {
    if (rows[0].key.empty())
        return;
    json reply = json::array();
    for (const HostRow& row : rows)
        reply.push_back({{"name", row.key}, {"stage", row.name}});
    rest_register_unique(
        unique_name + "/chains", [reply](connectionInstance& conn) { conn.send_json_reply(reply); },
        paths);
}

void release_host(const std::vector<HostRow>& rows, const std::vector<std::string>& paths) {
    restServer& rest = restServer::instance();
    for (const std::string& path : paths) {
        rest.remove_get_callback(path);
        rest.remove_json_callback(path);
    }
    std::lock_guard<std::mutex> lock(host_lock);
    for (const HostRow& row : rows) {
        kotekan::prometheus::Metrics::instance().remove_stage_metrics(row.name);
        row_names.erase(row.name);
    }
}

void run_rows(size_t n_rows, const std::vector<int>& cpus, const std::function<void(size_t)>& run) {
    if (n_rows == 1) {
        run(0);
        return;
    }
    std::vector<std::thread> threads;
    for (size_t r = 1; r < n_rows; ++r) {
        threads.emplace_back(run_row, std::cref(run), r);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int c : cpus)
            CPU_SET(c, &cpuset);
        pthread_setaffinity_np(threads.back().native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    run_row(run, 0);
    for (std::thread& t : threads)
        t.join();
}

} // namespace gnss
