#include "gnssHostRows.hpp"

#include "kotekanLogging.hpp"    // for FATAL_ERROR_NON_OO, ERROR_NON_OO, FatalError
#include "prometheusMetrics.hpp" // for Metrics
#include "util.h"                // for string_tail

#include <mutex>     // for mutex, lock_guard
#include <pthread.h> // for pthread_setaffinity_np, pthread_setname_np
#include <sched.h>   // for CPU_SETSIZE
#include <set>       // for set
#include <stdexcept> // for invalid_argument
#include <thread>    // for thread

using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
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
        ERROR_NON_OO("GNSS host thread {:d} caught FatalError: {:s}", row, e.what());
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

std::vector<HostWorker> host_workers(Config& config, const std::string& unique_name,
                                     const std::vector<HostRow>& rows) {
    std::vector<HostWorker> out;
    if (!config.exists(unique_name, "workers")) {
        for (size_t i = 0; i < rows.size(); ++i)
            out.push_back({rows[i].name, {}, {i}});
        return out;
    }
    if (rows[0].key.empty())
        FATAL_ERROR_NON_OO("{:s}: workers needs a chains block", unique_name);
    const json ws = config.get_value(unique_name, "workers");
    if (!ws.is_array() || ws.empty())
        FATAL_ERROR_NON_OO("{:s}: workers must be a non-empty list", unique_name);
    // The worker of each row, ws.size() while it has none.
    std::vector<size_t> of(rows.size(), ws.size());
    for (size_t k = 0; k < ws.size(); ++k) {
        const json& e = ws[k];
        if (!e.is_object() || !e.contains("rows") || !e["rows"].is_array() || e["rows"].empty()
            || !e.contains("cpu_affinity") || !e["cpu_affinity"].is_array()
            || e["cpu_affinity"].empty())
            FATAL_ERROR_NON_OO("{:s}: worker {:d} needs rows and cpu_affinity, each a non-empty "
                               "list",
                               unique_name, k);
        HostWorker w{unique_name + "/w" + std::to_string(k), {}, {}};
        for (const json& c : e["cpu_affinity"]) {
            if (!c.is_number_integer() || c.get<int64_t>() < 0 || c.get<int64_t>() >= CPU_SETSIZE)
                FATAL_ERROR_NON_OO("{:s}: worker {:d} has cpu_affinity entry {:s}, not a core",
                                   unique_name, k, c.dump());
            w.cpus.push_back(c.get<int>());
        }
        for (const json& key : e["rows"]) {
            size_t i = 0;
            while (i < rows.size() && !(key.is_string() && key.get<std::string>() == rows[i].key))
                ++i;
            if (i == rows.size())
                FATAL_ERROR_NON_OO("{:s}: worker {:d} runs {:s}, which is not a row of chains",
                                   unique_name, k, key.dump());
            if (of[i] < ws.size())
                FATAL_ERROR_NON_OO("{:s}: row {:s} is in workers {:d} and {:d}", unique_name,
                                   rows[i].key, of[i], k);
            of[i] = k;
            w.rows.push_back(i);
        }
        out.push_back(w);
    }
    for (size_t i = 0; i < rows.size(); ++i)
        if (of[i] == ws.size())
            FATAL_ERROR_NON_OO("{:s}: row {:s} is in no worker", unique_name, rows[i].key);
    return out;
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

void rest_reply(connectionInstance& conn, const RestGet& get, const RestQuery& query) {
    json reply;
    try {
        reply = get(query);
    } catch (const std::invalid_argument& e) {
        conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_json_reply(reply);
}

void rest_register_verbs(const std::vector<RestVerb>& verbs, std::vector<std::string>& paths) {
    for (const RestVerb& v : verbs) {
        if (!v.get) {
            rest_register_unique(v.path, v.post, paths);
            continue;
        }
        const RestGet get = v.get;
        rest_register_unique(
            v.path, [get](connectionInstance& conn) { rest_reply(conn, get, conn.get_query()); },
            paths);
    }
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

void run_workers(const std::vector<std::string>& names, const std::vector<std::vector<int>>& cpus,
                 const std::function<void(size_t)>& run) {
    std::vector<std::thread> threads;
    for (size_t k = 0; k < names.size(); ++k) {
        threads.emplace_back(run_row, std::cref(run), k);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int c : cpus[k])
            CPU_SET(c, &cpuset);
        const int err =
            pthread_setaffinity_np(threads.back().native_handle(), sizeof(cpu_set_t), &cpuset);
        if (err)
            ERROR_NON_OO("Failed to bind {:s} to its cpu_affinity, error code {:d}", names[k], err);
        pthread_setname_np(threads.back().native_handle(), string_tail(names[k], 15).c_str());
    }
    for (std::thread& t : threads)
        t.join();
}

} // namespace gnss
