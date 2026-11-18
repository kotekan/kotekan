#include "gnssHostRows.hpp"

#include "kotekanLogging.hpp"    // for FATAL_ERROR_NON_OO, ERROR_NON_OO, FatalError
#include "prometheusMetrics.hpp" // for Metrics
#include "util.h"                // for string_tail

#include "fmt.hpp" // for format

#include <algorithm> // for find
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

/// The rows of a host that have one verb, as (chain, verb), in row order.
using VerbRows = std::vector<std::pair<std::string, RestVerb>>;

std::string chain_list(const VerbRows& rows) {
    std::string out;
    for (const auto& r : rows)
        out += (out.empty() ? "" : ", ") + r.first;
    return out;
}

const RestVerb* verb_of(const VerbRows& rows, const std::string& chain) {
    for (const auto& r : rows)
        if (r.first == chain)
            return &r.second;
    return nullptr;
}

void no_chain(connectionInstance& conn, const std::string& path, const std::string& chain,
              const VerbRows& rows) {
    conn.send_error(fmt::format(fmt("{:s} has no chain {:s}; its chains are {:s}"), path, chain,
                                chain_list(rows)),
                    HTTP_RESPONSE::BAD_REQUEST);
}

void get_rows(connectionInstance& conn, const std::string& path, const VerbRows& rows) {
    RestQuery query = conn.get_query();
    const auto it = query.find("chain");
    if (it == query.end()) {
        json reply = json::object();
        for (const auto& [chain, v] : rows) {
            try {
                reply[chain] = v.get(query);
            } catch (const std::invalid_argument& e) {
                conn.send_error(chain + ": " + e.what(), HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
        }
        conn.send_json_reply(reply);
        return;
    }
    const RestVerb* v = verb_of(rows, it->second);
    if (v == nullptr) {
        no_chain(conn, path, it->second, rows);
        return;
    }
    query.erase(it);
    rest_reply(conn, v->get, query);
}

void post_rows(connectionInstance& conn, json& request, const std::string& path,
               const VerbRows& rows, const RestPost& batch) {
    if (batch && request.is_object() && request.size() == 1 && request.contains("chains")) {
        batch(conn, request);
        return;
    }
    if (!request.is_object() || request.size() != 2 || !request.contains("body")
        || !request.contains("chain") || !request.at("chain").is_string()) {
        conn.send_error(
            fmt::format(fmt("{:s} takes {{\"chain\": <chain>, \"body\": <request>}}{:s}; its "
                            "chains are {:s}"),
                        path, batch ? " or {\"chains\": {<chain>: <request>, ...}}" : "",
                        chain_list(rows)),
            HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const std::string chain = request.at("chain").get<std::string>();
    const RestVerb* v = verb_of(rows, chain);
    if (v == nullptr) {
        no_chain(conn, path, chain, rows);
        return;
    }
    v->post(conn, request.at("body"));
}

/// Register @p verbs at their own paths and add the paths to @p paths; with @p hits, each path
/// counts the requests that reach it there.
void register_verbs(const std::vector<RestVerb>& verbs, std::vector<std::string>& paths,
                    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>* hits) {
    for (const RestVerb& v : verbs) {
        kotekan::prometheus::Counter* hit = nullptr;
        if (hits) {
            hit = &hits->labels({v.path});
            // Shown from the start, so an unused path reads 0 rather than absent.
            hit->inc(0);
        }
        if (!v.get) {
            const RestPost post = v.post;
            if (!hit)
                rest_register_unique(v.path, post, paths);
            else
                rest_register_unique(
                    v.path,
                    [hit, post](connectionInstance& conn, json& request) {
                        hit->inc();
                        post(conn, request);
                    },
                    paths);
            continue;
        }
        const RestGet get = v.get;
        rest_register_unique(
            v.path,
            [hit, get](connectionInstance& conn) {
                if (hit)
                    hit->inc();
                rest_reply(conn, get, conn.get_query());
            },
            paths);
    }
}

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

void rest_register_chains(const std::string& unique_name, const std::vector<HostRow>& rows,
                          const std::vector<std::string>& chains, std::vector<std::string>& paths) {
    if (rows[0].key.empty())
        return;
    json reply = json::array();
    for (size_t i = 0; i < rows.size(); ++i) {
        json row = {{"name", rows[i].key}, {"stage", rows[i].name}};
        if (!chains[i].empty())
            row["chain"] = chains[i];
        reply.push_back(row);
    }
    rest_register_unique(
        unique_name + "/chains", [reply](connectionInstance& conn) { conn.send_json_reply(reply); },
        paths);
}

void rest_register_rows(const std::string& host, const std::vector<HostRow>& rows,
                        const std::vector<std::string>& chains,
                        const std::vector<std::vector<RestVerb>>& verbs,
                        std::vector<std::string>& paths,
                        const std::map<std::string, RestPost>& batch) {
    const bool hosted = !rows[0].key.empty();
    const bool by_chain = hosted && std::find(chains.begin(), chains.end(), "") == chains.end();
    if (hosted && !by_chain)
        INFO_NON_OO("{:s}: a row names no chain, so its verbs are served only at the rows' own "
                    "paths",
                    host);
    // Each verb with the rows that have it, in the order the rows first have them.
    std::vector<std::pair<std::string, VerbRows>> by_verb;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!by_chain) {
            register_verbs(verbs[i], paths, nullptr);
            continue;
        }
        for (size_t j = 0; j < i; ++j)
            if (chains[j] == chains[i])
                FATAL_ERROR_NON_OO("{:s}: rows {:s} and {:s} are both chain {:s}", host,
                                   rows[j].key, rows[i].key, chains[i]);
        register_verbs(verbs[i], paths,
                       &kotekan::prometheus::Metrics::instance().add_counter(
                           "kotekan_gnss_rest_legacy_hits_total", rows[i].name, {"path"}));
        for (const RestVerb& v : verbs[i]) {
            auto it = by_verb.begin();
            while (it != by_verb.end() && it->first != v.verb)
                ++it;
            if (it == by_verb.end())
                it = by_verb.insert(it, {v.verb, {}});
            it->second.emplace_back(chains[i], v);
        }
    }
    for (const auto& [verb, vrows] : by_verb) {
        const std::string path = host + "/" + verb;
        const VerbRows vr = vrows;
        if (vr[0].second.get) {
            rest_register_unique(
                path, [path, vr](connectionInstance& conn) { get_rows(conn, path, vr); }, paths);
            continue;
        }
        const auto b = batch.find(verb);
        const RestPost bp = b == batch.end() ? nullptr : b->second;
        rest_register_unique(
            path,
            [path, vr, bp](connectionInstance& conn, json& request) {
                post_rows(conn, request, path, vr, bp);
            },
            paths);
    }
    rest_register_chains(host, rows, chains, paths);
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
