/**
 * @file
 * @brief The rows of a GNSS host stage, and its REST registration.
 */
#ifndef GNSS_HOST_ROWS_HPP
#define GNSS_HOST_ROWS_HPP

#include "Config.hpp"     // for Config
#include "restServer.hpp" // for connectionInstance

#include "json.hpp" // for json

#include <functional> // for function
#include <map>        // for map
#include <string>     // for string
#include <vector>     // for vector

namespace gnss {

/// The query of a REST request (connectionInstance::get_query).
using RestQuery = std::map<std::string, std::string>;
/// A GET's reply, from the request's query. It throws std::invalid_argument to refuse the
/// request, which is then answered 400 with the exception's message.
using RestGet = std::function<nlohmann::json(const RestQuery&)>;
using RestPost = std::function<void(kotekan::connectionInstance&, nlohmann::json&)>;

/// One REST endpoint of a row's chain: its verb, its own path, and its GET or POST.
struct RestVerb {
    std::string verb;
    std::string path;
    RestGet get;
    RestPost post;
};

/// Answer @p conn with @p get's reply to @p query, or with its refusal.
void rest_reply(kotekan::connectionInstance& conn, const RestGet& get, const RestQuery& query);

/// One chain hosted by a GNSS host stage.
struct HostRow {
    /// Its key in the host's @c chains block; empty for a host without one.
    std::string key;
    /// The config path its chain reads.
    std::string cfg_path;
    /// Its name: the REST prefix, metric label and buffer client name of its chain.
    std::string name;
};

/**
 * @brief The rows of host stage @p unique_name.
 *
 * Without a @c chains key the host has one row, with cfg_path and name its unique_name. With
 * one, each entry `chains: {<key>: {...}}` is a row with cfg_path `<unique_name>/chains/<key>`,
 * named `/<key><suffix>`, the name of the one-row host it replaces. A row reads the keys it does
 * not set from the host block, so each key in @p row_keys must be set in every row itself, and a
 * key in @p row_only_keys must not be set on the host block. A row name may be held by one host
 * at a time, until release_host. Any violation is FATAL.
 */
std::vector<HostRow> host_rows(kotekan::Config& config, const std::string& unique_name,
                               const std::string& suffix, const std::vector<std::string>& row_keys,
                               const std::vector<std::string>& row_only_keys);

/// A thread of a host stage, which runs its rows in order and reads their inputs as one consumer.
struct HostWorker {
    /// Its consumer name.
    std::string name;
    /// The cores it is bound to; empty for a row's own thread, on the stage's cores.
    std::vector<int> cpus;
    /// Its rows, as indices into the host's rows, in the order it runs them.
    std::vector<size_t> rows;
};

/**
 * @brief The workers of host stage @p unique_name, whose rows are @p rows.
 *
 * Without a @c workers key, one per row, named for the row and with no cores. With one, worker k
 * of `workers: [{rows: [row keys], cpu_affinity: [cores]}, ...]`, named `<unique_name>/w<k>`.
 * Workers need a @c chains block, and every row must be in exactly one worker. Any violation is
 * FATAL.
 */
std::vector<HostWorker> host_workers(kotekan::Config& config, const std::string& unique_name,
                                     const std::vector<HostRow>& rows);

/// Register a GET callback at @p path and add @p path to @p paths, FATAL if @p path already has
/// a callback.
void rest_register_unique(const std::string& path,
                          std::function<void(kotekan::connectionInstance&)> callback,
                          std::vector<std::string>& paths);
/// Register a POST callback at @p path and add @p path to @p paths, FATAL if @p path already has
/// a callback.
void rest_register_unique(
    const std::string& path,
    std::function<void(kotekan::connectionInstance&, nlohmann::json&)> callback,
    std::vector<std::string>& paths);

/// With a @c chains block, register GET `<unique_name>/chains`, which lists @p rows, and add it
/// to @p paths.
void rest_register_chains(const std::string& unique_name, const std::vector<HostRow>& rows,
                          std::vector<std::string>& paths);

/// Register @p verbs at their own paths and add the paths to @p paths, FATAL if one already has a
/// callback.
void rest_register_verbs(const std::vector<RestVerb>& verbs, std::vector<std::string>& paths);

/// Release what a host claimed, for its destructor: remove its REST @p paths, waiting for any
/// request in flight on them, then the metrics and row names of its @p rows, so that a new host
/// may take them.
void release_host(const std::vector<HostRow>& rows, const std::vector<std::string>& paths);

/// Call @p run for each of @p n_rows rows, row 0 on the calling thread and every other row on a
/// thread of its own bound to @p cpus at its start, and return once every call has.
void run_rows(size_t n_rows, const std::vector<int>& cpus, const std::function<void(size_t)>& run);

/// Call @p run for each worker k on a thread of its own, named for the tail of @p names[k] and
/// bound to @p cpus[k] at its start, and return once every call has.
void run_workers(const std::vector<std::string>& names, const std::vector<std::vector<int>>& cpus,
                 const std::function<void(size_t)>& run);

} // namespace gnss

#endif // GNSS_HOST_ROWS_HPP
