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
#include <string>     // for string
#include <vector>     // for vector

namespace gnss {

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

/// Release what a host claimed, for its destructor: remove its REST @p paths, waiting for any
/// request in flight on them, then the metrics and row names of its @p rows, so that a new host
/// may take them.
void release_host(const std::vector<HostRow>& rows, const std::vector<std::string>& paths);

/// Call @p run for each of @p n_rows rows, row 0 on the calling thread and every other row on a
/// thread of its own bound to @p cpus at its start, and return once every call has.
void run_rows(size_t n_rows, const std::vector<int>& cpus, const std::function<void(size_t)>& run);

} // namespace gnss

#endif // GNSS_HOST_ROWS_HPP
