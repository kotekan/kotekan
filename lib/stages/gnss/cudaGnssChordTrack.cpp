#include "cudaGnssChordTrack.hpp"

#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

#include "fmt.hpp" // for format

#include <algorithm>  // for min
#include <functional> // for bind
#include <mutex>      // for mutex, unique_lock

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using nlohmann::json;

// The keys a row of a chains block must set itself; the command must not set them, nor the ones
// after them, which a row may set.
static const std::vector<std::string> ROW_KEYS = {"gnss_row",
                                                  "prns",
                                                  "signal",
                                                  "n_channels",
                                                  "channel_ids",
                                                  "f_offset_hz",
                                                  "dll_spacing",
                                                  "lanes_per_slot",
                                                  "seed_endpoint",
                                                  "set_trim_endpoint",
                                                  "get_prns_endpoint",
                                                  "set_prns_endpoint"};
static std::vector<std::string> row_only_keys() {
    std::vector<std::string> keys = ROW_KEYS;
    for (const char* k : {"prn_freq_offset_hz", "trim_endpoint", "num_synth", "synth_lane_base",
                          "gnss_local_channels", "gnss_ctl_name"})
        keys.push_back(k);
    return keys;
}

// The row named by gnss_row at @p path, which must not also set the keys the row replaces; or a
// row with no name.
static gnss_gpu::GpuRow inject_row(Config& config, const std::string& path) {
    if (!config.exists(path, "gnss_row"))
        return {};
    for (const char* k : {"num_synth", "synth_lane_base", "gnss_local_channels", "gnss_ctl_name"})
        if (config.exists(path, k))
            FATAL_ERROR_NON_OO("cudaGnssInject {:s}: {:s} comes from its gnss_rows row", path, k);
    const std::string name = config.get<std::string>(path, "gnss_row");
    for (const gnss_gpu::GpuRow& r : gnss_gpu::gpu_rows(config, path))
        if (r.name == name)
            return r;
    FATAL_ERROR_NON_OO("cudaGnssInject {:s}: gnss_rows has no row {:s}", path, name);
    return {};
}

// The engines of a hosted inject by broker chain, in row order.
using ChainEngines = std::vector<std::pair<std::string, gnss::TrackChain*>>;

// POST set_trim of several chains at once (see cudaGnssChordTrackState) at @p path.
static void set_trim_rows(connectionInstance& conn, json& request, const std::string& path,
                          const ChainEngines& engines) {
    const json& by_chain = request.at("chains");
    std::string have;
    for (const auto& e : engines)
        have += (have.empty() ? "" : ", ") + e.first;
    if (!by_chain.is_object()) {
        conn.send_error(fmt::format(fmt("{:s} takes {{\"chains\": {{<chain>: <request>, ...}}}}; "
                                        "its chains are {:s}"),
                                    path, have),
                        HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    std::string unknown;
    for (const auto& item : by_chain.items()) {
        bool found = false;
        for (const auto& e : engines)
            found = found || e.first == item.key();
        if (!found)
            unknown += (unknown.empty() ? "" : ", ") + item.key();
    }
    if (!unknown.empty()) {
        conn.send_error(
            fmt::format(fmt("{:s} has no chain {:s}; its chains are {:s}"), path, unknown, have),
            HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    std::vector<size_t> todo;
    for (size_t i = 0; i < engines.size(); ++i)
        if (by_chain.contains(engines[i].first))
            todo.push_back(i);
    std::vector<std::unique_lock<std::mutex>> locks;
    for (size_t i : todo)
        locks.emplace_back(engines[i].second->prn_mtx);
    std::vector<std::vector<std::pair<int, double>>> upd(engines.size());
    std::string refused;
    for (size_t i : todo) {
        const std::string err =
            engines[i].second->parse_trims(by_chain.at(engines[i].first), upd[i]);
        if (!err.empty())
            refused +=
                (refused.empty() ? "" : "; ") + engines[i].first + ": bad trim payload: " + err;
    }
    if (!refused.empty()) {
        conn.send_error(refused, HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    for (size_t i : todo)
        engines[i].second->apply_trims(upd[i]);
    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

cudaGnssChordTrackState::cudaGnssChordTrackState(Config& config, const std::string& unique_name,
                                                 bufferContainer& host_buffers,
                                                 cudaDeviceInterface& device) :
    cudaCommandState(config, unique_name, host_buffers, device) {
    const std::vector<gnss::HostRow> host =
        gnss::host_rows(config, unique_name, "_inject", ROW_KEYS, row_only_keys());
    if (host[0].key.empty()) {
        rows.push_back({host[0], inject_row(config, unique_name), nullptr});
    } else {
        std::vector<gnss_gpu::GpuRow> gpu;
        for (const gnss::HostRow& h : host)
            gpu.push_back(inject_row(config, h.cfg_path));
        for (const gnss_gpu::GpuRow& g : gnss_gpu::gpu_rows(config, unique_name)) {
            size_t found = host.size();
            for (size_t i = 0; i < host.size(); ++i)
                if (gpu[i].name == g.name) {
                    // Two engines would write one control block with one row tag.
                    if (found < host.size())
                        FATAL_ERROR_NON_OO("cudaGnssInject {:s}: rows {:s} and {:s} both take "
                                           "gnss_row {:s} (row tag {:#x})",
                                           unique_name, host[found].key, host[i].key, g.name,
                                           g.tag());
                    found = i;
                }
            if (found == host.size())
                FATAL_ERROR_NON_OO("cudaGnssInject {:s}: no row of chains takes gnss_row {:s}",
                                   unique_name, g.name);
            rows.push_back({host[found], g, nullptr});
        }
    }
    for (Row& r : rows)
        r.chain = std::make_unique<gnss::TrackChain>(config, r.host.cfg_path);
    std::vector<gnss::HostRow> row_hosts;
    std::vector<std::string> chains;
    std::vector<std::vector<gnss::RestVerb>> verbs;
    // The batch is bound to the engines, not to this state: the state is deleted through
    // cudaCommandState, whose destructor is not virtual, so its paths outlive it.
    ChainEngines engines;
    for (Row& r : rows) {
        row_hosts.push_back(r.host);
        chains.push_back(r.gpu.chain);
        verbs.push_back(r.chain->rest_verbs());
        engines.emplace_back(r.gpu.chain, r.chain.get());
    }
    // The path its rows' verbs are served at by chain: `/<half>_inject`.
    std::string host_path;
    if (hosted()) {
        // The rows of one GPU half: gnss<g>_<chain>, as the generator names them.
        for (const Row& r : rows) {
            const std::string& n = r.gpu.name;
            const size_t at = n.size() - std::min(n.size(), r.gpu.chain.size() + 1);
            if (at == 0 || n.compare(at, std::string::npos, "_" + r.gpu.chain) != 0)
                FATAL_ERROR_NON_OO("cudaGnssInject {:s}: gnss_rows row {:s} is not named "
                                   "<half>_{:s}",
                                   unique_name, n, r.gpu.chain);
            const std::string h = "/" + n.substr(0, at) + "_inject";
            if (!host_path.empty() && h != host_path)
                FATAL_ERROR_NON_OO("cudaGnssInject {:s}: rows of two halves, {:s} and {:s}",
                                   unique_name, host_path, h);
            host_path = h;
        }
    }
    using namespace std::placeholders;
    gnss::rest_register_rows(
        host_path.empty() ? unique_name : host_path, row_hosts, chains, verbs, _rest_paths,
        {{"set_trim", std::bind(set_trim_rows, _1, _2, host_path + "/set_trim", engines)}});
}

cudaGnssChordTrackState::~cudaGnssChordTrackState() {
    std::vector<gnss::HostRow> host;
    for (const Row& r : rows)
        host.push_back(r.host);
    gnss::release_host(host, _rest_paths);
}
