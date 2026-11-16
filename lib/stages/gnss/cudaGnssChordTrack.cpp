#include "cudaGnssChordTrack.hpp"

#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

using kotekan::bufferContainer;
using kotekan::Config;

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
        r.chain = std::make_unique<gnss::TrackChain>(config, r.host.cfg_path, _rest_paths);
}

cudaGnssChordTrackState::~cudaGnssChordTrackState() {
    std::vector<gnss::HostRow> host;
    for (const Row& r : rows)
        host.push_back(r.host);
    gnss::release_host(host, _rest_paths);
}
