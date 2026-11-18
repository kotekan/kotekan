/**
 * @file
 * @brief The state cudaGnssInject's instances share.
 *  - cudaGnssChordTrackState : public cudaCommandState
 */

#ifndef CUDA_GNSS_CHORD_TRACK_HPP
#define CUDA_GNSS_CHORD_TRACK_HPP

#include "Config.hpp"              // for Config
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommandState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "gnssGpuRows.hpp"         // for GpuRow
#include "gnssHostRows.hpp"        // for HostRow
#include "gnssTrackChain.hpp"      // for TrackChain

#include <memory> // for unique_ptr
#include <string> // for string
#include <vector> // for vector

/**
 * @class cudaGnssChordTrackState
 * @brief The gnss::TrackChain engines a cudaGnssInject's instances share: the command's own, or
 *        one per row of its @c chains block.
 *
 * Without a @c chains block the command is one chain, configured at its own path. With one,
 * each entry `chains: {<key>: {...}}` is a chain configured at `<command>/chains/<key>`
 * (gnss::host_rows), and the entries are the GPU half's gnss_rows: each row of gnss_rows is the
 * gnss_row of exactly one entry, and the chains are kept in gnss_rows order, the order the
 * command packs them in. Each entry sets the keys that differ per chain (the TrackChain keys
 * prns, signal, n_channels, channel_ids, f_offset_hz, dll_spacing and the four endpoints;
 * cudaGnssInject's gnss_row and lanes_per_slot; optionally prn_freq_offset_hz and
 * trim_endpoint) and the command none of them. Two rows that share a channel must not share a
 * lane.
 *
 * Each chain's REST endpoints are registered at the paths its config names. A path registered
 * twice is FATAL. With a @c chains block the command also serves them by chain at
 * `/<half>_inject/<verb>` (gnss::rest_register_rows), <half> the name its rows' gnss_rows
 * entries share before their chain. There, set_trim also takes
 * `{"chains": {"<chain>": <a set_trim request>, ...}}`: every named chain's prn_mtx is held, in
 * row order, while each request is checked as its own path checks it and, if none is refused,
 * applied; otherwise nothing is applied. The reply is its own path's, or the first refusal,
 * naming its chain.
 */
class cudaGnssChordTrackState : public cudaCommandState {
public:
    cudaGnssChordTrackState(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaGnssChordTrackState();

    /// One chain of the command.
    struct Row {
        /// Its name (the command's unique_name, or `/<key>_inject`) and config path.
        gnss::HostRow host;
        /// Its gnss_rows entry, or a row with no name for a chain without gnss_row.
        gnss_gpu::GpuRow gpu;
        std::unique_ptr<gnss::TrackChain> chain;
    };
    std::vector<Row> rows;

    /// True with a chains block.
    bool hosted() const {
        return !rows[0].host.key.empty();
    }

private:
    /// POST set_trim of several chains at once (see the class).
    void set_trim_rows(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// The path its rows' verbs are served at by chain: `/<half>_inject`.
    std::string _host;
    std::vector<std::string> _rest_paths;
};

#endif // CUDA_GNSS_CHORD_TRACK_HPP
