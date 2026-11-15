/**
 * @file
 * @brief The gnss_rows list of a GPU half: each chain's place in the half's shared tiles and
 *        control-block arrays.
 */

#ifndef GNSS_GPU_ROWS_HPP
#define GNSS_GPU_ROWS_HPP

#include "Config.hpp"         // for Config
#include "gnssGpuChain.hpp"   // for ctl_bytes
#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

#include "json.hpp" // for json

#include <cstddef> // for size_t
#include <cstdint> // for int32_t, int64_t
#include <set>     // for set
#include <string>  // for string
#include <vector>  // for vector

namespace gnss_gpu {

/// The row tag cudaGnssInject writes into FrameHdr::_pad1 of its block in a GPU half's shared
/// control array: a magic, then the row's chain id, its index in gnss_rows and its fg_offset,
/// 16 bits each. A reader compares it with its own row before it reads the block.
constexpr int64_t ROW_TAG_MAGIC = 0x5257;
constexpr int64_t row_tag(int chain_id, int row, int fg_offset) {
    return (ROW_TAG_MAGIC << 48) | ((int64_t)(chain_id & 0xFFFF) << 32)
           | ((int64_t)(row & 0xFFFF) << 16) | (int64_t)(fg_offset & 0xFFFF);
}

/**
 * One row of a GPU half: a chain's channels and lanes, and where its tiles and control block sit
 * in the half's arrays. The tiles are [Tc][Fg_total][Tile][16][16][2] int32, a row's channels at
 * [fg_offset, fg_offset + channels); its control block is ctl_bytes at ctl_offset.
 */
struct GpuRow {
    std::string name; ///< the chain's stage prefix, e.g. gnss0_gps_l5
    std::string chain;
    int chain_id = 0;
    int row = 0;                   ///< position in gnss_rows
    std::vector<int32_t> channels; ///< local channels, in the chain's order
    int lane_base = 0, lanes = 0;  ///< synth lanes [lane_base, lane_base + lanes)
    int fg_offset = 0;             ///< first channel's position on the tiles' channel axis
    size_t ctl_offset = 0, ctl_bytes = 0;
    int n_prn = 0;

    int64_t tag() const {
        return row_tag(chain_id, row, fg_offset);
    }
};

/// A row from its gnss_rows entry; every key is required.
inline GpuRow gpu_row(const nlohmann::json& j) {
    GpuRow r;
    r.name = j.at("name").get<std::string>();
    r.chain = j.at("chain").get<std::string>();
    r.chain_id = j.at("chain_id").get<int>();
    r.row = j.at("row").get<int>();
    r.channels = j.at("channels").get<std::vector<int32_t>>();
    r.lane_base = j.at("lane_base").get<int>();
    r.lanes = j.at("lanes").get<int>();
    r.fg_offset = j.at("fg_offset").get<int>();
    r.ctl_offset = j.at("ctl_offset").get<size_t>();
    r.ctl_bytes = j.at("ctl_bytes").get<size_t>();
    r.n_prn = j.at("n_prn").get<int>();
    return r;
}

/**
 * The gnss_rows list at @p path. FATAL unless the rows are the arrays' concatenation in list
 * order: row i has row i, its fg_offset and ctl_offset follow the rows before it, its ctl_bytes
 * is that of its n_prn and channels, and names, chain ids and tag fields are unique and fit the
 * tag.
 */
inline std::vector<GpuRow> gpu_rows(kotekan::Config& config, const std::string& path) {
    const std::vector<nlohmann::json> list =
        config.get<std::vector<nlohmann::json>>(path, "gnss_rows");
    if (list.empty())
        FATAL_ERROR_NON_OO("{:s}: gnss_rows is empty", path);
    std::vector<GpuRow> rows;
    std::set<std::string> names;
    std::set<int> ids;
    int fg = 0;
    size_t ctl = 0;
    for (const nlohmann::json& j : list) {
        const GpuRow r = gpu_row(j);
        const int n_chan = (int)r.channels.size();
        if (r.row != (int)rows.size() || r.fg_offset != fg || r.ctl_offset != ctl)
            FATAL_ERROR_NON_OO("{:s}: gnss_rows entry {:d} ({:s}) has row {:d}, fg_offset {:d} and "
                               "ctl_offset {:d}; the rows before it put it at row {:d}, fg_offset "
                               "{:d} and ctl_offset {:d}",
                               path, rows.size(), r.name, r.row, r.fg_offset, r.ctl_offset,
                               rows.size(), fg, ctl);
        if (n_chan < 1 || r.n_prn < 1 || r.ctl_bytes != ctl_bytes(r.n_prn, n_chan))
            FATAL_ERROR_NON_OO("{:s}: gnss_rows {:s}: ctl_bytes {:d} for {:d} PRN x {:d} channels, "
                               "need {:d}",
                               path, r.name, r.ctl_bytes, r.n_prn, n_chan,
                               ctl_bytes(r.n_prn, n_chan));
        if (!names.insert(r.name).second || !ids.insert(r.chain_id).second)
            FATAL_ERROR_NON_OO("{:s}: gnss_rows {:s}: its name or chain_id {:d} is not unique",
                               path, r.name, r.chain_id);
        if (r.chain_id < 0 || r.chain_id > 0xFFFF || fg + n_chan > 0xFFFF)
            FATAL_ERROR_NON_OO("{:s}: gnss_rows {:s}: chain_id {:d} or fg_offset {:d} does not fit "
                               "the row tag's 16 bits",
                               path, r.name, r.chain_id, r.fg_offset);
        fg += n_chan;
        ctl += r.ctl_bytes;
        rows.push_back(r);
    }
    return rows;
}

} // namespace gnss_gpu

#endif // GNSS_GPU_ROWS_HPP
