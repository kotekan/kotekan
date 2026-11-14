/**
 * @file
 * @brief The N^2 tiles and cudaGnssInject's control block to the epl frame (gnssGpuChain.hpp).
 *
 * The transform GnssN2RecordAssemble runs per frame; see its class note for the scale, lanes
 * and orientation.
 */

#ifndef GNSS_EPL_ASSEMBLE_HPP
#define GNSS_EPL_ASSEMBLE_HPP

#include <cstddef> // for size_t
#include <cstdint> // for int32_t, uint8_t

namespace gnss_gpu {

/// cudaGnssInject's control block: the epl layout without corr.
struct CtlView {
    const uint8_t* ptr;
};

/// The gathered mixed tiles, [Tc][channel][n_tile][16][16][2] int32. The tiles are compacted:
/// tile k of channel c lives at slice(r, c) + k*512, not at its full-triangle offset.
struct TilesView {
    const int32_t* ptr;
    size_t rec_stride; ///< int32 values from one record's tiles to the next
    int fg_offset;     ///< position of channel 0 in the gather's channel axis
    int n_chan;
    int n_tile; ///< mixed tiles per channel
    const int32_t* slice(int r, int c) const {
        return ptr + (size_t)r * rec_stride + (size_t)(fg_offset + c) * n_tile * 512;
    }
};

/**
 * @brief Write the epl frame for one control block and its tiles.
 *
 * The caller checks the control block's shape first: n_prn and lanes_per_slot as configured,
 * n_chan equal to tiles.n_chan, and n_rec at most MAX_REC.
 *
 * @param out_bytes at least frame_bytes(n_prn, n_chan, ROWS_PLAIN, n_live).
 * @param n_live    live antennas the tiles cover.
 * @return the active (record, PRN) job groups written; the header's n_jobs is that times
 *         ROWS_PLAIN.
 */
int assemble_epl(const CtlView& ctl, const TilesView& tiles, uint8_t* out, size_t out_bytes,
                 int n_prn, int lanes_per_slot, int num_elements, int n_live, int hops_per_record);

} // namespace gnss_gpu

#endif // GNSS_EPL_ASSEMBLE_HPP
