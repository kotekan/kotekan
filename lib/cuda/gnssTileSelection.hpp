#ifndef GNSS_TILE_SELECTION_HPP
#define GNSS_TILE_SELECTION_HPP

#include <algorithm> // for find
#include <cstddef>   // for size_t
#include <cstdint>   // for int32_t
#include <stdexcept> // for runtime_error
#include <string>    // for to_string
#include <vector>    // for vector

/**
 * @file
 * @brief cudaCorrelatorDual's tile-selection list, header-only and free of CUDA so a CPU test
 *        can build it. @c Sel is int2 in cudaCorrelatorDual; any aggregate of two int32 (x =
 *        slice, y = offset) gives the same list.
 */

// One gather: the tiles of `chans` (a subset of the comb) for the synth rows holding lanes
// [lane_base, lane_base + n_lanes). Several chains can share one correlator pass: each gather
// holds only its own chain's rows, so its buffer has the layout a one-chain correlator with
// num_synth = n_lanes would produce. The order is the tile list in cudaCorrelatorDual.hpp, and
// GnssN2RecordAssemble recomputes it.
template<class Sel>
std::vector<Sel> build_tile_selection(const std::vector<std::int32_t>& comb,
                                      const std::vector<std::int32_t>& chans, int num_elements,
                                      const std::vector<int>& live_cols, int lane_base, int n_lanes,
                                      bool gather_aa, bool gather_bb) {
    std::vector<Sel> sel;
    const int na16 = num_elements / 16;
    const int row_lo = na16 + lane_base / 16;
    const int row_hi = na16 + (lane_base + n_lanes) / 16;
    for (size_t ci = 0; ci < chans.size(); ci++) {
        // The kernel's freq map writes slice k for comb[k], so the gather indexes by k, not
        // by the channel number.
        const auto it = std::find(comb.begin(), comb.end(), chans[ci]);
        if (it == comb.end())
            throw std::runtime_error("cudaCorrelatorDual: a gather asks for local channel "
                                     + std::to_string(chans[ci])
                                     + ", which is not in gnss_local_channels");
        const std::int32_t f = (std::int32_t)(it - comb.begin());
        // Mixed (synth row x live antenna column) first. GnssN2RecordAssemble reads the tile
        // of row ihi and list position slot at mix_k = (ihi - row_lo)*n_live_cols + slot. The
        // slot is the position in live_cols, not the column, so the gather order defines the
        // record's element axis and the consumer needs no knowledge of which columns are live.
        // Blocks appended after this one do not move the mixed tiles.
        for (int ihi = row_lo; ihi < row_hi; ihi++)
            for (size_t k = 0; k < live_cols.size(); k++)
                sel.push_back({f, 512 * (ihi * (ihi + 1) / 2 + live_cols[k])});
        // The tile count is one contract stated in three places: here, GnssN2RecordAssemble's
        // _n_tile (mixed only), and config/gen_chord_gnss_config.py's n2_tiles_per_chan, which
        // sizes the tiles buffer. A mismatch is fatal: GnssN2RecordAssemble checks the frame
        // size at startup, and gpuDeviceInterface's size check catches this side when
        // cudaOutputData first copies the buffer out.
        //
        // AA (antenna x antenna, the live N^2) next, only on request: the lower triangle over
        // positions in live_cols, row-major with k2 <= k1, so the (k1, k2) tile sits at
        // k1*(k1+1)/2 + k2 in this block. These are the triangle's tiles only if live_cols is
        // ascending. The kernel writes the AA block only when BLOCK_MASK_AA is in its mask; the
        // constructor adds it iff a gather wants AA.
        if (gather_aa)
            for (size_t k1 = 0; k1 < live_cols.size(); k1++)
                for (size_t k2 = 0; k2 <= k1; k2++)
                    sel.push_back(
                        {f, 512 * (live_cols[k1] * (live_cols[k1] + 1) / 2 + live_cols[k2])});
        // BB (synth x synth) last, only on request: the lower triangle over this gather's synth
        // tile rows, (ihi - row_lo, jhi - row_lo) at k1*(k1+1)/2 + k2 in this block. The kernel
        // computes it regardless; only the visibility capture reads it (36 tiles per channel
        // at 128 lanes).
        if (gather_bb)
            for (int ihi = row_lo; ihi < row_hi; ihi++)
                for (int jhi = row_lo; jhi <= ihi; jhi++)
                    sel.push_back({f, 512 * (ihi * (ihi + 1) / 2 + jhi)});
    }
    return sel;
}

#endif // GNSS_TILE_SELECTION_HPP
