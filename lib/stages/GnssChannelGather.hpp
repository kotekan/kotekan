/**
 * @file
 * @brief Gather scattered single-channel voltage streams into one contiguous
 *        multi-channel buffer for a centralized channelized search.
 *  - GnssChannelGather : public kotekan::Stage
 */

#ifndef GNSS_CHANNEL_GATHER_HPP
#define GNSS_CHANNEL_GATHER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssChannelGather
 * @brief Interleave N single-channel hop streams into one [hop][N] buffer.
 *
 * The inverse of @ref GnssSubbandSplit, and the transport landing point for the
 * unified (centralized) search: in the CHORD comb a carrier's covering channels
 * are scattered across nodes, each shipping its one channel's raw channelized
 * voltage here (via @ref bufferSend / @ref bufferRecv -- the *small* search data,
 * not the expanded P_c surface). This gathers the i-th frame of every input
 * (lock-step, same absolute hops -- they share one F-engine cadence) into a
 * contiguous @c [hop][n_in] frame that the existing @ref GnssChannelizedSearch
 * consumes unchanged, with @c channel_offset set to the first covering channel's
 * global frequency index. The fpga_seq reference is propagated from input 0.
 *
 * @buffer in_bufs  Per-channel voltage, each [hop] cfloat32 (one channel).
 * @buffer out_buf  Gathered voltage, [hop][n_in] cfloat32 (channel-minor).
 *
 * @author Keith Vanderlinde
 */
class GnssChannelGather : public kotekan::Stage {
public:
    GnssChannelGather(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~GnssChannelGather() override = default;
    void main_thread() override;

private:
    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
};

#endif // GNSS_CHANNEL_GATHER_HPP
