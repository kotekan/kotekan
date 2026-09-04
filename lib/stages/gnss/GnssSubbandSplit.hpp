/**
 * @file
 * @brief Fan a PFB-channelized voltage stream out into per-subband channel slices.
 *  - GnssSubbandSplit : public kotekan::Stage
 */

#ifndef GNSS_SUBBAND_SPLIT_HPP
#define GNSS_SUBBAND_SPLIT_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssSubbandSplit
 * @brief Fan the PFB F-engine's channelized stream out into per-subband channel
 *        slices for independent search/track.
 *
 * Consumes the @c [hop][channel] complex-float stream from @ref fftwEngine and,
 * for each configured subband, copies a contiguous channel range into its own
 * output buffer. This is the data-distribution seam of the distributed-band
 * pipeline -- on CHORD each slice would be owned by a different node (shipped via
 * bufferSend/Recv); here all slices stay local. Each subband is then searched and
 * tracked independently, and the per-subband despread products are coherently
 * recombined downstream (see @ref GnssCoherentCombiner): because the despread is a
 * sum over channels, partitioning the carrier's covering channels across subbands
 * and recombining reproduces the full-band measurement exactly.
 *
 * The per-subband channel ranges are given as two parallel arrays @c subband_lo /
 * @c subband_hi (hi exclusive), one entry per @c out_bufs buffer.
 *
 * @conf spectrum_length Int. Channels per hop in the input (the F-engine N).
 * @conf subband_lo      Array<Int>. Low channel (inclusive) of each subband.
 * @conf subband_hi      Array<Int>. High channel (exclusive) of each subband.
 *
 * @buffer in_buf   Channelized voltages, [hop][spectrum_length] cfloat32.
 * @buffer out_bufs One per subband, [hop][hi_i - lo_i] cfloat32.
 *
 * @author Keith Vanderlinde
 */
class GnssSubbandSplit : public kotekan::Stage {
public:
    GnssSubbandSplit(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    void main_thread() override;

private:
    Buffer* in_buf;
    std::vector<Buffer*> out_bufs;
    int _N;                    ///< input channels per hop
    std::vector<int> _lo, _hi; ///< per-subband channel range [lo, hi)
};

#endif // GNSS_SUBBAND_SPLIT_HPP
