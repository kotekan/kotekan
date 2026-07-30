/**
 * @file
 * @brief Align N channelized GNSS streams on their absolute sample counter and merge
 *        them into one wider-comb stream.
 *  - GnssChanAlignMerge : public kotekan::Stage
 */

#ifndef GNSS_CHAN_ALIGN_MERGE_HPP
#define GNSS_CHAN_ALIGN_MERGE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <string>
#include <vector>

/**
 * @class GnssChanAlignMerge
 * @brief N x [hop][chan_i] cfloat32 -> [hop][sum chan_i] cfloat32, aligned on sample_seq.
 *
 * THE AGGREGATION SEAM of the distributed search. Each CHORD node holds a mod-8 comb of the
 * science band, so one node's covering channels for a GNSS carrier are 7 of the ~106 that
 * exist: -11.8 dB, unrecoverable by any amount of local integration. The design therefore
 * ships raw voltage from every node to ONE search over the gathered union (the search
 * products are ~38x larger than the voltage, so voltage is the cheap thing to move). This
 * stage is the gather: one input per node feed, one output holding the union comb.
 *
 * WHY ALIGNMENT IS THE WHOLE JOB. The per-node send legs run with drop_frames: true --
 * acquisition must never back-pressure a node's science chain -- so each input can be
 * missing arbitrary frames, independently. Concatenating columns frame-by-frame would
 * silently combine different epochs after the first drop anywhere, and a cross-node phase
 * error is precisely the thing the union search cannot tolerate. Alignment is on
 * GnssChanMetadata::sample_seq, which all feeds inherit from the F-engine's GLOBAL
 * fpga_seq_num (GPS-disciplined, shared by every node), so equality of sample_seq IS
 * simultaneity. The stage advances whichever inputs are behind and emits only hops that
 * exist on ALL inputs; a frame missing on any input drops that epoch for everyone
 * (gnss_merge_dropped_frames_total counts per input). With 24-deep node-side buffers the
 * steady state is lockstep; the machinery exists for the transients.
 *
 * Output column order is the concatenation of the inputs' channel lists in input order --
 * the search takes an explicit channel_ids list, so no sortedness is assumed anywhere.
 *
 * WHY NOT @ref GnssChannelGather. That stage is the same idea for the airspy distributed
 * band, and it is INDEX-LOCKSTEP: it merges the i-th frame of every input without ever
 * reading sample_seq. On an in-process pipeline with no drops that is sound; behind
 * bufferRecv + drop_frames senders it silently combines DIFFERENT EPOCHS from the first
 * drop anywhere, forever after -- no error, just a noise-only search. It also takes one
 * channel per input where the CHORD feeds are 7-channel blocks. Kept untouched for the
 * airspy chain; this stage exists because the network seam makes alignment the hard
 * requirement, not the transpose.
 *
 * @conf in_bufs     list of cfloat32 [hop][chan_i] buffers (GnssChordDequantize outputs)
 * @conf out_buf     cfloat32 [hop][sum chan_i]
 * @conf in_channels channels per input, in in_bufs order
 * @conf samples_per_data_set  hops per frame (all inputs and the output)
 *
 * @author Keith Vanderlinde
 */
class GnssChanAlignMerge : public kotekan::Stage {
public:
    GnssChanAlignMerge(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& buffer_container);
    ~GnssChanAlignMerge() override = default;
    void main_thread() override;

private:
    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
    std::vector<int> _in_chans; ///< channels per input
    int _out_chan = 0;          ///< sum of _in_chans
    int _n_hops = 0;
};

#endif // GNSS_CHAN_ALIGN_MERGE_HPP
