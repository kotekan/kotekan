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
 * The gather for the distributed search. Each CHORD node holds a mod-8 comb of the science
 * band, split across its two GPUs as mod-16 combs, so one GPU's feed carries about 1/16 of a
 * GNSS signal's covering channels (7 of ~106, -11.8 dB), a loss no local integration recovers.
 * Each feed therefore ships one element's voltage, which is cheaper to move than search
 * products, and one search runs over the union. This stage takes one input per feed and emits
 * the union.
 *
 * Alignment is the core of the job. The send and receive legs run with drop_frames, so each
 * input can miss frames independently, and merging the i-th frame of every input would combine
 * different epochs after the first drop, with no error. Inputs are aligned on
 * GnssChanMetadata::sample_seq, which every feed derives from the F-engine's global
 * fpga_seq_num, so equal sample_seq means the same instant. The stage advances whichever inputs
 * are behind and emits a frame only once every present input holds the same epoch; an epoch
 * missing on one present input is skipped on all of them. An input that delivers nothing for
 * input_timeout_s is treated as absent and its channels are zeroed until it returns.
 *
 * Output columns are the inputs' channels concatenated in input order; the search takes an
 * explicit channel_ids list, so no sort order is assumed.
 *
 * @conf in_bufs     list of cfloat32 [hop][chan_i] buffers (GnssChordDequantize outputs)
 * @conf out_buf     cfloat32 [hop][sum chan_i]
 * @conf in_channels channels per input, in in_bufs order
 * @conf samples_per_data_set  hops per frame (all inputs and the output)
 * @conf input_timeout_s  seconds an input may deliver nothing before it is treated as absent
 *                        (default 5; must be > 0)
 *
 * @metric kotekan_gnss_merge_skipped_frames_total  input frames released unmerged to align the
 *                                                  inputs, labelled by input
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
    /// Seconds to wait on one input before declaring it absent and zeroing its channels.
    /// Without it, a config declaring more inputs than there are running nodes stalls the merge
    /// with no error (see main_thread).
    double _input_timeout_s = 5.0;
    /// Seconds to wait on an input already declared absent. A short poll, because the reacquire
    /// sweep visits the inputs in turn: at the full timeout, n absent inputs would make the
    /// output period n x _input_timeout_s. Long enough to pick up a frame already queued, which
    /// is all a rejoining sender needs, and short enough that absent inputs cost almost nothing.
    static constexpr double absent_poll_s = 0.002;
};

#endif // GNSS_CHAN_ALIGN_MERGE_HPP
