/**
 * @file
 * @brief merge, upchannelize, and apply weights to FRB beamformer gain files
 *  - processFRBFeedGains : public processFeedGains
 */

#ifndef PROCESS_FRB_FEED_GAINS_HPP
#define PROCESS_FRB_FEED_GAINS_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "processFeedGains.hpp"

#include <cstdint>
#include <string>

/**
 * @class processFRBFeedGains
 * @brief Merge, upchannelize, and apply weights to gain files.
 *
 * Applies the same processing as the parent, but sets the buffer metadata
 * expected by `CHIMEFRBBeamformer_chime_U16_K4` and
 * `CHIMEFRBBeamformer_chime_U16_K8`; the gain buffer is the same for both input
 * bit depths.
 *
 * The output frames have a leading length-1 `TW` axis whose `dimscaling` is
 * `frb1_phase_lifetime_in_samples`. That must equal the lifetime of the bad feed
 * mask frames, which clock the output.
 *
 * @conf frb1_phase_lifetime_in_samples Int. How many FPGA samples one output frame covers.
 *
 * @author Liam Gray
 *
 */
class processFRBFeedGains : public processFeedGains {
public:
    processFRBFeedGains(kotekan::Config& config_, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);

private:
    void copy_upchannelize_f(const float* src_f, float16_t* dst_f, size_t fid) override;
    void set_frame_desc(Buffer* buf) override;
    /// Warn if the gains can make the FRB1 kernel overflow Float16
    void check_gains(const float16_t* frame) override;

    // config parameters required for metadata
    uint32_t num_polarizations;
    std::int64_t frb1_phase_lifetime_in_samples;

    bool frb1_swap_MN;
    int num_dishes_M;
    int num_dishes_N;
};

#endif
