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

#include <string>

/**
 * @class processFRBFeedGains
 * @brief Merge, upchannelize, and apply weights to gain files.
 *
 * Applies the same processing as the parent, but sets the buffer metadata
 * expected by `CHIMEFRBBeamformer_chime_U16`.
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

    // config parameters required for metadata
    uint32_t num_polarizations;

    bool frb1_swap_MN;
    int num_dishes_M;
    int num_dishes_N;
};

#endif
