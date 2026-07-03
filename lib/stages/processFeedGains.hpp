/**
 * @file
 * @brief merge, upchannelize, and apply weights to gain files
 *  - processFeedGains : public kotekan::Stage
 */

#ifndef PROCESS_FEED_GAINS_HPP
#define PROCESS_FEED_GAINS_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <queue>    // for queue
#include <stdint.h> // for int32_t, uint8_t, int16_t, uint32_t
#include <string>   // for string
#include <vector>   // for vector

using std::queue;
using std::vector;

/**
 * @class processFeedGains
 * @brief Merge, upchannelize, and apply weights to gain files.
 *
 * Gains are upchannelized by a factor of `upchan_factor` in frequency. The
 * current implementation does nothing more than copy each coarse frequency gain
 * `upchan_factor` times.
 *
 * For beamformers which support multiple beams, this stage expects to receive a
 * separate buffer for each beam, as implemented in `loadFeedGains`.
 *
 * In order to be compatible with existing copy-to-GPU methods, this stage will
 * continuously emit kotekan buffers containing the current gains.
 *
 * @par Buffers
 * @buffer gain_buf_<n> Each buffer is an array of arbitrary gains for a given beam
 *     @buffer_format float32
 *     @buffer_shape [freq][element][real/imag]
 *     @buffer_metadata chordMetadata
 * @buffer processed_gains_buf Array of processed gains
 *     @buffer_format float32
 *     @buffer_shape [beams][freq][upchan_factor][element][real/imag]
 *     @buffer_metadata chordMetadata
 *
 * @conf   num_elements                          Int. Number of elements
 * @conf   num_beams                             Int. Number of beams with unique gain files
 * @conf   num_local_freq                        Int. Number of frequencies per node
 * @conf   upchan_factor                         Int. Frequency upchannelization factor.
 *
 * @author Liam Gray
 *
 */
class processFeedGains : public kotekan::Stage {
public:
    /// Constructor.
    processFeedGains(kotekan::Config& config_, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);

    /// Destructor.
    ~processFeedGains();

    void main_thread() override;

private:
    /// Copy gain from a buffer and apply an upchannelization algorithm to gains for a single
    /// beam. Default duplicates the gain for each subfrequency.
    void copy_upchannelize(float* frame, size_t beam_id);

    /// Implement logic to upchannelize a single fine frequency, given a coarse frequency.
    /// Default will just duplicate the coarse frequency gain `N` times.
    virtual void copy_upchannelize_f(const float* src_f, float* dst_f, size_t fid);

    /// Create a frame desc for the output buffer. Default creates a frame desc
    /// containing [beam, freq, subfreq, element, ReIm] axes.
    virtual void set_frame_desc(Buffer* buf);

    std::vector<Buffer*> gain_buffers;
    Buffer* in_mask_buf;
    Buffer* out_buf;

    /// Number of elements, should be 2048
    uint32_t num_elements;
    /// Number of pulsar beams, should be 10
    uint32_t num_beams;
    /// Number of frequencies per node
    uint32_t num_local_freq;
    /// Frequency upchannelization factor
    uint32_t upchan_factor;

    /// Number of elements in the output buffer
    uint32_t out_num_values;

    /// Fixed buffers used to hold gains separately from the kotekan buffers
    std::vector<float> gain_store_buf;
    std::vector<uint8_t> mask_store_buf;

    /// Store gain upchannelization factors
    std::vector<int> freq_upchan_factor;
    std::vector<int> freq_upchan_index;
};


#endif
