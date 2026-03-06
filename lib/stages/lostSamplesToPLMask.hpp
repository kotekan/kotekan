#ifndef LOST_SAMPLES_TO_PL_MASK_HPP
#define LOST_SAMPLES_TO_PL_MASK_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t, uint32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @brief Merges and expands CHIME lost_samples buffers (1 bytes per sample for 4
 * frequencies) into a CHORD style package loss mask (1 bit per downsampled
 * time, per 4 frequencies, per polarization, per dish).
 *
 * @par Buffers
 * @buffer pl_mask_buf Kotekan buffer for package loss mask
 *     @buffer_format [time / 2 / 64][freq / 4][polr][dish / 8][time / 2 % 64]
 *     @buffer_metadata chordMetadata
 * @buffer lost_samples_buf Array of flags which indicate if a sample in a given location is lost
 *     @buffer_format Array of flags uint8_t flags which are either 0 (unset) or 1 (set)
 *     @buffer_metadata chimeMetadata
 *
 * @conf  in_lost_sample_buffers    Buffers to hold the lost samples buffer. For
 * example: in_lost_sample_buffers:
 *                                        - lost_samples_buffer_0
 *                                        - lost_samples_buffer_1
 *                                        - lost_samples_buffer_2
 *                                        - lost_samples_buffer_3
 * @conf  num_polarizations         Number of polarizations present in data set.
 * @conf  num_dishes                Number of dishes present in data set.
 *
 * @author Roland Haas
 */
class lostSamplesToPLMask : public kotekan::Stage {
public:
    /// Standard constructor
    lostSamplesToPLMask(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~lostSamplesToPLMask();

    /// Main thead which zeros the data from the lost_samples_buf
    void main_thread() override;

private:
    /// The number of polarizations in data set
    const int num_polarizations;

    /// The number of dishes in data set
    const int num_dishes;

    /// The buffer with the package loss data
    Buffer* pl_mask_buf;

    /// The buffers with the array of flags indicating lost data.
    std::vector<Buffer*> lost_samples_bufs;

    /// Current ID for pl_mask_buf
    int32_t pl_mask_buf_frame_id = 0;

    /// Current
    int32_t lost_samples_buf_frame_id = 0;
};

#endif /* LOST_SAMPLES_TO_PL_MASK_HPP */
