/**
 * @file
 * @brief Takes the RFI mask and the lost packet mask buffers and updates the metadata
 *  - rfiUpdateMetadata : public kotekan::Stage
 */

#ifndef RFI_UPDATE_METADATA_H
#define RFI_UPDATE_METADATA_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @class rfiUpdateMetadata
 * @brief Takes the RFI mask and the lost packet mask buffers and updates the metadata
 *
 * @par Buffers
 * @buffer rfi_mask_buf Mask of RFI flagged samples
 *     @buffer_format Array of @c uint8
 *     @buffer_metadata chimeMetadata
 * @buffer lost_samples_buf Mask of lost samples from packet loss/errors
 *     @buffer_format Array of @c uint8
 *     @buffer_metadata none
 * @buffer gpu_correlation_buf GPU N2 output buffer.  co-producer on this buffer
 *     @buffer_format Array of @c uint
 *     @buffer_metadata chimeMetadata
 *
 * @conf   sk_step               Int   The number of samples zeroed at once as RFI
 * @conf   num_sub_frames        Int   The number sub frames in the GPU stage
 * @conf   samples_per_data_set  Int   Number of samples in the GPU stage input frame
 *
 * @author Andre Renard
 *
 */
class rfiUpdateMetadata : public kotekan::Stage {
public:
    /// Constructor.
    rfiUpdateMetadata(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);

    /// Destructor.
    virtual ~rfiUpdateMetadata();

    /// Primary loop, which waits on input frames, integrates, and dumps to output.
    void main_thread() override;

private:
    /// The mask of which samples were flagged as RFI
    Buffer* rfi_mask_buf;

    /// The mask of which samples were lost to packet loss/network errors.
    Buffer* lost_samples_buf;

    /// The GPU N2 data, we are a producer of this data
    Buffer* gpu_correlation_buf;

    /// The number of N2 output frames for each input RFI/packet loss frame.
    uint32_t _num_sub_frames;

    /// The number of samples in each N2 output frame
    uint32_t _sub_frame_samples;

    /// Length of the subframe RFI mask
    uint32_t _sub_frame_mask_len;

    /// The number of samples zeroed at once as RFI
    uint32_t _sk_step;
};

#endif
