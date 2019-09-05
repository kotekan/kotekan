/**
 * @file
 * @brief FRB beamformer CPU verification
 */

#ifndef TEST_DATA_GEN_MISSING_FRAMES_HPP
#define TEST_DATA_GEN_MISSING_FRAMES_HPP

#include "Stage.hpp"
#include "buffer.h"

/**
 * @class testDataGenMissingFrames
 * @brief CPU verification for FRB beamformer
 *
 * This is CPU only pipeline to verify the FRB beamformer.
 * This CPU pipeline does the equivalent of:
 *   - hsaBeamformReorder
 *   - name: hsaBeamformKernel
 *   - name: hsaBeamformTranspose
 *   - name: hsaBeamformUpchan
 *
 * @author Cherry Ng
 **/

class testDataGenMissingFrames : public kotekan::Stage {
public:
    /// Constructor
    testDataGenMissingFrames(kotekan::Config& config, const string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~testDataGenMissingFrames();
    void main_thread() override;

private:
    /// Initializes internal variables from config, allocates reorder_map, gain, get metadata buffer
    struct Buffer* input_buf;
    struct Buffer* output_buf;

    /// Number of elements, should be 2048
    uint32_t _num_elements;
    /// Number of time samples, should be a multiple of 3x128 for FRB, standard ops is 49152
    uint32_t _samples_per_data_set;
    /// Index to create a missing frames 
    uint32_t _missing_frame_index;

};

#endif
