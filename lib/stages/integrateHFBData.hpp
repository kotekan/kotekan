/**
 * @file
 * @brief Integrate 21-cm data over frames.
 *  - integrateHFBData : public kotekan::Stage
 */

#ifndef INTEGRATE_HFB_DATA_PROCESS
#define INTEGRATE_HFB_DATA_PROCESS

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t, int64_t
#include <string>   // for string

/**
 * @class integrateHFBData
 * @brief Post-processing engine for output of the CHIME/HFB kernel,
 *        integrates data over 80 frames to create 10s worth of data.
 *        num_beams * num_sub_freq = 1024 * 128
 *
 * This engine sums CHIME/HFB data from 1 GPU stream in each CHIME node,
 * which are stored in the output buffer.
 * Note: _num_frames_to_integrate cannot go below 16 frames as _num_frames_to_integrate cannot be
 * lower than the max_frames_missing
 *
 * @par Buffers
 * @buffer hfb_input_buffer Kotekan buffer feeding data from any GPU.
 *     @buffer_format Array of @c floats
 * @buffer hfb_out_buf Kotekan buffer that will be populated with integrated data.
 *     @buffer_format Array of @c floats
 *
 * @conf   num_frames_to_integrate Int. No. of frames to integrate over.
 * @conf   num_frb_total_beams  Int. No. of total FRB beams (should be 1024).
 * @conf   factor_upchan  Int. Upchannelise factor (should be 128).
 *
 * @author James Willis
 *
 */

class integrateHFBData : public kotekan::Stage {
public:
    /// Constructor.
    integrateHFBData(kotekan::Config& config_, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~integrateHFBData();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;
    /// Copy the first frame of the integration
    void initFirstFrame(float* input_data, float* sum_data, const uint32_t in_buffer_ID);
    /// Add a frame to the integration
    void integrateFrame(float* input_data, float* sum_data, const uint32_t in_buffer_ID);
    /// Normalise frame after integration has been completed
    float normaliseFrame(float* sum_data, const uint32_t in_buffer_ID);

private:
    struct Buffer* in_buf;
    struct Buffer* compressed_lost_samples_buf;
    struct Buffer* out_buf;

    /// Config variables
    uint32_t _num_frames_to_integrate;
    uint32_t _num_frb_total_beams;
    uint32_t _factor_upchan;
    uint32_t _samples_per_data_set;
    float _good_samples_threshold;

    /// Stage variables
    uint32_t total_timesamples;
    uint32_t total_lost_timesamples;
    uint32_t frame;
    int64_t fpga_seq_num;
    int64_t fpga_seq_num_end;
};

#endif
