/**
 * @file
 * @brief Integrate 21-cm absorber data over frames.
 *  - HFBAccumulate : public kotekan::Stage
 */

#ifndef HFB_ACCUMULATE_STAGE
#define HFB_ACCUMULATE_STAGE

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "dataset.hpp"         // for dset_id_t
#include "HFBFrameView.hpp" // for HFBFrameView

#include <stdint.h> // for uint32_t, int64_t
#include <string>   // for string

/**
 * @class HFBAccumulate
 * @brief Post-processing engine for output of the CHIME/HFB kernel,
 *        integrates data over 80 frames to create 10s worth of data.
 *        num_beams * num_sub_freq = 1024 * 128
 *
 * This engine sums CHIME/HFB data from 1 GPU stream in each CHIME node,
 * which are stored in the output buffer.
 * Note: _num_frames_to_integrate cannot go below 16 frames as _num_frames_to_integrate cannot be
 * lower than the max_frames_missing
 *
 * The output of this stage is written to a raw file where each chunk is
 * the metadata followed by the frame and indexed by frequency ID.
 * This raw file is then transposed and compressed into a structured
 * HDF5 format by gossec.
 *
 * @par Buffers
 * @buffer hfb_input_buffer Kotekan buffer feeding data from any GPU.
 *     @buffer_format Array of @c floats
 * @buffer hfb_out_buf Kotekan buffer that will be populated with integrated data.
 *     @buffer_format Array of @c floats
 *
 * @conf   num_frames_to_integrate  Int. No. of frames to integrate over.
 * @conf   num_frb_total_beams      Int. No. of total FRB beams (should be 1024).
 * @conf   factor_upchan            Int. Upchannelise factor (should be 128).
 * @conf   samples_per_data_set     Int. The number of samples each GPU buffer has
 *                                  been integrated for.
 * @conf   good_samples_threshold   Float. Required fraction of good samples in
 *                                  integration before it is recorded.
 *
 * @author James Willis
 *
 */

class HFBAccumulate : public kotekan::Stage {
public:
    /// Constructor.
    HFBAccumulate(kotekan::Config& config_, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~HFBAccumulate();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;
    /// Copy the first frame of the integration
    void init_first_frame(HFBFrameView& in_frame, float* sum_data);
    /// Add a frame to the integration
    void integrate_frame(HFBFrameView& in_frame, float* sum_data);
    /// Normalise frame after integration has been completed
    void normalise_frame(HFBFrameView& in_frame, float* sum_data);

private:
    Buffer* in_buf;
    Buffer* cls_buf;
    Buffer* out_buf;

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
    dset_id_t ds_id;
};

#endif
