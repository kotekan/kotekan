// Compare the FRB2 beamforming weights produced by two stages (calcFRB2Weights on the CPU
// and cudaCalcFRB2Weights on the GPU); see config/ci-tests/gpu_batch/test_calc_frb2_weights.yaml
//
// The weights are a stream, one frame per lifetime. This compares the first `num_frames` frames of
// both streams, including their `fpga_seq_num`, which must advance by one lifetime per frame.

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for float16_t
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "errors.h"            // for TEST_PASSED
#include "kotekanLogging.hpp"  // for DEBUG, INFO, FATAL_ERROR

#include "fmt.hpp" // for compile_string_to_view

#include <cassert>    // for assert
#include <cmath>      // for fabs, isnan
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int64_t
#include <functional> // for function
#include <memory>     // for shared_ptr
#include <string>     // for allocator, string

class testCalcFRB2Weights : public kotekan::Stage {
    // Maximum acceptable absolute difference. The CPU (`cosf(pi * x)`) and GPU (`cospif(x)`)
    // weights differ at the ~1e-4 level before float16 rounding; the weights themselves are
    // O(1), where one float16 ulp is ~1e-3.
    const float tolerance = config.get_default<float>(unique_name, "tolerance", 2.0e-3f);

    // Number of frames to compare
    const int num_frames = config.get_default<int>(unique_name, "num_frames", 1);

    Buffer* const weights_a_buffer;
    Buffer* const weights_b_buffer;

public:
    testCalcFRB2Weights(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        weights_a_buffer(get_buffer("weights_a")), weights_b_buffer(get_buffer("weights_b"))
    //
    {
        assert(weights_a_buffer);
        assert(weights_b_buffer);
        weights_a_buffer->register_consumer(unique_name);
        weights_b_buffer->register_consumer(unique_name);
    }

    virtual ~testCalcFRB2Weights() {}

    void main_thread() override {
        if (stop_thread)
            return;

        std::int64_t seq0 = -1;
        for (int frame_index = 0; frame_index < num_frames; ++frame_index)
            if (!compare_frame(frame_index, seq0))
                return;

        TEST_PASSED();
    }

private:
    // Returns false if the pipeline is shutting down
    bool compare_frame(const int frame_index, std::int64_t& seq0) {
        const int frame_id_a = frame_index % weights_a_buffer->num_frames;
        const int frame_id_b = frame_index % weights_b_buffer->num_frames;

        DEBUG("[{:s}/{:d}] Waiting for buffer...", weights_a_buffer->buffer_name, frame_index);
        const float16_t* const weights_a = static_cast<const float16_t*>(
            static_cast<void*>(weights_a_buffer->wait_for_full_frame(unique_name, frame_id_a)));
        if (!weights_a)
            return false;

        DEBUG("[{:s}/{:d}] Waiting for buffer...", weights_b_buffer->buffer_name, frame_index);
        const float16_t* const weights_b = static_cast<const float16_t*>(
            static_cast<void*>(weights_b_buffer->wait_for_full_frame(unique_name, frame_id_b)));
        if (!weights_b)
            return false;

        // Both streams must describe the same time span
        const std::shared_ptr<const chordMetadata> meta_a =
            get_chord_metadata(weights_a_buffer, frame_id_a);
        const std::shared_ptr<const chordMetadata> meta_b =
            get_chord_metadata(weights_b_buffer, frame_id_b);
        if (meta_a->get_fpga_seq_num() != meta_b->get_fpga_seq_num()
            || meta_a->get_time_downsampling_fpga() != meta_b->get_time_downsampling_fpga())
            FATAL_ERROR("Frame {:d}: {:s} has fpga_seq_num {:d} and time_downsampling_fpga {:d}, "
                        "but {:s} has {:d} and {:d}",
                        frame_index, weights_a_buffer->buffer_name, meta_a->get_fpga_seq_num(),
                        meta_a->get_time_downsampling_fpga(), weights_b_buffer->buffer_name,
                        meta_b->get_fpga_seq_num(), meta_b->get_time_downsampling_fpga());
        if (frame_index == 0)
            seq0 = meta_a->get_fpga_seq_num();
        const std::int64_t expected_seq_num =
            seq0 + frame_index * std::int64_t(meta_a->get_time_downsampling_fpga());
        if (meta_a->get_fpga_seq_num() != expected_seq_num)
            FATAL_ERROR("Frame {:d} has fpga_seq_num {:d}, expected {:d}", frame_index,
                        meta_a->get_fpga_seq_num(), expected_seq_num);

        if (weights_a_buffer->frame_size != weights_b_buffer->frame_size)
            FATAL_ERROR("Frame sizes differ: {:s} has {:d} bytes, {:s} has {:d} bytes",
                        weights_a_buffer->buffer_name, weights_a_buffer->frame_size,
                        weights_b_buffer->buffer_name, weights_b_buffer->frame_size);
        const std::ptrdiff_t size = weights_a_buffer->frame_size / sizeof(float16_t);

        float max_difference = 0;
        std::ptrdiff_t num_errors = 0;
        for (std::ptrdiff_t i = 0; i < size; ++i) {
            const float a = float(weights_a[i]);
            const float b = float(weights_b[i]);
            if (std::isnan(a) && std::isnan(b))
                continue;
            const float difference = std::fabs(a - b);
            if (!(difference <= tolerance)) {
                ++num_errors;
                if (num_errors <= 10)
                    INFO("Mismatch at index {:d}: {:f} vs {:f}", i, a, b);
            }
            max_difference = std::fmax(max_difference, difference);
        }
        INFO("Frame {:d} (fpga_seq_num {:d}): compared {:d} weights, max difference {:g} "
             "(tolerance {:g})",
             frame_index, meta_a->get_fpga_seq_num(), size, max_difference, tolerance);
        if (num_errors > 0)
            FATAL_ERROR("{:d} of {:d} weights differ by more than {:g}", num_errors, size,
                        tolerance);

        weights_a_buffer->mark_frame_empty(unique_name, frame_id_a);
        weights_b_buffer->mark_frame_empty(unique_name, frame_id_b);

        return true;
    }
};

REGISTER_KOTEKAN_STAGE(testCalcFRB2Weights);
