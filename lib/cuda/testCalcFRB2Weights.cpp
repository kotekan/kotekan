// Compare the FRB2 beamforming weights produced by two stages (calcFRB2Weights on the CPU
// and cudaCalcFRB2Weights on the GPU); see config/ci-tests/gpu_batch/test_calc_frb2_weights.yaml

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for float16_t
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "errors.h"            // for TEST_PASSED
#include "kotekanLogging.hpp"  // for DEBUG, INFO, FATAL_ERROR

#include "fmt.hpp" // for compile_string_to_view

#include <cassert>    // for assert
#include <cmath>      // for fabs, isnan
#include <cstddef>    // for ptrdiff_t
#include <functional> // for function
#include <string>     // for allocator, string

class testCalcFRB2Weights : public kotekan::Stage {
    // Maximum acceptable absolute difference. The CPU (`cosf(pi * x)`) and GPU (`cospif(x)`)
    // weights differ at the ~1e-4 level before float16 rounding; the weights themselves are
    // O(1), where one float16 ulp is ~1e-3.
    const float tolerance = config.get_default<float>(unique_name, "tolerance", 2.0e-3f);

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
        const int frame_id = 0;

        if (stop_thread)
            return;

        DEBUG("[{:s}/{:d}] Waiting for buffer...", weights_a_buffer->buffer_name, frame_id);
        const float16_t* const weights_a = static_cast<const float16_t*>(
            static_cast<void*>(weights_a_buffer->wait_for_full_frame(unique_name, frame_id)));
        if (!weights_a)
            return;

        DEBUG("[{:s}/{:d}] Waiting for buffer...", weights_b_buffer->buffer_name, frame_id);
        const float16_t* const weights_b = static_cast<const float16_t*>(
            static_cast<void*>(weights_b_buffer->wait_for_full_frame(unique_name, frame_id)));
        if (!weights_b)
            return;

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
        INFO("Compared {:d} weights, max difference {:g} (tolerance {:g})", size, max_difference,
             tolerance);
        if (num_errors > 0)
            FATAL_ERROR("{:d} of {:d} weights differ by more than {:g}", num_errors, size,
                        tolerance);

        weights_a_buffer->mark_frame_empty(unique_name, frame_id);
        weights_b_buffer->mark_frame_empty(unique_name, frame_id);

        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testCalcFRB2Weights);
