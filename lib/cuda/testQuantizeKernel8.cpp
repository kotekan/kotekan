// Test 8-bit FRB beam quantizer for CHIME, using the float16 FRB beamformer and CHIME's
// sending-to-Bonsai network format

#include <algorithm>                // for fill, fill_n
#include <cassert>                  // for assert
#include <cfloat>                   // for FLT_MAX
#include <cmath>                    // for fabs, isfinite, isnan, fmax, fmin
#include <cstdint>                  // for uint8_t
#include <cstdlib>                  // for abort
#include <string>                   // for allocator, string
#include <vector>                   // for vector
#include <functional>               // for function

#include "Config.hpp"               // for Config
#include "DataType.hpp"             // for float16_t
#include "Stage.hpp"                // for Stage
#include "StageFactory.hpp"         // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp"      // for bufferContainer
#include "cudaQuantizeKernel8.hpp"  // for cpu_quantize8, gpu_quantize8
#include "cudaUtils.hpp"            // for CHECK_CUDA_ERROR
#include "cuda_fp16.h"              // for __half, __half::operator float
#include "cuda_runtime.h"           // for cudaMalloc
#include "cuda_runtime_api.h"       // for cudaMemcpy
#include "driver_types.h"           // for cudaMemcpyKind
#include "errors.h"                 // for TEST_PASSED
#include "fmt.hpp"                  // for compile_string_to_view
#include "kotekanLogging.hpp"       // for FATAL_ERROR, INFO

class testQuantizeKernel8 : public kotekan::Stage {
public:
    testQuantizeKernel8(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    virtual ~testQuantizeKernel8() {}

    void main_thread() override {
        const float16_t fpoison(0.0f / 0.0f);
        const std::uint8_t ipoison(0);

        const float epsilon = 1.0e-3; // We're using float16

        // Input array layout
        const int in_ntimes = 256;
        const int in_nfreqs = 256; // 16 coarse channels, upchannelized by 16
        const int in_nbeams = 1024;

        // Output array layout
        // The output is quantized in "chunks". Each chunk has a different offset and scale.
        const int out_ntimes_chunk = 16;
        const int out_nfreqs_chunk = 16;
        // Chunks are combined into packets.
        const int out_nfreqs_packet = 4;
        const int out_nbeams_packet = 8;
        // There are several packets.
        const int out_ntimes_outer = 16;
        const int out_nfreqs_outer = 4;
        const int out_nbeams_outer = 128;

        static_assert(out_ntimes_chunk * out_ntimes_outer == in_ntimes);
        static_assert(out_nfreqs_chunk * out_nfreqs_packet * out_nfreqs_outer == in_nfreqs);
        static_assert(out_nbeams_packet * out_nbeams_outer == in_nbeams);

        ////////////////////////////////////////////////////////////////////////////////

        // layout: input[beam, freq, time]
        std::vector<float16_t> input(in_nbeams * in_nfreqs * in_ntimes);
        const auto inputat = [&](int beam, int freq, int time) -> float16_t& {
            assert(0 <= beam && beam < in_nbeams);
            assert(0 <= freq && freq < in_nfreqs);
            assert(0 <= time && time < in_ntimes);
            return input.at(time + in_ntimes * (freq + in_nfreqs * beam));
        };
        std::fill(input.begin(), input.end(), 1);
        for (int beam = 0; beam < in_nbeams; ++beam)
            for (int freq = 0; freq < in_nfreqs; ++freq)
                for (int time = 0; time < in_ntimes; ++time)
                    inputat(beam, freq, time) = 0;
        if (std::find(input.begin(), input.end(), float16_t(1)) != input.end())
            FATAL_ERROR("input array indexing error");

        // layout: offsetscale[out_nbeams_outer, out_nfreqs_outer, out_ntimes_outer,
        //                     out_nbeams_packet, out_nfreqs_packet]
        std::vector<float16_t> offsetscale(2 * out_nbeams_outer * out_nfreqs_outer
                                           * out_ntimes_outer * out_nbeams_packet
                                           * out_nfreqs_packet);
        const auto offsetscaleat = [&](int beam_outer, int freq_outer, int time_outer,
                                       int beam_packet, int freq_packet, int offsca) -> float16_t& {
            assert(0 <= beam_outer && beam_outer < out_nbeams_outer);
            assert(0 <= freq_outer && freq_outer < out_nfreqs_outer);
            assert(0 <= time_outer && time_outer < out_ntimes_outer);
            assert(0 <= beam_packet && beam_packet < out_nbeams_packet);
            assert(0 <= freq_packet && freq_packet < out_nfreqs_packet);
            assert(0 <= offsca && offsca < 2);
            return offsetscale.at(
                offsca
                + 2
                      * (freq_packet
                         + out_nfreqs_packet
                               * (beam_packet
                                  + out_nbeams_packet
                                        * (time_outer
                                           + out_ntimes_outer
                                                 * (freq_outer + out_nfreqs_outer * beam_outer)))));
        };
        std::fill(offsetscale.begin(), offsetscale.end(), 1);
        for (int beam_outer = 0; beam_outer < out_nbeams_outer; ++beam_outer)
            for (int freq_outer = 0; freq_outer < out_nfreqs_outer; ++freq_outer)
                for (int time_outer = 0; time_outer < out_ntimes_outer; ++time_outer)
                    for (int beam_packet = 0; beam_packet < out_nbeams_packet; ++beam_packet)
                        for (int freq_packet = 0; freq_packet < out_nfreqs_packet; ++freq_packet) {
                            offsetscaleat(beam_outer, freq_outer, time_outer, beam_packet,
                                          freq_packet, 0) = 0;
                            offsetscaleat(beam_outer, freq_outer, time_outer, beam_packet,
                                          freq_packet, 1) = 0;
                        }
        if (std::find(offsetscale.begin(), offsetscale.end(), float16_t(1)) != offsetscale.end())
            FATAL_ERROR("offsetscale array indexing error");

        // layout: beams[out_nbeams_outer, out_nfreqs_outer, out_ntimes_outer,
        //               out_nbeams_packet, out_nfreqs_packet,
        //               out_nfreqs_chunk, out_ntimes_chunk]
        std::vector<std::uint8_t> beams(out_nbeams_outer * out_nfreqs_outer * out_ntimes_outer
                                        * out_nbeams_packet * out_nfreqs_packet * out_nfreqs_chunk
                                        * out_ntimes_chunk);
        const auto beamsat = [&](int beam_outer, int freq_outer, int time_outer, int beam_packet,
                                 int freq_packet, int freq_chunk, int time_chunk) -> std::uint8_t& {
            assert(0 <= beam_outer && beam_outer < out_nbeams_outer);
            assert(0 <= freq_outer && freq_outer < out_nfreqs_outer);
            assert(0 <= time_outer && time_outer < out_ntimes_outer);
            assert(0 <= beam_packet && beam_packet < out_nbeams_packet);
            assert(0 <= freq_packet && freq_packet < out_nfreqs_packet);
            assert(0 <= freq_chunk && freq_chunk < out_nfreqs_chunk);
            assert(0 <= time_chunk && time_chunk < out_ntimes_chunk);
            return beams.at(
                time_chunk
                + out_ntimes_chunk
                      * (freq_chunk
                         + out_nfreqs_chunk
                               * (freq_packet
                                  + out_nfreqs_packet
                                        * (beam_packet
                                           + out_nbeams_packet
                                                 * (time_outer
                                                    + out_ntimes_outer
                                                          * (freq_outer
                                                             + out_nfreqs_outer * beam_outer))))));
        };
        std::fill(beams.begin(), beams.end(), 1);
        for (int beam_outer = 0; beam_outer < out_nbeams_outer; ++beam_outer)
            for (int freq_outer = 0; freq_outer < out_nfreqs_outer; ++freq_outer)
                for (int time_outer = 0; time_outer < out_ntimes_outer; ++time_outer)
                    for (int beam_packet = 0; beam_packet < out_nbeams_packet; ++beam_packet)
                        for (int freq_packet = 0; freq_packet < out_nfreqs_packet; ++freq_packet)
                            for (int freq_chunk = 0; freq_chunk < out_nfreqs_chunk; ++freq_chunk)
                                for (int time_chunk = 0; time_chunk < out_ntimes_chunk;
                                     ++time_chunk)
                                    beamsat(beam_outer, freq_outer, time_outer, beam_packet,
                                            freq_packet, freq_chunk, time_chunk) = 0;
        if (std::find(beams.begin(), beams.end(), 1) != beams.end())
            FATAL_ERROR("beams array indexing error");

        ////////////////////////////////////////////////////////////////////////////////

        // Invent some data
        const auto f = [&](const int beam, const int freq, const int time) -> float {
            const float x = (time + 0.5) / in_ntimes;
            switch ((beam + freq) % 11) {
                case 0:
                    return 0.0f;
                case 1:
                    return 0.1f * (beam % 64);
                case 2:
                    return time % 2;
                case 3:
                    return x;
                case 4:
                    return 0.5f * x + 0.5f * x * x * x;
                case 5:
                    return 0.5f * x + 0.5f * x * x * x + 10 * (time % 23 == 0);
                case 6:
                    return 10000.0f * (time % 512); // large but not infinity
                case 7:
                    return 100000.0f * (time % 512); // some values are infinity
                case 8:
                    return (time % 512) / 10000.0f; // small, but 1/x is less than infinity
                case 9:
                    return (time % 512) / 100000.0f; // small, and some 1/x are infinity
                case 10:
                    return 0.5f * x + 0.5f * x * x * x + (time % 23 == 0 ? 0.0f / 0.0f : 0.0f);
            }
            std::abort();
        };

        ////////////////////////////////////////////////////////////////////////////////

        // Set input
        for (int beam = 0; beam < in_nbeams; ++beam) {
            for (int freq = 0; freq < in_nfreqs; ++freq) {
                for (int time = 0; time < in_ntimes; ++time) {
                    const float x = f(beam, freq, time);
                    inputat(beam, freq, time) = float16_t(x);
                }
            }
        }

        // A function to check the result
        const auto check_result = [&]() {
            using std::fabs, std::fmin, std::fmax, std::isfinite, std::isnan;

            // We love ourselves a good indentation
            for (int beam_outer = 0; beam_outer < out_nbeams_outer; ++beam_outer) {
                for (int freq_outer = 0; freq_outer < out_nfreqs_outer; ++freq_outer) {
                    for (int time_outer = 0; time_outer < out_ntimes_outer; ++time_outer) {
                        for (int beam_packet = 0; beam_packet < out_nbeams_packet; ++beam_packet) {
                            for (int freq_packet = 0; freq_packet < out_nfreqs_packet;
                                 ++freq_packet) {
                                const int beam = beam_packet + out_nbeams_packet * beam_outer;

                                // Calculate expected result
                                //
                                // When comparing to the expected result, two things can go wrong:
                                // 1. The scale can be zero, or can be close to zero
                                //    (`scale_may_collapse`)
                                // 2. The input can contain large numbers, or infinities, or
                                //    not-a-numbers (`may_overflow`)
                                //
                                // We need to handle both cases, and we need to be lenient when
                                // comparing when this happens. For example, there might be an
                                // overflow on the GPU, but not in the CPU implementation. This
                                // should not be flagged as error.
                                float minval = +FLT_MAX, maxval = -FLT_MAX;
                                bool may_overflow = false;
                                for (int freq_chunk = 0; freq_chunk < out_nfreqs_chunk;
                                     ++freq_chunk) {
                                    for (int time_chunk = 0; time_chunk < out_ntimes_chunk;
                                         ++time_chunk) {
                                        const int freq =
                                            freq_chunk
                                            + out_nfreqs_chunk
                                                  * (freq_packet + out_nfreqs_packet * freq_outer);
                                        const int time = time_chunk + out_ntimes_chunk * time_outer;
                                        const float x = f(beam, freq, time);
                                        minval = isnan(minval) || isnan(x) ? 0.0f / 0.0f
                                                                           : fmin(minval, x);
                                        maxval = isnan(maxval) || isnan(x) ? 0.0f / 0.0f
                                                                           : fmax(maxval, x);
                                        // Allow the respective float16 calculations to overflow
                                        may_overflow |= !isfinite(x) || fabs(x) >= 2048.0f;
                                    }
                                }
                                may_overflow |= !isfinite(minval) || fabs(minval) >= 2048.0f
                                                || !isfinite(maxval) || fabs(maxval) >= 2048.0f;
                                // Allow the scale to be inaccurate when all inputs are close
                                // together
                                const bool scale_may_collapse =
                                    fabs(maxval - minval) < 10 * epsilon;
                                const int uint8_range = 254; // avoid 0 and 255
                                float expected_scale = (maxval - minval) / uint8_range;
                                float expected_offset = minval - 0.5f * expected_scale;
                                if (!isfinite(expected_offset) || !isfinite(expected_scale)) {
                                    may_overflow = true;
                                    expected_offset = 0.0f;
                                    expected_scale = 0.0f;
                                }

                                const float offset =
                                    offsetscaleat(beam_outer, freq_outer, time_outer, beam_packet,
                                                  freq_packet, 0);
                                const float scale =
                                    offsetscaleat(beam_outer, freq_outer, time_outer, beam_packet,
                                                  freq_packet, 1);
                                if (!isfinite(offset))
                                    FATAL_ERROR("Found non-finite offset {}", offset);

                                // Check relative error of offset
                                bool offset_is_good =
                                    fabs(offset - expected_offset)
                                    <= epsilon
                                           * fmax(1.0f, fmax(fabs(offset), fabs(expected_offset)));
                                // If we think an overflow is allowed, and if the other
                                // implementation might have detected one (offset=0), then
                                // all is fine as well
                                if (may_overflow && scale == 0.0f && offset == 0.0f)
                                    offset_is_good = true;
                                if (!offset_is_good)
                                    FATAL_ERROR("Found inaccurate offset: beam {}, want {}, have "
                                                "{}, scale_may_collapse={}",
                                                beam, expected_offset, offset, scale_may_collapse);

                                // Check absolute error of scale
                                bool scale_is_good = fabs(scale - expected_scale) <= epsilon;
                                // If the scale is close to zero then don't worry about the
                                // scale at all
                                if (scale_may_collapse)
                                    scale_is_good = true;
                                // If we think an overflow is allowed, and if the other
                                // implementation might have detected one (scale=0,
                                // offset=0), then all is fine as well
                                if (may_overflow && scale == 0.0f && offset == 0.0f)
                                    scale_is_good = true;
                                if (!scale_is_good)
                                    FATAL_ERROR("Found inaccurate scale: beam{}, want {}, have {}, "
                                                "scale_may_collapse={}",
                                                beam, expected_scale, scale, scale_may_collapse);

                                for (int freq_chunk = 0; freq_chunk < out_nfreqs_chunk;
                                     ++freq_chunk) {
                                    for (int time_chunk = 0; time_chunk < out_ntimes_chunk;
                                         ++time_chunk) {
                                        const int freq =
                                            freq_chunk
                                            + out_nfreqs_chunk
                                                  * (freq_packet + out_nfreqs_packet * freq_outer);
                                        const int time = time_chunk + out_ntimes_chunk * time_outer;
                                        const float expected_x = inputat(beam, freq, time);

                                        const std::uint8_t i =
                                            beamsat(beam_outer, freq_outer, time_outer, beam_packet,
                                                    freq_packet, freq_chunk, time_chunk);
                                        if (i == 255)
                                            FATAL_ERROR("Unexpected value {}", i);
                                        const float x = i == 0 ? 0.0f / 0.0f : offset + scale * i;

                                        bool isgood = false;

                                        // Allow the value to be clamped
                                        if (i == 0 && isnan(x))
                                            isgood = true;
                                        if (i == 1 && x > expected_x)
                                            isgood = true;
                                        if (i == 254 && x < expected_x)
                                            isgood = true;
                                        // The value should differ by no more than scale/2
                                        isgood |= fabs(x - expected_x)
                                                  <= fmax(scale, expected_scale) / 2 + epsilon;
                                        // Handle non-finite inputs
                                        if (may_overflow && offset == 0.0f && scale == 0.0f)
                                            isgood = true;
                                        if (scale_may_collapse && scale == 0.0f)
                                            isgood = true;
                                        if (!isgood)
                                            FATAL_ERROR("Found inaccurate value: beam {}, time {}, "
                                                        "want {}, have {}, offset {}, scale {}, "
                                                        "may_overflow={}, scale_may_collapse={}, "
                                                        "expected_offset={}, expected_scale={}",
                                                        beam, time, expected_x, x, offset, scale,
                                                        may_overflow, scale_may_collapse,
                                                        expected_offset, expected_scale);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        ////////////////////////////////////////////////////////////////////////////////

        // Poison output
        std::fill(offsetscale.begin(), offsetscale.end(), fpoison);
        std::fill(beams.begin(), beams.end(), ipoison);

        // Quantize on the CPU
        INFO("Testing on CPU...");
        cpu_quantize8(input.data(), offsetscale.data(), beams.data());

        // Check CPU result
        check_result();

        ////////////////////////////////////////////////////////////////////////////////

        // Allocate memory on the GPU
        float16_t* gpu_input;
        CHECK_CUDA_ERROR(cudaMalloc(&gpu_input, input.size() * sizeof *input.data()));
        float16_t* gpu_offsetscale;
        CHECK_CUDA_ERROR(
            cudaMalloc(&gpu_offsetscale, offsetscale.size() * sizeof *offsetscale.data()));
        std::uint8_t* gpu_beams;
        CHECK_CUDA_ERROR(cudaMalloc(&gpu_beams, beams.size() * sizeof *beams.data()));

        // Poison output
        std::fill(offsetscale.begin(), offsetscale.end(), fpoison);
        std::fill(beams.begin(), beams.end(), ipoison);

        // Copy input and poisoned outputs to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input.data(), input.size() * sizeof *input.data(),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_offsetscale, offsetscale.data(),
                                    offsetscale.size() * sizeof *offsetscale.data(),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_beams, beams.data(), beams.size() * sizeof *beams.data(),
                                    cudaMemcpyHostToDevice));

        // Quantize on the GPU
        INFO("Testing on GPU...");
        gpu_quantize8(gpu_input, gpu_offsetscale, gpu_beams, nullptr);

        // Copy output from GPU
        CHECK_CUDA_ERROR(cudaMemcpy(offsetscale.data(), gpu_offsetscale,
                                    offsetscale.size() * sizeof *offsetscale.data(),
                                    cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(beams.data(), gpu_beams, beams.size() * sizeof *beams.data(),
                                    cudaMemcpyDeviceToHost));

        // Check GPU result
        check_result();

        // Done.
        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testQuantizeKernel8);
