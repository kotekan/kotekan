#include "Config.hpp"
#include "DataType.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "bufferContainer.hpp"
#include "cudaQuantizeKernel4.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

class testQuantizeKernel4 : public kotekan::Stage {
public:
    testQuantizeKernel4(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    virtual ~testQuantizeKernel4() {}

    void main_thread() override {
        const float16_t fpoison(0.0f / 0.0f);
        const kotekan::int4x2_t ipoison(-8, -8);

        const float epsilon = 1.0e-3; // We're using float16

        const int chunk_size = 256;

        const int ntimes = 512;
        const int nfreqs = 32;
        const int nbeams = 64;

        const float stddev_cutoff = 3.0f;

        ////////////////////////////////////////////////////////////////////////////////

        // layout: input[beam, freq, time]
        std::vector<float16_t> input(ntimes * nfreqs * nbeams);
        assert(ntimes % chunk_size == 0);
        const int nchunks = ntimes / chunk_size;
        assert(ntimes % 2 == 0);
        std::vector<float16_t> offsetscale(2 * nchunks * nfreqs * nbeams);
        std::vector<kotekan::int4x2_t> beams(ntimes / 2 * nfreqs * nbeams);

        // Invent some data
        const auto f = [&](const int beam, const int freq, const int time) -> float {
            const float x = (time + 0.5) / ntimes;
            switch ((beam + freq) % 11) {
                case 0:
                    return 0.0f;
                case 1:
                    return 0.1f * beam;
                case 2:
                    return time % 2;
                case 3:
                    return x;
                case 4:
                    return 0.5f * x + 0.5f * x * x * x;
                case 5:
                    return 0.5f * x + 0.5f * x * x * x + 10 * (time % 23 == 0);
                case 6:
                    return 10000.0f * time; // large but not infinity
                case 7:
                    return 100000.0f * time; // some values are infinity
                case 8:
                    return time / 10000.0f; // small, but 1/x is less than infinity
                case 9:
                    return time / 100000.0f; // small, and some 1/x are infinity
                case 10:
                    return 0.5f * x + 0.5f * x * x * x + (time % 23 == 0 ? 0.0f / 0.0f : 0.0f);
            }
            std::abort();
        };


        ////////////////////////////////////////////////////////////////////////////////

        // Set input
        for (int beam = 0; beam < nbeams; ++beam) {
            for (int freq = 0; freq < nfreqs; ++freq) {
                for (int time = 0; time < ntimes; ++time) {
                    const float x = f(beam, freq, time % chunk_size);
                    const int idx = time + ntimes * (freq + nfreqs * beam);
                    input.at(idx) = float16_t(x);
                }
            }
        }

        // A function to check the result
        const auto check_result = [&]() {
            using std::fabs, std::fmax, std::fmin, std::isfinite, std::isnan, std::sqrt;

            for (int beam = 0; beam < nbeams; ++beam) {
                for (int freq = 0; freq < nfreqs; ++freq) {

                    // Calculate expected result
                    //
                    // When comparing to the expected result, two things can go wrong:
                    // 1. The scale can be zero, or can be close to zero (`scale_may_collapse`)
                    // 2. The input can contain large numbers, or infinities, or not-a-numbers
                    //    (`may_overflow`)
                    //
                    // We need to handle both cases, and we need to be lenient when comparing when
                    // this happens. For example, there might be an overflow on the GPU, but not in
                    // the CPU implementation. This should not be flagged as error.
                    float sum = 0, sum2 = 0, minval = +FLT_MAX, maxval = -FLT_MAX;
                    bool may_overflow = false;
                    for (int time = 0; time < chunk_size; ++time) {
                        const float x = f(beam, freq, time);
                        sum += x;
                        sum2 += x * x;
                        minval = isnan(minval) || isnan(x) ? 0.0f / 0.0f : fmin(minval, x);
                        maxval = isnan(maxval) || isnan(x) ? 0.0f / 0.0f : fmax(maxval, x);
                        // Allow the respective float16 calculations to overflow
                        may_overflow |= !isfinite(x) || fabs(x) >= 2048.0f;
                    }
                    may_overflow |= !isfinite(sum) || fabs(sum) >= 2048.0f || !isfinite(sum2)
                                    || fabs(sum2) >= 2048.0f;
                    // Allow the scale to be inaccurate when all inputs are close together
                    const bool scale_may_collapse = fabs(maxval - minval) < 10 * epsilon;
                    const float mean = sum / chunk_size;
                    const float stddev = sqrt(fmax(0.0f, sum2 / chunk_size - mean * mean));
                    const int int4_range = 15; // avoid -8
                    float expected_offset = mean;
                    float expected_scale = 2 * stddev_cutoff * stddev / int4_range;
                    if (!isfinite(expected_offset) || !isfinite(expected_scale)) {
                        may_overflow = true;
                        expected_offset = 0.0f;
                        expected_scale = 0.0f;
                    }

                    for (int chunk = 0; chunk < nchunks; ++chunk) {
                        const float offset =
                            offsetscale.at(2 * chunk + 2 * nchunks * (freq + nfreqs * beam) + 0);
                        const float scale =
                            offsetscale.at(2 * chunk + 2 * nchunks * (freq + nfreqs * beam) + 1);
                        if (!isfinite(offset))
                            FATAL_ERROR("Found non-finite offset {}", offset);
                        if (!isfinite(scale))
                            FATAL_ERROR("Found non-finite scale {}", scale);

                        // Check relative error of offset
                        bool offset_is_good =
                            fabs(offset - expected_offset)
                            <= epsilon * fmax(1.0f, fmax(fabs(offset), fabs(expected_offset)));
                        // If we think an overflow is allowed, and if the other implementation might
                        // have detected one (offset=0), then all is fine as well
                        if (may_overflow && offset == 0.0f)
                            offset_is_good = true;
                        if (!offset_is_good)
                            FATAL_ERROR("Found inaccurate offset: beam {}, want {}, have {}, "
                                        "scale_may_collapse={}",
                                        beam, expected_offset, offset, scale_may_collapse);

                        // Check absolute error of scale
                        bool scale_is_good = abs(scale - expected_scale) <= epsilon;
                        // If the scale is close to zero then don't worry about the scale at all
                        if (scale_may_collapse)
                            scale_is_good = true;
                        // If we think an overflow is allowed, and if the other implementation might
                        // have detected one (offset=0), then all is fine as well
                        if (may_overflow && scale == 0.0f)
                            scale_is_good = true;
                        if (!scale_is_good)
                            FATAL_ERROR("Found inaccurate scale: beam{}, want {}, have {}, "
                                        "scale_may_collapse={}",
                                        beam, expected_scale, scale, scale_may_collapse);

                        for (int time2 = chunk * chunk_size; time2 < (chunk + 1) * chunk_size;
                             time2 += 2) {
                            const kotekan::int4x2_t i01 =
                                beams.at(time2 / 2 + ntimes / 2 * (freq + nfreqs * beam));
                            for (int time = time2; time < time2 + 2; ++time) {
                                const int idx = time + ntimes * (freq + nfreqs * beam);
                                const float expected_x = input.at(idx);

                                const int i = i01[time - time2];
                                const float x = i == -8 ? 0.0f / 0.0f : offset + scale * i;

                                bool isgood = false;

                                // Allow the value to be clamped
                                if (i == -8 && isnan(expected_x))
                                    isgood = true;
                                if (i == -7 && x > expected_x)
                                    isgood = true;
                                if (i == +7 && x < expected_x)
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
                                    FATAL_ERROR("Found inaccurate value: "
                                                "beam {}, freq {}, time {}, "
                                                "want {}, have {}, "
                                                "offset {}, scale {}, "
                                                "may_overflow={}, "
                                                "expected_offset={}, expected_scale={}",
                                                beam, freq, time, expected_x, x, offset, scale,
                                                may_overflow, expected_offset, expected_scale);
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
        cpu_quantize4(input.data(), offsetscale.data(), beams.data(),                 //
                      ntimes, nfreqs, nbeams, ntimes, ntimes * nfreqs,                //
                      2 * nchunks, nfreqs, nbeams, 2 * nchunks, 2 * nchunks * nfreqs, //
                      ntimes / 2, nfreqs, nbeams, ntimes / 2, ntimes / 2 * nfreqs,    //
                      stddev_cutoff                                                   //
        );

        // Check CPU result
        check_result();

        ////////////////////////////////////////////////////////////////////////////////

        // Allocate memory on the GPU
        float16_t* gpu_input;
        CHECK_CUDA_ERROR(cudaMalloc(&gpu_input, input.size() * sizeof *input.data()));
        float16_t* gpu_offsetscale;
        CHECK_CUDA_ERROR(
            cudaMalloc(&gpu_offsetscale, offsetscale.size() * sizeof *offsetscale.data()));
        kotekan::int4x2_t* gpu_beams;
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
        gpu_quantize4(gpu_input, gpu_offsetscale, gpu_beams,                          //
                      ntimes, nfreqs, nbeams, ntimes, ntimes * nfreqs,                //
                      2 * nchunks, nfreqs, nbeams, 2 * nchunks, 2 * nchunks * nfreqs, //
                      ntimes / 2, nfreqs, nbeams, ntimes / 2, ntimes / 2 * nfreqs,    //
                      stddev_cutoff,                                                  //
                      nullptr                                                         //
        );

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

REGISTER_KOTEKAN_STAGE(testQuantizeKernel4);
