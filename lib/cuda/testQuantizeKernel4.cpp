#include "Config.hpp"
#include "DataType.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "bufferContainer.hpp"
#include "cudaQuantizeKernel4.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <cassert>
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
        const float16_t fpoison(0.0 / 0.0);
        const kotekan::int4x2_t ipoison(-8, -8);

        const float epsilon = 1.0e-3; // We're using float16

        const int chunk_size = 256;

        const int ntimes = 512;
        const int nbeams = 64;
        const int nfreqs = 32;

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
        const auto f = [&](const int beam, const int time) -> float {
            const float x = (time + 0.5) / ntimes;
            switch (beam % 5) {
                case 0:
                    return 0.1f * beam;
                case 1:
                    return time % 2;
                case 2:
                    return x;
                case 3:
                    return 0.5f * x + 0.5f * x * x * x;
                case 4:
                    return 0.5f * x + 0.5f * x * x * x + 10 * (time % 23 == 0);
            }
            std::abort();
        };


        ////////////////////////////////////////////////////////////////////////////////

        // Set input
        for (int beam = 0; beam < nbeams; ++beam) {
            for (int freq = 0; freq < nfreqs; ++freq) {
                for (int time = 0; time < ntimes; ++time) {
                    const float x = f(beam, time % chunk_size);
                    const int idx = time + ntimes * (freq + nfreqs * beam);
                    input.at(idx) = (float16_t)x;
                }
            }
        }

        // A function to check the result
        const auto check_result = [&]() {
            for (int beam = 0; beam < nbeams; ++beam) {

                // Calculate expected result
                float sum = 0, sum2 = 0;
                for (int time = 0; time < chunk_size; ++time) {
                    const float x = f(beam, time);
                    sum += x;
                    sum2 += x * x;
                }
                const float mean = sum / chunk_size;
                using std::fmax, std::sqrt;
                const float stddev = sqrt(fmax(0.0f, sum2 / chunk_size - mean * mean));
                const int int4_range = 15;
                const float stddev_cutoff = 3.0;
                const float expected_offset = mean;
                const float expected_scale = 2 * stddev_cutoff * stddev / int4_range;

                for (int freq = 0; freq < nfreqs; ++freq) {
                    for (int chunk = 0; chunk < nchunks; ++chunk) {
                        const float offset =
                            offsetscale.at(2 * chunk + 2 * nchunks * (freq + nfreqs * beam) + 0);
                        const float scale =
                            offsetscale.at(2 * chunk + 2 * nchunks * (freq + nfreqs * beam) + 1);
                        using std::isfinite;
                        if (!isfinite(offset))
                            FATAL_ERROR("Found non-finite offset {}", offset);
                        if (!isfinite(scale))
                            FATAL_ERROR("Found non-finite scale {}", scale);
                        if (!(fabs(offset - expected_offset) <= epsilon))
                            FATAL_ERROR("Found inaccurate offset: want {}, have {}",
                                        expected_offset, offset);
                        if (!(fabs(scale - expected_scale) <= epsilon))
                            FATAL_ERROR("Found inaccurate scale: want {}, have {}", expected_scale,
                                        scale);
                        for (int time2 = chunk * chunk_size; time2 < (chunk + 1) * chunk_size;
                             time2 += 2) {
                            const kotekan::int4x2_t i01 =
                                beams.at(time2 / 2 + ntimes / 2 * (freq + nfreqs * beam));
                            for (int time = time2; time < time2 + 2; ++time) {
                                const int i = i01[time - time2];
                                assert(i != -8);
                                const float x = offset + scale * i;

                                const int idx = time + ntimes * (freq + nfreqs * beam);
                                const float expected_x = input.at(idx);

                                bool isgood = false;
                                // Allow the value to be clamped
                                if (i == -7 && x > expected_x)
                                    isgood = true;
                                if (i == +7 && x < expected_x)
                                    isgood = true;
                                // The value should differ by no more than scale/2
                                isgood |= fabs(x - expected_x) <= scale / 2 + epsilon;
                                if (!isgood)
                                    FATAL_ERROR(
                                        "Found inaccurate value: want {}, have {}, scale is {}",
                                        expected_x, x, scale);
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
