// Standalone test for the FRB1 PL mask GPU kernel: runs the GPU launcher and the CPU reference on
// the same made-up data and compares them byte for byte. This tests the kernel only; it does NOT
// exercise the kotekan ring-buffer / pipeline integration. Modeled on testPLMaskUpchannelizer.

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "cudaPLMaskFRB1.hpp"  // for launch_frb1_pl_mask, cpu_frb1_pl_mask
#include "cudaUtils.hpp"       // for CHECK_CUDA_ERROR
#include "cuda_runtime.h"      // for cudaMalloc, cudaMemcpy, ...
#include "errors.h"            // for TEST_PASSED
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO

#include <cmath>      // for pow
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for uint64_t, uint8_t
#include <functional> // for function
#include <random>     // for mt19937_64, uniform_real_distribution
#include <string>     // for string
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

class testPLMaskFRB1 : public kotekan::Stage {
public:
    testPLMaskFRB1(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    virtual ~testPLMaskFRB1() {}

    void main_thread() override {
        using std::ptrdiff_t;
        using std::uint64_t;
        using std::uint8_t;

        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);

        // Run one test case: invent input [size_in][num_frequencies][num_elements], run GPU + CPU,
        // and compare the whole output, including the bytes that must remain untouched.
        const auto run_case = [&](const ptrdiff_t num_elements, const int U, const int Td,
                                  const ptrdiff_t num_frequencies, const int Fmin, const int Fmax,
                                  const ptrdiff_t num_frequencies_out, const int Fbar_out_min,
                                  const ptrdiff_t size_in, const ptrdiff_t pos_in,
                                  const int bit_offset, const ptrdiff_t size_out,
                                  const ptrdiff_t pos_out, const double ones_prob,
                                  const bool expect_mixed) {
            // Produce as many output samples as the input ring holds
            const ptrdiff_t num_times_out = (64 * size_in - bit_offset) / Td;
            if (!(num_times_out > 0 && num_times_out <= size_out))
                FATAL_ERROR("test misconfigured: num_times_out {} for size_out {}", num_times_out,
                            size_out);

            const ptrdiff_t in_count = size_in * num_frequencies * num_elements;
            const ptrdiff_t out_count = size_out * num_frequencies_out;

            // Invent input data
            std::vector<uint64_t> h_in(in_count);
            std::uniform_real_distribution<double> ud(0.0, 1.0);
            for (auto& x : h_in) {
                uint64_t v = 0;
                for (int b = 0; b < 64; ++b)
                    if (ud(rng) < ones_prob)
                        v |= uint64_t(1) << b;
                x = v;
            }

            // Bytes that are not written must keep this value
            const uint8_t untouched = 0x55;

            // GPU: copy input up, run the kernel, copy output back
            uint64_t* d_in = nullptr;
            uint8_t* d_out = nullptr;
            CHECK_CUDA_ERROR(cudaMalloc(&d_in, in_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_out, out_count * sizeof(uint8_t)));
            CHECK_CUDA_ERROR(
                cudaMemcpy(d_in, h_in.data(), in_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemset(d_out, untouched, out_count * sizeof(uint8_t)));

            launch_frb1_pl_mask(d_out, d_in, num_elements, num_frequencies, num_frequencies_out,
                                Fmin, Fmax, Fbar_out_min, U, Td, num_times_out, size_in, pos_in,
                                bit_offset, size_out, pos_out, /*stream=*/0);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            std::vector<uint8_t> out_gpu(out_count);
            CHECK_CUDA_ERROR(cudaMemcpy(out_gpu.data(), d_out, out_count * sizeof(uint8_t),
                                        cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(d_in));
            CHECK_CUDA_ERROR(cudaFree(d_out));

            // CPU reference
            std::vector<uint8_t> out_cpu(out_count, untouched);
            cpu_frb1_pl_mask(out_cpu.data(), h_in.data(), num_elements, num_frequencies,
                             num_frequencies_out, Fmin, Fmax, Fbar_out_min, U, Td, num_times_out,
                             size_in, pos_in, bit_offset, size_out, pos_out);

            ptrdiff_t num_valid = 0;
            for (ptrdiff_t i = 0; i < out_count; ++i) {
                if (out_gpu[i] != out_cpu[i])
                    FATAL_ERROR("Mismatch: num_elements={}, U={}, Td={}, nfreq={}, Fmin={}, "
                                "Fmax={}, nfreq_out={}, Fbar_out_min={}, size_in={}, pos_in={}, "
                                "bit_offset={}, size_out={}, pos_out={}, density={}, row={}, "
                                "fbar={}: gpu={:d} cpu={:d}",
                                num_elements, U, Td, num_frequencies, Fmin, Fmax,
                                num_frequencies_out, Fbar_out_min, size_in, pos_in, bit_offset,
                                size_out, pos_out, ones_prob, i / num_frequencies_out,
                                i % num_frequencies_out, out_gpu[i], out_cpu[i]);
                num_valid += out_cpu[i] == 1;
            }
            // At the "half" density the mask should be neither all valid nor all invalid, unless
            // there are only few output samples
            const ptrdiff_t num_written = num_times_out * U * (Fmax - Fmin);
            if (expect_mixed && num_written >= 32 * U
                && !(num_valid > 0 && num_valid < num_written))
                FATAL_ERROR("test misconfigured: {} of {} output samples are valid", num_valid,
                            num_written);
        };

        // Test cases: the downsampling factors of the CHIME, CHORD, pathfinder, HIRAX and CHARTS
        // FRB1 beamformers; element counts below and above the kernel's block size; a frequency
        // sub-range written into a larger output; positions that make both rings wrap; several
        // input bit offsets (93 % 64 = 29 occurs for U=1, max_upchannelization_factor=64).
        // Data dimensions are small so that the naive CPU oracle stays cheap.
        // {U, Td}
        const int factors[][2] = {{128, 1}, {64, 3}, {32, 10}, {16, 12},
                                  {16, 24}, {2, 96}, {1, 192}, {1, 384}};
        const ptrdiff_t element_counts[] = {4, 200};
        // {num_frequencies_in, Fmin, Fmax, num_frequencies_out factor, Fbar_out_min factor}
        const ptrdiff_t ranges[][5] = {
            {3, 0, 3, 3, 0}, // full range
            {5, 1, 3, 6, 1}, // interior range, written into the middle of a larger output
        };
        // {size_in, pos_in, size_out, pos_out}
        const ptrdiff_t geoms[][4] = {
            {16, 3, 2048, 37},    // independent positions
            {16, 13, 2048, 2040}, // positions near the end -> wrap
        };
        const int bit_offsets[] = {0, 1, 29, 63};

        for (const auto& UTd : factors) {
            const int U = UTd[0];
            const int Td = UTd[1];
            for (const ptrdiff_t E : element_counts) {
                // With density 0.5 almost every output sample (an AND of Td*E input bits) is
                // invalid. At this density about half of them are valid, so that a misplaced input
                // bit shows.
                const double half_density = std::pow(0.5, 1.0 / (double(Td) * E));
                const double densities[] = {0.5, half_density, 1.0};
                for (const auto& r : ranges)
                    for (const auto& g : geoms)
                        for (const int b : bit_offsets)
                            for (const double d : densities)
                                run_case(E, U, Td, r[0], int(r[1]), int(r[2]), r[3] * U,
                                         int(r[4]) * U, g[0], g[1], b, g[2], g[3], d,
                                         d == half_density);
            }
        }

        INFO("testPLMaskFRB1: all GPU-vs-CPU cases matched.");
        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testPLMaskFRB1);
