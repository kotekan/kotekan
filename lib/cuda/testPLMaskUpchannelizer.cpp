// Standalone test for the PL-mask upchannelizer GPU kernel: runs the GPU launcher and the CPU
// reference on the same made-up data and compares them bit-for-bit. This tests the kernel only; it
// does NOT exercise the kotekan ring-buffer / pipeline integration. Modeled on testQuantizeKernel8.

#include "Config.hpp"                   // for Config
#include "Stage.hpp"                    // for Stage
#include "StageFactory.hpp"             // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp"          // for bufferContainer
#include "cudaPLMaskUpchannelizer.hpp"  // for launch_upchannelize_pl_mask, cpu_upchannelize_pl_mask
#include "cudaUtils.hpp"                // for CHECK_CUDA_ERROR
#include "cuda_runtime.h"               // for cudaMalloc, cudaMemcpy, ...
#include "errors.h"                     // for TEST_PASSED
#include "kotekanLogging.hpp"           // for FATAL_ERROR, INFO

#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for uint64_t
#include <functional> // for function
#include <random>     // for mt19937_64
#include <string>     // for string
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

class testPLMaskUpchannelizer : public kotekan::Stage {
public:
    testPLMaskUpchannelizer(Config& config, const std::string& unique_name,
                            bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    virtual ~testPLMaskUpchannelizer() {}

    void main_thread() override {
        using std::ptrdiff_t;
        using std::uint64_t;

        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);

        // Run one test case: invent input, run GPU + CPU, compare the written output rows.
        const auto run_case = [&](const int U, const ptrdiff_t num_spectators,
                                  const ptrdiff_t size_in, const ptrdiff_t pos_in,
                                  const ptrdiff_t size_out, const ptrdiff_t pos_out,
                                  const double ones_prob) {
            // `num_times_64` consumes the whole input ring; the kernel's forward look-ahead then
            // wraps within that ring, and both GPU and CPU wrap identically.
            const ptrdiff_t num_times_64 = size_in;
            const ptrdiff_t num_out_words = num_times_64 / U;
            if (num_out_words > size_out)
                FATAL_ERROR("test misconfigured: num_out_words {} > size_out {}", num_out_words,
                            size_out);

            // Invent input data.
            std::vector<uint64_t> h_in(size_in * num_spectators);
            std::uniform_real_distribution<double> ud(0.0, 1.0);
            for (auto& x : h_in) {
                uint64_t v = 0;
                for (int b = 0; b < 64; ++b)
                    if (ud(rng) < ones_prob)
                        v |= uint64_t(1) << b;
                x = v;
            }

            const ptrdiff_t in_count = size_in * num_spectators;
            const ptrdiff_t out_count = size_out * num_spectators;

            // GPU: copy input up, run the kernel, copy output back.
            uint64_t* d_in = nullptr;
            uint64_t* d_out = nullptr;
            CHECK_CUDA_ERROR(cudaMalloc(&d_in, in_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_out, out_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), in_count * sizeof(uint64_t),
                                        cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemset(d_out, 0, out_count * sizeof(uint64_t)));

            cudaPLMaskUpchannelizer::launch_upchannelize_pl_mask(
                d_out, d_in, num_spectators, num_times_64, size_in, pos_in, size_out, pos_out, U,
                /*stream=*/0);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            std::vector<uint64_t> out_gpu(out_count, 0);
            CHECK_CUDA_ERROR(cudaMemcpy(out_gpu.data(), d_out, out_count * sizeof(uint64_t),
                                        cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(d_in));
            CHECK_CUDA_ERROR(cudaFree(d_out));

            // CPU reference.
            std::vector<uint64_t> out_cpu(out_count, 0);
            cudaPLMaskUpchannelizer::cpu_upchannelize_pl_mask(out_cpu.data(), h_in.data(),
                                                              num_spectators, num_times_64, size_in,
                                                              pos_in, size_out, pos_out, U);

            // Compare exactly the physical output rows that were written.
            const ptrdiff_t out_mask = size_out - 1;
            for (ptrdiff_t o = 0; o < num_out_words; ++o) {
                const ptrdiff_t phys = (pos_out + o) & out_mask;
                for (ptrdiff_t s = 0; s < num_spectators; ++s) {
                    const ptrdiff_t i = phys * num_spectators + s;
                    if (out_gpu[i] != out_cpu[i])
                        FATAL_ERROR("Mismatch: U={}, size_in={}, pos_in={}, size_out={}, pos_out={}, "
                                    "density={}, out_row={}, spectator={}: gpu={:#018x} cpu={:#018x}",
                                    U, size_in, pos_in, size_out, pos_out, ones_prob, o, s,
                                    out_gpu[i], out_cpu[i]);
                }
            }
        };

        // Test cases: every U, independent in/out ring sizes, non-U-aligned positions (incl. one
        // that forces an output-ring wrap), a non-round spectator count, several bit densities.
        const ptrdiff_t num_spectators = 70;
        const int factors[] = {2, 4, 8, 16, 32, 64, 128};
        // {size_in, pos_in, size_out, pos_out}
        const ptrdiff_t geoms[][4] = {
            {256, 5, 256, 37},   // independent sizes, offset positions
            {256, 251, 256, 250}, // positions near the end -> wrap
            {512, 100, 256, 9},  // larger input ring, smaller output ring
        };
        const double densities[] = {0.5, 0.9, 1.0};

        for (const int U : factors)
            for (const auto& g : geoms)
                for (const double d : densities)
                    run_case(U, num_spectators, g[0], g[1], g[2], g[3], d);

        INFO("testPLMaskUpchannelizer: all GPU-vs-CPU cases matched.");
        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testPLMaskUpchannelizer);
