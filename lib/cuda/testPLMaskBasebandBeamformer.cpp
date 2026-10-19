// Standalone test for the baseband beamformer PL-mask GPU kernel: runs the GPU launcher and the CPU
// reference on the same made-up data and compares them bit-for-bit. This tests the kernel only; it
// does NOT exercise the kotekan ring-buffer / pipeline integration. Modeled on
// testPLMaskUpchannelizer.

#include "Config.hpp"                       // for Config
#include "Stage.hpp"                        // for Stage
#include "StageFactory.hpp"                 // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp"              // for bufferContainer
#include "cudaPLMaskBasebandBeamformer.hpp" // for launch_bb_pl_mask, cpu_bb_pl_mask
#include "cudaUtils.hpp"                    // for CHECK_CUDA_ERROR
#include "cuda_runtime.h"                   // for cudaMalloc, cudaMemcpy, ...
#include "errors.h"                         // for TEST_PASSED
#include "kotekanLogging.hpp"               // for FATAL_ERROR, INFO

#include <cmath>      // for pow
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for uint64_t
#include <functional> // for function
#include <random>     // for mt19937_64, uniform_real_distribution
#include <string>     // for string
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

class testPLMaskBasebandBeamformer : public kotekan::Stage {
public:
    testPLMaskBasebandBeamformer(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    virtual ~testPLMaskBasebandBeamformer() {}

    void main_thread() override {
        using std::ptrdiff_t;
        using std::uint64_t;

        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);

        // Run GPU and CPU on the input `h_in` [size_in][num_inner][num_d8] and compare the written
        // output rows.
        const auto compare = [&](const std::vector<uint64_t>& h_in, const ptrdiff_t num_inner,
                                 const ptrdiff_t num_d8, const ptrdiff_t num_rows,
                                 const ptrdiff_t size_in, const ptrdiff_t pos_in,
                                 const ptrdiff_t size_out, const ptrdiff_t pos_out) {
            const ptrdiff_t in_count = size_in * num_inner * num_d8;
            const ptrdiff_t out_count = size_out * num_inner;

            uint64_t* d_in = nullptr;
            uint64_t* d_out = nullptr;
            CHECK_CUDA_ERROR(cudaMalloc(&d_in, in_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_out, out_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(
                cudaMemcpy(d_in, h_in.data(), in_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemset(d_out, 0, out_count * sizeof(uint64_t)));

            launch_bb_pl_mask(d_out, d_in, num_inner, num_d8, num_rows, size_in, pos_in, size_out,
                              pos_out, /*stream=*/0);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            std::vector<uint64_t> out_gpu(out_count, 0);
            CHECK_CUDA_ERROR(cudaMemcpy(out_gpu.data(), d_out, out_count * sizeof(uint64_t),
                                        cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(d_in));
            CHECK_CUDA_ERROR(cudaFree(d_out));

            std::vector<uint64_t> out_cpu(out_count, 0);
            cpu_bb_pl_mask(out_cpu.data(), h_in.data(), num_inner, num_d8, num_rows, size_in,
                           pos_in, size_out, pos_out);

            // Rows outside the written range are zero in both
            for (ptrdiff_t i = 0; i < out_count; ++i)
                if (out_gpu[i] != out_cpu[i])
                    FATAL_ERROR("Mismatch: num_inner={}, num_d8={}, num_rows={}, size_in={}, "
                                "pos_in={}, size_out={}, pos_out={}, index={}: gpu={:#018x} "
                                "cpu={:#018x}",
                                num_inner, num_d8, num_rows, size_in, pos_in, size_out, pos_out, i,
                                out_gpu[i], out_cpu[i]);
            return out_cpu;
        };

        // {num_inner, num_d8}: CHORD (48 frequencies, 512 dishes), CHIME (1024 dishes), and an
        // inner size that is not a multiple of the thread block size
        const ptrdiff_t sizes[][2] = {{96, 64}, {32, 128}, {300, 3}};
        // {size_in, pos_in, size_out, pos_out, num_rows}
        const ptrdiff_t geoms[][5] = {
            {64, 5, 64, 37, 16},  // independent positions
            {64, 60, 32, 30, 16}, // both rings wrap
            {48, 0, 48, 0, 48},   // whole ring, not a power of two
        };

        for (const auto& sz : sizes) {
            const ptrdiff_t num_inner = sz[0];
            const ptrdiff_t num_d8 = sz[1];
            // With density 0.5 almost every output bit (an AND of num_d8 input bits) is 0. At this
            // density about half of them are 1, so that a misplaced input word shows.
            const double half_density = std::pow(0.5, 1.0 / num_d8);
            const double densities[] = {0.5, half_density, 1.0};
            for (const auto& g : geoms) {
                const ptrdiff_t size_in = g[0], pos_in = g[1], size_out = g[2], pos_out = g[3],
                                num_rows = g[4];
                std::vector<uint64_t> h_in(size_in * num_inner * num_d8);

                for (const double d : densities) {
                    std::uniform_real_distribution<double> ud(0.0, 1.0);
                    for (auto& x : h_in) {
                        uint64_t v = 0;
                        for (int b = 0; b < 64; ++b)
                            if (ud(rng) < d)
                                v |= uint64_t(1) << b;
                        x = v;
                    }
                    compare(h_in, num_inner, num_d8, num_rows, size_in, pos_in, size_out, pos_out);
                }

                // One lost sample in one dish group clears exactly that output bit
                const ptrdiff_t r = num_rows / 2, s = num_inner / 3, d8 = num_d8 - 1;
                const int b = 17;
                for (auto& x : h_in)
                    x = ~uint64_t(0);
                h_in[(((pos_in + r) % size_in) * num_inner + s) * num_d8 + d8] &=
                    ~(uint64_t(1) << b);
                const std::vector<uint64_t> out =
                    compare(h_in, num_inner, num_d8, num_rows, size_in, pos_in, size_out, pos_out);
                for (ptrdiff_t o = 0; o < num_rows; ++o)
                    for (ptrdiff_t i = 0; i < num_inner; ++i) {
                        const uint64_t expected =
                            o == r && i == s ? ~(uint64_t(1) << b) : ~uint64_t(0);
                        const uint64_t actual = out[((pos_out + o) % size_out) * num_inner + i];
                        if (actual != expected)
                            FATAL_ERROR("Single lost sample: num_inner={}, num_d8={}, row={}, "
                                        "inner={}: expected={:#018x} actual={:#018x}",
                                        num_inner, num_d8, o, i, expected, actual);
                    }
            }
        }

        INFO("testPLMaskBasebandBeamformer: all GPU-vs-CPU cases matched.");
        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testPLMaskBasebandBeamformer);
