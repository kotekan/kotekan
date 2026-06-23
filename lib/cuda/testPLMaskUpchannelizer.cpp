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

        // Inner (fast) element count per frequency, a power of two (= num_polarizations * num_dishes/8).
        const ptrdiff_t num_elements = 16;

        // Run one test case: invent input [size_in][num_frequencies][num_elements], run GPU + CPU,
        // compare the written output rows. The kernel reads input channels [Fmin, Fmax) and writes
        // output channels [0, Fmax-Fmin).
        const auto run_case = [&](const int U, const ptrdiff_t num_frequencies,
                                  const ptrdiff_t num_frequencies_out, const int Fmin,
                                  const int Fmax, const ptrdiff_t size_in, const ptrdiff_t pos_in,
                                  const ptrdiff_t size_out, const ptrdiff_t pos_out,
                                  const double ones_prob) {
            // `num_times_64` consumes the whole input ring; the kernel's forward look-ahead then
            // wraps within that ring, and both GPU and CPU wrap identically.
            const ptrdiff_t num_times_64 = size_in;
            const ptrdiff_t num_out_words = num_times_64 / U;
            if (num_out_words > size_out)
                FATAL_ERROR("test misconfigured: num_out_words {} > size_out {}", num_out_words,
                            size_out);

            const ptrdiff_t n_inner = ptrdiff_t(Fmax - Fmin) * num_elements; // inner positions written
            const ptrdiff_t out_stride = num_frequencies_out * num_elements; // output per-row stride
            const ptrdiff_t in_count = size_in * num_frequencies * num_elements;
            const ptrdiff_t out_count = size_out * out_stride;

            // Invent input data.
            std::vector<uint64_t> h_in(in_count);
            std::uniform_real_distribution<double> ud(0.0, 1.0);
            for (auto& x : h_in) {
                uint64_t v = 0;
                for (int b = 0; b < 64; ++b)
                    if (ud(rng) < ones_prob)
                        v |= uint64_t(1) << b;
                x = v;
            }

            // GPU: copy input up, run the kernel, copy output back.
            uint64_t* d_in = nullptr;
            uint64_t* d_out = nullptr;
            CHECK_CUDA_ERROR(cudaMalloc(&d_in, in_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_out, out_count * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), in_count * sizeof(uint64_t),
                                        cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemset(d_out, 0, out_count * sizeof(uint64_t)));

            launch_upchannelize_pl_mask(d_out, d_in, num_elements, num_frequencies,
                                        num_frequencies_out, Fmin, Fmax, num_times_64, size_in,
                                        pos_in, size_out, pos_out, U, /*stream=*/0);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            std::vector<uint64_t> out_gpu(out_count, 0);
            CHECK_CUDA_ERROR(cudaMemcpy(out_gpu.data(), d_out, out_count * sizeof(uint64_t),
                                        cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(d_in));
            CHECK_CUDA_ERROR(cudaFree(d_out));

            // CPU reference.
            std::vector<uint64_t> out_cpu(out_count, 0);
            cpu_upchannelize_pl_mask(out_cpu.data(), h_in.data(), num_elements, num_frequencies,
                                     num_frequencies_out, Fmin, Fmax, num_times_64, size_in, pos_in,
                                     size_out, pos_out, U);

            // Compare the written output rows (s in [0, n_inner)); the over-allocated tail of each
            // row is left zero by both and not compared explicitly.
            const ptrdiff_t out_mask = size_out - 1;
            for (ptrdiff_t o = 0; o < num_out_words; ++o) {
                const ptrdiff_t phys = (pos_out + o) & out_mask;
                for (ptrdiff_t s = 0; s < n_inner; ++s) {
                    const ptrdiff_t i = phys * out_stride + s;
                    if (out_gpu[i] != out_cpu[i])
                        FATAL_ERROR("Mismatch: U={}, nfreq={}, Fmin={}, Fmax={}, size_in={}, "
                                    "pos_in={}, size_out={}, pos_out={}, density={}, out_row={}, "
                                    "inner={}: gpu={:#018x} cpu={:#018x}",
                                    U, num_frequencies, Fmin, Fmax, size_in, pos_in, size_out,
                                    pos_out, ones_prob, o, s, out_gpu[i], out_cpu[i]);
                }
            }
        };

        // Test cases: every U; a few frequency sub-ranges (full, offset interior, single channel,
        // non-power-of-2 nfreq); an offset geometry and one that forces an output-ring wrap; a couple
        // of bit densities. (Correctness over the full cross-product is covered by an off-line host
        // test against the same CPU reference; this matrix is kept small so the deliberately-slow CPU
        // oracle finishes within the stage join timeout in debug builds.)
        const int factors[] = {2, 4, 8, 16, 32, 64, 128};
        // {num_frequencies_in, Fmin, Fmax, num_frequencies_out}
        const ptrdiff_t ranges[][4] = {
            {16, 0, 16, 16}, // full range, output not over-allocated
            {17, 3, 11, 32}, // offset interior; output over-allocated (writes 8 of 32 channels)
        };
        // {size_in, pos_in, size_out, pos_out}
        const ptrdiff_t geoms[][4] = {
            {256, 5, 256, 37},    // independent positions
            {256, 251, 256, 250}, // positions near the end -> wrap
        };
        const double densities[] = {0.5, 1.0};

        for (const int U : factors)
            for (const auto& r : ranges)
                for (const auto& g : geoms)
                    for (const double d : densities)
                        run_case(U, r[0], r[3], int(r[1]), int(r[2]), g[0], g[1], g[2], g[3], d);

        INFO("testPLMaskUpchannelizer: all GPU-vs-CPU cases matched.");
        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testPLMaskUpchannelizer);
