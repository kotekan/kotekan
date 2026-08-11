// Standalone test for the PilotProxy detector datapath: runs the packer GPU
// kernel and the vendored F-statistic CUDA library on made-up data and
// compares every product bit-for-bit against CPU references:
//
//   - packer:        launch_pilotproxy_pack  vs cpu_pilotproxy_pack;
//   - coarse powers: FStat_Compute_Powers_U64 vs fstat_ref_powers_u64
//                    (vendored bit-accurate CPU reference);
//   - row sums:      FStat_Compute_RowSums_I32 vs fstat_ref_row_sums_i32;
//   - fine powers:   FStat_Compute_FusedFine_U64 vs the frozen fxfft256 v1
//                    reference (vendored fxfft256_ref.c) plus exact feed
//                    sums over the CPU row sums;
//   - coarse mask:   FStat_Compute_NumDen_Mask_RationalHalf vs the exact
//                    rational comparison;
//   - fine mask:     FStat_Compute_FusedFineMask_U64 vs a bit-exact C++
//                    port of pilot-proxy's fine_decision.py (fine decision
//                    v1: rank-based null-bulk CFAR + designated-set
//                    compare, exact wide-integer arithmetic).
//
// This tests the kernels only; it does NOT exercise the kotekan ring-buffer
// / pipeline integration of cudaPilotProxyDetector. Modeled on
// testPLMaskUpchannelizer.

#include "Config.hpp"                         // for Config
#include "Stage.hpp"                          // for Stage
#include "StageFactory.hpp"                   // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp"                // for bufferContainer
#include "cudaPilotProxyPacker.hpp"           // for launch_pilotproxy_pack, cpu_pilotproxy_pack
#include "cudaUtils.hpp"                      // for CHECK_CUDA_ERROR
#include "cuda_runtime.h"                     // for cudaMalloc, cudaMemcpy, ...
#include "errors.h"                           // for TEST_PASSED
#include "kotekanLogging.hpp"                 // for FATAL_ERROR, INFO
#include "pilotproxy/f_statistic.h"           // for FStat_* (vendored)
#include "pilotproxy/f_statistic_reference.h" // for fstat_ref_* (vendored)

// Frozen fxfft256 v1 reference transform (vendored, C source). Compiled into
// this translation unit without its standalone file harness.
#define FXFFT256_REF_NO_MAIN
#include "pilotproxy/fxfft256_ref.c"

#include <array>      // for array
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for uint64_t, int32_t, int8_t, uint8_t
#include <functional> // for function
#include <random>     // for mt19937_64
#include <string>     // for string
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

namespace {

constexpr int NUM_TERMS = 3;   // target, lower reference, upper reference
constexpr int FINE_BINS = 256; // frozen fxfft256 v1 output bins

////////////////////////////////////////////////////////////////////////////////
// Exact wide-integer helpers for the fine decision reference.

using u128 = unsigned __int128;

struct u256 {
    u128 hi, lo;
};

// Exact 128x128 -> 256 bit product.
u256 mul_u128(const u128 a, const u128 b) {
    const std::uint64_t a0 = std::uint64_t(a), a1 = std::uint64_t(a >> 64);
    const std::uint64_t b0 = std::uint64_t(b), b1 = std::uint64_t(b >> 64);
    const u128 p00 = u128(a0) * b0;
    const u128 p01 = u128(a0) * b1;
    const u128 p10 = u128(a1) * b0;
    const u128 p11 = u128(a1) * b1;
    u128 mid = p01 + p10;
    const u128 mid_carry = mid < p01 ? (u128(1) << 64) : 0; // wrapped 2^128 -> bit 64 of hi
    const u128 lo = p00 + (mid << 64);
    const u128 lo_carry = lo < p00 ? 1 : 0;
    const u128 hi = p11 + (mid >> 64) + mid_carry + lo_carry;
    return {hi, lo};
}

bool less_u256(const u256& x, const u256& y) {
    return x.hi != y.hi ? x.hi < y.hi : x.lo < y.lo;
}

////////////////////////////////////////////////////////////////////////////////
// CPU model of the fine reduction: frozen fxfft256 v1 over each stream's row
// sums, exact uint64 feed sums. Row-sum layout is the
// FStat_Compute_RowSums_I32 contract: term-major, stream-major rows,
// interleaved re/im int32.

void cpu_fine_powers(const std::vector<std::int32_t>& row_sums, const int num_streams,
                     const int windows_per_stream,
                     std::vector<std::uint64_t>& fine_powers_out /* [3][256] */) {
    if (windows_per_stream != FX_NIN)
        FATAL_ERROR_NON_OO("cpu_fine_powers requires the frozen window count {:d}", FX_NIN);
    fine_powers_out.assign(std::size_t(NUM_TERMS) * FINE_BINS, 0);
    const std::ptrdiff_t rows = std::ptrdiff_t(num_streams) * windows_per_stream;
    std::int32_t in[FX_NIN][2];
    std::int32_t out[FX_N][2];
    for (int term = 0; term < NUM_TERMS; ++term) {
        for (int s = 0; s < num_streams; ++s) {
            for (int w = 0; w < windows_per_stream; ++w) {
                const std::ptrdiff_t m = std::ptrdiff_t(s) * windows_per_stream + w;
                const std::ptrdiff_t base = (std::ptrdiff_t(term) * rows + m) * 2;
                in[w][0] = row_sums[base + 0];
                in[w][1] = row_sums[base + 1];
            }
            fxfft256(in, out);
            for (int bin = 0; bin < FX_N; ++bin) {
                const std::int64_t re = out[bin][0], im = out[bin][1];
                fine_powers_out[std::size_t(term) * FINE_BINS + bin] +=
                    std::uint64_t(re * re + im * im);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Bit-exact C++ port of pilot-proxy fine decision v1
// (src/pilot_proxy/fine_decision.py). All comparisons are exact via 256-bit
// products; Python's arbitrary-precision reference stays within these
// widths for kernel-produced fine powers.

struct FineDecisionRef {
    int mask;   // 1 = reject (pilot present)
    bool valid; // false when the bulk is not deeper than the rank
    int n_bulk;
    int rank_bin;  // representative bin of the rank value (-1 invalid)
    int fired_bin; // first designated bin that fired (-1 when mask=0)
};

// Exact F2_i < F2_j by cross multiplication (dens > 0).
bool f2_less(const u128 num_i, const u128 den_i, const u128 num_j, const u128 den_j) {
    return less_u256(mul_u128(num_i, den_j), mul_u128(num_j, den_i));
}

FineDecisionRef cpu_fine_decision(const std::vector<std::uint64_t>& S /* [3][256] */,
                                  const int anchor_bin, const int designated_half_width,
                                  const std::array<unsigned long long, 4>& bulk_mask_words,
                                  const int cfar_rank, const std::uint64_t multiplier_q16) {
    std::array<u128, FINE_BINS> num;
    std::array<u128, FINE_BINS> den;
    for (int b = 0; b < FINE_BINS; ++b) {
        num[b] = u128(2) * S[std::size_t(0) * FINE_BINS + b];
        den[b] = u128(S[std::size_t(1) * FINE_BINS + b]) + S[std::size_t(2) * FINE_BINS + b];
    }
    std::vector<int> bulk;
    for (int b = 0; b < FINE_BINS; ++b)
        if (((bulk_mask_words[b >> 6] >> (b & 63)) & 1) && den[b] > 0)
            bulk.push_back(b);
    const int n_bulk = int(bulk.size());
    if (cfar_rank >= n_bulk)
        return {0, false, n_bulk, -1, -1};

    // Rank selection by counting: bin i holds the rank value iff
    // (# strictly below) <= rank < (# strictly below) + (# equal).
    int rank_bin = -1;
    for (const int i : bulk) {
        int count_less = 0, count_equal = 0;
        for (const int j : bulk) {
            if (f2_less(num[j], den[j], num[i], den[i]))
                ++count_less;
            else if (!f2_less(num[i], den[i], num[j], den[j]))
                ++count_equal;
        }
        if (count_less <= cfar_rank && cfar_rank < count_less + count_equal) {
            rank_bin = i;
            break;
        }
    }
    if (rank_bin < 0)
        FATAL_ERROR_NON_OO("cpu_fine_decision: rank selection failed (internal error)");
    const u128 num_r = num[rank_bin];
    const u128 den_r = den[rank_bin];

    // Designated set: anchor +/- half_width, modulo 256, ascending k.
    int fired_bin = -1;
    for (int k = -designated_half_width; k <= designated_half_width; ++k) {
        const int b = ((anchor_bin + k) % FINE_BINS + FINE_BINS) % FINE_BINS;
        if (den[b] <= 0)
            continue; // zero-reference forced 0, per bin
        // num[b] * 2^16 * den_r > multiplier_q16 * num_r * den[b], exact.
        const u256 lhs = mul_u128(num[b] << 16, den_r);
        const u256 rhs = mul_u128(u128(multiplier_q16) * num_r, den[b]);
        if (less_u256(rhs, lhs)) {
            fired_bin = b;
            break;
        }
    }
    return {fired_bin >= 0 ? 1 : 0, true, n_bulk, rank_bin, fired_bin};
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

class testPilotProxyDetector : public kotekan::Stage {
public:
    testPilotProxyDetector(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    virtual ~testPilotProxyDetector() {}

    void main_thread() override {
        using std::ptrdiff_t;
        using std::uint64_t;

        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);

        int K = 0, terms = 0, bits = 0;
        FStat_GetSpecs(&K, &terms, &bits, nullptr);
        if (terms != NUM_TERMS || bits != 4)
            FATAL_ERROR("unexpected compiled kernel contract: terms={:d} bits={:d}", terms, bits);
        int fine_windows = 0, fine_bins = 0;
        FStat_GetFineSpecs(&fine_windows, nullptr, &fine_bins);
        if (fine_bins != FINE_BINS)
            FATAL_ERROR("unexpected fine bins {:d}", fine_bins);

        // Small geometry so the CPU oracles stay cheap while preserving the
        // full logical structure: F coarse channels on the node, P*D streams,
        // the frozen 128 windows per stream.
        const ptrdiff_t num_dishes = 4;
        const ptrdiff_t num_polarizations = 2;
        const ptrdiff_t num_frequencies = 3;
        const ptrdiff_t num_streams = num_polarizations * num_dishes;
        const int windows = fine_windows; // 128: enables the fine entries
        const ptrdiff_t num_time_samples = ptrdiff_t(windows) * K;
        const ptrdiff_t detector_rows = num_streams * windows;
        const ptrdiff_t packed_count = detector_rows * K;

        // Random weights: packed complex int4 with the symmetric component
        // range [-7, 7] (the generated-ROM convention).
        std::uniform_int_distribution<int> weight_component(-7, 7);
        std::vector<std::int8_t> weights(std::size_t(NUM_TERMS) * K);
        for (auto& w : weights) {
            const int real = weight_component(rng), imag = weight_component(rng);
            w = std::int8_t(((real & 0xF) << 4) | (imag & 0xF));
        }

        // Ring geometries: no wrap, and a position that forces the block to
        // wrap around the ring end.
        const ptrdiff_t ring_size = 2 * num_time_samples; // power of two
        const std::array<ptrdiff_t, 3> ring_positions = {0, 5 * ring_size / 8, ring_size - K / 2};

        std::uniform_int_distribution<int> byte_dist(0, 255);
        std::vector<std::uint8_t> ring(std::size_t(ring_size) * num_frequencies * num_polarizations
                                       * num_dishes);
        for (auto& v : ring)
            v = std::uint8_t(byte_dist(rng));

        // Device buffers.
        std::uint8_t* d_ring = nullptr;
        std::int8_t* d_packed = nullptr;
        unsigned long long* d_powers = nullptr;
        std::int32_t* d_row_sums = nullptr;
        unsigned long long* d_fine_powers = nullptr;
        unsigned long long* d_numden = nullptr;
        unsigned char* d_mask_u8 = nullptr;
        int* d_mask_i32 = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d_ring, ring.size()));
        CHECK_CUDA_ERROR(cudaMalloc(&d_packed, packed_count));
        CHECK_CUDA_ERROR(cudaMalloc(&d_powers, NUM_TERMS * sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(
            cudaMalloc(&d_row_sums, std::size_t(NUM_TERMS) * detector_rows * 2 * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_fine_powers, std::size_t(NUM_TERMS) * FINE_BINS
                                                        * sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_numden, 2 * sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mask_u8, 1));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mask_i32, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_ring, ring.data(), ring.size(), cudaMemcpyHostToDevice));

        void* const handle = FStat_Create(d_packed, nullptr, int(detector_rows));
        if (!handle)
            FATAL_ERROR("FStat_Create failed: {:s}", FStat_LastError());

        std::vector<std::int8_t> packed_gpu(packed_count), packed_cpu(packed_count);
        std::vector<std::int32_t> row_sums_gpu(std::size_t(NUM_TERMS) * detector_rows * 2);
        std::vector<std::int32_t> row_sums_cpu(row_sums_gpu.size());
        std::vector<uint64_t> fine_gpu(std::size_t(NUM_TERMS) * FINE_BINS);
        std::vector<uint64_t> fine_cpu;

        int cases = 0;
        for (const bool reverse : {false, true}) {
            for (int freq_index = 0; freq_index < num_frequencies; ++freq_index) {
                for (const ptrdiff_t pos : ring_positions) {
                    // --- Packer: GPU vs CPU, bit for bit.
                    launch_pilotproxy_pack(d_packed, d_ring, num_dishes, num_polarizations,
                                           num_frequencies, freq_index, num_time_samples, ring_size,
                                           pos, K, reverse, /*stream=*/nullptr);
                    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                    CHECK_CUDA_ERROR(cudaMemcpy(packed_gpu.data(), d_packed, packed_count,
                                                cudaMemcpyDeviceToHost));
                    cpu_pilotproxy_pack(packed_cpu.data(), ring.data(), num_dishes,
                                        num_polarizations, num_frequencies, freq_index,
                                        num_time_samples, ring_size, pos, K, reverse);
                    for (ptrdiff_t i = 0; i < packed_count; ++i)
                        if (packed_gpu[i] != packed_cpu[i])
                            FATAL_ERROR("packer mismatch: case(rev={:d},f={:d},pos={:d}) "
                                        "index {:d}: gpu={:d} cpu={:d}",
                                        int(reverse), freq_index, pos, i, packed_gpu[i],
                                        packed_cpu[i]);

                    // --- Coarse powers: GPU vs vendored CPU reference.
                    unsigned long long powers_gpu[NUM_TERMS];
                    unsigned long long powers_cpu[NUM_TERMS];
                    FStat_Compute_Powers_U64(handle, weights.data(), d_powers);
                    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                    CHECK_CUDA_ERROR(cudaMemcpy(powers_gpu, d_powers, sizeof powers_gpu,
                                                cudaMemcpyDeviceToHost));
                    fstat_ref_powers_u64(packed_cpu.data(), weights.data(), int(detector_rows),
                                         powers_cpu);
                    for (int t = 0; t < NUM_TERMS; ++t)
                        if (powers_gpu[t] != powers_cpu[t])
                            FATAL_ERROR("coarse power mismatch: term {:d}: gpu={:d} cpu={:d}", t,
                                        powers_gpu[t], powers_cpu[t]);

                    // --- Row sums: GPU vs vendored CPU reference.
                    FStat_Compute_RowSums_I32(handle, weights.data(), d_row_sums);
                    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                    CHECK_CUDA_ERROR(cudaMemcpy(row_sums_gpu.data(), d_row_sums,
                                                row_sums_gpu.size() * sizeof(std::int32_t),
                                                cudaMemcpyDeviceToHost));
                    fstat_ref_row_sums_i32(packed_cpu.data(), weights.data(), int(detector_rows),
                                           row_sums_cpu.data());
                    if (row_sums_gpu != row_sums_cpu)
                        FATAL_ERROR("row-sum mismatch (rev={:d},f={:d},pos={:d})", int(reverse),
                                    freq_index, pos);

                    // --- Fused fine powers: GPU vs fxfft256 v1 + exact feed
                    // sums over the CPU row sums; coarse marginals must
                    // reproduce Powers_U64 exactly (the marginal identity).
                    unsigned long long fused_powers_gpu[NUM_TERMS];
                    FStat_Compute_FusedFine_U64(handle, weights.data(), d_fine_powers, d_powers,
                                                nullptr);
                    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                    CHECK_CUDA_ERROR(cudaMemcpy(fine_gpu.data(), d_fine_powers,
                                                fine_gpu.size() * sizeof(uint64_t),
                                                cudaMemcpyDeviceToHost));
                    CHECK_CUDA_ERROR(cudaMemcpy(fused_powers_gpu, d_powers, sizeof fused_powers_gpu,
                                                cudaMemcpyDeviceToHost));
                    cpu_fine_powers(row_sums_cpu, int(num_streams), windows, fine_cpu);
                    if (fine_gpu != fine_cpu)
                        FATAL_ERROR("fine power mismatch (rev={:d},f={:d},pos={:d})", int(reverse),
                                    freq_index, pos);
                    for (int t = 0; t < NUM_TERMS; ++t)
                        if (fused_powers_gpu[t] != powers_cpu[t])
                            FATAL_ERROR("fused coarse marginal mismatch: term {:d}", t);

                    // --- Coarse rational mask: exercise both outcomes by
                    // bracketing the measured num/den ratio.
                    const unsigned long long num = powers_cpu[0];
                    const unsigned long long den = powers_cpu[1] + powers_cpu[2];
                    const struct {
                        unsigned long long half_num, half_den;
                    } thresholds[] = {
                        {1, 1000000000ULL},                 // ~always fires (den > 0)
                        {~0ULL / 4, 1},                     // ~never fires
                        {num, 2 * den > den ? 2 * den : 1}, // near the operating point
                    };
                    for (const auto& threshold : thresholds) {
                        unsigned char mask_gpu = 0;
                        unsigned long long numden_gpu[2];
                        FStat_Compute_NumDen_Mask_RationalHalf(
                            handle, weights.data(), threshold.half_num, threshold.half_den,
                            d_numden, d_numden + 1, d_mask_u8);
                        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                        CHECK_CUDA_ERROR(
                            cudaMemcpy(&mask_gpu, d_mask_u8, 1, cudaMemcpyDeviceToHost));
                        CHECK_CUDA_ERROR(cudaMemcpy(numden_gpu, d_numden, sizeof numden_gpu,
                                                    cudaMemcpyDeviceToHost));
                        if (numden_gpu[0] != num || numden_gpu[1] != den)
                            FATAL_ERROR("numden mismatch: gpu=({:d},{:d}) cpu=({:d},{:d})",
                                        numden_gpu[0], numden_gpu[1], num, den);
                        const int mask_cpu =
                            den != 0
                            && less_u256(mul_u128(u128(threshold.half_num), u128(den)),
                                         mul_u128(u128(num), u128(threshold.half_den)));
                        if (int(mask_gpu) != mask_cpu)
                            FATAL_ERROR("coarse mask mismatch: half={:d}/{:d} gpu={:d} cpu={:d}",
                                        threshold.half_num, threshold.half_den, int(mask_gpu),
                                        mask_cpu);
                    }

                    // --- Fine designated-set CFAR mask: GPU epilogue vs the
                    // fine decision v1 port, over several calibrations
                    // (independent-bin bulk; guard-excluded designated set;
                    // a degenerate bulk that must force mask 0 / invalid).
                    const struct {
                        int anchor, half_width, rank;
                        std::uint64_t mult_q16;
                        bool degenerate_bulk;
                    } calibrations[] = {
                        {0, 2, 8, 3ULL << 16, false},
                        {7, 1, 0, 1ULL << 16, false},
                        {250, 3, 20, 5ULL << 15, false},
                        {13, 2, 200, 2ULL << 16, true}, // rank >= |bulk| -> invalid
                    };
                    for (const auto& calibration : calibrations) {
                        std::array<unsigned long long, 4> words = {0, 0, 0, 0};
                        if (calibration.degenerate_bulk) {
                            // Two bulk bins only; rank 200 exceeds them.
                            words[0] = (1ULL << 40) | (1ULL << 50);
                        } else {
                            // Even (independent) bins, excluding the
                            // designated set and one guard bin each side.
                            for (int b = 0; b < FINE_BINS; b += 2) {
                                int distance = (b - calibration.anchor) % FINE_BINS;
                                if (distance < 0)
                                    distance += FINE_BINS;
                                const int circular =
                                    distance <= FINE_BINS / 2 ? distance : FINE_BINS - distance;
                                if (circular <= calibration.half_width + 1)
                                    continue;
                                words[b >> 6] |= 1ULL << (b & 63);
                            }
                        }
                        int mask_gpu_i32 = -1;
                        FStat_Compute_FusedFineMask_U64(
                            handle, weights.data(), calibration.anchor, calibration.half_width,
                            words.data(), calibration.rank, calibration.mult_q16, d_fine_powers,
                            d_mask_i32, nullptr, nullptr);
                        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                        CHECK_CUDA_ERROR(cudaMemcpy(&mask_gpu_i32, d_mask_i32, sizeof(int),
                                                    cudaMemcpyDeviceToHost));
                        const FineDecisionRef ref =
                            cpu_fine_decision(fine_cpu, calibration.anchor, calibration.half_width,
                                              words, calibration.rank, calibration.mult_q16);
                        if (mask_gpu_i32 != ref.mask)
                            FATAL_ERROR("fine mask mismatch: anchor={:d} w={:d} rank={:d} "
                                        "mult={:d}: gpu={:d} cpu={:d} (n_bulk={:d} valid={:d})",
                                        calibration.anchor, calibration.half_width,
                                        calibration.rank, calibration.mult_q16, mask_gpu_i32,
                                        ref.mask, ref.n_bulk, int(ref.valid));
                    }

                    ++cases;
                }
            }
        }

        FStat_Destroy(handle);
        CHECK_CUDA_ERROR(cudaFree(d_ring));
        CHECK_CUDA_ERROR(cudaFree(d_packed));
        CHECK_CUDA_ERROR(cudaFree(d_powers));
        CHECK_CUDA_ERROR(cudaFree(d_row_sums));
        CHECK_CUDA_ERROR(cudaFree(d_fine_powers));
        CHECK_CUDA_ERROR(cudaFree(d_numden));
        CHECK_CUDA_ERROR(cudaFree(d_mask_u8));
        CHECK_CUDA_ERROR(cudaFree(d_mask_i32));

        INFO("testPilotProxyDetector: all GPU-vs-CPU products matched over {:d} cases.", cases);
        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testPilotProxyDetector);
