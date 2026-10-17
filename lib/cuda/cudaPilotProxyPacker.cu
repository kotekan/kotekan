// PilotProxy detector-input packer kernel.
//
// See cudaPilotProxyPacker.hpp for the input and output layouts.

#include "cudaPilotProxyPacker.hpp"
#include "cudaUtils.hpp" // for CHECK_CUDA_ERROR_NON_OO

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Kernel

// Streams per tile: one block transposes one K-sample window of TILE_STREAMS streams.
constexpr int TILE_STREAMS = 128;

// Shared-memory transpose from [T,F,P,D] to [stream,tap], so that both the
// ring reads (along P*D) and the packed writes (along T) are contiguous.
// The tile row is padded by 4 bytes to spread the column reads over banks.
__global__ void
pilotproxy_pack(std::int8_t* __restrict__ const packed_out,
                const std::uint8_t* __restrict__ const voltage_ring,
                const std::ptrdiff_t num_streams, const std::ptrdiff_t num_frequencies,
                const int freq_index, const std::ptrdiff_t num_time_samples,
                const std::ptrdiff_t ringbuf_mask_t, const std::ptrdiff_t ringbuf_pos_t,
                const int detector_window_samples, const bool time_reverse_windows) {
    extern __shared__ std::uint8_t tile[]; // [K][TILE_STREAMS + 4]
    constexpr int row = TILE_STREAMS + 4;
    const int K = detector_window_samples;
    const std::ptrdiff_t w = blockIdx.x;
    const std::ptrdiff_t s0 = std::ptrdiff_t(blockIdx.y) * TILE_STREAMS;
    const std::ptrdiff_t sample_stride = num_frequencies * num_streams; // ring bytes per sample
    const int tid = threadIdx.x;

    // Load: adjacent threads read adjacent streams of one sample. The tile
    // row is the output tap, so a reversed window is stored reversed.
    // s = p * D + d and the ring inner axes are [P, D], so s is the inner offset directly.
    for (int k = tid / TILE_STREAMS; k < K; k += blockDim.x / TILE_STREAMS) {
        const int sl = tid % TILE_STREAMS;
        const std::ptrdiff_t phys = (ringbuf_pos_t + w * K + k) & ringbuf_mask_t;
        if (s0 + sl < num_streams)
            tile[(time_reverse_windows ? K - 1 - k : k) * row + sl] =
                voltage_ring[phys * sample_stride + std::ptrdiff_t(freq_index) * num_streams + s0
                             + sl];
    }
    __syncthreads();

    // Store: adjacent threads write adjacent taps of one stream's window,
    // 64 threads per stream.
    constexpr int lanes = 64;
    for (int sl = tid / lanes; sl < TILE_STREAMS && s0 + sl < num_streams; sl += blockDim.x / lanes)
        for (int k = tid % lanes; k < K; k += lanes)
            packed_out[(s0 + sl) * num_time_samples + w * K + k] =
                std::int8_t(tile[k * row + sl] ^ std::uint8_t(0x88));
}

////////////////////////////////////////////////////////////////////////////////
// Launcher (externally visible)

void launch_pilotproxy_pack(std::int8_t* const packed_out, const std::uint8_t* const voltage_ring,
                            const std::ptrdiff_t num_dishes, const std::ptrdiff_t num_polarizations,
                            const std::ptrdiff_t num_frequencies, const int freq_index,
                            const std::ptrdiff_t num_time_samples,
                            const std::ptrdiff_t ringbuf_size_t, const std::ptrdiff_t ringbuf_pos_t,
                            const int detector_window_samples, const bool time_reverse_windows,
                            const cudaStream_t stream) {
    assert(packed_out);
    assert(voltage_ring);
    assert(num_dishes > 0);
    assert(num_polarizations > 0);
    assert(num_frequencies > 0);
    assert(0 <= freq_index && freq_index < num_frequencies);
    assert(detector_window_samples > 0);
    assert(num_time_samples > 0 && num_time_samples % detector_window_samples == 0);
    // Ring size must be a power of two (wrap by mask, matching the other
    // ring-buffer kernels in this directory).
    assert(ringbuf_size_t > 0 && (ringbuf_size_t & (ringbuf_size_t - 1)) == 0);
    assert(ringbuf_pos_t >= 0);

    const std::ptrdiff_t num_streams = num_polarizations * num_dishes;
    if (num_streams <= 0 || num_time_samples <= 0)
        return;

    constexpr int threads = 256;
    const dim3 blocks(unsigned(num_time_samples / detector_window_samples),
                      unsigned((num_streams + TILE_STREAMS - 1) / TILE_STREAMS));
    const std::size_t shared_bytes = std::size_t(detector_window_samples) * (TILE_STREAMS + 4);
    pilotproxy_pack<<<blocks, threads, shared_bytes, stream>>>(
        packed_out, voltage_ring, num_streams, num_frequencies, freq_index, num_time_samples,
        ringbuf_size_t - 1, ringbuf_pos_t, detector_window_samples, time_reverse_windows);
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference used to check the packer output.

void cpu_pilotproxy_pack(std::int8_t* const packed_out, const std::uint8_t* const voltage_ring,
                         const std::ptrdiff_t num_dishes, const std::ptrdiff_t num_polarizations,
                         const std::ptrdiff_t num_frequencies, const int freq_index,
                         const std::ptrdiff_t num_time_samples, const std::ptrdiff_t ringbuf_size_t,
                         const std::ptrdiff_t ringbuf_pos_t, const int detector_window_samples,
                         const bool time_reverse_windows) {
    assert(packed_out);
    assert(voltage_ring);
    assert(0 <= freq_index && freq_index < num_frequencies);
    assert(detector_window_samples > 0);
    assert(num_time_samples > 0 && num_time_samples % detector_window_samples == 0);
    assert(ringbuf_size_t > 0 && (ringbuf_size_t & (ringbuf_size_t - 1)) == 0);
    assert(ringbuf_pos_t >= 0);

    const std::ptrdiff_t K = detector_window_samples;
    const std::ptrdiff_t windows = num_time_samples / K;
    const std::ptrdiff_t mask = ringbuf_size_t - 1;
    std::ptrdiff_t out = 0;
    for (std::ptrdiff_t p = 0; p < num_polarizations; ++p) {
        for (std::ptrdiff_t d = 0; d < num_dishes; ++d) {
            for (std::ptrdiff_t w = 0; w < windows; ++w) {
                for (std::ptrdiff_t k = 0; k < K; ++k) {
                    const std::ptrdiff_t k_in = time_reverse_windows ? (K - 1 - k) : k;
                    const std::ptrdiff_t phys = (ringbuf_pos_t + w * K + k_in) & mask;
                    const std::ptrdiff_t byte =
                        ((phys * num_frequencies + freq_index) * num_polarizations + p) * num_dishes
                        + d;
                    packed_out[out++] = std::int8_t(voltage_ring[byte] ^ std::uint8_t(0x88));
                }
            }
        }
    }
}
