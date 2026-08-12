// PilotProxy detector-input packer kernel.
//
// See cudaPilotProxyPacker.hpp for the layout contract.

#include "cudaPilotProxyPacker.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Kernel

// One thread per output element, grid-stride. Output writes are fully
// coalesced (adjacent threads write adjacent taps of one detector row).
// The gathered reads stride by F*P*D bytes between taps as a result of
// the [T, F, P, D] -> [row, tap] corner turn. At the deployed geometry
// (<= 1024 streams x 8192 samples = 8 MiB in + 8 MiB out per 41.9 ms
// detector block), even a fully uncoalesced gather is far below 1% of an
// A40's bandwidth budget. Thus, the simple form is preferred over a
// tiled transpose.
__global__ void pilotproxy_pack(std::int8_t* __restrict__ const packed_out,
                                const std::uint8_t* __restrict__ const voltage_ring,
                                const std::ptrdiff_t num_dishes,
                                const std::ptrdiff_t num_polarizations,
                                const std::ptrdiff_t num_frequencies, const int freq_index,
                                const std::ptrdiff_t num_time_samples,
                                const std::ptrdiff_t ringbuf_mask_t,
                                const std::ptrdiff_t ringbuf_pos_t,
                                const int detector_window_samples, const bool time_reverse_windows) {
    const std::ptrdiff_t num_streams = num_polarizations * num_dishes;
    const std::ptrdiff_t num_elements = num_streams * num_time_samples;
    const std::ptrdiff_t sample_stride = num_frequencies * num_streams; // ring bytes per sample
    const std::ptrdiff_t K = detector_window_samples;

    for (std::ptrdiff_t idx = std::ptrdiff_t(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < num_elements; idx += std::ptrdiff_t(gridDim.x) * blockDim.x) {
        // Output index decomposition: idx = (s * windows + w) * K + k = s * T + w*K + k.
        const std::ptrdiff_t s = idx / num_time_samples;
        const std::ptrdiff_t t_out = idx % num_time_samples; // = w*K + k
        const std::ptrdiff_t w = t_out / K;
        const std::ptrdiff_t k = t_out % K;
        const std::ptrdiff_t k_in = time_reverse_windows ? (K - 1 - k) : k;
        const std::ptrdiff_t t_logical = w * K + k_in;
        const std::ptrdiff_t phys = (ringbuf_pos_t + t_logical) & ringbuf_mask_t;
        // s = p * D + d and the ring inner axes are [P, D], so s is the inner offset directly.
        const std::uint8_t raw =
            voltage_ring[phys * sample_stride + std::ptrdiff_t(freq_index) * num_streams + s];
        packed_out[idx] = std::int8_t(raw ^ std::uint8_t(0x88));
    }
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

    const std::ptrdiff_t num_elements = num_polarizations * num_dishes * num_time_samples;
    constexpr int threads = 256;
    const int max_blocks = 4096;
    const std::ptrdiff_t want_blocks = (num_elements + threads - 1) / threads;
    const int blocks = int(want_blocks < max_blocks ? want_blocks : max_blocks);
    if (num_elements <= 0)
        return;

    pilotproxy_pack<<<blocks, threads, 0, stream>>>(
        packed_out, voltage_ring, num_dishes, num_polarizations, num_frequencies, freq_index,
        num_time_samples, ringbuf_size_t - 1, ringbuf_pos_t, detector_window_samples,
        time_reverse_windows);
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference (the slow, "obviously correct" version; the oracle for the kernel)

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
                        ((phys * num_frequencies + freq_index) * num_polarizations + p)
                            * num_dishes
                        + d;
                    packed_out[out++] = std::int8_t(voltage_ring[byte] ^ std::uint8_t(0x88));
                }
            }
        }
    }
}
