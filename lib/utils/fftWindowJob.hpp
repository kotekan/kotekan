/**
 * @file
 * @brief The FFT window job: the frame layout crsWindowAlign produces and the F-engine
 *        stages (GPU and CPU reference) consume, and the output spectrum convention.
 *
 * A job describes one FFT window of one ADC: fft_length contiguous samples starting at an
 * ADC sample number that is a multiple of fft_length, held in samples_per_packet-sample
 * packets. The lookup table maps each packet position of the window, in time order, to the
 * address of its payload slot in the receive payload buffer, or, for a packet that never
 * arrived, to a slot of the noise buffer. Slot addresses stay valid because frames of those
 * buffers are never reallocated or swapped (ibvFrameMRs checks this on the receive side).
 *
 * Producer/consumer protocol. The aligner consumes only the header buffer and never touches
 * payload data; the FFT stage is the sole registered consumer of the payload buffer. Both
 * number payload frames with an absolute counter that starts at 0 on the first frame. The
 * FFT stage keeps two cursors, acquired and released, both starting at 0. Per job it
 * acquires: while acquired < acquire_up_to, wait_for_full_frame(acquired++ % num_frames) —
 * required even for frames the window never references (alignment skips), and done frame by
 * frame because the receive stage marks a header frame full one statement before its payload
 * frame, so a job can exist a moment before its last payload frame is full. After processing
 * the window it releases: while released < release_up_to, mark_frame_empty(released++ %
 * num_frames). A frame shared with the next window stays held: release_up_to <
 * acquire_up_to then. The aligner guarantees released <= acquired at every job and sizes
 * nothing beyond the frames that exist: the payload buffer must hold at least the frames one
 * window spans plus what the receive stage posts ahead, plus slack.
 *
 * Output spectrum convention, common to every implementation: with x[n] = (raw[n] - 2048)
 * / 2048 the window's samples, the spectrum is X[k] = (2 / fft_length) * sum_n x[n] *
 * exp(-2 pi i k n / fft_length) for k = 0 .. fft_length/2 - 1 (DC kept as bin 0, Nyquist
 * dropped), so a tone of amplitude A ADC half-scales lands at |X[k]| = A for any
 * fft_length. Bin k is quantized as int4x2_swapped_withoffset of gain[k] * X[k]: real and
 * imaginary parts rounded half to even, clamped to [-7, 7], offset by +8, real in the high
 * nibble (0x88 is complex zero).
 */
#ifndef FFT_WINDOW_JOB_HPP
#define FFT_WINDOW_JOB_HPP

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint64_t

#pragma pack(push, 1)
struct fft_window_job {
    /// ADC sample number of the window start; a multiple of the FFT length.
    uint64_t start_seq;
    /// start_seq / fft_length: the output time code, and the CRS seq_number of the
    /// window's output packets.
    uint64_t window_index;
    /// fft_length / samples_per_packet: the length of the lookup table.
    uint32_t num_packets;
    /// Lookup table entries that point into the noise buffer.
    uint32_t lost_packets;
    /// The consumer's acquired cursor after this window: every payload frame below this
    /// absolute counter is full and must have been waited for.
    uint64_t acquire_up_to;
    /// The consumer's released cursor after this window: every payload frame below this
    /// absolute counter is done and must be released. At most acquire_up_to.
    uint64_t release_up_to;
    /// Bit 0: the first window after a (re)alignment.
    uint32_t flags;
    uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(fft_window_job) == 48);

constexpr uint32_t FFT_WINDOW_REALIGNED = 1;

/// The lookup table stored right after the job in its frame: num_packets slot addresses.
inline uint64_t* fft_window_lut(fft_window_job* job) {
    return reinterpret_cast<uint64_t*>(job + 1);
}

inline const uint64_t* fft_window_lut(const fft_window_job* job) {
    return reinterpret_cast<const uint64_t*>(job + 1);
}

/// The job-buffer frame size that holds a window of num_packets packets.
inline size_t fft_window_job_size(size_t num_packets) {
    return sizeof(fft_window_job) + num_packets * sizeof(uint64_t);
}

#endif /* FFT_WINDOW_JOB_HPP */
