/*****************************************
@file
@brief Gather N2 autocorrelation power across frequency.
- N2AutoSpectrum : public kotekan::Stage
*****************************************/
#ifndef N2_AUTO_SPECTRUM_HPP
#define N2_AUTO_SPECTRUM_HPP

#include "Config.hpp"          // for Config
#include "N2FrameDesc.hpp"     // for N2FrameDesc
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint> // for uint32_t, int64_t
#include <memory>  // for shared_ptr
#include <string>  // for string
#include <vector>  // for vector


/**
 * @class N2AutoSpectrum
 * @brief ``kotekan::Stage`` that gathers the autocorrelations of N2 frames
 * across frequency into a float32 [freq, element] power spectrum.
 *
 * N2 frames arrive one per (accumulation bin, frequency). For each bin, this
 * stage fills one output frame with the real part of the autocorrelations of
 * each incoming frame at row ``freq_id - freq_id_base``. Rows for
 * frequencies that never arrive stay NaN.
 *
 * Up to ``num_pending_bins`` output frames are held open at once; the oldest
 * is flushed when a frame from a newer bin needs a slot, so bin lifetime is
 * driven entirely by frame arrival (no timers). Frames for bins at or below
 * the newest open bin that are not themselves open are dropped.
 *
 * The output buffer is intended to be watched live through its
 * ``/buffer/<name>/frame`` peek endpoint (``peek_hold: true``), so it may
 * have no downstream consumer.
 *
 * @par Buffers
 * @buffer in_buf The accumulated visibilities, one frame per (bin, frequency).
 *     @buffer_format N2Buffer (any layout containing all autocorrelations)
 *     @buffer_metadata N2Metadata
 * @buffer out_buf Autocorrelation power gathered across frequency.
 *     @buffer_format NDArray float32 "autospectrum" [num_local_freq, num_elements] {F, E}
 *     @buffer_metadata chordMetadata
 *
 * @conf num_local_freq   Int. Number of frequency rows in the output.
 * @conf freq_id_base     Int, default 0. freq_id mapped to output row 0.
 * @conf num_pending_bins Int, default 2. Accumulation bins held open at once.
 *                        Each extra bin tolerates one more bin of arrival skew
 *                        between senders, at the cost of one bin of latency.
 *                        Must leave at least one free output buffer frame.
 */
class N2AutoSpectrum : public kotekan::Stage {

public:
    /// Constructor. Validates buffer descriptors and locates the autocorrelations.
    N2AutoSpectrum(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Primary loop: scatter incoming autos into per-bin output frames.
    void main_thread() override;

private:
    /// Input buffer of N2 visibilities
    Buffer* in_buf;

    /// Output spectrum buffer
    Buffer* out_buf;

    /// Frame descriptor for the input buffer
    std::shared_ptr<const kotekan::N2FrameDesc> in_desc;

    /// Number of frequency rows in the output
    const int64_t _num_local_freq;

    /// freq_id mapped to output row 0
    const int64_t _freq_id_base;

    /// Number of accumulation bins held open at once
    const uint32_t _num_pending_bins;

    /// Number of elements in the input buffer
    uint32_t _num_elements;

    /// For each element, the index of its autocorrelation in the input products
    std::vector<uint32_t> auto_prod_index;
};


#endif
