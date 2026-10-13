/**
 * @file
 * @brief Unpack a CHORD 4+4b channelized stream to complex float for the search.
 *  - GnssChordDequantize : public kotekan::Stage
 */

#ifndef GNSS_CHORD_DEQUANTIZE_HPP
#define GNSS_CHORD_DEQUANTIZE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <string>

/**
 * @class GnssChordDequantize
 * @brief [hop][chan] 4+4b bytes -> [hop][chan] cfloat32.
 *
 * Converts CHORD's 4+4-bit channelized stream to the complex float @ref GnssChannelizedSearch
 * takes (the format of @ref fftwEngine's output). Acquisition runs on one element, so this is a
 * small stream and the conversion is not on a hot path.
 *
 * Decode: gnss44::unpack (high nibble = real, low nibble = imag, each stored as value + 8, as in
 * lib/testing/gpuSimulate.cpp). This is the layout of int4x2_swapped_withoffset_t (DataType.hpp),
 * whose component [0], the low nibble, is imag. Swapping the nibbles gives i*conj(x): the
 * magnitudes are unchanged, so power-only checks cannot see it, but the despread does not
 * match the replica. The imaginary part is negated, which the CHORD F-engine's output needs.
 *
 * Output values are in lsb; the search normalizes.
 *
 * @conf in_buf        [hop][chan] 4+4b bytes (a GnssChordVoltageTap output, one element)
 * @conf out_buf       [hop][chan] cfloat32
 * @conf n_channels    channels in the input stream
 * @conf n_elements    must be 1 (default 1)
 * @conf element       must be 0 (default 0)
 * @conf samples_per_data_set  hops per frame
 * @conf conjugate     must be true (default true)
 *
 * Both buffers carry GnssChanMetadata; each frame's sample_seq is copied to its output.
 */
class GnssChordDequantize : public kotekan::Stage {
public:
    GnssChordDequantize(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~GnssChordDequantize() override = default;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    int _n_chan;
    int _n_hops;
};

#endif // GNSS_CHORD_DEQUANTIZE_HPP
