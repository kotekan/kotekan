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
#include <vector>

/**
 * @class GnssChordDequantize
 * @brief [hop][chan][elem] 4+4b bytes -> [hop][chan] cfloat32, one element.
 *
 * Converts CHORD's 4+4-bit channelized stream to the complex float @ref GnssChannelizedSearch
 * takes (the format of @ref fftwEngine's output). Acquisition runs on one element, so this is a
 * small stream and the conversion is not on a hot path.
 *
 * Decode: high nibble = real, low nibble = imag, each stored as value + 8, as in
 * lib/testing/gpuSimulate.cpp. This is the layout of int4x2_swapped_withoffset_t (DataType.hpp),
 * whose component [0], the low nibble, is imag. Swapping the nibbles gives i*conj(x): the
 * magnitudes are unchanged, so power-only checks cannot see it, but the despread does not
 * match the replica. @c conjugate negates the imaginary part, which the CHORD F-engine's
 * output needs.
 *
 * @c scale matters only for absolute amplitude; the search normalizes.
 *
 * Optionally zero-fills to a contiguous span: with @c channel_ids set, input channel c goes to
 * output slot channel_ids[c] - out_offset of an out_channels-wide row. The frame is zeroed
 * first only when out_channels differs from n_channels. GnssChannelizedSearch does not need
 * this; it takes a sparse comb through its own channel_ids.
 *
 * @conf in_buf        [hop][chan][elem] 4+4b bytes (a GnssChordVoltageTap output)
 * @conf out_buf       [hop][out_channels] cfloat32
 * @conf n_channels    channels in the input stream
 * @conf n_elements    element stride of the input (default 1)
 * @conf element       which element to extract (default 0)
 * @conf samples_per_data_set  hops per frame
 * @conf scale         lsb -> output units (default 1.0)
 * @conf conjugate     negate the imaginary part (default false)
 * @conf out_channels  width of the output span (default n_channels, i.e. no fill)
 * @conf out_offset    global index of output slot 0 (default 0)
 * @conf channel_ids   global index of each input channel, in input order; required when
 *                     out_channels differs from n_channels
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
    int _n_elem;
    int _element;
    int _n_hops;
    float _scale;
    bool _conjugate = false;   ///< negate imag (CHORD F-engine convention)
    int _out_chan;             ///< output span width (== _n_chan when not zero-filling)
    std::vector<int> _out_idx; ///< output slot for each input channel
};

#endif // GNSS_CHORD_DEQUANTIZE_HPP
