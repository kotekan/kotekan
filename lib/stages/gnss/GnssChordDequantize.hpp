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
 * @brief [hop][chan][elem] 4+4b bytes -> [hop][chan] cfloat32, one element.
 *
 * The format seam between the CHORD ingest and @ref GnssChannelizedSearch. The search was
 * written against @ref fftwEngine's complex-float output; CHORD delivers 4+4-bit from the
 * RFSoC. Rather than teach the search a second input type, unpack once here -- acquisition is
 * SINGLE-ANTENNA and runs on one element, so this is a small stream (one element of one
 * subband) and the conversion is not on any hot path.
 *
 * DECODE, and why it is spelled out rather than shared: HIGH nibble = REAL, LOW nibble = IMAG,
 * each stored as value+8 (lib/testing/gpuSimulate.cpp is the authoritative reference model).
 * The type is named int4x2_SWAPPED_withoffset precisely because component [0] is imag. Getting
 * this backwards conjugates the data, which leaves magnitudes -- and therefore the acquisition
 * SNR -- completely unchanged while inverting the Doppler sign. An acquisition search would
 * still "find" the satellite, at the wrong Doppler, and the tracker would never pull in.
 *
 * SCALE is arbitrary for acquisition (the search normalizes), so this emits raw lsb units by
 * default. It matters only if the output is ever used for absolute amplitude.
 *
 * @conf in_buf        [hop][chan][elem] 4+4b bytes (a GnssChordVoltageTap output)
 * @conf out_buf       [hop][chan] cfloat32
 * @conf n_channels    channels in the stream
 * @conf n_elements    element stride of the input (>= 1)
 * @conf element       which element to extract (default 0)
 * @conf samples_per_data_set  hops per frame
 * @conf scale         lsb -> volts (default 1.0)
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
};

#endif // GNSS_CHORD_DEQUANTIZE_HPP
