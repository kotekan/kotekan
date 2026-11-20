/**
 * @file
 * @brief Quantize channelized voltages to CHORD's 4+4-bit format.
 *  - GnssQuantize44 : public kotekan::Stage
 */

#ifndef GNSS_QUANTIZE_44_HPP
#define GNSS_QUANTIZE_44_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssQuantize44
 * @brief Scale each channel of @ref fftwEngine's output to a fixed noise level and quantize it
 *        to 4+4 bits (gnss44.hpp).
 *
 * In CHORD the F-engine quantizes its PFB output to 4+4 bits before kotekan sees it. This stage
 * does the same for a software F-engine, so the stages after it take the format they take on
 * CHORD.
 *
 * The bandpass is not flat, so each channel has its own scale. The stage measures each
 * channel's rms over the first @c measure_frames frames, sets the scale that puts it at
 * @c target_lsb, and keeps that scale from then on. Frames of the measurement are written as
 * zeros (0x88).
 *
 * @par Buffers
 * @buffer in_buf The channelized voltages, [hop][spectrum_length].
 *     @buffer_format Array of @c complex<float>
 *     @buffer_metadata none, or GnssChanMetadata
 * @buffer out_buf The same samples, one byte each.
 *     @buffer_format Array of @c uint8_t: real in the high nibble, imaginary in the low, each
 *                    offset by 8
 *     @buffer_metadata none, or GnssChanMetadata: when both buffers carry it, each frame's
 *                      @c sample_seq is copied across.
 *
 * @conf   spectrum_length  Int. Channels per hop.
 * @conf   target_lsb       Float (default 2.1). Rms of a channel's complex samples after
 *                          scaling, in lsb: 2.1 is 1.5 lsb per component, CHORD's level.
 * @conf   measure_frames   Int (default 64). Frames the scales are measured over.
 *
 * @author Keith Vanderlinde
 */
class GnssQuantize44 : public kotekan::Stage {
public:
    GnssQuantize44(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~GnssQuantize44() override = default;
    void main_thread() override;

private:
    /// Set the scales from the measured power.
    void freeze_scales();

    Buffer* in_buf;
    Buffer* out_buf;

    int _spectrum_length;
    float _target_lsb;
    int _measure_frames;

    /// Per channel, the summed power of the measurement and the samples in it.
    std::vector<double> _sumsq;
    long long _num_measured = 0;
    int _frames_measured = 0;
    /// Volts to lsb, per channel; empty until the measurement is done.
    std::vector<float> _inv_scale;
};

#endif // GNSS_QUANTIZE_44_HPP
