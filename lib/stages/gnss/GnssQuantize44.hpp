/**
 * @file
 * @brief Gain + quantize the PFB output to CHORD 4+4b voltages (the RFSoC's job, done in software).
 *  - GnssQuantize44 : public kotekan::Stage
 */

#ifndef GNSS_QUANTIZE_44_HPP
#define GNSS_QUANTIZE_44_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <atomic>
#include <string>
#include <vector>

/**
 * @class GnssQuantize44
 * @brief Measure the bandpass, scale each channel to ~N lsb rms, and demote [hop][chan] cfloat32
 *        to CHORD 4+4b bytes.
 *
 * THIS STAGE IS AN RFSoC. In CHORD the PFB and the 4-bit quantizer are upstream in the F-engine
 * hardware and kotekan only ever consumes 4+4b off the wire; here @ref fftwEngine plays the PFB
 * and this stage plays the quantizer, so everything downstream sees exactly what a CHORD node
 * sees. Doing it HERE and not on the GPU is the point: the H2D copy shrinks 8x, and the boundary
 * lands where CHORD's boundary actually is.
 *
 * The bandpass is not flat, so each channel gets its OWN scale. We measure sum|v|^2 per channel
 * over the first @c measure_frames frames, set inv_scale[c] = target_lsb / rms[c] (rms of the
 * COMPLEX sample -- the convention CHORD's "2.1" is stated in), then FREEZE --
 * a drifting gain would smear the very calibration this feeds. (CHORD's RFSoC does the same thing
 * at startup.) Per-channel rail counts are exported: a rail is a clamp, and a rising railfrac is
 * the RFI/CW watchdog (at 2.1 lsb the Gaussian floor is ~0.1%, so a sustained excess is real).
 *
 * Frame layout is UNCHANGED except the sample type: in [hop][N] cfloat32, out [hop][N] uint8.
 * That order is already CHORD's [time][freq] with pol=dish=1 degenerate, so nothing transposes.
 *
 * @conf spectrum_length  int    N channels per hop (must match fftwEngine).
 * @conf target_lsb       float  Noise rms to aim each channel at, in lsb, measured on the
 *                               COMPLEX sample sqrt(E|v|^2) (default 2.1 = CHORD's 4b operating
 *                               point = 1.5 lsb per component; the rail sits ~5 sigma out).
 * @conf measure_frames   int    Frames to average the bandpass over before freezing (default 64).
 * @conf fixed_scales     [float] Optional: skip the measurement and use these inv_scales (volts
 *                               -> lsb, one per channel). For replay/bench determinism.
 *
 * @buffer in_buf   [hop][N] cfloat32 -- the PFB output.
 * @buffer out_buf  [hop][N] uint8    -- 4+4b, real = high nibble, imag = low, offset -8.
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
    Buffer* in_buf;
    Buffer* out_buf;

    int _N = 0;
    float _target_lsb = 2.1f;
    int _measure_frames = 64;

    // Bandpass measurement -> frozen per-channel scales.
    std::vector<double> _sumsq;     // per channel, accumulated over the measure window
    long long _nsamp = 0;           // samples per channel accumulated
    int _frames_measured = 0;
    bool _frozen = false;
    std::vector<float> _inv_scale;  // volts -> lsb, per channel (what the encoder uses)
    std::vector<float> _scale;      // lsb -> volts, per channel (what the despread needs back)

    // Watchdog: per-channel clamp counts + the total samples they are a fraction of.
    std::vector<std::atomic<long long>> _rails;
    std::atomic<long long> _rail_total{0};

    void freeze_scales();
};

#endif // GNSS_QUANTIZE_44_HPP
