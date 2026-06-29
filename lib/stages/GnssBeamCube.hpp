/**
 * @file
 * @brief Per-channel (frequency-resolved) deep-integrated GNSS amplitude -- the beam cube.
 *  - GnssBeamCube : public kotekan::Stage
 */

#ifndef GNSS_BEAM_CUBE_HPP
#define GNSS_BEAM_CUBE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssBeamCube
 * @brief Records the per-channel matched-filter amplitude over time -- the antenna beam
 *        pattern resolved in FREQUENCY as well as time.
 *
 * @ref GnssCoherentCombiner sums the per-subband despread products into ONE full-band
 * amplitude per PRN, which is the optimal detector but COLLAPSES the frequency axis. The
 * primary science goal here is mapping the antenna beam as a satellite transits, and for a
 * wide signal (L5 spans ~10 MHz) the beam evolves across the band -- so we also want the
 * response per frequency bin. This stage consumes the SAME per-channel tracker record
 * streams the combiner does (one @ref GnssChannelizedTracker output per covering channel),
 * and for each PRN forms the per-channel amplitude
 *   @f$ |A_c| = |G_c| / E_c @f$
 * (G_c = that channel's un-normalized correlation, E_c its replica energy), integrating
 * @f$ \langle|A_c|^2\rangle @f$ over time and emitting @f$ \sqrt{\langle|A_c|^2\rangle} @f$
 * -- the robust incoherent per-channel amplitude (~sqrt(K), no phase needed). The result is
 * a [PRN][channel] beam cube vs time; each channel c is a sub-band centred at its r2c bin,
 * so the recorded amplitudes trace beam(t, frequency).
 *
 * No extra correlation work: it reuses the per-channel tracker outputs (the data is already
 * there) -- it is the combiner without the cross-channel sum. Run it ALONGSIDE the combiner
 * to keep both the optimal full-band |A| and the frequency-resolved cube.
 *
 * Integration matches the combiner: @c block (accumulate K, emit, reset) or @c rolling (EMA
 * with time constant K, bias-corrected, emitted every @c output_every records -- so a weak
 * per-channel response climbs out continuously). The incoherent |A_c| has no nav-bit cap.
 *
 * @conf n_prn  Int. Records (PRNs) per frame; default from the first in-frame size.
 * @conf channel0 Int (default 0). Global r2c bin index of the FIRST input channel, recorded
 *       in each output record so the reader maps channel -> frequency (covering channels are
 *       a contiguous block channel0 .. channel0+n_chan-1).
 * @conf integration_length Int (default 1). block: records/output; rolling: EMA time constant.
 * @conf integration_mode String (default "block"). "block" or "rolling".
 * @conf output_every Int (rolling only; default max(1, integration_length/10)). Records/emit.
 *
 * @buffer in_bufs Per-channel tracker record streams (the combiner's inputs), in channel order.
 * @buffer out_buf Beam-cube records, OUT_HEADER + n_chan |A_c| + UTC(double) floats per PRN:
 *                 [0]=PRN [1]=dop [2]=cp [3]=channel0  then |A_0|..|A_{n_chan-1}|  then UTC at
 *                 slot (OUT_HEADER + n_chan), as a float64. out_record_floats = 4 + n_chan + 2.
 *
 * @author Keith Vanderlinde
 */
class GnssBeamCube : public kotekan::Stage {
public:
    GnssBeamCube(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    void main_thread() override;

    /// Input record layout (matches GnssChannelizedTracker / GnssCoherentCombiner):
    /// 0=PRN 1=dop 2=cp 3=G.re 4=G.im 5=E 6=n_chan, UTC float64 at slot 9.
    static constexpr int IN_RECORD_FLOATS = 11;
    static constexpr int IN_UTC_SLOT = 9;
    /// Output header floats before the per-channel |A| block.
    static constexpr int OUT_HEADER = 4; // PRN, dop, cp, channel0

private:
    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
    int _n_prn;
    int _n_chan;             ///< number of input channels (= in_bufs.size())
    int _channel0;           ///< global r2c bin of the first input channel
    int _out_record_floats;  ///< OUT_HEADER + _n_chan + 2 (UTC double)
    int _out_utc_slot;       ///< = OUT_HEADER + _n_chan
    int _integration_length; ///< block: records/output; rolling: EMA time constant
    bool _rolling;           ///< rolling EMA vs block-and-reset
    int _emit_every;         ///< rolling: records between emits
};

#endif // GNSS_BEAM_CUBE_HPP
