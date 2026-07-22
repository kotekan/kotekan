#ifndef GNSS_GPU_RECORD_ASSEMBLE_HPP
#define GNSS_GPU_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <complex>
#include <vector>

/**
 * @class GnssGpuRecordAssemble
 * @brief Host tail of the phase-F GPU tracking chain: gnssGpuChain frames -> tracker records.
 *
 * Consumes the cudaProcess output (control block + raw per-channel E/P/L correlations, layout
 * gnssGpuChain.hpp) and performs GnssChannelizedTracker's pass-2: cross-channel summation over
 * each PRN's covering mask, the carrier-NCO phase integration + derotation (phase continuity
 * state lives here; the slope f_nco = ctrim + ff rides in the control block), and the
 * gnssRecord.hpp record floats. Emits one rec_buf frame per record window with the window's
 * absolute sample in GnssChanMetadata -- byte-compatible with the CPU tracker's output, so the
 * combiner/broker/viewer are untouched.
 *
 * @conf in_buf   gnssGpuChain frames from the cudaProcess chain
 * @conf out_buf  tracker record frames (n_prn * record_floats * float)
 * @conf prns     PRN list (must match the cudaGnssTrack command's)
 * @conf sample_rate  (for the NCO dt; default 5e6)
 */
class GnssGpuRecordAssemble : public kotekan::Stage {
public:
    GnssGpuRecordAssemble(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~GnssGpuRecordAssemble() override;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    std::vector<int> _prns;
    double _sample_rate;

    // NCO state per PRN slot (pass-2's half of the carrier machinery).
    std::vector<double> _phi;
    std::vector<double> _phi_cyc;   ///< NCO phase, UNWRAPPED, in cycles (the export's time base;
                                    ///< _phi is the same phase wrapped for the rotation)
    std::vector<double> _phi_cmd_prev; ///< previous record's commanded phase (cycles)
    std::vector<uint8_t> _phi_cmd_ok;
    std::vector<double> _fcar_prev; ///< previous record's replica f_ref (to size the re-pin step)
    std::vector<uint8_t> _fcar_prev_ok;
    std::vector<std::complex<double>> _a_prev;
    std::vector<uint8_t> _a_prev_ok;
    std::vector<int64_t> _wstart_prev;

    /// Per-channel PROMPT-phase dump (chan_dump_prn / chan_dump_decim / chan_dump_path):
    /// DIAGNOSTIC (2026-07-21, L5 ADR-wander): the channel-width A/B showed the wander
    /// amplitude depends on the despread channel set (narrow 5-ch = 5-6x WORSE than the
    /// full 10) -> the mechanism lives in the per-channel phases the cross-channel sum
    /// normally hides. For the one listed PRN, every decim-th record writes one line per
    /// covering channel: "utc ch corr_re corr_im energy" (raw, pre-NCO-rotation -- the
    /// cross-channel RELATIVE phases are the observable). ~60 KB/s at 100 Hz x 10 ch.
    int _chan_dump_prn = -1;   ///< PRN number to dump (-1 = disabled)
    int _chan_dump_decim = 10; ///< dump every Nth record of that PRN
    long long _chan_dump_ctr = 0;
    FILE* _chan_dump = nullptr;
};

#endif
