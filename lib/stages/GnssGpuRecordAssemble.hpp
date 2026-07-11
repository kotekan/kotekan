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
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    std::vector<int> _prns;
    double _sample_rate;

    // NCO state per PRN slot (pass-2's half of the carrier machinery).
    std::vector<double> _phi;
    std::vector<std::complex<double>> _a_prev;
    std::vector<uint8_t> _a_prev_ok;
    std::vector<int64_t> _wstart_prev;
};

#endif
