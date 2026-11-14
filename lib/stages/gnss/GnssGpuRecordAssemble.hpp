#ifndef GNSS_GPU_RECORD_ASSEMBLE_HPP
#define GNSS_GPU_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssRecordChain.hpp"

#include <string>

/**
 * @class GnssGpuRecordAssemble
 * @brief Host tail of the GPU tracking chain: gnssGpuChain frames -> tracker records.
 *
 * Consumes the cudaProcess output (a control block and raw per-channel E/P/L correlations,
 * laid out in gnssGpuChain.hpp). For each PRN it sums the correlations over the PRN's covering
 * channel mask, integrates and applies the carrier-NCO phase (the phase state lives here; the
 * slope f_nco comes in the control block), and writes the gnssRecord.hpp record that the
 * combiner, broker and viewer read. Emits one out_buf frame per record, with the record's
 * start sample (wstart) in GnssChanMetadata::sample_seq. The assembly itself is a
 * gnss::RecordChain; this stage moves its frames.
 *
 * @conf in_buf   gnssGpuChain frames from the cudaProcess chain
 * @conf out_buf  tracker records: n_prn * record_stride floats, plus the per-channel block
 *                with chan_export
 * @conf prns     initial slot -> PRN map; its length must equal the frame's n_prn
 * @conf sample_rate  rate of the wstart sample counter, Hz
 * @conf record_cp_mod_chips  Float (default 0 = off). Reduce record slot 2 (the code-phase
 *                argument, float32) mod this many chips: the primary code length on a chain
 *                whose baked code is long enough to cost float32 precision (GLO_L2OC_P_OC).
 * @conf elem_proj_sys  String (default "G"). The constellation letter of the chain's PRNs
 *                (G, E, C or R), which names its satellites in the bright-satellite projection.
 */
class GnssGpuRecordAssemble : public kotekan::Stage {
public:
    /// @p clock_s is the chain's clock (gnss::RecordChain); tests pass their own.
    GnssGpuRecordAssemble(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container,
                          double (*clock_s)() = gnss::steady_now_s);
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    /// Push leg: each completed window is pushed to cube_buf with its address (chain, window
    /// index, wstart). It never blocks the tracker: if the next frame is not empty (safe to test,
    /// as this stage is the only producer) the window is dropped and counted in _cube_dropped,
    /// which every frame carries, so a gap in the window index without a rise in the count was
    /// not dropped here.
    Buffer* _cube_out_buf = nullptr;
    int _cube_out_id = 0;
    gnss::RecordChain _chain;
    /// Pack the chain's completed window into `_cube_out_buf` and mark it full, or count it
    /// dropped if the next frame is not empty.
    void emit_cube_window();
};

#endif
