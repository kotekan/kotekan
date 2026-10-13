/**
 * @file
 * @brief Turn the GNSS N^2 tiles and cudaGnssInject's control block into the epl frame.
 */

#ifndef GNSS_N2_RECORD_ASSEMBLE_HPP
#define GNSS_N2_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <string>

/**
 * @class GnssN2RecordAssemble
 * @brief N^2 mixed tiles to the epl frame (gnssGpuChain.hpp) that GnssGpuRecordAssemble reads.
 *
 * Emitting the gnssGpuChain.hpp layout lets the record assembler, combiner, record schema and
 * broker serve the N^2 path unchanged. The record assembler reads the record count from the
 * frame header, never hops_per_record.
 *
 * Inputs
 *  - @c ctl_buf   : cudaGnssInject's control block, the epl layout without corr: the frame's
 *                   seq0/utc0, the per-(record, PRN) PrnCtl and the replica energies.
 *  - @c tiles_buf : cudaCorrelatorDual's gathered tiles, [Tc][n_gnss_chan][n_tile][16][16][2]
 *                   int32, in cudaCorrelatorDual::build_tile_selection's order: per channel,
 *                   the mixed tiles (synth tile rows x live tile columns, row-major, a column
 *                   indexed by its position in the correlator's live_element_tiles), then the
 *                   optional AA and BB triangles.
 *
 * Scale. The pack quantises each replica lane to 4+4 bits with a per-lane scale s, so the
 * tile is V = s * corr. The quantiser holds M^2 nearly constant across lanes, so s scales as
 * 1/sqrt(E_R) and V/M^2 would weight the channel combine by the PFB response instead of by
 * SNR. cudaGnssInject has the pack take s from record 0's energy for every record of the
 * frame, so this stage recomputes s from the control block's record-0 energy and emits
 * corr = V/s with the record's own E_R, the units the GPU despread reports. For the P_HEAD
 * lane the pack divides that energy by the record's m_head, not hops_per_record, so the s
 * recomputed here differs from the pack's (see main_thread).
 *
 * Cadence. cudaCorrelatorDual integrates sub_integration_ntime = hops_per_record, so Tc
 * equals the number of records and each tile slice maps 1:1 onto a record the injector
 * synthesised. Records then sit on the hops_per_record grid, which the combiner's per-record
 * derotation and the fleet DLL's exact-hop grouping rely on.
 *
 * Orientation. With the replica conjugated (`conjugate` on, the F-engine convention) the
 * mixed tile equals the despread's sum_t conj(D) conj(R) directly, not its
 * conjugate; the wrong orientation reads as noise at every code phase.
 *
 * @conf  num_elements   Int. Antenna stations in the correlator (128 on the pathfinder).
 * @conf  num_synth      Int. Synthetic stations. Default 128.
 * @conf  n_live_elements Int. Live antennas the tiles cover.
 * @conf  n_gnss_channels Int. Comb channels gathered.
 * @conf  gnss_gather_aa  Bool, default false. The correlator's flag: the tiles frame then
 *                       carries nlive16*(nlive16+1)/2 AA tiles after the mixed block.
 * @conf  gnss_gather_bb  Bool, default false. The correlator's flag: the tiles frame then
 *                       carries nsb16*(nsb16+1)/2 BB tiles after the AA block.
 * @conf  n_prn, hops_per_record, samples_per_data_set
 */
class GnssN2RecordAssemble : public kotekan::Stage {
public:
    GnssN2RecordAssemble(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    ~GnssN2RecordAssemble() = default;
    void main_thread() override;

private:
    Buffer* _ctl_buf;
    Buffer* _tiles_buf;
    Buffer* _out_buf;

    int _num_elements, _num_synth, _n_live, _n_gnss_chan, _n_prn;
    int _hops_per_record, _n_hops_frame;

    /// Tile geometry, recomputed from config exactly as cudaCorrelatorDual does.
    int _na16, _nt16, _nlive16, _n_mixed, _n_aa, _n_bb, _n_tile;
};

#endif // GNSS_N2_RECORD_ASSEMBLE_HPP
