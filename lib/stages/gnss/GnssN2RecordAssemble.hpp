/**
 * @file
 * @brief M5: turn path B's N^2 tiles + control block into the SHIPPED epl frame layout.
 */

#ifndef GNSS_N2_RECORD_ASSEMBLE_HPP
#define GNSS_N2_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstdint>
#include <string>
#include <vector>

/**
 * @class GnssN2RecordAssemble
 * @brief Path B's consumer: N^2 mixed tiles -> the epl frame GnssGpuRecordAssemble already eats.
 *
 * This is the whole point of M5: rather than inventing a second record path, path B emits the
 * EXACT layout cudaGnssChordTrack emits (gnssGpuChain.hpp: FrameHdr + window_start[MAX_REC] +
 * PrnCtl[MAX_REC][n_prn] + corr[jobs][chan][elem] + energy[jobs][chan]), so the shipped
 * assembler, combiner, record schema and broker all work unchanged. The assembler is
 * header-driven -- it never reads hops_per_record -- which is what makes the drop-in possible.
 *
 * INPUTS
 *  - @c ctl_buf   : cudaGnssInject's control block, the epl layout MINUS corr. Carries the
 *                   frame's seq0/utc0, the per-(record,PRN) PrnCtl, and the TRUE replica
 *                   energies (which the scale recovery below needs).
 *  - @c tiles_buf : cudaCorrelatorDual's gathered tiles, [Tc][n_gnss_chan][n_tile][16][16][2]
 *                   int32. Tile order is fixed by cudaCorrelatorDual::build_tile_selection:
 *                   per channel, mixed tiles (ihi in [NA/16, NT/16), jhi in [0, ceil(n_live/16)))
 *                   row-major, then the synthetic triangle.
 *
 * THE TWO THINGS THAT ARE NOT A COPY
 *
 * 1. SCALE. The pack quantizes the replica to 4+4b with a per-lane scale s, so the tile is
 *    V = s * corr and the M^2 diagonal is s^2 * E_R. Because the quantizer normalizes every
 *    lane to a CONSTANT energy (measured: M^2 agrees within 1% across PRNs), s ~ 1/sqrt(E_R):
 *    handing V/M^2 downstream would weight the channel combine by the PFB response instead of
 *    by SNR. We therefore emit corr = V and energy = M^2/s, so corr/energy is the true
 *    amplitude. s is recovered from the control block's record-0 energy, which is exactly the
 *    number the pack used (cudaGnssInject freezes it for the whole frame so ONE s factors out).
 *
 * 2. CADENCE. The N^2 integrates the whole frame (sub_integration_ntime = samples_per_data_set,
 *    production's shape), so path B has ONE correlation per frame while the injector
 *    synthesized n_rec sub-records for replica accuracy. We therefore COLLAPSE to n_rec = 1:
 *    window_start[0] and PrnCtl[0] describe the frame, and the energies are summed over the
 *    sub-records (the frame integral's replica energy is the sum). The replica is
 *    phase-continuous across sub-record boundaries, so the frame tile is a coherent sum, not a
 *    smear. Downstream sees a 4x longer record -- see docs/gnss_gpu_search.md 11.11.
 *
 * ORIENTATION: with conj_replica on (CHORD's F-engine convention) path A == V directly, NOT
 * conj(V) -- measured in n2dualxval. Getting this backwards yields noise at every code phase.
 *
 * @conf  num_elements   Int. Antenna stations in the correlator (128 on the pathfinder).
 * @conf  num_synth      Int. Synthetic stations. Default 128.
 * @conf  n_live_elements Int. Live antennas the tiles cover.
 * @conf  n_gnss_channels Int. Comb channels gathered.
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
    int _na16, _nt16, _nlive16, _n_mixed, _n_bb, _n_tile;

    /// int32 offset of tile (ihi, jhi) within one (t, f) slice of the FULL triangle.
    static size_t tri_off(int ihi, int jhi) {
        return 512 * ((size_t)ihi * (ihi + 1) / 2 + jhi);
    }
};

#endif // GNSS_N2_RECORD_ASSEMBLE_HPP
