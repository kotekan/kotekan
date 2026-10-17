#include "GnssN2RecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "gnssGpuChain.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssN2RecordAssemble);

GnssN2RecordAssemble::GnssN2RecordAssemble(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssN2RecordAssemble::main_thread, this)) {
    _ctl_buf = get_buffer("ctl_buf");
    _ctl_buf->register_consumer(unique_name);
    _tiles_buf = get_buffer("tiles_buf");
    _tiles_buf->register_consumer(unique_name);
    _out_buf = get_buffer("out_buf");
    _out_buf->register_producer(unique_name);

    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_synth = config.get_default<int>(unique_name, "num_synth", 128);
    _lanes_per_slot =
        config.get_default<int>(unique_name, "lanes_per_slot", gnss_gpu::LANES_FULL);
    _n_live = config.get<int>(unique_name, "n_live_elements");
    _n_gnss_chan = config.get<int>(unique_name, "n_gnss_channels");
    _n_prn = config.get<int>(unique_name, "n_prn");
    _hops_per_record = config.get<int>(unique_name, "hops_per_record");
    _n_hops_frame = config.get<int>(unique_name, "samples_per_data_set");

    // Must mirror cudaCorrelatorDual::build_tile_selection and gen_chord_gnss_config.py's
    // n2_tiles_per_chan. The frame-size check below catches a count mismatch; an order
    // mismatch reads as garbage correlations.
    _na16 = _num_elements / 16;
    _nt16 = (_num_elements + _num_synth) / 16;
    _nlive16 = (_n_live + 15) / 16;
    // A chain's gather carries the mixed block alone; the AA and BB triangles are gathered
    // only for the visibility capture, which has its own gather. The live configs still carry
    // gnss_gather_aa, so it is read only to refuse true.
    if (config.get_default<bool>(unique_name, "gnss_gather_aa", false))
        FATAL_ERROR("GnssN2RecordAssemble: gnss_gather_aa true is not supported; a chain's "
                    "tiles carry the mixed block only");
    _n_tile = (_nt16 - _na16) * _nlive16;

    if (_lanes_per_slot != gnss_gpu::LANES_FULL && _lanes_per_slot != gnss_gpu::LANES_PROMPT)
        FATAL_ERROR("GnssN2RecordAssemble: lanes_per_slot {:d} must be {:d} or {:d}",
                    _lanes_per_slot, gnss_gpu::LANES_FULL, gnss_gpu::LANES_PROMPT);
    if (_lanes_per_slot == gnss_gpu::LANES_PROMPT && _n_gnss_chan > gnss_gpu::MAX_ROT_CHAN)
        FATAL_ERROR("GnssN2RecordAssemble: {:d} channels exceed the {:d} PrnCtl::rot carries; "
                    "lanes_per_slot {:d} needs every E/L lane packed",
                    _n_gnss_chan, gnss_gpu::MAX_ROT_CHAN, _lanes_per_slot);
    if (_lanes_per_slot * _n_prn > _num_synth)
        FATAL_ERROR("GnssN2RecordAssemble: {:d}*n_prn ({:d}) exceeds num_synth ({:d})",
                    _lanes_per_slot, _lanes_per_slot * _n_prn, _num_synth);

    const size_t need = gnss_gpu::frame_bytes(_n_prn, _n_gnss_chan, gnss_gpu::ROWS_PLAIN, _n_live);
    if ((size_t)_out_buf->frame_size < need)
        FATAL_ERROR("GnssN2RecordAssemble: out_buf frame_size {:d} < {:d} needed for {:d} PRN x "
                    "{:d} chan x {:d} elem",
                    (size_t)_out_buf->frame_size, need, _n_prn, _n_gnss_chan, _n_live);

    // The tiles frame is n_rec x n_chan x n_tile x (16x16 complex int32), sized by the config
    // generator. A mismatch means its geometry (n2_tiles_per_chan) and this stage's disagree.
    const size_t tiles_need =
        (size_t)(_n_hops_frame / _hops_per_record) * _n_gnss_chan * _n_tile * 512 * sizeof(int32_t);
    if ((size_t)_tiles_buf->frame_size != tiles_need)
        FATAL_ERROR("GnssN2RecordAssemble: tiles_buf frame_size {:d} != {:d} for {:d} rec x {:d} "
                    "chan x {:d} mixed tiles",
                    (size_t)_tiles_buf->frame_size, tiles_need, _n_hops_frame / _hops_per_record,
                    _n_gnss_chan, _n_tile);

    INFO("GnssN2RecordAssemble: {:d} PRN x {:d} chan x {:d} elem; {:d} mixed tiles per channel, "
         "{:d} lanes per slot",
         _n_prn, _n_gnss_chan, _n_live, _n_tile, _lanes_per_slot);
}

void GnssN2RecordAssemble::main_thread() {
    frameID ctl_id(_ctl_buf), tiles_id(_tiles_buf), out_id(_out_buf);
    uint64_t n_frames = 0;

    while (!stop_thread) {
        uint8_t* ctl = _ctl_buf->wait_for_full_frame(unique_name, ctl_id);
        if (ctl == nullptr)
            break;
        int32_t* tiles = (int32_t*)_tiles_buf->wait_for_full_frame(unique_name, tiles_id);
        if (tiles == nullptr)
            break;
        uint8_t* out = _out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        const auto* ihdr = (const gnss_gpu::FrameHdr*)ctl;
        const auto* iwin = (const int64_t*)(ctl + gnss_gpu::off_winstart());
        const auto* ictl = (const gnss_gpu::PrnCtl*)(ctl + gnss_gpu::off_prnctl());
        const size_t ctl_off_energy =
            gnss_gpu::off_prnctl() + sizeof(gnss_gpu::PrnCtl) * gnss_gpu::MAX_REC * _n_prn;
        const auto* ienergy = (const double*)(ctl + ctl_off_energy);

        const int n_rec = ihdr->n_rec;
        const int n_chan = ihdr->n_chan;
        // The producer's lane count is in the frame; a mismatch would read P's tile as E's.
        if (gnss_gpu::lanes_per_slot_of(ihdr->lanes_per_slot) != _lanes_per_slot)
            FATAL_ERROR("GnssN2RecordAssemble: the control block says {:d} lanes per slot, this "
                        "stage is configured for {:d} -- cudaGnssInject and this stage must "
                        "carry the same lanes_per_slot",
                        gnss_gpu::lanes_per_slot_of(ihdr->lanes_per_slot), _lanes_per_slot);

        std::memset(out, 0, _out_buf->frame_size);
        auto* ohdr = (gnss_gpu::FrameHdr*)out;
        auto* owin = (int64_t*)(out + gnss_gpu::off_winstart());
        auto* octl = (gnss_gpu::PrnCtl*)(out + gnss_gpu::off_prnctl());
        auto* ocorr = (double*)(out + gnss_gpu::off_corr(_n_prn)); // [job][chan][elem] x2
        auto* oenergy =
            (double*)(out + gnss_gpu::off_energy(_n_prn, n_chan, gnss_gpu::ROWS_PLAIN, _n_live));

        // One output record per sub-integration: with sub_integration_ntime = hops_per_record
        // the N^2 emits n_rec visibilities per frame, one per record the injector synthesised
        // (see the class note).
        ohdr->n_rec = n_rec;
        ohdr->n_prn = _n_prn;
        ohdr->n_chan = n_chan;
        ohdr->n_rows_spec = gnss_gpu::ROWS_PLAIN;
        ohdr->seq0 = ihdr->seq0;
        ohdr->utc0 = ihdr->utc0;

        const size_t tile_slice = (size_t)_n_gnss_chan * _n_tile * 512; // one sub-integration
        int n_out_jobs = 0;
        for (int r = 0; r < n_rec; ++r) {
            owin[r] = iwin[r];
            for (int p = 0; p < _n_prn; ++p) {
                const gnss_gpu::PrnCtl& in0 = ictl[(size_t)r * _n_prn + p];
                gnss_gpu::PrnCtl& oc = octl[(size_t)r * _n_prn + p];
                // Copy the PRN before the run check: a dark slot would otherwise ship 0 ("not
                // stamped"), and the assembler must learn that a slot changed satellite while
                // dark, since that is when it drops the old satellite's element cal.
                oc.prn = in0.prn;
                if (!in0.run)
                    continue;
                oc = in0;
                oc.job0 = n_out_jobs * gnss_gpu::ROWS_PLAIN;

                // Rows in lane order, P first: under LANES_PROMPT the E and L rows are formed
                // from the finished P row, so it must be in the frame before they are written.
                static constexpr int ROW_ORDER[gnss_gpu::ROWS_PLAIN] = {
                    gnss_gpu::ROW_P, gnss_gpu::ROW_PH, gnss_gpu::ROW_E, gnss_gpu::ROW_L};
                for (int ti = 0; ti < gnss_gpu::ROWS_PLAIN; ++ti) {
                    const int t = ROW_ORDER[ti];
                    const int lr = gnss_gpu::lane_of_row(_lanes_per_slot, t);
                    const size_t orow = (size_t)(oc.job0 + t);
                    if (lr < 0) {
                        // No lane for this trial: the prompt row rotated per channel by the
                        // kernel's rotation, corr_E = corr_P e^{-i rot}, corr_L = corr_P
                        // e^{+i rot} (PrnCtl::rot and the sign convention there), with the
                        // kernel's own E/L energy rows, which equal the prompt's on this path.
                        const size_t prow = (size_t)(oc.job0 + gnss_gpu::ROW_P);
                        const double sgn = (t == gnss_gpu::ROW_E) ? -1.0 : 1.0;
                        for (int c = 0; c < n_chan && c < _n_gnss_chan; ++c) {
                            oenergy[orow * n_chan + c] =
                                ienergy[(size_t)(in0.job0 + t) * n_chan + c];
                            const double cr = std::cos(sgn * (double)in0.rot[c]);
                            const double ci = std::sin(sgn * (double)in0.rot[c]);
                            for (int e = 0; e < _n_live; ++e) {
                                const size_t pi = ((prow * n_chan + c) * (size_t)_n_live + e) * 2;
                                const size_t oi = ((orow * n_chan + c) * (size_t)_n_live + e) * 2;
                                const double re = ocorr[pi], im = ocorr[pi + 1];
                                ocorr[oi + 0] = re * cr - im * ci;
                                ocorr[oi + 1] = re * ci + im * cr;
                            }
                        }
                        continue;
                    }
                    const int lane = _lanes_per_slot * p + lr;
                    const int gi = _num_elements + lane, ihi = gi >> 4, ilo = gi & 15;

                    for (int c = 0; c < n_chan && c < _n_gnss_chan; ++c) {
                        // The gathered tiles are compacted: tile k of channel c lives at
                        // c*_n_tile + k, not at its full-triangle offset.
                        const int32_t* slice =
                            tiles + (size_t)r * tile_slice + (size_t)c * _n_tile * 512;

                        // The pack's scale is frozen frame-wide at record 0's energy
                        // (cudaGnssInject), so read s from record 0's rows for this PRN; the
                        // active PRN set is fixed per frame, so those rows exist. This matches the
                        // pack for the E, P and L lanes; the pack takes the P_HEAD lane's rms over
                        // its m_head hops instead of hops_per_record.
                        const double e0 = ienergy[(size_t)(ictl[p].job0 + t) * n_chan + c];
                        const double rms =
                            (e0 > 0.0) ? std::sqrt(e0 / (double)_hops_per_record) : 0.0;
                        const double s = (rms > 0.0) ? 7.0 / (3.0 * rms) : 0.0;

                        // Absolute units, not just the right ratio: the combiner uses energy as
                        // an ML combining weight, so it must be the physical replica energy. Emit
                        // this record's E_R as-is and divide the correlation by s (record 0's).
                        const double e_rec = ienergy[(size_t)(in0.job0 + t) * n_chan + c];
                        oenergy[orow * n_chan + c] = e_rec;

                        for (int e = 0; e < _n_live; ++e) {
                            const int jhi = e >> 4, jlo = e & 15;
                            const int mix_k = (ihi - _na16) * _nlive16 + jhi;
                            const int32_t* tv = slice + (size_t)mix_k * 512 + 32 * ilo + 2 * jlo;
                            // The replica is conjugated in the pack, so V needs no conjugate
                            // here (see the class note).
                            const size_t oi = ((orow * n_chan + c) * (size_t)_n_live + e) * 2;
                            const double inv_s = (s > 0.0) ? 1.0 / s : 0.0;
                            ocorr[oi + 0] = (double)tv[0] * inv_s;
                            ocorr[oi + 1] = (double)tv[1] * inv_s;
                        }
                    }
                }
                ++n_out_jobs;
            }
        }
        ohdr->n_jobs = n_out_jobs * gnss_gpu::ROWS_PLAIN;

        // Carry the control block's metadata (the time reference) through.
        auto in_meta = std::dynamic_pointer_cast<GnssChanMetadata>(_ctl_buf->metadata[ctl_id]);
        if (in_meta) {
            auto om = std::dynamic_pointer_cast<GnssChanMetadata>(_out_buf->metadata[out_id]);
            if (om)
                *om = *in_meta;
        }

        if ((++n_frames & 0xFF) == 1)
            INFO("GnssN2RecordAssemble: seq0 {:d}, {:d} job groups -> {:d} rows over {:d} records",
                 (long long)ohdr->seq0, n_out_jobs, (int)ohdr->n_jobs, n_rec);

        _out_buf->mark_frame_full(unique_name, out_id++);
        _ctl_buf->mark_frame_empty(unique_name, ctl_id++);
        _tiles_buf->mark_frame_empty(unique_name, tiles_id++);
    }
}
