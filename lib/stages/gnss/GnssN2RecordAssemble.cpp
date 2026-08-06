#include "GnssN2RecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "gnssGpuChain.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp"

#include <functional>
#include <cmath>
#include <cstring>

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
    _n_live = config.get<int>(unique_name, "n_live_elements");
    _n_gnss_chan = config.get<int>(unique_name, "n_gnss_channels");
    _n_prn = config.get<int>(unique_name, "n_prn");
    _hops_per_record = config.get<int>(unique_name, "hops_per_record");
    _n_hops_frame = config.get<int>(unique_name, "samples_per_data_set");

    // MUST mirror cudaCorrelatorDual::build_tile_selection exactly -- the two are one contract
    // stated twice, and a mismatch reads as garbage correlations rather than an error.
    _na16 = _num_elements / 16;
    _nt16 = (_num_elements + _num_synth) / 16;
    _nlive16 = (_n_live + 15) / 16;
    _n_mixed = (_nt16 - _na16) * _nlive16;
    _n_bb = (_nt16 - _na16) * (_nt16 - _na16 + 1) / 2;
    _n_tile = _n_mixed + _n_bb;

    if (4 * _n_prn > _num_synth)
        FATAL_ERROR("GnssN2RecordAssemble: 4*n_prn ({:d}) exceeds num_synth ({:d})", 4 * _n_prn,
                    _num_synth);

    const size_t need = gnss_gpu::frame_bytes(_n_prn, _n_gnss_chan, gnss_gpu::ROWS_PLAIN, _n_live);
    if ((size_t)_out_buf->frame_size < need)
        FATAL_ERROR("GnssN2RecordAssemble: out_buf frame_size {:d} < {:d} needed for {:d} PRN x "
                    "{:d} chan x {:d} elem",
                    (size_t)_out_buf->frame_size, need, _n_prn, _n_gnss_chan, _n_live);

    INFO("GnssN2RecordAssemble: {:d} PRN x {:d} chan x {:d} elem; tiles {:d} ({:d} mixed + {:d} "
         "BB) per channel",
         _n_prn, _n_gnss_chan, _n_live, _n_tile, _n_mixed, _n_bb);
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

        std::memset(out, 0, _out_buf->frame_size);
        auto* ohdr = (gnss_gpu::FrameHdr*)out;
        auto* owin = (int64_t*)(out + gnss_gpu::off_winstart());
        auto* octl = (gnss_gpu::PrnCtl*)(out + gnss_gpu::off_prnctl());
        auto* ocorr = (double*)(out + gnss_gpu::off_corr(_n_prn)); // [job][chan][elem] x2
        auto* oenergy =
            (double*)(out
                      + gnss_gpu::off_energy(_n_prn, n_chan, gnss_gpu::ROWS_PLAIN, _n_live));

        // ONE OUTPUT RECORD PER SUB-INTEGRATION. With sub_integration_ntime = hops_per_record
        // the N^2 emits nt_outer = n_rec visibilities per frame, which map 1:1 onto the
        // sub-records the injector synthesized -- so path B's records land on the SAME hop grid
        // as the tracker's. That matters twice: the combiner can rate-search and derotate the
        // per-record phase ramp (path B's amplitude was 0.55 of path A's without it, because
        // the ramp was baked inside a 4x longer coherent sum), and the broker's fleet DLL groups
        // powers by an EXACT pow_hop match, which a frame-length record can never satisfy.
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
            if (!in0.run)
                continue;
            oc = in0;
            oc.job0 = n_out_jobs * gnss_gpu::ROWS_PLAIN;

            for (int t = 0; t < gnss_gpu::ROWS_PLAIN; ++t) {
                const int lane = 4 * p + t;
                const int gi = _num_elements + lane, ihi = gi >> 4, ilo = gi & 15;
                const size_t orow = (size_t)(oc.job0 + t);

                for (int c = 0; c < n_chan && c < _n_gnss_chan; ++c) {
                    // The gathered tiles are a COMPACTED triangle: tile k of channel c lives at
                    // c*_n_tile + k, not at the full-triangle offset.
                    const int32_t* slice =
                        tiles + (size_t)r * tile_slice + (size_t)c * _n_tile * 512;

                    // M^2 diagonal: the quantized replica's own (frame-integrated) energy.
                    const int bb_k = _n_mixed + (ihi - _na16) * (ihi - _na16 + 1) / 2
                                     + (ihi - _na16);
                    const double m2 = (double)slice[(size_t)bb_k * 512 + 32 * ilo + 2 * ilo];

                    // The pack's scale is FROZEN frame-wide at record 0's energy
                    // (cudaGnssInject), so every sub-integration shares one s -- read it from
                    // record 0's rows, not this record's.
                    const double e0 = ienergy[(size_t)(ictl[p].job0 + t) * n_chan + c];
                    const double rms = (e0 > 0.0) ? std::sqrt(e0 / (double)_hops_per_record) : 0.0;
                    const double s = (rms > 0.0) ? 7.0 / (3.0 * rms) : 0.0;

                    // energy = M^2 / s  =>  corr/energy is the TRUE amplitude (see the header).
                    oenergy[orow * n_chan + c] = (s > 0.0) ? m2 / s : 0.0;

                    for (int e = 0; e < _n_live; ++e) {
                        const int jhi = e >> 4, jlo = e & 15;
                        const int mix_k = (ihi - _na16) * _nlive16 + jhi;
                        const int32_t* tv =
                            slice + (size_t)mix_k * 512 + 32 * ilo + 2 * jlo;
                        // Orientation: conj_replica ON => path A == V directly.
                        const size_t oi = ((orow * n_chan + c) * (size_t)_n_live + e) * 2;
                        ocorr[oi + 0] = (double)tv[0];
                        ocorr[oi + 1] = (double)tv[1];
                    }
                }
            }
            ++n_out_jobs;
            }
        }
        ohdr->n_jobs = n_out_jobs * gnss_gpu::ROWS_PLAIN;

        // Carry the tap's time reference through, as the tracker's output does.
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
