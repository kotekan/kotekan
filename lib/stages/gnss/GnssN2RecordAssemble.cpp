#include "GnssN2RecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "gnssEplAssemble.hpp"
#include "gnssGpuChain.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp"

#include <cstdint>
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
    _lanes_per_slot = config.get_default<int>(unique_name, "lanes_per_slot", gnss_gpu::LANES_FULL);
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
        // The producer's lane count is in the frame; a mismatch would read P's tile as E's.
        if (gnss_gpu::lanes_per_slot_of(ihdr->lanes_per_slot) != _lanes_per_slot)
            FATAL_ERROR("GnssN2RecordAssemble: the control block says {:d} lanes per slot, this "
                        "stage is configured for {:d} -- cudaGnssInject and this stage must "
                        "carry the same lanes_per_slot",
                        gnss_gpu::lanes_per_slot_of(ihdr->lanes_per_slot), _lanes_per_slot);

        // Every row below n_jobs is written whole, for every channel and element, which needs
        // the frame to have this stage's shape.
        if (ihdr->n_prn != _n_prn || ihdr->n_chan != _n_gnss_chan
            || ihdr->n_rec != _n_hops_frame / _hops_per_record || ihdr->n_rec > gnss_gpu::MAX_REC)
            FATAL_ERROR("GnssN2RecordAssemble: the control block is {:d} PRN x {:d} chan x {:d} "
                        "records, this stage {:d} x {:d} x {:d} (at most {:d} records)",
                        ihdr->n_prn, ihdr->n_chan, ihdr->n_rec, _n_prn, _n_gnss_chan,
                        _n_hops_frame / _hops_per_record, gnss_gpu::MAX_REC);
        const gnss_gpu::TilesView tv{tiles, (size_t)_n_gnss_chan * _n_tile * 512, 0, _n_gnss_chan,
                                     _n_tile};
        const int n_out_jobs =
            gnss_gpu::assemble_epl({ctl}, tv, out, _out_buf->frame_size, _n_prn, _lanes_per_slot,
                                   _num_elements, _n_live, _hops_per_record);
        const auto* ohdr = (const gnss_gpu::FrameHdr*)out;

        // Carry the control block's metadata (the time reference) through.
        auto in_meta = std::dynamic_pointer_cast<GnssChanMetadata>(_ctl_buf->metadata[ctl_id]);
        if (in_meta) {
            auto om = std::dynamic_pointer_cast<GnssChanMetadata>(_out_buf->metadata[out_id]);
            if (om)
                *om = *in_meta;
        }

        if ((++n_frames & 0xFF) == 1)
            INFO("GnssN2RecordAssemble: seq0 {:d}, {:d} job groups -> {:d} rows over {:d} records",
                 (long long)ohdr->seq0, n_out_jobs, (int)ohdr->n_jobs, (int)ohdr->n_rec);

        _out_buf->mark_frame_full(unique_name, out_id++);
        _ctl_buf->mark_frame_empty(unique_name, ctl_id++);
        _tiles_buf->mark_frame_empty(unique_name, tiles_id++);
    }
}
