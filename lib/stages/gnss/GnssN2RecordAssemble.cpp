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

    const int hops_per_record = config.get<int>(unique_name, "hops_per_record");
    _shape = {config.get<int>(unique_name, "num_elements"),
              config.get_default<int>(unique_name, "num_synth", 128),
              config.get<int>(unique_name, "n_live_elements"),
              config.get<int>(unique_name, "n_gnss_channels"),
              config.get<int>(unique_name, "n_prn"),
              config.get_default<int>(unique_name, "lanes_per_slot", gnss_gpu::LANES_FULL),
              hops_per_record,
              config.get<int>(unique_name, "samples_per_data_set") / hops_per_record};

    // A chain's gather carries the mixed block alone; the AA and BB triangles are gathered
    // only for the visibility capture, which has its own gather. The live configs still carry
    // gnss_gather_aa, so it is read only to refuse true.
    if (config.get_default<bool>(unique_name, "gnss_gather_aa", false))
        FATAL_ERROR("GnssN2RecordAssemble: gnss_gather_aa true is not supported; a chain's "
                    "tiles carry the mixed block only");
    _shape.check(unique_name, _tiles_buf->frame_size);

    const size_t need =
        gnss_gpu::frame_bytes(_shape.n_prn, _shape.n_chan, gnss_gpu::ROWS_PLAIN, _shape.n_live);
    if ((size_t)_out_buf->frame_size < need)
        FATAL_ERROR("GnssN2RecordAssemble: out_buf frame_size {:d} < {:d} needed for {:d} PRN x "
                    "{:d} chan x {:d} elem",
                    (size_t)_out_buf->frame_size, need, _shape.n_prn, _shape.n_chan, _shape.n_live);

    INFO("GnssN2RecordAssemble: {:d} PRN x {:d} chan x {:d} elem; {:d} mixed tiles per channel, "
         "{:d} lanes per slot",
         _shape.n_prn, _shape.n_chan, _shape.n_live, _shape.n_tile(), _shape.lanes_per_slot);
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

        const int n_out_jobs = _shape.assemble(unique_name, ctl, tiles, out, _out_buf->frame_size);
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
