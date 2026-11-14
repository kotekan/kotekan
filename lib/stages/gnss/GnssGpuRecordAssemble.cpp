#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "gnssChanPower.hpp"    // for ChanPower
#include "gnssProjSubspace.hpp" // for ProjBoard
#include "visUtil.hpp"          // for frameID

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssGpuRecordAssemble);

const std::vector<std::string> GnssGpuRecordAssemble::ROW_KEYS = {"in_buf", "out_buf", "prns",
                                                                  "channel_ids"};
const std::vector<std::string> GnssGpuRecordAssemble::ROW_ONLY_KEYS = {
    "cube_buf", "elem_proj_sys", "elem_sum_shared_ref", "lane_transfer_freq_ids",
    "record_cp_mod_chips"};

GnssGpuRecordAssemble::Row::Row(Config& config, const gnss::HostRow& row, double (*clock_s)()) :
    chain(config, row.cfg_path, row.name, gnss::ProjBoard::instance(), gnss::ChanPower::instance(),
          clock_s) {}

GnssGpuRecordAssemble::GnssGpuRecordAssemble(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container,
                                             double (*clock_s)()) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssGpuRecordAssemble::main_thread, this)),
    _host_rows(gnss::host_rows(config, unique_name, "_n2assemble", ROW_KEYS, ROW_ONLY_KEYS)) {
    for (const gnss::HostRow& hr : _host_rows) {
        Row& row = _rows.emplace_back(config, hr, clock_s);
        row.in_buf = buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "in_buf"));
        row.in_buf->register_consumer(hr.name);
        row.out_buf = buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "out_buf"));
        row.out_buf->register_producer(hr.name);
        row.chain.check_out_buf(row.out_buf);
        if (row.chain.cube_on()) {
            const std::string cbuf = config.get<std::string>(hr.cfg_path, "cube_buf");
            row.cube_out_buf = buffer_container.get_buffer(cbuf);
            row.cube_out_buf->register_producer(hr.name);
            row.chain.check_cube_buf(cbuf, row.cube_out_buf);
        }
        for (const auto& v : row.chain.rest_verbs()) {
            if (v.get)
                gnss::rest_register_unique(hr.name + "/" + v.verb, v.get);
            else
                gnss::rest_register_unique(hr.name + "/" + v.verb, v.post);
        }
    }
    gnss::rest_register_chains(unique_name, _host_rows);
}

void GnssGpuRecordAssemble::main_thread() {
    gnss::run_rows(_rows.size(), get_cpu_affinity(),
                   std::bind(&GnssGpuRecordAssemble::run_row, this, std::placeholders::_1));
}

void GnssGpuRecordAssemble::run_row(size_t i) {
    Row& row = _rows[i];
    const std::string& name = _host_rows[i].name;
    Buffer* in_buf = row.in_buf;
    Buffer* out_buf = row.out_buf;
    frameID frame_in(in_buf), frame_out(out_buf);

    while (!stop_thread) {
        const uint8_t* in = in_buf->wait_for_full_frame(name, frame_in);
        if (in == nullptr)
            return;
        const int n_rec = row.chain.begin_frame(in);
        for (int r = 0; r < n_rec && !stop_thread; ++r) {
            float* out = (float*)out_buf->wait_for_empty_frame(name, frame_out);
            if (out == nullptr)
                return;
            const int64_t wstart = row.chain.assemble_record(r, out);
            emit_cube_window(i);
            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = wstart;
            }
            out_buf->mark_frame_full(name, frame_out);
            frame_out++;
        }
        in_buf->mark_frame_empty(name, frame_in);
        frame_in++;
    }
}

void GnssGpuRecordAssemble::emit_cube_window(size_t i) {
    Row& row = _rows[i];
    if (!row.chain.cube_window_ready())
        return;
    // Never blocks (see the header): this stage is in the real-time path, so a full buffer
    // costs the archive a window, never the tracker a record.
    // is_frame_empty() is a safe pre-check because this row is the buffer's only producer.
    if (!row.cube_out_buf->is_frame_empty(row.cube_out_id)) {
        row.chain.take_cube_window(nullptr);
        return;
    }
    const std::string& name = _host_rows[i].name;
    uint8_t* f = row.cube_out_buf->wait_for_empty_frame(name, row.cube_out_id);
    if (f == nullptr)
        return; // shutting down
    const int64_t w0 = row.chain.take_cube_window(f);
    // A frame that feeds bufferSend needs a metadata object: bufferSend drops a frame without
    // one. Stamped like the record leg: the window's first sample, so a consumer after
    // bufferRecv can address the frame without parsing it.
    if (row.cube_out_buf->metadata_pool) {
        row.cube_out_buf->allocate_new_metadata_object(row.cube_out_id);
        if (auto* m = get_gnss_chan_metadata(row.cube_out_buf, row.cube_out_id))
            m->sample_seq = w0;
    }
    row.cube_out_buf->mark_frame_full(name, row.cube_out_id);
    row.cube_out_id = (row.cube_out_id + 1) % row.cube_out_buf->num_frames;
}
