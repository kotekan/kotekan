#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "gnssChanPower.hpp"    // for ChanPower
#include "gnssProjSubspace.hpp" // for ProjBoard
#include "restServer.hpp"
#include "visUtil.hpp" // for frameID

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssGpuRecordAssemble);

GnssGpuRecordAssemble::GnssGpuRecordAssemble(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssGpuRecordAssemble::main_thread, this)),
    _chain(config, unique_name, unique_name, gnss::ProjBoard::instance(),
           gnss::ChanPower::instance()) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    _chain.check_out_buf(out_buf);
    if (_chain.cube_on()) {
        const std::string cbuf = config.get<std::string>(unique_name, "cube_buf");
        _cube_out_buf = buffer_container.get_buffer(cbuf);
        _cube_out_buf->register_producer(unique_name);
        _chain.check_cube_buf(cbuf, _cube_out_buf);
    }
    kotekan::restServer& rest = kotekan::restServer::instance();
    for (const auto& v : _chain.rest_verbs()) {
        if (v.get)
            rest.register_get_callback(unique_name + "/" + v.verb, v.get);
        else
            rest.register_post_callback(unique_name + "/" + v.verb, v.post);
    }
}

void GnssGpuRecordAssemble::main_thread() {
    frameID frame_in(in_buf), frame_out(out_buf);

    while (!stop_thread) {
        const uint8_t* in = in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in == nullptr)
            return;
        const int n_rec = _chain.begin_frame(in);
        for (int r = 0; r < n_rec && !stop_thread; ++r) {
            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            if (out == nullptr)
                return;
            const int64_t wstart = _chain.assemble_record(r, out);
            emit_cube_window();
            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = wstart;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out++;
        }
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in++;
    }
}

void GnssGpuRecordAssemble::emit_cube_window() {
    if (!_chain.cube_window_ready())
        return;
    // Never blocks (see the header): this stage is in the real-time path, so a full buffer
    // costs the archive a window, never the tracker a record.
    // is_frame_empty() is a safe pre-check because this stage is the buffer's only producer.
    if (!_cube_out_buf->is_frame_empty(_cube_out_id)) {
        _chain.take_cube_window(nullptr);
        return;
    }
    uint8_t* f = _cube_out_buf->wait_for_empty_frame(unique_name, _cube_out_id);
    if (f == nullptr)
        return; // shutting down
    const int64_t w0 = _chain.take_cube_window(f);
    // A frame that feeds bufferSend needs a metadata object: bufferSend drops a frame without
    // one. Stamped like the record leg: the window's first sample, so a consumer after
    // bufferRecv can address the frame without parsing it.
    if (_cube_out_buf->metadata_pool) {
        _cube_out_buf->allocate_new_metadata_object(_cube_out_id);
        if (auto* m = get_gnss_chan_metadata(_cube_out_buf, _cube_out_id))
            m->sample_seq = w0;
    }
    _cube_out_buf->mark_frame_full(unique_name, _cube_out_id);
    _cube_out_id = (_cube_out_id + 1) % _cube_out_buf->num_frames;
}
