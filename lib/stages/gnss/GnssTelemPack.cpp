#include "GnssTelemPack.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"

#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssTelemPack);

GnssTelemPack::GnssTelemPack(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssTelemPack::main_thread, this)),
    _chain(config, unique_name, unique_name) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    _chain.check_bufs(in_buf, out_buf);
}

bool GnssTelemPack::flush(frameID& out_id) {
    if (!_chain.window_open())
        return true;

    uint8_t* frame = out_buf->wait_for_empty_frame(unique_name, out_id);
    if (frame == nullptr)
        return false;

    const int64_t wstart0 = _chain.pack(frame);

    // bufferSend ships only frames that carry a metadata object. Its sample_seq mirrors the
    // header's wstart0, so the frame can be addressed from its metadata without parsing the
    // payload; readers of the wire key on the header's `win`.
    out_buf->allocate_new_metadata_object(out_id);
    if (GnssChanMetadata* m = get_gnss_chan_metadata(out_buf, out_id))
        m->sample_seq = wstart0;
    out_buf->mark_frame_full(unique_name, out_id++);
    return true;
}

void GnssTelemPack::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);

    while (!stop_thread) {
        float* in = (float*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;

        int64_t wstart = -1;
        if (metadata_is_gnss_chan(in_buf)) {
            const GnssChanMetadata* m = get_gnss_chan_metadata(in_buf, in_id);
            if (m)
                wstart = m->sample_seq;
        }

        if (!_chain.on_grid(wstart)) {
            in_buf->mark_frame_empty(unique_name, in_id++);
            continue;
        }
        if (_chain.window_changes(wstart))
            if (!flush(out_id))
                break;
        _chain.add_record(in, wstart);

        in_buf->mark_frame_empty(unique_name, in_id++);

        // Emit a full window at once rather than when the next record crosses the boundary:
        // that record may be a whole correlator frame away, and a chain whose records stop
        // leaves no complete window unsent.
        if (_chain.window_full())
            if (!flush(out_id))
                break;
    }
    flush(out_id);
}
