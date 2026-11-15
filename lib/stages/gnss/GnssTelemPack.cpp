#include "GnssTelemPack.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"

#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssTelemPack);

const std::vector<std::string> GnssTelemPack::ROW_KEYS = {
    "in_buf", "out_buf", "chain", "n_prn", "max_prn", "n_chan", "channel_ids"};

GnssTelemPack::Row::Row(Config& config, const gnss::HostRow& row) :
    chain(config, row.cfg_path, row.name) {}

GnssTelemPack::GnssTelemPack(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssTelemPack::main_thread, this)),
    _host_rows(gnss::host_rows(config, unique_name, "_telem_pack", ROW_KEYS, {})) {
    for (const gnss::HostRow& hr : _host_rows) {
        Row& row = _rows.emplace_back(config, hr);
        row.in_buf = buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "in_buf"));
        row.in_buf->register_consumer(hr.name);
        row.out_buf = buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "out_buf"));
        row.out_buf->register_producer(hr.name);
        row.chain.check_bufs(row.in_buf, row.out_buf);
    }
    gnss::rest_register_chains(unique_name, _host_rows, _rest_paths);
}

GnssTelemPack::~GnssTelemPack() {
    gnss::release_host(_host_rows, _rest_paths);
}

bool GnssTelemPack::flush(size_t i, frameID& out_id) {
    Row& row = _rows[i];
    if (!row.chain.window_open())
        return true;

    uint8_t* frame = row.out_buf->wait_for_empty_frame(_host_rows[i].name, out_id);
    if (frame == nullptr)
        return false;

    const int64_t wstart0 = row.chain.pack(frame);

    // bufferSend ships only frames that carry a metadata object. Its sample_seq mirrors the
    // header's wstart0, so the frame can be addressed from its metadata without parsing the
    // payload; readers of the wire key on the header's `win`.
    row.out_buf->allocate_new_metadata_object(out_id);
    if (GnssChanMetadata* m = get_gnss_chan_metadata(row.out_buf, out_id))
        m->sample_seq = wstart0;
    row.out_buf->mark_frame_full(_host_rows[i].name, out_id++);
    return true;
}

void GnssTelemPack::main_thread() {
    gnss::run_rows(_rows.size(), get_cpu_affinity(),
                   std::bind(&GnssTelemPack::run_row, this, std::placeholders::_1));
}

void GnssTelemPack::run_row(size_t i) {
    Buffer* in_buf = _rows[i].in_buf;
    gnss::TelemChain& chain = _rows[i].chain;
    const std::string& name = _host_rows[i].name;
    frameID in_id(in_buf);
    frameID out_id(_rows[i].out_buf);

    while (!stop_thread) {
        float* in = (float*)in_buf->wait_for_full_frame(name, in_id);
        if (in == nullptr)
            break;

        int64_t wstart = -1;
        if (metadata_is_gnss_chan(in_buf)) {
            const GnssChanMetadata* m = get_gnss_chan_metadata(in_buf, in_id);
            if (m)
                wstart = m->sample_seq;
        }

        if (!chain.on_grid(wstart)) {
            in_buf->mark_frame_empty(name, in_id++);
            continue;
        }
        if (chain.window_changes(wstart))
            if (!flush(i, out_id))
                break;
        chain.add_record(in, wstart);

        in_buf->mark_frame_empty(name, in_id++);

        // Emit a full window at once rather than when the next record crosses the boundary:
        // that record may be a whole correlator frame away, and a chain whose records stop
        // leaves no complete window unsent.
        if (chain.window_full())
            if (!flush(i, out_id))
                break;
    }
    flush(i, out_id);
}
