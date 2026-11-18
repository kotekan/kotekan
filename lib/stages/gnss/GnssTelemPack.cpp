#include "GnssTelemPack.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"

#include <functional>
#include <set>
#include <sys/syscall.h> // for SYS_gettid
#include <unistd.h>      // for syscall
#include <utility>       // for pair

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssTelemPack);

const std::vector<std::string> GnssTelemPack::ROW_KEYS = {"in_buf",  "chain",  "n_prn",
                                                          "max_prn", "n_chan", "channel_ids"};

GnssTelemPack::Out::Out(Buffer* buf, const std::string& producer) :
    buf(buf), producer(producer), id(buf) {}

GnssTelemPack::Row::Row(Config& config, const gnss::HostRow& row, Buffer* in, Out* out) :
    in_buf(in), in_id(in), out(out), chain(config, row.cfg_path, row.name) {}

std::vector<std::string> GnssTelemPack::row_keys(Config& config, const std::string& unique_name) {
    std::vector<std::string> keys = ROW_KEYS;
    if (!config.exists(unique_name, "out_buf"))
        keys.push_back("out_buf");
    return keys;
}

GnssTelemPack::GnssTelemPack(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssTelemPack::main_thread, this)),
    _host_rows(
        gnss::host_rows(config, unique_name, "_telem_pack", row_keys(config, unique_name), {})),
    _workers(gnss::host_workers(config, unique_name, _host_rows)) {
    // A host block's out_buf is every row's: one stream for the host.
    const bool shared = !_host_rows[0].key.empty() && config.exists(unique_name, "out_buf");
    if (shared)
        _outs.emplace_back(get_buffer("out_buf"), unique_name);
    std::set<std::pair<std::string, std::string>> senders;
    for (const gnss::HostRow& hr : _host_rows) {
        if (shared && config.exists(hr.cfg_path, "out_buf"))
            FATAL_ERROR("{:s}: row {:s} sets out_buf, which the host block sets for every row",
                        unique_name, hr.key);
        if (!shared)
            _outs.emplace_back(
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "out_buf")),
                hr.name);
        Row& row = _rows.emplace_back(
            config, hr, buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "in_buf")),
            &_outs.back());
        row.chain.check_bufs(row.in_buf, row.out->buf, !shared);
        if (!senders.insert({row.chain.chain(), row.chain.inst()}).second)
            FATAL_ERROR("{:s}: two rows send chain {:s} inst {:s}; the gather would merge their "
                        "streams",
                        unique_name, row.chain.chain(), row.chain.inst());
        if (shared)
            row.dropped =
                &kotekan::prometheus::Metrics::instance()
                     .add_counter("kotekan_gnss_telem_dropped_frames_total", hr.name, {"chain"})
                     .labels({row.chain.chain()});
    }
    for (Out& o : _outs)
        o.buf->register_producer(o.producer);
    for (const gnss::HostWorker& w : _workers)
        register_inputs(w);
    gnss::rest_register_chains(unique_name, _host_rows, _rest_paths);
}

GnssTelemPack::~GnssTelemPack() {
    gnss::release_host(_host_rows, _rest_paths);
}

void GnssTelemPack::register_inputs(const gnss::HostWorker& w) {
    std::set<Buffer*> taken;
    for (size_t i : w.rows) {
        Buffer* b = _rows[i].in_buf;
        if (!taken.insert(b).second)
            FATAL_ERROR("{:s}: {:s} reads {:s} for two rows", _host_rows[i].name, w.name,
                        b->buffer_name);
        b->register_consumer(consumer_name(w));
    }
}

const std::string& GnssTelemPack::consumer_name(const gnss::HostWorker& w) const {
    return w.rows.size() == 1 ? _host_rows[w.rows[0]].name : w.name;
}

bool GnssTelemPack::flush(size_t i) {
    Row& row = _rows[i];
    if (!row.chain.window_open())
        return true;

    Out& out = *row.out;
    std::lock_guard<std::mutex> lock(out.lock);
    if (row.dropped && !out.buf->is_frame_empty(out.id)) {
        row.chain.drop();
        row.dropped->inc();
        return true;
    }
    uint8_t* frame = out.buf->wait_for_empty_frame(out.producer, out.id);
    if (frame == nullptr)
        return false;

    const int64_t wstart0 = row.chain.pack(frame);

    // bufferSend ships only frames that carry a metadata object. Its sample_seq mirrors the
    // header's wstart0, so the frame can be addressed from its metadata without parsing the
    // payload; readers of the wire key on the header's `win`.
    out.buf->allocate_new_metadata_object(out.id);
    if (GnssChanMetadata* m = get_gnss_chan_metadata(out.buf, out.id))
        m->sample_seq = wstart0;
    out.buf->mark_frame_full(out.producer, out.id++);
    return true;
}

void GnssTelemPack::main_thread() {
    if (_workers[0].cpus.empty()) {
        gnss::run_rows(_workers.size(), get_cpu_affinity(),
                       std::bind(&GnssTelemPack::run_worker, this, std::placeholders::_1));
        return;
    }
    std::vector<std::string> names;
    std::vector<std::vector<int>> cpus;
    for (const gnss::HostWorker& w : _workers) {
        names.push_back(consumer_name(w));
        cpus.push_back(w.cpus);
    }
    gnss::run_workers(names, cpus,
                      std::bind(&GnssTelemPack::run_pinned, this, std::placeholders::_1));
}

void GnssTelemPack::run_pinned(size_t w) {
    const pid_t tid = syscall(SYS_gettid);
    register_tid(tid);
    run_worker(w);
    unregister_tid(tid);
}

void GnssTelemPack::run_worker(size_t w) {
    const gnss::HostWorker& wk = _workers[w];
    bool running = true;
    while (running && !stop_thread)
        for (size_t r : wk.rows)
            if (!take_record(r, consumer_name(wk))) {
                running = false;
                break;
            }
    for (size_t r : wk.rows)
        flush(r);
}

bool GnssTelemPack::take_record(size_t i, const std::string& consumer) {
    Row& row = _rows[i];
    Buffer* in_buf = row.in_buf;

    float* in = (float*)in_buf->wait_for_full_frame(consumer, row.in_id);
    if (in == nullptr)
        return false;

    int64_t wstart = -1;
    if (metadata_is_gnss_chan(in_buf)) {
        const GnssChanMetadata* m = get_gnss_chan_metadata(in_buf, row.in_id);
        if (m)
            wstart = m->sample_seq;
    }

    if (!add_record(i, in, wstart))
        return false;
    in_buf->mark_frame_empty(consumer, row.in_id++);
    return flush_full(i);
}

bool GnssTelemPack::add_record(size_t i, const float* in, int64_t wstart) {
    gnss::TelemChain& chain = _rows[i].chain;
    if (!chain.on_grid(wstart))
        return true;
    if (chain.window_changes(wstart))
        if (!flush(i))
            return false;
    chain.add_record(in, wstart);
    return true;
}

bool GnssTelemPack::flush_full(size_t i) {
    // Emit a full window at once rather than when the next record crosses the boundary:
    // that record may be a whole correlator frame away, and a chain whose records stop
    // leaves no complete window unsent.
    if (_rows[i].chain.window_full())
        if (!flush(i))
            return false;
    return true;
}
