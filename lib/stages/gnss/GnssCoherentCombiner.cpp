#include "GnssCoherentCombiner.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata (the record's absolute hop)
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "gnssHostRows.hpp"     // for host_rows, host_workers, rest_register_unique, run_rows

#include <functional>    // for bind
#include <set>           // for set
#include <string>        // for string
#include <sys/syscall.h> // for SYS_gettid
#include <unistd.h>      // for syscall

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssCoherentCombiner);

const std::vector<std::string> GnssCoherentCombiner::ROW_KEYS = {"in_bufs", "n_prn", "chain"};
const std::vector<std::string> GnssCoherentCombiner::ROW_ONLY_KEYS = {"out_buf"};

GnssCoherentCombiner::Row::Row(Config& config, const gnss::HostRow& row) :
    chain(config, row.cfg_path, row.name,
          row.key.empty() ? "" : config.get<std::string>(row.cfg_path, "chain")) {}

GnssCoherentCombiner::GnssCoherentCombiner(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssCoherentCombiner::main_thread, this)),
    _host_rows(gnss::host_rows(config, unique_name, "_n2combine", ROW_KEYS, ROW_ONLY_KEYS)),
    _workers(gnss::host_workers(config, unique_name, _host_rows)) {
    for (const gnss::HostRow& hr : _host_rows) {
        Row& row = _rows.emplace_back(config, hr);
        // One record stream: the node's assembler sums every covering channel, and the broker
        // combines across nodes.
        const auto in_bufs = config.get<std::vector<std::string>>(hr.cfg_path, "in_bufs");
        if (in_bufs.size() != 1) {
            FATAL_ERROR("GnssCoherentCombiner: in_bufs has {:d} buffers; exactly one is supported",
                        in_bufs.size());
            return;
        }
        row.in_buf = buffer_container.get_buffer(in_bufs[0]);
        if (config.exists(hr.cfg_path, "out_buf"))
            row.out_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "out_buf"));
        if (row.out_buf)
            row.out_buf->register_producer(hr.name);
        row.chain.check_bufs(row.in_buf, row.out_buf);
        if (!row.out_buf)
            row.scratch.assign((size_t)row.chain.n_prn() * row.chain.rec_stride(), 0.0f);
    }
    for (const gnss::HostWorker& w : _workers)
        register_inputs(w);
}

void GnssCoherentCombiner::register_inputs(const gnss::HostWorker& w) {
    std::set<Buffer*> taken;
    for (size_t i : w.rows) {
        Buffer* b = _rows[i].in_buf;
        if (!taken.insert(b).second)
            FATAL_ERROR("{:s}: {:s} reads {:s} for two rows", _host_rows[i].name, w.name,
                        b->buffer_name);
        b->register_consumer(consumer_name(w));
    }
}

const std::string& GnssCoherentCombiner::consumer_name(const gnss::HostWorker& w) const {
    return w.rows.size() == 1 ? _host_rows[w.rows[0]].name : w.name;
}

GnssCoherentCombiner::~GnssCoherentCombiner() {
    gnss::release_host(_host_rows, _rest_paths);
}

void GnssCoherentCombiner::main_thread() {
    for (size_t r = 0; r < _rows.size(); ++r)
        for (const auto& v : _rows[r].chain.rest_verbs())
            gnss::rest_register_unique(_host_rows[r].name + "/" + v.verb, v.get, _rest_paths);
    gnss::rest_register_chains(unique_name, _host_rows, _rest_paths);
    if (_workers[0].cpus.empty()) {
        gnss::run_rows(_workers.size(), get_cpu_affinity(),
                       std::bind(&GnssCoherentCombiner::run_worker, this, std::placeholders::_1));
        return;
    }
    std::vector<std::string> names;
    std::vector<std::vector<int>> cpus;
    for (const gnss::HostWorker& w : _workers) {
        names.push_back(consumer_name(w));
        cpus.push_back(w.cpus);
    }
    gnss::run_workers(names, cpus,
                      std::bind(&GnssCoherentCombiner::run_pinned, this, std::placeholders::_1));
}

void GnssCoherentCombiner::run_pinned(size_t w) {
    const pid_t tid = syscall(SYS_gettid);
    register_tid(tid);
    run_worker(w);
    unregister_tid(tid);
}

void GnssCoherentCombiner::run_worker(size_t w) {
    const gnss::HostWorker& wk = _workers[w];
    while (!stop_thread)
        for (size_t r : wk.rows)
            if (!take_record(r, consumer_name(wk)))
                return;
}

bool GnssCoherentCombiner::take_record(size_t r, const std::string& consumer) {
    Row& row = _rows[r];
    Buffer* in_buf = row.in_buf;

    float* in = (float*)in_buf->wait_for_full_frame(consumer, row.in_id);
    if (in == nullptr)
        return false;

    const GnssChanMetadata* mi = get_gnss_chan_metadata(in_buf, row.in_id);
    const bool emit = ingest_record(r, in, mi ? mi->sample_seq : -1);

    in_buf->mark_frame_empty(consumer, row.in_id);
    row.in_id = (row.in_id + 1) % in_buf->num_frames;
    return finish_record(r, emit);
}

bool GnssCoherentCombiner::ingest_record(size_t r, const float* in, int64_t sample_seq) {
    gnss::CombinerChain& chain = _rows[r].chain;
    // Absolute hop index of this record's window, from the window start GnssGpuRecordAssemble
    // sets. Every node's sample_seq derives from the same F-engine counter, so equal hop is the
    // same window exactly, with no tolerance as UTC would need; the fleet DLL groups on it.
    // cur_hop is this record's hop, for the per-record export; win_hop labels the emitted window
    // (its newest record, as ref_utc is).
    const long long cur_hop = sample_seq >= 0 ? sample_seq / chain.fft_len() : -1;
    return chain.ingest(in, cur_hop);
}

bool GnssCoherentCombiner::finish_record(size_t r, bool emit) {
    Row& row = _rows[r];
    const std::string& name = _host_rows[r].name;
    Buffer* out_buf = row.out_buf;
    if (emit) {
        float* out = row.scratch.data();
        if (out_buf) {
            out = (float*)out_buf->wait_for_empty_frame(name, row.out_id);
            if (out == nullptr)
                return false;
        }
        row.chain.emit(out);
        if (out_buf) {
            out_buf->mark_frame_full(name, row.out_id);
            row.out_id = (row.out_id + 1) % out_buf->num_frames;
        }
    }
    row.records_done.store(row.records_done.load() + 1);
    return true;
}
