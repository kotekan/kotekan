#include "GnssCoherentCombiner.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata (the record's absolute hop)
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "gnssHostRows.hpp"     // for host_rows, host_workers, rest_register_verbs, run_rows

#include <algorithm>     // for replace
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
const std::vector<std::string> GnssCoherentCombiner::ROW_ONLY_KEYS = {"out_buf", "rec_row"};

std::vector<std::string> GnssCoherentCombiner::row_keys(Config& config,
                                                        const std::string& unique_name) {
    std::vector<std::string> keys = ROW_KEYS;
    if (config.exists(unique_name, "in_bufs"))
        std::replace(keys.begin(), keys.end(), std::string("in_bufs"), std::string("rec_row"));
    return keys;
}

GnssCoherentCombiner::Row::Row(Config& config, const gnss::HostRow& row) :
    chain(config, row.cfg_path, row.name,
          row.key.empty() ? "" : config.get<std::string>(row.cfg_path, "chain")) {}

GnssCoherentCombiner::GnssCoherentCombiner(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssCoherentCombiner::main_thread, this)),
    _host_rows(gnss::host_rows(config, unique_name, "_n2combine", row_keys(config, unique_name),
                               ROW_ONLY_KEYS)),
    _workers(gnss::host_workers(config, unique_name, _host_rows)) {
    // A host block's in_bufs are every row's: the GPU half's record buffer.
    const bool shared = !_host_rows[0].key.empty() && config.exists(unique_name, "in_bufs");
    std::vector<gnss::RecRow> rec_rows;
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
        if (shared) {
            if (config.exists(hr.cfg_path, "in_bufs"))
                FATAL_ERROR("{:s}: row {:s} sets in_bufs, which the host block sets for every row",
                            unique_name, hr.key);
            const gnss::RecRow rec =
                gnss::rec_row(config.get<nlohmann::json>(hr.cfg_path, "rec_row"));
            gnss::check_rec_row(hr.name, rec, row.chain.n_prn(), row.chain.n_elements(),
                                (int)rec.n_chan);
            row.rec_offset = rec.offset;
            rec_rows.push_back(rec);
        } else if (config.exists(hr.cfg_path, "rec_row")) {
            FATAL_ERROR("{:s}: rec_row is set, but the host block sets no in_bufs for its rows",
                        hr.name);
        }
        if (config.exists(hr.cfg_path, "out_buf"))
            row.out_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "out_buf"));
        if (row.out_buf)
            row.out_buf->register_producer(hr.name);
        row.chain.check_bufs(row.in_buf, row.out_buf);
        if (!row.out_buf)
            row.scratch.assign((size_t)row.chain.n_prn() * row.chain.rec_stride(), 0.0f);
    }
    if (shared) {
        _rec_buf = _rows[0].in_buf;
        _rec_dir = std::make_unique<gnss::RecDir>(unique_name, rec_rows, _rec_buf);
        for (Row& row : _rows)
            row.dir_row = _rec_dir->find((int64_t)row.rec_offset);
        for (const gnss::HostWorker& w : _workers)
            _rec_buf->register_consumer(consumer_name(w));
    } else {
        for (const gnss::HostWorker& w : _workers)
            register_inputs(w);
    }
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
    for (Row& row : _rows)
        gnss::rest_register_verbs(row.chain.rest_verbs(), _rest_paths);
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
    if (_rec_dir) {
        run_frames(wk);
        return;
    }
    while (!stop_thread)
        for (size_t r : wk.rows)
            if (!take_record(r, consumer_name(wk)))
                return;
}

void GnssCoherentCombiner::run_frames(const gnss::HostWorker& w) {
    const std::string& consumer = consumer_name(w);
    std::vector<bool> emit(w.rows.size());
    for (int id = 0; !stop_thread; id = (id + 1) % _rec_buf->num_frames) {
        const uint8_t* f = _rec_buf->wait_for_full_frame(consumer, id);
        if (f == nullptr)
            return;
        _rec_dir->check(consumer, id, f);
        for (size_t k = 0; k < w.rows.size(); ++k) {
            const Row& row = _rows[w.rows[k]];
            emit[k] = ingest_record(w.rows[k], (const float*)(f + row.rec_offset),
                                    _rec_dir->wstart(f, row.dir_row));
        }
        _rec_buf->mark_frame_empty(consumer, id);
        for (size_t k = 0; k < w.rows.size(); ++k)
            if (!finish_record(w.rows[k], emit[k]))
                return;
    }
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
