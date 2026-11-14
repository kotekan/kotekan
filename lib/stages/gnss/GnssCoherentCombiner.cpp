#include "GnssCoherentCombiner.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata (the record's absolute hop)
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "gnssHostRows.hpp"     // for host_rows, rest_register_unique, run_rows
#include "visUtil.hpp"          // for frameID

#include <functional> // for bind
#include <optional>   // for optional
#include <string>     // for string

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssCoherentCombiner);

const std::vector<std::string> GnssCoherentCombiner::ROW_KEYS = {"in_bufs", "n_prn"};
const std::vector<std::string> GnssCoherentCombiner::ROW_ONLY_KEYS = {"out_buf"};

GnssCoherentCombiner::Row::Row(Config& config, const gnss::HostRow& row) :
    chain(config, row.cfg_path, row.name) {}

GnssCoherentCombiner::GnssCoherentCombiner(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssCoherentCombiner::main_thread, this)),
    _host_rows(gnss::host_rows(config, unique_name, "_n2combine", ROW_KEYS, ROW_ONLY_KEYS)) {
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
        row.in_buf->register_consumer(hr.name);
        if (config.exists(hr.cfg_path, "out_buf"))
            row.out_buf =
                buffer_container.get_buffer(config.get<std::string>(hr.cfg_path, "out_buf"));
        if (row.out_buf)
            row.out_buf->register_producer(hr.name);
        row.chain.check_bufs(row.in_buf, row.out_buf);
        if (!row.out_buf)
            row.scratch.assign((size_t)row.chain.n_prn() * row.chain.rec_stride(), 0.0f);
    }
}

void GnssCoherentCombiner::main_thread() {
    for (size_t r = 0; r < _rows.size(); ++r)
        for (const auto& v : _rows[r].chain.rest_verbs())
            gnss::rest_register_unique(_host_rows[r].name + "/" + v.verb, v.get);
    gnss::rest_register_chains(unique_name, _host_rows);
    gnss::run_rows(_rows.size(), get_cpu_affinity(),
                   std::bind(&GnssCoherentCombiner::run_row, this, std::placeholders::_1));
}

void GnssCoherentCombiner::run_row(size_t r) {
    Row& row = _rows[r];
    const std::string& name = _host_rows[r].name;
    Buffer* in_buf = row.in_buf;
    Buffer* out_buf = row.out_buf;
    gnss::CombinerChain& chain = row.chain;

    frameID in_id(in_buf);
    std::optional<frameID> out_id;
    if (out_buf)
        out_id.emplace(out_buf);

    long long n_roll = 0; // total records seen

    while (!stop_thread) {
        row.records_done.store((uint64_t)n_roll);
        float* in = (float*)in_buf->wait_for_full_frame(name, in_id);
        if (in == nullptr)
            break;

        // Absolute hop index of this record's window, from the frame metadata
        // (GnssGpuRecordAssemble sets sample_seq to the window start). Every node's sample_seq
        // derives from the same F-engine counter, so equal hop is the same window exactly, with
        // no tolerance as UTC would need; the fleet DLL groups on it. cur_hop is this record's
        // hop, for the per-record export; win_hop labels the emitted window (its newest record,
        // as ref_utc is).
        long long cur_hop = -1;
        {
            const GnssChanMetadata* mi = get_gnss_chan_metadata(in_buf, in_id);
            cur_hop = (mi && mi->sample_seq >= 0) ? mi->sample_seq / chain.fft_len() : -1;
        }
        const bool emit = chain.ingest(in, cur_hop);

        in_buf->mark_frame_empty(name, in_id++);

        ++n_roll;
        if (!emit)
            continue;

        float* out = row.scratch.data();
        if (out_buf) {
            out = (float*)out_buf->wait_for_empty_frame(name, *out_id);
            if (out == nullptr)
                break;
        }
        chain.emit(out);
        if (out_buf)
            out_buf->mark_frame_full(name, (*out_id)++);
    }
}
