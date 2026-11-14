#include "GnssCoherentCombiner.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata (the record's absolute hop)
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "restServer.hpp"       // for restServer
#include "visUtil.hpp"          // for frameID

#include <functional> // for bind
#include <optional>   // for optional
#include <string>     // for string

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssCoherentCombiner);

GnssCoherentCombiner::GnssCoherentCombiner(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssCoherentCombiner::main_thread, this)),
    _chain(config, unique_name, unique_name) {
    // One record stream: the node's assembler sums every covering channel, and the broker
    // combines across nodes.
    const std::vector<Buffer*> in_bufs = get_buffer_array("in_bufs");
    if (in_bufs.size() != 1) {
        FATAL_ERROR("GnssCoherentCombiner: in_bufs has {:d} buffers; exactly one is supported",
                    in_bufs.size());
        return;
    }
    in_buf = in_bufs[0];
    in_buf->register_consumer(unique_name);
    out_buf = config.exists(unique_name, "out_buf") ? get_buffer("out_buf") : nullptr;
    if (out_buf)
        out_buf->register_producer(unique_name);
    _chain.check_bufs(in_buf, out_buf);
    if (!out_buf)
        _scratch.assign((size_t)_chain.n_prn() * _chain.rec_stride(), 0.0f);
}

void GnssCoherentCombiner::main_thread() {
    for (const auto& v : _chain.rest_verbs())
        kotekan::restServer::instance().register_get_callback(unique_name + "/" + v.verb, v.get);

    frameID in_id(in_buf);
    std::optional<frameID> out_id;
    if (out_buf)
        out_id.emplace(out_buf);

    long long n_roll = 0; // total records seen

    while (!stop_thread) {
        _records_done.store((uint64_t)n_roll);
        float* in = (float*)in_buf->wait_for_full_frame(unique_name, in_id);
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
            cur_hop = (mi && mi->sample_seq >= 0) ? mi->sample_seq / _chain.fft_len() : -1;
        }
        const bool emit = _chain.ingest(in, cur_hop);

        in_buf->mark_frame_empty(unique_name, in_id++);

        ++n_roll;
        if (!emit)
            continue;

        float* out = _scratch.data();
        if (out_buf) {
            out = (float*)out_buf->wait_for_empty_frame(unique_name, *out_id);
            if (out == nullptr)
                break;
        }
        _chain.emit(out);
        if (out_buf)
            out_buf->mark_frame_full(unique_name, (*out_id)++);
    }
}
