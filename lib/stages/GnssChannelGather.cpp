#include "GnssChannelGather.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp" // for FATAL_ERROR
#include "visUtil.hpp"        // for frameID

#include <complex> // for complex
#include <cstring> // for memcpy

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssChannelGather);

GnssChannelGather::GnssChannelGather(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChannelGather::main_thread, this)) {
    in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : in_bufs)
        b->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    const int n_in = (int)in_bufs.size();
    if (n_in == 0) {
        FATAL_ERROR("GnssChannelGather: no in_bufs");
        return;
    }
    // out frame holds n_in channels per hop; each input frame is one channel's hops.
    if (out_buf->frame_size != in_bufs[0]->frame_size * n_in)
        FATAL_ERROR("GnssChannelGather: out frame_size {:d} != {:d} (in {:d} x {:d} channels)",
                    out_buf->frame_size, in_bufs[0]->frame_size * n_in, in_bufs[0]->frame_size,
                    n_in);
}

void GnssChannelGather::main_thread() {
    const int n_in = (int)in_bufs.size();
    std::vector<frameID> in_ids;
    for (Buffer* b : in_bufs)
        in_ids.emplace_back(b);
    frameID out_id(out_buf);

    while (!stop_thread) {
        cf* out = (cf*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        // Gather the i-th frame of every channel (lock-step: same absolute hops).
        std::vector<cf*> ins;
        bool stopping = false;
        for (int i = 0; i < n_in; ++i) {
            cf* in = (cf*)in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
            if (in == nullptr) {
                stopping = true;
                break;
            }
            ins.push_back(in);
        }
        if (stopping)
            break;

        const int hops = in_bufs[0]->frame_size / (int)sizeof(cf);
        for (int i = 0; i < n_in; ++i)
            for (int h = 0; h < hops; ++h)
                out[(size_t)h * n_in + i] = ins[i][h]; // [hop][channel], channel-minor

        // Carry the absolute sample reference (fpga_seq) from input 0 so the search
        // keeps a node-independent "sample 0".
        if (in_bufs[0]->metadata_pool && out_buf->metadata_pool) {
            out_buf->allocate_new_metadata_object(out_id);
            in_bufs[0]->copy_metadata(in_ids[0], out_buf, out_id);
        }

        for (int i = 0; i < n_in; ++i)
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
        out_buf->mark_frame_full(unique_name, out_id++);
    }
}
