#include "GnssRecordCollector.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "visUtil.hpp"      // for frameID

#include <cstring> // for memcpy

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssRecordCollector);

GnssRecordCollector::GnssRecordCollector(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssRecordCollector::main_thread, this)) {
    in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : in_bufs)
        b->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
}

void GnssRecordCollector::main_thread() {
    std::vector<frameID> in_ids;
    for (Buffer* b : in_bufs)
        in_ids.emplace_back(b);
    frameID out_id(out_buf);

    while (!stop_thread) {
        uint8_t* out = out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        size_t off = 0;
        bool stopping = false;
        for (size_t i = 0; i < in_bufs.size(); ++i) {
            uint8_t* in = in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
            if (in == nullptr) {
                stopping = true;
                break;
            }
            const size_t n = in_bufs[i]->frame_size;
            memcpy(out + off, in, n); // record-agnostic byte concat
            off += n;
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
        }
        if (stopping)
            break;

        out_buf->mark_frame_full(unique_name, out_id++);
    }
}
