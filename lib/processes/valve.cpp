#include "valve.hpp"

#include <string>
#include <pthread.h>
#include <cstring>
#include <signal.h>

#include "fmt.hpp"

#include "visUtil.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "bufferContainer.hpp"
#include "prometheusMetrics.hpp"
#include "visBuffer.hpp"


REGISTER_KOTEKAN_PROCESS(Valve);

Valve::Valve(Config& config,
             const std::string& unique_name,
             bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&Valve::main_thread, this)) {

    _dropped_total = 0;

    _buf_in = get_buffer("in_buf");
    register_consumer(_buf_in, unique_name.c_str());
    _buf_out = get_buffer("out_buf");
    register_producer(_buf_out, unique_name.c_str());
}

void Valve::main_thread() {
    frameID frame_id_in(_buf_in);
    frameID frame_id_out(_buf_out);

    while (!stop_thread) {
        // Fetch a new frame and get its sequence id
        uint8_t* frame_in = wait_for_full_frame(_buf_in, unique_name.c_str(),
                                            frame_id_in);
        if(frame_in == nullptr) break;

        // check if there is space for it in the output buffer
        if (is_frame_empty(_buf_out, frame_id_out)) {
            visFrameView::copy_frame(_buf_in, frame_id_in,
                                     _buf_out, frame_id_out);
            mark_frame_full(_buf_out, unique_name.c_str(), frame_id_out++);
        } else {
            WARN("Output buffer full. Dropping incoming frame %d.",
                 frame_id_in);
            prometheusMetrics::instance().add_process_metric(
                        "dropped_frames_total", unique_name, ++_dropped_total);
        }
        mark_frame_empty(_buf_in, unique_name.c_str(), frame_id_in++);
    }
}

