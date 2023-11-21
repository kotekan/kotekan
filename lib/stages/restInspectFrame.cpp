#include "restInspectFrame.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for CHECK_MEM, WARN
#include "restServer.hpp"      // for restServer, connectionInstance

#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdlib.h>   // for free, malloc
#include <string.h>   // for memcpy
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(restInspectFrame);

restInspectFrame::restInspectFrame(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&restInspectFrame::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf_config_name = config.get<std::string>(unique_name, "in_buf");
    in_buf->register_consumer(unique_name);

    len = config.get_default<int32_t>(unique_name, "len", 0);

    registered = false;
    endpoint = "/inspect_frame/" + in_buf_config_name;

    if (len == 0) {
        len = in_buf->frame_size;
    } else if ((size_t)len > in_buf->frame_size) {
        WARN("Requested len ({:d}) is greater than the frame_size ({:d}).", len,
             in_buf->frame_size);
        len = in_buf->frame_size;
    }

    frame_copy = (uint8_t*)malloc(len);
    CHECK_MEM(frame_copy);
}

restInspectFrame::~restInspectFrame() {
    if (registered) {
        restServer::instance().remove_get_callback(endpoint);
    }
    free(frame_copy);
}

void restInspectFrame::rest_callback(connectionInstance& conn) {
    frame_copy_lock.lock();
    conn.send_binary_reply(frame_copy, len);
    frame_copy_lock.unlock();
}

void restInspectFrame::main_thread() {

    uint8_t* frame = nullptr;
    uint32_t frame_id = 0;

    while (!stop_thread) {
        frame = in_buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        if (frame_copy_lock.try_lock()) {
            // TODO Enforce alignemnt needed to use nt_memcpy() here.
            memcpy((void*)frame_copy, (void*)frame, len);
            frame_copy_lock.unlock();
        }

        // Only register the callback once we have something to return
        if (!registered) {
            using namespace std::placeholders;
            restServer::instance().register_get_callback(
                endpoint, std::bind(&restInspectFrame::rest_callback, this, _1));
            registered = true;
        }

        in_buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}
