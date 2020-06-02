#include "pyPlotN2.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_empty, Buffer, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for restServer, connectionInstance, HTTP_RESPONSE, HTTP_RESPO...

#include "json.hpp" // for json_ref, json

#include <atomic>      // for atomic_bool
#include <cstdio>      // for fwrite, fflush, popen, FILE
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, _Placeholder, bind, _1, function
#include <regex>       // for match_results<>::_Base_type
#include <stdint.h>    // for uint32_t, uint8_t
#include <stdlib.h>    // for free, malloc
#include <string.h>    // for memcpy
#include <sys/types.h> // for uint
#include <thread>      // for thread
#include <unistd.h>    // for usleep
#include <vector>      // for vector


using json = nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(pyPlotN2);

pyPlotN2::pyPlotN2(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&pyPlotN2::main_thread, this))

{
    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    gpu_id = config.get<int>(unique_name, "gpu_id");
    in_local = (unsigned char*)malloc(buf->frame_size);
    endpoint = unique_name + "/plot_corr_matrix/" + std::to_string(gpu_id);
}

pyPlotN2::~pyPlotN2() {
    restServer::instance().remove_get_callback(endpoint);
    free(in_local);
}

void pyPlotN2::request_plot_callback(connectionInstance& conn) {
    //    std::lock_guard<std::mutex> lock(_packet_frame_lock);
    dump_plot = true;
    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

void pyPlotN2::main_thread() {
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    rest_server.register_get_callback(endpoint,
                                      std::bind(&pyPlotN2::request_plot_callback, this, _1));

    int frame_id = 0;
    uint8_t* frame = nullptr;

    while (!stop_thread) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        // INFO("Got buffer, id: {:d}", bufferID);

        if ((!busy) && (dump_plot)) {
            dump_plot = false;
            busy = true;
            // make a local copy so the rest of kotekan can carry along happily.
            memcpy(in_local, frame, buf->frame_size);
            stream_id = ice_get_stream_id_t(buf, frame_id);

            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;

            std::thread thr = std::thread(&pyPlotN2::make_plot, this);
            thr.detach();
        } else {
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
        }
    }
}


void pyPlotN2::make_plot(void) {
    FILE* python_script;
    python_script = popen("python -u /usr/sbin/pyPlotN2.py", "w");
    { // N^2
        uint num_elements = config.get<uint>(unique_name, "num_elements");
        uint block_dim = 32;
        uint num_blocks = (num_elements / block_dim) * (num_elements / block_dim + 1) / 2;
        uint block_size = block_dim * block_dim * 2; // real, complex

        usleep(10000);

        json header = {
            {"data_length", num_blocks * block_size},
            {"type", "CORR_MATRIX"},
            {"num_elements", num_elements},
            {"block_dim", {block_dim, block_dim, 2}},
            {"stream_id",
             {stream_id.crate_id, stream_id.slot_id, stream_id.link_id, stream_id.unused}}};
        std::string s = header.dump() + "\n";
        fwrite(s.c_str(), 1, s.length(), python_script);
        for (uint32_t i = 0; i < num_blocks; i++) {
            fwrite(in_local + i * sizeof(int) * block_size, sizeof(int), block_size, python_script);
            fflush(python_script);
        }
    }
    busy = false;
}
