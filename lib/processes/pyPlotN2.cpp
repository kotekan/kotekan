#include "pyPlotN2.hpp"

#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using json = nlohmann::json;

REGISTER_KOTEKAN_PROCESS(pyPlotN2);

pyPlotN2::pyPlotN2(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&pyPlotN2::main_thread, this))

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
    uint8_t* frame = NULL;

    while (!stop_thread) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL)
            break;

        // INFO("Got buffer, id: %d", bufferID);

        if ((!busy) && (dump_plot)) {
            dump_plot = false;
            busy = true;
            // make a local copy so the rest of kotekan can carry along happily.
            memcpy(in_local, frame, buf->frame_size);
            stream_id = get_stream_id_t(buf, frame_id);

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
