#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "pyPlotResult.hpp"
#include "buffer.h"
#include "errors.h"
#include "chimeMetadata.h"
#include "accumulate.hpp"
#include "fpga_header_functions.h"

using json = nlohmann::json;

REGISTER_KOTEKAN_PROCESS(pyPlotResult);

pyPlotResult::pyPlotResult(Config& config, const string& unique_name,
                           bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&pyPlotResult::main_thread, this))

{
    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    gpu_id = config.get_int(unique_name, "gpu_id");
}

pyPlotResult::~pyPlotResult() {
}

void pyPlotResult::request_plot_callback(connectionInstance& conn, json& json_request) {
//    std::lock_guard<std::mutex> lock(_packet_frame_lock);
    dump_plot=true;
    conn.send_empty_reply(STATUS_OK);
}

void pyPlotResult::apply_config(uint64_t fpga_seq) {
}

void pyPlotResult::main_thread() {
    using namespace std::placeholders;
    restServer * rest_server = get_rest_server();
    string endpoint = "/plot_corr_matrix/" + std::to_string(gpu_id);
    rest_server->register_json_callback(endpoint,
            std::bind(&pyPlotResult::request_plot_callback, this, _1, _2));

    int frame_id = 0;
    uint8_t * frame = NULL;
    unsigned char *in_local = (unsigned char*)malloc(buf->frame_size);

    while (!stop_thread) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        //INFO("Got buffer, id: %d", bufferID);

        if (dump_plot)
        {
            dump_plot=false;
            //make a local copy so the rest of kotekan can carry along happily.
            memcpy(in_local, frame, buf->frame_size);
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            frame_id = ( frame_id + 1 ) % buf->num_frames;

            FILE *python_script;
            python_script = popen("python -u pyPlotResult.py","w");

            { // N^2
                uint num_elements = config.get_int(unique_name, "num_elements");
                uint block_dim = 32;
                uint num_blocks = (num_elements/block_dim)*(num_elements/block_dim + 1)/2;
                uint block_size = block_dim*block_dim*2; //real, complex

                usleep(10000);

                stream_id_t stream_id = get_stream_id_t(buf, frame_id);

                json header = {
                    {"data_length",num_blocks*block_size},
                    {"type","CORR_MATRIX"},
                    {"num_elements",num_elements},
                    {"block_dim",{block_dim,block_dim,2}},
                    {"stream_id", {stream_id.crate_id, stream_id.slot_id, stream_id.link_id, stream_id.unused}}
                };
                std::string s = header.dump()+"\n";
                fwrite(s.c_str(),1,s.length(),python_script);
                for (int i=0; i<num_blocks; i++) {
                    fwrite(in_local +i*sizeof(int)*block_size,sizeof(int),block_size,python_script);
                    fflush(python_script);
                }
            }
        }
        else{
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            frame_id = ( frame_id + 1 ) % buf->num_frames;
        }
    }
    free(in_local);
}
