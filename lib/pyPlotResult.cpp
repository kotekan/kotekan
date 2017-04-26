#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "pyPlotResult.hpp"
#include "stream_raw_vdif.h"
#include "buffers.h"
#include "errors.h"

using json = nlohmann::json;

pyPlotResult::pyPlotResult(Config& config, const string& unique_name, Buffer& buf_, int gpu_id_,
                 const std::string &base_dir_,
                 const std::string &file_name_,
                 const std::string &file_ext_) :
        KotekanProcess(config, unique_name, std::bind(&pyPlotResult::main_thread, this)),
        buf(buf_),
        base_dir(base_dir_),
        file_name(file_name_),
        file_ext(file_ext_),
        gpu_id(gpu_id_)
{
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

    int buffer_id = 0;
    unsigned char *in_local = (unsigned char*)malloc(buf.buffer_size);

    for (;;) {

        // This call is blocking.
        buffer_id = get_full_buffer_from_list(&buf, &buffer_id, 1);

        //INFO("Got buffer, id: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (buffer_id == -1) {
            return;
        }


        if (dump_plot)
        {
            dump_plot=false;
            //make a local copy so the rest of kotekan can carry along happily.
            memcpy(in_local,buf.data[buffer_id],buf.buffer_size);
            mark_buffer_empty(&buf, buffer_id);
            buffer_id = ( buffer_id + 1 ) % buf.num_buffers;

            FILE *python_script;
            python_script = popen("python -u pyPlotResult.py","w");

            { // N^2
                uint num_elements = config.get_int("/processing/num_elements");
                uint block_dim = 32;
                uint num_blocks = (num_elements/block_dim)*(num_elements/block_dim + 1)/2;
                uint block_size = block_dim*block_dim*2; //real, complex

                usleep(10000);

                json header = {
                    {"data_length",num_blocks*block_size},
                    {"type","CORR_MATRIX"},
                    {"num_elements",num_elements},
                    {"block_dim",{block_dim,block_dim,2}}
                };
                std::string s = header.dump()+"\n";
                fwrite(s.c_str(),1,s.length(),python_script);
                for (int i=0; i<num_blocks; i++) {
                    fwrite(in_local+i*sizeof(int)*block_size,sizeof(int),block_size,python_script);
                    fflush(python_script);
                }
            }
        }
        else{
            mark_buffer_empty(&buf, buffer_id);
            buffer_id = ( buffer_id + 1 ) % buf.num_buffers;
        }
    }
    free(in_local);
}
