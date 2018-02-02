#include "eigenVis.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"

eigenVis::eigenVis(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&eigenVis::main_thread, this)) {

    apply_config(0);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
}

eigenVis::~eigenVis() {
}

void eigenVis::apply_config(uint64_t fpga_seq) {
}

void eigenVis::main_thread() {

    unsigned int in_frame_id = 0, out_frame_id = 0;


    while (!stop_thread) {
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               in_frame_id) == nullptr) {
            break;
        }
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                               out_frame_id) == nullptr) {
            break;
        }



    }
}
