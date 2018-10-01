#include "gpuSimulate.hpp"
#include "errors.h"

REGISTER_KOTEKAN_PROCESS(gpuSimulate);

gpuSimulate::gpuSimulate(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&gpuSimulate::main_thread, this)) {

    apply_config(0);

    input_buf = get_buffer("network_in_buf");
    register_consumer(input_buf, unique_name.c_str());
    output_buf = get_buffer("corr_out_buf");
    register_producer(output_buf, unique_name.c_str());

    int block_map_len = _num_blocks * 2 * sizeof(uint32_t);
    host_block_map = (uint32_t *)malloc(block_map_len);
    assert(host_block_map != nullptr);
    int block_id = 0;
    for (int y = 0; block_id < _num_blocks; y++) {
        for (int x = y; x < _num_elements/32; x++) {
            host_block_map[2*block_id+0] = x;
            host_block_map[2*block_id+1] = y;
            block_id++;
        }
    }
}

gpuSimulate::~gpuSimulate() {
    free(host_block_map);
}

void gpuSimulate::apply_config(uint64_t fpga_seq) {
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_blocks = config.get<int32_t>(unique_name, "num_blocks");
}

void gpuSimulate::main_thread() {

    int input_frame_id = 0;
    int output_frame_id = 0;

    while(!stop_thread) {
        int * input = (int *)wait_for_full_frame(input_buf, unique_name.c_str(), input_frame_id);
        if (input == NULL) break;
        int * output = (int *)wait_for_empty_frame(output_buf, unique_name.c_str(), output_frame_id);
        if (output == NULL) break;

        // TODO adjust to allow for more than one frequency.
        // TODO remove all the 32's in here with some kind of constant/define
        INFO("Simulating GPU processing for %s[%d] putting result in %s[%d]",
                input_buf->buffer_name, input_frame_id,
                output_buf->buffer_name, output_frame_id);
        for (int b = 0; b < _num_blocks; ++b){
            for (int y = 0; y < 32; ++y){
                for (int x = 0; x < 32; ++x){
                    int real = 0;
                    int imag = 0;
                    for (int t = 0; t < _samples_per_data_set; ++t){
                        int xi = (input[host_block_map[2*b+0]*32 + x + t*_num_elements] & 0x0f)       - 8;
                        int xr =((input[host_block_map[2*b+0]*32 + x + t*_num_elements] & 0xf0) >> 4) - 8;
                        int yi = (input[host_block_map[2*b+1]*32 + y + t*_num_elements] & 0x0f)       - 8;
                        int yr =((input[host_block_map[2*b+1]*32 + y + t*_num_elements] & 0xf0) >> 4) - 8;
                        real += xr*yr + xi*yi;
                        imag += xi*yr - yi*xr;
                    }
                    output[b*32*32*2 + x*2 + y*32*2 + 0] = imag;
                    output[b*32*32*2 + x*2 + y*32*2 + 1] = real;
                    //INFO("real: %d, imag: %d", real, imag);
                }
            }
            INFO("Done block %d of %d...", b, _num_blocks);
        }

        INFO("Simulating GPU processing done for %s[%d] result is in %s[%d]",
                input_buf->buffer_name, input_frame_id,
                output_buf->buffer_name, output_frame_id);

        pass_metadata(input_buf, input_frame_id, output_buf, output_frame_id);
        mark_frame_empty(input_buf, unique_name.c_str(), input_frame_id);
        mark_frame_full(output_buf, unique_name.c_str(), output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
