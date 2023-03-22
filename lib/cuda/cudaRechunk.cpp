#include "cudaRechunk.hpp"

#include "cudaUtils.hpp"

#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaRechunk);

cudaRechunk::cudaRechunk(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaRechunk", "") {
    _len_inner_input = config.get<int>(unique_name, "len_inner_input");
    _len_inner_output = config.get<int>(unique_name, "len_inner_output");
    _len_outer = config.get<int>(unique_name, "len_outer");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    set_command_type(gpuCommandType::KERNEL);
    num_accumulated = 0;

    assert(_len_inner_output % _len_inner_input == 0);
    // leftover_memory =
    //device.get_gpu_memory("leftover", _len_inner_input * _len_outer);
    // num_leftover = 0;
}

cudaRechunk::~cudaRechunk() {}

cudaEvent_t cudaRechunk::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events, bool* quit) {
    (void)pre_events;
    pre_execute(gpu_frame_id);

    size_t input_frame_len = _len_inner_input * _len_outer;
    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_input, gpu_frame_id, input_frame_len);

    size_t output_frame_len = _len_inner_output * _len_outer;
    void* accum_memory = device.get_gpu_memory("accum", output_frame_len);

    size_t n_copy = _len_inner_input;
    if (num_accumulated + _len_inner_input > _len_inner_output) {
        n_copy = _len_inner_output - num_accumulated;
        // Copy the remainder into the leftover_memory.
    }

    record_start_event(gpu_frame_id);

    // if (num_leftover) copy leftover_memory to output, incr. num_accumulated

    CHECK_CUDA_ERROR(cudaMemcpy2DAsync((void*)((char*)accum_memory + num_accumulated),
                                       _len_inner_output,
                                       input_memory,
                                       _len_inner_input,
                                       n_copy,
                                       _len_outer,
                                       cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    num_accumulated += n_copy;
    if (num_accumulated >= _len_inner_output) {
        // emit an output frame!
        void* output_memory =
            device.get_gpu_memory_array(_gpu_mem_output, gpu_frame_id, output_frame_len);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, accum_memory, output_frame_len,
                                         cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));

        num_accumulated -= _len_inner_output;
        // (copy any overflow into the "leftover" array)
    } else {
        // partial output frame -- don't run further GPU kernels.
        *quit = true;
    }

    return record_end_event(gpu_frame_id);
}
