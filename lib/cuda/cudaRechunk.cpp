#include "cudaRechunk.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaRechunk);

cudaRechunk::cudaRechunk(Config& config, const std::string& unique_name,
                         bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaRechunk", ""),
    output_frame_id(0) {
    _len_inner_input = config.get<int>(unique_name, "len_inner_input");
    _len_inner_output = config.get<int>(unique_name, "len_inner_output");
    _len_outer = config.get<int>(unique_name, "len_outer");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _set_flag = config.get_default<std::string>(unique_name, "set_flag", "");
    _output_frame_counter = config.get_default<std::string>(unique_name, "output_frame_counter", "");
    set_command_type(gpuCommandType::KERNEL);
    num_accumulated = 0;

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));

    assert(_len_inner_output % _len_inner_input == 0);
    // leftover_memory =
    // device.get_gpu_memory("leftover", _len_inner_input * _len_outer);
    // num_leftover = 0;
}

cudaRechunk::~cudaRechunk() {}

cudaEvent_t cudaRechunk::execute(cudaPipelineState& pipestate,
                                 const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    size_t input_frame_len = _len_inner_input * _len_outer;
    void* input_memory = device.get_gpu_memory_array(_gpu_mem_input, pipestate.gpu_frame_id, input_frame_len);

    size_t output_frame_len = _len_inner_output * _len_outer;
    void* accum_memory = device.get_gpu_memory("accum", output_frame_len);

    size_t n_copy = _len_inner_input;
    if (num_accumulated + _len_inner_input > _len_inner_output) {
        n_copy = _len_inner_output - num_accumulated;
        // Copy the remainder into the leftover_memory.
    }

    record_start_event(pipestate.gpu_frame_id);

    // if (num_leftover) copy leftover_memory to output, incr. num_accumulated

    CHECK_CUDA_ERROR(cudaMemcpy2DAsync((void*)((char*)accum_memory + num_accumulated),
                                       _len_inner_output, input_memory, _len_inner_input, n_copy,
                                       _len_outer, cudaMemcpyDeviceToDevice,
                                       device.getStream(cuda_stream_id)));
    num_accumulated += n_copy;
    if (num_accumulated >= _len_inner_output) {
        DEBUG("cudaRechunk: accumulated {:d}, output size {:d} -- producing output!",
              num_accumulated, _len_inner_output);
        // emit an output frame!
        void* output_memory =
            device.get_gpu_memory_array(_gpu_mem_output, output_frame_id, output_frame_len);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, accum_memory, output_frame_len,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));

        num_accumulated -= _len_inner_output;
        // (copy any overflow into the "leftover" array)

        // Set the flag to indicate that we have emitted a frame!
        if (_set_flag.size()) {
            pipestate.set_flag(_set_flag, true);
        }
        if (_output_frame_counter.size()) {
            DEBUG("cudaRechunk: set frame counter {:s} = {:d}", _output_frame_counter, output_frame_id);
            pipestate.set_frame_id(_output_frame_counter, output_frame_id);
        }
        output_frame_id = (output_frame_id + 1) % _gpu_buffer_depth;

    } else {
        DEBUG("cudaRechunk: accumulated {:d}, output size {:d} -- NOT producing output!",
              num_accumulated, _len_inner_output);
        // partial output frame -- don't run further GPU kernels.
    }
    return record_end_event(pipestate.gpu_frame_id);
}
