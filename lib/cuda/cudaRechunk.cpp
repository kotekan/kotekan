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
    _cols_input = config.get<int>(unique_name, "cols_input");
    _cols_output = config.get<int>(unique_name, "cols_output");
    _rows = config.get<int>(unique_name, "rows");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _set_flag = config.get_default<std::string>(unique_name, "set_flag", "");
    set_command_type(gpuCommandType::KERNEL);
    cols_accumulated = 0;

    gpu_mem_accum = unique_name + "/accum";

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "/accum", false, true, true));

    assert(_cols_output % _cols_input == 0);
    // leftover_memory =
    // device.get_gpu_memory("leftover", _cols_input * _rows);
    // num_leftover = 0;
}

cudaRechunk::~cudaRechunk() {}

cudaEvent_t cudaRechunk::execute(cudaPipelineState& pipestate,
                                 const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    size_t input_frame_len = _cols_input * _rows;
    void* input_memory = device.get_gpu_memory_array(_gpu_mem_input, pipestate.gpu_frame_id, input_frame_len);

    size_t output_frame_len = _cols_output * _rows;
    void* accum_memory = device.get_gpu_memory(gpu_mem_accum, output_frame_len);

    size_t cols_to_copy = _cols_input;
    if (cols_accumulated + _cols_input > _cols_output) {
        cols_to_copy = _cols_output - cols_accumulated;
        // Copy the remainder into the leftover_memory.
    }

    record_start_event(pipestate.gpu_frame_id);

    // if (num_leftover) copy leftover_memory to output, incr. cols_accumulated

    CHECK_CUDA_ERROR(cudaMemcpy2DAsync((void*)((char*)accum_memory + cols_accumulated),
                                       _cols_output, input_memory, _cols_input, cols_to_copy,
                                       _rows, cudaMemcpyDeviceToDevice,
                                       device.getStream(cuda_stream_id)));
    cols_accumulated += cols_to_copy;
    if (cols_accumulated >= _cols_output) {
        DEBUG("cudaRechunk: accumulated {:d} columns, output columns {:d} -- producing output!",
              cols_accumulated, _cols_output);
        // emit an output frame!
        void* output_memory =
            device.get_gpu_memory_array(_gpu_mem_output, pipestate.gpu_frame_id, output_frame_len);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, accum_memory, output_frame_len,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));

        cols_accumulated -= _cols_output;
        // (copy any overflow into the "leftover" array)

        // Set the flag to indicate that we have emitted a frame!
        if (_set_flag.size()) {
            DEBUG("cudaRechunk: set pipeline flag {:s}", _set_flag);
            pipestate.set_flag(_set_flag, true);
        }
    } else {
        DEBUG("cudaRechunk: accumulated {:d} columns, output columns {:d} -- NOT producing output!",
              cols_accumulated, _cols_output);
        // partial output frame -- don't run further GPU kernels.
    }
    return record_end_event(pipestate.gpu_frame_id);
}
