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
    num_accumulated = 0;

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));

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
    void* accum_memory = device.get_gpu_memory("accum", output_frame_len);

    size_t n_copy = _cols_input;
    if (num_accumulated + _cols_input > _cols_output) {
        n_copy = _cols_output - num_accumulated;
        // Copy the remainder into the leftover_memory.
    }

    record_start_event(pipestate.gpu_frame_id);

    // if (num_leftover) copy leftover_memory to output, incr. num_accumulated

    CHECK_CUDA_ERROR(cudaMemcpy2DAsync((void*)((char*)accum_memory + num_accumulated),
                                       _cols_output, input_memory, _cols_input, n_copy,
                                       _rows, cudaMemcpyDeviceToDevice,
                                       device.getStream(cuda_stream_id)));
    num_accumulated += n_copy;
    if (num_accumulated >= _cols_output) {
        DEBUG("cudaRechunk: accumulated {:d}, output size {:d} -- producing output!",
              num_accumulated, _cols_output);
        // emit an output frame!
        void* output_memory =
            device.get_gpu_memory_array(_gpu_mem_output, pipestate.gpu_frame_id, output_frame_len);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, accum_memory, output_frame_len,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));

        num_accumulated -= _cols_output;
        // (copy any overflow into the "leftover" array)

        // Set the flag to indicate that we have emitted a frame!
        if (_set_flag.size()) {
            DEBUG("cudaRechunk: set pipeline flag {:s}", _set_flag);
            pipestate.set_flag(_set_flag, true);
        }
    } else {
        DEBUG("cudaRechunk: accumulated {:d}, output size {:d} -- NOT producing output!",
              num_accumulated, _cols_output);
        // partial output frame -- don't run further GPU kernels.
    }
    return record_end_event(pipestate.gpu_frame_id);
}
