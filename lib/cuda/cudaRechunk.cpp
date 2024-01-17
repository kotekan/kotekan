#include "cudaRechunk.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaRechunk, cudaRechunkState);

cudaRechunk::cudaRechunk(Config& config, const std::string& unique_name,
                         bufferContainer& host_buffers, cudaDeviceInterface& device, int inst,
                         std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, inst, state) {
    _cols_input = config.get<int>(unique_name, "cols_input");
    _cols_output = config.get<int>(unique_name, "cols_output");
    _rows = config.get<int>(unique_name, "rows");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _set_flag = config.get_default<std::string>(unique_name, "set_flag", "");
    _input_columns_field = config.get_default<std::string>(unique_name, "input_columns_field", "");
    _output_async = config.get_default<bool>(unique_name, "output_async", false);
    output_id = 0;
    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaRechunk");

    gpu_mem_accum = unique_name + "/accum";

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "/accum", false, true, true));

    // pre-allocate memory buffers
    size_t output_len = _cols_output * _rows;
    device.get_gpu_memory(gpu_mem_accum, output_len);
}

cudaRechunk::~cudaRechunk() {}

cudaRechunkState* cudaRechunk::get_state() {
    return static_cast<cudaRechunkState*>(command_state.get());
}

cudaEvent_t cudaRechunk::execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>&) {
    pre_execute();

    size_t input_frame_len = _cols_input * _rows;
    void* input_memory = device.get_gpu_memory_array(_gpu_mem_input, gpu_frame_id,
                                                     _gpu_buffer_depth, input_frame_len);

    size_t output_len = _cols_output * _rows;
    void* accum_memory = device.get_gpu_memory(gpu_mem_accum, output_len);

    size_t cols_input = _cols_input;
    if (_input_columns_field.length()) {
        cols_input = pipestate.get_int(_input_columns_field);
        DEBUG("cudaRechunk: copying input elements: {:d}, vs avail buffer size {:d}", cols_input,
              _cols_input);
        assert(cols_input <= _cols_input);
    }

    size_t cols_to_copy = cols_input;
    size_t cols_leftover = 0;

    size_t cols_accumulated = get_state()->cols_accumulated;

    if (cols_accumulated + cols_to_copy > _cols_output) {
        cols_to_copy = _cols_output - cols_accumulated;
        // Copy the remainder into the leftover_memory.
        cols_leftover = cols_input - cols_to_copy;
    }

    record_start_event();

    CHECK_CUDA_ERROR(cudaMemcpy2DAsync((void*)((char*)accum_memory + cols_accumulated),
                                       _cols_output, input_memory, cols_input, cols_to_copy, _rows,
                                       cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    cols_accumulated += cols_to_copy;
    if (cols_accumulated >= _cols_output) {
        DEBUG("cudaRechunk: accumulated {:d} columns, output columns {:d} -- producing output!",
              cols_accumulated, _cols_output);
        // emit an output frame!
        int out_id = (_output_async ? output_id : pipestate.gpu_frame_id);
        output_id++;
        void* output_memory =
            device.get_gpu_memory_array(_gpu_mem_output, out_id, _gpu_buffer_depth, output_len);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, accum_memory, output_len,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));
        cols_accumulated -= _cols_output;
        // cols_accumulated should be zero at this point!
        assert(cols_accumulated == 0);

        // After copying 'accum' to 'output', if there were any inputs left over, copy them
        // to the start of the 'accum' array for next time.
        if (cols_leftover) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                accum_memory, (void*)((char*)input_memory + cols_to_copy), cols_leftover * _rows,
                cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
            cols_accumulated = cols_leftover;
        }

        // Set the flag to indicate that we have emitted a frame!
        if (_set_flag.size()) {
            DEBUG("cudaRechunk: set pipeline flag {:s}", _set_flag);
            pipestate.set_flag(_set_flag, true);
        }
    } else {
        DEBUG("cudaRechunk: accumulated {:d} columns, output columns {:d} -- NOT producing output!",
              cols_accumulated, _cols_output);
    }

    get_state()->cols_accumulated = cols_accumulated;

    return record_end_event();
}
