#include "cudaFRBBeamformer.hpp"

#include "math.h"

#include "fmt.hpp"

#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaFRBBeamformer);

static const size_t sizeof_float16_t = 2;

cudaFRBBeamformer::cudaFRBBeamformer(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, cudaDeviceInterface& device,
                                     int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "FRB_beamformer", "frb.ptx") {

    // HACK -- at the very beginning of the run, we pretend we have 40 samples of padding,
    // so that we get 51 outputs (otherwise we would get 50), so that the rechunker produces
    // an output frame of 256 outputs after every 5 input frames.
    padded_samples = 40;
    // padded_samples = 0;

    _num_dishes = config.get<int>(unique_name, "num_dishes");
    _dish_grid_size = config.get<int>(unique_name, "dish_grid_size");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    // Number of input samples per frame
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    // Number of input samples worth of padding on the voltage array
    _samples_padding = config.get<int>(unique_name, "samples_padding");
    _time_downsampling = config.get<int>(unique_name, "time_downsampling");
    _gpu_mem_dishlayout = config.get<std::string>(unique_name, "gpu_mem_dishlayout");
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_beamgrid = config.get<std::string>(unique_name, "gpu_mem_beamgrid");
    gpu_mem_info = unique_name + "/info";
    gpu_mem_length = unique_name + "/length";

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_dishlayout, false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_phase, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beamgrid, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_info", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_length", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    if (inst == 0) {
        std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_name}, opts);
    }

    /*if (_samples_per_data_set != cuda_samples_per_data_set)
        throw std::runtime_error(fmt::format(
            "The samples_per_data_set config setting must be {:d} for the CUDA FRB beamformer",
            cuda_samples_per_data_set));
    */
    if (_samples_padding < (cuda_input_chunk + cuda_time_downsampling))
        throw std::runtime_error(fmt::format(
            "The samples_padding config setting must be >= {:d} for the CUDA FRB beamformer",
            cuda_input_chunk + cuda_time_downsampling));
    if (_time_downsampling != cuda_time_downsampling)
        throw std::runtime_error(fmt::format(
            "The time_downsampling config setting must be {:d} for the CUDA FRB beamformer",
            cuda_time_downsampling));
    if (_num_dishes != cuda_num_dishes)
        throw std::runtime_error(
            fmt::format("The num_dishes config setting must be {:d} for the CUDA FRB beamformer",
                        cuda_num_dishes));
    if (_dish_grid_size != cuda_dish_grid_size)
        throw std::runtime_error(fmt::format(
            "The dish_grid_size config setting must be {:d} for the CUDA FRB beamformer",
            cuda_dish_grid_size));
    if (_num_local_freq != cuda_num_local_freq)
        throw std::runtime_error(fmt::format(
            "The num_local_freq config setting must be {:d} for the CUDA FRB beamformer",
            cuda_num_local_freq));

    // Assumption:
    int32_t dish_m = _dish_grid_size;
    int32_t dish_n = _dish_grid_size;
    int32_t beam_p = _dish_grid_size * 2;
    int32_t beam_q = _dish_grid_size * 2;
    if (dish_m * dish_n < _num_dishes)
        throw std::runtime_error("Config parameter dish_grid_size^2 must be >= num_dishes");
    int32_t Td = (_samples_per_data_set + _time_downsampling - 1) / _time_downsampling;

    dishlayout_len = (size_t)dish_m * dish_n * 2 * sizeof(int16_t);
    // 2 polarizations x 2 for complex
    phase_len = (size_t)dish_m * dish_n * _num_local_freq * 2 * 2 * sizeof_float16_t;
    // 2 polarizations
    voltage_len_per_sample = (size_t)_num_dishes * _num_local_freq * 2;
    voltage_len = voltage_len_per_sample * (_samples_per_data_set + _samples_padding);
    beamgrid_len = (size_t)beam_p * beam_q * _num_local_freq * Td * sizeof_float16_t;
    info_len = (size_t)(threads_x * threads_y * blocks_x * sizeof(int32_t));
    length_len = sizeof(int32_t);

    host_info.resize(_gpu_buffer_depth);
    for (int i = 0; i < _gpu_buffer_depth; i++)
        host_info[i].resize(info_len / sizeof(int32_t));

    host_length.resize(_gpu_buffer_depth);

    // Allocate GPU memory for dish-layout array, and fill from the config file entry!
    int16_t* dishlayout_memory =
        (int16_t*)device.get_gpu_memory(_gpu_mem_dishlayout, dishlayout_len);
    std::vector<int> dishlayout_config =
        config.get<std::vector<int>>(unique_name, "frb_beamformer_dish_layout");
    if (dishlayout_config.size() != (size_t)(dish_m * dish_n * 2))
        throw std::runtime_error(fmt::format("Config parameter frb_beamformer_dish_layout (length "
                                             "{}) must have length = 2 * dish_grid_size^2 = {}",
                                             dishlayout_config.size(), 2 * dish_m * dish_n));
    int16_t* dishlayout_cpu_memory = (int16_t*)malloc(dishlayout_len);
    for (size_t i = 0; i < dishlayout_len / sizeof(int16_t); i++)
        dishlayout_cpu_memory[i] = dishlayout_config[i];
    CHECK_CUDA_ERROR(cudaMemcpy(dishlayout_memory, dishlayout_cpu_memory, dishlayout_len,
                                cudaMemcpyHostToDevice));
    free(dishlayout_cpu_memory);

    // DEBUG
    if (inst == 0) {
        for (int i = 0; i < _gpu_buffer_depth; i++) {
            void* voltage_memory = device.get_gpu_memory_array(_gpu_mem_voltage, i, voltage_len);
            DEBUG("GPUMEM memory_array({:p}, {:d}, {:d}, \"{:s}\", \"{:s}\", "
                  "\"frb_bf_input_voltage[{:d}]\")",
                  voltage_memory, voltage_len, i, get_name(), _gpu_mem_voltage, i);
        }
    }
}

cudaFRBBeamformer::~cudaFRBBeamformer() {}

// This struct is Erik's interpretation of what Julia is expecting for its "CuDevArray" type.
template<typename T, int64_t N>
struct CuDeviceArray {
    T* ptr;
    int64_t maxsize;
    int64_t dims[N];
    int64_t len;
};
typedef CuDeviceArray<int32_t, 1> kernel_arg;

cudaEvent_t cudaFRBBeamformer::execute(cudaPipelineState& pipestate,
                                       const std::vector<cudaEvent_t>&) {
    pre_execute();

    void* dishlayout_memory = device.get_gpu_memory(_gpu_mem_dishlayout, dishlayout_len);
    void* phase_memory = device.get_gpu_memory_array(_gpu_mem_phase, gpu_frame_id, phase_len);
    void* voltage_memory = device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, voltage_len);
    void* beamgrid_memory =
        device.get_gpu_memory_array(_gpu_mem_beamgrid, gpu_frame_id, beamgrid_len);
    int32_t* length_memory = (int32_t*)device.get_gpu_memory(gpu_mem_length, length_len);
    int32_t* info_memory = (int32_t*)device.get_gpu_memory(gpu_mem_info, info_len);

    record_start_event();

    // Initialize info_memory return codes
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_len, device.getStream(cuda_stream_id)));

    int gpu_frame_index = gpu_frame_id % _gpu_buffer_depth;

    int32_t valid = _samples_per_data_set + padded_samples;
    int32_t process = (valid / cuda_input_chunk) * cuda_input_chunk;
    int32_t output_frames = (process / cuda_time_downsampling);
    int32_t output_samples = (process / cuda_time_downsampling) * cuda_time_downsampling;
    host_length[gpu_frame_index] = process;
    int32_t padding_next = valid - output_samples;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(length_memory, host_length.data() + gpu_frame_index,
                                     length_len, cudaMemcpyHostToDevice,
                                     device.getStream(cuda_stream_id)));

    // Compute the offset of the voltage array to pass to the GPU kernel.
    void* voltage_input = (char*)voltage_memory
                          + (size_t)(_samples_padding - padded_samples) * voltage_len_per_sample;
    // Compute the length of the voltage array to pass to the GPU kernel.
    size_t voltage_input_len = (size_t)process * voltage_len_per_sample;

    DEBUG(
        "CUDA FRB Beamformer, GPU frame {:d}: {:d} new samples + {:d} left over = {:d}, processing "
        "{:d}, producing {:d} output samples = {:d} input samples, leaving {:d} for next time",
        gpu_frame_id, _samples_per_data_set, padded_samples, valid, process, output_frames,
        output_samples, padding_next);

    // Padding for next frame...
    padded_samples = padding_next;
    // Copy these padding samples into place!!
    void* voltage_pad = (char*)voltage_input + (size_t)output_samples * voltage_len_per_sample;
    void* voltage_next =
        device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id + 1, voltage_len);
    voltage_next =
        (char*)voltage_next + (size_t)(_samples_padding - padded_samples) * voltage_len_per_sample;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(voltage_next, voltage_pad,
                                     padded_samples * voltage_len_per_sample,
                                     cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    DEBUG("GPUMEM copyasync({:p}, {:p}, {:d}, \"{:s}\", \"padding: {:d} samples for frame {:d}\")",
          voltage_next, voltage_pad, padded_samples * voltage_len_per_sample, get_name(),
          padded_samples, gpu_frame_id);

    // Set the number of output samples produced!
    pipestate.set_int("frb_bf_samples", output_frames);
    int32_t beam_p = _dish_grid_size * 2;
    int32_t beam_q = _dish_grid_size * 2;
    pipestate.set_int("frb_bf_bytes_per_freq",
                      (size_t)output_frames * beam_p * beam_q * sizeof_float16_t);

    // input samples, dishlayout (S), phase (W), voltage (E), beamgrid (I), info
    const char* exc = "exception";
    kernel_arg arr[7];

    arr[0].ptr = (int32_t*)length_memory;
    arr[0].maxsize = length_len;
    arr[0].dims[0] = length_len / sizeof(int32_t);
    arr[0].len = length_len / sizeof(int32_t);

    arr[1].ptr = (int32_t*)dishlayout_memory;
    arr[1].maxsize = dishlayout_len;
    arr[1].dims[0] = dishlayout_len / sizeof(int16_t);
    arr[1].len = dishlayout_len / sizeof(int16_t);

    arr[2].ptr = (int32_t*)phase_memory;
    arr[2].maxsize = phase_len;
    arr[2].dims[0] = phase_len / sizeof_float16_t;
    arr[2].len = phase_len / sizeof_float16_t;

    arr[3].ptr = (int32_t*)voltage_input;
    arr[3].maxsize = voltage_input_len;
    arr[3].dims[0] = voltage_input_len;
    arr[3].len = voltage_input_len;

    arr[4].ptr = (int32_t*)beamgrid_memory;
    arr[4].maxsize = beamgrid_len;
    arr[4].dims[0] = beamgrid_len / sizeof_float16_t;
    arr[4].len = beamgrid_len / sizeof_float16_t;

    arr[5].ptr = (int32_t*)info_memory;
    arr[5].maxsize = info_len;
    arr[5].dims[0] = info_len / sizeof(int32_t);
    arr[5].len = info_len / sizeof(int32_t);

    arr[6].ptr = (int32_t*)nullptr;
    arr[6].maxsize = 0;
    arr[6].dims[0] = 0;
    arr[6].len = 0;

    void* parameters[] = {
        &(exc), //&(arr[0]),
        &(arr[1]), &(arr[2]), &(arr[3]), &(arr[4]), &(arr[5]), &(arr[6]),
    };

    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shared_mem_bytes));

    DEBUG("Running CUDA FRB Beamformer on GPU frame {:d}", gpu_frame_id);
    CHECK_CU_ERROR(cuLaunchKernel(device.runtime_kernels[kernel_name], blocks_x, blocks_y, 1,
                                  threads_x, threads_y, 1, shared_mem_bytes,
                                  device.getStream(cuda_stream_id), parameters, NULL));

    DEBUG("GPUMEM kernel_in({:p}, {:d}, \"{:s}\", \"frb beamformer for frame {:d}\")",
          voltage_input, voltage_input_len, get_name(), gpu_frame_id);

    // Copy "info" result code back to host memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info[gpu_frame_index].data(), info_memory, info_len,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    return record_end_event();
}

void cudaFRBBeamformer::finalize_frame() {
    cudaCommand::finalize_frame();
    for (size_t i = 0; i < host_info[gpu_frame_id % _gpu_buffer_depth].size(); i++)
        if (host_info[gpu_frame_id % _gpu_buffer_depth][i] != 0)
            ERROR("cudaFRBBeamformer returned 'info' value {:d} at index {:d} (zero indicates no "
                  "error)",
                  host_info[gpu_frame_id % _gpu_buffer_depth][i], i);
}
