/**
 * @file
 * @brief CUDA FRBBeamformer kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

#include <array>
#include <bufferContainer.hpp>
#include <fmt.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

/**
 * @class cudaFRBBeamformer
 * @brief cudaCommand for FRBBeamformer
 */
class cudaFRBBeamformer : public cudaCommand {
public:
    cudaFRBBeamformer(Config& config, const std::string& unique_name, bufferContainer& host_buffers,
                      cudaDeviceInterface& device);
    virtual ~cudaFRBBeamformer();

    // int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>& pre_events) override;        
    void finalize_frame(int gpu_frame_id) override;

private:
    // Julia's `CuDevArray` type
    template<typename T, std::int64_t N>
    struct CuDeviceArray {
        T* ptr;
        std::int64_t maxsize; // bytes
        std::int64_t dims[N]; // elements
        std::int64_t len;     // elements
        CuDeviceArray(void* const ptr, const std::size_t bytes) :
            ptr(static_cast<T*>(ptr)), maxsize(bytes), dims{std::int64_t(maxsize / sizeof(T))},
            len(maxsize / sizeof(T)) {}
    };
    using kernel_arg = CuDeviceArray<int32_t, 1>;

    // Kernel design parameters:
    static constexpr int cuda_beam_layout_M = 48;
    static constexpr int cuda_beam_layout_N = 48;
    static constexpr int cuda_dish_layout_M = 24;
    static constexpr int cuda_dish_layout_N = 24;
    static constexpr int cuda_downsampling_factor = 40;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 256;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 2064;

    // Kernel compile parameters:
    static constexpr int minthreads = 768;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 24;
    static constexpr int blocks = 256;
    static constexpr int shmem_bytes = 76896;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z15julia_frb_1025413CuDeviceArrayI7Int16x2Li1ELi1EES_I9Float16x2Li1ELi1EES_"
        "I6Int4x8Li1ELi1EES_IS1_Li1ELi1EES_I5Int32Li1ELi1EE";

    // Kernel arguments:
    static constexpr std::size_t S_length = 2304UL;
    static constexpr std::size_t W_length = 1179648UL;
    static constexpr std::size_t E_length = 541065216UL;
    static constexpr std::size_t I_length = 239616UL;
    static constexpr std::size_t info_length = 786432UL;

    // Runtime parameters:

    // GPU memory:
    const std::string S_memname;
    const std::string W_memname;
    const std::string E_memname;
    const std::string I_memname;
    const std::string info_memname;

    /// Samples of padding on the voltage buffer.  Must be >= cuda_input_chunk +
    /// cuda_time_downsampling.
    int32_t _samples_padding;
    int padded_samples;
    /// Host-side buffer array for GPU kernel status/info output
    std::vector<std::vector<int32_t>> host_info;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamformer);

cudaFRBBeamformer::cudaFRBBeamformer(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "FRBBeamformer", "FRBBeamformer.ptx"),
    S_memname(config.get<std::string>(unique_name, "gpu_mem_dishlayout")),
    W_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    I_memname(config.get<std::string>(unique_name, "gpu_mem_beamgrid")),

    info_memname(unique_name + "/info"),

    // HACK -- at the very beginning of the run, we pretend we have 40 samples of padding,
    // so that we get 51 outputs (otherwise we would get 50), so that the rechunker produces
    // an output frame of 256 outputs after every 5 input frames.
    padded_samples(40) {
    // padded_samples = 0;

    //gpu_mem_length = unique_name + "/length";

    gpu_buffers_used.push_back(std::make_tuple(S_memname, false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(W_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I_memname, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_info", false, true, true));
    //gpu_buffers_used.push_back(std::make_tuple(get_name() + "_length", false, true, true));

    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    if (num_dishes != (cuda_number_of_dishes))
        throw std::runtime_error("The num_dishes config setting must be "
                                 + std::to_string(cuda_number_of_dishes)
                                 + " for the CUDA Baseband Beamformer");
    const int dish_grid_size = config.get<int>(unique_name, "dish_grid_size");
    if (dish_grid_size != (cuda_dish_layout_M))
        throw std::runtime_error("The dish_grid_size config setting must be "
                                 + std::to_string(cuda_dish_layout_M)
                                 + " for the CUDA Baseband Beamformer");
    const int num_local_freq = config.get<int>(unique_name, "num_local_freq");
    if (num_local_freq != (cuda_number_of_frequencies))
        throw std::runtime_error("The num_local_freq config setting must be "
                                 + std::to_string(cuda_number_of_frequencies)
                                 + " for the CUDA Baseband Beamformer");
    const int samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    if (samples_per_data_set != (cuda_number_of_timesamples))
        throw std::runtime_error("The samples_per_data_set config setting must be "
                                 + std::to_string(cuda_number_of_timesamples)
                                 + " for the CUDA Baseband Beamformer");
    const int time_downsampling = config.get<int>(unique_name, "time_downsampling");
    if (time_downsampling != (cuda_downsampling_factor))
        throw std::runtime_error("The time_downsampling config setting must be "
                                 + std::to_string(cuda_downsampling_factor)
                                 + " for the CUDA Baseband Beamformer");

    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_symbol}, opts);

    /*
     if (_samples_padding < (cuda_input_chunk + cuda_time_downsampling))
     throw std::runtime_error(fmt::format(
     "The samples_padding config setting must be >= {:d} for the CUDA FRB beamformer",
     cuda_input_chunk + cuda_time_downsampling));
     */

    // Assumption:
    int32_t dish_m = dish_grid_size;
    int32_t dish_n = dish_grid_size;
    int32_t beam_p = dish_grid_size * 2;
    int32_t beam_q = dish_grid_size * 2;
    if (dish_m * dish_n < num_dishes)
        throw std::runtime_error("Config parameter dish_grid_size^2 must be >= num_dishes");
    int32_t Td = (samples_per_data_set + time_downsampling - 1) / time_downsampling;

    size_t dishlayout_len = (size_t)dish_m * dish_n * 2 * sizeof(int16_t);
    // 2 polarizations x 2 for complex
    size_t phase_len = (size_t)dish_m * dish_n * num_local_freq * 2 * 2 * sizeof(float16_t); //sizeof_float16_t;
    // 2 polarizations
    size_t voltage_len_per_sample = (size_t)num_dishes * num_local_freq * 2;
    size_t voltage_len = voltage_len_per_sample * (samples_per_data_set + _samples_padding);
    size_t beamgrid_len = (size_t)beam_p * beam_q * num_local_freq * Td * sizeof(float16_t);//sizeof_float16_t;
    //info_length = (size_t)(threads_x * threads_y * blocks_x * sizeof(int32_t));
    //length_len = sizeof(int32_t);

    host_info.resize(_gpu_buffer_depth);
    for (int i = 0; i < _gpu_buffer_depth; i++)
        host_info[i].resize(info_length / sizeof(int32_t));
    //host_length.resize(_gpu_buffer_depth);

    // Allocate GPU memory for dish-layout array, and fill from the config file entry!
    int16_t* dishlayout_memory =
        (int16_t*)device.get_gpu_memory(S_memname, dishlayout_len);
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
    for (int i = 0; i < _gpu_buffer_depth; i++) {
        void* voltage_memory = device.get_gpu_memory_array(E_memname, i, voltage_len);
        DEBUG("GPUMEM memory_array({:p}, {:d}, {:d}, \"{:s}\", \"{:s}\", "
              "\"frb_bf_input_voltage[{:d}]\")",
              voltage_memory, voltage_len, i, get_name(), E_memname, i);
    }

    //
    // const std::string S_buffer_name = "host_" + S_memname;
    // Buffer* const S_buffer = host_buffers.get_buffer(S_buffer_name.c_str());
    // assert(S_buffer);
    //
    // register_consumer(S_buffer, unique_name.c_str());
    //
    //
    //
    // const std::string W_buffer_name = "host_" + W_memname;
    // Buffer* const W_buffer = host_buffers.get_buffer(W_buffer_name.c_str());
    // assert(W_buffer);
    //
    // register_consumer(W_buffer, unique_name.c_str());
    //
    //
    //
    // const std::string E_buffer_name = "host_" + E_memname;
    // Buffer* const E_buffer = host_buffers.get_buffer(E_buffer_name.c_str());
    // assert(E_buffer);
    //
    // register_consumer(E_buffer, unique_name.c_str());
    //
    //
    //
    // const std::string I_buffer_name = "host_" + I_memname;
    // Buffer* const I_buffer = host_buffers.get_buffer(I_buffer_name.c_str());
    // assert(I_buffer);
    //
    //
    // register_producer(I_buffer, unique_name.c_str());
    //
    //
    // const std::string info_buffer_name = "host_" + info_memname;
    // Buffer* const info_buffer = host_buffers.get_buffer(info_buffer_name.c_str());
    // assert(info_buffer);
    //
    //
    // register_producer(info_buffer, unique_name.c_str());
    //
    //
}

cudaFRBBeamformer::~cudaFRBBeamformer() {}

// int cudaFRBBeamformer::wait_on_precondition(const int gpu_frame_id) {
//
//
//     const std::string S_buffer_name = "host_" + S_memname;
//     Buffer* const S_buffer = host_buffers.get_buffer(S_buffer_name.c_str());
//     assert(S_buffer);
//     uint8_t* const S_frame = wait_for_full_frame(S_buffer, unique_name.c_str(), gpu_frame_id);
//     if (!S_frame)
//         return -1;
//
//
//
//     const std::string W_buffer_name = "host_" + W_memname;
//     Buffer* const W_buffer = host_buffers.get_buffer(W_buffer_name.c_str());
//     assert(W_buffer);
//     uint8_t* const W_frame = wait_for_full_frame(W_buffer, unique_name.c_str(), gpu_frame_id);
//     if (!W_frame)
//         return -1;
//
//
//
//     const std::string E_buffer_name = "host_" + E_memname;
//     Buffer* const E_buffer = host_buffers.get_buffer(E_buffer_name.c_str());
//     assert(E_buffer);
//     uint8_t* const E_frame = wait_for_full_frame(E_buffer, unique_name.c_str(), gpu_frame_id);
//     if (!E_frame)
//         return -1;
//
//
//
//
//
//
//
//     return 0;
// }

cudaEvent_t cudaFRBBeamformer::execute(cudaPipelineState& pipestate,
                                       const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute(pipestate.gpu_frame_id);

    void* const S_memory = device.get_gpu_memory_array(S_memname, pipestate.gpu_frame_id, S_length);
    void* const W_memory = device.get_gpu_memory_array(W_memname, pipestate.gpu_frame_id, W_length);
    void* const E_memory = device.get_gpu_memory_array(E_memname, pipestate.gpu_frame_id, E_length);
    void* const I_memory = device.get_gpu_memory_array(I_memname, pipestate.gpu_frame_id, I_length);
    //void* const info_memory = device.get_gpu_memory_array(info_memname, gpu_frame_id, info_length);
    //int32_t* length_memory = (int32_t*)device.get_gpu_memory(gpu_mem_length, length_len);
    int32_t* info_memory = (int32_t*)device.get_gpu_memory(info_memname, info_length);

    record_start_event(pipestate.gpu_frame_id);

    // Initialize info_memory return codes
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    int samples_per_data_set = cuda_number_of_timesamples;
    // Compiled-in constants in the CUDA kernel
    const int cuda_input_chunk = 48;
    const int cuda_time_downsampling = 40;
    int num_dishes = cuda_number_of_dishes;
    int num_local_freq = cuda_number_of_frequencies;
    size_t voltage_len_per_sample = (size_t)num_dishes * num_local_freq * 2;
    
    int32_t valid = samples_per_data_set + padded_samples;
    int32_t process = (valid / cuda_input_chunk) * cuda_input_chunk;
    int32_t output_frames = (process / cuda_time_downsampling);
    int32_t output_samples = (process / cuda_time_downsampling) * cuda_time_downsampling;
    //host_length[pipestate.gpu_frame_id] = process;
    int32_t padding_next = valid - output_samples;

    //CHECK_CUDA_ERROR(cudaMemcpyAsync(length_memory, host_length.data() + pipestate.gpu_frame_id,
    //                                 length_len, cudaMemcpyHostToDevice,
    //                                 device.getStream(cuda_stream_id)));

    // Compute the offset of the voltage array to pass to the GPU kernel.
    void* voltage_input = (char*)E_memory
                          + (size_t)(_samples_padding - padded_samples) * voltage_len_per_sample;
    // Compute the length of the voltage array to pass to the GPU kernel.
    size_t voltage_input_len = (size_t)process * voltage_len_per_sample;

    DEBUG(
        "CUDA FRB Beamformer, GPU frame {:d}: {:d} new samples + {:d} left over = {:d}, processing "
        "{:d}, producing {:d} output samples = {:d} input samples, leaving {:d} for next time",
        pipestate.gpu_frame_id, samples_per_data_set, padded_samples, valid, process,
        output_frames, output_samples, padding_next);

    // Padding for next frame...
    padded_samples = padding_next;
    // Copy these padding samples into place!!
    void* voltage_pad = (char*)voltage_input + (size_t)output_samples * voltage_len_per_sample;
    void* voltage_next = device.get_gpu_memory_array(
                                                     E_memname, (pipestate.gpu_frame_id + 1) % _gpu_buffer_depth, E_length);
    voltage_next =
        (char*)voltage_next + (size_t)(_samples_padding - padded_samples) * voltage_len_per_sample;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(voltage_next, voltage_pad,
                                     padded_samples * voltage_len_per_sample,
                                     cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    DEBUG("GPUMEM copyasync({:p}, {:p}, {:d}, \"{:s}\", \"padding: {:d} samples for frame {:d}\")",
          voltage_next, voltage_pad, padded_samples * voltage_len_per_sample, get_name(),
          padded_samples, pipestate.get_int("gpu_frame_counter"));

    // Set the number of output samples produced!
    pipestate.set_int("frb_bf_samples", output_frames);
    int _dish_grid_size = cuda_dish_layout_M;
    int32_t beam_p = _dish_grid_size * 2;
    int32_t beam_q = _dish_grid_size * 2;
    pipestate.set_int("frb_bf_bytes_per_freq",
                      (size_t)output_frames * beam_p * beam_q * sizeof(float16_t));

    const char* exc_arg = "exception";
    kernel_arg S_arg(S_memory, S_length);
    kernel_arg W_arg(W_memory, W_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg I_arg(I_memory, I_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &S_arg, &W_arg, &E_arg, &I_arg, &info_arg,
        // &length_arg...
    };

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));

    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA FRBBeamformer on GPU frame {:d}", pipestate.gpu_frame_id);
    DEBUG("GPUMEM kernel_in({:p}, {:d}, \"{:s}\", \"frb beamformer for frame {:d}\")",
          voltage_input, voltage_input_len, get_name(), pipestate.get_int("gpu_frame_counter"));

    const CUresult err =
        cuLaunchKernel(runtime_kernels[kernel_symbol], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("Error number: {}", err);
        ERROR("cuLaunchKernel: {}", errStr);
    }

    // Copy "info" result code back to host memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info[pipestate.gpu_frame_id].data(), info_memory,
                                     info_length, cudaMemcpyDeviceToHost,
                                     device.getStream(cuda_stream_id)));

    return record_end_event(pipestate.gpu_frame_id);
}

void cudaFRBBeamformer::finalize_frame(int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);
    for (size_t i = 0; i < host_info[gpu_frame_id].size(); i++)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaFRBBeamformer returned 'info' value {:d} at index {:d} (zero indicates no "
                  "error)",
                  host_info[gpu_frame_id][i], i);
}
// void cudaFRBBeamformer::finalize_frame(const int gpu_frame_id) {
//
//
//
//
//
//
//
//
//     const std::string I_buffer_name = "host_" + I_memname;
//     Buffer* const I_buffer = host_buffers.get_buffer(I_buffer_name.c_str());
//     assert(I_buffer);
//     mark_frame_full(I_buffer, unique_name.c_str(), gpu_frame_id);
//
//
//
//     const std::string info_buffer_name = "host_" + info_memname;
//     Buffer* const info_buffer = host_buffers.get_buffer(info_buffer_name.c_str());
//     assert(info_buffer);
//     mark_frame_full(info_buffer, unique_name.c_str(), gpu_frame_id);
//
//
// }

