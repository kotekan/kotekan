#include "cudaUpchannelize.hpp"

#include "chordMetadata.hpp"
#include "cudaUtils.hpp"
#include "math.h"

#include "fmt.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

const int sizeof_float16_t = 2;

REGISTER_CUDA_COMMAND(cudaUpchannelize);

cudaUpchannelize::cudaUpchannelize(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, cudaDeviceInterface& device,
                                   std::string name, std::string kernel_fn, int nsamples,
                                   std::string kernel_symbol) :
    cudaCommand(config, unique_name, host_buffers, device, no_cuda_state, name, kernel_fn),
    kernel_name(kernel_symbol) {
    _num_dishes = config.get<int>(unique_name, "num_dishes");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _upchan_factor = config.get<int>(unique_name, "upchan_factor");
    _gpu_mem_input_voltage = config.get<std::string>(unique_name, "gpu_mem_input_voltage");
    _gpu_mem_output_voltage = config.get<std::string>(unique_name, "gpu_mem_output_voltage");
    _gpu_mem_gain = config.get<std::string>(unique_name, "gpu_mem_gain");
    _gpu_mem_info = unique_name + "/info";
    // Try config value "freq_gains" as either a scalar float or a vector of floats
    float gain0 = config.get_default<float>(unique_name, "freq_gains", 1.);
    std::vector<float> gains =
        config.get_default<std::vector<float>>(unique_name, "freq_gains", std::vector<float>());

    if (_num_dishes != cuda_ndishes)
        throw std::runtime_error(fmt::format(
            "The num_dishes config setting must be {:d} for the CUDA Upchannelizer", cuda_ndishes));
    if (_num_local_freq != cuda_nfreq)
        throw std::runtime_error(
            fmt::format("The num_local_freq config setting must be {:d} for the CUDA Upchannelizer",
                        cuda_nfreq));
    if (_samples_per_data_set != nsamples)
        throw std::runtime_error(fmt::format(
            "The samples_per_data_set config setting must be {:d} for the CUDA Upchannelizer",
            nsamples));

    size_t ngains = _num_local_freq * _upchan_factor;
    if (gains.size() == 0) {
        for (size_t i = 0; i < ngains; i++)
            gains.push_back(gain0);
    }
    if (gains.size() != ngains)
        throw std::runtime_error(
            fmt::format("The number of elements in the 'freq_gains' config setting array must be "
                        "{:d} for the CUDA Upchannelizer",
                        ngains));
    // number of polarizations
    const int P = 2;
    gain_len = ngains * sizeof(float16_t);
    // 2 complex terms, int4+4
    voltage_input_len = (size_t)_samples_per_data_set * P * _num_local_freq * _num_dishes;
    voltage_output_len = voltage_input_len;
    info_len = (size_t)(threads_x * threads_y * blocks_x * sizeof(int32_t));

    host_info.resize(_gpu_buffer_depth);
    for (int i = 0; i < _gpu_buffer_depth; i++)
        host_info[i].resize(info_len / sizeof(int32_t));

    set_command_type(gpuCommandType::KERNEL);

    std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_symbol}, opts);

    std::vector<float16_t> gains16;
    gains16.resize(gains.size());
    for (size_t i = 0; i < gains.size(); i++)
        gains16[i] = gains[i];

    const float16_t* gain_host = gains16.data();
    float16_t* gain_gpu = (float16_t*)device.get_gpu_memory(_gpu_mem_gain, gain_len);
    CHECK_CUDA_ERROR(cudaMemcpy(gain_gpu, gain_host, gain_len, cudaMemcpyHostToDevice));

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_gain, false, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input_voltage, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output_voltage, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_info", false, true, true));

    // DEBUG
    for (int i = 0; i < _gpu_buffer_depth; i++) {
        void* out_memory =
            device.get_gpu_memory_array(_gpu_mem_output_voltage, i, voltage_output_len);
        DEBUG("GPUMEM memory_array({:p}, {:d}, {:d}, \"{:s}\", \"{:s}\", "
              "\"upchan_output_voltage[{:d}]\")",
              out_memory, voltage_output_len, i, get_name(), _gpu_mem_output_voltage, i);
    }
}

cudaUpchannelize::~cudaUpchannelize() {}


// This struct is Erik's interpretation of what Julia is expecting for its "CuDevArray" type.
template<typename T, int64_t N>
struct CuDeviceArray {
    T* ptr;
    int64_t maxsize;
    int64_t dims[N];
    int64_t len;
};
typedef CuDeviceArray<int32_t, 1> kernel_arg;

cudaEvent_t cudaUpchannelize::execute(cudaPipelineState& pipestate,
                                      const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    void* voltage_input_memory = device.get_gpu_memory_array(
        _gpu_mem_input_voltage, pipestate.gpu_frame_id, voltage_input_len);
    void* voltage_output_memory = device.get_gpu_memory_array(
        _gpu_mem_output_voltage, pipestate.gpu_frame_id, voltage_output_len);
    float16_t* gain_memory = (float16_t*)device.get_gpu_memory(_gpu_mem_gain, gain_len);
    int32_t* info_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_info, info_len);

    // If input voltage array has metadata, create new metadata for output.
    metadataContainer* mc =
        device.get_gpu_memory_array_metadata(_gpu_mem_input_voltage, pipestate.gpu_frame_id);
    if (mc && metadata_container_is_chord(mc)) {
        metadataContainer* mc_out = device.create_gpu_memory_array_metadata(
            _gpu_mem_output_voltage, pipestate.gpu_frame_id, mc->parent_pool);
        chordMetadata* meta_out = get_chord_metadata(mc_out);
        chordMetadata* meta_in = get_chord_metadata(mc);
        chord_metadata_copy(meta_out, meta_in);
        DEBUG("cudaUpchannelize: input array shape: {:s}", meta_in->get_dimensions_string());
        assert(meta_in->get_dimension_name(0) == "T");
        assert(meta_in->get_dimension_name(2) == "F");
        meta_out->dim[0] /= _upchan_factor;
        meta_out->dim[2] *= _upchan_factor;
        DEBUG("cudaUpchannelize: output array shape: {:s}", meta_out->get_dimensions_string());
        for (int i = 0; i < meta_in->nfreq; i++) {
            meta_out->freq_upchan_factor[i] *= _upchan_factor;
            // TODO -- compute this complicated quantity!!!
            // meta_out->half_fpga_sample0[i] = ;
            meta_out->time_downsampling_fpga[i] *= _upchan_factor;
        }
    }

    record_start_event(pipestate.gpu_frame_id);

    // Initialize info_memory return codes
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_len, device.getStream(cuda_stream_id)));

    // A, E, s, J
    const char* exc = "exception";
    kernel_arg arr[4];

    arr[0].ptr = (int32_t*)gain_memory;
    arr[0].maxsize = gain_len;
    arr[0].dims[0] = gain_len / sizeof(float16_t);
    arr[0].len = gain_len / sizeof(float16_t);

    arr[1].ptr = (int32_t*)voltage_input_memory;
    arr[1].maxsize = voltage_input_len;
    arr[1].dims[0] = voltage_input_len;
    arr[1].len = voltage_input_len;

    arr[2].ptr = (int32_t*)voltage_output_memory;
    arr[2].maxsize = voltage_output_len;
    arr[2].dims[0] = voltage_output_len;
    arr[2].len = voltage_output_len;

    arr[3].ptr = (int32_t*)info_memory;
    arr[3].maxsize = info_len;
    arr[3].dims[0] = info_len / sizeof(int32_t);
    arr[3].len = info_len / sizeof(int32_t);

    void* parameters[] = {
        &(exc), &(arr[0]), &(arr[1]), &(arr[2]), &(arr[3]),
    };

    DEBUG("Kernel_name: {}", kernel_name);
    DEBUG("runtime_kernels[kernel_name]: {}", (void*)runtime_kernels[kernel_name]);
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shared_mem_bytes));

    DEBUG("Running CUDA Upchannelizer on GPU frame {:d}", pipestate.gpu_frame_id);
    CHECK_CU_ERROR(cuLaunchKernel(runtime_kernels[kernel_name], blocks_x, blocks_y, 1, threads_x,
                                  threads_y, 1, shared_mem_bytes, device.getStream(cuda_stream_id),
                                  parameters, NULL));

    DEBUG("GPUMEM kernel_out({:p}, {:d}, \"{:s}\", \"upchannelizer for frame {:d}\")",
          voltage_output_memory, voltage_output_len, get_name(),
          pipestate.get_int("gpu_frame_counter"));

    // Copy "info" result code back to host memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info[pipestate.gpu_frame_id].data(), info_memory,
                                     info_len, cudaMemcpyDeviceToHost,
                                     device.getStream(cuda_stream_id)));

    return record_end_event(pipestate.gpu_frame_id);
}

void cudaUpchannelize::finalize_frame(int gpu_frame_id) {
    device.release_gpu_memory_array_metadata(_gpu_mem_output_voltage, gpu_frame_id);

    cudaCommand::finalize_frame(gpu_frame_id);
    for (size_t i = 0; i < host_info[gpu_frame_id].size(); i++)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaUpchannelize returned 'info' value {:d} at index {:d} (zero indicates no "
                  "error)",
                  host_info[gpu_frame_id][i], i);
}
