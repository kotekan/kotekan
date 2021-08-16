#include "clKVCorr.hpp"

#include <string>
using std::string;

#define small_array (_num_elements < 32)

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clKVCorr);

clKVCorr::clKVCorr(Config& config, const std::string& unique_name, bufferContainer& host_buffers,
                   clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "corr", "kv_corr.cl") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _data_format = config.get_default<string>(unique_name, "data_format", "4+4b");
    _full_complicated = config.get_default<bool>(unique_name, "full_complicated", false);
    _legacy_opencl = config.get_default<bool>(unique_name, "legacy_opencl", false);

    if (_data_format == "4+4b") {
        if (small_array) {
            if (_legacy_opencl) {
                kernel_file_name =
                    config.get_default<string>(unique_name, "kernel_path", ".") + "/"
                    + config.get_default<string>(unique_name, "kernel", "kv_corr_sm_legacy.cl");
            } else {
                kernel_file_name =
                    config.get_default<string>(unique_name, "kernel_path", ".") + "/"
                    + config.get_default<string>(unique_name, "kernel", "kv_corr_sm.cl");
            }
        } else if (_full_complicated) {
            if (small_array)
                throw std::invalid_argument("Can't do full_complicated with num_elements < 32");
            else
                kernel_file_name =
                    config.get_default<string>(unique_name, "kernel_path", ".") + "/"
                    + config.get_default<string>(unique_name, "kernel", "kv_corr_amd.cl");
        }
    } else if (_data_format == "dot4b") {
        kernel_file_name = config.get_default<string>(unique_name, "kernel_path", ".") + "/"
                           + config.get_default<string>(unique_name, "kernel", "kv_corr_dot4b.cl");
    } else {
        throw std::invalid_argument(fmt::format(fmt("Unknown Data Format: {:s}"), _data_format));
    }

    defineOutputDataMap(); // id_x_map and id_y_map depend on this call.

    command_type = gpuCommandType::KERNEL;
}

clKVCorr::~clKVCorr() {
    free(zeros);

    clReleaseMemObject(id_x_map);
    clReleaseMemObject(id_y_map);
}

void clKVCorr::build() {
    clCommand::build();

    cl_int err;

    std::string cl_options = "";

    if (_data_format == "4+4b") {
        INFO("Running 4+4b CHIME-like data");
        int accum_length;
        // Correlation kernel global and local work space sizes.
        if (small_array) {
            accum_length = 4096 * 4;
            int num_accum = _samples_per_data_set / accum_length;
            gws[0] = _num_elements / 4 * _num_local_freq;
            gws[1] = _num_elements / 4 * num_accum;
            gws[2] = _num_blocks;
            lws[0] = _num_elements / 4;
            lws[1] = _num_elements / 4;
            lws[2] = 1;
        } else {
            accum_length = _samples_per_data_set;
            gws[0] = (_full_complicated ? 16 : 8) * _num_local_freq;
            gws[1] = (_full_complicated ? 4 : 8);
            gws[2] = _num_blocks;
            lws[0] = (_full_complicated ? 16 : 8);
            lws[1] = (_full_complicated ? 4 : 8);
            lws[2] = 1;
        }
        cl_options += " -D NUM_ELEMENTS=" + std::to_string(_num_elements);
        cl_options += " -D BLOCK_SIZE=" + std::to_string(_block_size);
        cl_options += " -D SAMPLES_PER_DATA_SET=" + std::to_string(accum_length);
        cl_options += " -D COARSE_BLOCK_SIZE=" + std::to_string(_block_size / 4);
    } else if (_data_format == "dot4b") {
        INFO("Running experimental dot-product data");
        int _wi_size = 4;
        gws[0] = _block_size / _wi_size;
        gws[1] = _block_size / _wi_size * _num_blocks;
        gws[2] = _num_local_freq;
        lws[0] = _block_size / _wi_size;
        lws[1] = _block_size / _wi_size;
        lws[2] = 1;
        cl_options += " -D NUM_ELEMENTS=" + std::to_string(_num_elements);
        cl_options += " -D BLOCK_SIZE=" + std::to_string(_block_size);
        cl_options += " -D SAMPLES_PER_DATA_SET=" + std::to_string(_samples_per_data_set);
        cl_options += " -D WI_SIZE=" + std::to_string(_wi_size);
        cl_options += " -D COARSE_BLOCK_SIZE=" + std::to_string(_block_size / _wi_size);
    } else {
        throw std::invalid_argument("Unknown Data Format: " + _data_format);
    }

    cl_device_id dev_id = device.get_id();

    err = clBuildProgram(program, 1, &dev_id, cl_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t len = 0;
        CHECK_CL_ERROR(clGetProgramBuildInfo(program, device.get_id(), CL_PROGRAM_BUILD_LOG, 0,
                                             nullptr, &len));
        char* buffer = (char*)calloc(len, sizeof(char));
        CHECK_CL_ERROR(clGetProgramBuildInfo(program, device.get_id(), CL_PROGRAM_BUILD_LOG, len,
                                             buffer, nullptr));
        INFO("CL failed. Build log follows: \n {:s}", buffer);
        free(buffer);
    }
    CHECK_CL_ERROR(err);
    kernel = clCreateKernel(program, "corr", &err);
    CHECK_CL_ERROR(err);

    // set other parameters that will be fixed for the kernels (changeable parameters will be set in
    // run loops)
    CHECK_CL_ERROR(clSetKernelArg(kernel, (cl_uint)3, sizeof(id_x_map),
                                  (void*)&id_x_map)); // this should maybe be sizeof(void *)?

    CHECK_CL_ERROR(clSetKernelArg(kernel, (cl_uint)4, sizeof(id_y_map), (void*)&id_y_map));

    zeros = (cl_int*)calloc(_num_blocks * _num_local_freq, sizeof(cl_int)); // for the output
                                                                            // buffers
}

cl_event clKVCorr::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                          * _num_data_sets * sizeof(int32_t);
    uint32_t presum_len = _num_elements * _num_local_freq * 2 * sizeof(int32_t);

    cl_mem input_memory = device.get_gpu_memory_array("voltage", gpu_frame_id, input_frame_len);
    cl_mem output_memory_frame = device.get_gpu_memory_array("output", gpu_frame_id, output_len);
    cl_mem presum_memory = device.get_gpu_memory_array("presum", gpu_frame_id, presum_len);

    setKernelArg(0, input_memory);
    setKernelArg(1, presum_memory);
    setKernelArg(2, output_memory_frame);

    CHECK_CL_ERROR(clEnqueueNDRangeKernel(device.getQueue(1), kernel, 3, nullptr, gws, lws, 1,
                                          &pre_event, &post_events[gpu_frame_id]));

    return post_events[gpu_frame_id];
}

void clKVCorr::defineOutputDataMap() {
    cl_int err;
    // Create lookup tables

    // upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[_num_blocks];
    unsigned int global_id_y_map[_num_blocks];

    // TODO: p260 OpenCL in Action has a clever while loop that changes 1 D addresses to X & Y
    // indices for an upper triangle.
    // Time Test kernels using them compared to the lookup tables for NUM_ELEM = 256
    int largest_num_blocks_1D = _num_elements / _block_size;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++) {
        for (int i = j; i < largest_num_blocks_1D; i++) {
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }

    id_x_map = clCreateBuffer(device.get_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              _num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err) {
        printf("Error in clCreateBuffer %i\n", err);
    }

    id_y_map = clCreateBuffer(device.get_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              _num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err) {
        printf("Error in clCreateBuffer %i\n", err);
    }
}
