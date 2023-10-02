#include "clPreseedKernel.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clPreseedKernel);

clPreseedKernel::clPreseedKernel(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, clDeviceInterface& device,
				 int inst) :
    clCommand(config, unique_name, host_buffers, device, inst, no_cl_state, "preseed", "preseed_multifreq.cl") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");

    defineOutputDataMap(); // id_x_map and id_y_map depend on this call.

    command_type = gpuCommandType::KERNEL;
}

clPreseedKernel::~clPreseedKernel() {
    clReleaseMemObject(id_x_map);
    clReleaseMemObject(id_y_map);
}

void clPreseedKernel::build() {
    clCommand::build();

    cl_int err;

    cl_device_id dev_id = device.get_id();

    std::string cl_options = "";
    cl_options += " -D NUM_ELEMENTS=" + std::to_string(_num_elements);
    cl_options += " -D NUM_BLOCKS=" + std::to_string(_num_blocks);
    cl_options += " -D NUM_TIMESAMPLES=" + std::to_string(_samples_per_data_set);

    CHECK_CL_ERROR(clBuildProgram(program, 1, &dev_id, cl_options.c_str(), nullptr, nullptr));

    kernel = clCreateKernel(program, kernel_command.c_str(), &err);
    CHECK_CL_ERROR(err);

    CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(id_x_map),
                                  (void*)&id_x_map)); // this should maybe be sizeof(void *)?

    CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(id_y_map), (void*)&id_y_map));

    CHECK_CL_ERROR(clSetKernelArg(kernel, 4, 64 * sizeof(cl_uint), nullptr));

    CHECK_CL_ERROR(clSetKernelArg(kernel, 5, 64 * sizeof(cl_uint), nullptr));

    // Pre-seed kernel global and local work space sizes.
    gws[0] = 8 * _num_data_sets;
    gws[1] = 8 * _num_local_freq;
    gws[2] = _num_blocks;

    lws[0] = 8;
    lws[1] = 8;
    lws[2] = 1;
}

cl_event clPreseedKernel::execute(cl_event pre_event) {
    pre_execute();

    uint32_t presum_len = _num_elements * _num_local_freq * 2 * sizeof(int32_t);
    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                          * _num_data_sets * sizeof(int32_t);

    cl_mem output_memory_frame = device.get_gpu_memory_array("output", gpu_frame_id, output_len);
    cl_mem presum_memory = device.get_gpu_memory_array("presum", gpu_frame_id, presum_len);

    setKernelArg(0, presum_memory);
    setKernelArg(1, output_memory_frame);

    CHECK_CL_ERROR(clEnqueueNDRangeKernel(device.getQueue(1), kernel, 3, nullptr, gws, lws, 1,
                                          &pre_event, &post_event));

    return post_event;
}
void clPreseedKernel::defineOutputDataMap() {
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
        ERROR("Error in clCreateBuffer {:d}\n", err);
    }

    id_y_map = clCreateBuffer(device.get_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              _num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err) {
        ERROR("Error in clCreateBuffer {:d}\n", err);
    }
}
