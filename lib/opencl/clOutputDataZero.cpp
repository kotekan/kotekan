#include "clOutputDataZero.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clOutputDataZero);

clOutputDataZero::clOutputDataZero(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, clDeviceInterface& device,
                                   int instance_num) :
    clCommand(config, unique_name, host_buffers, device, instance_num, no_cl_command_state, "clOutputDataZero", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _num_blocks = config.get<int>(unique_name, "num_blocks");

    output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2 * _num_data_sets
                 * sizeof(int32_t);


    int err;
    // Array used to zero the output memory on the device.
    // TODO should this be in it's own function?
    err = posix_memalign((void**)&output_zeros, PAGESIZE_MEM, output_len);
    if (err != 0) {
        ERROR("Error creating aligned memory for accumulate zeros");
        exit(err);
    }

    // Ask that all pages be kept in memory
    err = mlock((void*)output_zeros, output_len);
    if (err == -1) {
        ERROR("Error locking memory - check ulimit -a to check memlock limits");
        exit(errno);
    }
    memset(output_zeros, 0, output_len);


    cl_mem_prt =
        clCreateBuffer(device.get_context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                        output_len, output_zeros, &err);
    CHECK_CL_ERROR(err);
    void* pinned_ptr =
        clEnqueueMapBuffer(device.getQueue(0), cl_mem_prt, CL_TRUE, CL_MAP_READ, 0,
                            output_len, 0, nullptr, nullptr, &err);
    CHECK_CL_ERROR(err);
    assert(pinned_ptr == output_zeros);

    command_type = gpuCommandType::COPY_IN;
}

clOutputDataZero::~clOutputDataZero() {
    cl_event wait_event;
    clEnqueueUnmapMemObject(device.getQueue(0), cl_mem_prt,
                            output_len, 0, nullptr, &wait_event);
    // Block here to make sure the memory actually gets unmapped.
    clWaitForEvents(1, &wait_event);

    free(output_zeros);
}

cl_event clOutputDataZero::execute(cl_event pre_event) {
    pre_execute();

    cl_mem gpu_memory_frame = device.get_gpu_memory_array("output", gpu_frame_id, output_len);

    // Data transfer to GPU
    CHECK_CL_ERROR(clEnqueueWriteBuffer(device.getQueue(0), gpu_memory_frame, CL_FALSE,
                                        0, // offset
                                        output_len, output_zeros, 1, &pre_event, &post_event));
    return post_event;
}
