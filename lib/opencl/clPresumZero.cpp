#include "clPresumZero.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clPresumZero);

clPresumZero::clPresumZero(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, clDeviceInterface& device,
                           int instance_num) :
    clCommand(config, unique_name, host_buffers, device, instance_num, no_cl_command_state, "clPresumZero", "") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    presum_len = _num_elements * _num_local_freq * 2 * sizeof(int32_t);

    int err;
    // Array used to zero the output memory on the device.
    // TODO should this be in it's own function?
    err = posix_memalign((void**)&presum_zeros, PAGESIZE_MEM, presum_len);
    if (err != 0) {
        ERROR("Error creating aligned memory for accumulate zeros");
        exit(err);
    }

    // Ask that all pages be kept in memory
    err = mlock((void*)presum_zeros, presum_len);
    if (err == -1) {
        ERROR("Error locking memory - check ulimit -a to check memlock limits");
        exit(errno);
    }
    memset(presum_zeros, 0, presum_len);

    cl_mem_prt =
        clCreateBuffer(device.get_context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                        presum_len, presum_zeros, &err);
    CHECK_CL_ERROR(err);
    void* pinned_ptr =
        clEnqueueMapBuffer(device.getQueue(0), cl_mem_prt, CL_TRUE, CL_MAP_READ, 0,
                            presum_len, 0, nullptr, nullptr, &err);
    CHECK_CL_ERROR(err);
    assert(pinned_ptr == presum_zeros);

    command_type = gpuCommandType::COPY_IN;
}

clPresumZero::~clPresumZero() {
    cl_event wait_event;
    clEnqueueUnmapMemObject(device.getQueue(0), cl_mem_prt,
                            presum_zeros, 0, nullptr, &wait_event);
    // Block here to make sure the memory actually gets unmapped.
    clWaitForEvents(1, &wait_event);

    free(presum_zeros);
}

cl_event clPresumZero::execute(cl_event pre_event) {
    pre_execute();

    device.async_copy_host_to_gpu("presum", gpu_frame_id, presum_zeros, presum_len, pre_event, post_event);                                    
    return post_event;
}
