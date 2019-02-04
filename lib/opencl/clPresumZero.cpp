#include "clPresumZero.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clPresumZero);

clPresumZero::clPresumZero(Config& config, const string& unique_name, bufferContainer& host_buffers,
                           clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "", "") {
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
    // TODO: DO WITH STANDARD BUFFERS?
    err = mlock((void*)presum_zeros, presum_len);
    if (err == -1) {
        ERROR("Error locking memory - check ulimit -a to check memlock limits");
        exit(errno);
    }
    memset(presum_zeros, 0, presum_len);

    command_type = gpuCommandType::COPY_IN;
}

clPresumZero::~clPresumZero() {
    free(presum_zeros);
}

cl_event clPresumZero::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    cl_mem gpu_memory_frame = device.get_gpu_memory_array("presum", gpu_frame_id, presum_len);

    // Data transfer to GPU
    CHECK_CL_ERROR(clEnqueueWriteBuffer(device.getQueue(0), gpu_memory_frame, CL_FALSE,
                                        0, // offset
                                        presum_len, presum_zeros, 1, &pre_event,
                                        &post_events[gpu_frame_id]));
    return post_events[gpu_frame_id];
}
