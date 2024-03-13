#ifndef CL_INPUT_DATA_H
#define CL_INPUT_DATA_H

#include "clCommand.hpp"
#include "gpuBufferHandler.hpp"

class clInputData : public clCommand {
public:
    clInputData(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, clDeviceInterface& device,
                int instance_num);
    ~clInputData();
    int wait_on_precondition() override;
    cl_event execute(cl_event pre_event) override;
    void finalize_frame() override;

protected:
    /// Helper class to manage the buffers
    gpuBufferHandler in_bufs;

    cl_event* data_staged_event;

    /// Name of the GPU side memory to transfer data into.
    std::string _gpu_memory;
};

#endif // CL_INPUT_DATA_H
