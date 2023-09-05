#ifndef CL_INPUT_DATA_H
#define CL_INPUT_DATA_H

#include "clCommand.hpp"
#include "gpuBufferHandler.hpp"

class clInputData : public clCommand {
public:
    clInputData(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, clDeviceInterface& device);
    ~clInputData();
    int wait_on_precondition(int gpu_frame_id) override;
    cl_event execute(int gpu_frame_id, cl_event pre_event) override;
    void finalize_frame(int frame_id) override;

protected:
    /// Helper class to manage the buffers
    gpuBufferHandler in_bufs;

    cl_event* data_staged_event;

    /// Name of the GPU side memory to transfer data into.
    std::string _gpu_memory;
};

#endif // CL_INPUT_DATA_H
