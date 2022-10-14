#ifndef CL_OUTPUT_DATA_H
#define CL_OUTPUT_DATA_H

#include "clCommand.hpp"
#include "gpuBufferHandler.hpp"

class clOutputData : public clCommand {
public:
    clOutputData(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& host_buffers, clDeviceInterface& device);
    ~clOutputData();
    int wait_on_precondition(int gpu_frame_id) override;
    virtual cl_event execute(int buf_frame_id, cl_event pre_event) override;
    void finalize_frame(int frame_id) override;

private:
    /// Helper class to manage the input buffers
    gpuBufferHandler in_bufs;
    /// Helper class to manage the output buffers
    gpuBufferHandler out_bufs;

    /// Name of the GPU side memory to transfer data out of.
    std::string _gpu_memory;
};

#endif // CL_OUTPUT_DATA_H
