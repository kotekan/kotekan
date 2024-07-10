#ifndef CL_OUTPUT_DATA_H
#define CL_OUTPUT_DATA_H

#include "clCommand.hpp"

class clOutputData : public clCommand {
public:
    clOutputData(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& host_buffers, clDeviceInterface& device,
                 int instance_num);
    ~clOutputData();
    int wait_on_precondition() override;
    virtual cl_event execute(cl_event pre_event) override;
    void finalize_frame() override;

private:
    /// Helper class to manage the input buffers
    std::vector<Buffer*> in_bufs;
    /// Helper class to manage the output buffers
    std::vector<Buffer*> out_bufs;

    /// Name of the GPU side memory to transfer data out of.
    std::string _gpu_memory;
};

#endif // CL_OUTPUT_DATA_H
