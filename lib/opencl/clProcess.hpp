#ifndef CL_PROCESS_H
#define CL_PROCESS_H

#include "clCommand.hpp"
#include "clDeviceInterface.hpp"
#include "clEventContainer.hpp"
#include "gpuProcess.hpp"

class clProcess final : public gpuProcess {
public:
    clProcess(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);
    virtual ~clProcess();

protected:
    std::vector<gpuCommand*> create_command(const std::string& cmd_name,
                                            const std::string& unique_name) override;
    gpuEventContainer* create_signal() override;
    void queue_commands(int gpu_frame_counter) override;
    void register_host_memory(Buffer* host_buffer) override;

    clDeviceInterface* device;

    /// Keep track of the OpenCL registered host memory corresponding
    /// to the host memory frames.
    std::vector<std::tuple<cl_mem, void*>> opencl_host_frames;
};

#endif // CL_PROCESS_H
