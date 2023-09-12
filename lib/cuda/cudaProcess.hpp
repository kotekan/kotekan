/**
 * @file
 * @brief Stage for running a set of CUDA commands
 *  - cudaProcess : public gpuProcess
 */

#ifndef CUDA_PROCESS_H
#define CUDA_PROCESS_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b)&0x0F)

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaEventContainer.hpp"
#include "gpuProcess.hpp"

/**
 * @class cudaProcess
 * @brief Stage to manage all the kernels and copy commands for a GPU
 *
 * This stage is responsible for running the cudaCommandObjects which in turn run the
 * various host<->device copies and kernel calls.  Much of the logic exists in the base
 * class @c gpuProcess, so that class for more details.
 *
 * @conf num_cuda_streams The number of CUDA streams to setup, the default is 3 for one
 *                        host->device, one device->host, and one kernel stream.
 *                        Can be set higher if more than one stream is need for each type
 *                        of operation.  See @c cudaCommand and @c cudaSyncStream for more details.
 *
 * @author Keith Vanderlinde and Andre Renard
 */
class cudaProcess final : public gpuProcess {
public:
    cudaProcess(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~cudaProcess();

    gpuCommand* create_command(const std::string& cmd_name,
                               const std::string& unique_name) override;
    gpuEventContainer* create_signal() override;
    void queue_commands(int gpu_frame_id, int gpu_frame_counter) override;

    void register_host_memory(struct Buffer* host_buffer) override;

    cudaDeviceInterface* device;
};

#endif // CUDA_PROCESS_H
