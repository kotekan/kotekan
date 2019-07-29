/**
 * @file cudaCommand.h
 * @brief Base class for defining openCL commands to execute on GPUs
 *  - cudaCommand
 */

#ifndef CUDA_COMMAND_H
#define CUDA_COMMAND_H

#include "cuda_runtime_api.h"

#include "Config.hpp"
#include "assert.h"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaEventContainer.hpp"
#include "errors.h"
#include "factory.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "cudaUtils.hpp"

#include <signal.h>
#include <stdio.h>
#include <string>

#define CUDA_INPUT_STREAM 0
#define CUDA_COMPUTE_STREAM 1
#define CUDA_OUTPUT_STREAM 2


/**
 * @class cudaCommand
 * @brief Base class for defining openCL commands to execute on GPUs
 *
 */

class cudaCommand : public gpuCommand {
public:
    /** Kernel file name is optional.
     * @param device  The instance of the clDeviceInterface class that abstracts the interfacing
     *                      layer between the software and hardware.
     **/
    cudaCommand(kotekan::Config& config, const string& unique_name,
              kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
              const string& default_kernel_command = "",
              const string& default_kernel_file_name = "");
    /// Destructor that frees memory for the kernel and name.
    virtual ~cudaCommand();

    /** Execute a kernel, copy, etc.
     * @param gpu_frame_id  The bufferID associated with the GPU commands.
     * @param pre_event     The preceeding event in a sequence of chained event sequence of
     *commands.
     **/
    virtual cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) = 0;

    /** Releases the memory of the event chain arrays per buffer_id
     * @param gpu_frame_id    The bufferID to release all the memory references for.
     **/
    virtual void finalize_frame(int gpu_frame_id) override;

protected:
    cudaEvent_t* post_events; // tracked locally for cleanup
    cudaEvent_t* pre_events; // tracked locally for cleanup

    cudaDeviceInterface& device;
};

// Create a factory for cudaCommands
CREATE_FACTORY(cudaCommand, // const string &, const string &,
               kotekan::Config&, const string&, kotekan::bufferContainer&, cudaDeviceInterface&);
#define REGISTER_CUDA_COMMAND(newCommand)                                                            \
    REGISTER_NAMED_TYPE_WITH_FACTORY(cudaCommand, newCommand, #newCommand)

#endif // CUDA_COMMAND_H
