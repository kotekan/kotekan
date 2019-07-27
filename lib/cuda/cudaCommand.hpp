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

    /** The build function creates the event to return as the post event in an event chaining
     * sequence. If a kernel is part of the cudaCommand object definition the resources to run it are
     * allocated on the gpu here.
     **/
    virtual void build();

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
    /// Compiled instance of the kernel that will execute on the GPU once enqueued.
    //cl_kernel kernel;
    /// Allocates resources on the GPU for the kernel.
    //cl_program program;

    // Kernel values.
    /// global work space dimension
    size_t gws[3];
    /// local work space dimension
    size_t lws[3];
    cudaEvent_t* post_events; // tracked locally for cleanup
    cudaEvent_t* pre_events; // tracked locally for cleanup

    cudaDeviceInterface& device;
};

// Create a factory for cudaCommands
CREATE_FACTORY(cudaCommand, // const string &, const string &,
               kotekan::Config&, const string&, kotekan::bufferContainer&, cudaDeviceInterface&);
#define REGISTER_CUDA_COMMAND(newCommand)                                                            \
    REGISTER_NAMED_TYPE_WITH_FACTORY(cudaCommand, newCommand, #newCommand)

#endif // CL_COMMAND_H
