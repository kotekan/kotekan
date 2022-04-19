/**
 * @file
 * @brief Base class for defining CUDA commands to execute on GPUs
 *  - cudaCommand
 */

#ifndef CUDA_COMMAND_H
#define CUDA_COMMAND_H

#include "Config.hpp"
#include "assert.h"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaEventContainer.hpp"
#include "cudaUtils.hpp"
#include "errors.h"
#include "factory.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"

#include <cuda.h>
#include <signal.h>
#include <stdio.h>
#include <string>

/**
 * @class cudaCommand
 * @brief Base class for defining CUDA commands to execute on GPUs
 *
 * This is a base class for CUDA commands to run on NVidia hardware.
 * Kernels and other opersations (I/O) should derive from this class,
 * which handles a lot of queueing and device interface issues.
 *
 * @author Keith Vanderlinde
 */
class cudaCommand : public gpuCommand {
public:
    /**
     * @brief Base constructor
     * @param config       The system config, passed by factory.
     * @param unique_name  The stage + command name.
     * @param host_buffers The list of bufferes handled by this GPU stage.
     * @param device       Abstracted GPU API interface for managing memory and common operations.
     * @param default_kernel_command   Name of the kernel for profiling read out.
     * @param default_kernel_file_name Kernel dile name, not really needed with cuda kernels.
     */
    cudaCommand(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                const std::string& default_kernel_command = "",
                const std::string& default_kernel_file_name = "");
    /// Destructor that frees memory for the kernel and name.
    virtual ~cudaCommand();

    /**
     * @brief Builds a list of kernels from the file with name: @c kernel_file_name
     *
     * @param kernel_names Vector list of kernel names in the kernel file
     * @param opts         List of options to pass to nvrtc
     **/
    virtual void build(const std::vector<std::string>& kernel_names,
                       std::vector<std::string>& opts);

    /**
     * @brief Execute a kernel, copy, etc.
     * @param gpu_frame_id  The bufferID associated with the GPU commands.
     * @param pre_event     The preceeding event in a sequence of chained event sequence of
     *                      commands.
     **/
    virtual cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) = 0;

    /** Releases the memory of the event chain arrays per buffer_id
     * @param gpu_frame_id    The bufferID to release all the memory references for.
     **/
    virtual void finalize_frame(int gpu_frame_id) override;

protected:
    cudaEvent_t* post_events; // tracked locally for cleanup
    cudaEvent_t* pre_events;  // tracked locally for cleanup

    cudaDeviceInterface& device;

    // Map containing the runtime kernels built with nvrtc from the kernel file (if needed)
    std::map<std::string, CUfunction> runtime_kernels;
};

// Create a factory for cudaCommands
CREATE_FACTORY(cudaCommand, // const std::string &, const std::string &,
               kotekan::Config&, const std::string&, kotekan::bufferContainer&,
               cudaDeviceInterface&);
#define REGISTER_CUDA_COMMAND(newCommand)                                                          \
    REGISTER_NAMED_TYPE_WITH_FACTORY(cudaCommand, newCommand, #newCommand)

#endif // CUDA_COMMAND_H
