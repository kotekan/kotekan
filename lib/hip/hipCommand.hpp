/**
 * @file
 * @brief Base class for defining HIP commands to execute on GPUs
 *  - hipCommand
 */

#ifndef HIP_COMMAND_H
#define HIP_COMMAND_H

#include "Config.hpp"
#include "assert.h"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "errors.h"
#include "factory.hpp"
#include "gpuCommand.hpp"
#include "hip/hip_runtime_api.h"
#include "hipDeviceInterface.hpp"
#include "hipEventContainer.hpp"
#include "hipUtils.hpp"
#include "kotekanLogging.hpp"

#include <signal.h>
#include <stdio.h>
#include <string>

/**
 * @class hipCommand
 * @brief Base class for defining HIP commands to execute on GPUs
 *
 * This is a base class for HIP commands to run on NVidia hardware.
 * Kernels and other opersations (I/O) should derive from this class,
 * which handles a lot of queueing and device interface issues.
 *
 * @todo Clean up and refactor GPU code to not require external kernels.
 *
 * @author Andre Renard
 */
class hipCommand : public gpuCommand {
public:
    /**
     * @brief Base constructor
     * @param config       The system config, passed by factory.
     * @param unique_name  The stage + command name.
     * @param host_buffers The list of bufferes handled by this GPU stage.
     * @param device       Abstracted GPU API interface for managing memory and common operations.
     * @param default_kernel_command   Name of the kernel for profiling read out.
     * @param default_kernel_file_name Kernel dile name, not really needed with hip kernels.
     */
    hipCommand(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& host_buffers, hipDeviceInterface& device,
               const std::string& default_kernel_command = "",
               const std::string& default_kernel_file_name = "");
    /// Destructor that frees memory for the kernel and name.
    virtual ~hipCommand();

    /** Execute a kernel, copy, etc.
     * @param gpu_frame_id  The bufferID associated with the GPU commands.
     * @param pre_event     The preceeding event in a sequence of chained event sequence of
     *commands.
     **/
    virtual hipEvent_t execute(int gpu_frame_id, hipEvent_t pre_event) = 0;

    /** Releases the memory of the event chain arrays per buffer_id
     * @param gpu_frame_id    The bufferID to release all the memory references for.
     **/
    virtual void finalize_frame(int gpu_frame_id) override;

protected:
    hipEvent_t* post_events; // tracked locally for cleanup
    hipEvent_t* pre_events;  // tracked locally for cleanup

    hipDeviceInterface& device;
};

// Create a factory for hipCommands
CREATE_FACTORY(hipCommand, // const string &, const string &,
               kotekan::Config&, const std::string&, kotekan::bufferContainer&,
               hipDeviceInterface&);
#define REGISTER_HIP_COMMAND(newCommand)                                                           \
    REGISTER_NAMED_TYPE_WITH_FACTORY(hipCommand, newCommand, #newCommand)

#endif // HIP_COMMAND_H
