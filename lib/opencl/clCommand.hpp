/**
 * @file clCommand.hpp
 * @brief Base class for defining openCL commands to execute on GPUs
 *  - clCommand
 */

#ifndef CL_COMMAND_H
#define CL_COMMAND_H

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include "Config.hpp"
#include "assert.h"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "clDeviceInterface.hpp"
#include "clEventContainer.hpp"
#include "errors.h"
#include "factory.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"

#include <signal.h>
#include <stdio.h>
#include <string>


/**
 * @class clCommand
 * @brief Base class for defining openCL commands to execute on GPUs
 *
 */

class clCommand : public gpuCommand {
public:
    /** Kernel file name is optional.
     * @param config                    kotekan config object
     * @param unique_name               kotekan unique name
     * @param host_buffers              kotekan host-side buffers
     * @param device                    The instance of the clDeviceInterface class that abstracts
     *                                  the interfacing layer between the software and hardware.
     * @param default_kernel_command    (optional) function name / proper name for a derived command
     * @param default_kernel_file_name  (optional) external file (e.g. CL) used by a command
     **/
    clCommand(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& host_buffers, clDeviceInterface& device,
              const std::string& default_kernel_command = "",
              const std::string& default_kernel_file_name = "");
    /// Destructor that frees memory for the kernel and name.
    virtual ~clCommand();

    /** The build function creates the event to return as the post event in an event chaining
     * sequence. If a kernel is part of the clCommand object definition the resources to run it are
     * allocated on the gpu here.
     **/
    virtual void build();

    /// This method appends arguements to the kernel execution statement that's run when the kernel
    /// is enqueued on the GPU.
    void setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer);

    /** Execute a kernel, copy, etc.
     * @param gpu_frame_id  The bufferID associated with the GPU commands.
     * @param pre_event     The preceeding event in a sequence of chained event sequence of
     *commands.
     **/
    virtual cl_event execute(int gpu_frame_id, cl_event pre_event) = 0;

    /** Releases the memory of the event chain arrays per buffer_id
     * @param gpu_frame_id    The bufferID to release all the memory references for.
     **/
    virtual void finalize_frame(int gpu_frame_id) override;

protected:
    /// Compiled instance of the kernel that will execute on the GPU once enqueued.
    cl_kernel kernel;
    /// Allocates resources on the GPU for the kernel.
    cl_program program;

    // Kernel values.
    /// global work space dimension
    size_t gws[3];
    /// local work space dimension
    size_t lws[3];
    cl_event* post_events; // tracked locally for cleanup

    clDeviceInterface& device;
};

// Create a factory for clCommands
CREATE_FACTORY(clCommand, // const std::string &, const std::string &,
               kotekan::Config&, const std::string&, kotekan::bufferContainer&, clDeviceInterface&);
#define REGISTER_CL_COMMAND(newCommand)                                                            \
    REGISTER_NAMED_TYPE_WITH_FACTORY(clCommand, newCommand, #newCommand)

#endif // CL_COMMAND_H
