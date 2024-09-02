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
#include <vector>


class clCommandState : public gpuCommandState {
public:
    clCommandState(kotekan::Config&, const std::string&, kotekan::bufferContainer&,
                   clDeviceInterface&) {}
};

// use this to avoid having to write "std::shared_ptr<clCommandState>()"
extern std::shared_ptr<clCommandState> no_cl_command_state;

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
              kotekan::bufferContainer& host_buffers, clDeviceInterface& device, int instance_num,
              std::shared_ptr<gpuCommandState> = std::shared_ptr<gpuCommandState>(),
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
    virtual cl_event execute(cl_event pre_event) = 0;

    /** Releases the memory of the event chain arrays per buffer_id
     * @param gpu_frame_id    The bufferID to release all the memory references for.
     **/
    virtual void finalize_frame() override;

protected:

    /**
     * Retrieves an array of Buffer objects from a list of buffer names in the config.
     *
     * @param array_name The name of the array.
     * @param register_buffer Flag indicating whether to register the buffer against the command.
     * @param producer Flag indicating whether the command is a producer.
     * @return A vector of Buffer pointers.
     */
    std::vector<Buffer*> get_buffer_array(const std::string array_name, bool register_buffer = false, bool producer = false);

    /// Compiled instance of the kernel that will execute on the GPU once enqueued.
    cl_kernel kernel;
    /// Allocates resources on the GPU for the kernel.
    cl_program program;

    // Kernel values.
    /// global work space dimension
    size_t gws[3];
    /// local work space dimension
    size_t lws[3];
    cl_event post_event;

    clDeviceInterface& device;
};

// Create a factory for clCommands
CREATE_FACTORY(clCommand, // const std::string &, const std::string &,
               kotekan::Config&, const std::string&, kotekan::bufferContainer&, clDeviceInterface&,
               int);

// ... and another factory for clCommands that take a CommandState argument!
CREATE_FACTORY_VARIANT(state, clCommand, kotekan::Config&, const std::string&,
                       kotekan::bufferContainer&, clDeviceInterface&, int,
                       const std::shared_ptr<clCommandState>&);

// ... and a factory for clCommandStates
CREATE_FACTORY(clCommandState, kotekan::Config&, const std::string&, kotekan::bufferContainer&,
               clDeviceInterface&);

#define REGISTER_CL_COMMAND(newCommand)                                                            \
    REGISTER_NAMED_TYPE_WITH_FACTORY(clCommand, newCommand, #newCommand)

#define REGISTER_CL_COMMAND_WITH_STATE(newCommand, newCommandState)                                \
    REGISTER_NAMED_TYPE_WITH_FACTORY_VARIANT(state, clCommand, newCommand, #newCommand);           \
    REGISTER_NAMED_TYPE_WITH_FACTORY(clCommandState, newCommandState, #newCommand)

#endif // CL_COMMAND_H
