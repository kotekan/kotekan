/**
 * @file clCommand.h
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
#include "errors.h"
#include <stdio.h>
#include "clDeviceInterface.hpp"
#include "clEventContainer.hpp"
#include "factory.hpp"
#include "bufferContainer.hpp"
#include "kotekanLogging.hpp"
#include "assert.h"
#include "buffer.h"
#include <string>
#include <signal.h>
#include "gpuCommand.hpp"


/**
 * @class clCommand
 * @brief Base class for defining openCL commands to execute on GPUs
 *
 * Commands executed on a GPU can either be kernels that perform a simple calculation
 * or resource management instructions to support kernel execution. Any openCL instruction
 * sent to a GPU requires a set of common support functions. The code common to all openCL classes has
 * been abstracted into this base class along with the signature of fundamental methods
 * a child class should implement when defining an openCL class.
 * 
 * This object and its subclasses is managed by a command_factory instance that initializes
 * memory resources and determines execution sequence.
 * 
 * 
 * @conf num_adjusted_elements     Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
 * @conf num_elements              Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
 * @conf num_local_freq            Number of frequencies per data stream sent to each node.
 * @conf samples_per_data_set      Total samples in each dataset. Must be a value that is a power of 2.
 * @conf num_data_sets             Number of independent integrations within a single dataset. (eg. 8 means samples_per_data_set/8= amount of integration per dataset.)
 * @conf num_adjusted_local_freq   Number of frequencies per data stream sent to each node.
 * @conf num_blocks                Calculated value: num_adjusted_elements/block_size * (num_adjusted_elements/block_size + 1)/2
 * @conf block_size                This is a kernel tuning parameter for a global work space dimension that sets data sizes for GPU work items.
 * @conf buffer_depth              Global buffer depth for all buffers in system. Sets the number of frames to be queued up in each buffer.
 * 
 * 
 * @todo    Clean up redundancy in config values used.
 * @todo    Add dynamic memory allocation logic.
 * @todo    Move some of the correlator specific config values into the correlator kernels.
 * @todo    Rename variables to frame where buffer is current used. In some cases, frame is the correct usage.
 * @todo    BufferID used when FrameID is more appropriate. Change this naming.
 *
 * @author Ian Tretyakov
 *
 */

class clCommand: public gpuCommand
{
public:
    /** Kernel file name is optional.
     * @param param_Device  The instance of the clDeviceInterface class that abstracts the interfacing
     *                      layer between the software and hardware.
     *                      In this method, it returns the current context of the device when
     *                      allocating resources for the kernel.
    **/
    clCommand(Config &config, const string &unique_name,
              bufferContainer &host_buffers, clDeviceInterface &device,
              const string &default_kernel_command="",
              const string &default_kernel_file_name=""
              );
    /// Destructor that frees memory for the kernel and name.
    virtual ~clCommand();

    /** The build function creates the event to return as the post event in an event chaining sequence.
     * If a kernel is part of the clCommand object definition the resources to run it are allocated on
     * the gpu here.
    **/
    virtual void build();

    /// This method appends arguements to the kernel execution statement that's run when the kernel is enqueued on the GPU.
    void setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer);

    /** The execute command does very little in the base case. The child class must provide an implementation of the 
     * logic under the signature of the method defined here. Basic functions to execute a gpu command are done in the
     * base class such as checking that the buffer_ID is positive and is less than the number of frames in the buffer. 
     * @param gpu_frame_id  The bufferID associated with the GPU commands.
     * 
     * @param fpga_seq      Passed to apply_config.
     * 
     * @param pre_event     The preceeding event in a sequence of chained event sequence of commands.
    **/
    virtual cl_event execute(int gpu_frame_id, const uint64_t& fpga_seq, cl_event pre_event) = 0;

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
    cl_event *post_events; //tracked locally for cleanup

    clDeviceInterface &device;
};

// Create a factory for clCommands
CREATE_FACTORY(clCommand, //const string &, const string &,
                Config &, const string &,
                bufferContainer &, clDeviceInterface &);
#define REGISTER_CL_COMMAND(newCommand) REGISTER_NAMED_TYPE_WITH_FACTORY(clCommand, newCommand, #newCommand)

#endif // CL_COMMAND_H

