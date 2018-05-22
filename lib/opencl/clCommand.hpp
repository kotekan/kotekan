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
#include "clCommandFactory.hpp"
#include "bufferContainer.hpp"
#include "kotekanLogging.hpp"
#include "assert.h"
#include "buffer.h"
#include <string>

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

class clCommand: public kotekanLogging
{
public:
    /** Kernel file name is optional.
     * @param param_Device  The instance of the clDeviceInterface class that abstracts the interfacing
     *                      layer between the software and hardware.
     *                      In this method, it returns the current context of the device when
     *                      allocating resources for the kernel.
    **/
    clCommand(const string &default_kernel_command, const string &default_kernel_file_name,
                Config &config, const string &unique_name,
                bufferContainer &host_buffers, clDeviceInterface &device);
    /// Destructor that frees memory for the kernel and name.
    virtual ~clCommand();
    /// gettor that returns the preceeding event in an event chain.
    cl_event getPreceedEvent();
    /// gettor that returns the next event in an event chain.
    cl_event getPostEvent();
    /// gettor that returns the name given to this clCommand object.
    string &get_name();
    /** The build function creates the event to return as the post event in an event chaining sequence.
     * If a kernel is part of the clCommand object definition the resources to run it are allocated on
     * the gpu here.
    **/
    virtual void build();

    // This function blocks on whatever resource is required by this command
    // for example if this command requires a full buffer frame to copy
    // then it should block on that.  It should also block on having any
    // free output buffers that might be referenced by this command.
    virtual int wait_on_precondition(int gpu_frame_id);

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
    virtual cl_event execute(int gpu_frame_id, const uint64_t& fpga_seq, cl_event pre_event);
    /** Releases the memory of the event chain arrays per buffer_id
     * @param gpu_frame_id    The bufferID to release all the memory references for.
    **/
    virtual void finalize_frame(int gpu_frame_id);
    /// Reads all the relevant config values out of the config file references into the protected scope variables of the class.
    virtual void apply_config(const uint64_t &fpga_seq);
protected:
    /// Compiled instance of the kernel that will execute on the GPU once enqueued.
    cl_kernel kernel;
    /// Allocates resources on the GPU for the kernel.
    cl_program program;

    // Kernel values.
    /// global work space dimension
    size_t gws[3]; // TODO Rename to something more meaningful - or comment.
    /// local work space dimension
    size_t lws[3];

    // Kernel Events
    /// The next event in an event chain when building an event chain of commands.
    cl_event *post_event;
    /// A unique name used for the gpu command. Used in indexing commands in a list and referencing them by this value.
    string kernel_command;
    /// File reference for the openCL file (.cl) where the kernel is written.
    string kernel_file_name;
    /// reference to the config file for the current run
    Config &config;
    /// Name to use with consumer and producer assignment for buffers defined in yaml files.
    string unique_name;
    bufferContainer host_buffers;
    clDeviceInterface &device;

    /// Global buffer depth for all buffers in system. Sets the number of frames to be queued up in each buffer.
    int32_t _buffer_depth;
    int32_t _gpu_buffer_depth;
};

#endif // CL_COMMAND_H

