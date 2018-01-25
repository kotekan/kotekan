/**
 * @class gpu_command
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
 * @conf _num_adjusted_elements     Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
 * @conf _num_elements              Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
 * @conf _num_local_freq            Number of frequencies per data stream sent to each node.
 * @conf _samples_per_data_set      Total samples in each dataset. Must be a value that is a power of 2.
 * @conf _num_data_sets             Number of independent integrations within a single dataset. (eg. 8 means samples_per_data_set/8= amount of integration per dataset.)
 * @conf _num_adjusted_local_freq   Number of frequencies per data stream sent to each node.
 * @conf _num_blocks                Calculated value: num_adjusted_elements/block_size * (num_adjusted_elements/block_size + 1)/2
 * @conf _block_size                This is a kernel tuning parameter for a global work space dimension that sets data sizes for GPU work items.
 * @conf _buffer_depth              Global buffer depth for all buffers in system. Sets the number of frames to be queued up in each buffer.
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

#ifndef GPU_COMMAND_H
#define GPU_COMMAND_H

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

#include "Config.hpp"
#include "errors.h"
#include <stdio.h>
#include "device_interface.h"
#include "assert.h"
#include "buffer.h"
#include <string>

class gpu_command
{
public:
    /// Constructor when no kernel is defined for the gpu_command object.
    gpu_command(const char* param_name, Config &param_config, const string &unique_name_);
    /// Overloaded constructor that sets a given openCL kernel file (.cl) to execute for this command object.
    gpu_command(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name);//, cl_device_id *param_DeviceID, cl_context param_Context);
    /// Destructor that frees memory for the kernel and name.
    virtual ~gpu_command();
    /// gettor that returns the preceeding event in an event chain.
    cl_event getPreceedEvent();
    /// gettor that returns the next event in an event chain.
    cl_event getPostEvent();
    /// gettor that returns the name given to this gpu_command object.
    char* get_name();
    /// gettor that returns the config values for a kernel formatted as a cl_options string to append to the kernel execution statement.
    string get_cl_options();
    /** The build function creates the event to return as the post event in an event chaining sequence.
     * If a kernel is part of the gpu_command object definition the resources to run it are allocated on
     * the gpu here.
     * @param param_Device  The instance of the device_interface class that abstracts the interfacing
     *                      layer between the software and hardware.
     *                      In this method, it returns the current context of the device when
     *                      allocating resources for the kernel.
    **/
    virtual void build(device_interface& param_Device);
    /// This method appends arguements to the kernel execution statement that's run when the kernel is enqueued on the GPU.
    void setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer);
    /** The execute command does very little in the base case. The child class must provide an implementation of the 
     * logic under the signature of the method defined here. Basic functions to execute a gpu command are done in the
     * base class such as checking that the buffer_ID is positive and is less than the number of frames in the buffer. 
    **/
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, device_interface& param_Device, cl_event param_PrecedeEvent);
    /// Releases the memory of the event chain arrays per buffer_id
    virtual void cleanMe(int param_BufferID);
    /// Releases the memory of GPU resource allocation, events, and compiled GPU kernel.
    virtual void freeMe();
    /// Reads all the relevant config values out of the config file references into the protected scope variables of the class.
    virtual void apply_config(const uint64_t &fpga_seq);
protected:
    /// Compiled instance of the kernel that will execute on the GPU once enqueued.
    cl_kernel kernel;
    /// Allocates resources on the GPU for the kernel.
    cl_program program;
    
    /// reference to the config file for the current run
    Config &config;

    // Kernel values.
    /// global work space dimension
    size_t gws[3]; // TODO Rename to something more meaningful - or comment.
    /// local work space dimension
    size_t lws[3];

    // Kernel Events
    /// The preceding event when building an event chain of commands.
    cl_event * precedeEvent;
    /// The next event in an event chain when building an event chain of commands.
    cl_event * postEvent;
    /// Default state to non-kernel executing command. 1 means kernel is defined with this command.
    int gpuCommandState; 
    /// File reference for the openCL file (.cl) where the kernel is written.
    char * gpuKernel;
    /// A unique name used for the gpu command. Used in indexing commands in a list and referencing them by this value.
    char* name;

    // Common configuration values (which do not change in a run)
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_adjusted_elements;
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    /// Number of independent integrations within a single dataset. (eg. 8 means samples_per_data_set/8= amount of integration per dataset.)
    int32_t _num_data_sets;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_adjusted_local_freq;
    /// Calculated value: num_adjusted_elements/block_size * (num_adjusted_elements/block_size + 1)/2
    int32_t _num_blocks;
    /// This is a kernel tuning parameter for a global work space dimension that sets data sizes for GPU work items.
    int32_t _block_size;
    /// Global buffer depth for all buffers in system. Sets the number of frames to be queued up in each buffer.
    int32_t _buffer_depth;
    /// Name to use with consumer and producer assignment for buffers defined in yaml files.
    string unique_name;
};

#endif // GPU_COMMAND_H

