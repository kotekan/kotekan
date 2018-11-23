/**
 * @file gpuCommand.h
 * @brief Base class for defining GPU commands to execute on GPUs
 *  - gpuCommand
 */ 

#ifndef GPU_COMMAND_H
#define GPU_COMMAND_H

#include "Config.hpp"
#include "errors.h"
#include <stdio.h>
#include "gpuDeviceInterface.hpp"
#include "gpuEventContainer.hpp"
#include "factory.hpp"
#include "bufferContainer.hpp"
#include "kotekanLogging.hpp"
#include "assert.h"
#include "buffer.h"
#include <string>
#include <signal.h>


enum class gpuCommandType {COPY_IN, BARRIER, KERNEL, COPY_OUT, NOT_SET};
/**
 * @class gpuCommand
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

class gpuCommand: public kotekanLogging
{
public:
    /** Kernel file name is optional.
     * @param param_Device  The instance of the clDeviceInterface class that abstracts the interfacing
     *                      layer between the software and hardware.
     *                      In this method, it returns the current context of the device when
     *                      allocating resources for the kernel.
    **/
    gpuCommand(Config &config, const string &unique_name,
              bufferContainer &host_buffers, gpuDeviceInterface &device,
              const string &default_kernel_command="",
              const string &default_kernel_file_name=""
              );
    /// Destructor that frees memory for the kernel and name.
    virtual ~gpuCommand();
    /// gettor that returns the name given to this gpuCommand object.
    string &get_name();

    // This function blocks on whatever resource is required by this command
    // for example if this command requires a full buffer frame to copy
    // then it should block on that.  It should also block on having any
    // free output buffers that might be referenced by this command.
    virtual int wait_on_precondition(int gpu_frame_id);

    /** Basic functions to execute a gpu command are done in the
     * base class such as checking that the buffer_ID is positive and is
     * less than the number of frames in the buffer. 
     * @param gpu_frame_id  The bufferID associated with the GPU commands.
    **/
    void pre_execute(int gpu_frame_id);

    /** Releases the memory of the event chain arrays per buffer_id
     * @param gpu_frame_id    The bufferID to release all the memory references for.
    **/
    virtual void finalize_frame(int gpu_frame_id);

    double get_last_gpu_execution_time();
    gpuCommandType get_command_type();
protected:

    /// A unique name used for the gpu command. Used in indexing commands in a list and referencing them by this value.
    string kernel_command;
    /// File reference for the openCL file (.cl) where the kernel is written.
    string kernel_file_name;
    /// reference to the config file for the current run
    Config &config;

    /// Name to use with consumer and producer assignment for buffers defined in yaml files.
    string unique_name;
    bufferContainer host_buffers;
    gpuDeviceInterface &dev;

    /// Global buffer depth for all buffers in system. Sets the number of frames to be queued up in each buffer.
    int32_t _gpu_buffer_depth;

    // Profiling time for the last signal
    double last_gpu_execution_time = 0;
    gpuCommandType command_type = gpuCommandType::NOT_SET;
};

#endif // GPU_COMMAND_H

