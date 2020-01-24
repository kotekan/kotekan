/**
 * @file gpuCommand.hpp
 * @brief Base class for defining commands to execute on GPUs
 *  - gpuCommand
 */

#ifndef GPU_COMMAND_H
#define GPU_COMMAND_H

#include "Config.hpp"          // for Config
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for kotekanLogging

#include <stdint.h> // for int32_t
#include <string>   // for string, allocator

class gpuDeviceInterface;

/// Enumeration of known GPU command types.
enum class gpuCommandType { COPY_IN, BARRIER, KERNEL, COPY_OUT, NOT_SET };

/**
 * @class gpuCommand
 * @brief Base class for defining commands to execute on GPUs
 *
 * Commands executed on a GPU can either be kernels that perform a simple calculation
 * or resource management instructions to support kernel execution, generally these
 * break down into three categories, copy-in, copy-out, and kernels.
 *
 * @conf buffer_depth  Global buffer depth for all buffer arrays in the subsystem.
 *                     Generally sets the number of frames used for staging input / output.
 * @conf kernel        If an external file (CL, binary, etc) is used, this gives the filename.
 * @conf kernel_path   If an external file is used, this gives the path to search.
 * @conf command       The name used bby this kernel internally for logging, and also the name
 *                     name of the kernel function (where) that applies).
 *
 * @author Keith Vanderlinde
 */
class gpuCommand : public kotekan::kotekanLogging {
public:
    /**
     * @brief Constructor, needs to be initialized by any derived classes.
     * @param config        kotekan config object
     * @param unique_name   kotekan unique name
     * @param host_buffers  kotekan host-side buffers
     * @param device        instance of a derived GPU device interface.
     * @param default_kernel_command (optional) function name / proper name
     *                               for a derived command
     * @param default_kernel_file_name  (optional) external file (e.g. CL) used by a command
     */
    gpuCommand(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& host_buffers, gpuDeviceInterface& device,
               const std::string& default_kernel_command = "",
               const std::string& default_kernel_file_name = "");
    /// Destructor that frees memory for the kernel and name.
    virtual ~gpuCommand();
    /// Get that returns the name given to this gpuCommand object.
    std::string& get_name();

    /**
     * @brief This function blocks on whatever resource is required by the command.
     * For example if this command requires a full buffer frame to copy
     * then it should block on that. It should also block on having any
     * free output buffers that might be referenced by this command.
     * @param gpu_frame_id  index of the frame, used to check I/O buffer status.
     */
    virtual int wait_on_precondition(int gpu_frame_id);

    /**
     * @brief Runs some quick sanity checks before, should be called by
     *        derived GPU processes before running execution stages.
     * @param gpu_frame_id  The bufferID associated with the GPU commands.
     */
    void pre_execute(int gpu_frame_id);

    /**
     * @brief Releases the memory of the event chain arrays per frame_id
     * @param gpu_frame_id    The frame id to release all the memory references for.
     */
    virtual void finalize_frame(int gpu_frame_id);

    /// Get to return the results of profiling / timing.
    double get_last_gpu_execution_time();
    /// Get to distinguish the flavour of command (copy,kernel,etc)
    gpuCommandType get_command_type();

protected:
    /// A unique name used for the gpu command. Used in indexing commands in a list and referencing
    /// them by this value.
    std::string kernel_command;
    /// File reference for the openCL file (.cl) where the kernel is written.
    std::string kernel_file_name;
    /// reference to the config file for the current run
    kotekan::Config& config;

    /// Name to use with consumer and producer assignment for buffers defined in yaml files.
    std::string unique_name;
    kotekan::bufferContainer host_buffers;

    /// Reference to a derived device interface.
    gpuDeviceInterface& dev;

    /// Sets the number of frames to be queued up in each buffer.
    int32_t _gpu_buffer_depth;

    /// Profiling time for the last signal
    double last_gpu_execution_time = 0;

    /// Type of command
    gpuCommandType command_type = gpuCommandType::NOT_SET;
};

#endif // GPU_COMMAND_H
