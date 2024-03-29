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
#include "kotekanTrackers.hpp" // for kotekanTrackers

#include "fmt.hpp"

#include <memory>
#include <stdint.h> // for int32_t
#include <string>   // for string, allocator

class gpuDeviceInterface;

/// Enumeration of known GPU command types.
enum class gpuCommandType { COPY_IN, BARRIER, KERNEL, COPY_OUT, NOT_SET };

/**
 * @class gpuCommandState
 * @brief Base class for shared state between peer gpuCommand objects.
 *
 * Each gpuCommand is responsible for processing one frame of data at
 * a time for one stage in a GPU pipeline.  Since multiple frames can
 * be active at once, multiple gpuCommand objects are created for each
 * each step in the GPU pipeline.  Sometimes, these peers need to
 * share state, and this base class gives them a mechanism to do that.
 */
class gpuCommandState : public kotekan::kotekanLogging {};

/**
 * @class gpuCommand
 * @brief Base class for defining commands to execute on GPUs
 *
 * Commands executed on a GPU can either be kernels that perform a simple calculation
 * or resource management instructions to support kernel execution, generally these
 * break down into three categories, copy-in, copy-out, and kernels.
 *
 * @conf buffer_depth          The number of GPU frames used for pipelining commands
 * @conf kernel                Filename, if an external file (CL, binary, etc) is used.
 * @conf kernel_path           If an external file is used, this gives the path to search.
 * @conf command               The name used by this kernel internally for logging,
 *                             and also the name of the kernel function (where that applies).
 * @conf profiling             Enable the recording of the command runtime
 * @conf frame_arrival_period  The time between frames, used for some profiling functions
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
     * @param instance_num  a single gpuCommand object is responsible for handling a single frame
     *                      of data at a time; there will be @c buffer_depth objects created for
     *                      each step in the GPU pipeline; this counter from zero says which of the
     *                      @c buffer_depth objects this one is.
     * @param shared_state  some gpuCommand types will require state that is shared between the
     *                      @c buffer_depth instances; this is a pointer to the shared state object
     *                      that has been created.
     * @param default_kernel_command (optional) function name / proper name
     *                               for a derived command
     * @param default_kernel_file_name  (optional) external file (e.g. CL) used by a command
     */
    gpuCommand(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& host_buffers, gpuDeviceInterface& device, int instance_num,
               std::shared_ptr<gpuCommandState> shared_state = std::shared_ptr<gpuCommandState>(),
               const std::string& default_kernel_command = "",
               const std::string& default_kernel_file_name = "");
    /// Destructor that frees memory for the kernel and name.
    virtual ~gpuCommand();
    /// Get that returns the name given to this gpuCommand object.
    std::string& get_name();
    /// Set the name (used for logging & profiling) for this gpuCommand object.
    void set_name(const std::string&);

    /**
     * @brief This function is called before a new GPU frame is going to be processed.
     * This is just a monotonic counter, starting from zero!
     */
    virtual void start_frame(int64_t gpu_frame_id);

    /**
     * @brief This function blocks on whatever resource is required by the command.
     * For example if this command requires a full buffer frame to copy
     * then it should block on that. It should also block on having any
     * free output buffers that might be referenced by this command.
     */
    virtual int wait_on_precondition();

    /**
     * @brief Runs some quick sanity checks before, should be called by
     *        derived GPU processes before running execution stages.
     */
    void pre_execute();

    /**
     * @brief Releases the memory of the event chain arrays per frame_id
     */
    virtual void finalize_frame();

    /// Get to return the results of profiling / timing.
    double get_last_gpu_execution_time();
    /// Get to distinguish the flavour of command (copy,kernel,etc)
    gpuCommandType get_command_type();

    /// Returns performance information, can be customized to give more detailed stats.
    virtual std::string get_performance_metric_string() {
        return fmt::format("Time: {:.3f} ms", get_last_gpu_execution_time() * 1e3);
    }

    /**
     * @brief Returns the unique name of the command object.
     * @return The command object unique name.
     */
    virtual std::string get_unique_name() const;

    /**
     * @brief For DOT / graphviz, return the list of GPU buffers
     * read/written by this command.
     * @return A list of GPU buffers touched by this command:
     *    [ (name, is_array, does_read, does_write) ]
     */
    virtual std::vector<std::tuple<std::string, bool, bool, bool>> get_gpu_buffers() const {
        return gpu_buffers_used;
    }

    /**
     * @brief For DOT / graphviz, return an extra string that will be inserted into the DOT
     * output after the GPU nodes and edges are drawn.
     */
    virtual std::string get_extra_dot(const std::string& prefix) const {
        (void)prefix;
        return "";
    }

    /// Track the time the command was active on the GPU.
    /// This is just the time the command is running, and doesn't include time waiting
    /// in the queue.
    std::shared_ptr<StatTracker> excute_time;

    /// Almost the same as excute_time, but divided by the frame arrival period
    std::shared_ptr<StatTracker> utilization;

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

    /// Instance number: [0, gpu_buffer_depth) for the command
    /// instances at this point in the GPU pipeline.
    int instance_num;

    /// State that is shared by instances of this command (at one point in the pipeline)
    std::shared_ptr<gpuCommandState> command_state;

    /// The counter for the GPU frame we are currently processing.
    int64_t gpu_frame_id;

    /// Sets the number of frames to be queued up in each buffer.
    int32_t _gpu_buffer_depth;

    /// Set to true if we have enabled profiling
    bool profiling;

    /// The expected time between new frames, used to compute utilization
    double frame_arrival_period;

    /// Type of command
    gpuCommandType command_type = gpuCommandType::NOT_SET;

    /// For get_gpu_buffers: a list of GPU buffers used by this command.
    std::vector<std::tuple<std::string, bool, bool, bool>> gpu_buffers_used;
};

#endif // GPU_COMMAND_H
