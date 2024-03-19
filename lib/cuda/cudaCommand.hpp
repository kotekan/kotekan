/**
 * @file
 * @brief Base class for defining CUDA commands to execute on GPUs
 *  - cudaCommand
 */

#ifndef CUDA_COMMAND_H
#define CUDA_COMMAND_H

#include "Config.hpp"
#include "assert.h"
#include "buffer.hpp"
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
#include <vector>

class cudaPipelineState : public kotekan::kotekanLogging {
public:
    cudaPipelineState(int _gpu_frame_id);
    virtual ~cudaPipelineState();
    void set_flag(const std::string&, bool val);
    bool flag_exists(const std::string&) const;
    bool flag_is_set(const std::string&) const;
    void set_int(const std::string&, int64_t val);
    int64_t get_int(const std::string&) const;

    int gpu_frame_id;

protected:
    std::map<std::string, bool> flags;
    std::map<std::string, int64_t> intmap;
};

class cudaCommandState : public gpuCommandState {
public:
    cudaCommandState(kotekan::Config&, const std::string&, kotekan::bufferContainer&,
                     cudaDeviceInterface&) {}
};

// use this to avoid having to write "std::shared_ptr<cudaCommandState>()"
extern std::shared_ptr<cudaCommandState> no_cuda_command_state;

/**
 * @class cudaCommand
 * @brief Base class for defining CUDA commands to execute on GPUs
 *
 * This is a base class for CUDA commands to run on NVidia hardware.
 * Kernels and other operations (I/O) should derive from this class,
 * which handles a lot of queueing and device interface issues.
 *
 * @conf cuda_stream  The ID of the CUDA stream to use for this command, defaults to one of
 *                    0, 1, 2, for command types COPY_IN, COPY_OUT, and KERNEL respectively.
 *                    This number must be less than @c num_cuda_streams set in cudaProcess.
 * @conf required_flag  A string flag name.  If set, the @c cudaPipelineState object will be
 *                    checked for this flag, and this command will only run if that flag is set.
 *
 * @author Keith Vanderlinde and Andre Renard
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
                int instance_num,
                std::shared_ptr<cudaCommandState> = std::shared_ptr<cudaCommandState>(),
                const std::string& default_kernel_command = "",
                const std::string& default_kernel_file_name = "");
    /// Destructor that frees memory for the kernel and name.
    virtual ~cudaCommand();

    /**
     * @brief Execute a kernel, with more control over the *cudaPipelineState* object.
     *        Most subclassers should implement *execute* instead.
     * @param pipestate  The pipeline state object.
     * @param pre_events Array of the last events from each cuda stream, indexed by stream
     *                   number.
     */
    virtual cudaEvent_t execute_base(cudaPipelineState& pipestate,
                                     const std::vector<cudaEvent_t>& pre_events);

    virtual bool should_execute(cudaPipelineState& pipestate,
                                const std::vector<cudaEvent_t>& pre_events);

    /**
     * @brief Execute a kernel, copy, etc.
     * @param pipestate     Pipeline state for this GPU frame.
     * @param pre_events    Array of the last events from each cuda stream, indexed by stream
     *                      number.
     **/
    virtual cudaEvent_t execute(cudaPipelineState& pipestate,
                                const std::vector<cudaEvent_t>& pre_events) = 0;

    /** Releases the memory of the event chain.
     **/
    virtual void finalize_frame() override;

    /**
     * Returns the id of the cuda stream used by this command object.
     */
    int32_t get_cuda_stream_id();

protected:
    void set_command_type(const gpuCommandType& type);

    // For subclassers to call to create & record a GPU starting event, IFF profiling is on.
    void record_start_event();

    // For subclassers to call to create & record a GPU ending event.
    cudaEvent_t record_end_event();

    /// Event queued after the kernel/copy for synchronization and profiling
    cudaEvent_t end_event;
    /// Extra event created at the start of kernels/copies for profiling
    cudaEvent_t start_event;

    cudaDeviceInterface& device;

    /// The ID of the cuda stream to run operations on
    int32_t cuda_stream_id;

    // cudaPipelineState flag required for this command to run, set from config "required_flag"
    std::string _required_flag;
};

/*
 * cudaCommand objects are responsible for handling a single frame of
 * GPU data as it makes its way through the pipeline.  Multiple frames
 * of data can be flowing through the GPU at a time, so @c
 * buffer_depth cudaCommand objects are created for each step in the
 * GPU pipeline.  These are called "instances" here, and the @c
 * instance_num is passed as an argument to the cudaCommand constructor.
 *
 * Some cudaCommand types will want to share state between the
 * instances.  This is done by using the @c
 * REGISTER_CUDA_COMMAND_WITH_STATE registration macro, passing in the
 * classes for the cudaCommand subclass and its cudaCommandState
 * subclass.  Before the instances are created, a single state object
 * will be created, and its @c shared_ptr will be passed to each
 * cudaCommand constructor.
 *
 * cudaCommand subclasses that don't need shared state can use the
 * plain old @c REGISTER_CUDA_COMMAND macro, and will not get passes a
 * state pointer.
 */

// Create a factory for cudaCommands
CREATE_FACTORY(cudaCommand, kotekan::Config&, const std::string&, kotekan::bufferContainer&,
               cudaDeviceInterface&, int);
// ... and another factory for cudaCommands that take a CommandState argument!
CREATE_FACTORY_VARIANT(state, cudaCommand, kotekan::Config&, const std::string&,
                       kotekan::bufferContainer&, cudaDeviceInterface&, int,
                       const std::shared_ptr<cudaCommandState>&);

// ... and a factory for cudaCommandStates
CREATE_FACTORY(cudaCommandState, kotekan::Config&, const std::string&, kotekan::bufferContainer&,
               cudaDeviceInterface&);

#define REGISTER_CUDA_COMMAND(newCommand)                                                          \
    REGISTER_NAMED_TYPE_WITH_FACTORY(cudaCommand, newCommand, #newCommand)

#define REGISTER_CUDA_COMMAND_WITH_STATE(newCommand, newCommandState)                              \
    REGISTER_NAMED_TYPE_WITH_FACTORY_VARIANT(state, cudaCommand, newCommand, #newCommand);         \
    REGISTER_NAMED_TYPE_WITH_FACTORY(cudaCommandState, newCommandState, #newCommand)

#endif // CUDA_COMMAND_H
