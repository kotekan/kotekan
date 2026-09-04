/**
 * @file
 * @brief Stage for running a set of CUDA commands
 *  - cudaProcess : public gpuProcess
 */

#ifndef CUDA_PROCESS_H
#define CUDA_PROCESS_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

#include "Config.hpp"              // for Config
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "gpuCommand.hpp"          // for gpuCommand
#include "gpuEventContainer.hpp"   // for gpuEventContainer
#include "gpuProcess.hpp"          // for gpuProcess

#include <memory> // for shared_ptr
#include <string> // for string
#include <cstdint>  // for int32_t
#include <vector> // for vector

/**
 * @class cudaProcess
 * @brief Stage to manage all the kernels and copy commands for a GPU
 *
 * This stage is responsible for running the cudaCommandObjects which in turn run the
 * various host<->device copies and kernel calls.  Much of the logic exists in the base
 * class @c gpuProcess, so that class for more details.
 *
 * @conf num_cuda_streams The number of CUDA streams to setup, the default is 3 for one
 *                        host->device, one device->host, and one kernel stream.
 *                        Can be set higher if more than one stream is need for each type
 *                        of operation.  See @c cudaCommand and @c cudaSyncStream for more details.
 *
 * @author Keith Vanderlinde and Andre Renard
 */
class cudaProcess final : public gpuProcess {
public:
    cudaProcess(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~cudaProcess();

    std::vector<gpuCommand*> create_command(const std::string& cmd_name,
                                            const std::string& unique_name) override;
    gpuEventContainer* create_signal() override;
    void queue_commands(int gpu_frame_counter) override;

private:
    /// The CUDA streams this pipeline's commands enqueue onto, ASCENDING and unique. Only
    /// these streams' mutexes are taken while queuing a frame, so a pipeline with private
    /// streams never blocks another one (see cudaDeviceInterface::stream_mutex).
    std::vector<std::int32_t> _my_stream_ids;

    /// Which of our streams carries the end-of-frame join. Must be one WE own: the join makes
    /// it wait on every other stream's last event, and pointing that at a stream another
    /// pipeline enqueues onto puts that pipeline's work behind our events -- a false
    /// dependency that becomes a hang if our event never completes. It was hardcoded to
    /// stream 0, which every pipeline shares by default.
    std::int32_t _join_stream_id = 0;

    /// Fill the two above from the constructed command list. Called once, after init().
    void collect_stream_ids();

    /// per-frame-slot join events for multi-stream command chains (see queue_commands:
    /// frame completion must wait on every stream's last event, not just the last
    /// command's). Destroyed lazily on slot reuse.
    std::vector<cudaEvent_t> join_events;

    void register_host_memory(Buffer* host_buffer) override;

    std::shared_ptr<cudaDeviceInterface> device;
};

#endif // CUDA_PROCESS_H
