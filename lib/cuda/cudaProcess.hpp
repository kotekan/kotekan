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

#include <cstdint> // for int32_t
#include <memory>  // for shared_ptr
#include <string>  // for string
#include <vector>  // for vector

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

    /// Fill `_my_stream_ids` from the constructed command list. Called once, after init().
    void collect_stream_ids();

    /// Per-frame-slot join events for multi-stream command chains (see queue_commands:
    /// frame completion must wait on every stream's last event, not just the last
    /// command's). Destroyed on slot reuse, and the survivors in ~cudaProcess.
    std::vector<cudaEvent_t> join_events;

    void register_host_memory(Buffer* host_buffer) override;

    // ---------------------------------------------------------------- the wedge probe
    //
    /// Take ONE device-wide lock instead of this pipeline's stream mutexes, the behaviour
    /// before per-stream locking. `gpu_command_lock: device`; diagnostic only, so the wedge
    /// can be re-excited without a rebuild.
    bool device_wide_lock = false;

    /// Warn when a single command takes longer than this to RETURN. The command that never
    /// returns is the watchdog's business, not this one's.
    double slow_command_warn_s = 0.5;

    /// This pipeline's record in the device's registry. Written before every command so a
    /// watchdog can name a thread that is still inside one.
    std::shared_ptr<cudaDeviceInterface::InFlight> in_flight;

    void probe_enter(const std::string& command, int32_t stream, int64_t frame, bool holds_lock,
                     bool waiting);
    void probe_clear();

    std::shared_ptr<cudaDeviceInterface> device;
};

#endif // CUDA_PROCESS_H
