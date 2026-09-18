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
 *                        Every command in this stage must resolve to a stream below this value,
 *                        so a stage with a kernel at @c cuda_stream_base b needs at least 3*b+3.
 * @conf cuda_stream_base Int, default 0. A pipeline INDEX: the default triple of every command
 *                        in this stage is streams 3*base+0 (copy-in), 3*base+1 (copy-out),
 *                        3*base+2 (kernel); see @c cudaCommand. Distinct bases give disjoint
 *                        triples, so two stages on one GPU share either all three streams or
 *                        none, and take either the same queuing mutexes or none in common.
 *
 *                        \warning Two stages on one GPU that chain kernels through a block of
 *                        GPU memory named the same in both -- @c gpuDeviceInterface::get_gpu_memory
 *                        keys those by name across the whole device -- must share a base. The
 *                        queuing lock is per stream, so pipelines on disjoint streams queue
 *                        concurrently and one can overwrite the block between the other's two
 *                        commands. Memory passed between stages through a ring buffer is not
 *                        affected: the host ring is signalled only once the producing stage's
 *                        final event has completed.
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

    void register_host_memory(Buffer* host_buffer) override;

    std::shared_ptr<cudaDeviceInterface> device;

private:
    /// The CUDA streams this pipeline's commands enqueue onto, ascending and unique. Only these
    /// streams' mutexes are taken while queuing a frame, so a pipeline with private streams
    /// never blocks another one (see cudaDeviceInterface::stream_mutex).
    std::vector<std::int32_t> _my_stream_ids;

    /// Fill `_my_stream_ids` from the constructed command list, refusing a command whose stream
    /// is not below `num_cuda_streams` and a stage with no commands that enqueue. Called once,
    /// after init().
    void collect_stream_ids(uint32_t num_cuda_streams);
};

#endif // CUDA_PROCESS_H
