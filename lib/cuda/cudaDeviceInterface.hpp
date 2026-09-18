/**
 * @file
 * @brief Class to handle CUDA interactions with GPU hardware
 *  - cudaCommand
 */

#ifndef CUDA_DEVICE_INTERFACE_H
#define CUDA_DEVICE_INTERFACE_H

#include "Config.hpp"             // for Config
#include "driver_types.h"         // for cudaEvent_t, cudaStream_t
#include "gpuDeviceInterface.hpp" // for gpuDeviceInterface

#include <atomic>   // for atomic
#include <cuda.h>   // for CUfunction
#include <map>      // for map
#include <memory>   // for allocator, shared_ptr, weak_ptr
#include <mutex>    // for recursive_mutex
#include <stddef.h> // for size_t
#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string
#include <thread>   // for thread
#include <vector>   // for vector

/**
 * @class cudaDeviceInterface
 * @brief Class to handle CUDA interactions with GPU hardware.
 *
 * @par GPU Memory
 * @gpu_mem  bf_output       Output from the FRB pipeline, size 1024x128x16
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     chimeMetadata
 *
 * @author Keith Vanderlinde
 */
class cudaDeviceInterface final : public gpuDeviceInterface {
public:
    /**
     * @brief Get/create a cudaDeviceInterface for the given gpu_id.
     */
    static std::shared_ptr<cudaDeviceInterface> get(int32_t gpu_id, const std::string& name,
                                                    kotekan::Config& config);

    cudaDeviceInterface(kotekan::Config& config, const std::string& unique_name, int32_t gpu_id);
    ~cudaDeviceInterface();

    void prepareStreams(uint32_t num_streams);
    cudaStream_t getStream(int32_t cuda_stream_id);

    /// Returns the number of streams available
    int32_t get_num_streams();

    /// This function calls cudaSetDevice and must be called from every thread operating with this
    /// gpuDeviceInterface, or making calls directly to one of the cuda streams
    void set_thread_device() override;

    /**
     * @brief Asynchronous copies memory from the host (CPU RAM) to the device GPU (global memory)
     *
     * @param dst The GPU memory pointer
     * @param src The CPU memory pointer
     * @param len The amount of data to copy in bytes
     * @param cuda_stream_id The stream to run the copy on
     * @param pre_event The event before this one to wait on, if NULL will not wait
     * @param copy_start_event The profiling event at the start of this copy
     * @param copy_end_event The event at the end of the copy.
     */
    void async_copy_host_to_gpu(void* dst, void* src, size_t len, uint32_t cuda_stream_id,
                                cudaEvent_t pre_event, cudaEvent_t* copy_start_event,
                                cudaEvent_t* copy_end_event);

    /**
     * @brief Asynchronous Copies memory from the device GPU (global memory) to host (CPU RAM).
     *
     * @param dst The CPU memory pointer
     * @param src The GPU memory pointer
     * @param len The amount of data to copy in bytes
     * @param cuda_stream_id The stream to run the copy on
     * @param pre_event The event before this one to wait on, if NULL will not wait
     * @param copy_start_event The profiling event at the start of this copy
     * @param copy_end_event The event at the end of the copy.
     */
    void async_copy_gpu_to_host(void* dst, void* src, size_t len, uint32_t cuda_stream_id,
                                cudaEvent_t pre_event, cudaEvent_t* copy_start_event,
                                cudaEvent_t* copy_end_event);

    /**
     * @brief Builds a list of kernels from the file with name: @c kernel_file_name
     *
     * @param kernel_names Vector list of kernel names in the kernel file
     * @param opts         List of options to pass to nvrtc
     **/
    void build(const std::string& kernel_filename, const std::vector<std::string>& kernel_names,
               const std::vector<std::string>& opts);

    /**
     * @brief Builds a list of kernels from the PTX file with name: @c kernel_file_name
     *
     * Any @c --gpu-name option in @c opts is replaced with the compute capability of
     * the local GPU, so PTX generated for one GPU model runs on others as well.
     *
     * @param kernel_names       Vector list of kernel names in the kernel file
     * @param opts               List of options to pass to the PTX compiler
     * @param kernel_name_prefix Prefix to add to the kernel names in @c runtime_kernels
     **/
    void build_ptx(const std::string& kernel_filename, const std::vector<std::string>& kernel_names,
                   const std::vector<std::string>& opts,
                   const std::string& kernel_name_prefix = "");

    // Map containing the runtime kernels built with nvrtc from the kernel file (if needed)
    std::map<std::string, CUfunction> runtime_kernels;

    /// The most CUDA streams one device will ever have. Fixed so the mutex array below
    /// never reallocates: `prepareStreams` can be called from several stage constructors,
    /// and a reallocating container of live mutexes is not something to be clever about.
    static constexpr int32_t MAX_CUDA_STREAMS = 64;

    /// ⚠️ ONE MUTEX PER STREAM, NOT ONE PER DEVICE (2026-08-31). This used to be a single
    /// `gpu_command_mutex` covering the whole device, and that made every cudaProcess on a
    /// GPU serialize its command queuing against every other one -- across blocking CUDA
    /// driver calls. With 10 pipelines per GPU (7 GNSS chains + N2 + RFI + the copies) one
    /// thread stuck in `cuEventRecord` inside that lock stopped the entire GPU half, and the
    /// back-pressure propagated all the way to the NIC: the voltage ring filled, the
    /// cudaCopyToRingbuffer producer blocked, `host_voltage_buffer_N` pinned (peek_hold frees
    /// a frame only when EVERY consumer releases), transpose blocked, and the dpdk
    /// distributor dropped. Half the aperture went dark with nothing logging an error
    /// (proven by backtrace on cx19; see chord-gpu-command-mutex-wedge).
    ///
    /// Locking per STREAM instead is safe because intra-frame ordering never depended on
    /// exclusive stream access: each pipeline chains its own commands with explicit
    /// `cudaStreamWaitEvent` calls on events held in a vector LOCAL to its own
    /// `queue_commands` (see cudaSyncStream::execute). What the lock actually protects is
    /// shared `cudaCommandState` (cudaRechunkState is the documented case), which is shared
    /// only WITHIN one pipeline -- so a pipeline that locks the streams it uses keeps that
    /// invariant exactly.
    ///
    /// Pipelines with DISJOINT stream sets therefore never contend, which is what makes the
    /// wedge impossible rather than merely unlikely. Pipelines that do share streams (all
    /// the production processes still default to 0/1/2) still mutually exclude, byte-for-byte
    /// the old behaviour.
    ///
    /// ⚠️ ALWAYS LOCK IN ASCENDING STREAM ORDER -- that global order is the whole reason
    /// locking several of these cannot deadlock.
    std::recursive_mutex& stream_mutex(int32_t stream_id);

    // ------------------------------------------------------------------ the wedge probe
    //
    // Instrumentation for RE-EXCITING the half-node wedge on purpose, and for naming its
    // cause if it fires. The fault was diagnosed by backtrace to a thread inside
    // `cuEventRecord` holding the device-wide queuing lock while every other pipeline on the
    // GPU piled up behind it, but WHY that thread never returned was never established, so
    // the per-stream lock above is a blast-radius fix on an unexplained fault.
    //
    // ⚠️ A STUCK CALL CANNOT BE TIMED. If a thread never returns from the driver, code that
    // measures a call's duration after it returns prints nothing at all -- which is exactly
    // what happened the first time. So each pipeline PUBLISHES what it is about to do before
    // it does it, and a watchdog thread reads those records. That is what turns "the node is
    // dark and nothing logged" into a line naming the stage, command, stream and frame.

    /// The device-wide queuing lock this class replaced, kept so the fault can be re-excited
    /// with a config key rather than a rebuild. Selected by `gpu_command_lock: device`; see
    /// cudaProcess. Nothing takes it in the default `stream` mode.
    std::recursive_mutex gpu_command_mutex;

    /// What one pipeline is doing right now. Published BEFORE the call and cleared after, so
    /// a call that never returns leaves its record standing for the watchdog to find.
    struct InFlight {
        std::mutex m; ///< guards the fields below; never held across a CUDA call
        std::string stage;
        std::string command;
        int32_t stream = -1;
        int64_t frame = -1;
        int64_t entered_us = 0;  ///< steady clock, microseconds
        bool active = false;     ///< inside a command
        bool holds_lock = false; ///< past the queuing lock, i.e. blocking other pipelines
        bool waiting = false;    ///< blocked ON the queuing lock
        /// When this pipeline last FINISHED queuing a frame, and how many it has done. A
        /// pipeline that is not active and has not finished one recently is starved --
        /// blocked in wait_on_precondition, outside the lock and outside every record here.
        /// That is what the wedge's VICTIMS look like, and without it they stay silent.
        int64_t last_done_us = 0;
        int64_t frames_done = 0;
    };

    /// Register one record per cudaProcess. The device keeps it alive, so the watchdog can
    /// read it without racing a stage's destruction.
    std::shared_ptr<InFlight> register_in_flight();

    /// CUDA events created minus destroyed on this device. A monotonic rise means frames are
    /// not being finalized, which is the other way this fault could present.
    std::atomic<int64_t> events_outstanding{0};

    /// Start a thread that every `period_s` reports any pipeline stuck in one command for
    /// more than `stuck_s`, the busy/idle state of every stream, and events_outstanding.
    /// `cudaStreamQuery` is non-blocking and thread-safe, and it is the measurement that
    /// splits the hypothesis: all streams idle while a thread sits in the driver means the
    /// GPU has drained and the block is host-side. Idempotent; a period of 0 does nothing.
    void start_wedge_watchdog(double period_s, double stuck_s);

protected:
    void* alloc_gpu_memory(size_t len) override;
    void free_gpu_memory(void*) override;

    // Cuda Streams
    std::vector<cudaStream_t> streams;

    /// Per-stream command-queuing mutexes; see stream_mutex(). Fixed-size on purpose.
    std::recursive_mutex stream_mutexes[MAX_CUDA_STREAMS];

    /// Every registered pipeline record, and the watchdog reading them.
    std::mutex in_flight_mutex;
    std::vector<std::shared_ptr<InFlight>> in_flight;
    std::thread watchdog_thread;
    std::atomic<bool> watchdog_stop{false};
    double watchdog_period_s = 0.0;
    double watchdog_stuck_s = 0.0;
    void watchdog_loop();

    // Cache of device instances (weak to avoid lifetime extension)
    static std::map<int32_t, std::weak_ptr<cudaDeviceInterface>> inst_map;
};

#endif // CUDA_DEVICE_INTERFACE_H
