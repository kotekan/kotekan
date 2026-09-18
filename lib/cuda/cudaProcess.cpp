#include "cudaProcess.hpp"

#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE
#include "cudaCommand.hpp"        // for cudaCommand, _factory_aliascudaCommandState, _factory_...
#include "cudaEventContainer.hpp" // for cudaEventContainer
#include "cudaFrameJoin.hpp"      // for cudaFrameJoin, plan_cuda_frame_join
#include "cudaUtils.hpp"          // for CHECK_CUDA_ERROR
#include "cuda_profiler_api.h"    // for cudaProfilerStart, cudaProfilerStop
#include "cuda_runtime_api.h"     // for cudaHostRegister
#include "driver_types.h"         // for CUevent_st, cudaEvent_t, cudaHostRegisterDefault
#include "factory.hpp"            // for FACTORY, FACTORY_VARIANT
#include "kotekanLogging.hpp"     // for DEBUG, DEBUG2, WARN

#include "fmt.hpp" // for compile_string_to_view

#include <chrono>    // for steady_clock
#include <mutex>     // for recursive_mutex, unique_lock
#include <set>       // for set (stream-id dedup in collect_stream_ids)
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t, int32_t

using kotekan::bufferContainer;
using kotekan::Config;

using namespace std;

REGISTER_KOTEKAN_STAGE(cudaProcess);

// TODO Remove the GPU_ID from this constructor
cudaProcess::cudaProcess(Config& config_, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    gpuProcess(config_, unique_name, buffer_container) {
    std::string device_name =
        config_.get_default<std::string>(unique_name, "device", "device_" + std::to_string(gpu_id));
    device = cudaDeviceInterface::get(gpu_id, device_name, config_);
    dev = device.get();
    // Tell the Cuda runtime to associate this gpu_id with this thread/Stage.
    device->set_thread_device();

    uint32_t num_streams = config.get_default<uint32_t>(unique_name, "num_cuda_streams", 3);

    device->prepareStreams(num_streams);

    // ---- the wedge probe. `gpu_command_lock: device` restores the older whole-device lock, one
    // per GPU across every pipeline's whole command loop, so the half-node wedge can be
    // re-excited deliberately. The stream-mutex path is the default and the shipping
    // behaviour; this exists so the experiment needs a config key, not a rebuild.
    const std::string lock_mode =
        config.get_default<std::string>(unique_name, "gpu_command_lock", "stream");
    if (lock_mode == "device")
        device_wide_lock = true;
    else if (lock_mode != "stream")
        throw std::runtime_error(
            fmt::format("{:s}: gpu_command_lock must be \"stream\" or \"device\", not \"{:s}\"",
                        unique_name, lock_mode));
    slow_command_warn_s = config.get_default<double>(unique_name, "slow_command_warn_s", 0.5);
    in_flight = device->register_in_flight();
    device->start_wedge_watchdog(
        config.get_default<double>(unique_name, "wedge_watchdog_s", 0.0),
        config.get_default<double>(unique_name, "wedge_watchdog_stuck_s", 5.0));
    if (device_wide_lock)
        WARN("{:s}: gpu_command_lock=device -- this pipeline serialises its command queuing "
             "against EVERY other pipeline on GPU {:d}. Diagnostic only.",
             unique_name, gpu_id);
    CHECK_CUDA_ERROR(cudaProfilerStart());
    init();
    collect_stream_ids();
}

// Which streams do our commands actually enqueue onto? Ascending and unique, so every
// cudaProcess takes these mutexes in the same global order and locking several cannot
// deadlock. (The end-of-frame join uses the FINAL COMMAND's stream, not one of these by
// position -- see queue_commands.)
void cudaProcess::collect_stream_ids() {
    std::set<std::int32_t> ids;
    for (auto& command : commands)
        for (auto* c : command)
            if (c != nullptr) {
                const std::int32_t sid = ((cudaCommand*)c)->get_cuda_stream_id();
                if (sid >= 0)
                    ids.insert(sid);
            }
    // A pipeline with no commands still needs a join stream; fall back to 0.
    if (ids.empty())
        ids.insert(0);
    _my_stream_ids.assign(ids.begin(), ids.end());
    INFO("cudaProcess[{:s}]: queuing streams [{:s}]", unique_name,
         fmt::format("{}", fmt::join(_my_stream_ids, ", ")));
}

cudaProcess::~cudaProcess() {
    // Ensure this thread is set to the correct device before final CUDA cleanup.
    if (device)
        device->set_thread_device();
    // Destroy the per-slot join events. They are otherwise only destroyed on slot REUSE,
    // which never happens for the last frame in each slot, so teardown would leak one event
    // per buffer-depth slot per pipeline. Safe here: gpuProcess has already joined its
    // threads, and every one of these was synchronized by the results thread before the slot
    // could be reused.
    for (cudaEvent_t& e : join_events)
        if (e != nullptr) {
            CHECK_CUDA_ERROR(cudaEventDestroy(e));
            e = nullptr;
        }
    CHECK_CUDA_ERROR(cudaProfilerStop());
    // With weak_ptr cache, simply letting the last shared_ptr go triggers cleanup.
}

gpuEventContainer* cudaProcess::create_signal() {
    return new cudaEventContainer();
}

std::vector<gpuCommand*> cudaProcess::create_command(const std::string& cmd_name,
                                                     const std::string& unique_name) {
    std::vector<gpuCommand*> cmds;
    // Create the cudaCommandState object, if used, for this command class.
    std::shared_ptr<cudaCommandState> st;
    if (FACTORY(cudaCommandState)::exists(cmd_name))
        st = FACTORY(cudaCommandState)::create_shared(cmd_name, config, unique_name,
                                                      local_buffer_container, *device);
    for (uint32_t i = 0; i < _gpu_buffer_depth; i++) {
        gpuCommand* cmd;
        if (st)
            // Create the cudaCommand object (with state arg)
            cmd = FACTORY_VARIANT(state, cudaCommand)::create_bare(
                cmd_name, config, unique_name, local_buffer_container, *device, i, st);
        else
            // Create the cudaCommand object (without state arg)
            cmd = FACTORY(cudaCommand)::create_bare(cmd_name, config, unique_name,
                                                    local_buffer_container, *device, i);
        cmds.push_back(cmd);
    }
    DEBUG("Command added: {:s}", cmd_name.c_str());
    return cmds;
}

void cudaProcess::queue_commands(int gpu_frame_counter) {
    std::vector<cudaEvent_t> events;
    events.resize(device->get_num_streams(), nullptr);
    if (join_events.size() < (size_t)_gpu_buffer_depth)
        join_events.resize(_gpu_buffer_depth, nullptr);
    cudaEvent_t final_event = nullptr;
    int32_t final_stream_id = _my_stream_ids.front(); // only a fallback: no events, no join

    int icommand = gpu_frame_counter % _gpu_buffer_depth;
    {
        // Publish that we are about to take the lock, so a watchdog can tell a pipeline
        // blocked ON the lock from the one pipeline stuck INSIDE it. A call that never
        // returns cannot be timed after the fact, which is why this is written first.
        probe_enter("(queuing lock)", _my_stream_ids.front(), gpu_frame_counter, false, true);

        // Grab the queuing locks for THE STREAMS WE USE -- not one device-wide lock. See
        // cudaDeviceInterface::stream_mutex for why that was a whole-GPU outage waiting to
        // happen. _my_stream_ids is ascending, which is what makes taking several safe.
        // `gpu_command_lock: device` puts the old whole-device lock back, for re-exciting it.
        std::vector<std::unique_lock<std::recursive_mutex>> locks;
        if (device_wide_lock) {
            locks.emplace_back(device->gpu_command_mutex);
        } else {
            locks.reserve(_my_stream_ids.size());
            for (const std::int32_t sid : _my_stream_ids)
                locks.emplace_back(device->stream_mutex(sid));
        }

        // Create the state object that will get passed through this pipeline
        cudaPipelineState pipestate(gpu_frame_counter);

        for (auto& command : commands) {
            cudaCommand* cmd = (cudaCommand*)command[icommand];
            probe_enter(cmd->get_unique_name(), cmd->get_cuda_stream_id(), gpu_frame_counter, true,
                        false);
            const auto t0 = std::chrono::steady_clock::now();
            // Feed the last signal into the next operation
            cudaEvent_t event = cmd->execute_base(pipestate, events);
            const double held_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // Catches a command that is slow but RETURNS. The one that never returns is the
            // watchdog's job; it reads the record written above.
            if (held_s > slow_command_warn_s)
                WARN("SLOW COMMAND: {:s} took {:.2f} s on stream {:d} (frame {:d}), holding "
                     "the queuing lock throughout",
                     cmd->get_unique_name(), held_s, cmd->get_cuda_stream_id(), gpu_frame_counter);
            if (event != nullptr) {
                int32_t command_stream_id = cmd->get_cuda_stream_id();
                events[command_stream_id] = event;
                final_event = event;
                final_stream_id = command_stream_id;
            }
        }

        // ⚠️⚠️ THE FRAME IS NOT DONE WHEN THE LAST COMMAND IS DONE. Streams are independent
        // in-order queues and a command waits only on `pre_events[its own stream]`, so with
        // parallel per-stream chains the final command can finish while its siblings are
        // still running. The results thread then calls finalize_frame(), which releases host
        // frames (`cudaInputData` -> mark_frame_empty, `cudaOutputData` -> mark_frame_full),
        // so a consumer can read a half-written frame or a producer can overwrite a buffer a
        // DMA is still reading. Measured 2026-07-19: gal/bds epl frames with n_prn 0.
        //
        // ⚠️ JOIN ON THE FINAL COMMAND'S OWN STREAM, NEVER ON min(streams). The chain ENDS
        // there, so making it wait on the others adds no false dependency. Joining on the
        // lowest stream we own means stream 0 for any pipeline using the default triple --
        // the COPY-IN stream -- which puts the NEXT frame's host-to-device copy behind this
        // frame's compute and output, destroying the copy/compute overlap the three-stream
        // layout exists for. Stream 0 is shared by every pipeline left at the default base,
        // so that also couples every pipeline on the GPU through it: the same wedge this
        // class of change exists to remove, rebuilt in the stream graph where no backtrace
        // would ever show it.
        //
        // ⚠️ INSIDE THE LOCK. These are stream enqueues like any other; done outside, another
        // pipeline sharing this stream can interleave between the waits and the record.
        //
        // Pipelines that already terminate in a `cudaSyncOutput` (every one in this tree)
        // fold their compute streams in themselves, so `multi` stays false and this costs
        // nothing. It is here for the shapes that do not.
        std::vector<bool> has_event(events.size(), false);
        for (size_t i = 0; i < events.size(); ++i)
            has_event[i] = (events[i] != nullptr);
        const cudaFrameJoin join = plan_cuda_frame_join(has_event, final_stream_id);
        if (join.needed) {
            cudaStream_t join_stream = device->getStream(join.join_stream);
            for (const std::int32_t sid : join.wait_streams)
                CHECK_CUDA_ERROR(cudaStreamWaitEvent(join_stream, events[sid], 0));
            if (join_events[icommand] != nullptr)
                CHECK_CUDA_ERROR(cudaEventDestroy(join_events[icommand]));
            CHECK_CUDA_ERROR(
                cudaEventCreateWithFlags(&join_events[icommand], cudaEventDisableTiming));
            CHECK_CUDA_ERROR(cudaEventRecord(join_events[icommand], join_stream));
            final_event = join_events[icommand];
        }
    }
    probe_clear();
    final_signals[icommand]->set_signal(final_event);
    DEBUG2("Commands executed.");
}

void cudaProcess::probe_enter(const std::string& command, int32_t stream, int64_t frame,
                              bool holds_lock, bool waiting) {
    if (!in_flight)
        return;
    std::lock_guard<std::mutex> lock(in_flight->m);
    in_flight->stage = unique_name;
    in_flight->command = command;
    in_flight->stream = stream;
    in_flight->frame = frame;
    in_flight->entered_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
    in_flight->active = true;
    in_flight->holds_lock = holds_lock;
    in_flight->waiting = waiting;
}

void cudaProcess::probe_clear() {
    if (!in_flight)
        return;
    std::lock_guard<std::mutex> lock(in_flight->m);
    in_flight->active = false;
    in_flight->holds_lock = false;
    in_flight->waiting = false;
    in_flight->last_done_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch())
                                  .count();
    in_flight->frames_done++;
}

void cudaProcess::register_host_memory(Buffer* host_buffer) {
    // Register the host memory in buffers with the Cuda run time.
    for (int i = 0; i < host_buffer->num_frames; i++) {
        cudaHostRegister(host_buffer->frames[i], host_buffer->aligned_frame_size,
                         cudaHostRegisterDefault);
        DEBUG("Registered frame: {:s}[{:d}]", host_buffer->buffer_name, i);
    }
}
