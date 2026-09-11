#ifndef CUDA_FRAME_JOIN_HPP
#define CUDA_FRAME_JOIN_HPP

#include <cstdint>
#include <vector>

/**
 * @brief Which streams an end-of-frame join must wait on, and where to record it.
 *
 * A cudaProcess enqueues one frame's commands across several CUDA streams. Streams are
 * INDEPENDENT in-order queues -- a command waits only on the previous event of its OWN stream
 * (cudaInputData/cudaOutputData pass `pre_events[cuda_stream_id]`), and the only cross-stream
 * barriers are cudaSyncInput/cudaSyncOutput, which a pipeline has to be configured to use. So
 * "the last command finished" does NOT mean "the frame finished": with parallel per-stream
 * chains the final command can complete while its siblings are still running.
 *
 * That matters because frame completion drives gpuProcess::results_thread ->
 * gpuCommand::finalize_frame(), which RELEASES HOST FRAMES -- cudaInputData::finalize_frame
 * calls mark_frame_empty() (a producer may then overwrite a buffer a DMA is still reading) and
 * cudaOutputData::finalize_frame calls mark_frame_full() (a consumer may then read a
 * half-written frame). Observed 2026-07-19 as gal/bds epl frames carrying n_prn 0.
 *
 * This is split out of cudaProcess::queue_commands so the decision can be unit tested without
 * a GPU: it is pure, and every CUDA call stays at the call site. See
 * tests/boost/test_cuda_frame_join.cpp.
 */
struct cudaFrameJoin {
    /// False when the final command's stream already carries every event: no join, no cost.
    bool needed = false;
    /// The stream to record the join event on. ALWAYS the final command's stream.
    std::int32_t join_stream = 0;
    /// The streams whose last event the join stream must wait on. Never contains join_stream.
    std::vector<std::int32_t> wait_streams;
};

/**
 * @brief Plan the end-of-frame join.
 *
 * @param stream_has_event  indexed by stream id; true where that stream carries an event from
 *                          this frame (i.e. `events[i] != nullptr` in queue_commands).
 * @param final_stream      the stream the LAST command to return an event ran on.
 *
 * ⚠️ THE JOIN STREAM IS THE FINAL COMMAND'S, NOT THE LOWEST ONE THE PIPELINE OWNS. The chain
 * ends on the final command's stream, so making it wait on the others adds no false
 * dependency. Joining on min(streams) instead means stream 0 for any pipeline left on the
 * default triple -- the COPY-IN stream -- which puts the NEXT frame's host-to-device copy
 * behind this frame's compute and output, destroying the copy/compute overlap the three-stream
 * layout exists for; and since every default pipeline shares stream 0, it couples every
 * pipeline on the GPU through it. That was a real regression, caught in review 2026-09-11, and
 * `join_stream_is_the_final_stream_never_the_lowest` below is its regression test.
 */
inline cudaFrameJoin plan_cuda_frame_join(const std::vector<bool>& stream_has_event,
                                          std::int32_t final_stream) {
    cudaFrameJoin plan;
    plan.join_stream = final_stream;
    for (std::size_t i = 0; i < stream_has_event.size(); ++i) {
        // Start at 0, not 1: a non-final event on stream 0 is exactly the copy-in whose DMA
        // must not outlive the frame signal, and skipping it was how the first version of
        // this could release a host input frame mid-transfer.
        const std::int32_t sid = static_cast<std::int32_t>(i);
        if (stream_has_event[i] && sid != final_stream)
            plan.wait_streams.push_back(sid);
    }
    plan.needed = !plan.wait_streams.empty();
    return plan;
}

#endif // CUDA_FRAME_JOIN_HPP
