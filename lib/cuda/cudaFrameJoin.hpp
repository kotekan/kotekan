#ifndef CUDA_FRAME_JOIN_HPP
#define CUDA_FRAME_JOIN_HPP

#include <cstdint>
#include <vector>

/**
 * @brief Which streams an end-of-frame join must wait on, and where to record it.
 *
 * A cudaProcess enqueues one frame's commands across several CUDA streams. Streams are
 * independent in-order queues: a command waits only on the previous event of its own stream
 * (cudaInputData/cudaOutputData pass `pre_events[cuda_stream_id]`), and the only cross-stream
 * barriers are the cudaSyncStream commands (cudaSyncInput, cudaSyncOutput), which a pipeline
 * has to be configured to use. So the final command completing does not mean the frame is
 * done: with parallel per-stream chains its siblings can still be running.
 *
 * Frame completion drives gpuProcess::results_thread -> gpuCommand::finalize_frame(), which
 * releases host frames: cudaInputData::finalize_frame calls mark_frame_empty() (a producer may
 * then overwrite a buffer a DMA is still reading) and cudaOutputData::finalize_frame calls
 * mark_frame_full() (a consumer may then read a half-written frame).
 *
 * Kept out of cudaProcess::queue_commands so the decision can be unit tested without a GPU
 * (tests/boost/test_cuda_frame_join.cpp): it is pure, and every CUDA call stays at the call site.
 */
struct cudaFrameJoin {
    /// False when no stream other than the final command's carries an event: no join is needed.
    bool needed = false;
    /// The stream to record the join event on: always the final command's stream.
    std::int32_t join_stream = 0;
    /// The streams whose last event the join stream must wait on, ascending. Never contains
    /// join_stream.
    std::vector<std::int32_t> wait_streams;
};

/**
 * @brief Plan the end-of-frame join.
 *
 * @param stream_has_event  indexed by stream id; true where that stream carries an event from
 *                          this frame (`events[i] != nullptr` in queue_commands).
 * @param final_stream      the stream of the last command that returned an event.
 *
 * The join goes on the final command's stream, not the lowest stream the pipeline owns. The
 * chain ends there, so waiting on the others adds no false dependency. The lowest stream of
 * the default triple is the copy-in stream: a join there would queue the next frame's
 * host-to-device copy behind this frame's compute and output, losing the copy/compute overlap
 * the three streams exist for, and would couple every pipeline that shares that stream.
 */
inline cudaFrameJoin plan_cuda_frame_join(const std::vector<bool>& stream_has_event,
                                          std::int32_t final_stream) {
    cudaFrameJoin plan;
    plan.join_stream = final_stream;
    for (std::size_t i = 0; i < stream_has_event.size(); ++i) {
        // Include stream 0: at cuda_stream_base 0 it is the copy-in stream, whose DMA must
        // finish before the host input frame is released.
        const std::int32_t sid = static_cast<std::int32_t>(i);
        if (stream_has_event[i] && sid != final_stream)
            plan.wait_streams.push_back(sid);
    }
    plan.needed = !plan.wait_streams.empty();
    return plan;
}

#endif // CUDA_FRAME_JOIN_HPP
