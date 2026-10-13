/**
 * @file
 * @brief Stage that passes a REST-selected window of frames, dropping everything else.
 *  - FrameWindowGate : public kotekan::Stage
 */
#ifndef FRAME_WINDOW_GATE_HPP
#define FRAME_WINDOW_GATE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "prometheusMetrics.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class FrameWindowGate
 * @brief Pass frames whose sequence number falls in an armed [start_seq, end_seq) window;
 *        drop every other frame without blocking the producer.
 *
 * Sits between a live pipeline and an expensive sink (a rawFileWrite) that should run only for
 * a window an operator picks at runtime. Unarmed, it consumes and drops every frame. Armed, it
 * copies each in-window frame to the outputs if every output has an empty frame, and otherwise
 * drops and counts it: the gate never waits on the sink, so the sink never holds up the producer.
 *
 * Several input buffers can be gated as a tuple: frame k of every input is read, passed or
 * dropped together, so the sink files stay frame-aligned (a tiles frame and its control block,
 * say). The clock is the sequence number of one of them (@c clock_buf): an int64 at a byte
 * offset inside that frame.
 *
 * REST (all under this stage's unique_name):
 *  - ``POST .../arm``     ``{"start_seq": X, "end_seq": Y}``: pass frames with
 *                          ``start_seq <= seq < end_seq``. ``end_seq`` is optional; <= 0 means
 *                          until disarmed, else the gate disarms itself at the first frame with
 *                          ``seq >= end_seq``. Re-arming replaces the window.
 *  - ``POST .../disarm``  close the window now.
 *  - ``GET  .../status``  ``{armed, start_seq, end_seq, last_seq, seq_per_frame, passed, dropped,
 *                          dropped_in_window}``. ``seq_per_frame`` is the last observed step
 *                          between consecutive frames, so a caller can align a window to frame
 *                          edges without knowing the pipeline's geometry.
 *
 * @par Buffers
 * @buffer in_bufs   Input frames, consumed in lockstep.
 *         @buffer_format any
 *         @buffer_metadata any; deep-copied if the output has a metadata pool (same type required)
 * @buffer out_bufs  One per input, same frame_size; receives the in-window frames.
 *
 * @conf in_bufs        Array of buffer names.
 * @conf out_bufs       Array of buffer names, same length as in_bufs.
 * @conf clock_buf      Int, default 0. Index into in_bufs of the buffer that carries the clock.
 * @conf clock_source   String, default "frame", the only value accepted: the clock is the int64
 *                      at byte offset @c clock_offset of the clock frame.
 * @conf clock_offset   Int, default 0. Byte offset of the clock in the clock frame.
 *
 * @par Metrics
 * @metric kotekan_framewindowgate_passed_frames_total   Tuples copied to the outputs.
 * @metric kotekan_framewindowgate_dropped_frames_total  Tuples dropped outside the window.
 * @metric kotekan_framewindowgate_window_dropped_frames_total  Tuples dropped inside the window
 *                      because an output was full: the sink is not keeping up.
 * @metric kotekan_framewindowgate_armed  1 while frames are inside the window (not merely armed).
 */
class FrameWindowGate : public kotekan::Stage {
public:
    FrameWindowGate(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    ~FrameWindowGate() override;
    void main_thread() override;

private:
    int64_t read_seq(const uint8_t* frame) const;
    void copy_frame(Buffer* src, int src_id, Buffer* dst, int dst_id);

    std::vector<Buffer*> _in_bufs;
    std::vector<Buffer*> _out_bufs;
    int _clock_buf;
    size_t _clock_offset;

    // Window state and counters, guarded by _mtx: shared by the REST callbacks and main_thread.
    std::mutex _mtx;
    bool _armed = false;
    int64_t _start_seq = 0;
    int64_t _end_seq = 0;
    int64_t _last_seq = -1;
    int64_t _seq_per_frame = 0;
    uint64_t _passed = 0;
    uint64_t _dropped = 0;
    uint64_t _dropped_in_window = 0;
};

#endif // FRAME_WINDOW_GATE_HPP
