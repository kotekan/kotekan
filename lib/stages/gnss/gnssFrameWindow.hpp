/**
 * @file
 * @brief The armed windows of the FrameWindowGate stages, for the stages upstream of a gate.
 *  - gnss::FrameWindowBoard
 */
#ifndef GNSS_FRAME_WINDOW_HPP
#define GNSS_FRAME_WINDOW_HPP

#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace gnss {

/**
 * @class FrameWindowBoard
 * @brief Process-wide board of the FrameWindowGate windows, by gate unique_name.
 *
 * A gate publishes its window whenever it changes (under its own lock, so the board follows
 * the gate's order of changes). A stage upstream of the gate (cudaGnssBeamDump) asks whether
 * the frame it is about to produce would pass, and skips the work the gate would drop.
 */
class FrameWindowBoard {
public:
    static FrameWindowBoard& instance() {
        static FrameWindowBoard board;
        return board;
    }

    /// The gate's window now: it passes a frame with clock seq when armed and
    /// start_seq <= seq < end_seq (end_seq <= 0: open-ended).
    void publish(const std::string& gate, bool armed, int64_t start_seq, int64_t end_seq) {
        std::lock_guard<std::mutex> lk(_mtx);
        _windows[gate] = Window{armed, start_seq, end_seq};
    }

    void remove(const std::string& gate) {
        std::lock_guard<std::mutex> lk(_mtx);
        _windows.erase(gate);
    }

    /// Whether @p gate passes a frame whose clock reads @p seq. False when no such gate has
    /// published, which @p found reports.
    bool in_window(const std::string& gate, int64_t seq, bool& found) const {
        std::lock_guard<std::mutex> lk(_mtx);
        const auto it = _windows.find(gate);
        found = it != _windows.end();
        if (!found)
            return false;
        const Window& w = it->second;
        return w.armed && seq >= w.start_seq && (w.end_seq <= 0 || seq < w.end_seq);
    }

private:
    struct Window {
        bool armed;
        int64_t start_seq;
        int64_t end_seq;
    };
    mutable std::mutex _mtx;
    std::map<std::string, Window> _windows;
};

} // namespace gnss

#endif // GNSS_FRAME_WINDOW_HPP
