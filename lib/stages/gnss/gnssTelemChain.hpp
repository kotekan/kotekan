#ifndef GNSS_TELEM_CHAIN_HPP
#define GNSS_TELEM_CHAIN_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "kotekanLogging.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gnss {

/**
 * @class TelemChain
 * @brief The telemetry packing of one chain (see GnssTelemPack), without the buffers: a host
 *        feeds it records and takes its wire frames.
 *
 * Per input record the host calls on_grid(), flushes the open window when window_changes(),
 * calls add_record(), and flushes again once window_full(). A flush is pack() into an out_buf
 * frame. The config is read at @c cfg_path; @c name prefixes the log lines. A one-row host passes
 * its unique_name as both.
 */
class TelemChain : public kotekan::kotekanLogging {
public:
    TelemChain(kotekan::Config& config, const std::string& cfg_path, const std::string& name);

    /// FATAL unless @p out_buf's frames hold one wire frame, exactly when @p exact, and
    /// @p in_buf's a record frame.
    void check_bufs(const Buffer* in_buf, const Buffer* out_buf, bool exact = true);
    /// The chain and instance tags every frame carries.
    const std::string& chain() const {
        return _chain;
    }
    const std::string& inst() const {
        return _inst;
    }

    /// Whether a record at @p wstart has a slot; a record without one is counted dropped.
    bool on_grid(int64_t wstart);
    /// Whether a record at @p wstart closes the open window.
    bool window_changes(int64_t wstart) const;
    /// Place the record frame @p in, at @p wstart on the record grid, in the open window.
    void add_record(const float* in, int64_t wstart);
    /// A window is open.
    bool window_open() const {
        return _cur_win >= 0;
    }
    /// Every record slot of the open window is filled.
    bool window_full() const {
        return _cur_present == ((1u << _rec_per_frame) - 1u);
    }
    /// Write the open window into @p frame, an out_buf frame, and reset. Returns its wstart0.
    int64_t pack(uint8_t* frame);
    /// Drop the open window and reset. Its seq is skipped, so the receiver reads it as lost.
    void drop();

private:
    /// Close the open window: none is open after.
    void reset();

    const std::string _name;

    std::string _chain, _inst;
    int _n_prn = 0;
    int _n_elem = 0;
    int _max_prn = 0;
    int _rec_per_frame = 4;
    int64_t _hops_per_record = 0;
    int64_t _fft_len = 0;
    /// Comb columns this sender ships (the header's max_chan): the unsummed comb (gnssRecord.hpp
    /// chan block) goes after each row's record header, with the column freq_ids in the frame
    /// header, so the broker can fit a delay across channels.
    int _n_chan = 0;
    std::vector<int> _chan_ids; ///< [n_chan] freq_id per comb column; rides on every frame

    /// records_per_frame * hops_per_record * fft_len: the window length in samples.
    int64_t _win_samples = 0;
    /// hops_per_record * fft_len: one record in samples, the slot quantum.
    int64_t _rec_samples = 0;

    // The frame being accumulated. -1 = none open.
    int64_t _cur_win = -1;
    uint32_t _cur_present = 0;
    double _cur_utc0 = 0.0;
    std::vector<float> _rows; ///< [rec_per_frame][max_prn][telem_row_floats(_n_chan)]

    /// Wire row of input PRN slot p in the open window, or -1 if it is not carried. Fixed for
    /// the window; see GnssTelemPack on row compaction.
    std::vector<int> _row_of;
    bool _map_built = false;

    uint64_t _seq = 0;          ///< frames emitted; the receiver reads gaps as loss
    uint64_t _dropped = 0;      ///< records that could not be placed (no metadata / bad wstart)
    uint64_t _prn_overflow = 0; ///< windows where more PRNs were live than the wire can carry
};

} // namespace gnss

#endif // GNSS_TELEM_CHAIN_HPP
