#ifndef GNSS_TELEM_PACK_HPP
#define GNSS_TELEM_PACK_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssTelem.hpp"
#include "visUtil.hpp" // for frameID

#include <string>
#include <vector>

/**
 * @class GnssTelemPack
 * @brief Node side of the tracker-to-broker telemetry transport: packs tracker records into
 *        fixed-size telemetry frames addressed on the F-engine sample clock.
 *
 * A second consumer of the tracker's record buffer, beside the combiner. Each record is cut to
 * its gnssRecord.hpp header and its per-channel comb block, dropping the element blocks, and
 * @c records_per_frame records go into one wire frame (gnssTelem.hpp) for bufferSend.
 *
 * The batch boundary is absolute. A frame holds window
 *     win = wstart / (records_per_frame * hops_per_record * fft_len)
 * on the F-engine's sample counter, not N consecutive local records: a local count starts
 * wherever the stage started and slips on every dropped record, so instances would batch
 * different records. Computed from three integers every sender shares, the boundary needs no
 * negotiation and cannot drift.
 *
 * Record slot r within the frame is addressed, never appended:
 *     slot r <=> wstart == win*window_samples + r*hops_per_record*fft_len
 * so a dropped record leaves a hole at a known index (its @c present bit unset) instead of
 * shifting the records after it.
 *
 * Wire rows are compacted: only PRNs despread in the first record of the window to arrive get a
 * row, so @c max_prn is the number of live PRNs a window can carry, not the slot count. Readers
 * take each row's PRN from the first filled record slot and apply it to every slot: only the
 * slots whose record arrived carry rows.
 *
 * @par buffers
 * @buffer in_buf  tracker record frames (one per record; GnssChanMetadata::sample_seq = wstart)
 * @buffer out_buf telemetry frames of exactly
 *                  gnss::telem_frame_bytes(records_per_frame, max_prn, n_chan)
 *
 * @conf chain             String. Chain tag carried on every frame ("gps_l5", "gal_e5a", ...).
 *                           Shorter than gnss::TELEM_NAME.
 * @conf inst              String. Instance tag ("<node>.<gpu>"), shorter than gnss::TELEM_NAME.
 *                           With @c chain it identifies the sender, so it must be unique within
 *                           a chain.
 * @conf n_prn             Int. PRN slots in the source record frame.
 * @conf n_elements        Int, default 0. Element blocks in the source record (for the stride).
 * @conf max_prn           Int. PRN rows on the wire, >= 1. May be less than n_prn; a window
 *                           with more despread PRNs ships the lowest slots and WARNs.
 * @conf records_per_frame Int, default 4. Records per wire frame, in [1, gnss::TELEM_MAX_REC].
 *                           Set it to divide the records per correlator frame so a wire frame
 *                           does not straddle two.
 * @conf hops_per_record   Int. Record length in hops.
 * @conf fft_len           Int. Samples per hop.
 * @conf n_chan            Int. Covering channels this instance despreads: the columns per PRN of
 *                           the source frame's comb block, shipped after each PRN's record
 *                           header. In [1, gnss::TELEM_MAX_CHAN].
 * @conf chan_export       Bool, must be true (default true).
 * @conf channel_ids       Int list. freq_id of each comb column, sent in every frame header;
 *                           n_chan entries.
 */
class GnssTelemPack : public kotekan::Stage {
public:
    GnssTelemPack(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~GnssTelemPack() override = default;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

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
    /// the window; see the class note on row compaction.
    std::vector<int> _row_of;
    bool _map_built = false;

    uint64_t _seq = 0;          ///< frames emitted; the receiver reads gaps as loss
    uint64_t _dropped = 0;      ///< records that could not be placed (no metadata / bad wstart)
    uint64_t _prn_overflow = 0; ///< windows where more PRNs were live than the wire can carry

    /// Write the accumulated window out and reset. No-op if nothing is open.
    /// Returns false if the output buffer shut down (the caller must exit).
    bool flush(frameID& out_id);
};

#endif // GNSS_TELEM_PACK_HPP
