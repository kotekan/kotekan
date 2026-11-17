#ifndef GNSS_TELEM_PACK_HPP
#define GNSS_TELEM_PACK_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssHostRows.hpp"
#include "gnssTelemChain.hpp"
#include "visUtil.hpp" // for frameID

#include <cstddef>
#include <deque>
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
 * The packing itself is a gnss::TelemChain; this stage moves its frames. With a @c chains block
 * it hosts one chain per row (see gnss::host_rows), each with its own buffers and thread, under
 * the buffer client names of the one-row stage `<key>_telem_pack` it replaces, and serves the
 * rows at `<unique_name>/chains`.
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
 * @conf chains            Map (optional). One row per entry; each sets ROW_KEYS itself and reads
 *                           every other key from this block.
 */
class GnssTelemPack : public kotekan::Stage {
public:
    GnssTelemPack(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~GnssTelemPack() override;
    void main_thread() override;

    /// Keys each row of a @c chains block must set itself.
    static const std::vector<std::string> ROW_KEYS;

private:
    struct Row {
        Row(kotekan::Config& config, const gnss::HostRow& row, Buffer* in, Buffer* out);
        Buffer* in_buf;
        frameID in_id;
        Buffer* out_buf;
        frameID out_id;
        gnss::TelemChain chain;
    };
    const std::vector<gnss::HostRow> _host_rows;
    std::deque<Row> _rows;
    /// The REST paths this host registered.
    std::vector<std::string> _rest_paths;

    void run_row(size_t row);
    /// Pack row @p row's next record, reading it as consumer @p consumer, and write its window
    /// out once complete. Returns false when shutting down.
    bool take_record(size_t row, const std::string& consumer);
    /// Write row @p row's accumulated window out and reset. No-op if nothing is open.
    /// Returns false if the output buffer shut down (the caller must exit).
    bool flush(size_t row);
};

#endif // GNSS_TELEM_PACK_HPP
