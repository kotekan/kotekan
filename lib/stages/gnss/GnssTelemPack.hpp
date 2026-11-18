#ifndef GNSS_TELEM_PACK_HPP
#define GNSS_TELEM_PACK_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssHostRows.hpp"
#include "gnssTelemChain.hpp"
#include "prometheusMetrics.hpp" // for Counter
#include "visUtil.hpp"           // for frameID

#include <cstddef>
#include <deque>
#include <mutex>
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
 * it hosts one chain per row (see gnss::host_rows), each with its own in_buf and thread, under
 * the buffer client names of the one-row stage `<key>_telem_pack` it replaces, and serves the
 * rows at `<unique_name>/chains`. With @c workers its rows run on pinned threads instead, each
 * taking one record of each of its rows in turn and reading their in_bufs as one consumer, named
 * for its row when it runs one, else `<unique_name>/w<k>`.
 *
 * Each row writes its own out_buf, or with an out_buf on the host block every row writes that
 * one, a frame at a time, and GnssTelemSend ships each frame at its own length. Rows that share
 * an out_buf never wait for it: a window that finds the next frame still full is dropped and
 * counted, and its seq is skipped, so the receiver reads the gap.
 *
 * @par buffers
 * @buffer in_buf  tracker record frames (one per record; GnssChanMetadata::sample_seq = wstart)
 * @buffer out_buf telemetry frames, of exactly
 *                  gnss::telem_frame_bytes(records_per_frame, max_prn, n_chan) for a row's own,
 *                  of at least that for every row of a shared one
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
 * @conf chains            Map (optional). One row per entry; each sets ROW_KEYS itself, and
 *                           out_buf unless this block sets it, and reads every other key from
 *                           this block. No two rows may carry one chain and inst.
 * @conf workers           List (optional, with chains). Worker k, `{rows: [row keys],
 *                           cpu_affinity: [cores]}`, runs its rows in that order on one thread
 *                           bound to its cores. Every row is in exactly one worker, and the rows
 *                           of a worker read distinct in_bufs. A row whose record has not come
 *                           holds the others of its worker.
 *
 * @par Metrics
 * @metric kotekan_gnss_telem_dropped_frames_total
 *         Per row of a shared out_buf (stage_name its one-row stage, chain its chain): windows
 *         dropped because the out_buf was full. Frames the send drops itself are counted once
 *         for all rows, in its kotekan_buffer_send_dropped_frame_count; a row's seq gaps count
 *         both.
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
    /// An out_buf and its writer; the rows that share it take turns.
    struct Out {
        Out(Buffer* buf, const std::string& producer);
        Buffer* buf;
        const std::string producer;
        frameID id;
        std::mutex lock;
    };
    struct Row {
        Row(kotekan::Config& config, const gnss::HostRow& row, Buffer* in, Out* out);
        Buffer* in_buf;
        frameID in_id;
        Out* out;
        /// Windows dropped on a full shared out_buf; nullptr for a row's own out_buf.
        kotekan::prometheus::Counter* dropped = nullptr;
        gnss::TelemChain chain;
    };
    const std::vector<gnss::HostRow> _host_rows;
    /// The out_bufs: one per row, or one the rows share.
    std::deque<Out> _outs;
    std::deque<Row> _rows;
    /// Its threads: each takes a record of each of its rows in turn (gnss::host_workers).
    const std::vector<gnss::HostWorker> _workers;
    /// The REST paths this host registered.
    std::vector<std::string> _rest_paths;

    /// The keys each row sets itself: ROW_KEYS, and out_buf unless the host block of
    /// @p unique_name sets it.
    static std::vector<std::string> row_keys(kotekan::Config& config,
                                             const std::string& unique_name);
    /// Register worker @p w as the consumer of its rows' in_bufs.
    void register_inputs(const gnss::HostWorker& w);
    /// Worker @p w's consumer and thread name: its row's name when it runs one, else its own.
    const std::string& consumer_name(const gnss::HostWorker& w) const;
    void run_worker(size_t w);
    /// run_worker on a thread of its own, counted among the stage's threads.
    void run_pinned(size_t w);
    /// Pack row @p row's next record, reading it as consumer @p consumer, and write its window
    /// out once complete. Returns false when shutting down.
    bool take_record(size_t row, const std::string& consumer);
    /// Write row @p row's accumulated window out, or drop it on a full shared out_buf, and
    /// reset. No-op if nothing is open. Returns false if the output buffer shut down (the
    /// caller must exit).
    bool flush(size_t row);
};

#endif // GNSS_TELEM_PACK_HPP
