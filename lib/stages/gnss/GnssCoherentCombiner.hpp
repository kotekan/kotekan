/**
 * @file
 * @brief Full-band complex amplitude per PRN and its temporal and deep coherent integration.
 *  - GnssCoherentCombiner : public kotekan::Stage
 */

#ifndef GNSS_COHERENT_COMBINER_HPP
#define GNSS_COHERENT_COMBINER_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "gnssCombinerChain.hpp" // for CombinerChain
#include "gnssHostRows.hpp"      // for HostRow
#include "gnssRecRows.hpp"       // for RecDir
#include "gnssRecord.hpp"        // for RECORD_FLOATS + slot names (the record schema)

#include <atomic>  // for atomic
#include <cstddef> // for size_t
#include <cstdint> // for uint64_t
#include <deque>   // for deque
#include <memory>  // for unique_ptr
#include <string>  // for string
#include <vector>  // for vector

/**
 * @class GnssCoherentCombiner
 * @brief Integrates one node's per-record despread into amplitudes and loop observables.
 *
 * Each record carries, per PRN, the un-normalized coherent correlation @f$ G @f$ and replica
 * energy @f$ E @f$ summed over the covering channels, and this stage forms the matched-filter
 * amplitude @f$ \hat A = G / E @f$. The broker combines across nodes. The output uses the
 * combiner record layout of gnssRecord.hpp.
 *
 * The integration itself is a gnss::CombinerChain; this stage moves its frames. With a
 * @c chains block it hosts one chain per row (see gnss::host_rows), each with its own buffers and
 * thread, under the REST paths, buffer client names and metric stage_name of the one-row stage
 * `<key>_n2combine` it replaces, its metrics also labelled with the row's @c chain, and serves
 * the rows at `<unique_name>/chains` and their verbs by chain at `<unique_name>/<verb>`
 * (gnss::rest_register_rows). A chain may be hosted on both GPU halves, so a series is
 * told apart by its stage_name, never by the label alone. With @c workers its rows run on pinned
 * threads instead, each taking one record of each of its rows in turn and reading their in_bufs as
 * one consumer, named for its row when it runs one, else `<unique_name>/w<k>`.
 *
 * With in_bufs on the host block every row reads that one buffer, a GPU half's record buffer
 * (GnssGpuRecordAssemble): each frame holds one record of every row, each at its @c rec_row,
 * after a directory that each worker checks against its rows' before it reads a frame. A
 * worker then takes each frame whole, folding in each of its rows' records in turn.
 *
 * Temporal integration is an exponential moving average with time constant
 * K = @c integration_length records (alpha = 1/K), updated every record and emitted every K/10
 * records. It gives both the incoherent amplitude
 * @f$ \sqrt{\langle|A|^2\rangle} @f$ (slot 3, ~sqrt(K) SNR, no phase needed) and the coherent
 * mean @f$ \langle A\rangle @f$ (slots 4-6, up to K SNR, but only while the carrier phase is
 * stable across the K records). Each is divided by 1-(1-alpha)^n at emit, n the slot's records
 * since its last reset, so it reads a true running mean from the first record. A slot is reset
 * only when its records carry a new PRN (@c reset_on_prn_change); its state then restarts as at
 * the stage's start, except that its ADR arc id keeps counting.
 *
 * Slot 8 is a deep coherent |A| over the buffered per-record A, by one of three paths:
 *  - nav-bit wipe (@c navwipe_bit_records): records are grouped into data-bit epochs, each
 *    bit's +-1 is estimated and wiped, and the bits are summed coherently, so |A| keeps
 *    growing past the data bit where slot 6 decorrelates.
 *  - overlay and nav-bit wipe (@c secondary_overlay with @c navwipe_bit_records): a known
 *    shared overlay is wiped at its best-ranked alignment, then the nav-bit wipe runs on the
 *    result (a data component under an overlay, e.g. GPS L5-I).
 *  - plain coherent sum (@c deep_coherent), for a replica that already carries the overlay.
 * Each path is evaluated over a ladder of octave-shorter trailing windows; a rung counts
 * as a coherent detection only if its SNR clears twice its noise-rectification floor (or, where
 * a rate search was applied, the search's peak/median gate).
 *
 * @conf n_prn               Int. Records (PRNs) per frame.
 * @conf n_elements          Int. Per-antenna blocks per PRN record, >= 1.
 * @conf integration_length  Int (default 1). EMA time constant in records, and the length of
 *                           the deep window.
 * @conf integration_mode    String. Must be "rolling". Default "rolling".
 * @conf navwipe_bit_records Int (default 0 = off). Records per data bit, e.g. 20 for 1 ms GPS
 *                           L1 C/A records. Needs integration_length >> this.
 * @conf secondary_overlay   String (default ""). Overlay name from the gnssOverlay registry
 *                           (shared overlays under a data component), wiped ahead of the
 *                           nav-bit wipe; needs navwipe_bit_records. Any other name is refused.
 * @conf deep_coherent       Bool. Must be true: without a nav-bit wipe the deep |A| is the plain
 *                           coherent sum, correct only when the despread replica already carries
 *                           the overlay. Default true.
 * @conf deep_rate_search    Bool. Must be true: the plain sum runs after a phase-rate search.
 *                           Default true.
 * @conf sky_deep            Bool. Must be true: the sky-corrected prompt is a plain-sum
 *                           candidate. Default true.
 * @conf deep_rate_min_q     Float. Must be 10, the peak/median the rate search must reach.
 *                           Default 10.
 * @conf phase_track         Bool. Must be false. Default false.
 * @conf fft_len             Int. F-engine FFT length, to turn sample_seq into hops.
 * @conf record_export       Int (default 0 = off). Per-PRN record ring length for get_records.
 * @conf bit_export          Bool (default false). Publish nav_obs in get_status.
 * @conf bit_export_track    Int (default 0). Half-width in bits of the local carrier phase that
 *                           each exported bit is decided against; 0 = the window's one phase.
 * @conf reset_on_prn_change Bool (default true). When a slot's records carry a new PRN (a live
 *                           swap), its state restarts cold, as GnssGpuRecordAssemble's does.
 * @conf chain               String, set by each row of @c chains. The row's broker chain (e.g.
 *                           gal_e5a), which labels its metrics.
 * @conf chains              Map (optional). One row per entry; each sets ROW_KEYS itself, alone
 *                           sets ROW_ONLY_KEYS, and reads every other key from this block.
 * @conf rec_row             With in_bufs on the host block, each row's: its block in each frame,
 *                           {chain_id, n_prn, n_chan, offset, bytes} (gnss::RecRow), which the
 *                           frame's directory must list.
 * @conf workers             List (optional, with chains). Worker k, `{rows: [row keys],
 *                           cpu_affinity: [cores]}`, runs its rows in that order on one thread
 *                           bound to its cores, named for its row when it runs one, else
 *                           `<unique_name>/w<k>`. Every row is in exactly one worker, and the rows
 *                           of a worker read distinct in_bufs. They also share the thread: a row
 *                           whose record has not come, or whose out_buf is full, holds the others,
 *                           and on a busy core they get one thread's share of it. So rows fed by
 *                           one thread that writes n records of a row before the next row's
 *                           deadlock unless their in_bufs hold at least n - 1 frames. A GPU
 *                           half's record buffer, which every row reads, holds every row's record
 *                           in each frame, so its rows wait only for each other's emits.
 *
 * @buffer in_bufs A list of one buffer: the node's records (gnssRecord.hpp tracker layout: a
 *                 RECORD_FLOATS header plus n_elements element blocks per PRN).
 * @buffer out_buf Optional. Combined records (gnssRecord.hpp combiner layout: 3 = |A|_incoh,
 *                 4,5 = <A>, 6 = |<A>|_coh, 7 = n_chan, 8 = deep |A|, 9-10 = UTC). Without it
 *                 each emit is written to a member record, which only the REST snapshot reads.
 *
 * @author Keith Vanderlinde
 */
class GnssCoherentCombiner : public kotekan::Stage {
public:
    GnssCoherentCombiner(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    ~GnssCoherentCombiner() override;
    void main_thread() override;

    static constexpr int RECORD_UTC_SLOT = gnss::RECORD_UTC_SLOT;

    /// Keys each row of a @c chains block must set itself.
    static const std::vector<std::string> ROW_KEYS;
    /// Keys only a row of a @c chains block may set.
    static const std::vector<std::string> ROW_ONLY_KEYS;

    /// Input records row @p row has folded in, each with the emit it completed; a test reads it
    /// to wait for the stage to go idle.
    uint64_t records_done(size_t row = 0) const {
        return _rows[row].records_done.load();
    }
    /// Row @p row's last emit when it has no out_buf (n_prn x record stride floats). Stable
    /// only while the row waits for input.
    const std::vector<float>& scratch(size_t row = 0) const {
        return _rows[row].scratch;
    }

private:
    struct Row {
        Row(kotekan::Config& config, const gnss::HostRow& row);
        Buffer* in_buf = nullptr;
        int in_id = 0;
        Buffer* out_buf = nullptr; ///< nullptr when the config has no out_buf
        int out_id = 0;
        /// The emit's records when there is no out_buf.
        std::vector<float> scratch;
        /// With a GPU half's record buffer: its block's offset and its row of the directory.
        size_t rec_offset = 0, dir_row = 0;
        std::atomic<uint64_t> records_done{0};
        gnss::CombinerChain chain;
    };
    const std::vector<gnss::HostRow> _host_rows;
    std::deque<Row> _rows;
    /// Its threads: each takes a record of each of its rows in turn (gnss::host_workers).
    const std::vector<gnss::HostWorker> _workers;
    /// The REST paths this host registered.
    std::vector<std::string> _rest_paths;
    /// With in_bufs on the host block: that buffer and its directory.
    Buffer* _rec_buf = nullptr;
    std::unique_ptr<gnss::RecDir> _rec_dir;

    /// The keys each row sets itself: ROW_KEYS, with rec_row for in_bufs when the host block of
    /// @p unique_name sets it.
    static std::vector<std::string> row_keys(kotekan::Config& config,
                                             const std::string& unique_name);
    /// Register worker @p w as the consumer of its rows' in_bufs.
    void register_inputs(const gnss::HostWorker& w);
    /// Worker @p w's consumer and thread name: its row's name when it runs one, else its own.
    const std::string& consumer_name(const gnss::HostWorker& w) const;
    void run_worker(size_t w);
    /// As worker @p w, take each frame of the GPU half's record buffer and fold in its rows'
    /// records. Returns when shutting down.
    void run_frames(const gnss::HostWorker& w);
    /// run_worker on a thread of its own, counted among the stage's threads.
    void run_pinned(size_t w);
    /// Fold row @p row's next record into its chain, reading it as consumer @p consumer, and
    /// emit when one is due. Returns false when shutting down.
    bool take_record(size_t row, const std::string& consumer);
    /// Fold record @p in of row @p row, whose window starts at sample @p sample_seq (-1 if
    /// unknown), into its chain. Returns true when an emit is due.
    bool ingest_record(size_t row, const float* in, int64_t sample_seq);
    /// Emit row @p row's records if @p emit, and count its record taken. Returns false when
    /// shutting down.
    bool finish_record(size_t row, bool emit);
};

#endif // GNSS_COHERENT_COMBINER_HPP
