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
#include "gnssRecord.hpp"        // for RECORD_FLOATS + slot names (the record schema)

#include <atomic>  // for atomic
#include <cstddef> // for size_t
#include <cstdint> // for uint64_t
#include <deque>   // for deque
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
 * thread, under the REST paths, buffer client names and metric labels of the one-row stage
 * `<key>_n2combine` it replaces, and serves the rows at `<unique_name>/chains`.
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
 * @conf reset_on_prn_change Bool (default true). When a slot's records carry a new PRN (a live
 *                           swap), its state restarts cold, as GnssGpuRecordAssemble's does.
 * @conf chains              Map (optional). One row per entry; each sets ROW_KEYS itself, alone
 *                           sets ROW_ONLY_KEYS, and reads every other key from this block.
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
        Buffer* out_buf = nullptr; ///< nullptr when the config has no out_buf
        /// The emit's records when there is no out_buf.
        std::vector<float> scratch;
        std::atomic<uint64_t> records_done{0};
        gnss::CombinerChain chain;
    };
    const std::vector<gnss::HostRow> _host_rows;
    std::deque<Row> _rows;

    void run_row(size_t row);
};

#endif // GNSS_COHERENT_COMBINER_HPP
