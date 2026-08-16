#ifndef GNSS_TELEM_HPP
#define GNSS_TELEM_HPP
/**
 * @file gnssTelem.hpp
 * @brief THE TRACKER -> BROKER TELEMETRY WIRE FORMAT (task #59) -- one frame-synced record
 *        stream per (chain, instance), carried by kotekan's bufferSend/bufferRecv.
 *
 * WHY THIS EXISTS. The broker used to build its fleet products from ~60 REST round trips per
 * cycle (12 instances x 5 chains), each landing at a different wall time, and then had to work
 * out AFTERWARDS which instance and which window each reply described. Every one of those
 * ambiguities produced a real defect inside a single week:
 *   * #53 -- /get_spectrum windows were "whatever accumulated since your last GET", so no two
 *     instances ever summed the same records and the cross-instance phase was free.
 *   * #52 -- the delay fit absorbed that free phase per instance, i.e. fitted what it should
 *     have derived.
 *   * #33 -- `res_cycles` is a per-instance ACCUMULATOR whose origin is that instance's arc
 *     start; differencing it across a "served row" (which silently changes instance) read
 *     4.92 Hz where one instance read 0.82 Hz. 6x, from addressing alone.
 *   * #46 -- 0.105 s of record-time spread between instances that nothing in the design buffers.
 *
 * None of those are physics. They are all the same defect: THE ADDRESS WAS INFERRED. So the
 * address now travels WITH the data, on the F-engine's own sample clock, and the broker stops
 * inferring anything.
 *
 * TWO CONTRACTS, both learned the hard way; break either and this is no better than the polls.
 *
 * (1) SHIP INCREMENTS, NEVER ACCUMULATIONS. Every float here is a PER-RECORD quantity, and the
 *     two phase slots that could have been accumulators (REC_CPHASE, REC_TRIM_INC) are already
 *     defined as increments since the PRN's previous record precisely because a float32 cannot
 *     hold 1e7 cycles (see gnssRecord.hpp). Accumulation happens ONCE, in the broker, over a
 *     run of records it can see is contiguous. An accumulator on the wire cannot be repaired
 *     downstream by any amount of transport synchronisation, because its ORIGIN is not on the
 *     wire.
 *
 * (2) KEY ON `win`/`wstart`, NEVER ON THE RECEIVER'S FRAME ORDER. bufferRecv makes NO ordering
 *     promise across connections (its own `frame_id` is a local ring index,
 *     `current_frame_id = (current_frame_id + 1) % num_frames`), and 60 senders share one
 *     listener. Collating on arrival order silently mis-pairs instances the moment one drops a
 *     frame or restarts -- which is exactly the bug class this file exists to remove. The
 *     window index below is computed from the ABSOLUTE F-engine sample counter with a divisor
 *     every sender derives identically, so two instances agree on it without negotiating.
 *
 * LAYOUT
 *
 *     [ TelemHeader, 112 B ][ float32 rows[n_rec][n_prn][gnss::TELEM_ROW_FLOATS] ]
 *
 * and a row is
 *
 *     [ gnss::RECORD_FLOATS -- the tracker record header ][ the UNSUMMED COMB ]
 *       ...................................................  CHAN_FLOATS x TELEM_MAX_CHAN
 *
 * ⚠️ THE COMB IS THE POINT (v2, 2026-08-14). The record header's prompt (slots 3/4) is SUMMED
 * over the instance's covering channels by GnssGpuRecordAssemble -- "the one combine the broker
 * can never undo". That sum destroys the frequency axis a delay lives on, which is why
 * fleet_coherent has to FIT a free constant per instance instead of DERIVING one from the ramp
 * across the fleet's ~106 channels (#52 in its original form). KV, 2026-08-14: "purge the idea
 * of summing across channels in each instance, that's *never* what we want to do." The summed
 * slots stay for now so the existing consumers keep working while the un-summed path is proven.
 *
 * A row is the TRACKER RECORD HEADER of gnssRecord.hpp verbatim -- byte-for-byte the airspy
 * single-antenna layout (`record_stride(0) == RECORD_FLOATS`). Deliberately not a new schema:
 * a translation table between the record slots and a wire struct is one more thing to get
 * subtly wrong, and slots keep being added (SKY, RES, TRIM_INC all arrived this month). The
 * ELEMENT BLOCKS are dropped -- 32 antennas x 12 floats is 94% of a CHORD record, the broker
 * closes no loop on them, and the per-element beam product is accumulated node-side and
 * written to disk there (CMB_ELEM_*). If a broker-side per-element product is ever wanted, it
 * is a second stream, not a fatter row.
 *
 * Record slot r of a frame holds the record whose window starts at
 *     wstart0 + r * hops_per_record * fft_len
 * ALWAYS -- the slot is addressed, not appended -- and `present` says which slots were filled.
 * A missing record is therefore a hole at a known place rather than a shift, which is what
 * makes a dropped record harmless instead of a silent re-pairing.
 *
 * SIZE / RATE (CHORD, 2026-08-14): 4 records/frame, 40 PRN rows, 26 floats
 *     = 112 + 4*40*50*4 = 32112 B per frame, 23.84 frames/s per (chain, instance)
 *     = 766 kB/s x 12 instances x 5 chains = 46 MB/s into the gather host.
 * (v1 was 24 MB/s with the comb summed away; carrying the comb costs 1.9x for the axis the
 * fleet combine actually needs.)
 * The rows are mostly zero for chains carrying fewer PRNs, and that is on purpose: ONE frame
 * size for every sender means ONE bufferRecv, one port, one buffer, and no per-chain plumbing
 * that could be wired up crooked. bufferRecv rejects a size mismatch by closing the connection,
 * so a stale sender fails loudly rather than corrupting the stream.
 */

#include "gnssRecord.hpp" // for gnss::RECORD_FLOATS

#include <cstddef> // for offsetof
#include <cstdint>
#include <cstring>
#include <string>

namespace gnss {

/// "GTL1" -- bumped only for an INCOMPATIBLE layout change; `version` covers the rest.
constexpr uint32_t TELEM_MAGIC = 0x314c5447u;
/// v5 (2026-08-16): RECORD_FLOATS 28 -> 29 for REC_PHI0, the comb's phase currency -- #72's
/// root cause. Without it the comb carries a per-instance arbitrary phase constant that no
/// consumer can undo, which is what held the fleet combine to within-instance coherence.
/// v4 (2026-08-16): RECORD_FLOATS 26 -> 28 for REC_ANG0 + REC_PHI_DDOP (#72). The row grew, so
/// the frame did: a v3 sender against a v4 gather (or the reverse) mis-strides every row, which
/// is why this is a version bump and not a quiet append -- the header's `n_row` check below
/// rejects the mismatch instead of reading plausible numbers at the wrong offsets.
/// v3 (2026-08-14): the comb carries EARLY, PROMPT and LATE per channel (CHAN_FLOATS
/// 3 -> 9), which is what lets the DLL move off the tracker's summed slots and so what
/// makes the sum deletable. Columns 0-2 keep their v2 meaning (prompt re, im, energy).
/// v2: the row carries the UNSUMMED COMB after the record header, and the wire
/// header carries the channels' freq_ids. ⚠️ KV: "purge the idea of summing across channels in
/// each instance, that's *never* what we want to do" -- the cross-channel sum destroys the
/// frequency axis a delay lives on, so the broker was left FITTING a per-instance constant
/// where it should DERIVE one from the ramp across ~106 channels.
constexpr uint16_t TELEM_VERSION = 5;

/// Comb columns reserved per row. Instances hold 6-7 covering channels today; the frame is a
/// fixed size for every sender (one bufferRecv, one buffer, no per-chain plumbing), so this is
/// the ceiling and the header's n_chan says how many are real.
constexpr int TELEM_MAX_CHAN = 8;

/// Hard ceiling on records batched into one wire frame. Not the configured value -- that is
/// `records_per_frame`, and it must divide the upstream frame's record count so a wire frame
/// never straddles two source frames.
constexpr int TELEM_MAX_REC = 8;

/// Fixed-width name fields. Names, not enums: an integer chain id is a lookup table that has to
/// be identical in the generator, the C++ and the broker, and the failure mode of a drifted one
/// is data attributed to the wrong constellation with nothing to show it.
constexpr int TELEM_NAME = 16;

/// Wire header size. Stated as a constexpr int (not just `sizeof`) so the CONFIG GENERATOR can
/// parse it out of this file the same way it parses RECORD_FLOATS -- the receive buffer's
/// frame_size is yaml and the wire format is C++, and config/gnss_record_layout.py exists
/// because that gap silently drifted once already (RECORD_FLOATS 24 -> 26, 34 stages dead at
/// construction). The static_assert below is what keeps the two honest.
constexpr int TELEM_HEADER_BYTES = 112;

/**
 * @struct TelemHeader
 * @brief Fixed 96-byte frame header. POD, little-endian x86-64 on both ends (the same
 *        assumption every other bufferSend user in this tree makes).
 */
struct TelemHeader {
    uint32_t magic;   ///< TELEM_MAGIC
    uint16_t version; ///< TELEM_VERSION
    uint16_t n_rec;   ///< record slots in this frame (constant per sender)
    uint16_t n_prn;   ///< PRN rows per record slot (constant per sender)
    uint16_t n_row;   ///< floats per row == gnss::RECORD_FLOATS at the SENDER's build
    uint16_t n_chan;  ///< covering channels this instance despreads (its share of the band)
    uint16_t n_elem;  ///< elements summed behind the header slots (provenance; rows are scalar)

    uint32_t hops_per_record; ///< record length in hops
    uint32_t fft_len;         ///< samples per hop -- with hops_per_record this converts
                              ///< wstart (samples) <-> hop, which is what every other GNSS
                              ///< product is keyed on (CMB_HOP_SLOT, seeds' ref_hop, the
                              ///< search). Carried rather than assumed: mixing a sample index
                              ///< with a hop index is a 16384x error that looks like a clock.

    /// THE COLLATION KEY. win = wstart0 / (n_rec * hops_per_record * fft_len), an exact integer
    /// division on the F-engine's global sample counter. Every sender computes it from the same
    /// three configured integers, so equal `win` IS the same sky with no tolerance -- unlike
    /// UTC, which is a stamp each instance derives independently.
    uint64_t win;

    /// Sender's own frame counter, monotone from stage start. Gaps mean this sender dropped or
    /// the link did; the broker can report loss without guessing from timestamps.
    uint64_t seq;

    int64_t wstart0; ///< absolute sample index of slot 0 == win * window_samples
    double utc0;     ///< UTC the assembler stamped on slot 0's record, or 0 if it had none.
                     ///< DIAGNOSTIC ONLY. Never a key: #46 measured 0.105 s of spread between
                     ///< instances on records with identical wstart.

    uint32_t present; ///< bit r set => record slot r was filled this frame

    uint16_t max_chan; ///< comb columns RESERVED per row (== TELEM_MAX_CHAN at the sender)
    uint16_t n_row_total; ///< floats per row == n_row + max_chan*CHAN_FLOATS. Derivable, but
                          ///< stated: a parser that computes a stride wrong reads plausible
                          ///< numbers at the wrong offsets rather than failing.

    char chain[TELEM_NAME]; ///< chain tag, NUL-padded ("gps_l5", "gal_e5a", ...)
    char inst[TELEM_NAME];  ///< instance tag, NUL-padded ("cx19.0" = node cx19, GPU 0)

    /// F-engine freq_id of each comb column, [0, n_chan) valid. ⚠️ WITHOUT THESE THE COMB IS
    /// USELESS: a delay is a phase ramp across FREQUENCY, so a fit over unlabelled columns --
    /// or worse, columns whose labels the consumer assumed -- returns a confident wrong tau.
    /// They ride on every frame rather than being configured into the broker, because a
    /// configured copy is one more thing that can drift out of step with the node it describes.
    uint16_t chan_id[TELEM_MAX_CHAN];
};

static_assert(sizeof(TelemHeader) == TELEM_HEADER_BYTES,
              "TelemHeader must stay TELEM_HEADER_BYTES -- it is a wire format, and the config "
              "generator sizes the receive buffer from that constant by parsing this file");
static_assert(offsetof(TelemHeader, win) == 24, "TelemHeader layout is a wire format");
static_assert(offsetof(TelemHeader, wstart0) == 40, "TelemHeader layout is a wire format");
static_assert(offsetof(TelemHeader, chain) == 64, "TelemHeader layout is a wire format");
static_assert(offsetof(TelemHeader, inst) == 80, "TelemHeader layout is a wire format");
static_assert(offsetof(TelemHeader, chan_id) == 96, "TelemHeader layout is a wire format");

/// Floats per row: the gnssRecord.hpp header, then the UNSUMMED COMB.
///
///     [ RECORD_FLOATS ][ CHAN_FLOATS x TELEM_MAX_CHAN ]
///
/// The record header keeps its summed prompt (slots 3/4) so nothing downstream has to change on
/// the same day; the comb is what the broker should actually combine, because only it carries
/// the frequency axis a delay lives on.
constexpr int TELEM_ROW_FLOATS = RECORD_FLOATS + TELEM_MAX_CHAN * CHAN_FLOATS;

/// Bytes of one wire frame carrying @c n_rec record slots of @c n_prn rows.
/// ⚠️ The senders and the gather's receive buffer MUST agree on this exactly; it is computed
/// once in the config generator from the same integers and written into both.
constexpr size_t telem_frame_bytes(int n_rec, int n_prn) {
    return sizeof(TelemHeader) + (size_t)n_rec * n_prn * TELEM_ROW_FLOATS * sizeof(float);
}

/// Float offset of PRN row @c p in record slot @c r, within the payload (i.e. AFTER the header).
constexpr size_t telem_row_offset(int r, int p, int n_prn) {
    return ((size_t)r * n_prn + p) * TELEM_ROW_FLOATS;
}

/// Float offset of comb column @c ch within a row (relative to the row's start).
constexpr size_t telem_chan_offset(int ch) {
    return (size_t)RECORD_FLOATS + (size_t)ch * CHAN_FLOATS;
}

/// The payload floats of a frame.
inline float* telem_rows(void* frame) {
    return (float*)((uint8_t*)frame + sizeof(TelemHeader));
}
inline const float* telem_rows(const void* frame) {
    return (const float*)((const uint8_t*)frame + sizeof(TelemHeader));
}

/// Copy a NUL-padded fixed-width name in. Truncates rather than overflowing; the caller checks
/// length at construction so truncation is never silent in practice.
inline void telem_set_name(char (&dst)[TELEM_NAME], const std::string& s) {
    std::memset(dst, 0, TELEM_NAME);
    std::memcpy(dst, s.data(), s.size() < TELEM_NAME ? s.size() : TELEM_NAME - 1);
}

} // namespace gnss

#endif // GNSS_TELEM_HPP
