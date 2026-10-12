#ifndef GNSS_TELEM_HPP
#define GNSS_TELEM_HPP
/**
 * @file gnssTelem.hpp
 * @brief Tracker -> broker telemetry wire format: one frame-synced record stream per
 *        (chain, instance), carried by kotekan's bufferSend/bufferRecv.
 *
 * Every frame carries its own address (chain, instance, and an absolute window on the
 * F-engine sample clock), so the broker never infers which instance or which window a value
 * describes. Two rules keep it that way:
 *
 * (1) Ship increments, never accumulations. Every float in a row is a per-record quantity; the
 *     two phase-advance slots (REC_CPHASE, REC_TRIM_INC) are increments since the PRN's previous
 *     record, and REC_PHI0 is the rotation applied to this record, wrapped to [-pi, pi] (see
 *     gnssRecord.hpp). Accumulation happens once, in the broker, over a run of records it
 *     can see is contiguous. An accumulator's origin is not on the wire, so no amount of
 *     transport synchronisation could repair it downstream.
 *
 * (2) Key on `win`/`wstart`, never on the receiver's frame order. bufferRecv makes no ordering
 *     promise across connections (its `frame_id` is a local ring index), and many senders share
 *     one listener, so collating on arrival order mis-pairs instances as soon as one drops a
 *     frame or restarts. The window index is computed from the absolute F-engine sample counter
 *     with a divisor every sender derives identically, so instances agree on it without
 *     negotiating.
 *
 * LAYOUT
 *
 *     [ TelemHeader, TELEM_HEADER_BYTES ][ float32 rows[n_rec][n_prn][n_row_total] ]
 *
 * and a row is
 *
 *     [ RECORD_FLOATS: the tracker record header ][ max_chan comb columns, CHAN_FLOATS each ]
 *
 * The row starts with the gnssRecord.hpp tracker record header verbatim, not a separate wire
 * schema, so slots added to the record need no translation table. The element blocks are not
 * shipped: the broker closes no loop on them, and the per-element products are accumulated on
 * the node (CMB_ELEM_*, the beam cube).
 *
 * The comb columns are the record's per-channel E, P and L (the gnssRecord.hpp chan block),
 * labelled by `chan_id`. The record header's prompt (slots 3/4) is summed over the instance's
 * covering channels, which loses the frequency axis a delay lives on; the comb keeps it, so the
 * broker can derive a delay from the phase ramp across the fleet's channels instead of fitting
 * a free constant per instance.
 *
 * Rows are not PRN slots: GnssTelemPack packs the PRNs despread in a window into the first
 * rows, with one PRN-to-row map for every record slot of the frame, and leaves the rest zero.
 * Read each row's PRN from REC_PRN.
 *
 * Record slot r of a frame holds the record whose window starts at
 *     wstart0 + r * hops_per_record * fft_len
 * always: the slot is addressed, not appended, and `present` says which slots were filled. A
 * missing record is a hole at a known place rather than a shift.
 *
 * The frame shape (n_rec, n_prn, max_chan) is the sender's own. The gather's single receive
 * buffer is sized for the widest sender and accepts shorter frames (bufferRecv
 * `allow_short_frames`); bufferRecv closes the connection of a sender whose frame does not fit,
 * so a mis-sized sender fails loudly rather than corrupting the stream.
 */

#include "gnssRecord.hpp" // for gnss::RECORD_FLOATS

#include <cstddef> // for offsetof
#include <cstdint>
#include <cstring>
#include <string>

namespace gnss {

/// "GTL1" in little-endian byte order.
constexpr uint32_t TELEM_MAGIC = 0x314c5447u;
/// Wire format version. GnssTelemGather and gnss::FleetDll accept only their own, and
/// python/scripts/gnss/gnss_broker/telem.py mirrors it. Bump it on any header or row layout
/// change, including a change to RECORD_FLOATS or CHAN_FLOATS: the row changes width, so nodes,
/// gather and broker move together.
constexpr uint16_t TELEM_VERSION = 8;

/// Comb columns a row may carry: the format's ceiling and the size of the header's chan_id
/// array, not the row stride. Each sender ships exactly the columns it despreads and declares
/// that in `max_chan`. The config generator sizes the gather's receive buffer with this ceiling
/// (parsed by config/gnss_record_layout.py), so senders of different widths share one listener.
constexpr int TELEM_MAX_CHAN = 8;

/// Ceiling on record slots per wire frame (GnssTelemPack's configured `records_per_frame`);
/// readers reject a header with more.
constexpr int TELEM_MAX_REC = 8;

/// Width of the chain and instance name fields. Names, not enums: an integer chain id is a
/// lookup table that must match in the generator, the C++ and the broker, and a mismatch
/// attributes data to the wrong constellation with nothing to show it.
constexpr int TELEM_NAME = 16;

/// Wire header size, stated as a literal (not just `sizeof`) so config/gnss_record_layout.py can
/// parse it out of this file: the receive buffer's frame_size is set in yaml, the wire format
/// here. The static_assert below ties it to sizeof(TelemHeader).
constexpr int TELEM_HEADER_BYTES = 112;

/**
 * @struct TelemHeader
 * @brief Fixed TELEM_HEADER_BYTES frame header. POD, little-endian x86-64 on both ends (the
 *        same assumption every other bufferSend user in this tree makes).
 */
struct TelemHeader {
    uint32_t magic;   ///< TELEM_MAGIC
    uint16_t version; ///< TELEM_VERSION
    uint16_t n_rec;   ///< record slots in this frame (constant per sender)
    uint16_t n_prn;   ///< PRN rows per record slot (constant per sender; see the file note)
    uint16_t n_row;   ///< record-header floats per row == gnss::RECORD_FLOATS at the sender's build
    uint16_t n_chan;  ///< comb columns carrying data: the instance's covering channels, 0 = no comb
    uint16_t n_elem;  ///< element count of the source record (provenance; rows have no elements)

    uint32_t hops_per_record; ///< record length in hops
    uint32_t fft_len;         ///< samples per hop. With hops_per_record it converts wstart
                              ///< (samples) to hops, which other GNSS products key on
                              ///< (CMB_HOP_SLOT, seeds' ref_hop). Carried rather than assumed:
                              ///< mixing a sample index with a hop index is an fft_len-fold
                              ///< error that looks like a clock error.

    /// The collation key. win = wstart0 / (n_rec * hops_per_record * fft_len), an exact integer
    /// division on the F-engine's global sample counter. Every sender computes it from the same
    /// three configured integers, so equal `win` is the same window with no tolerance, unlike
    /// UTC, which each instance stamps independently.
    uint64_t win;

    /// Sender's own frame counter, monotone from stage start. Gaps mean this sender dropped or
    /// the link did; the broker can report loss without guessing from timestamps.
    uint64_t seq;

    int64_t wstart0; ///< absolute sample index of slot 0 == win * window_samples
    double utc0;     ///< UTC the assembler stamped on slot 0's record, 0 if slot 0 is absent.
                     ///< Diagnostic only, never a key: each instance derives it from its own
                     ///< epoch (frame0_utc, else its host clock), so equal wstart need not give
                     ///< equal utc0.

    uint32_t present; ///< bit r set => record slot r was filled this frame

    uint16_t max_chan;    ///< comb columns per row from this sender (0 = no comb), at most
                          ///< TELEM_MAX_CHAN. Readers stride by it, never by the ceiling.
    uint16_t n_row_total; ///< floats per row == n_row + max_chan*CHAN_FLOATS. Derivable, but
                          ///< stated: a parser that computes a stride wrong reads plausible
                          ///< numbers at the wrong offsets rather than failing.

    char chain[TELEM_NAME]; ///< chain tag, NUL-padded ("gps_l5", "gal_e5a", ...)
    char inst[TELEM_NAME];  ///< instance tag, NUL-padded ("<node>.<gpu>")

    /// F-engine freq_id of each comb column, [0, n_chan) valid. A delay is a phase ramp across
    /// frequency, so a fit over unlabelled or assumed columns returns a confident wrong delay.
    /// The labels ride on every frame rather than being configured into the broker, so they
    /// cannot drift out of step with the node they describe.
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

/// Widest row the format admits: a ceiling, not the stride. Every sender declares its own row
/// width in `max_chan`/`n_row_total`, and a reader that strides by this constant walks a
/// narrower sender's frame at the wrong offsets. Use telem_row_floats(h.max_chan).
constexpr int TELEM_MAX_ROW_FLOATS = RECORD_FLOATS + TELEM_MAX_CHAN * CHAN_FLOATS;

/// Floats per row for a sender carrying @c n_cols comb columns (0 when the comb is off).
constexpr int telem_row_floats(int n_cols) {
    return RECORD_FLOATS + n_cols * CHAN_FLOATS;
}

/// Bytes of one wire frame carrying @c n_rec record slots of @c n_prn rows of @c n_cols columns.
/// The shape is the sender's own: an instance despreading one channel ships one comb column.
constexpr size_t telem_frame_bytes(int n_rec, int n_prn, int n_cols) {
    return sizeof(TelemHeader) + (size_t)n_rec * n_prn * telem_row_floats(n_cols) * sizeof(float);
}

/// Float offset of PRN row @c p in record slot @c r, within the payload (after the header).
/// @c row_floats is the sender's stride (header n_row_total), never the compile-time ceiling.
constexpr size_t telem_row_offset(int r, int p, int n_prn, int row_floats) {
    return ((size_t)r * n_prn + p) * row_floats;
}

/// Float offset of comb column @c ch within a row (relative to the row's start).
constexpr size_t telem_chan_offset(int ch) {
    return (size_t)RECORD_FLOATS + (size_t)ch * CHAN_FLOATS;
}

/// Bytes this header says its own frame is. A consumer reading a frame out of a buffer sized for
/// the widest sender takes the frame length from here, never from the buffer's frame_size.
inline size_t telem_frame_bytes(const TelemHeader& h) {
    return telem_frame_bytes(h.n_rec, h.n_prn, h.max_chan);
}

/// Does this header describe a self-consistent frame of @c bytes? The shape travels on the wire
/// and the payload is walked with it, so a reader that skips this walks off the end of a short
/// frame instead of rejecting it.
inline bool telem_shape_ok(const TelemHeader& h, size_t bytes) {
    return h.max_chan <= TELEM_MAX_CHAN && h.n_row == RECORD_FLOATS
           && h.n_row_total == telem_row_floats(h.max_chan) && h.n_chan <= h.max_chan && h.n_rec > 0
           && h.n_rec <= TELEM_MAX_REC && h.n_prn > 0
           && telem_frame_bytes(h.n_rec, h.n_prn, h.max_chan) == bytes;
}

/// The payload floats of a frame.
inline float* telem_rows(void* frame) {
    return (float*)((uint8_t*)frame + sizeof(TelemHeader));
}
inline const float* telem_rows(const void* frame) {
    return (const float*)((const uint8_t*)frame + sizeof(TelemHeader));
}

/// Copy a name into a NUL-padded fixed-width field, truncating to TELEM_NAME - 1 chars.
/// GnssTelemPack rejects longer names at construction.
inline void telem_set_name(char (&dst)[TELEM_NAME], const std::string& s) {
    std::memset(dst, 0, TELEM_NAME);
    std::memcpy(dst, s.data(), s.size() < TELEM_NAME ? s.size() : TELEM_NAME - 1);
}

} // namespace gnss

#endif // GNSS_TELEM_HPP
