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
 * @brief NODE SIDE of the frame-synced tracker->broker gather (task #59): tracker record frames
 *        -> fixed-size telemetry frames addressed on the F-engine sample clock.
 *
 * Reads the SAME record buffer the combiner reads -- as a second, independent consumer, so
 * nothing about the existing chain changes -- strips each record down to its gnssRecord.hpp
 * HEADER (the element blocks are 94% of a CHORD record and the broker closes no loop on them),
 * and batches @c records_per_frame records into one wire frame for bufferSend.
 *
 * ⚠️ THE BATCH BOUNDARY IS ABSOLUTE, NOT LOCAL. A frame closes when the record's
 *     win = wstart / (records_per_frame * hops_per_record * fft_len)
 * changes, i.e. on the F-engine's own global sample counter, NOT after N records have been
 * seen locally. Those are different: a local counter starts wherever the stage started and
 * drifts on every dropped record, so two instances would batch different record sets and the
 * receiver would have to work out the offset -- which is exactly the inference this whole
 * change exists to delete (it is #53's defect, in a new place). Derived from three integers
 * every sender is given identically, the boundary needs no negotiation and cannot drift.
 *
 * Record slot r within the frame is ADDRESSED, never appended:
 *     slot r <=> wstart == win*window_samples + r*hops_per_record*fft_len
 * so a dropped record leaves a hole at a known index (reported in @c present) instead of
 * shifting everything after it by one.
 *
 * @par buffers
 * @buffer in_buf  tracker record frames (one per record; GnssChanMetadata::sample_seq = wstart)
 * @buffer out_buf telemetry frames, exactly gnss::telem_frame_bytes(records_per_frame, max_prn)
 *
 * @conf chain             String. Chain tag carried on every frame ("gps_l5", "gal_e5a", ...).
 * @conf inst              String. Instance tag ("cx19.0" = node cx19, GPU 0). Must be unique
 *                           across the fleet -- it is how the broker tells senders apart.
 * @conf n_prn             Int. PRN slots in the SOURCE record frame.
 * @conf n_elements        Int, default 0. Element blocks in the source record (for the stride).
 * @conf max_prn           Int. PRN rows on the wire; >= n_prn, identical on every sender.
 * @conf records_per_frame Int, default 4. Records batched per wire frame; must divide the
 *                           upstream frame's record count so a wire frame never straddles two.
 * @conf hops_per_record   Int. Record length in hops.
 * @conf fft_len           Int. Samples per hop.
 * @conf n_chan            Int, default 0. Covering channels this instance despreads (metadata
 *                           only -- it is the instance's share of the band, which the broker
 *                           uses as a combining weight sanity check).
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
    int _n_chan = 0;

    /// records_per_frame * hops_per_record * fft_len -- the absolute window length in SAMPLES.
    int64_t _win_samples = 0;
    /// hops_per_record * fft_len -- one record in samples; the slot quantum.
    int64_t _rec_samples = 0;

    // The frame being accumulated. -1 = none open.
    int64_t _cur_win = -1;
    uint32_t _cur_present = 0;
    double _cur_utc0 = 0.0;
    std::vector<float> _rows; ///< [rec_per_frame][max_prn][RECORD_FLOATS]

    uint64_t _seq = 0;      ///< frames emitted; the receiver reads gaps as loss
    uint64_t _dropped = 0;  ///< records that could not be placed (no metadata / bad wstart)
    uint64_t _n_records = 0;

    /// Write the accumulated window out and reset. No-op if nothing is open.
    /// Returns false if the output buffer shut down (the caller must exit).
    bool flush(frameID& out_id);
};

#endif // GNSS_TELEM_PACK_HPP
