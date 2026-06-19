/**
 * @file
 * @brief Merge per-worker GNSS record streams into one unified stream.
 *  - GnssRecordCollector : public kotekan::Stage
 */

#ifndef GNSS_RECORD_COLLECTOR_HPP
#define GNSS_RECORD_COLLECTOR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssRecordCollector
 * @brief Concatenates one frame from each worker's record buffer into a single
 *        output frame, per measurement window.
 *
 * The data-plane side of the broker: several @ref GnssChannelizedCorrelator
 * workers (a PRN/band split) each emit a record frame per window; this stage
 * gathers the i-th frame from every input and writes them back to back, giving
 * one record stream for the downstream writer/visibility tools. Workers run in
 * lockstep on the same channelized input, so their i-th frames are the same
 * window. The stage is record-format agnostic -- it copies bytes -- so the
 * output frame size must equal the sum of the input frame sizes.
 *
 * @par Buffers
 * @buffer in_bufs Array of per-worker record buffers.
 *     @buffer_format float records (e.g. GpsReplicaCorrelator layout)
 *     @buffer_metadata none
 * @buffer out_buf Merged records (concatenation of the input frames).
 *     @buffer_format float records
 *     @buffer_metadata none
 *
 * @author Keith Vanderlinde
 */
class GnssRecordCollector : public kotekan::Stage {
public:
    GnssRecordCollector(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~GnssRecordCollector() override = default;

    void main_thread() override;

private:
    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
};

#endif // GNSS_RECORD_COLLECTOR_HPP
