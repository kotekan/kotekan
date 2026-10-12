/**
 * @file
 * @brief Minimal metadata for the channelized GNSS streams.
 *  - GnssChanMetadata : public metadataObject
 */

#ifndef GNSS_CHAN_METADATA_HPP
#define GNSS_CHAN_METADATA_HPP

#include "buffer.hpp"   // for Buffer, GenericBuffer
#include "metadata.hpp" // for metadataObject

#include "json.hpp" // for json

#include <cstdint> // for int64_t
#include <memory>  // for shared_ptr
#include <vector>  // for chan_scale

/**
 * @class GnssChanMetadata
 * @brief Frame context for the channelized GNSS streams: the absolute sample index of the
 *        frame's first hop (@c sample_seq) and optional per-channel 4+4b dequantization scales
 *        (@c chan_scale).
 *
 * The search and track stages reference code phase to an absolute sample 0, so that reference
 * travels with each frame, including between kotekan instances (bufferSend/bufferRecv), and
 * every node agrees on it. chordMetadata carries the whole CHORD array structure; this type
 * carries only these fields and serializes trivially.
 *
 * Scales travel with the frame because a consumer must decode the bytes with exactly the
 * scales they were encoded with.
 */
class GnssChanMetadata : public metadataObject {
public:
    void deepCopy(std::shared_ptr<const metadataObject> other) override;
    size_t get_serialized_size() override;
    size_t set_from_bytes(const char* bytes, size_t length) override;
    size_t serialize(char* bytes) override;
    nlohmann::json to_json() override;

    int64_t sample_seq = -1; ///< absolute sample index of the frame's first hop (-1 = unset)
    /// 4+4b dequantization scales, lsb -> volts, one per frame channel; empty if none were set.
    /// See gnss44.hpp for the format.
    std::vector<float> chan_scale;
};

/// The GnssChanMetadata of a frame, or nullptr if it has none.
inline GnssChanMetadata* get_gnss_chan_metadata(Buffer* buf, int frame_id) {
    return dynamic_cast<GnssChanMetadata*>(buf->get_metadata(frame_id).get());
}

/// True if the buffer's metadata pool is GnssChanMetadata.
inline bool metadata_is_gnss_chan(Buffer* buf) {
    return buf && buf->metadata_pool && buf->metadata_pool->type_name == "GnssChanMetadata";
}

#endif // GNSS_CHAN_METADATA_HPP
