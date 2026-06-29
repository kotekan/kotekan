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
#include <cstdint>  // for int64_t
#include <memory>   // for shared_ptr

/**
 * @class GnssChanMetadata
 * @brief Carries one value across the GNSS channelized pipeline: the absolute
 *        sample index of the frame's first hop (@c sample_seq).
 *
 * The distributed search/track stages reference code phase to an absolute "sample
 * 0"; when subbands are shipped to other kotekan instances (bufferSend/Recv), that
 * reference has to ride the wire so every node agrees. chordMetadata is the
 * heavyweight CHORD array metadata (full dimension/frequency structure) and is
 * fragile to serialize when only one field is set -- so this purpose-built 8-byte
 * metadata is used instead. Trivially (and robustly) serializable.
 */
class GnssChanMetadata : public metadataObject {
public:
    void deepCopy(std::shared_ptr<const metadataObject> other) override;
    size_t get_serialized_size() override;
    size_t set_from_bytes(const char* bytes, size_t length) override;
    size_t serialize(char* bytes) override;
    nlohmann::json to_json() override;

    int64_t sample_seq = -1; ///< absolute sample index of the frame's first hop (-1 = unset)
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
