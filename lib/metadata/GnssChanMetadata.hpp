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
#include <vector>   // for chan_scale
#include <memory>   // for shared_ptr

/**
 * @class GnssChanMetadata
 * @brief Carries the channelized GNSS stream's frame context: the absolute sample index of the
 *        frame's first hop (@c sample_seq), and -- for 4+4b-quantized frames -- the per-channel
 *        dequantization scales (@c chan_scale).
 *
 * The distributed search/track stages reference code phase to an absolute "sample
 * 0"; when subbands are shipped to other kotekan instances (bufferSend/Recv), that
 * reference has to ride the wire so every node agrees. chordMetadata is the
 * heavyweight CHORD array metadata (full dimension/frequency structure) and is
 * fragile to serialize when only one field is set -- so this purpose-built
 * metadata is used instead. Trivially (and robustly) serializable.
 *
 * chan_scale rides WITH the frame deliberately: the quantizer (GnssQuantize44) freezes its
 * per-channel gains once at startup, and a consumer that despreads the bytes must use exactly the
 * gains those bytes were encoded with -- carrying them on every frame makes a stale-gain despread
 * impossible by construction (a quantizer restart re-freezes and the new scales simply arrive
 * with the first re-scaled frame). Empty = the frame is not quantized (fp32), or the quantizer is
 * still in its bandpass-measurement window (during which it emits offset-zero bytes, which decode
 * to zero volts under ANY scale).
 */
class GnssChanMetadata : public metadataObject {
public:
    void deepCopy(std::shared_ptr<const metadataObject> other) override;
    size_t get_serialized_size() override;
    size_t set_from_bytes(const char* bytes, size_t length) override;
    size_t serialize(char* bytes) override;
    nlohmann::json to_json() override;

    int64_t sample_seq = -1; ///< absolute sample index of the frame's first hop (-1 = unset)
    /// 4+4b dequantization scales, lsb -> volts, one per frame channel (empty = fp32 frame or
    /// quantizer warmup). See GnssQuantize44 / gnss44.hpp for the format contract.
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
