#ifndef N2_METADATA
#define N2_METADATA

#include "Config.hpp"   // for Config
#include "N2Layout.hpp" // for N2Layout
#include "N2Metadata.hpp"
#include "buffer.hpp"         // for Buffer
#include "kotekanLogging.hpp" // for WARN_NON_OO
#include "metadata.hpp"       // for metadataObject, metadataPool
#include "timeUtil.hpp"       // for EOP

#include "fmt.hpp"  // for compile_string_to_view
#include "json.hpp" // for json

using kotekan::Config;

#include <assert.h> // for assert
#include <memory>   // for shared_ptr, __shared_ptr_access, allocator, static_pointer...
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint64_t
#include <string>   // for operator==, char_traits, basic_string
#include <vector>   // for vector

// Struct containing metadata fields for an N2 frame
struct N2MetadataFormat {
    /// ID of the frequency bin
    uint32_t freq_id = 0; // this is an int in chordMetadata, maybe change later
    /// Physical frequency in Hz
    double freq_MHz = 0.0;

    /// absolute time index of this frame in its stream. Begins at 0 at instument start
    /// and counts monitonically afterwards.
    uint64_t abs_time_idx = 0;

    /// Earth Orientation Parameters at the center of the integration/accumulation time
    struct EOP time_center_eop = eop_null;
    /// Earth Orientation Parameters within an integration/accumulation bin
    struct EOP bin_eop = eop_null;
    // bin start/end info for convenience
    double bin_start_ERA_deg = 0.0; /// Earth Rotation Angle at start of bin
    double bin_end_ERA_deg = 0.0;   /// Earth Rotation Angle at end of bin
    double bin_start_LAST = 0.0;    /// local apparent sidereal time (nanoseconds) at start of bin
    double bin_end_LAST = 0.0;      /// local apparent sidereal time (nanoseconds) at end of bin

    /// The sequence number of the first FPGA frame integrated into this visibility frame
    uint64_t fpga_start_tick = 0;
    /// The time of the start of the integration frame in nanosec
    uint64_t frame_start_time_ns = 0;
    /// Nominal length of the frame in FPGA ticks
    uint64_t frame_length_fpga_ticks = 0;
    /// Amount of data that actually went into the frame (in FPGA ticks)
    uint64_t n_valid_fpga_ticks = 0;
    /// The number of FPGA frames flagged as containing RFI. NOTE: This value
    /// might contain overlap with lost samples, as that counts missing samples
    /// as well as RFI. For renormalization this value should NOT be used, use
    /// lost samples (= @c frame_length_fpga_ticks - @c n_valid_fpga_ticks) instead.
    uint64_t n_rfi_fpga_ticks = 0;
};

class N2Metadata : public metadataObject, public N2MetadataFormat {
public:
    N2Metadata();

    // ASSUMES the "other" is my type!
    void deepCopy(std::shared_ptr<const metadataObject> other) override;

    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;

    nlohmann::json to_json() override;

    /// Validates that the frame descriptor is an N2FrameDesc and checks internal metadata
    /// consistency (Currently, just valid/RFI ticks <= frame length). Issues a non-fatal error
    /// for any inconsistencies.
    void check_frame_desc(const std::shared_ptr<const kotekan::FrameDesc>& frame_desc) const;
};

void to_json(nlohmann::json& j, const N2Metadata& m);
void from_json(const nlohmann::json& j, N2Metadata& m);

inline bool metadata_is_N2(Buffer* buf, int) {
    return buf && buf->metadata_pool && (buf->metadata_pool->type_name == "N2Metadata");
}

inline bool metadata_is_N2(const std::shared_ptr<const metadataObject> mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "N2Metadata");
}

inline bool metadata_is_N2(const std::shared_ptr<metadataObject> mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "N2Metadata");
}

inline std::shared_ptr<N2Metadata> get_N2_metadata(const std::shared_ptr<metadataObject> mc) {
    if (!mc)
        return std::shared_ptr<N2Metadata>();
    if (!metadata_is_N2(mc)) {
        std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
        WARN_NON_OO("Expected metadata to be type \"N2Metadata\", got \"{:s}\".", pool->type_name);
        return std::shared_ptr<N2Metadata>();
    }

    return std::static_pointer_cast<N2Metadata>(mc);
}

inline std::shared_ptr<const N2Metadata>
get_N2_metadata(const std::shared_ptr<const metadataObject>& mc) {
    if (!mc)
        return std::shared_ptr<const N2Metadata>();
    if (!metadata_is_N2(mc)) {
        std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
        WARN_NON_OO("Expected metadata to be type \"N2Metadata\", got \"{:s}\".", pool->type_name);
        return std::shared_ptr<const N2Metadata>();
    }

    return std::static_pointer_cast<const N2Metadata>(mc);
}

inline std::shared_ptr<N2Metadata> get_N2_metadata(Buffer* buf, int frame_id) {
    if (!buf || frame_id < 0 || frame_id >= (int)buf->metadata.size())
        return std::shared_ptr<N2Metadata>();
    std::shared_ptr<metadataObject> meta = buf->metadata[frame_id];
    return get_N2_metadata(meta);
}

#endif
