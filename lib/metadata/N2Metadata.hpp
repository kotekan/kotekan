#ifndef N2_METADATA
#define N2_METADATA

#include "CHORDTelescope.hpp" // for EOP
#include "Config.hpp"         // for Config
#include "N2Metadata.hpp"
#include "buffer.hpp"         // for Buffer
#include "kotekanLogging.hpp" // for WARN_NON_OO
#include "metadata.hpp"       // for metadataObject, metadataPool

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

    /// Number of elements for data in buffer
    uint32_t num_elements;
    /// Number of products in the data
    uint32_t num_prod;
    /// Number of eigenvectors and values calculated
    uint32_t num_ev;
    /// Total number of frequencies in pipeline
    uint32_t nfreq;

    /// ID of the frequency bin
    uint32_t freq_id; // this is an int in chordMetadata, maybe change later
    /// Physical frequency in Hz
    double freq_Hz;

    /// Frame Earth Orientation Paramters
    struct EOP frame_eop;
    /// Bin Earth Orientation Parameters
    struct EOP bin_eop;

    /// Absolute frame index since start of observation
    uint64_t abs_frame_index;

    /// The sequence number of the first FPGA frame integrated into this visibility frame
    uint64_t fpga_start_tick;
    /// The time of the start of the integration frame in nanosec
    uint64_t frame_start_time_ns;
    /// Nominal length of the frame in FPGA ticks
    uint64_t frame_length_fpga_ticks;
    /// Amount of data that actually went into the frame (in FPGA ticks)
    uint64_t n_valid_fpga_ticks;
    /// The number of FPGA frames flagged as containing RFI. NOTE: This value
    /// might contain overlap with lost samples, as that counts missing samples
    /// as well as RFI. For renormalization this value should NOT be used, use
    /// lost samples (= @c frame_length_fpga_ticks - @c n_valid_fpga_ticks) instead.
    uint64_t n_rfi_fpga_ticks;
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
