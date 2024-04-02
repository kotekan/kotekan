#ifndef N2_METADATA
#define N2_METADATA

#include "N2Metadata.hpp"
#include "buffer.hpp"
#include "metadata.hpp"

#include <assert.h>

// Struct containing metadata fields for an N2 frame
struct N2MetadataFormat {

    /// Number of elements for data in buffer
    uint32_t num_elements;
    /// Number of products for data in buffer
    uint32_t num_prod;
    /// Number of eigenvectors and values calculated
    uint32_t num_ev;

    /// ID of the frequency bin
    int freq_id; // this is an int in chordMetadata, maybe change later

    /// The sequence number of the first FPGA frame integrated into this visibility frame
    uint64_t fpga_start_tick;
    /// The ctime of the integration frame
    timespec frame_start_ctime;
    /// Nominal length of the frame in FPGA ticks
    uint64_t frame_length_fpga_ticks;
    /// Amount of data that actually went into the frame (in FPGA ticks)
    uint64_t n_valid_fpga_ticks_in_frame;
    /// The number of FPGA frames flagged as containing RFI. NOTE: This value
    /// might contain overlap with lost samples, as that counts missing samples
    /// as well as RFI. For renormalization this value should NOT be used, use
    /// lost samples (= @c frame_length_fpga_ticks - @c n_valid_fpga_ticks_in_frame) instead.
    uint64_t n_rfi_fpga_ticks;
};

class N2Metadata :
    public metadataObject, public N2MetadataFormat {
public:
    N2Metadata();

    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;
};


inline bool metadata_is_N2(Buffer* buf, int) {
    return buf && buf->metadata_pool && (buf->metadata_pool->type_name == "N2Metadata");
}

inline bool metadata_is_N2(const std::shared_ptr<metadataObject> mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "N2Metadata");
}

inline std::shared_ptr<N2Metadata> get_N2_metadata(std::shared_ptr<metadataObject> mc) {
    if (!mc)
        return std::shared_ptr<N2Metadata>();
    if (!metadata_is_N2(mc)) {
        std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
        WARN_NON_OO("Expected metadata to be type \"N2Metadata\", got \"{:s}\".",
                    pool->type_name);
        return std::shared_ptr<N2Metadata>();
    }

    return std::static_pointer_cast<N2Metadata>(mc);
}

inline std::shared_ptr<N2Metadata> get_N2_metadata(Buffer* buf, int frame_id) {
    if (!buf || frame_id < 0 || frame_id >= (int)buf->metadata.size())
        return std::shared_ptr<N2Metadata>();
    std::shared_ptr<metadataObject> meta = buf->metadata[frame_id];
    return get_N2_metadata(meta);
}

#endif
