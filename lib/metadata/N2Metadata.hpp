#ifndef N2_METADATA
#define N2_METADATA

#include "N2Metadata.hpp"
#include "metadata.hpp"     // for metadataObject, metadataPool
#include "chordMetadata.hpp"// for chordMetadata
#include "N2Util.hpp"       // for frameID
#include "buffer.hpp"       // for Buffer
#include "Config.hpp"       // for Config
#include "N2Util.hpp"       // for get_num_prod

using kotekan::Config;

#include <assert.h>

// Struct containing metadata fields for an N2 frame
struct N2MetadataFormat {

    /// Number of elements for data in buffer
    uint32_t num_elements;
    /// Number of eigenvectors and values calculated
    uint32_t num_ev;
    /// Number of products in the data
    uint32_t num_prod;

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

inline std::shared_ptr<N2Metadata> alloc_N2_from_chord_metadata(Buffer* chord_buf, size_t chord_frame_id,
    Buffer* N2_buf, N2::frameID N2_frame_id, Config& config, const std::string& unique_name, int f) {
    
    assert(f >= 0 && f < CHORD_META_MAX_FREQ);
    
    N2_buf->allocate_new_metadata_object(N2_frame_id);

    std::shared_ptr<chordMetadata> chord_meta = get_chord_metadata(chord_buf, chord_frame_id);
    std::shared_ptr<N2Metadata> N2_meta = get_N2_metadata(N2_buf, N2_frame_id);

    N2_meta->num_elements = config.get<int32_t>(unique_name, "num_elements");
    N2_meta->num_prod = N2::get_num_prod(N2_meta->num_elements);
    N2_meta->num_ev = config.get<int32_t>(unique_name, "num_ev");

    N2_meta->freq_id = chord_meta->coarse_freq[f];
    N2_meta->fpga_start_tick = 0;
    N2_meta->frame_start_ctime = {0, 0};
    N2_meta->frame_length_fpga_ticks = 0;
    N2_meta->n_valid_fpga_ticks_in_frame = 0;
    N2_meta->n_rfi_fpga_ticks = 0;

    return N2_meta;
}

#endif
