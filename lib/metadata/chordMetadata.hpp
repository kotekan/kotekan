#ifndef CHORD_METADATA
#define CHORD_METADATA

#include "Telescope.hpp"
#include "buffer.hpp"
#include "metadata.hpp"

#include <assert.h>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

// One of the warning-silencing pragmas below only applied for gcc >= 8
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#pragma pack()

enum chordDataType {
    unknown_type,
    uint4p4,
    uint8,
    uint16,
    uint32,
    uint64,
    int4p4,
    int4p4chime, // offset-encoded (stored is value + 8), low and high values swapped
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64
};

constexpr std::size_t chord_datatype_bytes(chordDataType type) {
    switch (type) {
        case uint4p4:
            return 1;
        case uint8:
            return 1;
        case uint16:
            return 2;
        case uint32:
            return 4;
        case uint64:
            return 8;
        case int4p4:
            return 1;
        case int4p4chime:
            return 1;
        case int8:
            return 1;
        case int16:
            return 2;
        case int32:
            return 4;
        case int64:
            return 8;
        case float16:
            return 2;
        case float32:
            return 4;
        case float64:
            return 8;
        case unknown_type:
        default:
            return -1;
    }
}

const char* chord_datatype_string(chordDataType type);
chordDataType chord_datatype_from_string(const std::string& type);

// Maximum number of frequencies in metadata array
const int CHORD_META_MAX_FREQ = 1024;

// Maximum number of dimensions for arrays
const int CHORD_META_MAX_DIM = 10;

// Maximum length of dimension names for arrays
const int CHORD_META_MAX_DIMNAME = 20;

// Maximum number of visibility matrix samples in a frame
const int CHORD_META_MAX_VIS_SAMPLES = 64;

class chordMetadata : public metadataObject {
public:
    chordMetadata();

    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;

    /// The ICEBoard sequence number
    int64_t fpga_seq_num;
    /// The system time when the first packet in the frame was captured
    struct timeval first_packet_recv_time;
    /// The GPS time of @c fpga_seq_num.
    struct timespec gps_time;
    /// The stream ID from the ICEBoard
    /// Note in the case of CHIME-2048 the normally unused section
    /// Encodes the port-shuffle frequency information
    uint16_t stream_ID;

    int frame_counter;

    char name[CHORD_META_MAX_DIMNAME]; // "E", "J", "I", etc
    chordDataType type;

    /// Track the number of lost fpga samples in each gpu sub-integration
    int lost_fpga_samples[CHORD_META_MAX_FREQ][CHORD_META_MAX_VIS_SAMPLES];
    /// Track the number of rfi-flagged samples in each gpu sub-integration
    int rfi_flagged_samples[CHORD_META_MAX_FREQ][CHORD_META_MAX_VIS_SAMPLES];

    int dims;
    int dim[CHORD_META_MAX_DIM];
    char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME]; // "F", "T", "D", etc
    // The stride counts elements, not bytes
    int64_t stride[CHORD_META_MAX_DIM];
    // The offset counts elements, not bytes
    int64_t offset;

    // One-hot arrays?
    int n_one_hot;
    char onehot_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME];
    int onehot_index[CHORD_META_MAX_DIM];

    // All time samples in this buffer (or the whole buffer, if the
    // buffer does not have a time sample index) have `sample_offset`
    // added to the buffer's time sample index. (This allows quickly
    // shifting metadata in time to re-use metadata objects.)
    //
    // The actual (possibly fractional) time sample index is calculated as follows:
    //     T_actual = (sample0_offset + T + half_fpga_sample0[F] / 2) / time_downsampling_fpga[F]
    // where `T` is the time sample index and `F` is the coarse frequency index.
    int64_t sample0_offset;

    // Number of bytes per time sample
    size_t sample_bytes() const {
        // The number of bytes per sample is the number of bytes needed to store one array slice.
        return chord_datatype_bytes(type) * stride[0];
    }

    // Per-frequency arrays

    // Number of coarse frequency channels in this frame, or -1. The
    // actual number of frequencies will be larger after
    // upchannelization. This field continues to track the original
    // number of coarse frequency channels.
    int nfreq;

    // frequencies -- integer (0-8192) identifier for FPGA coarse frequencies
    // This is the FPGA frequency channel index, indexed by the local coarse frequency channel.
    int coarse_freq[CHORD_META_MAX_FREQ];

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    // Also indexed by the local coarse frequency channel.
    int freq_upchan_factor[CHORD_META_MAX_FREQ];

    // Time sampling -- for each coarse frequency channel, 2x the FPGA
    // sample number of the first sample.  The 2x is there to handle
    // the upchannelization case, where 2 or more samples may get
    // averaged, producing a new sample that is effectively halfway in
    // between them, ie, at a half-FPGAsample time.
    int64_t half_fpga_sample0[CHORD_META_MAX_FREQ];

    // Time sampling -- for each coarse frequency channel, the factor
    // by which the time samples have been downsampled relative to
    // FPGA samples.
    int time_downsampling_fpga[CHORD_META_MAX_FREQ];

    // Dish layout
    int ndishes;                                  // number of dishes
    int n_dish_locations_ew, n_dish_locations_ns; // number of possible dish locations
    int* dish_index; // [non-owning pointer] dish index for a possible dish location, or -1
    int get_dish_index(int dish_loc_ew, int dish_loc_ns) const {
        // The east-west dish index runs faster because this is the
        // convenient way to specify dish indices in a YAML file
        assert(dish_loc_ew >= 0 && dish_loc_ew < n_dish_locations_ew);
        assert(dish_loc_ns >= 0 && dish_loc_ns < n_dish_locations_ns);
        return dish_index[dish_loc_ew + n_dish_locations_ew * dish_loc_ns];
    }

    std::string get_dimension_name(size_t i) const {
        return std::string(dim_name[i], strnlen(dim_name[i], CHORD_META_MAX_DIMNAME));
    }

    std::string get_type_string() const {
        return std::string(chord_datatype_string(type));
    }

    std::string get_dimensions_string() const {
        std::ostringstream s;
        for (int i = 0; i < this->dims; i++) {
            if (i)
                s << " x ";
            s << get_dimension_name(i) << "(" << dim[i] << ")";
        }
        return s.str();
    }

    std::string get_onehot_name(size_t i) const {
        return std::string(onehot_name[i], strnlen(onehot_name[i], CHORD_META_MAX_DIMNAME));
    }

    std::string get_onehot_string() const {
        std::ostringstream s;
        for (int i = 0; i < this->n_one_hot; i++) {
            if (i)
                s << ", ";
            s << get_onehot_name(i) << "=" << onehot_index[i];
        }
        return s.str();
    }

    void set_array_dimension(int dim, int size, const std::string& name) {
        assert(dim < CHORD_META_MAX_DIM);
        this->dim[dim] = size;
        // GCC helpfully tries to warn us that the destination string may end up not
        // NUL-terminated, which we know.
#pragma GCC diagnostic push
#if GCC_VERSION > 80000
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
        strncpy(this->dim_name[dim], name.c_str(), CHORD_META_MAX_DIMNAME);
#pragma GCC diagnostic pop
    }

    void set_name(const std::string& name) {
        // GCC helpfully tries to warn us that the destination string may end up not
        // NUL-terminated, which we know.
#pragma GCC diagnostic push
#if GCC_VERSION > 80000
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
        strncpy(this->name, name.c_str(), CHORD_META_MAX_DIMNAME);
#pragma GCC diagnostic pop
    }

    std::string get_name() const {
        return std::string(name, strnlen(name, CHORD_META_MAX_DIMNAME));
    }

    void set_onehot_dimension(int dim, int i, const std::string& name) {
        assert(dim < CHORD_META_MAX_DIM);
        this->onehot_index[dim] = i;
        strncpy(this->onehot_name[dim], name.c_str(), CHORD_META_MAX_DIMNAME);
    }
};

inline bool metadata_is_chord(Buffer* buf, int) {
    return buf && buf->metadata_pool && (buf->metadata_pool->type_name == "chordMetadata");
}

inline bool metadata_is_chord(const std::shared_ptr<metadataObject>& mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "chordMetadata");
}

inline bool metadata_is_chord(const std::shared_ptr<const metadataObject>& mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "chordMetadata");
}

inline std::shared_ptr<chordMetadata>
get_chord_metadata(const std::shared_ptr<metadataObject>& mc) {
    if (!mc)
        return std::shared_ptr<chordMetadata>();
    if (!metadata_is_chord(mc)) {
        std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
        WARN_NON_OO("Expected metadata to be type \"chordMetadata\", got \"{:s}\".",
                    pool->type_name);
        return std::shared_ptr<chordMetadata>();
    }
    return std::static_pointer_cast<chordMetadata>(mc);
}

inline std::shared_ptr<const chordMetadata>
get_chord_metadata(const std::shared_ptr<const metadataObject>& mc) {
    if (!mc)
        return std::shared_ptr<const chordMetadata>();
    if (!metadata_is_chord(mc)) {
        std::shared_ptr<const metadataPool> pool = mc->parent_pool.lock();
        WARN_NON_OO("Expected metadata to be type \"chordMetadata\", got \"{:s}\".",
                    pool->type_name);
        return std::shared_ptr<const chordMetadata>();
    }
    return std::static_pointer_cast<const chordMetadata>(mc);
}

inline std::shared_ptr<chordMetadata> get_chord_metadata(Buffer* buf, int frame_id) {
    if (!buf || frame_id < 0 || frame_id >= (int)buf->metadata.size())
        return std::shared_ptr<chordMetadata>();
    std::shared_ptr<metadataObject> meta = buf->metadata[frame_id];
    return get_chord_metadata(meta);
}

#endif
