#ifndef CHORD_METADATA
#define CHORD_METADATA

#include "DataType.hpp"       // for type_to_string, type_total_bytes, DataType
#include "Telescope.hpp"
#include "buffer.hpp"         // for Buffer
#include "kotekanLogging.hpp" // for WARN_NON_OO
#include "metadata.hpp"       // for metadataObject, metadataPool
#include "jsonMetadata.hpp"
// TODO: CHIME and CHORD differ whether they use the datasetManager
#include "dataset.hpp"

#include "fmt.hpp" // for compile_string_to_view

#include <atomic>
#include <cassert>    // for assert
#include <cstddef>    // for size_t, ptrdiff_t
#include <cstdint>    // for int64_t, uint16_t
#include <memory>     // for shared_ptr, allocator, __shared_ptr_access, static_pointer...
#include <sstream>    // for basic_ostream, operator<<, basic_ostringstream, basic_ostr...
#include <string.h>   // for strncpy, strnlen, size_t
#include <string>     // for char_traits, basic_string, string, operator==, operator<<
#include <sys/time.h> // for timeval
#include <time.h>     // for size_t, timespec
#include <vector>     // for vector

// One of the warning-silencing pragmas below only applied for gcc >= 8
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#pragma pack()

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

    int frame_counter;

    // TODO: Replace by NDArray
    char name[CHORD_META_MAX_DIMNAME]; // "E", "J", "I", etc
    kotekan::DataType type;

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
    //     T_actual = (sample0_offset + T / offset_downsampling + half_fpga_sample0[F] / 2) /
    //                time_downsampling_fpga[F]
    // where `T` is the time sample index (the slowest varying index)
    // and `F` is the coarse frequency index.
    int64_t sample0_offset;
    int offset_downsampling;

    size_t sample_bytes() const {
        // The number of bytes per sample is the number of bytes needed to store one array slice.
        return type_total_bytes(type) * stride[0];
    }

    // Per-frequency arrays

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    // Also indexed by the local coarse frequency channel.
    int freq_upchan_factor[CHORD_META_MAX_FREQ];

    // TODO: Store upchannelization index as well

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
        return type_to_string(type);
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

    void set_strides_simple() {
        // Compute the strides from the set dims assuming simple contiguous
        // access.
        assert(this->dims >= 0);
        std::ptrdiff_t np = 1;
        for (int d = this->dims - 1; d >= 0; --d) {
            this->stride[d] = np;
            assert(this->dim[d] >= 0);
            np *= this->dim[d];
        }
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

    // science metadata
    using beamCoord = jsonMetadata::beamCoord;

    /// The coordinates of the tracking beam (if applicable)
    beamCoord get_beam_coord() const {
        return metadata[jsonMetadata::BEAM_COORD].template get<beamCoord>();
    }

    // TODO: add set_beam_coord

    int64_t get_fpga_seq_num() const {
        return metadata[jsonMetadata::FPGA_SEQ_NUM].template get<int64_t>();
    }

    void set_fpga_seq_num(const int64_t fpga_seq_num) {
        metadata[jsonMetadata::FPGA_SEQ_NUM] = fpga_seq_num;
    }

    int get_nfreq() const {
        return static_cast<int>(metadata[jsonMetadata::COARSE_FREQ].size());
    }

    // TODO: this should really be a freq_id_t array
    const std::vector<int> get_coarse_freq() const {
        return metadata[jsonMetadata::COARSE_FREQ].template get<std::vector<int>>();
    }

    void set_coarse_freq(const std::vector<int>& coarse_freq) {
        assert(coarse_freq.size() < CHORD_META_MAX_FREQ);
        metadata[jsonMetadata::COARSE_FREQ] = coarse_freq;
    }

    // TODO: remove this, its redundant
    struct timespec get_gps_time() const {
        const Telescope& tel = Telescope::instance();
        return tel.to_time(this->get_fpga_seq_num());
    }

    // TODO: remove this, it's not setting anything anymore (and assumes that
    // fpga_seq_num is set)
    void set_gps_time(const timespec gps_time) {
        const Telescope& tel = Telescope::instance();
        const timespec my_gps_time = tel.to_time(this->get_fpga_seq_num());
        assert(gps_time.tv_sec == my_gps_time.tv_sec);
        assert(gps_time.tv_nsec == my_gps_time.tv_nsec);
    }

    /// The number of bad inputs in the RFI systems bad input list.
    /// This value is mostly needed for renormalization of the SK values.
    uint32_t get_rfi_num_bad_inputs() const {
        return metadata[jsonMetadata::RFI_NUM_BAD_INPUTS].template get<uint32_t>();
    }

    void set_rfi_num_bad_inputs(const uint32_t rfi_num_bad_inputs) {
        metadata[jsonMetadata::RFI_NUM_BAD_INPUTS] = rfi_num_bad_inputs;
    }

    /// The number of FPGA frames flagged as containing RFI.
    /// NOTE: This value might contain overlap with lost samples, so it can count
    /// missing samples as samples with RFI.  For renormalization this value
    /// should NOT be used, use @c lost_timesamples instead.
    /// This value will be filled even if RFI zeroing is disabled.
    int32_t get_rfi_flagged_samples() const {
        return metadata[jsonMetadata::RFI_FLAGGED_SAMPLES].template get<int32_t>();
    }

    void set_rfi_flagged_samples(const int32_t flagged_samples)  {
        // very much non-atomic, due to json dict entry creation
        metadata[jsonMetadata::RFI_FLAGGED_SAMPLES] = flagged_samples;
    }

    int32_t get_lost_timesamples() const {
        return metadata[jsonMetadata::LOST_TIMESAMPLES].template get<int32_t>();
    }

    void set_lost_timesamples(int32_t lost_timesamples) {
        // very much non-atomic, due to json dict entry creation
        metadata[jsonMetadata::LOST_TIMESAMPLES] = lost_timesamples;
    }

    void atomic_add_lost_timesamples(const int32_t lost_samples) {
        // RH: this is almost certainly not admissible code, but also almost certain, "works".
        // RH: this is not actually atomic and has race conditions if the underlying json dict changes
        static_assert(std::is_same<std::int64_t, nlohmann::json::number_integer_t>::value, "Roland's horrible hack fails");
        *reinterpret_cast<std::atomic_int64_t*>(metadata[jsonMetadata::LOST_TIMESAMPLES].template get_ptr<std::int64_t*>()) += lost_samples;
    }

    // non-science metadata

   timeval get_first_packet_recv_time() const {
        return metadata[jsonMetadata::FIRST_PACKET_RECV_TIME].template get<timeval>();
   }

   void set_first_packet_recv_time(const timeval time_v) {
        metadata[jsonMetadata::FIRST_PACKET_RECV_TIME] = time_v;
   }

    // links to other data

    stream_t get_stream_id() const {
        return stream_t{.id = metadata[jsonMetadata::STREAM_ID].template get<uint64_t>()};
    }

    void set_stream_id(const stream_t stream_id) {
        metadata[jsonMetadata::STREAM_ID] = stream_id.id;
    }

    /// ID of the dataset
    dset_id_t get_dataset_id() const {
        return metadata[jsonMetadata::DATASET_ID].template get<dset_id_t>();
    }

    void set_dataset_id(const dset_id_t dset_id) {
        metadata[jsonMetadata::DATASET_ID] = dset_id;
    }

private:
    jsonMetadata::metadata metadata;
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
