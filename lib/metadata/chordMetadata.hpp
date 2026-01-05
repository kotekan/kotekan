#ifndef CHORD_METADATA
#define CHORD_METADATA

#include "CHORDTelescope.hpp" // for dish_index_t
#include "DataType.hpp"       // for type_to_string, type_total_bytes, DataType
#include "NDArray.hpp"        // for GenericNDArray
#include "Telescope.hpp"      // for Telescope, stream_t
#include "buffer.hpp"         // for Buffer

#include <cassert>    // for assert
#include <cstddef>    // for size_t, ptrdiff_t
#include <cstdint>    // for int64_t, int32_t, uint32_t, uint64_t
#include <memory>     // for shared_ptr, __shared_ptr_access, allocator, static_pointer...
#include <mutex>      // for mutex, lock_guard
#include <sstream>    // for basic_ostream, operator<<, basic_ostringstream, basic_ostr...
#include <string.h>   // for strnlen, strncpy
#include <string>     // for basic_string, char_traits, operator==, string, operator<<
#include <sys/time.h> // for timeval
#include <time.h>     // for timespec
#include <vector>     // for vector
// TODO: CHIME and CHORD differ whether they use the datasetManager
#include "dataset.hpp"        // for dset_id_t
#include "kotekanLogging.hpp" // for WARN_NON_OO
#include "metadata.hpp"       // for metadataObject, metadataPool

#include "fmt/format.h"     // for compile_string_to_view
#include "json.hpp"         // for basic_json, json
#include "jsonMetadata.hpp" // for COARSE_FREQ, LOST_TIMESAMPLES, STREAM_ID, BEAM_COORD, DATA...

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
    bool operator==(const chordMetadata& other) const;

    /// Helper function to compare data during conversion to NDArray
    void check_frame_desc(const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) const;

    /// Helper function to set CHORD metadata during conversion to NDArray
    void set_from_frame_desc(const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc);

    /// copy object
    void deepCopy(std::shared_ptr<const metadataObject> other) override;

    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;

    /// serialize to json
    nlohmann::json to_json() override;

    /// controls access to this object
    mutable class almost_copyable_mutex : public std::mutex {
        // chordMetadata::deepCopy copies chordMetadata and locks it, so this must not itself lock
    public:
        almost_copyable_mutex() : std::mutex() {}
        almost_copyable_mutex(const almost_copyable_mutex& /*other*/) : std::mutex() {}
        almost_copyable_mutex operator=(const almost_copyable_mutex& /*other*/) {
            return *this;
        }
    } lock;

    // TODO: Replace by NDArray
    char name[CHORD_META_MAX_DIMNAME]; // "E", "J", "I", etc
    kotekan::DataType type;

    int dims;
    int dim[CHORD_META_MAX_DIM];
    char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME]; // "F", "T", "D", etc
    // The stride counts elements, not bytes
    int64_t stride[CHORD_META_MAX_DIM];
    // The offset counts elements, not bytes
    int64_t offset;

    size_t sample_bytes() const {
        // The number of bytes per sample is the number of bytes needed to store one array slice.
        return type_total_bytes(type) * stride[0];
    }

    // Dish layout
    int ndishes;                                  // number of dishes
    int n_dish_locations_ew, n_dish_locations_ns; // number of possible dish locations
    dish_index_t* dish_index; // [non-owning pointer] dish index for a possible dish location, or -1
    dish_index_t get_dish_index(int dish_loc_ew, int dish_loc_ns) const {
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
        // Manually copying in a for loop to avoid possibly buggy GCC warning
        // about array bounds and stringop-truncation.

        int len = name.length() < CHORD_META_MAX_DIMNAME ? name.length() : CHORD_META_MAX_DIMNAME;

        for (int i = 0; i < len; i++)
            this->name[i] = name[i];
        // Fill the remaining space with 0s
        for (int i = len; i < CHORD_META_MAX_DIMNAME; i++)
            this->name[i] = '\0';
    }

    std::string get_name() const {
        return std::string(name, strnlen(name, CHORD_META_MAX_DIMNAME));
    }

    // science metadata
    using beamCoord = jsonMetadata::beamCoord;

    /// The coordinates of the tracking beam (if applicable)
    bool has_beam_coord() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::BEAM_COORD);
    }

    beamCoord get_beam_coord() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::BEAM_COORD).template get<beamCoord>();
    }

    void set_beam_coord(const beamCoord& beam_coord) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::BEAM_COORD] = beam_coord;
    }

    void set_fpga_seq_num(const int64_t fpga_seq_num) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::FPGA_SEQ_NUM] = fpga_seq_num;
    }

    bool has_fpga_seq_num() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FPGA_SEQ_NUM);
    }

    int64_t get_fpga_seq_num() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FPGA_SEQ_NUM).template get<int64_t>();
    }

    void set_frame_counter(const int frame_counter) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::FRAME_COUNTER] = frame_counter;
    }

    bool has_frame_counter() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FRAME_COUNTER);
    }

    int get_frame_counter() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FRAME_COUNTER).template get<int>();
    }

    bool has_nfreq() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::COARSE_FREQ);
    }

    int get_nfreq() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return static_cast<int>(metadata.at(jsonMetadata::COARSE_FREQ).size());
    }

    // TODO: this should really be a freq_id_t array
    void set_coarse_freq(const std::vector<int>& coarse_freq) {
        std::lock_guard<std::mutex> lock(this->lock);
        assert(coarse_freq.size() <= CHORD_META_MAX_FREQ);
        metadata[jsonMetadata::COARSE_FREQ] = coarse_freq;
    }

    bool has_coarse_freq() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::COARSE_FREQ);
    }

    std::vector<int> get_coarse_freq() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::COARSE_FREQ).template get<std::vector<int>>();
    }

    // TODO: remove this, it's not setting anything anymore (and assumes that
    // fpga_seq_num is set)
    void set_gps_time([[maybe_unused]] const timespec gps_time) {
        // this must not request the lock
        const Telescope& tel = Telescope::instance();
        [[maybe_unused]] const timespec my_gps_time = tel.to_time(this->get_fpga_seq_num());
        assert(gps_time.tv_sec == my_gps_time.tv_sec);
        assert(gps_time.tv_nsec == my_gps_time.tv_nsec);
    }

    // TODO: remove this, it's redundant
    bool has_gps_time() const {
        return has_fpga_seq_num();
    }

    // TODO: remove this, it's redundant
    struct timespec get_gps_time() const {
        // this must not request the lock
        const Telescope& tel = Telescope::instance();
        return tel.to_time(this->get_fpga_seq_num());
    }

    /// The number of bad inputs in the RFI systems bad input list.
    /// This value is mostly needed for renormalization of the SK values.
    void set_rfi_num_bad_inputs(const uint32_t rfi_num_bad_inputs) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::RFI_NUM_BAD_INPUTS] = rfi_num_bad_inputs;
    }

    bool has_rfi_num_bad_inputs() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::RFI_NUM_BAD_INPUTS);
    }

    uint32_t get_rfi_num_bad_inputs() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::RFI_NUM_BAD_INPUTS).template get<uint32_t>();
    }

    /// The number of FPGA frames flagged as containing RFI.
    /// NOTE: This value might contain overlap with lost samples, so it can count
    /// missing samples as samples with RFI.  For renormalization this value
    /// should NOT be used, use @c lost_timesamples instead.
    /// This value will be filled even if RFI zeroing is disabled.
    void set_rfi_flagged_samples(const int32_t flagged_samples) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::RFI_FLAGGED_SAMPLES] = flagged_samples;
    }

    bool has_rfi_flagged_samples() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::RFI_FLAGGED_SAMPLES);
    }

    int32_t get_rfi_flagged_samples() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::RFI_FLAGGED_SAMPLES).template get<int32_t>();
    }

    void set_lost_timesamples(int32_t lost_timesamples) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::LOST_TIMESAMPLES] = lost_timesamples;
    }

    bool has_lost_timesamples() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::LOST_TIMESAMPLES);
    }

    int32_t get_lost_timesamples() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::LOST_TIMESAMPLES).template get<int32_t>();
    }

    void atomic_add_lost_timesamples(const int32_t lost_samples) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata.at(jsonMetadata::LOST_TIMESAMPLES) =
            metadata.at(jsonMetadata::LOST_TIMESAMPLES).template get<int32_t>() + lost_samples;
    }

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
    void set_sample0_offset(const int64_t sample0_offset) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::SAMPLE0_OFFSET] = sample0_offset;
    }

    bool has_sample0_offset() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::SAMPLE0_OFFSET);
    }

    int64_t get_sample0_offset() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::SAMPLE0_OFFSET).template get<int64_t>();
    }

    void set_offset_downsampling(const int offset_downsampling) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::OFFSET_DOWNSAMPLING] = offset_downsampling;
    }

    bool has_offset_downsampling() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::OFFSET_DOWNSAMPLING);
    }

    int get_offset_downsampling() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::OFFSET_DOWNSAMPLING).template get<int>();
    }

    // Per-frequency arrays

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    // Also indexed by the local coarse frequency channel.
    void set_freq_upchan_factor(const std::vector<int>& freq_upchan_factor) {
        std::lock_guard<std::mutex> lock(this->lock);
        assert(freq_upchan_factor.size() <= CHORD_META_MAX_FREQ);
        metadata[jsonMetadata::FREQ_UPCHAN_FACTOR] = freq_upchan_factor;
    }

    bool has_freq_upchan_factor() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FREQ_UPCHAN_FACTOR);
    }

    std::vector<int> get_freq_upchan_factor() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FREQ_UPCHAN_FACTOR).template get<std::vector<int>>();
    }

    // TODO: Store upchannelization index as well

    // Time sampling -- for each coarse frequency channel, 2x the FPGA
    // sample number of the first sample.  The 2x is there to handle
    // the upchannelization case, where 2 or more samples may get
    // averaged, producing a new sample that is effectively halfway in
    // between them, ie, at a half-FPGAsample time.
    void set_half_fpga_sample0(const std::vector<int64_t>& half_fpga_sample0) {
        std::lock_guard<std::mutex> lock(this->lock);
        assert(half_fpga_sample0.size() <= CHORD_META_MAX_FREQ);
        metadata[jsonMetadata::HALF_FPGA_SAMPLE0] = half_fpga_sample0;
    }

    bool has_half_fpga_sample0() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::HALF_FPGA_SAMPLE0);
    }

    std::vector<int64_t> get_half_fpga_sample0() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::HALF_FPGA_SAMPLE0).template get<std::vector<int64_t>>();
    }

    // Time sampling -- for each coarse frequency channel, the factor
    // by which the time samples have been downsampled relative to
    // FPGA samples.
    void set_time_downsampling_fpga(const std::vector<int>& time_downsampling_fpga) {
        std::lock_guard<std::mutex> lock(this->lock);
        assert(time_downsampling_fpga.size() <= CHORD_META_MAX_FREQ);
        metadata[jsonMetadata::TIME_DOWNSAMPLING_FPGA] = time_downsampling_fpga;
    }

    bool has_time_downsampling_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::TIME_DOWNSAMPLING_FPGA);
    }

    std::vector<int> get_time_downsampling_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::TIME_DOWNSAMPLING_FPGA);
    }

    // non-science metadata

    void set_first_packet_recv_time(const timeval time_v) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::FIRST_PACKET_RECV_TIME] = time_v;
    }

    bool has_first_packet_recv_time() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FIRST_PACKET_RECV_TIME);
    }

    timeval get_first_packet_recv_time() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FIRST_PACKET_RECV_TIME).template get<timeval>();
    }

    // links to other data

    void set_stream_id(const stream_t stream_id) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::STREAM_ID] = stream_id.id;
    }

    bool has_stream_id() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::STREAM_ID);
    }

    stream_t get_stream_id() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return stream_t{.id = metadata.at(jsonMetadata::STREAM_ID).template get<uint64_t>()};
    }

    /// ID of the dataset
    void set_dataset_id(const dset_id_t dset_id) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::DATASET_ID] = dset_id;
    }

    bool has_dataset_id() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::DATASET_ID);
    }

    dset_id_t get_dataset_id() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::DATASET_ID).template get<dset_id_t>();
    }

    std::string get_string_repr_of_json() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.dump();
    }

private:
    jsonMetadata::metadata metadata;

    // these are not thread safe
    chordMetadata& operator=(const chordMetadata&) = default;
    chordMetadata(const chordMetadata&) = default;

    friend void to_json(nlohmann::json& j, const chordMetadata& m);
    friend void from_json(const nlohmann::json& j, chordMetadata& m);
};

void to_json(nlohmann::json& j, const chordMetadata& m);
void from_json(const nlohmann::json& j, chordMetadata& m);

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
    std::shared_ptr<metadataObject> meta = buf->metadata.at(frame_id);
    return get_chord_metadata(meta);
}

#endif
