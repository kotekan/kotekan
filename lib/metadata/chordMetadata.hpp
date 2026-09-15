#ifndef CHORD_METADATA
#define CHORD_METADATA

#include "DataType.hpp"  // for type_to_string, type_total_bytes, DataType
#include "NDArray.hpp"   // for GenericNDArray
#include "Telescope.hpp" // for Telescope, stream_t
#include "buffer.hpp"    // for Buffer

#include <array>      // for array
#include <cassert>    // for assert
#include <cstddef>    // for size_t, ptrdiff_t
#include <cstdint>    // for int32_t, uint32_t, int64_t, uint64_t
#include <memory>     // for shared_ptr, __shared_ptr_access, allocator, static_pointer...
#include <mutex>      // for mutex, lock_guard
#include <sstream>    // for basic_ostream, operator<<, basic_ostringstream, basic_ostr...
#include <stdexcept>  // for runtime_error
#include <string.h>   // for strnlen
#include <string>     // for basic_string, char_traits, operator==, string, operator<<
#include <sys/time.h> // for timeval
#include <time.h>     // for timespec
#include <vector>     // for vector
// TODO: CHIME and CHORD differ whether they use the datasetManager
#include "dataset.hpp"        // for dset_id_t
#include "kotekanLogging.hpp" // for WARN_NON_OO
#include "metadata.hpp"       // for metadataObject, metadataPool

#include "fmt.hpp"          // for compile_string_to_view
#include "json.hpp"         // for basic_json, json
#include "jsonMetadata.hpp" // for COARSE_FREQ, LOST_TIMESAMPLES, STREAM_ID, BEAM_COORD, DATA...

#pragma pack()

// Maximum number of frequencies in metadata array
const int CHORD_META_MAX_FREQ = 12288;

// Maximum number of dimensions for arrays
const int CHORD_META_MAX_DIM = 10;
static_assert(CHORD_META_MAX_DIM == int(kotekan::GenericNDArray::max_rank),
              "CHORD_META_MAX_DIM must match GenericNDArray::max_rank");

// Maximum length of array names and dimension names.
//
// These names are stored in fixed-size char arrays that are NOT NUL-terminated:
// a name of exactly CHORD_META_MAX_DIMNAME characters fills its field
// completely. The fields are NUL-padded, so shorter names do end in a NUL, but
// no code may rely on that. Read them via get_name() and get_dimension_name(),
// which bound the length with strnlen; never pass them to strlen, strcpy,
// printf("%s") or fmt, all of which read until they find a NUL.
const int CHORD_META_MAX_DIMNAME = 24;

// Maximum number of stream IDs in metadata array
const int CHORD_META_MAX_STREAM_IDS = 64;

// Maximum number of visibility matrix samples in a frame
const int CHORD_META_MAX_VIS_SAMPLES = 64;

class chordMetadata : public metadataObject {
public:
    chordMetadata();
    bool operator==(const chordMetadata& other) const;

    /// Validates that this metadata's array structure (name, type, dimensions, strides) matches
    /// the given frame descriptor, issuing a non-fatal error for any inconsistencies.
    void check_frame_desc(const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) const;

    /// Copies array structure information (name, type, dimensions, dimension names, extents,
    /// strides) from the given frame descriptor into this metadata object.
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
        almost_copyable_mutex& operator=(const almost_copyable_mutex& /*other*/) {
            // A mutex is not copied; this object keeps its own.
            return *this;
        }
    } lock;

    // TODO: Replace by NDArray
    /// The name of the array, e.g. "E", "J", "I". NUL-padded but not
    /// NUL-terminated when the name uses all CHORD_META_MAX_DIMNAME characters;
    /// use set_name() and get_name() instead of accessing the field directly.
    char name[CHORD_META_MAX_DIMNAME];
    kotekan::DataType type;

    int dims;
    int dim[CHORD_META_MAX_DIM];
    /// The names of the dimensions, e.g. "F", "T", "D". NUL-padded but not
    /// NUL-terminated when a name uses all CHORD_META_MAX_DIMNAME characters; use
    /// set_dimension_name() and get_dimension_name() instead of accessing the
    /// fields directly.
    char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME];
    int64_t dim_scaling[CHORD_META_MAX_DIM];
    // The stride counts elements, not bytes
    int64_t stride[CHORD_META_MAX_DIM];
    // The offset counts elements, not bytes
    int64_t offset;

    size_t sample_bytes() const {
        // The number of bytes per sample is the number of bytes needed to store one array slice.
        if (dims < 1)
            FATAL_ERROR("sample_bytes: the array description is not set (dims={:d})", dims);
        if (stride[0] < 0)
            FATAL_ERROR("sample_bytes: stride[0] is not set (stride[0]={:d})", stride[0]);
        return type_total_bytes(type) * stride[0];
    }

    std::string get_dimension_name(size_t i) const {
        if (i >= size_t(CHORD_META_MAX_DIM))
            FATAL_ERROR("get_dimension_name: dimension {:d} is out of range [0, {:d})", i,
                        CHORD_META_MAX_DIM);
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

    void set_array_dimension(int dim, int size, const std::string& name, int64_t scaling) {
        set_dimension_name(dim, name); // checks the range of `dim` before we index anything
        this->dim[dim] = size;
        this->dim_scaling[dim] = scaling;
    }

    /// Sets the name of dimension @p dim, truncating it (with a warning) to
    /// CHORD_META_MAX_DIMNAME characters.
    void set_dimension_name(int dim, const std::string& name) {
        if (dim < 0 || dim >= CHORD_META_MAX_DIM)
            FATAL_ERROR("set_dimension_name: dimension {:d} is out of range [0, {:d})", dim,
                        CHORD_META_MAX_DIM);
        set_string_field(this->dim_name[dim], name, "dimension name");
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

    bool has_name() const {
        return (strnlen(this->name, CHORD_META_MAX_DIMNAME) > 0);
    }

    /// Sets the name of the array, truncating it (with a warning) to
    /// CHORD_META_MAX_DIMNAME characters.
    void set_name(const std::string& name) {
        set_string_field(this->name, name, "array name");
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

    // The FPGA sequence number of the first time sample in the buffer.
    // With time downsampling, this is the first FPGA sequence number
    // of the range of FPGA samples that correspond to the first time
    // sample in the buffer.
    //
    // Specifically, the FPGA sequence number defines an instant in
    // time. Each time sample in a buffer corresponds to a certain
    // time duration. Time downsampling affects the duration of each
    // sample, but it does not affect the instance in time at which
    // this buffer begins.
    //
    // For ring buffers things are slightly different. Formally, ring
    // buffers are infinitely large, they just reuse storage. The FPGA
    // sequence number describes the logical beginning of the buffer.
    // This information does not change during the life time of a ring
    // buffer. Since ring buffers reuse storage, the FPGA sequence
    // number of the time sample that happens to be stored at index 0
    // of the ring buffer will change over time, but the buffer's FPGA
    // sequence number will not.
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

    // Time downsampling -- the factor by which the time samples have
    // been downsampled relative to FPGA samples.
    void set_time_downsampling_fpga(const int time_downsampling_fpga) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::TIME_DOWNSAMPLING_FPGA] = time_downsampling_fpga;
    }

    bool has_time_downsampling_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::TIME_DOWNSAMPLING_FPGA);
    }

    int get_time_downsampling_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::TIME_DOWNSAMPLING_FPGA);
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
    /// Sets the coarse frequencies. Like the other per-frequency arrays this is
    /// either unset or non-empty; see @c has_coarse_freq.
    void set_coarse_freq(const std::vector<int>& coarse_freq) {
        std::lock_guard<std::mutex> guard(this->lock);
        if (coarse_freq.empty())
            FATAL_ERROR("set_coarse_freq: the per-frequency arrays must not be empty");
        if (coarse_freq.size() > size_t(CHORD_META_MAX_FREQ))
            FATAL_ERROR("set_coarse_freq: {:d} frequencies exceed CHORD_META_MAX_FREQ={:d}",
                        coarse_freq.size(), CHORD_META_MAX_FREQ);
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

    // Stream IDs - the stream identifiers for packets received
    void set_stream_ids(const std::vector<uint32_t>& stream_ids) {
        std::lock_guard<std::mutex> guard(this->lock);
        if (stream_ids.size() > size_t(CHORD_META_MAX_STREAM_IDS))
            FATAL_ERROR("set_stream_ids: {:d} stream IDs exceed CHORD_META_MAX_STREAM_IDS={:d}",
                        stream_ids.size(), CHORD_META_MAX_STREAM_IDS);
        metadata[jsonMetadata::STREAM_IDS] = stream_ids;
    }

    bool has_stream_ids() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::STREAM_IDS);
    }

    std::vector<uint32_t> get_stream_ids() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::STREAM_IDS).template get<std::vector<uint32_t>>();
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

    /// Adds to the lost time sample count. The count must already be set;
    /// otherwise there is no way to tell "no samples lost" from "not counted".
    void atomic_add_lost_timesamples(const int32_t lost_samples) {
        std::lock_guard<std::mutex> guard(this->lock);
        const auto it = metadata.find(jsonMetadata::LOST_TIMESAMPLES);
        if (it == metadata.end())
            throw std::runtime_error(
                "atomic_add_lost_timesamples: LOST_TIMESAMPLES has not been set");
        *it = it->template get<int32_t>() + lost_samples;
    }

    // Per-frequency arrays

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    /// Either unset or non-empty, like the other per-frequency arrays.
    void set_freq_upchan_factor(const std::vector<int>& freq_upchan_factor) {
        std::lock_guard<std::mutex> guard(this->lock);
        if (freq_upchan_factor.empty())
            FATAL_ERROR("set_freq_upchan_factor: the per-frequency arrays must not be empty");
        if (freq_upchan_factor.size() > size_t(CHORD_META_MAX_FREQ))
            FATAL_ERROR("set_freq_upchan_factor: {:d} frequencies exceed CHORD_META_MAX_FREQ={:d}",
                        freq_upchan_factor.size(), CHORD_META_MAX_FREQ);
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

    // the upchannelization index for each frequency (0 ... upchannelization factor - 1)
    /// Either unset or non-empty, like the other per-frequency arrays.
    void set_freq_upchan_index(const std::vector<int>& freq_upchan_index) {
        std::lock_guard<std::mutex> guard(this->lock);
        if (freq_upchan_index.empty())
            FATAL_ERROR("set_freq_upchan_index: the per-frequency arrays must not be empty");
        if (freq_upchan_index.size() > size_t(CHORD_META_MAX_FREQ))
            FATAL_ERROR("set_freq_upchan_index: {:d} frequencies exceed CHORD_META_MAX_FREQ={:d}",
                        freq_upchan_index.size(), CHORD_META_MAX_FREQ);
        metadata[jsonMetadata::FREQ_UPCHAN_INDEX] = freq_upchan_index;
    }

    bool has_freq_upchan_index() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FREQ_UPCHAN_INDEX);
    }

    std::vector<int> get_freq_upchan_index() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FREQ_UPCHAN_INDEX).template get<std::vector<int>>();
    }

    // Whether second stage RFI excision (at the GPU frame level) is enabled
    void set_rfi_frame_excision_enabled(const bool rfi_frame_excision_enabled) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::RFI_FRAME_EXCISION_ENABLED] = rfi_frame_excision_enabled;
    }

    bool has_rfi_frame_excision_enabled() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::RFI_FRAME_EXCISION_ENABLED);
    }

    bool get_rfi_frame_excision_enabled() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::RFI_FRAME_EXCISION_ENABLED).template get<bool>();
    }

    // Second stage RFI excision (whole GPU frames) thresholds
    void set_rfi_frame_excision_thresholds(const std::vector<std::array<float, 2>>& thresholds) {
        std::lock_guard<std::mutex> guard(this->lock);
        if (thresholds.size() > size_t(MAX_NUM_RFI_THRESHOLDS))
            FATAL_ERROR("set_rfi_frame_excision_thresholds: {:d} thresholds exceed "
                        "MAX_NUM_RFI_THRESHOLDS={:d}",
                        thresholds.size(), MAX_NUM_RFI_THRESHOLDS);
        metadata[jsonMetadata::RFI_FRAME_EXCISION_THRESHOLDS] = thresholds;
    }

    bool has_rfi_frame_excision_thresholds() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::RFI_FRAME_EXCISION_THRESHOLDS);
    }

    std::vector<std::array<float, 2>> get_rfi_frame_excision_thresholds() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::RFI_FRAME_EXCISION_THRESHOLDS)
            .template get<std::vector<std::array<float, 2>>>();
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
    /// Copies @p str into a fixed-size name field, padding it with NULs and
    /// truncating (with a warning naming @p what) if it does not fit. A name that
    /// fills the field leaves it without a terminating NUL; see
    /// CHORD_META_MAX_DIMNAME.
    void set_string_field(char (&field)[CHORD_META_MAX_DIMNAME], const std::string& str,
                          const char* what);

    jsonMetadata::metadata metadata;

    // these are not thread safe
    chordMetadata& operator=(const chordMetadata&) = default;
    chordMetadata(const chordMetadata&) = default;

    friend void to_json(nlohmann::json& j, const chordMetadata& m);
    friend void from_json(const nlohmann::json& j, chordMetadata& m);
};

void to_json(nlohmann::json& j, const chordMetadata& m);
void from_json(const nlohmann::json& j, chordMetadata& m);

bool metadata_is_chord(Buffer* buf, int);
bool metadata_is_chord(const std::shared_ptr<metadataObject>& mc);
bool metadata_is_chord(const std::shared_ptr<const metadataObject>& mc);

std::shared_ptr<chordMetadata> get_chord_metadata(const std::shared_ptr<metadataObject>& mc);
std::shared_ptr<const chordMetadata>
get_chord_metadata(const std::shared_ptr<const metadataObject>& mc);
std::shared_ptr<chordMetadata> get_chord_metadata(Buffer* buf, int frame_id);

#endif
