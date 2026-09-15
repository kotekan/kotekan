#include "chordMetadata.hpp"

#include "Symbol.hpp"  // for Symbol
#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <algorithm>   // for copy_n, copy, max, find_if
#include <array>       // for array
#include <cstring>     // for memset, memcpy
#include <json.hpp>    // for operator==, json
#include <stdexcept>   // for runtime_error
#include <string.h>    // for strncmp, strnlen, memset
#include <type_traits> // for is_pod_v

REGISTER_TYPE_WITH_FACTORY(metadataObject, chordMetadata);

chordMetadata::chordMetadata() : type(kotekan::unknown_type), dims(-1), offset(0) {
    name[0] = '\0';
    for (int d = 0; d < CHORD_META_MAX_DIM; ++d) {
        dim[d] = -1;
        dim_name[d][0] = '\0';
        dim_scaling[d] = -1;
        stride[d] = -1;
    }
}

void chordMetadata::set_string_field(char (&field)[CHORD_META_MAX_DIMNAME], const std::string& str,
                                     const char* const what) {
    // A name that fills the field is stored without a terminating NUL; the field
    // is not a C string. See CHORD_META_MAX_DIMNAME.
    const std::size_t max_length = sizeof(field);
    const std::size_t length = std::min(str.size(), max_length);
    if (str.size() > max_length)
        WARN("Truncating {:s} \"{:s}\" to {:d} characters", what, str, max_length);
    std::memcpy(field, str.data(), length);
    std::memset(field + length, '\0', max_length - length);
}

bool chordMetadata::operator==(const chordMetadata& other) const {
    if (this == &other)
        return true;

    std::scoped_lock<std::mutex, std::mutex> guard(this->lock, other.lock);

    if (0 != strncmp(name, other.name, CHORD_META_MAX_DIMNAME))
        return false;

    if (type != other.type)
        return false;

    if (dims != other.dims)
        return false;

    for (int d = 0; d < dims; ++d)
        if (dim[d] != other.dim[d])
            return false;
    for (int d = 0; d < dims; ++d)
        if (0 != strncmp(dim_name[d], other.dim_name[d], CHORD_META_MAX_DIMNAME))
            return false;
    for (int d = 0; d < dims; ++d)
        if (dim_scaling[d] != other.dim_scaling[d])
            return false;
    for (int d = 0; d < dims; ++d)
        if (stride[d] != other.stride[d])
            return false;
    if (offset != other.offset)
        return false;

    return metadata == other.metadata;
}

void chordMetadata::check_frame_desc(
    const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) const {

    bool failed = false;

    // An invalid Symbol has a null string; never hand that to strncmp or fmt.
    const kotekan::Symbol quantity_name = frame_desc->get_quantity_name();
    if (!quantity_name.valid()) {
        ERROR("Frame descriptor has no quantity name");
        failed = true;
    } else if (this->get_name() != quantity_name.get_string()) {
        ERROR("Names differ: {:s} != {:s}", this->get_name(), quantity_name.get_string());
        failed = true;
    }
    if (this->type != frame_desc->get_value_datatype()) {
        ERROR("Types differ for {:s}: {:s} != {:s}", quantity_name,
              kotekan::type_to_string(this->type),
              kotekan::type_to_string(frame_desc->get_value_datatype()));
        failed = true;
    }
    if (size_t(this->dims) != frame_desc->get_rank()) {
        ERROR("Ranks differ for {:s}: {:d} != {:d}", quantity_name, this->dims,
              frame_desc->get_rank());
        failed = true;
    }
    for (int d = this->dims - 1; d >= 0; --d) {

        const kotekan::Symbol dimname = frame_desc->get_dimname(d);
        if (!dimname.valid()) {
            ERROR("Frame descriptor for {:s} has no name for dimension {:d}", quantity_name, d);
            failed = true;
        } else if (this->get_dimension_name(d) != dimname.get_string()) {
            ERROR("Dim_name[{:d}] differs for {:s}: {:s} != {:s}", d, quantity_name,
                  this->get_dimension_name(d), dimname.get_string());
            failed = true;
        }

        if (this->dim_scaling[d] != frame_desc->get_dimscaling(d)) {
            ERROR("Dim_scaling[{:d}] differs for {:s}: {:d} != {:d}", d, quantity_name,
                  this->dim_scaling[d], frame_desc->get_dimscaling(d));
            failed = true;
        }

        if (this->dim[d] != frame_desc->get_extent(d)) {
            ERROR("Dim[{:d}] differs for {:s}: {:d} != {:d}", d, quantity_name, this->dim[d],
                  frame_desc->get_extent(d));
            failed = true;
        }
        if (this->stride[d] != frame_desc->get_stride(d)) {
            ERROR("Stride[{:d}] differs for {:s}: {:d} != {:d}", d, quantity_name, this->stride[d],
                  frame_desc->get_stride(d));
            failed = true;
        }
    }

    if (failed)
        FATAL_ERROR("Inconsistent array description between CHORDMetadata and FrameDesc");
}

void chordMetadata::set_from_frame_desc(
    const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) {
    // An invalid Symbol has no string; Symbol::get_string() would throw.
    const kotekan::Symbol quantity_name = frame_desc->get_quantity_name();
    if (!quantity_name.valid())
        FATAL_ERROR("Cannot set metadata from a frame descriptor without a quantity name");
    const std::size_t rank = frame_desc->get_rank();
    if (rank > std::size_t(CHORD_META_MAX_DIM))
        FATAL_ERROR("Frame descriptor for {:s} has rank {:d}, which exceeds "
                    "CHORD_META_MAX_DIM={:d}",
                    quantity_name, rank, CHORD_META_MAX_DIM);

    set_name(quantity_name.get_string());
    this->type = frame_desc->get_value_datatype();
    this->dims = int(rank);
    for (int d = this->dims - 1; d >= 0; --d) {
        const kotekan::Symbol dimname = frame_desc->get_dimname(d);
        if (!dimname.valid())
            FATAL_ERROR("Frame descriptor for {:s} has no name for dimension {:d}", quantity_name,
                        d);
        set_array_dimension(d, frame_desc->get_extent(d), dimname.get_string(),
                            frame_desc->get_dimscaling(d));
        this->stride[d] = frame_desc->get_stride(d);
    }
}

void chordMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto chord_other = std::dynamic_pointer_cast<const chordMetadata>(other);
    assert(chord_other);

    if (this == chord_other.get())
        return;

    // order locks so that there is no race condition if two chordMetadata a and
    // b are deepCopy'ed into each other a the same time
    std::scoped_lock<std::mutex, std::mutex> guard(this->lock, chord_other->lock);
    // This copies the base class too, so `this` adopts the other object's metadata
    // pool. That is load-bearing rather than accidental: parent_pool doubles as the
    // type tag that metadata_is_chord() reads, and cudaCopyFromRingbuffer,
    // cudaCopyToRingbuffer and cudaCopyNToRingbuffer deliberately create detached
    // copies with make_shared<chordMetadata>(). Those have no pool of their own, so
    // without inheriting one here they trip metadata_is_chord()'s assert(pool).
    *this = *chord_other;
}

struct chordMetadataFormat {
    int32_t max_dim;
    int32_t max_dimname;
    int32_t max_freq;
    int32_t max_stream_ids;
    int32_t max_rfi_thresholds;

    int32_t frame_counter;
    int64_t fpga_seq_num;

    // Time sampling -- the factor by which the time samples have been
    // downsampled relative to FPGA samples.
    int32_t time_downsampling_fpga;

    char name[CHORD_META_MAX_DIMNAME]; // "E", "J", "I", etc
    // chordDataType type;
    int32_t type;

    int32_t dims;
    int32_t dim[CHORD_META_MAX_DIM];
    char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME]; // "F", "Tbar", "D", etc
    int64_t dim_scaling[CHORD_META_MAX_DIM];
    int64_t stride[CHORD_META_MAX_DIM];
    int64_t offset;

    // Per-frequency arrays. Each array is either unset or has `nfreq` entries,
    // and `nfreq` is at least 1 whenever any of them is set. An array is unset
    // when its first element holds the invalid value noted below, which is
    // unambiguous exactly because a set array is never empty.
    int32_t nfreq;

    // frequencies -- integer (0-2047) identifier for FPGA coarse frequencies;
    // unset when coarse_freq[0] == -1, an invalid frequency index
    int32_t coarse_freq[CHORD_META_MAX_FREQ];

    // the upchannelization factor that each frequency has gone through (1 for =
    // FPGA); unset when freq_upchan_factor[0] == 0, an invalid factor
    int32_t freq_upchan_factor[CHORD_META_MAX_FREQ];

    // the upchannelization index for each frequency (0 ... freq_upchan_factor -
    // 1); unset when freq_upchan_index[0] == -1, an invalid index
    int32_t freq_upchan_index[CHORD_META_MAX_FREQ];

    // Stream IDs. `num_stream_ids` is -1 when unset.
    int32_t num_stream_ids;
    uint32_t stream_ids[CHORD_META_MAX_STREAM_IDS];

    // Second stage (whole GPU frame) RFI excision.
    // `rfi_frame_excision_enabled` is -1 when unset, 0 or 1 otherwise.
    // `num_rfi_frame_excision_thresholds` is -1 when unset.
    int32_t rfi_frame_excision_enabled;
    int32_t num_rfi_frame_excision_thresholds;
    float rfi_frame_excision_thresholds[MAX_NUM_RFI_THRESHOLDS][2];

    int32_t rfi_flagged_samples;
    int32_t lost_timesamples;

    int32_t have_beam_coord;
    chordMetadata::beamCoord beam_coord;
    static_assert(std::is_pod_v<chordMetadata::beamCoord> == true,
                  "beamCoord contains C++ only data");

    stream_t stream_id;
    static_assert(std::is_pod_v<stream_t> == true, "stream_t contains C++ only data");

    // cannot use dset_id_t here since Hash is not a pod (has a constructor)
    char dataset_id[32];

    timeval first_packet_recv_time;
};

size_t chordMetadata::get_serialized_size() {
    return sizeof(chordMetadataFormat);
}

// Note: The pointers to chordMetadataFormat below must not be called `fmt`. That
// would hide the `fmt` namespace used here and by the logging macros.
//
// These two functions validate data that is exchanged with other kotekan
// instances (see bufferSend/bufferRecv), so they throw instead of calling
// FATAL_ERROR; the caller decides how bad a malformed frame is. This also matches
// from_json below.

size_t chordMetadata::set_from_bytes(const char* bytes, size_t length) {
    if (length < sizeof(chordMetadataFormat))
        throw std::runtime_error(
            fmt::format("Cannot deserialize chordMetadata: got {:d} bytes, need at least {:d}",
                        length, sizeof(chordMetadataFormat)));

    const chordMetadataFormat* fmt_data = reinterpret_cast<const chordMetadataFormat*>(bytes);

    // These describe the layout of the byte array, so they must be checked
    // before anything else is read from it.
    if (fmt_data->max_dim != CHORD_META_MAX_DIM || fmt_data->max_dimname != CHORD_META_MAX_DIMNAME
        || fmt_data->max_freq != CHORD_META_MAX_FREQ
        || fmt_data->max_stream_ids != CHORD_META_MAX_STREAM_IDS
        || fmt_data->max_rfi_thresholds != MAX_NUM_RFI_THRESHOLDS)
        throw std::runtime_error(fmt::format(
            "Cannot deserialize chordMetadata: the sender uses incompatible limits "
            "(max_dim={:d}, max_dimname={:d}, max_freq={:d}, max_stream_ids={:d}, "
            "max_rfi_thresholds={:d}; expected {:d}, {:d}, {:d}, {:d}, {:d})",
            fmt_data->max_dim, fmt_data->max_dimname, fmt_data->max_freq, fmt_data->max_stream_ids,
            fmt_data->max_rfi_thresholds, CHORD_META_MAX_DIM, CHORD_META_MAX_DIMNAME,
            CHORD_META_MAX_FREQ, CHORD_META_MAX_STREAM_IDS, MAX_NUM_RFI_THRESHOLDS));

    if (fmt_data->dims < -1 || fmt_data->dims > CHORD_META_MAX_DIM)
        throw std::runtime_error(
            fmt::format("Cannot deserialize chordMetadata: dims={:d} is not in [-1, {:d}]",
                        fmt_data->dims, CHORD_META_MAX_DIM));
    if (fmt_data->nfreq < 0 || fmt_data->nfreq > CHORD_META_MAX_FREQ)
        throw std::runtime_error(
            fmt::format("Cannot deserialize chordMetadata: nfreq={:d} is not in [0, {:d}]",
                        fmt_data->nfreq, CHORD_META_MAX_FREQ));
    if (fmt_data->num_stream_ids < -1 || fmt_data->num_stream_ids > CHORD_META_MAX_STREAM_IDS)
        throw std::runtime_error(fmt::format(
            "Cannot deserialize chordMetadata: num_stream_ids={:d} is not in [-1, {:d}]",
            fmt_data->num_stream_ids, CHORD_META_MAX_STREAM_IDS));
    if (fmt_data->num_rfi_frame_excision_thresholds < -1
        || fmt_data->num_rfi_frame_excision_thresholds > MAX_NUM_RFI_THRESHOLDS)
        throw std::runtime_error(fmt::format(
            "Cannot deserialize chordMetadata: num_rfi_frame_excision_thresholds={:d} is "
            "not in [-1, {:d}]",
            fmt_data->num_rfi_frame_excision_thresholds, MAX_NUM_RFI_THRESHOLDS));

    if (fmt_data->frame_counter != -1)
        this->set_frame_counter(fmt_data->frame_counter);
    if (fmt_data->fpga_seq_num != -1)
        this->set_fpga_seq_num(fmt_data->fpga_seq_num);
    if (fmt_data->time_downsampling_fpga != -1)
        this->set_time_downsampling_fpga(fmt_data->time_downsampling_fpga);

    // These fields are not NUL-terminated; copy them verbatim. They are only ever
    // read through get_name()/get_dimension_name(), which bound the length.
    std::memcpy(name, fmt_data->name, sizeof(name));

    type = (kotekan::DataType)fmt_data->type;
    dims = fmt_data->dims;
    for (int i = 0; i < dims; i++) {
        dim[i] = fmt_data->dim[i];
        std::memcpy(dim_name[i], fmt_data->dim_name[i], sizeof(dim_name[i]));
        dim_scaling[i] = fmt_data->dim_scaling[i];
        stride[i] = fmt_data->stride[i];
    }
    offset = fmt_data->offset;

    const int nfreq = fmt_data->nfreq;
    if (fmt_data->coarse_freq[0] != -1)
        this->set_coarse_freq(
            std::vector<int>(fmt_data->coarse_freq, fmt_data->coarse_freq + nfreq));
    if (fmt_data->freq_upchan_factor[0] != 0)
        this->set_freq_upchan_factor(
            std::vector<int>(fmt_data->freq_upchan_factor, fmt_data->freq_upchan_factor + nfreq));
    if (fmt_data->freq_upchan_index[0] != -1)
        this->set_freq_upchan_index(
            std::vector<int>(fmt_data->freq_upchan_index, fmt_data->freq_upchan_index + nfreq));

    if (fmt_data->num_stream_ids >= 0)
        this->set_stream_ids(std::vector<uint32_t>(
            fmt_data->stream_ids, fmt_data->stream_ids + fmt_data->num_stream_ids));

    if (fmt_data->rfi_frame_excision_enabled >= 0)
        this->set_rfi_frame_excision_enabled(fmt_data->rfi_frame_excision_enabled != 0);
    if (fmt_data->num_rfi_frame_excision_thresholds >= 0) {
        std::vector<std::array<float, 2>> thresholds;
        thresholds.reserve(fmt_data->num_rfi_frame_excision_thresholds);
        for (int i = 0; i < fmt_data->num_rfi_frame_excision_thresholds; ++i)
            thresholds.push_back({fmt_data->rfi_frame_excision_thresholds[i][0],
                                  fmt_data->rfi_frame_excision_thresholds[i][1]});
        this->set_rfi_frame_excision_thresholds(thresholds);
    }

    if (fmt_data->rfi_flagged_samples != -1)
        this->set_rfi_flagged_samples(fmt_data->rfi_flagged_samples);
    if (fmt_data->lost_timesamples != -1)
        this->set_lost_timesamples(fmt_data->lost_timesamples);

    if (fmt_data->stream_id.id != uint64_t(-1))
        this->set_stream_id(fmt_data->stream_id);
    if (fmt_data->dataset_id[0] != '\0')
        this->set_dataset_id(dset_id_t::from_string(
            std::string(fmt_data->dataset_id, sizeof(fmt_data->dataset_id))));

    if (fmt_data->have_beam_coord)
        this->set_beam_coord(fmt_data->beam_coord);

    if (fmt_data->first_packet_recv_time.tv_sec != 0) // sometime in 1970
        this->set_first_packet_recv_time(fmt_data->first_packet_recv_time);

    // TODO: this misses dish_positions etc
    return sizeof(chordMetadataFormat);
}

size_t chordMetadata::serialize(char* bytes) {
    chordMetadataFormat* fmt_data = reinterpret_cast<chordMetadataFormat*>(bytes);
    memset(fmt_data, 0, sizeof(chordMetadataFormat));

    fmt_data->max_dim = CHORD_META_MAX_DIM;
    fmt_data->max_dimname = CHORD_META_MAX_DIMNAME;
    fmt_data->max_freq = CHORD_META_MAX_FREQ;
    fmt_data->max_stream_ids = CHORD_META_MAX_STREAM_IDS;
    fmt_data->max_rfi_thresholds = MAX_NUM_RFI_THRESHOLDS;

    if (this->has_frame_counter())
        fmt_data->frame_counter = this->get_frame_counter();
    else
        fmt_data->frame_counter = -1;
    if (this->has_fpga_seq_num())
        fmt_data->fpga_seq_num = this->get_fpga_seq_num();
    else
        fmt_data->fpga_seq_num = -1;
    if (this->has_time_downsampling_fpga())
        fmt_data->time_downsampling_fpga = this->get_time_downsampling_fpga();
    else
        fmt_data->time_downsampling_fpga = -1;

    if (dims > CHORD_META_MAX_DIM)
        throw std::runtime_error(
            fmt::format("Cannot serialize chordMetadata: dims={:d} exceeds CHORD_META_MAX_DIM={:d}",
                        dims, CHORD_META_MAX_DIM));
    std::memcpy(fmt_data->name, name, sizeof(fmt_data->name));
    fmt_data->type = (int32_t)type;
    fmt_data->dims = dims;
    for (int i = 0; i < dims; i++) {
        fmt_data->dim[i] = dim[i];
        std::memcpy(fmt_data->dim_name[i], dim_name[i], sizeof(fmt_data->dim_name[i]));
        fmt_data->dim_scaling[i] = dim_scaling[i];
        fmt_data->stride[i] = stride[i];
    }
    fmt_data->offset = offset;

    // The per-frequency arrays are optional, but they are never empty, so their
    // first element can encode "unset" (the struct was zeroed above, which is
    // already the sentinel for freq_upchan_factor). The byte format stores a
    // single length, so the arrays that are set must agree on it.
    const std::vector<int> coarse_freq =
        this->has_coarse_freq() ? this->get_coarse_freq() : std::vector<int>();
    const std::vector<int> freq_upchan_factor =
        this->has_freq_upchan_factor() ? this->get_freq_upchan_factor() : std::vector<int>();
    const std::vector<int> freq_upchan_index =
        this->has_freq_upchan_index() ? this->get_freq_upchan_index() : std::vector<int>();

    const std::size_t nfreq =
        std::max({coarse_freq.size(), freq_upchan_factor.size(), freq_upchan_index.size()});
    for (const std::vector<int>* const array :
         {&coarse_freq, &freq_upchan_factor, &freq_upchan_index})
        if (!array->empty() && array->size() != nfreq)
            throw std::runtime_error(fmt::format(
                "Cannot serialize chordMetadata: the per-frequency arrays have different "
                "lengths (coarse_freq={:d}, freq_upchan_factor={:d}, freq_upchan_index={:d}; "
                "0 means unset)",
                coarse_freq.size(), freq_upchan_factor.size(), freq_upchan_index.size()));
    if (nfreq > std::size_t(CHORD_META_MAX_FREQ))
        throw std::runtime_error(fmt::format(
            "Cannot serialize chordMetadata: nfreq={:d} exceeds CHORD_META_MAX_FREQ={:d}", nfreq,
            CHORD_META_MAX_FREQ));

    fmt_data->nfreq = int32_t(nfreq);
    if (coarse_freq.empty())
        fmt_data->coarse_freq[0] = -1;
    else
        std::copy_n(coarse_freq.data(), nfreq, fmt_data->coarse_freq);
    if (!freq_upchan_factor.empty())
        std::copy_n(freq_upchan_factor.data(), nfreq, fmt_data->freq_upchan_factor);
    if (freq_upchan_index.empty())
        fmt_data->freq_upchan_index[0] = -1;
    else
        std::copy_n(freq_upchan_index.data(), nfreq, fmt_data->freq_upchan_index);

    if (this->has_stream_ids()) {
        const std::vector<uint32_t> stream_ids = this->get_stream_ids();
        if (stream_ids.size() > std::size_t(CHORD_META_MAX_STREAM_IDS))
            throw std::runtime_error(
                fmt::format("Cannot serialize chordMetadata: {:d} stream IDs exceed "
                            "CHORD_META_MAX_STREAM_IDS={:d}",
                            stream_ids.size(), CHORD_META_MAX_STREAM_IDS));
        fmt_data->num_stream_ids = int32_t(stream_ids.size());
        std::copy_n(stream_ids.data(), stream_ids.size(), fmt_data->stream_ids);
    } else {
        fmt_data->num_stream_ids = -1;
    }

    if (this->has_rfi_frame_excision_enabled())
        fmt_data->rfi_frame_excision_enabled = this->get_rfi_frame_excision_enabled() ? 1 : 0;
    else
        fmt_data->rfi_frame_excision_enabled = -1;
    if (this->has_rfi_frame_excision_thresholds()) {
        const std::vector<std::array<float, 2>> thresholds =
            this->get_rfi_frame_excision_thresholds();
        if (thresholds.size() > std::size_t(MAX_NUM_RFI_THRESHOLDS))
            throw std::runtime_error(
                fmt::format("Cannot serialize chordMetadata: {:d} RFI frame excision thresholds "
                            "exceed MAX_NUM_RFI_THRESHOLDS={:d}",
                            thresholds.size(), MAX_NUM_RFI_THRESHOLDS));
        fmt_data->num_rfi_frame_excision_thresholds = int32_t(thresholds.size());
        for (std::size_t i = 0; i < thresholds.size(); ++i) {
            fmt_data->rfi_frame_excision_thresholds[i][0] = thresholds[i][0];
            fmt_data->rfi_frame_excision_thresholds[i][1] = thresholds[i][1];
        }
    } else {
        fmt_data->num_rfi_frame_excision_thresholds = -1;
    }

    if (this->has_rfi_flagged_samples())
        fmt_data->rfi_flagged_samples = this->get_rfi_flagged_samples();
    else
        fmt_data->rfi_flagged_samples = -1;
    if (this->has_lost_timesamples())
        fmt_data->lost_timesamples = this->get_lost_timesamples();
    else
        fmt_data->lost_timesamples = -1;

    if (this->has_stream_id())
        fmt_data->stream_id = this->get_stream_id();
    else
        fmt_data->stream_id = stream_t{uint64_t(-1)};
    if (this->has_dataset_id()) {
        const std::string dataset_id_str = this->get_dataset_id().to_string();
        if (dataset_id_str.size() != sizeof(fmt_data->dataset_id))
            throw std::runtime_error(
                fmt::format("Cannot serialize chordMetadata: the stringified dataset id has {:d} "
                            "characters, expected {:d}",
                            dataset_id_str.size(), sizeof(fmt_data->dataset_id)));
        std::copy_n(dataset_id_str.data(), sizeof(fmt_data->dataset_id), fmt_data->dataset_id);
    }

    if (this->has_beam_coord()) {
        fmt_data->have_beam_coord = 1;
        fmt_data->beam_coord = this->get_beam_coord();
    }

    if (this->has_first_packet_recv_time())
        fmt_data->first_packet_recv_time = this->get_first_packet_recv_time();
    else
        fmt_data->first_packet_recv_time = timeval{0, 0}; // unix epoch, 1970

    // TODO: this misses dish_positions etc
    return sizeof(chordMetadataFormat);
}

namespace {

// The json keys that describe the array structure. Everything else in the json
// is a json metadata entry; from_json copies those verbatim so that newly added
// entries do not have to be listed anywhere.
constexpr const char* KEY_MAX_DIM = "max_dim";
constexpr const char* KEY_MAX_DIMNAME = "max_dimname";
constexpr const char* KEY_MAX_FREQ = "max_freq";
constexpr const char* KEY_NAME = "name";
constexpr const char* KEY_TYPE = "type";
constexpr const char* KEY_DIMS = "dims";
constexpr const char* KEY_DIM = "dim";
constexpr const char* KEY_DIM_NAME = "dim_name";
constexpr const char* KEY_DIM_SCALING = "dim_scaling";
constexpr const char* KEY_STRIDE = "stride";
constexpr const char* KEY_OFFSET = "offset";

const std::array<const char*, 11> structural_keys = {
    KEY_MAX_DIM, KEY_MAX_DIMNAME, KEY_MAX_FREQ,    KEY_NAME,   KEY_TYPE,  KEY_DIMS,
    KEY_DIM,     KEY_DIM_NAME,    KEY_DIM_SCALING, KEY_STRIDE, KEY_OFFSET};

bool is_structural_key(const std::string& key) {
    return std::find_if(structural_keys.begin(), structural_keys.end(),
                        [&key](const char* const k) { return key == k; })
           != structural_keys.end();
}

} // namespace

nlohmann::json chordMetadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const chordMetadata& m) {
    assert(j.empty());

    // The json metadata is shared mutable state, and the plain fields are read
    // here as well, so hold the lock for the whole function.
    std::lock_guard<std::mutex> guard(m.lock);

    j = m.metadata;

    j.emplace(KEY_MAX_DIM, CHORD_META_MAX_DIM);
    j.emplace(KEY_MAX_DIMNAME, CHORD_META_MAX_DIMNAME);
    j.emplace(KEY_MAX_FREQ, CHORD_META_MAX_FREQ);

    // `dims` is -1 while the array description is unset; write empty arrays in
    // that case instead of constructing vectors from an inverted range.
    const int ndims = std::max(m.dims, 0);

    j.emplace(KEY_NAME, std::string(m.name, strnlen(m.name, sizeof(m.name))));
    j.emplace(KEY_TYPE, m.type);
    j.emplace(KEY_DIMS, m.dims);
    j.emplace(KEY_DIM, std::vector<int>(m.dim, m.dim + ndims));
    std::vector<std::string> dimnames;
    for (int i = 0; i < ndims; i++)
        dimnames.push_back(
            std::string(m.dim_name[i], strnlen(m.dim_name[i], sizeof(m.dim_name[i]))));
    j.emplace(KEY_DIM_NAME, dimnames);
    j.emplace(KEY_DIM_SCALING, std::vector<int64_t>(m.dim_scaling, m.dim_scaling + ndims));
    j.emplace(KEY_STRIDE, std::vector<int64_t>(m.stride, m.stride + ndims));
    j.emplace(KEY_OFFSET, m.offset);
    // TODO: this misses dish_positions etc
}

void from_json(const nlohmann::json& j, chordMetadata& m) {
    if (!j.is_object())
        throw std::runtime_error("Cannot deserialize chordMetadata: the json is not an object");

    std::lock_guard<std::mutex> guard(m.lock);
    if (!m.metadata.empty())
        throw std::runtime_error("Cannot deserialize chordMetadata: the target is not empty");

    if (j.at(KEY_MAX_DIM) != CHORD_META_MAX_DIM || j.at(KEY_MAX_DIMNAME) != CHORD_META_MAX_DIMNAME
        || j.at(KEY_MAX_FREQ) != CHORD_META_MAX_FREQ)
        throw std::runtime_error(
            "Cannot deserialize chordMetadata: the json was written with incompatible limits");

    // Copy every entry that is not part of the array description. Listing the
    // metadata keys here instead would silently drop every key that is added to
    // the setters but forgotten here.
    for (const auto& item : j.items())
        if (!is_structural_key(item.key()))
            m.metadata[item.key()] = item.value();

    m.set_name(j.at(KEY_NAME).template get<std::string>());
    m.type = j.at(KEY_TYPE).template get<kotekan::DataType>();

    const int dims = j.at(KEY_DIMS).template get<int>();
    if (dims < -1 || dims > CHORD_META_MAX_DIM)
        throw std::runtime_error("Cannot deserialize chordMetadata: dims=" + std::to_string(dims)
                                 + " is not in [-1, " + std::to_string(CHORD_META_MAX_DIM) + "]");
    m.dims = dims;
    const std::size_t ndims = std::size_t(std::max(dims, 0));

    const std::vector<int> extents = j.at(KEY_DIM).template get<std::vector<int>>();
    const std::vector<std::string> dimnames =
        j.at(KEY_DIM_NAME).template get<std::vector<std::string>>();
    const std::vector<int64_t> dim_scalings =
        j.at(KEY_DIM_SCALING).template get<std::vector<int64_t>>();
    const std::vector<int64_t> strides = j.at(KEY_STRIDE).template get<std::vector<int64_t>>();
    if (extents.size() != ndims || dimnames.size() != ndims || dim_scalings.size() != ndims
        || strides.size() != ndims)
        throw std::runtime_error(
            "Cannot deserialize chordMetadata: the array description has "
            + std::to_string(extents.size()) + " extents, " + std::to_string(dimnames.size())
            + " dimension names, " + std::to_string(dim_scalings.size()) + " dimension scalings, "
            + std::to_string(strides.size()) + " strides, but dims=" + std::to_string(dims));

    for (std::size_t i = 0; i < ndims; i++) {
        m.dim[i] = extents[i];
        m.set_dimension_name(int(i), dimnames[i]);
        m.dim_scaling[i] = dim_scalings[i];
        m.stride[i] = strides[i];
    }
    m.offset = j.at(KEY_OFFSET);
    // TODO: this misses dish_positions etc
}


bool metadata_is_chord(Buffer* buf, int) {
    return buf && buf->metadata_pool && (buf->metadata_pool->type_name == "chordMetadata");
}

bool metadata_is_chord(const std::shared_ptr<metadataObject>& mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "chordMetadata");
}

bool metadata_is_chord(const std::shared_ptr<const metadataObject>& mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "chordMetadata");
}

std::shared_ptr<chordMetadata> get_chord_metadata(const std::shared_ptr<metadataObject>& mc) {
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

std::shared_ptr<const chordMetadata>
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

std::shared_ptr<chordMetadata> get_chord_metadata(Buffer* buf, int frame_id) {
    if (!buf || frame_id < 0 || frame_id >= (int)buf->metadata.size())
        return std::shared_ptr<chordMetadata>();
    std::shared_ptr<metadataObject> meta = buf->metadata.at(frame_id);
    return get_chord_metadata(meta);
}
