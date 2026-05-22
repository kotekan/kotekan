#include "chordMetadata.hpp"

#include <string.h>     // for strncmp, strnlen, memset, strncpy
#include <json.hpp>     // for operator==, json
#include <algorithm>    // for copy_n, copy, max
#include <cstring>      // for memset
#include <type_traits>  // for is_pod_v

#include "Symbol.hpp"   // for Symbol
#include "factory.hpp"  // for REGISTER_TYPE_WITH_FACTORY

REGISTER_TYPE_WITH_FACTORY(metadataObject, chordMetadata);

chordMetadata::chordMetadata() :
    type(kotekan::unknown_type), dims(-1), offset(0), ndishes(-1), n_dish_locations_ew(-1),
    n_dish_locations_ns(-1), dish_index(nullptr) {
    name[0] = '\0';
    for (int d = 0; d < CHORD_META_MAX_DIM; ++d) {
        dim[d] = -1;
        dim_name[d][0] = '\0';
        stride[d] = -1;
    }
}

bool chordMetadata::operator==(const chordMetadata& other) const {
    if (this == &other)
        return true;

    std::scoped_lock<std::mutex, std::mutex> lock(this->lock, other.lock);

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
        if (stride[d] != other.stride[d])
            return false;
    if (offset != other.offset)
        return false;

    // TODO: this misses dish_positions etc
    return metadata == other.metadata;
}

void chordMetadata::check_frame_desc(
    const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) const {

    bool failed = false;

    if (strncmp(this->name, frame_desc->get_quantity_name().get_c_string(), sizeof(this->name))
        != 0) {
        ERROR("Names differ: {:s} != {:s}",
              std::string(this->name, strnlen(this->name, sizeof(this->name))),
              frame_desc->get_quantity_name());
        failed = true;
    }
    if (this->type != frame_desc->get_value_datatype()) {
        ERROR("Types differ for {:s}: {:s} != {:s}", frame_desc->get_quantity_name(),
              kotekan::type_to_string(this->type),
              kotekan::type_to_string(frame_desc->get_value_datatype()));
        failed = true;
    }
    if (size_t(this->dims) != frame_desc->get_rank()) {
        ERROR("Ranks differ for {:s}: {:d} != {:d}", frame_desc->get_quantity_name(), this->dims,
              frame_desc->get_rank());
        failed = true;
    }
    for (int d = this->dims - 1; d >= 0; --d) {

        if (strncmp(this->dim_name[d], frame_desc->get_dimname(d).get_c_string(),
                    sizeof this->dim_name[d])
            != 0) {
            ERROR("Dim_name[{:d}] differs for {:s}: {:s} != {:s}", d,
                  frame_desc->get_quantity_name(),
                  std::string(this->dim_name[d],
                              strnlen(this->dim_name[d], sizeof(this->dim_name[d]))),
                  frame_desc->get_dimname(d));
            failed = true;
        }

        if (this->dim[d] != frame_desc->get_extent(d)) {
            ERROR("Dim[{:d}] differs for {:s}: {:d} != {:d}", d, frame_desc->get_quantity_name(),
                  this->dim[d], frame_desc->get_extent(d));
            failed = true;
        }
        if (this->stride[d] != frame_desc->get_stride(d)) {
            ERROR("Stride[{:d}] differs for {:s}: {:d} != {:d}", d, frame_desc->get_quantity_name(),
                  this->stride[d], frame_desc->get_stride(d));
            failed = true;
        }
    }

    if (failed)
        FATAL_ERROR("Inconsistent array description between CHORDMetadata and FrameDesc");
}

void chordMetadata::set_from_frame_desc(
    const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) {
    set_name(frame_desc->get_quantity_name());
    this->type = frame_desc->get_value_datatype();
    this->dims = frame_desc->get_rank();
    for (int d = this->dims - 1; d >= 0; --d) {
        set_array_dimension(d, frame_desc->get_extent(d), frame_desc->get_dimname(d));
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
    std::scoped_lock<std::mutex, std::mutex> lock(this->lock, chord_other->lock);
    *this = *chord_other;
}

struct chordMetadataFormat {
    int32_t max_dim;
    int32_t max_dimname;
    int32_t max_freq;

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
    int64_t stride[CHORD_META_MAX_DIM];
    int64_t offset;

    // Per-frequency arrays
    int32_t nfreq;

    // frequencies -- integer (0-2047) identifier for FPGA coarse frequencies
    int32_t coarse_freq[CHORD_META_MAX_FREQ];

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    int32_t freq_upchan_factor[CHORD_META_MAX_FREQ];

    // the upchannelization index for each frequency (0 ... freq_upchan_factor - 1)
    int32_t freq_upchan_index[CHORD_META_MAX_FREQ];

    uint32_t rfi_num_bad_inputs;
    int32_t rfi_flagged_samples;
    int32_t lost_timesamples;

    chordMetadata::beamCoord beam_coord;
    static_assert(std::is_pod_v<chordMetadata::beamCoord> == true,
                  "beamCoord containes C++ only data");

    stream_t stream_id;
    static_assert(std::is_pod_v<stream_t> == true, "stream_t containes C++ only data");

    // cannot use dset_id_t here since Hash is not a pod (has a constructor)
    char dataset_id[32];

    timeval first_packet_recv_time;
};

size_t chordMetadata::get_serialized_size() {
    return sizeof(chordMetadataFormat);
}

size_t chordMetadata::set_from_bytes(const char* bytes, [[maybe_unused]] size_t length) {
    (void)length;
    assert(length >= get_serialized_size());
    assert(length >= sizeof(chordMetadataFormat));

    const chordMetadataFormat* fmt = reinterpret_cast<const chordMetadataFormat*>(bytes);

    if (fmt->frame_counter != -1)
        this->set_frame_counter(fmt->frame_counter);
    if (fmt->fpga_seq_num != -1)
        this->set_fpga_seq_num(fmt->fpga_seq_num);
    if (fmt->time_downsampling_fpga != -1)
        this->set_time_downsampling_fpga(fmt->time_downsampling_fpga);
    for (int i = 0; i < CHORD_META_MAX_DIMNAME; i++) {
        name[i] = fmt->name[i];
    }
    type = (kotekan::DataType)fmt->type;
    assert(CHORD_META_MAX_DIM == fmt->max_dim);
    assert(CHORD_META_MAX_DIMNAME == fmt->max_dimname);
    assert(CHORD_META_MAX_FREQ == fmt->max_freq);
    dims = fmt->dims;
    assert(dims < CHORD_META_MAX_DIM);
    for (int i = 0; i < dims; i++) {
        dim[i] = fmt->dim[i];
        for (int j = 0; j < CHORD_META_MAX_DIMNAME; j++) {
            dim_name[i][j] = fmt->dim_name[i][j];
        }
        stride[i] = fmt->stride[i];
    }
    offset = fmt->offset;
    const int nfreq = fmt->nfreq;
    assert(nfreq < CHORD_META_MAX_FREQ);
    if (fmt->freq_upchan_factor[0] != 0) // 0 should be an invalid value
        this->set_freq_upchan_factor(
            std::vector<int>(fmt->freq_upchan_factor, fmt->freq_upchan_factor + nfreq));
    if (fmt->freq_upchan_index[0] != -1) // -1 should be an invalid value
        this->set_freq_upchan_index(
            std::vector<int>(fmt->freq_upchan_index, fmt->freq_upchan_index + nfreq));
    if (fmt->coarse_freq[0] != -1) // -1 is an invalid frequency index
        this->set_coarse_freq(std::vector<int>(fmt->coarse_freq, fmt->coarse_freq + nfreq));

    if (fmt->rfi_num_bad_inputs != uint32_t(-1))
        this->set_rfi_num_bad_inputs(fmt->rfi_num_bad_inputs);
    if (fmt->rfi_flagged_samples != -1)
        this->set_rfi_flagged_samples(fmt->rfi_flagged_samples);
    if (fmt->lost_timesamples != -1)
        this->set_lost_timesamples(fmt->lost_timesamples);

    if (fmt->stream_id.id != uint64_t(-1))
        this->set_stream_id(fmt->stream_id);
    if (fmt->dataset_id[0] != '\0')
        this->set_dataset_id(
            dset_id_t::from_string(std::string(fmt->dataset_id, sizeof(fmt->dataset_id))));

    if (fmt->beam_coord.scaling[0] != 0)
        this->set_beam_coord(fmt->beam_coord);

    if (fmt->first_packet_recv_time.tv_sec != 0) // sometime in 1970
        this->set_first_packet_recv_time(fmt->first_packet_recv_time);

    // TODO: this misses dish_positions etc
    return sizeof(chordMetadataFormat);
}

size_t chordMetadata::serialize(char* bytes) {
    chordMetadataFormat* fmt = reinterpret_cast<chordMetadataFormat*>(bytes);
    memset(fmt, 0, sizeof(chordMetadataFormat));

    fmt->max_dim = CHORD_META_MAX_DIM;
    fmt->max_dimname = CHORD_META_MAX_DIMNAME;
    fmt->max_freq = CHORD_META_MAX_FREQ;

    if (this->has_frame_counter())
        fmt->frame_counter = this->get_frame_counter();
    else
        fmt->frame_counter = -1;
    if (this->has_fpga_seq_num())
        fmt->fpga_seq_num = this->get_fpga_seq_num();
    else
        fmt->fpga_seq_num = -1;
    if (this->has_time_downsampling_fpga())
        fmt->time_downsampling_fpga = this->get_time_downsampling_fpga();
    else
        fmt->time_downsampling_fpga = -1;
    for (int i = 0; i < CHORD_META_MAX_DIMNAME; i++) {
        fmt->name[i] = name[i];
    }
    fmt->type = (int32_t)type;
    fmt->dims = dims;
    for (int i = 0; i < dims; i++) {
        fmt->dim[i] = dim[i];
        for (int j = 0; j < CHORD_META_MAX_DIMNAME; j++) {
            fmt->dim_name[i][j] = dim_name[i][j];
        }
        fmt->stride[i] = stride[i];
    }
    fmt->offset = offset;
    fmt->nfreq = this->get_nfreq();
    assert(fmt->nfreq < CHORD_META_MAX_FREQ);
    if (this->has_freq_upchan_factor())
        std::copy_n(this->get_freq_upchan_factor().data(), this->get_nfreq(),
                    fmt->freq_upchan_factor);
    if (this->has_freq_upchan_index())
        std::copy_n(this->get_freq_upchan_index().data(), this->get_nfreq(),
                    fmt->freq_upchan_index);
    else
        std::memset(
            fmt->freq_upchan_index, -1,
            this->get_nfreq()
                * sizeof(fmt->freq_upchan_index[0])); // set bytes not ints but is ok for now
    if (this->has_coarse_freq())
        std::copy_n(this->get_coarse_freq().data(), this->get_nfreq(), fmt->coarse_freq);
    else
        std::memset(fmt->coarse_freq, -1,
                    this->get_nfreq()
                        * sizeof(fmt->coarse_freq[0])); // set bytes not ints but is ok for now

    if (this->has_rfi_num_bad_inputs())
        fmt->rfi_num_bad_inputs = this->get_rfi_num_bad_inputs();
    else
        fmt->rfi_num_bad_inputs = uint32_t(-1);
    if (this->has_rfi_flagged_samples())
        fmt->rfi_flagged_samples = this->get_rfi_flagged_samples();
    else
        fmt->rfi_flagged_samples = -1;
    if (this->has_lost_timesamples())
        fmt->lost_timesamples = this->get_lost_timesamples();
    else
        fmt->lost_timesamples = -1;

    if (this->has_stream_id())
        fmt->stream_id = this->get_stream_id();
    else
        fmt->stream_id = stream_t{uint64_t(-1)};
    if (this->has_dataset_id()) {
        const std::string dataset_id_str = this->get_dataset_id().to_string();
        assert(dataset_id_str.size() == sizeof(fmt->dataset_id)
               && "Sized of strigified hash is not 32");
        std::copy_n(dataset_id_str.data(), sizeof(fmt->dataset_id), fmt->dataset_id);
    }

    if (this->has_beam_coord())
        fmt->beam_coord = this->get_beam_coord();

    if (this->has_first_packet_recv_time())
        fmt->first_packet_recv_time = this->get_first_packet_recv_time();
    else
        fmt->first_packet_recv_time = timeval{0, 0}; // unix epoch, 1970

    // TODO: this misses dish_positions etc
    return sizeof(chordMetadataFormat);
}

nlohmann::json chordMetadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const chordMetadata& m) {
    assert(j.empty());

    j = m.metadata;

    j.emplace("max_dim", CHORD_META_MAX_DIM);
    j.emplace("max_dimname", CHORD_META_MAX_DIMNAME);
    j.emplace("max_freq", CHORD_META_MAX_FREQ);

    j.emplace("name", std::string(m.name, strnlen(m.name, sizeof(m.name))));
    j.emplace("type", m.type);
    j.emplace("dims", m.dims);
    j.emplace("dim", std::vector<int>(m.dim, m.dim + m.dims));
    std::vector<std::string> dimnames;
    for (int i = 0; i < m.dims; i++)
        dimnames.push_back(
            std::string(m.dim_name[i], strnlen(m.dim_name[i], sizeof(m.dim_name[i]))));
    j.emplace("dim_name", dimnames);
    j.emplace("stride", std::vector<int64_t>(m.stride, m.stride + m.dims));
    j.emplace("offset", m.offset);
    // TODO: this misses dish_positions etc
}

void from_json(const nlohmann::json& j, chordMetadata& m) {
    // TODO once everything is stored in json, can just copy in the json given
    assert(m.metadata.empty());

    if (j.contains(jsonMetadata::BEAM_COORD))
        m.metadata.emplace(jsonMetadata::BEAM_COORD, j.at(jsonMetadata::BEAM_COORD));
    if (j.contains(jsonMetadata::FPGA_SEQ_NUM))
        m.metadata.emplace(jsonMetadata::FPGA_SEQ_NUM, j.at(jsonMetadata::FPGA_SEQ_NUM));
    if (j.contains(jsonMetadata::TIME_DOWNSAMPLING_FPGA))
        m.metadata.emplace(jsonMetadata::TIME_DOWNSAMPLING_FPGA,
                           j.at(jsonMetadata::TIME_DOWNSAMPLING_FPGA));
    if (j.contains(jsonMetadata::COARSE_FREQ))
        m.metadata.emplace(jsonMetadata::COARSE_FREQ, j.at(jsonMetadata::COARSE_FREQ));
    if (j.contains(jsonMetadata::DATASET_ID))
        m.metadata.emplace(jsonMetadata::DATASET_ID, j.at(jsonMetadata::DATASET_ID));
    if (j.contains(jsonMetadata::RFI_NUM_BAD_INPUTS))
        m.metadata.emplace(jsonMetadata::RFI_NUM_BAD_INPUTS,
                           j.at(jsonMetadata::RFI_NUM_BAD_INPUTS));
    if (j.contains(jsonMetadata::RFI_FLAGGED_SAMPLES))
        m.metadata.emplace(jsonMetadata::RFI_FLAGGED_SAMPLES,
                           j.at(jsonMetadata::RFI_FLAGGED_SAMPLES));
    if (j.contains(jsonMetadata::LOST_TIMESAMPLES))
        m.metadata.emplace(jsonMetadata::LOST_TIMESAMPLES, j.at(jsonMetadata::LOST_TIMESAMPLES));
    if (j.contains(jsonMetadata::STREAM_ID))
        m.metadata.emplace(jsonMetadata::STREAM_ID, j.at(jsonMetadata::STREAM_ID));
    if (j.contains(jsonMetadata::FRAME_COUNTER))
        m.metadata.emplace(jsonMetadata::FRAME_COUNTER, j.at(jsonMetadata::FRAME_COUNTER));

    if (j.contains(jsonMetadata::FIRST_PACKET_RECV_TIME))
        m.metadata.emplace(jsonMetadata::FIRST_PACKET_RECV_TIME,
                           j.at(jsonMetadata::FIRST_PACKET_RECV_TIME));

    if (j.contains(jsonMetadata::FREQ_UPCHAN_FACTOR))
        m.metadata.emplace(jsonMetadata::FREQ_UPCHAN_FACTOR,
                           j.at(jsonMetadata::FREQ_UPCHAN_FACTOR));
    if (j.contains(jsonMetadata::FREQ_UPCHAN_INDEX))
        m.metadata.emplace(jsonMetadata::FREQ_UPCHAN_INDEX, j.at(jsonMetadata::FREQ_UPCHAN_INDEX));

    assert(j.at("max_dim") == CHORD_META_MAX_DIM);
    assert(j.at("max_dimname") == CHORD_META_MAX_DIMNAME);
    assert(j.at("max_freq") == CHORD_META_MAX_FREQ);

    // GCC helpfully tries to warn us that the destination string may end up not
    // NUL-terminated, which we know.
#pragma GCC diagnostic push
#if GCC_VERSION > 80000
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
    strncpy(m.name, j.at("name").template get<std::string>().c_str(), sizeof(m.name));
#pragma GCC diagnostic pop
    m.type = j.at("type").template get<kotekan::DataType>();
    m.dims = j.at("dims").template get<int>();
    std::vector<int> extents = j.at("dim").template get<std::vector<int>>();
    std::copy(extents.begin(), extents.end(), m.dim);
    std::vector<std::string> dimnames = j.at("dim_name").template get<std::vector<std::string>>();
    for (int i = 0; i < m.dims; i++)
        strncpy(m.dim_name[i], dimnames.at(i).c_str(), sizeof(m.dim_name[i]));
    std::vector<int64_t> strides = j.at("stride").template get<std::vector<int64_t>>();
    for (int i = 0; i < m.dims; i++)
        m.stride[i] = strides.at(i);
    m.offset = j.at("offset");
    // TODO: this misses dish_positions etc
}
