#include "BeamMetadata.hpp"

#include "chordMetadata.hpp" // for CHORD_META_MAX_FREQ
#include "factory.hpp"       // for REGISTER_TYPE_WITH_FACTORY

#include <assert.h> // for assert
#include <cstring>  // for memset
#include <json.hpp> // for json
#include <memory>   // for allocator, shared_ptr, dynamic_pointer_cast, __shared_ptr_a...
#include <string>   // for basic_string

REGISTER_TYPE_WITH_FACTORY(metadataObject, BeamMetadata);

struct BeamMetadataFormat {
    int64_t fpga_seq_start;
    struct timespec ctime;
    int nfreq;
    int coarse_freq[CHORD_META_MAX_FREQ];
    dset_id_t dataset_id;
    uint32_t beam_number;
    float ra;
    float dec;
    uint32_t scaling;
};

void BeamMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto beam_other = std::dynamic_pointer_cast<const BeamMetadata>(other);
    assert(beam_other);
    *this = *beam_other;
}

size_t BeamMetadata::get_serialized_size() {
    return sizeof(BeamMetadataFormat);
}

size_t BeamMetadata::set_from_bytes(const char* bytes, [[maybe_unused]] size_t length) {
    size_t sz = get_serialized_size();
    assert(length >= sz);
    const BeamMetadataFormat* fmt = reinterpret_cast<const BeamMetadataFormat*>(bytes);
    fpga_seq_start = fmt->fpga_seq_start;
    ctime = fmt->ctime;
    coarse_freq.resize(fmt->nfreq);
    // TODO: copy on nfreq?
    for (int i = 0; i < fmt->nfreq; i++) {
        coarse_freq[i] = fmt->coarse_freq[i];
    }
    dataset_id = fmt->dataset_id;
    beam_number = fmt->beam_number;
    ra = fmt->ra;
    dec = fmt->dec;
    scaling = fmt->scaling;
    return sz;
}

size_t BeamMetadata::serialize(char* bytes) {
    size_t sz = get_serialized_size();
    BeamMetadataFormat* fmt = reinterpret_cast<BeamMetadataFormat*>(bytes);
    // Zero first so struct padding is not written out uninitialized
    memset(bytes, 0, sz);
    fmt->fpga_seq_start = fpga_seq_start;
    fmt->ctime = ctime;
    fmt->nfreq = static_cast<int>(coarse_freq.size());
    for (int i = 0; i < static_cast<int>(coarse_freq.size()); i++) {
        fmt->coarse_freq[i] = coarse_freq[i];
    }
    fmt->dataset_id = dataset_id;
    fmt->beam_number = beam_number;
    fmt->ra = ra;
    fmt->dec = dec;
    fmt->scaling = scaling;
    return sz;
}

nlohmann::json BeamMetadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const BeamMetadata& m) {
    assert(j.empty());

    j.emplace("fpga_seq_start", m.fpga_seq_start);
    j.emplace("ctime", m.ctime);
    j.emplace("coarse_freq", m.coarse_freq);
    j.emplace("dataset_id", m.dataset_id);
    j.emplace("beam_number", m.beam_number);
    j.emplace("ra", m.ra);
    j.emplace("dec", m.dec);
    j.emplace("scaling", m.scaling);
}

void from_json(const nlohmann::json& j, BeamMetadata& m) {
    m.fpga_seq_start = j.at("fpga_seq_start");
    m.ctime = j.at("ctime");
    m.coarse_freq = j.at("coarse_freq").template get<std::vector<int>>();
    m.dataset_id = j.at("dataset_id");
    m.beam_number = j.at("beam_number");
    m.ra = j.at("ra");
    m.dec = j.at("dec");
    m.scaling = j.at("scaling");
}
