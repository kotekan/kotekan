#include "BeamMetadata.hpp"

#include "visUtil.hpp"

#include <assert.h>

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

size_t BeamMetadata::get_serialized_size() {
    return sizeof(BeamMetadataFormat);
}

size_t BeamMetadata::set_from_bytes(const char* bytes, size_t length) {
    size_t sz = get_serialized_size();
    assert(length >= sz);
    const BeamMetadataFormat* fmt = reinterpret_cast<const BeamMetadataFormat*>(bytes);
    fpga_seq_start = fmt->fpga_seq_start;
    ctime = fmt->ctime;
    nfreq = fmt->nfreq;
    // TODO: copy on nfreq?
    for (int i = 0 ; i < CHORD_META_MAX_FREQ ; i++) {
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
    fmt->fpga_seq_start = fpga_seq_start;
    fmt->ctime = ctime;
    fmt->nfreq = nfreq;
    // TODO: copy on nfreq?
    for (int i = 0 ; i < CHORD_META_MAX_FREQ ; i++) {
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
    j["fpga_seq_start"] = m.fpga_seq_start;
    j["ctime"] = m.ctime;
    j["nfreq"] = m.nfreq;
    // TODO: either make coarse_freq a std::vector or make it a std::array
    j["coarse_freq"] = std::vector<int>(m.coarse_freq, m.coarse_freq + m.nfreq);
    j["dataset_id"] = m.dataset_id;
    j["beam_number"] = m.beam_number;
    j["ra"] = m.ra;
    j["dec"] = m.dec;
    j["scaling"] = m.scaling;
}

void from_json(const nlohmann::json& j, BeamMetadata& m) {
    m.fpga_seq_start = j["fpga_seq_start"];
    m.ctime = j["ctime"];
    m.nfreq = j["nfreq"];
    const auto& v = j["coarse_freq"].template get<std::vector<int>>();
    std::copy(v.begin(), v.end(), m.coarse_freq);
    m.dataset_id = j["dataset_id"];
    m.beam_number = j["beam_number"];
    m.ra = j["ra"];
    m.dec = j["dec"];
    m.scaling = j["scaling"];
}
