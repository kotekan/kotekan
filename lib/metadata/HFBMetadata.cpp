#include "HFBMetadata.hpp"

#include "visUtil.hpp"

#include <assert.h>

REGISTER_TYPE_WITH_FACTORY(metadataObject, HFBMetadata);

struct HFBMetadataFormat {
    int64_t fpga_seq_start;
    struct timespec ctime;
    freq_id_t freq_id;
    uint64_t fpga_seq_total;
    uint64_t fpga_seq_length;
    uint32_t num_beams;
    uint32_t num_subfreq;
    dset_id_t dataset_id;
};

size_t HFBMetadata::get_serialized_size() {
    return sizeof(HFBMetadataFormat);
}

size_t HFBMetadata::set_from_bytes(const char* bytes, size_t length) {
    size_t sz = get_serialized_size();
    assert(length >= sz);
    const HFBMetadataFormat* fmt = reinterpret_cast<const HFBMetadataFormat*>(bytes);
    fpga_seq_start = fmt->fpga_seq_start;
    ctime = fmt->ctime;
    freq_id = fmt->freq_id;
    fpga_seq_length = fmt->fpga_seq_length;
    fpga_seq_total = fmt->fpga_seq_total;
    num_beams = fmt->num_beams;
    num_subfreq = fmt->num_subfreq;
    dataset_id = fmt->dataset_id;
    return sz;
}

size_t HFBMetadata::serialize(char* bytes) {
    size_t sz = get_serialized_size();
    HFBMetadataFormat* fmt = reinterpret_cast<HFBMetadataFormat*>(bytes);
    fmt->fpga_seq_start = fpga_seq_start;
    fmt->ctime = ctime;
    fmt->freq_id = freq_id;
    fmt->fpga_seq_length = fpga_seq_length;
    fmt->fpga_seq_total = fpga_seq_total;
    fmt->num_beams = num_beams;
    fmt->num_subfreq = num_subfreq;
    fmt->dataset_id = dataset_id;
    return sz;
}

nlohmann::json HFBMetadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const HFBMetadata& m) {
    j["fpga_seq_start"] = m.fpga_seq_start;
    j["ctime"] = m.ctime;
    j["freq_id"] = m.freq_id;
    j["fpga_seq_length"] = m.fpga_seq_length;
    j["fpga_seq_total"] = m.fpga_seq_total;
    j["num_beams"] = m.num_beams;
    j["num_subfreq"] = m.num_subfreq;
    j["dataset_id"] = m.dataset_id;
}

void from_json(const nlohmann::json& j, HFBMetadata& m) {
    m.fpga_seq_start = j["fpga_seq_start"];
    m.ctime = j["ctime"];
    m.freq_id = j["freq_id"];
    m.fpga_seq_length = j["fpga_seq_length"];
    m.fpga_seq_total = j["fpga_seq_total"];
    m.num_beams = j["num_beams"];
    m.num_subfreq = j["num_subfreq"];
    m.dataset_id = j["dataset_id"];
}
