#include "HFBMetadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY
#include "visUtil.hpp" // Needed for timespec conversion  // IWYU pragma: keep

#include <assert.h> // for assert
#include <string>   // for basic_string

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

void HFBMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto hfb_other = std::dynamic_pointer_cast<const HFBMetadata>(other);
    assert(hfb_other);
    *this = *hfb_other;
}

size_t HFBMetadata::get_serialized_size() {
    return sizeof(HFBMetadataFormat);
}

size_t HFBMetadata::set_from_bytes(const char* bytes, [[maybe_unused]] size_t length) {
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
    assert(j.empty());

    j.emplace("fpga_seq_start", m.fpga_seq_start);
    j.emplace("ctime", m.ctime);
    j.emplace("freq_id", m.freq_id);
    j.emplace("fpga_seq_length", m.fpga_seq_length);
    j.emplace("fpga_seq_total", m.fpga_seq_total);
    j.emplace("num_beams", m.num_beams);
    j.emplace("num_subfreq", m.num_subfreq);
    j.emplace("dataset_id", m.dataset_id);
}

void from_json(const nlohmann::json& j, HFBMetadata& m) {
    m.fpga_seq_start = j.at("fpga_seq_start");
    m.ctime = j.at("ctime");
    m.freq_id = j.at("freq_id");
    m.fpga_seq_length = j.at("fpga_seq_length");
    m.fpga_seq_total = j.at("fpga_seq_total");
    m.num_beams = j.at("num_beams");
    m.num_subfreq = j.at("num_subfreq");
    m.dataset_id = j.at("dataset_id");
}
