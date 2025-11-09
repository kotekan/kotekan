#include "N2Metadata.hpp"

#include <string.h> // for size_t, memset

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

REGISTER_TYPE_WITH_FACTORY(metadataObject, N2Metadata);
N2Metadata::N2Metadata() : N2MetadataFormat{0, 0, 0, 0, 0, 0.0, eop_null, 0, 0, 0, 0, 0} {
    ;
}

void N2Metadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    std::shared_ptr<const N2Metadata> o = std::dynamic_pointer_cast<const N2Metadata>(other);
    *this = *o;
}

size_t N2Metadata::get_serialized_size() {
    return sizeof(N2MetadataFormat);
}

size_t N2Metadata::set_from_bytes(const char* bytes, [[maybe_unused]] size_t length) {
    assert(length >= get_serialized_size());
    assert(length >= sizeof(N2MetadataFormat));

    const N2MetadataFormat* fmt = reinterpret_cast<const N2MetadataFormat*>(bytes);

    fpga_start_tick = fmt->fpga_start_tick;
    frame_start_time_ns = fmt->frame_start_time_ns;
    frame_length_fpga_ticks = fmt->frame_length_fpga_ticks;

    n_valid_fpga_ticks = fmt->n_valid_fpga_ticks;
    n_rfi_fpga_ticks = fmt->n_rfi_fpga_ticks;

    freq_id = fmt->freq_id; // this is an int in chordMetadata, maybe change later
    freq_Hz = fmt->freq_Hz;
    eop = fmt->eop;

    num_elements = fmt->num_elements;
    num_prod = fmt->num_prod;
    num_ev = fmt->num_ev;
    nfreq = fmt->nfreq;

    return sizeof(N2MetadataFormat);
}

size_t N2Metadata::serialize(char* bytes) {
    N2MetadataFormat* fmt = reinterpret_cast<N2MetadataFormat*>(bytes);
    memset(fmt, 0, sizeof(N2MetadataFormat));

    fmt->fpga_start_tick = fpga_start_tick;
    fmt->frame_start_time_ns = frame_start_time_ns;
    fmt->frame_length_fpga_ticks = frame_length_fpga_ticks;

    fmt->n_valid_fpga_ticks = n_valid_fpga_ticks;
    fmt->n_rfi_fpga_ticks = n_rfi_fpga_ticks;

    fmt->freq_id = freq_id; // this is an int in chordMetadata, maybe change later
    fmt->freq_Hz = freq_Hz;
    fmt->eop = eop;

    fmt->num_elements = num_elements;
    fmt->num_prod = num_prod;
    fmt->num_ev = num_ev;
    fmt->nfreq = nfreq;

    return sizeof(N2MetadataFormat);
}

nlohmann::json N2Metadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const N2Metadata& m) {
    assert(j.empty());

    j.emplace("fpga_start_tick", m.fpga_start_tick);
    j.emplace("frame_start_time_ns", m.frame_start_time_ns);
    j.emplace("frame_length_fpga_ticks", m.frame_length_fpga_ticks);

    j.emplace("n_valid_fpga_ticks", m.n_valid_fpga_ticks);
    j.emplace("n_rfi_fpga_ticks", m.n_rfi_fpga_ticks);

    j.emplace("freq_id", m.freq_id); // this is an int in chordMetadata, maybe change later
    j.emplace("freq_Hz", m.freq_Hz);
    j.emplace("eop", m.eop);

    j.emplace("num_elements", m.num_elements);
    j.emplace("num_prod", m.num_prod);
    j.emplace("num_ev", m.num_ev);
    j.emplace("nfreq", m.nfreq);
}

void from_json(const nlohmann::json& j, N2Metadata& m) {
    m.fpga_start_tick = j.at("fpga_start_tick");
    m.frame_start_time_ns = j.at("frame_start_time_ns");
    m.frame_length_fpga_ticks = j.at("frame_length_fpga_ticks");

    m.n_valid_fpga_ticks = j.at("n_valid_fpga_ticks");
    m.n_rfi_fpga_ticks = j.at("n_rfi_fpga_ticks");

    m.freq_id = j.at("freq_id"); // this is an int in chordMetadata, maybe change later
    m.freq_Hz = j.at("freq_Hz");
    m.eop = j.at("eop");

    m.num_elements = j.at("num_elements");
    m.num_prod = j.at("num_prod");
    m.num_ev = j.at("num_ev");
    m.nfreq = j.at("nfreq");
}
