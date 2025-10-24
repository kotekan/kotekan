#include "N2Metadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <string.h> // for size_t, memset

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
    j["fpga_start_tick"] = m.fpga_start_tick;
    j["frame_start_time_ns"] = m.frame_start_time_ns;
    j["frame_length_fpga_ticks"] = m.frame_length_fpga_ticks;

    j["n_valid_fpga_ticks"] = m.n_valid_fpga_ticks;
    j["n_rfi_fpga_ticks"] = m.n_rfi_fpga_ticks;

    j["freq_id"] = m.freq_id; // this is an int in chordMetadata, maybe change later
    j["freq_Hz"] = m.freq_Hz;
    j["eop"] = m.eop;

    j["num_elements"] = m.num_elements;
    j["num_prod"] = m.num_prod;
    j["num_ev"] = m.num_ev;
    j["nfreq"] = m.nfreq;
}

void from_json(const nlohmann::json& j, N2Metadata& m) {
    m.fpga_start_tick = j["fpga_start_tick"];
    m.frame_start_time_ns = j["frame_start_time_ns"];
    m.frame_length_fpga_ticks = j["frame_length_fpga_ticks"];

    m.n_valid_fpga_ticks = j["n_valid_fpga_ticks"];
    m.n_rfi_fpga_ticks = j["n_rfi_fpga_ticks"];

    m.freq_id = j["freq_id"]; // this is an int in chordMetadata, maybe change later
    m.freq_Hz = j["freq_Hz"];
    m.eop = j["eop"];

    m.num_elements = j["num_elements"];
    m.num_prod = j["num_prod"];
    m.num_ev = j["num_ev"];
    m.nfreq = j["nfreq"];
}
