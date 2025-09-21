#include "N2Metadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <string.h> // for size_t, memset

REGISTER_TYPE_WITH_FACTORY(metadataObject, N2Metadata);
N2Metadata::N2Metadata() : N2MetadataFormat{0, 0, 0, 0, 0, 0.0, eop_null, 0, 0, 0, 0, 0} {
    ;
}

void N2Metadata::deepCopy(std::shared_ptr<metadataObject> other) {
    std::shared_ptr<N2Metadata> o = std::dynamic_pointer_cast<N2Metadata>(other);
    *this = *o;
}

size_t N2Metadata::get_serialized_size() {
    return sizeof(N2MetadataFormat);
}

size_t N2Metadata::set_from_bytes(const char* bytes, size_t length) {
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
