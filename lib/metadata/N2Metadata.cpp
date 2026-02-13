#include "N2Metadata.hpp"

#include "N2FrameDesc.hpp"
#include "factory.hpp"  // for REGISTER_TYPE_WITH_FACTORY
#include "timeUtil.hpp" // for EOP()

#include <string.h> // for size_t, memset

REGISTER_TYPE_WITH_FACTORY(metadataObject, N2Metadata);
N2Metadata::N2Metadata() = default;

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

    abs_time_idx = fmt->abs_time_idx;
    freq_id = fmt->freq_id; // this is an int in chordMetadata, maybe change later
    freq_MHz = fmt->freq_MHz;

    time_center_eop = fmt->time_center_eop;
    bin_eop = fmt->bin_eop;
    bin_start_ERA_deg = fmt->bin_start_ERA_deg;
    bin_end_ERA_deg = fmt->bin_end_ERA_deg;
    bin_start_LAST = fmt->bin_start_LAST;
    bin_end_LAST = fmt->bin_end_LAST;

    return sizeof(N2MetadataFormat);
}

size_t N2Metadata::serialize(char* bytes) {
    N2MetadataFormat* fmt = reinterpret_cast<N2MetadataFormat*>(bytes);

    fmt->fpga_start_tick = fpga_start_tick;
    fmt->frame_start_time_ns = frame_start_time_ns;
    fmt->frame_length_fpga_ticks = frame_length_fpga_ticks;

    fmt->n_valid_fpga_ticks = n_valid_fpga_ticks;
    fmt->n_rfi_fpga_ticks = n_rfi_fpga_ticks;

    fmt->abs_time_idx = abs_time_idx;
    fmt->freq_id = freq_id; // this is an int in chordMetadata, maybe change later
    fmt->freq_MHz = freq_MHz;
    fmt->time_center_eop = time_center_eop;
    fmt->bin_eop = bin_eop;
    fmt->bin_start_ERA_deg = bin_start_ERA_deg;
    fmt->bin_end_ERA_deg = bin_end_ERA_deg;
    fmt->bin_start_LAST = bin_start_LAST;
    fmt->bin_end_LAST = bin_end_LAST;

    return sizeof(N2MetadataFormat);
}

nlohmann::json N2Metadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void N2Metadata::check_frame_desc(
    const std::shared_ptr<const kotekan::FrameDesc>& frame_desc) const {
    auto n2_desc = frame_desc->as<kotekan::N2FrameDesc>();
    if (!n2_desc) {
        FATAL_ERROR_NON_OO("Frame description is not an N2FrameDesc!");
        return;
    }

    // Validate internal consistency
    if (n_valid_fpga_ticks > frame_length_fpga_ticks) {
        ERROR_NON_OO("n_valid_fpga_ticks ({:d}) exceeds frame_length_fpga_ticks ({:d})",
                     n_valid_fpga_ticks, frame_length_fpga_ticks);
    }
    if (n_rfi_fpga_ticks > frame_length_fpga_ticks) {
        ERROR_NON_OO("n_rfi_fpga_ticks ({:d}) exceeds frame_length_fpga_ticks ({:d})",
                     n_rfi_fpga_ticks, frame_length_fpga_ticks);
    }

    // TODO: Perhaps validate elements/products/layout combination
}

void to_json(nlohmann::json& j, const N2Metadata& m) {
    assert(j.empty());

    j.emplace("fpga_start_tick", m.fpga_start_tick);
    j.emplace("frame_start_time_ns", m.frame_start_time_ns);
    j.emplace("frame_length_fpga_ticks", m.frame_length_fpga_ticks);

    j.emplace("n_valid_fpga_ticks", m.n_valid_fpga_ticks);
    j.emplace("n_rfi_fpga_ticks", m.n_rfi_fpga_ticks);

    j.emplace("abs_time_idx", m.abs_time_idx);
    j.emplace("freq_id", m.freq_id); // this is an int in chordMetadata, maybe change later
    j.emplace("freq_MHz", m.freq_MHz);

    j.emplace("time_center_eop", m.time_center_eop);
    j.emplace("bin_eop", m.bin_eop);
    j.emplace("bin_start_ERA_deg", m.bin_start_ERA_deg);
    j.emplace("bin_end_ERA_deg", m.bin_end_ERA_deg);
    j.emplace("bin_start_LAST", m.bin_start_LAST);
    j.emplace("bin_end_LAST", m.bin_end_LAST);
}

void from_json(const nlohmann::json& j, N2Metadata& m) {
    m.fpga_start_tick = j.at("fpga_start_tick");
    m.frame_start_time_ns = j.at("frame_start_time_ns");
    m.frame_length_fpga_ticks = j.at("frame_length_fpga_ticks");

    m.n_valid_fpga_ticks = j.at("n_valid_fpga_ticks");
    m.n_rfi_fpga_ticks = j.at("n_rfi_fpga_ticks");

    m.abs_time_idx = j.at("abs_time_idx");
    m.freq_id = j.at("freq_id"); // this is an int in chordMetadata, maybe change later
    m.freq_MHz = j.at("freq_MHz");

    m.time_center_eop = j.at("time_center_eop");
    m.bin_eop = j.at("bin_eop");
    m.bin_start_ERA_deg = j.at("bin_start_ERA_deg");
    m.bin_end_ERA_deg = j.at("bin_end_ERA_deg");
    m.bin_start_LAST = j.at("bin_start_LAST");
    m.bin_end_LAST = j.at("bin_end_LAST");
}
