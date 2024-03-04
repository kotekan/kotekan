#include "visBuffer.hpp"

#include "FrameView.hpp"     // for bind_span, bind_scalar, FrameView
#include "Telescope.hpp"     // for Telescope
#include "buffer.hpp"        // for Buffer, allocate_new_metadata_object
#include "chimeMetadata.hpp" // for chimeMetadata, get_stream_id_from_metadata
#include "metadata.hpp"      // for metadataContainer

#include "fmt.hpp" // for format, fmt

#include <algorithm> // for copy
#include <assert.h>
#include <complex>     // for complex
#include <cstdint>     // for uint64_t // IWYU pragma: keep
#include <ctime>       // for gmtime
#include <exception>   // for exception
#include <map>         // for map
#include <regex>       // for match_results<>::_Base_type
#include <set>         // for set
#include <stdexcept>   // for runtime_error
#include <string.h>    // for memset
#include <sys/time.h>  // for TIMEVAL_TO_TIMESPEC
#include <type_traits> // for __decay_and_strip<>::__type
#include <vector>      // for vector

REGISTER_TYPE_WITH_FACTORY(metadataObject, VisMetadata);

void VisMetadata::deepCopy(std::shared_ptr<metadataObject> other) {
    std::shared_ptr<VisMetadata> o = std::dynamic_pointer_cast<VisMetadata>(other);
    *this = *o;
}

struct VisMetadataFormat {
    uint64_t fpga_seq_start;
    timespec ctime;
    uint64_t fpga_seq_length;
    uint64_t fpga_seq_total;
    uint64_t rfi_total;
    freq_id_t freq_id;
    dset_id_t dataset_id;
    uint32_t num_elements;
    uint32_t num_prod;
    uint32_t num_ev;
};

size_t VisMetadata::get_serialized_size() {
    return sizeof(VisMetadataFormat);
}

size_t VisMetadata::set_from_bytes(const char* bytes, size_t length) {
    size_t sz = get_serialized_size();
    assert(length >= sz);
    const VisMetadataFormat* fmt = reinterpret_cast<const VisMetadataFormat*>(bytes);
    fpga_seq_start = fmt->fpga_seq_start;
    ctime = fmt->ctime;
    fpga_seq_length = fmt->fpga_seq_length;
    fpga_seq_total = fmt->fpga_seq_total;
    rfi_total = fmt->rfi_total;
    freq_id = fmt->freq_id;
    dataset_id = fmt->dataset_id;
    num_elements = fmt->num_elements;
    num_prod = fmt->num_prod;
    num_ev = fmt->num_ev;
    return sz;
}

size_t VisMetadata::serialize(char* bytes) {
    size_t sz = get_serialized_size();
    VisMetadataFormat* fmt = reinterpret_cast<VisMetadataFormat*>(bytes);
    fmt->fpga_seq_start = fpga_seq_start;
    fmt->ctime = ctime;
    fmt->fpga_seq_length = fpga_seq_length;
    fmt->fpga_seq_total = fpga_seq_total;
    fmt->rfi_total = rfi_total;
    fmt->freq_id = freq_id;
    fmt->dataset_id = dataset_id;
    fmt->num_elements = num_elements;
    fmt->num_prod = num_prod;
    fmt->num_ev = num_ev;
    return sz;
}

nlohmann::json VisMetadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const VisMetadata& m) {
    j["fpga_seq_start"] = m.fpga_seq_start;
    j["ctime"] = m.ctime;
    j["fpga_seq_length"] = m.fpga_seq_length;
    j["fpga_seq_total"] = m.fpga_seq_total;
    j["rfi_total"] = m.rfi_total;
    j["freq_id"] = m.freq_id;
    j["dataset_id"] = m.dataset_id;
    j["num_elements"] = m.num_elements;
    j["num_prod"] = m.num_prod;
    j["num_ev"] = m.num_ev;
}

void from_json(const nlohmann::json& j, VisMetadata& m) {
    m.fpga_seq_start = j["fpga_seq_start"];
    m.ctime = j["ctime"];
    m.fpga_seq_length = j["fpga_seq_length"];
    m.fpga_seq_total = j["fpga_seq_total"];
    m.rfi_total = j["rfi_total"];
    m.freq_id = j["freq_id"];
    m.dataset_id = j["dataset_id"];
    m.num_elements = j["num_elements"];
    m.num_prod = j["num_prod"];
    m.num_ev = j["num_ev"];
}

VisFrameView::VisFrameView(Buffer* buf, int frame_id) :
    FrameView(buf, frame_id), _metadata(std::static_pointer_cast<VisMetadata>(buf->metadata[id])),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(
        calculate_buffer_layout(_metadata->num_elements, _metadata->num_prod, _metadata->num_ev)),

    // Set the const refs to the structural metadata
    num_elements(_metadata->num_elements), num_prod(_metadata->num_prod), num_ev(_metadata->num_ev),

    // Set the refs to the general _metadata
    time(std::tie(_metadata->fpga_seq_start, _metadata->ctime)),
    fpga_seq_length(_metadata->fpga_seq_length), fpga_seq_total(_metadata->fpga_seq_total),
    rfi_total(_metadata->rfi_total), freq_id(_metadata->freq_id), dataset_id(_metadata->dataset_id),

    // Bind the regions of the buffer to spans and references on the view
    vis(bind_span<cfloat>(_frame, buffer_layout.second[VisField::vis])),
    weight(bind_span<float>(_frame, buffer_layout.second[VisField::weight])),
    flags(bind_span<float>(_frame, buffer_layout.second[VisField::flags])),
    eval(bind_span<float>(_frame, buffer_layout.second[VisField::eval])),
    evec(bind_span<cfloat>(_frame, buffer_layout.second[VisField::evec])),
    erms(bind_scalar<float>(_frame, buffer_layout.second[VisField::erms])),
    gain(bind_span<cfloat>(_frame, buffer_layout.second[VisField::gain]))
{
    // Check that the actual buffer size is big enough to contain the calculated
    // view
    size_t required_size = buffer_layout.first;

    if (required_size > (uint32_t)buffer->frame_size) {

        std::string s =
            fmt::format(fmt("Visibility buffer [{:s}] frames are too small with {:d} bytes. Must be a minimum of {:d} bytes "
                            "for elements={:d}, products={:d}, ev={:d}"),
                        buffer->buffer_name, (uint32_t)buffer->frame_size, required_size, num_elements, num_prod, num_ev);

        throw std::runtime_error(s);
    }
}

std::string VisFrameView::summary() const {

    struct tm* tm = std::gmtime(&(std::get<1>(time).tv_sec));

    std::string s =
        fmt::format("VisBuffer[name={:s}]: freq={:d} dataset={} fpga_start={:d} time={:%F %T}",
                    buffer->buffer_name, freq_id, dataset_id, std::get<0>(time), *tm);

    return s;
}

VisFrameView VisFrameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                      int frame_id_dest) {
    FrameView::copy_frame(buf_src, frame_id_src, buf_dest, frame_id_dest);
    return VisFrameView(buf_dest, frame_id_dest);
}


// Copy the non-const parts of the metadata
void VisFrameView::copy_metadata(VisFrameView frame_to_copy) {
    _metadata->fpga_seq_start = frame_to_copy.metadata()->fpga_seq_start;
    _metadata->fpga_seq_length = frame_to_copy.metadata()->fpga_seq_length;
    _metadata->fpga_seq_total = frame_to_copy.metadata()->fpga_seq_total;
    _metadata->rfi_total = frame_to_copy.metadata()->rfi_total;
    _metadata->ctime = frame_to_copy.metadata()->ctime;
    _metadata->freq_id = frame_to_copy.metadata()->freq_id;
    _metadata->dataset_id = frame_to_copy.metadata()->dataset_id;
}

// Copy the non-visibility parts of the buffer
void VisFrameView::copy_data(VisFrameView frame_to_copy, const std::set<VisField>& skip_members) {

    // Define some helper methods so we don't need to code up the same checks everywhere
    auto copy_member = [&](VisField member) { return (skip_members.count(member) == 0); };

    auto check_elements = [&]() {
        if (num_elements != frame_to_copy.num_elements) {
            auto msg = fmt::format(fmt("Number of inputs don't match for copy [src={}; dest={}]."),
                                   frame_to_copy.num_elements, num_elements);
            throw std::runtime_error(msg);
        }
    };

    auto check_prod = [&]() {
        if (num_elements != frame_to_copy.num_elements) {
            auto msg =
                fmt::format(fmt("Number of products don't match for copy [src={}; dest={}]."),
                            frame_to_copy.num_prod, num_prod);
            throw std::runtime_error(msg);
        }
    };

    auto check_ev = [&]() {
        if (num_ev != frame_to_copy.num_ev) {
            auto msg = fmt::format(fmt("Number of ev don't match for copy [src={}; dest={}]."),
                                   frame_to_copy.num_ev, num_ev);
            throw std::runtime_error(msg);
        }
    };

    if (copy_member(VisField::vis)) {
        check_prod();
        std::copy(frame_to_copy.vis.begin(), frame_to_copy.vis.end(), vis.begin());
    }

    if (copy_member(VisField::weight)) {
        check_prod();
        std::copy(frame_to_copy.weight.begin(), frame_to_copy.weight.end(), weight.begin());
    }


    if (copy_member(VisField::flags)) {
        check_elements();
        std::copy(frame_to_copy.flags.begin(), frame_to_copy.flags.end(), flags.begin());
    }

    if (copy_member(VisField::eval)) {
        check_ev();
        std::copy(frame_to_copy.eval.begin(), frame_to_copy.eval.end(), eval.begin());
    }

    if (copy_member(VisField::evec)) {
        check_ev();
        check_elements();
        std::copy(frame_to_copy.evec.begin(), frame_to_copy.evec.end(), evec.begin());
    }

    if (copy_member(VisField::erms))
        erms = frame_to_copy.erms;

    if (copy_member(VisField::gain)) {
        check_elements();
        std::copy(frame_to_copy.gain.begin(), frame_to_copy.gain.end(), gain.begin());
    }
}

struct_layout<VisField> VisFrameView::calculate_buffer_layout(uint32_t num_elements,
                                                              uint32_t num_prod, uint32_t num_ev) {
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<VisField, size_t, size_t>> buffer_members = {
        std::make_tuple(VisField::vis, sizeof(cfloat), num_prod),
        std::make_tuple(VisField::weight, sizeof(float), num_prod),
        std::make_tuple(VisField::flags, sizeof(float), num_elements),
        std::make_tuple(VisField::eval, sizeof(float), num_ev),
        std::make_tuple(VisField::evec, sizeof(cfloat), num_ev * num_elements),
        std::make_tuple(VisField::erms, sizeof(float), 1),
        std::make_tuple(VisField::gain, sizeof(cfloat), num_elements)};

    return struct_alignment(buffer_members);
}

size_t VisFrameView::calculate_frame_size(uint32_t num_elements, uint32_t num_prod,
                                          uint32_t num_ev) {

    return calculate_buffer_layout(num_elements, num_prod, num_ev).first;
}

size_t VisFrameView::calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {

    const int num_elements = config.get<int>(unique_name, "num_elements");
    const int num_ev = config.get<int>(unique_name, "num_ev");
    int num_prod = config.get_default<int>(unique_name, "num_prod", -1);

    if (num_prod < 0) {
        // num_prod = num_elements * (num_elements + 1) / 2;
        num_prod = 2 * num_elements * num_elements;
    }

    return calculate_buffer_layout(num_elements, num_prod, num_ev).first;
}

void VisFrameView::fill_chime_metadata(const chimeMetadata* chime_metadata, uint32_t ind) {

    auto& tel = Telescope::instance();

    // Set to zero as there's no information in chimeMetadata about it.
    dataset_id = dset_id_t::null;

    // Set the frequency index from the stream id of the metadata
    freq_id = tel.to_freq_id(get_stream_id_from_metadata(chime_metadata), ind);

    // Set the time
    uint64_t fpga_seq = chime_metadata->fpga_seq_num;

    timespec ts;

    // Use the GPS time if appropriate.
    if (tel.gps_time_enabled()) {
        ts = chime_metadata->gps_time;
    } else {
        TIMEVAL_TO_TIMESPEC(&(chime_metadata->first_packet_recv_time), &ts);
    }

    time = std::make_tuple(fpga_seq, ts);
}

void VisFrameView::set_metadata(VisMetadata* metadata, const uint32_t num_elements,
                                const uint32_t num_prod, const uint32_t num_ev) {
    metadata->num_elements = num_elements;
    metadata->num_prod = num_prod;
    metadata->num_ev = num_ev;
}

void VisFrameView::set_metadata(Buffer* buf, const uint32_t index, const uint32_t num_elements,
                                const uint32_t num_prod, const uint32_t num_ev) {
    VisMetadata* metadata = (VisMetadata*)buf->metadata[index].get();
    metadata->num_elements = num_elements;
    metadata->num_prod = num_prod;
    metadata->num_ev = num_ev;
}

VisFrameView VisFrameView::create_frame_view(Buffer* buf, const uint32_t index,
                                             const uint32_t num_elements, const uint32_t num_prod,
                                             const uint32_t num_ev,
                                             bool alloc_metadata /*= true*/) {

    if (alloc_metadata) {
        buf->allocate_new_metadata_object(index);
    }

    set_metadata(buf, index, num_elements, num_prod, num_ev);
    return VisFrameView(buf, index);
}

size_t VisFrameView::data_size() const {
    return buffer_layout.first;
}

void VisFrameView::zero_frame() {

    // Fill data with zeros
    std::memset(_frame, 0, data_size());
    erms = 0;

    // Set non-structural metadata
    freq_id = 0;
    dataset_id = dset_id_t::null;
    time = std::make_tuple(0, timespec{0, 0});

    // mark frame as empty by ensuring this is 0
    fpga_seq_length = 0;
    fpga_seq_total = 0;
}
