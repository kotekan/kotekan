#include "visBuffer.hpp"

#include "FrameView.hpp"           // for metadataContainer
#include "buffer.h"                // for Buffer, allocate_new_metadata_object, swap_frames
#include "chimeMetadata.h"         // for chimeMetadata
#include "fpga_header_functions.h" // for bin_number_chime, extract_stream_id, stream_id_t
#include "gpsTime.h"               // for is_gps_global_time_set
#include "metadata.h"              // for metadataContainer

#include "fmt.hpp" // for format, fmt

#include <algorithm>   // for copy
#include <complex>     // for complex  // IWYU pragma: keep
#include <cstdint>     // for uint64_t // IWYU pragma: keep
#include <cstring>     // for memcpy
#include <ctime>       // for gmtime
#include <map>         // for map
#include <set>         // for set
#include <stdexcept>   // for runtime_error
#include <sys/time.h>  // for TIMEVAL_TO_TIMESPEC
#include <type_traits> // for __decay_and_strip<>::__type
#include <vector>      // for vector

VisFrameView::VisFrameView(Buffer* buf, int frame_id) :
    FrameView(buf, frame_id),
    _metadata((visMetadata*)buf->metadata[id]->metadata),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(
        calculate_buffer_layout(_metadata->num_elements, _metadata->num_prod, _metadata->num_ev)),

    // Set the const refs to the structural metadata
    num_elements(_metadata->num_elements),
    num_prod(_metadata->num_prod),
    num_ev(_metadata->num_ev),

    // Set the refs to the general _metadata
    time(std::tie(_metadata->fpga_seq_start, _metadata->ctime)),
    fpga_seq_length(_metadata->fpga_seq_length),
    fpga_seq_total(_metadata->fpga_seq_total),
    rfi_total(_metadata->rfi_total),
    freq_id(_metadata->freq_id),
    dataset_id(_metadata->dataset_id),

    // Bind the regions of the buffer to spans and references on the view
    vis(bind_span<cfloat>(_frame, buffer_layout.second[visField::vis])),
    weight(bind_span<float>(_frame, buffer_layout.second[visField::weight])),
    flags(bind_span<float>(_frame, buffer_layout.second[visField::flags])),
    eval(bind_span<float>(_frame, buffer_layout.second[visField::eval])),
    evec(bind_span<cfloat>(_frame, buffer_layout.second[visField::evec])),
    erms(bind_scalar<float>(_frame, buffer_layout.second[visField::erms])),
    gain(bind_span<cfloat>(_frame, buffer_layout.second[visField::gain]))

{
    // Check that the actual buffer size is big enough to contain the calculated
    // view
    size_t required_size = buffer_layout.first;

    if (required_size > (uint32_t)buffer->frame_size) {

        std::string s =
            fmt::format(fmt("Visibility buffer [{:s}] too small. Must be a minimum of {:d} bytes "
                            "for elements={:d}, products={:d}, ev={:d}"),
                        buffer->buffer_name, required_size, num_elements, num_prod, num_ev);

        throw std::runtime_error(s);
    }
}

std::string VisFrameView::summary() const {

    struct tm* tm = std::gmtime(&(std::get<1>(time).tv_sec));

    std::string s =
        fmt::format("visBuffer[name={:s}]: freq={:d} dataset={} fpga_start={:d} time={:%F %T}",
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
void VisFrameView::copy_data(VisFrameView frame_to_copy, const std::set<visField>& skip_members) {

    // Define some helper methods so we don't need to code up the same checks everywhere
    auto copy_member = [&](visField member) { return (skip_members.count(member) == 0); };

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

    if (copy_member(visField::vis)) {
        check_prod();
        std::copy(frame_to_copy.vis.begin(), frame_to_copy.vis.end(), vis.begin());
    }

    if (copy_member(visField::weight)) {
        check_prod();
        std::copy(frame_to_copy.weight.begin(), frame_to_copy.weight.end(), weight.begin());
    }


    if (copy_member(visField::flags)) {
        check_elements();
        std::copy(frame_to_copy.flags.begin(), frame_to_copy.flags.end(), flags.begin());
    }

    if (copy_member(visField::eval)) {
        check_ev();
        std::copy(frame_to_copy.eval.begin(), frame_to_copy.eval.end(), eval.begin());
    }

    if (copy_member(visField::evec)) {
        check_ev();
        check_elements();
        std::copy(frame_to_copy.evec.begin(), frame_to_copy.evec.end(), evec.begin());
    }

    if (copy_member(visField::erms))
        erms = frame_to_copy.erms;

    if (copy_member(visField::gain)) {
        check_elements();
        std::copy(frame_to_copy.gain.begin(), frame_to_copy.gain.end(), gain.begin());
    }
}

struct_layout<visField> VisFrameView::calculate_buffer_layout(uint32_t num_elements,
                                                              uint32_t num_prod, uint32_t num_ev) {
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<visField, size_t, size_t>> buffer_members = {
        std::make_tuple(visField::vis, sizeof(cfloat), num_prod),
        std::make_tuple(visField::weight, sizeof(float), num_prod),
        std::make_tuple(visField::flags, sizeof(float), num_elements),
        std::make_tuple(visField::eval, sizeof(float), num_ev),
        std::make_tuple(visField::evec, sizeof(cfloat), num_ev * num_elements),
        std::make_tuple(visField::erms, sizeof(float), 1),
        std::make_tuple(visField::gain, sizeof(cfloat), num_elements)};

    return struct_alignment(buffer_members);
}

size_t VisFrameView::calculate_frame_size(uint32_t num_elements, uint32_t num_prod,
                                          uint32_t num_ev) {
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<visField, size_t, size_t>> buffer_members = {
        std::make_tuple(visField::vis, sizeof(cfloat), num_prod),
        std::make_tuple(visField::weight, sizeof(float), num_prod),
        std::make_tuple(visField::flags, sizeof(float), num_elements),
        std::make_tuple(visField::eval, sizeof(float), num_ev),
        std::make_tuple(visField::evec, sizeof(cfloat), num_ev * num_elements),
        std::make_tuple(visField::erms, sizeof(float), 1),
        std::make_tuple(visField::gain, sizeof(cfloat), num_elements)};

    struct_layout<visField> buf_layout = struct_alignment(buffer_members);

    return buf_layout.first;
}

size_t VisFrameView::calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {

    const int num_elements = config.get<int>(unique_name, "num_elements");
    const int num_ev = config.get<int>(unique_name, "num_ev");
    int num_prod = config.get_default<int>(unique_name, "num_prod", -1);

    if (num_prod < 0) {
        num_prod = num_elements * (num_elements + 1) / 2;
    }

    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<visField, size_t, size_t>> buffer_members = {
        std::make_tuple(visField::vis, sizeof(cfloat), num_prod),
        std::make_tuple(visField::weight, sizeof(float), num_prod),
        std::make_tuple(visField::flags, sizeof(float), num_elements),
        std::make_tuple(visField::eval, sizeof(float), num_ev),
        std::make_tuple(visField::evec, sizeof(cfloat), num_ev * num_elements),
        std::make_tuple(visField::erms, sizeof(float), 1),
        std::make_tuple(visField::gain, sizeof(cfloat), num_elements)};

    struct_layout<visField> buf_layout = struct_alignment(buffer_members);

    return buf_layout.first;
}

void VisFrameView::fill_chime_metadata(const chimeMetadata* chime_metadata) {

    // Set to zero as there's no information in chimeMetadata about it.
    dataset_id = dset_id_t::null;

    // Set the frequency index from the stream id of the metadata
    stream_id_t stream_id = extract_stream_id(chime_metadata->stream_ID);
    freq_id = bin_number_chime(&stream_id);

    // Set the time
    // TODO: get the GPS time instead
    uint64_t fpga_seq = chime_metadata->fpga_seq_num;

    timespec ts;

    // Use the GPS time if appropriate.
    if (is_gps_global_time_set()) {
        ts = chime_metadata->gps_time;
    } else {
        TIMEVAL_TO_TIMESPEC(&(chime_metadata->first_packet_recv_time), &ts);
    }

    time = std::make_tuple(fpga_seq, ts);
}

void VisFrameView::set_metadata(visMetadata* metadata, const uint32_t num_elements,
                                const uint32_t num_prod, const uint32_t num_ev) {
    metadata->num_elements = num_elements;
    metadata->num_prod = num_prod;
    metadata->num_ev = num_ev;
}
