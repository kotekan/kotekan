#include "HFBFrameView.hpp"

#include "FrameView.hpp"   // for metadataContainer
#include "buffer.h"        // for Buffer, allocate_new_metadata_object, swap_frames
#include "chimeMetadata.h" // for chimeMetadata
#include "metadata.h"      // for metadataContainer

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

HFBFrameView::HFBFrameView(Buffer* buf, int frame_id) :
    FrameView(buf, frame_id),
    _metadata((HFBMetadata*)buf->metadata[id]->metadata),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(calculate_buffer_layout(_metadata->num_beams, _metadata->num_subfreq)),

    // Set the const refs to the structural metadata
    num_beams(_metadata->num_beams),
    num_subfreq(_metadata->num_subfreq),

    // Set the refs to the general _metadata
    time(_metadata->ctime),
    fpga_seq_start(_metadata->fpga_seq_start),
    fpga_seq_total(_metadata->fpga_seq_total),
    fpga_seq_length(_metadata->fpga_seq_length),
    freq_id(_metadata->freq_id),
    dataset_id(_metadata->dataset_id),

    // Bind the regions of the buffer to spans and references on the view
    hfb(bind_span<float>(_frame, buffer_layout.second[HFBField::hfb])),
    weight(bind_span<float>(_frame, buffer_layout.second[HFBField::weight]))

{
    // Check that the actual buffer size is big enough to contain the calculated
    // view
    size_t required_size = buffer_layout.first;

    if (required_size > (uint32_t)buffer->frame_size) {

        std::string s = fmt::format(
            fmt("Hyper fine beam buffer [{:s}] too small. Must be a minimum of {:d} bytes "
                "for beams={:d}, sub-frequencies={:d}"),
            buffer->buffer_name, required_size, num_beams, num_subfreq);

        throw std::runtime_error(s);
    }
}

std::string HFBFrameView::summary() const {

    struct tm* tm = std::gmtime(&(time.tv_sec));

    std::string s = fmt::format("hfbBuffer[name={:s}]: freq={:d} fpga_start={:d} time={:%F %T}",
                                buffer->buffer_name, freq_id, fpga_seq_start, *tm);

    return s;
}

HFBFrameView HFBFrameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                      int frame_id_dest) {
    FrameView::copy_frame(buf_src, frame_id_src, buf_dest, frame_id_dest);
    return HFBFrameView(buf_dest, frame_id_dest);
}


// Copy the non-const parts of the metadata
void HFBFrameView::copy_metadata(HFBFrameView frame_to_copy) {
    _metadata->ctime = frame_to_copy.metadata()->ctime;
    _metadata->fpga_seq_start = frame_to_copy.metadata()->fpga_seq_start;
    _metadata->fpga_seq_total = frame_to_copy.metadata()->fpga_seq_total;
    _metadata->fpga_seq_length = frame_to_copy.metadata()->fpga_seq_length;
    _metadata->freq_id = frame_to_copy.metadata()->freq_id;
}

// Copy the non-hfb parts of the buffer
void HFBFrameView::copy_data(HFBFrameView frame_to_copy, const std::set<HFBField>& skip_members) {

    // Define some helper methods so we don't need to code up the same checks everywhere
    auto copy_member = [&](HFBField member) { return (skip_members.count(member) == 0); };

    auto check_beams = [&]() {
        if (num_beams != frame_to_copy.num_beams) {
            auto msg = fmt::format(fmt("Number of beams doesn't match for copy [src={}; dest={}]."),
                                   frame_to_copy.num_beams, num_beams);
            throw std::runtime_error(msg);
        }
    };

    auto check_subfreq = [&]() {
        if (num_subfreq != frame_to_copy.num_subfreq) {
            auto msg = fmt::format(
                fmt("Number of sub-frequencies doesn't match for copy [src={}; dest={}]."),
                frame_to_copy.num_subfreq, num_subfreq);
            throw std::runtime_error(msg);
        }
    };

    if (copy_member(HFBField::hfb)) {
        check_beams();
        check_subfreq();
        std::copy(frame_to_copy.hfb.begin(), frame_to_copy.hfb.end(), hfb.begin());
    }

    if (copy_member(HFBField::weight)) {
        check_beams();
        check_subfreq();
        std::copy(frame_to_copy.weight.begin(), frame_to_copy.weight.end(), weight.begin());
    }
}

struct_layout<HFBField> HFBFrameView::calculate_buffer_layout(uint32_t num_beams,
                                                              uint32_t num_subfreq) {
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<HFBField, size_t, size_t>> buffer_members = {
        std::make_tuple(HFBField::hfb, sizeof(float), num_beams * num_subfreq),
        std::make_tuple(HFBField::weight, sizeof(float), num_beams * num_subfreq)};

    return struct_alignment(buffer_members);
}

size_t HFBFrameView::calculate_frame_size(uint32_t num_beams, uint32_t num_subfreq) {

    return calculate_buffer_layout(num_beams, num_subfreq).first;
}

size_t HFBFrameView::calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {

    const uint32_t num_beams = config.get<uint32_t>(unique_name, "num_frb_total_beams");
    const uint32_t num_subfreq = config.get<uint32_t>(unique_name, "factor_upchan");

    return calculate_buffer_layout(num_beams, num_subfreq).first;
}

void HFBFrameView::set_metadata(HFBMetadata* metadata, const uint32_t num_beams,
                                const uint32_t num_subfreq) {
    metadata->num_beams = num_beams;
    metadata->num_subfreq = num_subfreq;
}

void HFBFrameView::set_metadata(Buffer* buf, const uint32_t index, const uint32_t num_beams,
                                const uint32_t num_subfreq) {
    HFBMetadata* metadata = (HFBMetadata*)buf->metadata[index]->metadata;
    metadata->num_beams = num_beams;
    metadata->num_subfreq = num_subfreq;
}

HFBFrameView HFBFrameView::create_frame_view(Buffer* buf, const uint32_t index,
                                             const uint32_t num_beams, const uint32_t num_subfreq,
                                             bool alloc_metadata /*= true*/) {

    if (alloc_metadata) {
        allocate_new_metadata_object(buf, index);
    }

    set_metadata(buf, index, num_beams, num_subfreq);
    return HFBFrameView(buf, index);
}

size_t HFBFrameView::data_size() {
    return buffer_layout.first;
}
