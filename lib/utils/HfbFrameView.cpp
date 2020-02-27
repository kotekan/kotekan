#include "HfbFrameView.hpp"

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

HfbFrameView::HfbFrameView(Buffer* buf, int frame_id) :
    FrameView(buf, frame_id),
    _metadata((hfbMetadata*)buf->metadata[id]->metadata),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(calculate_buffer_layout(_metadata->num_beams, _metadata->num_subfreq)),

    // Set the const refs to the structural metadata
    num_beams(_metadata->num_beams),
    num_subfreq(_metadata->num_subfreq),

    // Set the refs to the general _metadata
    time(_metadata->gps_time),
    fpga_seq_num(_metadata->fpga_seq_num),
    norm_frac(_metadata->norm_frac),
    num_samples_integrated(_metadata->num_samples_integrated),
    num_samples_expected(_metadata->num_samples_expected),
    freq_id(_metadata->freq_bin_num),
    dataset_id(_metadata->dataset_id),

    // Bind the regions of the buffer to spans and references on the view
    hfb(bind_span<float>(_frame, buffer_layout.second[hfbField::hfb]))

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

std::string HfbFrameView::summary() const {

    struct tm* tm = std::gmtime(&(time.tv_sec));

    std::string s = fmt::format("hfbBuffer[name={:s}]: freq={:d} fpga_start={:d} time={:%F %T}",
                                buffer->buffer_name, freq_id, fpga_seq_num, *tm);

    return s;
}

HfbFrameView HfbFrameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                      int frame_id_dest) {
    FrameView::copy_frame(buf_src, frame_id_src, buf_dest, frame_id_dest);
    return HfbFrameView(buf_dest, frame_id_dest);
}


// Copy the non-const parts of the metadata
void HfbFrameView::copy_metadata(HfbFrameView frame_to_copy) {
    _metadata->gps_time = frame_to_copy.metadata()->gps_time;
    _metadata->fpga_seq_num = frame_to_copy.metadata()->fpga_seq_num;
    _metadata->norm_frac = frame_to_copy.metadata()->norm_frac;
    _metadata->num_samples_integrated = frame_to_copy.metadata()->num_samples_integrated;
    _metadata->num_samples_expected = frame_to_copy.metadata()->num_samples_expected;
    _metadata->freq_bin_num = frame_to_copy.metadata()->freq_bin_num;
}

// Copy the non-hfb parts of the buffer
void HfbFrameView::copy_data(HfbFrameView frame_to_copy, const std::set<hfbField>& skip_members) {

    // Define some helper methods so we don't need to code up the same checks everywhere
    auto copy_member = [&](hfbField member) { return (skip_members.count(member) == 0); };

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

    if (copy_member(hfbField::hfb)) {
        check_beams();
        check_subfreq();
        std::copy(frame_to_copy.hfb.begin(), frame_to_copy.hfb.end(), hfb.begin());
    }
}

struct_layout<hfbField> HfbFrameView::calculate_buffer_layout(uint32_t num_beams,
                                                              uint32_t num_subfreq) {
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<hfbField, size_t, size_t>> buffer_members = {
        std::make_tuple(hfbField::hfb, sizeof(float), num_beams * num_subfreq)};

    // std::vector<std::tuple<hfbField, size_t, size_t, size_t>> buffer_members = {
    //    std::make_tuple(hfbField::hfb, sizeof(float), num_beams, num_subfreq)};

    return struct_alignment(buffer_members);
}

size_t HfbFrameView::calculate_frame_size(uint32_t num_beams, uint32_t num_subfreq) {
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<hfbField, size_t, size_t>> buffer_members = {
        std::make_tuple(hfbField::hfb, sizeof(float), num_beams * num_subfreq)};

    struct_layout<hfbField> buf_layout = struct_alignment(buffer_members);

    return buf_layout.first;
}

size_t HfbFrameView::calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {

    const uint32_t num_beams = config.get<uint32_t>(unique_name, "num_frb_total_beams");
    const uint32_t num_subfreq = config.get<uint32_t>(unique_name, "num_sub_freqs");

    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<hfbField, size_t, size_t>> buffer_members = {
        std::make_tuple(hfbField::hfb, sizeof(float), num_beams * num_subfreq)};

    struct_layout<hfbField> buf_layout = struct_alignment(buffer_members);

    return buf_layout.first;
}

void HfbFrameView::set_metadata(hfbMetadata* metadata, const uint32_t num_beams,
                                const uint32_t num_subfreq) {
    metadata->num_beams = num_beams;
    metadata->num_subfreq = num_subfreq;
}
