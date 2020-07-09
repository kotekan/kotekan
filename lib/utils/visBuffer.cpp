#include "visBuffer.hpp"

#include "FrameView.hpp" // for metadataContainer
#include "Telescope.hpp"
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

template<typename T>
gsl::span<T> bind_span(uint8_t* start, std::pair<size_t, size_t> range) {
    T* span_start = (T*)(start + range.first);
    T* span_end = (T*)(start + range.second);

    return gsl::span<T>(span_start, span_end);
}

template<typename T>
T& bind_scalar(uint8_t* start, std::pair<size_t, size_t> range) {
    T* loc = (T*)(start + range.first);

    return *loc;
}

// NOTE: this construct somewhat pointlessly reinitialises the structural
// elements of the metadata, but I think there's no other way to share the
// initialisation list
VisFrameView::VisFrameView(Buffer* buf, int frame_id) :
    VisFrameView(buf, frame_id, ((VisMetadata*)(buf->metadata[frame_id]->metadata))->num_elements,
                 ((VisMetadata*)(buf->metadata[frame_id]->metadata))->num_prod,
                 ((VisMetadata*)(buf->metadata[frame_id]->metadata))->num_ev) {}

VisFrameView::VisFrameView(Buffer* buf, int frame_id, uint32_t num_elements, uint32_t num_ev) :
    VisFrameView(buf, frame_id, num_elements, num_elements * (num_elements + 1) / 2, num_ev) {}

VisFrameView::VisFrameView(Buffer* buf, int frame_id, uint32_t n_elements, uint32_t n_prod,
                           uint32_t n_eigenvectors) :
    buffer(buf),
    id(frame_id),
    _metadata((VisMetadata*)buf->metadata[id]->metadata),
    _frame(buffer->frames[id]),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(calculate_buffer_layout(n_elements, n_prod, n_eigenvectors)),

    // Set the const refs to the structural metadata
    num_elements(_metadata->num_elements),
    num_prod(_metadata->num_prod),
    num_ev(_metadata->num_ev),
    data_size(buffer_layout.first),

    // Set the refs to the general _metadata
    time(std::tie(_metadata->fpga_seq_start, _metadata->ctime)),
    fpga_seq_length(_metadata->fpga_seq_length),
    fpga_seq_total(_metadata->fpga_seq_total),
    rfi_total(_metadata->rfi_total),
    freq_id(_metadata->freq_id),
    dataset_id(_metadata->dataset_id),

    // Bind the regions of the buffer to spans and references on the view
    vis(bind_span<cfloat>(_frame, buffer_layout.second[VisField::vis])),
    weight(bind_span<float>(_frame, buffer_layout.second[VisField::weight])),
    flags(bind_span<float>(_frame, buffer_layout.second[VisField::flags])),
    eval(bind_span<float>(_frame, buffer_layout.second[VisField::eval])),
    evec(bind_span<cfloat>(_frame, buffer_layout.second[VisField::evec])),
    erms(bind_scalar<float>(_frame, buffer_layout.second[VisField::erms])),
    gain(bind_span<cfloat>(_frame, buffer_layout.second[VisField::gain]))

{
    // Initialise the structure if not already done
    // NOTE: the provided structure params have already been used to calculate
    // the layout, but here we need to make sure the metadata tracks them too.
    _metadata->num_elements = n_elements;
    _metadata->num_prod = n_prod;
    _metadata->num_ev = n_eigenvectors;

    // Check that the actual buffer size is big enough to contain the calculated
    // view
    size_t required_size = buffer_layout.first;

    data_size = required_size;

    if (required_size > (uint32_t)buffer->frame_size) {

        std::string s =
            fmt::format(fmt("Visibility buffer [{:s}] too small. Must be a minimum of {:d} bytes "
                            "for elements={:d}, products={:d}, ev={:d}"),
                        buffer->buffer_name, required_size, n_elements, n_prod, n_eigenvectors);

        throw std::runtime_error(s);
    }
}

VisFrameView::VisFrameView(Buffer* buf, int frame_id, VisFrameView frame_to_copy) :
    VisFrameView(buf, frame_id, frame_to_copy.num_elements, frame_to_copy.num_prod,
                 frame_to_copy.num_ev) {
    // Copy over the metadata values
    *_metadata = *(frame_to_copy.metadata());

    // Copy the frame data here:
    // NOTE: this copies the full buffer memory, not only the individual components
    std::memcpy(buffer->frames[id], frame_to_copy.buffer->frames[frame_to_copy.id],
                frame_to_copy.buffer->frame_size);
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
    allocate_new_metadata_object(buf_dest, frame_id_dest);

    // Buffer sizes must match exactly
    if (buf_src->frame_size != buf_dest->frame_size) {
        std::string msg =
            fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->frame_size, buf_dest->frame_size);
        throw std::runtime_error(msg);
    }

    // Metadata sizes must match exactly
    if (buf_src->metadata[frame_id_src]->metadata_size
        != buf_dest->metadata[frame_id_dest]->metadata_size) {
        std::string msg =
            fmt::format(fmt("Metadata sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->metadata[frame_id_src]->metadata_size,
                        buf_dest->metadata[frame_id_dest]->metadata_size);
        throw std::runtime_error(msg);
    }

    // Calculate the number of consumers on the source buffer
    int num_consumers = 0;
    for (int i = 0; i < MAX_CONSUMERS; ++i) {
        if (buf_src->consumers[i].in_use == 1) {
            num_consumers++;
        }
    }

    // Copy or transfer the data part.
    if (num_consumers == 1) {
        // Transfer frame contents with directly...
        swap_frames(buf_src, frame_id_src, buf_dest, frame_id_dest);
    } else if (num_consumers > 1) {
        // Copy the frame data over, leaving the source intact
        std::memcpy(buf_dest->frames[frame_id_dest], buf_src->frames[frame_id_src],
                    buf_src->frame_size);
    }

    // Copy over the metadata
    std::memcpy(buf_dest->metadata[frame_id_dest]->metadata,
                buf_src->metadata[frame_id_src]->metadata,
                buf_src->metadata[frame_id_src]->metadata_size);

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
        num_prod = num_elements * (num_elements + 1) / 2;
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
    VisMetadata* metadata = (VisMetadata*)buf->metadata[index]->metadata;
    metadata->num_elements = num_elements;
    metadata->num_prod = num_prod;
    metadata->num_ev = num_ev;
}

VisFrameView VisFrameView::create_frame_view(Buffer* buf, const uint32_t index,
                                             const uint32_t num_elements, const uint32_t num_prod,
                                             const uint32_t num_ev,
                                             bool alloc_metadata /*= true*/) {

    if (alloc_metadata) {
        allocate_new_metadata_object(buf, index);
    }

    set_metadata(buf, index, num_elements, num_prod, num_ev);
    return VisFrameView(buf, index);
}

size_t VisFrameView::get_data_size() {
    return data_size;
}
