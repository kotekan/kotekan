#include "hfbBuffer.hpp"

#include "gpsTime.h"

#include "fmt.hpp"

#include <set>


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
hfbFrameView::hfbFrameView(Buffer* buf, int frame_id) :
    hfbFrameView(buf, frame_id, ((hfbMetadata*)(buf->metadata[frame_id]->metadata))->num_elements,
                 ((hfbMetadata*)(buf->metadata[frame_id]->metadata))->num_prod) {}

hfbFrameView::hfbFrameView(Buffer* buf, int frame_id, uint32_t num_elements) :
    hfbFrameView(buf, frame_id, num_elements, num_elements * (num_elements + 1) / 2) {}

hfbFrameView::hfbFrameView(Buffer* buf, int frame_id, uint32_t n_elements, uint32_t n_prod) :
    buffer(buf),
    id(frame_id),
    _metadata((hfbMetadata*)buf->metadata[id]->metadata),
    _frame(buffer->frames[id]),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(calculate_buffer_layout(n_elements, n_prod)),

    // Set the const refs to the structural metadata
    num_elements(_metadata->num_elements),
    num_prod(_metadata->num_prod),

    // Set the refs to the general _metadata
    time(std::tie(_metadata->fpga_seq_start, _metadata->ctime)),
    fpga_seq_length(_metadata->fpga_seq_length),
    fpga_seq_total(_metadata->fpga_seq_total),
    freq_id(_metadata->freq_id),
    dataset_id(_metadata->dataset_id),

    // Bind the regions of the buffer to spans and refernces on the view
    hfb(bind_span<cfloat>(_frame, buffer_layout.second[hfbField::hfb])),
    weight(bind_span<float>(_frame, buffer_layout.second[hfbField::weight])),
    flags(bind_span<float>(_frame, buffer_layout.second[hfbField::flags])),
    gain(bind_span<cfloat>(_frame, buffer_layout.second[hfbField::gain]))

{
    // Initialise the structure if not already done
    // NOTE: the provided structure params have already been used to calculate
    // the layout, but here we need to make sure the metadata tracks them too.
    _metadata->num_elements = n_elements;
    _metadata->num_prod = n_prod;

    // Check that the actual buffer size is big enough to contain the calculated
    // view
    size_t required_size = buffer_layout.first;

    if (required_size > (uint32_t)buffer->frame_size) {

        std::string s =
            fmt::format(fmt("Hyperfine beam buffer [{:s}] too small. Must be a minimum of {:d} bytes "
                            "for elements={:d}, products={:d}"),
                        buffer->buffer_name, required_size, n_elements, n_prod);

        throw std::runtime_error(s);
    }
}


hfbFrameView::hfbFrameView(Buffer* buf, int frame_id, hfbFrameView frame_to_copy) :
    hfbFrameView(buf, frame_id, frame_to_copy.num_elements, frame_to_copy.num_prod) {
    // Copy over the metadata values
    *_metadata = *(frame_to_copy.metadata());

    // Copy the frame data here:
    // NOTE: this copies the full buffer memory, not only the individual components
    std::memcpy(buffer->frames[id], frame_to_copy.buffer->frames[frame_to_copy.id],
                frame_to_copy.buffer->frame_size);
}


std::string hfbFrameView::summary() const {

    struct tm* tm = std::gmtime(&(std::get<1>(time).tv_sec));

    string s =
        fmt::format("hfbBuffer[name={:s}]: freq={:d} dataset={:#x} fpga_start={:d} time={:%F %T}",
                    buffer->buffer_name, freq_id, dataset_id, std::get<0>(time), *tm);

    return s;
}


hfbFrameView hfbFrameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
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

    return hfbFrameView(buf_dest, frame_id_dest);
}


// Copy the non-const parts of the metadata
void hfbFrameView::copy_metadata(hfbFrameView frame_to_copy) {
    _metadata->fpga_seq_start = frame_to_copy.metadata()->fpga_seq_start;
    _metadata->fpga_seq_length = frame_to_copy.metadata()->fpga_seq_length;
    _metadata->fpga_seq_total = frame_to_copy.metadata()->fpga_seq_total;
    _metadata->ctime = frame_to_copy.metadata()->ctime;
    _metadata->freq_id = frame_to_copy.metadata()->freq_id;
    _metadata->dataset_id = frame_to_copy.metadata()->dataset_id;
}

// Copy the non-visibility parts of the buffer
void hfbFrameView::copy_data(hfbFrameView frame_to_copy, const std::set<hfbField>& skip_members) {

    // Define some helper methods so we don't need to code up the same checks everywhere
    auto copy_member = [&](hfbField member) { return (skip_members.count(member) == 0); };

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

    if (copy_member(hfbField::hfb)) {
        check_prod();
        std::copy(frame_to_copy.hfb.begin(), frame_to_copy.hfb.end(), hfb.begin());
    }

    if (copy_member(hfbField::weight)) {
        check_prod();
        std::copy(frame_to_copy.weight.begin(), frame_to_copy.weight.end(), weight.begin());
    }


    if (copy_member(hfbField::flags)) {
        check_elements();
        std::copy(frame_to_copy.flags.begin(), frame_to_copy.flags.end(), flags.begin());
    }

    if (copy_member(hfbField::gain)) {
        check_elements();
        std::copy(frame_to_copy.gain.begin(), frame_to_copy.gain.end(), gain.begin());
    }
}

struct_layout<hfbField> hfbFrameView::calculate_buffer_layout(uint32_t num_elements,
                                                              uint32_t num_prod) {
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<hfbField, size_t, size_t>> buffer_members = {
        std::make_tuple(hfbField::hfb, sizeof(cfloat), num_prod),
        std::make_tuple(hfbField::weight, sizeof(float), num_prod),
        std::make_tuple(hfbField::flags, sizeof(float), num_elements),
        std::make_tuple(hfbField::gain, sizeof(cfloat), num_elements)};

    return struct_alignment(buffer_members);
}

void hfbFrameView::fill_chime_metadata(const chimeMetadata* chime_metadata) {

    // Set to zero as there's no information in chimeMetadata about it.
    dataset_id = 0;

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
