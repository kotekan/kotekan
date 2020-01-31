#include "frameView.hpp"

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
//frameView::frameView(Buffer* buf, int frame_id) :
//    frameView(buf, frame_id, ((visMetadata*)(buf->metadata[frame_id]->metadata))->num_elements,
//                 ((visMetadata*)(buf->metadata[frame_id]->metadata))->num_prod,
//                 ((visMetadata*)(buf->metadata[frame_id]->metadata))->num_ev) {}

frameView::frameView(Buffer* buf, int frame_id) :
    buffer(buf), id(frame_id), _frame(buffer->frames[id]) {}


frameView::frameView(Buffer* buf, int frame_id, frameView frame_to_copy) :
    frameView(buf, frame_id, frame_to_copy.num_elements, frame_to_copy.num_prod,
                 frame_to_copy.num_ev) {
    // Copy over the metadata values
    *_metadata = *(frame_to_copy.metadata());

    // Copy the frame data here:
    // NOTE: this copies the full buffer memory, not only the individual components
    std::memcpy(buffer->frames[id], frame_to_copy.buffer->frames[frame_to_copy.id],
                frame_to_copy.buffer->frame_size);
}


std::string frameView::summary() const {

    struct tm* tm = std::gmtime(&(std::get<1>(time).tv_sec));

    string s =
        fmt::format("visBuffer[name={:s}]: freq={:d} dataset={} fpga_start={:d} time={:%F %T}",
                    buffer->buffer_name, freq_id, dataset_id, std::get<0>(time), *tm);

    return s;
}


frameView frameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
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

    return frameView(buf_dest, frame_id_dest);
}


// Copy the non-const parts of the metadata
void frameView::copy_metadata(frameView frame_to_copy) {
    _metadata->fpga_seq_start = frame_to_copy.metadata()->fpga_seq_start;
    _metadata->fpga_seq_length = frame_to_copy.metadata()->fpga_seq_length;
    _metadata->fpga_seq_total = frame_to_copy.metadata()->fpga_seq_total;
    _metadata->rfi_total = frame_to_copy.metadata()->rfi_total;
    _metadata->ctime = frame_to_copy.metadata()->ctime;
    _metadata->freq_id = frame_to_copy.metadata()->freq_id;
    _metadata->dataset_id = frame_to_copy.metadata()->dataset_id;
}

// Copy the non-visibility parts of the buffer
void frameView::copy_data(frameView frame_to_copy, const std::set<visField>& skip_members) {

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

void frameView::fill_chime_metadata(const chimeMetadata* chime_metadata) {

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
