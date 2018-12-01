#include "visBuffer.hpp"
#include "gpsTime.h"
#include "fmt.hpp"


template<typename T>
gsl::span<T> bind_span(uint8_t * start, std::pair<size_t, size_t> range) {
    T* span_start = (T*)(start + range.first);
    T* span_end = (T*)(start + range.second);

    return gsl::span<T>(span_start, span_end);
}

template<typename T>
T& bind_scalar(uint8_t * start, std::pair<size_t, size_t> range) {
    T* loc = (T*)(start + range.first);

    return *loc;
}

// NOTE: this construct somewhat pointlessly reinitialises the structural
// elements of the metadata, but I think there's no other way to share the
// initialisation list
visFrameView::visFrameView(Buffer * buf, int frame_id) :
    visFrameView(buf, frame_id,
                 ((visMetadata *)(buf->metadata[frame_id]->metadata))->num_elements,
                 ((visMetadata *)(buf->metadata[frame_id]->metadata))->num_prod,
                 ((visMetadata *)(buf->metadata[frame_id]->metadata))->num_ev)
{
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                           uint32_t num_ev) :
    visFrameView(buf, frame_id, num_elements,
                 num_elements * (num_elements + 1) / 2, num_ev)
{
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t n_elements,
                           uint32_t n_prod, uint32_t n_eigenvectors) :
    buffer(buf),
    id(frame_id),
    _metadata((visMetadata *)buf->metadata[id]->metadata),
    _frame(buffer->frames[id]),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(calculate_buffer_layout(n_elements, n_prod, n_eigenvectors)),

    // Set the const refs to the structural metadata
    num_elements(_metadata->num_elements),
    num_prod(_metadata->num_prod),
    num_ev(_metadata->num_ev),

    // Set the refs to the general _metadata
    time(std::tie(_metadata->fpga_seq_start, _metadata->ctime)),
    fpga_seq_length(_metadata->fpga_seq_length),
    fpga_seq_total(_metadata->fpga_seq_total),
    freq_id(_metadata->freq_id),
    dataset_id(_metadata->dataset_id),

    // Bind the regions of the buffer to spans and refernces on the view
    vis(bind_span<cfloat>(_frame, buffer_layout.second[visField::vis])),
    weight(bind_span<float>(_frame, buffer_layout.second[visField::weight])),
    flags(bind_span<float>(_frame, buffer_layout.second[visField::flags])),
    eval(bind_span<float>(_frame, buffer_layout.second[visField::eval])),
    evec(bind_span<cfloat>(_frame, buffer_layout.second[visField::evec])),
    erms(bind_scalar<float>(_frame, buffer_layout.second[visField::erms])),
    gain(bind_span<cfloat>(_frame, buffer_layout.second[visField::gain]))

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

    if(required_size > (uint32_t)buffer->frame_size) {

        std::string s = fmt::format(
            "Visibility buffer [{}] too small. Must be a minimum of\
             {} bytes for elements={}, products={}, ev={}",
            buffer->buffer_name, required_size, n_elements, n_prod,
            n_eigenvectors
        );

        throw std::runtime_error(s);
    }
}


visFrameView::visFrameView(Buffer * buf, int frame_id,
                           visFrameView frame_to_copy) :
    visFrameView(buf, frame_id, frame_to_copy.num_elements,
                 frame_to_copy.num_prod, frame_to_copy.num_ev)
{
    // Copy over the metadata values
    *_metadata = *(frame_to_copy.metadata());

    // Copy the frame data here:
    // NOTE: this copies the full buffer memory, not only the individual components
    std::memcpy(buffer->frames[id],
                frame_to_copy.buffer->frames[frame_to_copy.id],
                frame_to_copy.buffer->frame_size);
}


std::string visFrameView::summary() const {

    auto tm = gmtime(&(std::get<1>(time).tv_sec));

    string s = fmt::format(
        "visBuffer[name={}]: freq={} dataset={} fpga_start={} time={:%F %T}",
        buffer->buffer_name, freq_id, dataset_id, std::get<0>(time), *tm
    );

    return s;
}


visFrameView visFrameView::copy_frame(Buffer* buf_src, int frame_id_src,
                                      Buffer* buf_dest, int frame_id_dest)
{
    allocate_new_metadata_object(buf_dest, frame_id_dest);

    // Buffer sizes must match exactly
    if (buf_src->frame_size != buf_dest->frame_size) {
        std::string msg = fmt::format(
            "Buffer sizes must match for direct copy (src %i != dest %i).",
            buf_src->frame_size, buf_dest->frame_size);
        throw std::runtime_error(msg);
    }

    // Metadata sizes must match exactly
    if (buf_src->metadata[frame_id_src]->metadata_size !=
        buf_dest->metadata[frame_id_dest]->metadata_size) {
        std::string msg = fmt::format(
            "Metadata sizes must match for direct copy (src %i != dest %i).",
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
        std::memcpy(buf_dest->frames[frame_id_dest],
                    buf_src->frames[frame_id_src], buf_src->frame_size);
    }

    // Copy over the metadata
    std::memcpy(buf_dest->metadata[frame_id_dest]->metadata,
                buf_src->metadata[frame_id_src]->metadata,
                buf_src->metadata[frame_id_src]->metadata_size);

    return visFrameView(buf_dest, frame_id_dest);
}


// Copy the non-const parts of the metadata
void visFrameView::copy_nonconst_metadata(visFrameView frame_to_copy) {
    _metadata->fpga_seq_start = frame_to_copy.metadata()->fpga_seq_start;
    _metadata->fpga_seq_length = frame_to_copy.metadata()->fpga_seq_length;
    _metadata->fpga_seq_total = frame_to_copy.metadata()->fpga_seq_total;
    _metadata->ctime = frame_to_copy.metadata()->ctime;
    _metadata->freq_id = frame_to_copy.metadata()->freq_id;
    _metadata->dataset_id = frame_to_copy.metadata()->dataset_id;
}

// Copy the non-visibility parts of the buffer
void visFrameView::copy_nonvis_buffer(visFrameView frame_to_copy) {

    // Copy eigenvector parts
    std::copy(frame_to_copy.eval.begin(),
              frame_to_copy.eval.end(),
              eval.begin());
    std::copy(frame_to_copy.evec.begin(),
              frame_to_copy.evec.end(),
              evec.begin());
    erms = frame_to_copy.erms;

    // Copy per input flags
    std::copy(frame_to_copy.flags.begin(),
              frame_to_copy.flags.end(),
              flags.begin());

    // Copy gains
    std::copy(frame_to_copy.gain.begin(),
              frame_to_copy.gain.end(),
              gain.begin());
}

struct_layout<visField> visFrameView::calculate_buffer_layout(
    uint32_t num_elements, uint32_t num_prod, uint32_t num_ev
)
{
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<visField, size_t, size_t>> buffer_members = {
        std::make_tuple(visField::vis, sizeof(cfloat), num_prod),
        std::make_tuple(visField::weight, sizeof(float),  num_prod),
        std::make_tuple(visField::flags, sizeof(float),  num_elements),
        std::make_tuple(visField::eval, sizeof(float),  num_ev),
        std::make_tuple(visField::evec, sizeof(cfloat), num_ev * num_elements),
        std::make_tuple(visField::erms, sizeof(float),  1),
        std::make_tuple(visField::gain, sizeof(cfloat), num_elements)
    };

    return struct_alignment(buffer_members);
}

void visFrameView::fill_chime_metadata(const chimeMetadata * chime_metadata) {

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
    if(is_gps_global_time_set()) {
        ts = chime_metadata->gps_time;
    } else{
        TIMEVAL_TO_TIMESPEC(&(chime_metadata->first_packet_recv_time), &ts);
    }

    time = std::make_tuple(fpga_seq, ts);
}
