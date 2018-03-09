#include "visBuffer.hpp"
#include "gpsTime.h"


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
                 ((visMetadata *)(buf->metadata[frame_id]->metadata))->num_eigenvectors)
{
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                           uint32_t num_eigenvectors) :
    visFrameView(buf, frame_id, num_elements,
                 num_elements * (num_elements + 1) / 2, num_eigenvectors)
{
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t n_elements,
                           uint32_t n_prod, uint32_t n_eigenvectors) :
    buffer(buf),
    id(frame_id),
    metadata((visMetadata *)buf->metadata[id]->metadata),
    frame(buffer->frames[id]),

    // Calculate the internal buffer layout from the given structure params
    buffer_layout(calculate_buffer_layout(n_elements, n_prod, n_eigenvectors)),

    // Set the const refs to the structural metadata
    num_elements(metadata->num_elements),
    num_prod(metadata->num_prod),
    num_eigenvectors(metadata->num_eigenvectors),

    // Set the refs to the general metadata
    time(std::tie(metadata->fpga_seq_num, metadata->ctime)),
    freq_id(metadata->freq_id),
    dataset_id(metadata->dataset_id),

    // Bind the regions of the buffer to spans and refernces on the view
    vis(bind_span<cfloat>(frame, buffer_layout["vis"])),
    weight(bind_span<float>(frame, buffer_layout["weight"])),
    eigenvalues(bind_span<float>(frame, buffer_layout["evals"])),
    eigenvectors(bind_span<cfloat>(frame, buffer_layout["evecs"])),
    rms(bind_scalar<float>(frame, buffer_layout["rms"]))

{
    // Initialise the structure if not already done
    // NOTE: the provided structure params have already been used to calculate
    // the layout, but here we need to make sure the metadata tracks them too.
    metadata->num_elements = n_elements;
    metadata->num_prod = n_prod;
    metadata->num_eigenvectors = n_eigenvectors;

    // Check that the actual buffer size is big enough to contain the calculated
    // view
    size_t required_size = buffer_layout["_struct"].second;

    if(required_size > (uint32_t)buffer->frame_size) {
        throw std::runtime_error(
            "Visibility buffer too small. Must be a minimum of " +
            std::to_string((int)required_size) + " bytes."
        );
    }
}


visFrameView::visFrameView(Buffer * buf, int frame_id,
                           visFrameView frame_to_copy) :
    visFrameView(buf, frame_id, frame_to_copy.num_elements,
                 frame_to_copy.num_prod, frame_to_copy.num_eigenvectors)
{
    // Copy over the metadata values
    *metadata = *(frame_to_copy.metadata);

    // Copy the frame data here:
    // NOTE: this copies the full buffer memory, not only the individual components
    std::memcpy(buffer->frames[id],
                frame_to_copy.buffer->frames[frame_to_copy.id],
                frame_to_copy.buffer->frame_size);
}


std::string visFrameView::summary() const {

    std::ostringstream s;

    auto tm = gmtime(&(std::get<1>(time).tv_sec));

    s << "visBuffer[name=" << buffer->buffer_name << "]:"
      << " freq=" << freq_id
      << " dataset=" << dataset_id
      << " fpga_seq=" << std::get<0>(time)
      << " time=" << std::put_time(tm, "%F %T");

    return s.str();
}


// Copy the non-const parts of the metadata
void visFrameView::copy_nonconst_metadata(visFrameView frame_to_copy) {
    metadata->fpga_seq_num = frame_to_copy.metadata->fpga_seq_num;
    metadata->ctime = frame_to_copy.metadata->ctime;
    metadata->freq_id = frame_to_copy.metadata->freq_id;
    metadata->dataset_id = frame_to_copy.metadata->dataset_id;
}

// Copy the non-visibility parts of the buffer
void visFrameView::copy_nonvis_buffer(visFrameView frame_to_copy) {
    std::copy(frame_to_copy.eigenvalues.begin(), 
              frame_to_copy.eigenvalues.end(), 
              eigenvalues.begin());
    std::copy(frame_to_copy.eigenvectors.begin(),
              frame_to_copy.eigenvectors.end(), 
              eigenvectors.begin());
    rms = frame_to_copy.rms;
}

struct_layout visFrameView::calculate_buffer_layout(
    uint32_t num_elements, uint32_t num_prod, uint32_t num_eigenvectors
)
{
    // TODO: get the types of each element using a template on the member
    // definition
    std::vector<std::tuple<std::string, size_t, size_t>> buffer_members = {
        std::make_tuple("vis", sizeof(cfloat), num_prod),
        std::make_tuple("weight", sizeof(float),  num_prod),
        std::make_tuple("evals", sizeof(float),  num_eigenvectors),
        std::make_tuple("evecs", sizeof(cfloat), num_eigenvectors * num_elements),
        std::make_tuple("rms", sizeof(float),  1)
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
