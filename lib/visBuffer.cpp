#include "visBuffer.hpp"


visFrameView::visFrameView(Buffer * buf, int frame_id) :
    buffer(buf),
    id(frame_id),
    metadata((visMetadata *)buf->metadata[id]->metadata)
{
    check_and_set();
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                           uint16_t num_eigenvector) :
    visFrameView(buf, frame_id, num_elements,
                 num_elements * (num_elements + 1) / 2, num_eigenvector )
{
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                           uint32_t num_prod, uint16_t num_eigenvector) :
    buffer(buf),
    id(frame_id),
    metadata((visMetadata *)buf->metadata[id]->metadata)
{
    metadata->num_elements = num_elements;
    metadata->num_prod = num_prod;
    metadata->num_eigenvectors = num_eigenvector;

    check_and_set();
}


std::string visFrameView::summary() {

    auto t = time();

    std::string msg = "visBuffer: freq=" + std::to_string(freq_id()) + " fpga_seq=" + std::to_string(std::get<0>(t));

    /*
    uint64_t fpga_seq = get_fpga_seq_num(buf, frame_id);
    stream_id_t stream_id = get_stream_id_t(buf, frame_id);
    timeval time_v = get_first_packet_recv_time(buf, frame_id);
    uint64_t lost_samples = get_lost_timesamples(buf, frame_id);

    char time_buf[64];
    time_t temp_time = time_v.tv_sec;
    struct tm* l_time = gmtime(&temp_time);
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", l_time);

    INFO("Metadata for %s[%d]: FPGA Seq: %" PRIu64
            ", stream ID = {create ID: %d, slot ID: %d, link ID: %d, freq ID: %d}, lost samples: %" PRIu64
            ", time stamp: %ld.%06ld (%s.%06ld)",
            buf->buffer_name, frame_id, fpga_seq,
            stream_id.crate_id, stream_id.slot_id,
            stream_id.link_id, stream_id.unused, lost_samples,
            time_v.tv_sec, time_v.tv_usec, time_buf, time_v.tv_usec);
    */

    return msg;

}


void visFrameView::check_and_set() {

    // This defines the packing of the buffer. The order is somewhat funky to
    // try and ensure alignment of the members. For that to be true the buffer
    // size must be a multiple of the 16 (i.e. the size of a complex double).
    evec_ptr = (std::complex<float> *)(buffer->frames[id]);
    eval_ptr = (float *)(evec_ptr + num_eigenvectors() * num_elements());
    vis_ptr = (std::complex<float> *)(eval_ptr + num_eigenvectors());
    rms_ptr = (float *)(vis_ptr + num_prod());

    // Reuse the pointer arithmetic we've already done to calculate the size
    size_t required_size = ((uint8_t *)(rms_ptr + 1) - buffer->frames[id]);

    if(required_size > buffer->frame_size) {
        throw std::runtime_error(
            "Visibility buffer too small. Must be a minimum of " +
            std::to_string((int)required_size) + " bytes."
        );
    }

}
uint32_t visFrameView::num_elements() {
    return metadata->num_elements;
}

uint32_t visFrameView::num_prod() {
    return metadata->num_prod;
}

uint32_t visFrameView::num_eigenvectors() {
    return metadata->num_eigenvectors;
}


std::tuple<uint64_t &, timespec &> visFrameView::time() {
    return std::tie(metadata->fpga_seq_num, metadata->ctime);
}

uint16_t & visFrameView::freq_id() {
    return metadata->freq_id;
}

uint16_t & visFrameView::dataset_id() {
    return metadata->dataset_id;
}


std::complex<float> * visFrameView::vis() {
    return vis_ptr;
}

float * visFrameView::eigenvalues() {
    return eval_ptr;
}

std::complex<float> * visFrameView::eigenvectors() {
    return evec_ptr;
}

float & visFrameView::rms() {
    return *rms_ptr;
}


// Buggy initial brinstorm for a member function that
// copies a frame to another buffer
// TODO: need to debug this and add definition to visBuffer.hpp
visFrameView visFrameView::copy_frame(Buffer * output_buffer, 
               int output_frame_id, uint32_t output_num_elements,
                                 uint16_t output_num_eigenvector) {

    auto output_frame = visFrameView(output_buffer, output_frame_id, 
                         output_num_elements,output_num_eigenvectors);


    //TODO: I would here do:
    //  output_frame.metadata = metadata
    //but this doesn't work because metadata is private
    //so I cannot directly access output_frame.metadata
    //What I might have to do is create a new member function that
    //a pointer to the metadata and call it here. 

}




