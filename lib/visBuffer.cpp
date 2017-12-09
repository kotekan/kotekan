#include "visBuffer.hpp"


visFrameView::visFrameView(Buffer * buf, int frame_id) {

    buffer = buf;
    id = frame_id;
    metadata = (visMetadata *)buffer->metadata[id]->metadata;

    check_and_set();
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                           uint16_t num_eigenvector) {
    visFrameView(buf, frame_id, num_elements,
                 num_elements * (num_elements + 1) / 2, num_eigenvector);
}

visFrameView::visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                           uint32_t num_prod, uint16_t num_eigenvector) {

    buffer = buf;
    id = frame_id;
    metadata = (visMetadata *)buffer->metadata[id]->metadata;

    metadata->num_elements = num_elements;
    metadata->num_prod = num_prod;
    metadata->num_eigenvectors = num_eigenvectors;

    check_and_set();
}

void visFrameView::check_and_set() {

    // This defines the packing of the buffer. The order is somewhat funky to
    // try and ensure alignment of the members. For that to be true the buffer
    // size must be a multiple of the 16 (i.e. the size of a complex double).
    evec_ptr = (std::complex<double> *)(buffer->frames[id]);
    eval_ptr = (double *)(evec_ptr + num_eigenvectors() * num_elements());
    vis_ptr = (complex_int *)(eval_ptr + num_eigenvectors());
    rms_ptr = (double *)(vis_ptr + num_prod());

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


std::tuple<int64_t &, timespec &> visFrameView::time() {
    return std::make_tuple(metadata->fpga_seq_num, metadata->ctime);
}

uint16_t & visFrameView::freq_id() {
    return metadata->freq_id;
}

uint16_t & visFrameView::dataset_id() {
    return metadata->dataset_id;
}


complex_int * visFrameView::vis() {
    return vis_ptr;
}

double * visFrameView::eigenvalues() {
    return eval_ptr;
}

std::complex<double> * visFrameView::eigenvectors() {
    return evec_ptr;
}

double & visFrameView::rms() {
    return *rms_ptr;
}
