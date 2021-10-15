#include "BasebandFrameView.hpp"

#include "metadata.h" // for metadataContainer

#include <string.h> // for memset

BasebandFrameView::BasebandFrameView(Buffer* buf, int frame_id) :
    FrameView(buf, frame_id),
    _metadata((BasebandMetadata*)buf->metadata[id]->metadata) {}


const BasebandMetadata* BasebandFrameView::metadata() const {
    return _metadata;
}


size_t BasebandFrameView::data_size() const {
    return buffer->frame_size;
}


void BasebandFrameView::zero_frame() {
    // Fill data with zeros
    std::memset(_frame, 0, buffer->frame_size);
}
