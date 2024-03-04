#include "FrameView.hpp"

#include "metadata.hpp" // for metadataContainer

#include "fmt.hpp" // for format, fmt

#include <stdexcept> // for runtime_error
#include <string.h>  // for memcpy
#include <string>    // for string

FrameView::FrameView(Buffer* buf, int frame_id) :
    buffer(buf), id(frame_id), _frame(buffer->frames[id]) {}

FrameView::~FrameView(){};

void FrameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest, int frame_id_dest) {
    buf_dest->allocate_new_metadata_object(frame_id_dest);

    // Buffer sizes must match exactly
    if (buf_src->frame_size != buf_dest->frame_size) {
        std::string msg =
            fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->frame_size, buf_dest->frame_size);
        throw std::runtime_error(msg);
    }

    // Metadata sizes must match exactly
    if (buf_src->metadata[frame_id_src]->get_object_size()
        != buf_dest->metadata[frame_id_dest]->get_object_size()) {
        std::string msg =
            fmt::format(fmt("Metadata sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->metadata[frame_id_src]->get_object_size(),
                        buf_dest->metadata[frame_id_dest]->get_object_size());
        throw std::runtime_error(msg);
    }

    // Calculate the number of consumers on the source buffer
    int num_consumers = buf_src->get_num_consumers();

    // Copy or transfer the data part.
    if (num_consumers == 1) {
        // Transfer frame contents with directly...
        buf_src->swap_frames(frame_id_src, buf_dest, frame_id_dest);
    } else if (num_consumers > 1) {
        // Copy the frame data over, leaving the source intact
        std::memcpy(buf_dest->frames[frame_id_dest], buf_src->frames[frame_id_src],
                    buf_src->frame_size);
    }

    // Copy over the metadata
    buf_dest->metadata[frame_id_dest]->deepCopy(buf_src->metadata[frame_id_src]);
}
