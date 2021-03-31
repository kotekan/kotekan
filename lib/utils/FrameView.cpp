#include "FrameView.hpp"

#include "metadata.h" // for metadataContainer

#include "fmt.hpp" // for format, fmt

#include <stdexcept> // for runtime_error
#include <string.h>  // for memcpy
#include <string>    // for string

FrameView::FrameView(Buffer* buf, int frame_id) :
    buffer(buf),
    id(frame_id),
    _frame(buffer->frames[id]) {}

FrameView::~FrameView(){};

void FrameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest, int frame_id_dest) {
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
}
