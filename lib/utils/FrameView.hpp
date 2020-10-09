/*****************************************
@file
@brief Base class for creating frame views.
- FrameView
*****************************************/
#ifndef FRAMEVIEW_HPP
#define FRAMEVIEW_HPP

#include <stdint.h>      // for uint8_t
#include <time.h>        // for size_t
#include <utility>       // for pair

#include "buffer.h"      // for Buffer
#include "gsl-lite.hpp"  // for span


/**
 * @class FrameView
 * @brief Provide a structured view of a buffer.
 *
 * This class sets up a base view on a buffer with the ability to
 * interact with the data. All classes which inherit from this should provide the following API:
 *
 * calculate_buffer_layout(...);
 * copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
 *            int frame_id_dest);
 * calculate_frame_size(kotekan::Config& config, const std::string& unique_name);
 * calculate_frame_size(...);
 * set_metdata(...);
 * create_frame_view(Buffer* buf, const uint32_t index, ...);
 * data_size();
 *
 * @author James Willis
 **/
class FrameView {

public:
    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    FrameView(Buffer* buf, int frame_id);

    virtual ~FrameView();

    /**
     * @brief Read only access to the frame data.
     * @returns The data.
     **/
    const uint8_t* data() const {
        return _frame;
    }

    /**
     * @brief Copy a whole frame from a buffer and create a view of it.
     *
     * This will attempt to do a zero copy transfer of the frame for speed, and
     * fall back on a full copy if any other stages consume from the input
     * buffer.
     *
     * @note This will allocate metadata for the destination.
     *
     * @warning This may invalidate anything pointing at the input buffer.
     *
     * @param buf_src        The buffer to copy from.
     * @param frame_id_src   The buffer location to copy from.
     * @param buf_dest       The buffer to copy into.
     * @param frame_id_dest  The buffer location to copy into.
     *
     **/
    static void copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest, int frame_id_dest);

    virtual size_t data_size() const = 0;

protected:
    // References to the buffer and metadata we are viewing
    Buffer* const buffer;
    const int id;

    // Pointer to frame data. In theory this is redundant as it can be derived
    // from buffer and id, but it's nice for brevity
    uint8_t* const _frame;
};

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

#endif
