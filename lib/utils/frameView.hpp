#ifndef FRAMEVIEW_HPP
#define FRAMEVIEW_HPP

#include "buffer.h"
#include "chimeMetadata.h"
#include "datasetManager.hpp"
#include "frameView.hpp"
#include "visUtil.hpp"
#include "Config.hpp"

#include "gsl-lite.hpp"

#include <complex>
#include <set>
#include <sys/time.h>
#include <time.h>
#include <tuple>

/**
 * @class frameView
 * @brief Provide a structured view of a visibility buffer.
 *
 * This class sets up a view on a visibility buffer with the ability to
 * interact with the data and metadata. Structural parameters can only be set at
 * creation, everything else is returned as a reference or pointer so can be
 * modified at will.
 *
 * @note There are multiple constructors: one for viewing already initialised
 *       buffers; one for initialising a buffer and returning a view of it; and
 *       one for copying an existing buffer into a new location and returning a
 *       view of that. Make sure to pick the right one!
 *
 * @todo This may want changing to use reference wrappers instead of bare
 *       references.
 *
 * @author Richard Shaw
 **/
class frameView {

public:
    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    frameView(Buffer* buf, int frame_id);

    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    frameView(Buffer* buf, int frame_id, kotekan::Config config, const string& unique_name);

    /**
     * @brief Copy frame to a new buffer and create view of copied frame
     *
     * This should be used for copying a frame from one buffer to another.
     *
     * @param buf              The buffer the frame is in.
     * @param frame_id         The id of the frame to read.
     * @param frame_to_copy    An instance of frameView corresponding to the frame to be copied.
     *
     * @warning The metadata object must already have been allocated.
     **/
    frameView(Buffer* buf, int frame_id, frameView *frame_to_copy);

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
     * @returns A frameView of the copied frame.
     *
     **/
    //static frameView copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
    //                               int frame_id_dest);

    /**
     * @brief Get the layout of the buffer from the structural parameters.
     *
     * @param config Config file variables.
     **/
    virtual void calculate_buffer_layout(kotekan::Config& config, const string& unique_name) = 0;

    /**
     * @brief Return a summary of the buffer contents.
     *
     * @returns A string summarising the contents.
     **/
    virtual std::string summary() const = 0;

    /**
     * @brief Read only access to the metadata.
     * @returns The metadata.
     **/
    //const visMetadata* metadata() const {
    //    return _metadata;
    //}

    /**
     * @brief Read only access to the frame data.
     * @returns The data.
     **/
    const uint8_t* data() const {
        return _frame;
    }

protected:
    // References to the buffer and metadata we are viewing
    Buffer* const buffer;
    const int id;
    //visMetadata* const _metadata;

    // Pointer to frame data. In theory this is redundant as it can be derived
    // from buffer and id, but it's nice for brevity
    uint8_t* const _frame;

    // The calculated layout of the buffer
    //struct_layout<visField> buffer_layout;

    // NOTE: these need to be defined in a final public block to ensure that they
    // are initialised after the above members.
};

#endif
