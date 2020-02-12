/*****************************************
@file
@brief Code for using the visBuffer formatted data.
- visMetadata
- frameView
*****************************************/
#ifndef FRAMEVIEW_HPP
#define FRAMEVIEW_HPP

#include "Hash.hpp"        // for Hash
#include "buffer.h"        // for Buffer
#include "chimeMetadata.h" // for chimeMetadata
#include "dataset.hpp"     // for dset_id_t
#include "visUtil.hpp"     // for cfloat

#include "gsl-lite.hpp" // for span

#include <set>      // for set
#include <stdint.h> // for uint32_t, uint64_t, uint8_t
#include <string>   // for string
#include <time.h>   // for timespec
#include <tuple>    // for tuple
#include <utility>  // for pair


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

    // Pointer to frame data. In theory this is redundant as it can be derived
    // from buffer and id, but it's nice for brevity
    uint8_t* const _frame;
};

#endif
