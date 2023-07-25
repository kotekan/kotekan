/*****************************************
@file
@brief Code for using the BasebandFrameView formatted data.
- BasebandFrameView
*****************************************/
#ifndef BASEBAND_FRAME_VIEW_HPP
#define BASEBAND_FRAME_VIEW_HPP

#include "BasebandMetadata.hpp" // for BasebandMetadata
#include "FrameView.hpp"        // for FrameView
#include "buffer.hpp"             // for Buffer

#include <cstddef> // for size_t

/**
 * @class HFBFrameView
 * @brief Provide a structured view of a baseband dump buffer.
 *
 * This class inherits from the FrameView base class and sets up a view on a
 * baseband dump buffer with the ability to interact with the data and metadata.
 *
 * @author Davor Cubranic
 **/
class BasebandFrameView : public FrameView {

public:
    /**
     * @brief Create the view from an existing buffer.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    BasebandFrameView(Buffer* buf, int frame_id);

    /**
     * @brief Read only access to the metadata.
     * @returns The metadata.
     **/
    const BasebandMetadata* metadata() const;

    size_t data_size() const override;

    void zero_frame() override;

private:
    // References to the metadata we are viewing
    BasebandMetadata* const _metadata;
};

#endif // BASEBAND_FRAME_VIEW_HPP
