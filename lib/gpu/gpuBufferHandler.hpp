// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

#ifndef KOTEKAN_GPU_BUFFER_HANDLER_HPP
#define KOTEKAN_GPU_BUFFER_HANDLER_HPP

#include "Config.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "visUtil.hpp"

#include <string>
#include <vector>

/**
 * @brief Structure for holding a frame, frameID and buffer used as a return value
 *        by the helper object @c gpuBufferHandler
 */
struct NextFrameCollection {
    /// The buffer associated with the frame in this structure.
    struct Buffer* buf;

    /// The frame data, use the @c buf to get details like frame_size
    uint8_t* frame;

    /// The frameID of the @c frame in the ring buffer @c buf
    uint32_t frame_id;
};

/**
 * @brief Class to abstract some of the complex buffer handling logic for commandObjects
 *
 * This class abstracts the buffer handling in command objects so that the command object
 * simply needs to the functions:
 *     get_next_frame_precondition()
 *     get_next_frame_execute()
 *     get_next_frame_finalize() // Optional
 *     release_frame_finalize()
 * In each of the command object functional stages.  It is important that every one of these
 * be called in each stage for the buffer handling to work correctly,
 * except for @c get_next_frame_finalize()
 *
 * Each of the get_next_* functions returns a @c NextFrameCollection which contains the
 * buffer in question, the frame pointer, and the frame_id.  This can be used to reference
 * the data location for GPU commands, and also access or pass the metadata as needed.
 *
 * The class is designed to work with either one buffer, or an array of buffers, in the case of
 * an array, it uses a round-robin model to get the next frame from each buffer in the array in
 * order.   This is useful when combining multiple streams of data into the same GPU.
 *
 * @author Andre Renard
 */
class gpuBufferHandler {
public:
    gpuBufferHandler(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& host_buffers, const std::string buffer_entry,
                     const bool producer);

    /**
     * @brief Gets the next frame in the @c wait_on_precondition() function of a commandObject
     *
     * This function calls a blocking function to get the next frame, it should be used only
     * in the @c wait_on_precondition function, and the return value's frame pointer
     * should be checked to make sure the frame isn't nullptr, which indicates the exit condition
     *
     * @return A @c NextFrameCollection struct with the buffer point, and current frame and frameID
     */
    NextFrameCollection get_next_frame_precondition();

    /**
     * @brief Gets the frame data in the @c() execute function of a commandObject
     *
     * This call is non-blocking, and assumes @c get_next_frame_precondition() has been called in
     * advance.
     *
     * @return A @c NextFrameCollection struct with the buffer point, and current frame and frameID
     */
    NextFrameCollection get_next_frame_execute();

    /**
     * @brief Gets the frame data in the @c() finalize_frame function of a commandObject
     *
     * This function is non-blocking, and assumes @c get_next_frame_precondition() has been
     * called in advance.
     *
     * It is not required to call this function while using this helper class, it is only called
     * if the commandObject needs the details on the buffer and frame in-use.  e.g. for transferring
     * metadata data between buffers.
     *
     * @return A @c NextFrameCollection struct with the buffer point, and current frame and frameID
     */
    NextFrameCollection get_next_frame_finalize();

    /**
     * @brief Releases the current frame in the @c finalize_frame() function of a commandObject
     *
     * This function is non-blocking and must be called in the @c finalize_frame(), and assumes
     * a call to @c get_next_frame_precondition() has already been and in the
     * @c wait_on_precondition function of a commandObject.
     */
    void release_frame_finalize();

private:
    /**
     * @brief A simple collection of the buffer and its corresponding FrameIDs for the different
     * stages of the command object.
     */
    struct FrameIDCollection {
        /**
         * @brief Constructor for the FrameIDCollection structure, only used in @c gpuBufferHandler
         * @param buf The buffer to use.
         */
        FrameIDCollection(Buffer* buf) :
            buf(buf),
            precondition_id(buf),
            execute_id(buf),
            finalize_id(buf) {}

        Buffer* buf;
        frameID precondition_id;
        frameID execute_id;
        frameID finalize_id;
    };

    /// Array of input buffers with host memory to copy to device.
    std::vector<FrameIDCollection> buffers;

    /// This the index in @c buffers for the precondition function.  This not a frameID.
    int precondition_buffer_id;

    /// This the index in @c buffers for the execute function.  This not a frameID.
    int execute_buffer_id;

    /// This the index in @c buffers for the finalize function.  This not a frameID.
    int finalize_buffer_id;

    /// Set to true if the user of this class is producing frames, false for a consumer
    const bool producer;

    /// Producer/consumer identifier
    const std::string unique_name;
};


#endif // KOTEKAN_GPU_BUFFER_HANDLER_HPP
