#ifndef REST_INSPECT_FRAME_HPP
#define REST_INSPECT_FRAME_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include <mutex>    // for mutex
#include <stdint.h> // for int32_t, uint8_t
#include <string>   // for string

/**
 * @class restInspectFrame
 * @brief Exposes the binary contents of a buffer frame (or subset there of) to
 *        the REST server via a GET request.
 *
 * Returns the latest data copied into the internal frame @c frame_copy.
 * The length of this data is either the frame size, or the @c len config option
 * When returning data to the REST client, this class will not update @c frame_copy
 * So it's possible that if too many requests are made, it will keep returning the
 * same (old) frame contents.  This is a side effect of trying to prevent the system
 * from locking up the buffer itself.  We might want to adjust this someday, but for
 * testing this seems like a reasonable compromise.
 *
 * @par REST Endpoints
 * @endpoint /inspect_frame/\<buffer name\> ``GET`` Returns binary data from the
 *           latest frame in the buffer given in @p in_buf
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer
 *     @buffer_format Any
 *     @buffer_metadata Any
 * @conf   len   Int. the amount of bindary data in bytes to return from the
 *               front of the latest frame. Default the frame size of @c in_buf
 *               Note if set to zero, this will be set to frame size of @c in_buf
 *
 * @warning This stage makes a copy of the data in each and every frame ( upto @c len ).
 *          So it should not be used in places where this extra memory copy would be
 *          expensive for the system to deal with, and should only be enabled when it
 *          is needed for trouble shooting.
 *
 * @todo Once the new buffers are implemented this should use the frame "freeze out"
 *       method instead of doing memory copies.
 *
 * @author Andre Renard
 */
class restInspectFrame : public kotekan::Stage {
public:
    /// Constructor
    restInspectFrame(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);

    /// Destructor
    virtual ~restInspectFrame();

    /// Gets the latest frame from @c in_buf and copies it to @c frame_copy
    void main_thread() override;

    /**
     * @brief Retruns the binary data in @c frame_copy to the REST client
     *
     * Internal callback function, shouldn't be directly called
     * outside the HTTP/REST server
     *
     * @param conn The HTTP connection object
     */
    void rest_callback(kotekan::connectionInstance& conn);

private:
    /// The buffer to allow inspections on.
    struct Buffer* in_buf;

    /// The name (from the config) of the buffer we are inspecting
    std::string in_buf_config_name;

    /// The part of the frame to hold off to side for
    /// the REST call back.
    uint8_t* frame_copy;

    /// Locks changes to @c frame_copy
    std::mutex frame_copy_lock;

    /// The REST server endpoint name
    std::string endpoint;

    /// Has the REST server end point been registered?
    bool registered;

    /// The length of the frame_copy array.
    int32_t len;
};

#endif /* REST_INSPECT_FRAME_HPP */
