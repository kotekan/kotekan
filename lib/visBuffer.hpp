/*****************************************
@file
@brief Code for using the visBuffer formatted data.
- visMetadata
- visFrameView
*****************************************/
#ifndef VISBUFFER_HPP
#define VISBUFFER_HPP

#include <time.h>
#include <sys/time.h>
#include <tuple>
#include <complex>
#include <gsl/gsl>

#include "visUtil.hpp"

#include "buffer.h"

/**
 * @struct visMetadata
 * @brief Metadata for the visibility style buffers
 *
 * @author Richard Shaw
 */
struct visMetadata {

    /// The FPGA sequence number of the integration frame
    uint64_t fpga_seq_num;
    /// The ctime of the integration frame
    timespec ctime;

    /// ID of the frequency bin
    uint16_t freq_id;

    /// ID of the dataset (vis, gatedvisX ...), main vis dataset = 0
    uint16_t dataset_id;

    /// Number of elements for data in buffer
    uint32_t num_elements;
    /// Number of products for data in buffer
    uint32_t num_prod;

    /// Number of eigenvectors and values calculated
    uint16_t num_eigenvectors;

};


/**
 * @class visFrameView
 * @brief Provide a structured view of a visibility buffer.
 *
 * This class sets up a view on a visibillity buffer with the ability to
 * interact with the data and metadata. Structural parameters can only be set at
 * creation, everything else is returned as a reference or pointer so can be
 * modified at will.
 *
 * @author Richard Shaw
 */
class visFrameView {

public:

    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    visFrameView(Buffer * buf, int frame_id);

    /**
     * @brief Create view and set structure metadata.
     *
     * This should be used for creating entirely new frames. This overload also
     * assumes the full visibility triangle is being stored.
     *
     * @param buf              The buffer the frame is in.
     * @param frame_id         The id of the frame to read.
     * @param num_elements     Number of elements in the data.
     * @param num_eigenvectors Number of eigenvectors to hold.
     *
     * @warning The metadata object must already have been allocated.
     */
    visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                 uint16_t num_eigenvectors);

    /**
     * @brief Create view and set structure metadata.
     *
     * This should be used for creating entirely new frames. This overload takes
     * the number of products as a parameter.
     *
     * @param buf              The buffer the frame is in.
     * @param frame_id         The id of the frame to read.
     * @param num_elements     Number of elements in the data.
     * @param num_prod         Number of products in the data.
     * @param num_eigenvectors Number of eigenvectors to hold.
     *
     * @warning The metadata object must already have been allocated.
     */
    visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                 uint32_t num_prod, uint16_t num_eigenvectors);

    /**
     * @brief Copy frame to a new buffer and create view of copied frame
     *
     * This should be used for copying a frame from one buffer to another.
     *
     * @param buf              The buffer the frame is in.
     * @param frame_id         The id of the frame to read.
     * @param frame_to_copy    An instance of visFrameView corresponding to the frame to be copied.
     *
     * @warning The metadata object must already have been allocated.
     */
    visFrameView(Buffer * buf, int frame_id, visFrameView frame_to_copy);

    /**
     * @brief Get the layout of the buffer from the structural parameters.
     *
     * @param num_elements     Number of elements.
     * @param num_prod         Number of products.
     * @param num_eigenvectors Number of eigenvectors.
     *
     * @returns A map from member name to start and end in bytes. The start
     *          (i.e. 0) and end (i.e. total size) of the buffer is contained in
     *          `_struct`.
     */
    static struct_layout bufferLayout(uint32_t num_elements,
                                      uint32_t num_prod,
                                      uint16_t num_eigenvectors);

    /// Return a summary of the visibility buffer contents
    std::string summary() const;

private:

    // References to the buffer and metadata we are viewing
    Buffer * const buffer;
    const int id;
    visMetadata * const metadata;

    // Pointer to frame data. In theory this is redundant as it can be derived
    // from buffer and id, but it's nice for brevity
    uint8_t * const frame;

    // The calculated layout of the buffer
    struct_layout buffer_layout;

// NOTE: these need to be defined in a final public block to ensure that they
// are initialised after the above members.
public:

    /// The number of elements in the data (read only).
    const uint32_t& num_elements;
    /// The number of products in the data (read only).
    const uint32_t& num_prod;
    /// The number of eigenvectors/values in the data (read only).
    const uint32_t& num_eigenvectors;

    /// A tuple of references to the underlying time parameters
    std::tuple<uint64_t&, timespec&> time;
    /// A reference to the frequency ID.
    uint16_t& freq_id;
    /// A reference to the dataset ID.
    uint16_t& dataset_id;

    /// View of the visibility data.
    const gsl::span<cfloat> vis;
    /// View of the weight data.
    const gsl::span<float> weight;
    /// View of the eigenvalues.
    const gsl::span<float> eigenvalues;
    /// View of the eigenvectors (packed as ev,feed).
    const gsl::span<cfloat> eigenvectors;
    /// The RMS of residual visibilities
    float& rms;

};



#endif
