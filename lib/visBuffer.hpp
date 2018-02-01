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
     * This should be used for viewing already create buffers.
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

    // Copy frame to new buffer and create view
    visFrameView(Buffer * buf, int frame_id,
                                visFrameView frame_to_copy);

    /// Return a tuple of references to the time parameters.
    std::tuple<uint64_t &, timespec &> time();
    /// Return a reference to the frequency ID.
    uint16_t & freq_id();
    /// Return a reference to the dataset ID.
    uint16_t & dataset_id();

    /// Return a pointer to the start of the visibility data.
    std::complex<float> * vis();
    /// Return a pointer to the start of the eigenvalues.
    float * eigenvalues();
    /// Return a pointer to the start of the eigenvector data.
    std::complex<float> * eigenvectors();
    /// Return a reference to the RMS parameteres.
    float & rms();

    /// Return a copy of the number of elements in the data.
    uint32_t num_elements();
    /// Return a copy of the number of products in the data.
    uint32_t num_prod();
    /// Return a copy of the number of eigenvectors/values in the data.
    uint32_t num_eigenvectors();

    /// Return a summary of the visibility buffer contents
    std::string summary();

private:

    // Pointers that will index into the buffer
    std::complex<float> * vis_ptr;
    float * eval_ptr;
    std::complex<float> * evec_ptr;
    float * rms_ptr;

    // References to the buffer and metadata we are using
    Buffer * const buffer;
    const int id;
    visMetadata * const  metadata;

    // Validate that the defined view fits in the space allocated
    void check_and_set();
};



#endif
