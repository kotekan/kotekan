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

#include "gsl-lite.hpp"

#include "visUtil.hpp"
#include "buffer.h"
#include "chimeMetadata.h"

/**
 * @struct visMetadata
 * @brief Metadata for the visibility style buffers
 *
 * @author Richard Shaw
 **/
struct visMetadata {

    /// The FPGA sequence number of the integration frame
    uint64_t fpga_seq_start;
    /// The ctime of the integration frame
    timespec ctime;
    /// Nominal length of the frame in FPGA ticks
    uint64_t fpga_seq_length;
    // Amount of data that actually went into the frame (in FPGA ticks)
    uint64_t fpga_seq_total;

    /// ID of the frequency bin
    uint32_t freq_id;
    /// ID of the dataset (vis, gatedvisX ...), main vis dataset = 0
    uint64_t dataset_id;

    /// Number of elements for data in buffer
    uint32_t num_elements;
    /// Number of products for data in buffer
    uint32_t num_prod;
    /// Number of eigenvectors and values calculated
    uint32_t num_ev;

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
     * @param num_ev           Number of eigenvectors to hold.
     *
     * @warning The metadata object must already have been allocated.
     **/
    visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                 uint32_t num_ev);

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
     * @param num_ev           Number of eigenvectors to hold.
     *
     * @warning The metadata object must already have been allocated.
     **/
    visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                 uint32_t num_prod, uint32_t num_ev);

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
     **/
    visFrameView(Buffer * buf, int frame_id, visFrameView frame_to_copy);

    /**
     * @brief Copy a whole frame from a buffer and create a view of it.
     *
     * This will attempt to do a zero copy transfer of the frame for speed, and
     * fall back on a full copy if any other processes consume from the input
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
     * @returns A visFrameView of the copied frame.
     *
     **/
    static visFrameView copy_frame(Buffer* buf_src, int frame_id_src,
                                   Buffer* buf_dest, int frame_id_dest);

    /**
     * @brief Get the layout of the buffer from the structural parameters.
     *
     * @param num_elements     Number of elements.
     * @param num_prod         Number of products.
     * @param num_ev           Number of eigenvectors.
     *
     * @returns A map from member name to start and end in bytes. The start
     *          (i.e. 0) and end (i.e. total size) of the buffer is contained in
     *          `_struct`.
     **/
    static struct_layout calculate_buffer_layout(uint32_t num_elements,
                                                 uint32_t num_prod,
                                                 uint32_t num_ev);

    /**
     * @brief Return a summary of the visibility buffer contents.
     *
     * @returns A string summarising the contents.
     **/
    std::string summary() const;

    /**
     * @brief Copy the non-const parts of the metadata.
     *
     * Transfers all the non-structural metadata from the source frame.
     *
     * @param frame_to_copy Frame to copy metadata from.
     *
     **/
    void copy_nonconst_metadata(visFrameView frame_to_copy);

    /**
     * @brief Copy the non-visibility parts of the buffer.
     *
     * Transfers all the datasets except the visibilities and their weights.
     *
     * @param frame_to_copy Frame to copy metadata from.
     *
     **/
    void copy_nonvis_buffer(visFrameView frame_to_copy);

    /**
     * @brief Fill the visMetadata from a chimeMetadata struct.
     *
     * The time field is filled with the GPS time if it is set (checked via
     * `is_gps_global_time_set`), otherwise the `first_packet_recv_time` is
     * used. Also note, there is no dataset information in chimeMetadata so the
     * `dataset_id` is set to zero.
     *
     * @param chime_metadata Metadata to fill from.
     *
     **/
    void fill_chime_metadata(const chimeMetadata* chime_metadata);

    /**
     * @brief Read only access to the metadata.
     * @returns The metadata.
     **/
    const visMetadata* metadata() const { return _metadata; }

    /**
     * @brief Read only access to the frame data.
     * @returns The data.
     **/
    const uint8_t* data() const { return _frame; }

private:

    // References to the buffer and metadata we are viewing
    Buffer * const buffer;
    const int id;
    visMetadata * const _metadata;

    // Pointer to frame data. In theory this is redundant as it can be derived
    // from buffer and id, but it's nice for brevity
    uint8_t * const _frame;

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
    const uint32_t& num_ev;

    /// A tuple of references to the underlying time parameters
    std::tuple<uint64_t&, timespec&> time;
    /// The nominal frame length in FPGA ticks
    uint64_t& fpga_seq_length;
    /// The actual amount of data accumulated in FPGA ticks
    uint64_t& fpga_seq_total;

    /// A reference to the frequency ID.
    uint32_t& freq_id;
    /// A reference to the dataset ID.
    uint64_t& dataset_id;

    /// View of the visibility data.
    const gsl::span<cfloat> vis;
    /// View of the weight data.
    const gsl::span<float> weight;
    /// View of the input flags
    const gsl::span<float> flags;
    /// View of the eigenvalues.
    const gsl::span<float> eval;
    /// View of the eigenvectors (packed as ev,feed).
    const gsl::span<cfloat> evec;
    /// The RMS of residual visibilities
    float& erms;
    /// View of the applied gains
    const gsl::span<cfloat> gain;

};



#endif
