/*****************************************
@file
@brief Code for using the VisFrameView formatted data.
- VisMetadata
- VisFrameView
*****************************************/
#ifndef VISBUFFER_HPP
#define VISBUFFER_HPP

#include "Config.hpp"         // for Config
#include "FrameView.hpp"      // for FrameView
#include "Telescope.hpp"      // for freq_id_t
#include "buffer.h"           // for Buffer
#include "chimeMetadata.hpp"  // for chimeMetadata
#include "datasetManager.hpp" // for dset_id_t
#include "visUtil.hpp"        // for cfloat

#include "gsl-lite.hpp" // for span

#include <set>      // for set
#include <stdint.h> // for uint32_t, uint64_t
#include <string>   // for string
#include <time.h>   // for size_t, timespec
#include <tuple>    // for tuple
#include <utility>  // for pair


/**
 * @brief The fields within the VisFrameView.
 *
 * Use this enum to refer to the fields.
 **/
enum class VisField { vis, weight, flags, eval, evec, erms, gain };


/**
 * @struct VisMetadata
 * @brief Metadata for the visibility style buffers
 *
 * @author Richard Shaw
 **/
struct VisMetadata {

    /// The FPGA sequence number of the integration frame
    uint64_t fpga_seq_start;
    /// The ctime of the integration frame
    timespec ctime;
    /// Nominal length of the frame in FPGA ticks
    uint64_t fpga_seq_length;
    /// Amount of data that actually went into the frame (in FPGA ticks)
    uint64_t fpga_seq_total;
    /// The number of FPGA frames flagged as containing RFI. NOTE: This value
    /// might contain overlap with lost samples, as that counts missing samples
    /// as well as RFI. For renormalization this value should NOT be used, use
    /// lost samples (= @c fpga_seq_length - @c fpga_seq_total) instead.
    uint64_t rfi_total;

    /// ID of the frequency bin
    freq_id_t freq_id;
    /// ID of the dataset
    dset_id_t dataset_id;

    /// Number of elements for data in buffer
    uint32_t num_elements;
    /// Number of products for data in buffer
    uint32_t num_prod;
    /// Number of eigenvectors and values calculated
    uint32_t num_ev;
};


/**
 * @class VisFrameView
 * @brief Provide a structured view of a visibility buffer.
 *
 * This class inherits from the FrameView base class and sets up a view on a visibility buffer with
 *the ability to interact with the data and metadata. Structural parameters can only be set at
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
 * @author Richard Shaw and James Willis
 **/
class VisFrameView : public FrameView {

public:
    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    VisFrameView(Buffer* buf, int frame_id);

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
     * @returns A VisFrameView of the copied frame.
     *
     **/
    static VisFrameView copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                   int frame_id_dest);

    /**
     * @brief Get the layout of the buffer from the structural parameters.
     *
     * @param num_elements     Number of elements.
     * @param num_prod         Number of products.
     * @param num_ev           Number of eigenvectors.
     *
     * @returns A mnonvis_bufferap from member name to start and end in bytes. The start
     *          (i.e. 0) and end (i.e. total size) of the buffer is contained in
     *          `_struct`.
     **/
    static struct_layout<VisField> calculate_buffer_layout(uint32_t num_elements, uint32_t num_prod,
                                                           uint32_t num_ev);
    /**
     * @brief Get the size of the frame using the config file.
     *
     * @param config      Config file.
     * @param unique_name Path to stage in config file.
     *
     * @returns Size of frame.
     **/
    static size_t calculate_frame_size(kotekan::Config& config, const std::string& unique_name);

    /**
     * @brief Get the size of the frame.
     *
     * @param num_elements     Number of elements.
     * @param num_prod         Number of products.
     * @param num_ev           Number of eigenvectors.
     *
     * @returns Size of frame.
     **/
    static size_t calculate_frame_size(uint32_t num_elements, uint32_t num_prod, uint32_t num_ev);

    size_t data_size() const override;

    void zero_frame() override;

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
     * @param  frame_to_copy  Frame to copy metadata from.
     *
     **/
    void copy_metadata(VisFrameView frame_to_copy);

    /**
     * @brief Copy over the data, skipping specified members.
     *
     * This routine copys member by member and the structural parameters of the
     * buffer only need to match for the members actually being copied. If they
     * don't match an exception is thrown.
     *
     * @note To copy the whole frame it is more efficient to use the copying
     * constructor.
     *
     * @param  frame_to_copy  Frame to copy metadata from.
     * @param  skip_members   Specify a set of data members to *not* copy.
     *
     **/
    void copy_data(VisFrameView frame_to_copy, const std::set<VisField>& skip_members);

    /**
     * @brief Fill the VisMetadata from a chimeMetadata struct.
     *
     * The time field is filled with the GPS time if it is set (checked via
     * `Telescope.gps_time_enabled`), otherwise the `first_packet_recv_time` is
     * used. Also note, there is no dataset information in chimeMetadata so the
     * `dataset_id` is set to zero.
     *
     * @param chime_metadata Metadata to fill from.
     * @param ind            Frequency ind for multifrequency buffers (use zero
     *                       if not multifrequency)
     *
     **/
    void fill_chime_metadata(const chimeMetadata* chime_metadata, uint32_t ind);

    /**
     * @brief Populate metadata.
     *
     * @param metadata     Metadata to populate.
     * @param num_elements Number of elements.
     * @param num_prod     Number of products.
     * @param num_ev       Number of eigenvectors.
     *
     **/
    static void set_metadata(VisMetadata* metadata, const uint32_t num_elements,
                             const uint32_t num_prod, const uint32_t num_ev);

    /**
     * @brief Populate metadata.
     *
     * @param buf          Buffer.
     * @param index        Index into buffer.
     * @param num_elements Number of elements.
     * @param num_prod     Number of products.
     * @param num_ev       Number of eigenvectors.
     *
     **/
    static void set_metadata(Buffer* buf, const uint32_t index, const uint32_t num_elements,
                             const uint32_t num_prod, const uint32_t num_ev);

    /**
     * @brief Populate metadata and frame view.
     *
     * @param buf            Buffer.
     * @param index          Index into buffer.
     * @param num_elements   Number of elements.
     * @param num_prod       Number of products.
     * @param num_ev         Number of eigenvectors.
     * @param alloc_metadata Bool to allocate metadata or not.
     *
     **/
    static VisFrameView create_frame_view(Buffer* buf, const uint32_t index,
                                          const uint32_t num_elements, const uint32_t num_prod,
                                          const uint32_t num_ev, bool alloc_metadata = true);

    /**
     * @brief Read only access to the metadata.
     * @returns The metadata.
     **/
    const VisMetadata* metadata() const {
        return _metadata;
    }

private:
    // References to the metadata we are viewing
    VisMetadata* const _metadata;

    // The calculated layout of the buffer
    struct_layout<VisField> buffer_layout;

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

    /// The number of lost samples due to RFI
    uint64_t& rfi_total;

    /// A reference to the frequency ID.
    uint32_t& freq_id;
    /// A reference to the dataset ID.
    dset_id_t& dataset_id;

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
