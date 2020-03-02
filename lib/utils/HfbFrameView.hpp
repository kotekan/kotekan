/*****************************************
@file
@brief Code for using the hfbBuffer formatted data.
- hfbMetadata
- HfbFrameView
*****************************************/
#ifndef HFBBUFFER_HPP
#define HFBBUFFER_HPP

#include "FrameView.hpp"   // for FrameView
#include "Hash.hpp"        // for Hash
#include "buffer.h"        // for Buffer
#include "dataset.hpp"     // for dset_id_t
#include "hfbMetadata.hpp" // for hfbMetadata
#include "visUtil.hpp"     // for cfloat

#include "gsl-lite.hpp" // for span

#include <set>      // for set
#include <stdint.h> // for uint32_t, uint64_t, uint8_t
#include <string>   // for string
#include <time.h>   // for timespec
#include <tuple>    // for tuple
#include <utility>  // for pair


/**
 * @brief The fields within the hfbBuffer.
 *
 * Use this enum to refer to the fields.
 **/
enum class hfbField { hfb };

/**
 * @class hfbFrameView
 * @brief Provide a structured view of a hyperfine beam buffer.
 *
 * This class sets up a view on a hyperfine beam buffer with the ability to
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
 * @author James Willis
 **/

class HfbFrameView : public FrameView {

public:
    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    HfbFrameView(Buffer* buf, int frame_id);

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
     * @param buf_src       The buffer to copy from.
     * @param frame_id_src  The buffer location to copy from.
     * @param buf_dest      The buffer to copy into.
     * @param frame_id_dest The buffer location to copy into.
     *
     * @returns A HfbFrameView of the copied frame.
     *
     **/
    static HfbFrameView copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                   int frame_id_dest);

    /**
     * @brief Get the layout of the buffer from the structural parameters.
     *
     * @param num_beams   Number of beams.
     * @param num_subfreq Number of sub-frequencies.
     *
     * @returns A mnonhfb_bufferap from member name to start and end in bytes. The start
     *          (i.e. 0) and end (i.e. total size) of the buffer is contained in
     *          `_struct`.
     **/
    static struct_layout<hfbField> calculate_buffer_layout(uint32_t num_beams,
                                                           uint32_t num_subfreq);

    /**
     * @brief Get the size of the frame.
     *
     * @param num_beams   Number of beams.
     * @param num_subfreq Number of sub-frequencies.
     *
     * @returns Size of frame.
     **/
    static size_t calculate_frame_size(uint32_t num_beams, uint32_t num_subfreq);

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
     * @brief Return a summary of the hyper fine beam buffer contents.
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
    void copy_metadata(HfbFrameView frame_to_copy);

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
    void copy_data(HfbFrameView frame_to_copy, const std::set<hfbField>& skip_members);

    /**
     * @brief Populate metadata.
     *
     * @param metadata    Metadata to populate.
     * @param num_beams   Number of beams.
     * @param num_subfreq Number of sub-frequencies.
     *
     **/
    static void set_metadata(hfbMetadata* metadata, const uint32_t num_beams,
                             const uint32_t num_subfreq);

    /**
     * @brief Read only access to the metadata.
     * @returns The metadata.
     **/
    const hfbMetadata* metadata() const {
        return _metadata;
    }

private:
    // References to the metadata we are viewing
    hfbMetadata* const _metadata;

    // The calculated layout of the buffer
    struct_layout<hfbField> buffer_layout;

    // NOTE: these need to be defined in a final public block to ensure that they
    // are initialised after the above members.
public:
    /// The number of beams in the data (read only).
    const uint32_t& num_beams;
    /// The number of sub-frequencies in the data (read only).
    const uint32_t& num_subfreq;

    /// GPS time
    timespec& time;
    /// The ICEBoard sequence number
    int64_t& fpga_seq_num;
    /// Normalisation fraction
    float& norm_frac;
    /// Number of samples integrated
    uint32_t& num_samples_integrated;
    /// Number of samples expected
    uint32_t& num_samples_expected;
    /// A reference to the frequency ID.
    uint32_t& freq_id;
    /// A reference to the dataset ID.
    dset_id_t& dataset_id;

    /// View of the hyperfine beam data.
    const gsl::span<float> hfb;
};

#endif
