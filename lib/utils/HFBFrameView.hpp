/*****************************************
@file
@brief Code for using the HFBFrameView formatted data.
- HFBMetadata
- HFBFrameView
*****************************************/
#ifndef HFBBUFFER_HPP
#define HFBBUFFER_HPP

#include "FrameView.hpp"   // for FrameView
#include "HFBMetadata.hpp" // for HFBMetadata
#include "Hash.hpp"        // for Hash
#include "buffer.h"        // for Buffer
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
 * @brief The fields within the hfbBuffer.
 *
 * Use this enum to refer to the fields.
 **/
enum class HFBField { hfb, weight };

/**
 * @class HFBFrameView
 * @brief Provide a structured view of a hyperfine beam buffer.
 *
 * This class inherits from the FrameView base class and sets up a view on a hyperfine beam buffer
 *with the ability to interact with the data and metadata. Structural parameters can only be set at
 * creation, everything else is returned as a reference or pointer so can be
 * modified at will.
 *
 * @author James Willis
 **/
class HFBFrameView : public FrameView {

public:
    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    HFBFrameView(Buffer* buf, int frame_id);

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
     * @returns A HFBFrameView of the copied frame.
     *
     **/
    static HFBFrameView copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
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
    static struct_layout<HFBField> calculate_buffer_layout(uint32_t num_beams,
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

    size_t data_size();

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
    void copy_metadata(HFBFrameView frame_to_copy);

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
    void copy_data(HFBFrameView frame_to_copy, const std::set<HFBField>& skip_members);

    /**
     * @brief Populate metadata.
     *
     * @param metadata    Metadata to populate.
     * @param num_beams   Number of beams.
     * @param num_subfreq Number of sub-frequencies.
     *
     **/
    static void set_metadata(HFBMetadata* metadata, const uint32_t num_beams,
                             const uint32_t num_subfreq);

    /**
     * @brief Populate metadata.
     *
     * @param buf         Buffer.
     * @param index       Index into buffer.
     * @param num_beams   Number of beams.
     * @param num_subfreq Number of sub-frequencies.
     *
     **/
    static void set_metadata(Buffer* buf, const uint32_t index, const uint32_t num_beams,
                             const uint32_t num_subfreq);

    /**
     * @brief Populate metadata and frame view.
     *
     * @param buf            Buffer.
     * @param index          Index into buffer.
     * @param num_beams      Number of beams.
     * @param num_subfreq    Number of sub-frequencies.
     * @param alloc_metadata Bool to allocate metadata or not.
     *
     **/
    static HFBFrameView create_frame_view(Buffer* buf, const uint32_t index,
                                          const uint32_t num_beams, const uint32_t num_subfreq,
                                          bool alloc_metadata = true);

    /**
     * @brief Read only access to the metadata.
     * @returns The metadata.
     **/
    const HFBMetadata* metadata() const {
        return _metadata;
    }

private:
    // References to the metadata we are viewing
    HFBMetadata* const _metadata;

    // The calculated layout of the buffer
    struct_layout<HFBField> buffer_layout;

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
    int64_t& fpga_seq_start;
    /// Number of samples integrated
    uint64_t& fpga_seq_total;
    /// Number of samples expected
    uint64_t& fpga_seq_length;
    /// A reference to the frequency ID.
    freq_id_t& freq_id;
    /// A reference to the dataset ID.
    dset_id_t& dataset_id;

    /// View of the hyperfine beam data.
    const gsl::span<float> hfb;
    /// View of the weight data.
    const gsl::span<float> weight;
};

#endif
