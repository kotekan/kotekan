/*****************************************
@file
@brief Code for using the hfbBuffer formatted data.
- hfbMetadata
- hfbFrameView
*****************************************/
#ifndef HFBBUFFER_HPP
#define HFBBUFFER_HPP

#include "buffer.h"
#include "chimeMetadata.h"
#include "visUtil.hpp"

#include "gsl-lite.hpp"

#include <complex>
#include <set>
#include <sys/time.h>
#include <time.h>
#include <tuple>


/**
 * @brief The fields within the hfbBuffer.
 *
 * Use this enum to refer to the fields.
 **/
enum class hfbField { hfb, weight, flags, gain };


/**
 * @struct hfbMetadata
 * @brief Metadata for the hyperfine beam style buffers
 *
 * @author James Willis
 **/
struct hfbMetadata {

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
    /// ID of the dataset, main hfb dataset = 0
    uint64_t dataset_id;

    /// Number of beams for data in buffer
    uint32_t num_beams;
    /// Number of sub-frequencies for data in buffer
    uint32_t num_subfreq;
};


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
class hfbFrameView {

public:
    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created buffers.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    hfbFrameView(Buffer* buf, int frame_id);

    /**
     * @brief Create view and set structure metadata.
     *
     * This should be used for creating entirely new frames. This overload takes
     * the number of sub-frequencies as a parameter.
     *
     * @param buf              The buffer the frame is in.
     * @param frame_id         The id of the frame to read.
     * @param num_beams     Number of beams in the data.
     * @param num_subfreq         Number of sub-frequencies in the data.
     *
     * @warning The metadata object must already have been allocated.
     **/
    hfbFrameView(Buffer* buf, int frame_id, uint32_t num_beams, uint32_t num_subfreq);

    /**
     * @brief Copy frame to a new buffer and create view of copied frame
     *
     * This should be used for copying a frame from one buffer to another.
     *
     * @param buf              The buffer the frame is in.
     * @param frame_id         The id of the frame to read.
     * @param frame_to_copy    An instance of hfbFrameView corresponding to the frame to be copied.
     *
     * @warning The metadata object must already have been allocated.
     **/
    hfbFrameView(Buffer* buf, int frame_id, hfbFrameView frame_to_copy);

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
     * @returns A hfbFrameView of the copied frame.
     *
     **/
    static hfbFrameView copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                   int frame_id_dest);

    /**
     * @brief Get the layout of the buffer from the structural parameters.
     *
     * @param num_beams     Number of beams.
     * @param num_subfreq         Number of sub-frequencies.
     *
     * @returns A mnonvis_bufferap from member name to start and end in bytes. The start
     *          (i.e. 0) and end (i.e. total size) of the buffer is contained in
     *          `_struct`.
     **/
    static struct_layout<hfbField> calculate_buffer_layout(uint32_t num_beams, uint32_t num_subfreq);

    /**
     * @brief Return a summary of the hyperfine beam buffer contents.
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
    void copy_metadata(hfbFrameView frame_to_copy);

    /**
     * @brief Copy over the data, skipping specified members.
     *
     * This routine copies member by member and the structural parameters of the
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
    void copy_data(hfbFrameView frame_to_copy, const std::set<hfbField>& skip_members);

    // TODO: CHIME specific
    /**
     * @brief Fill the hfbMetadata from a chimeMetadata struct.
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
    const hfbMetadata* metadata() const {
        return _metadata;
    }

    /**
     * @brief Read only access to the frame data.
     * @returns The data.
     **/
    const uint8_t* data() const {
        return _frame;
    }

private:
    // References to the buffer and metadata we are viewing
    Buffer* const buffer;
    const int id;
    hfbMetadata* const _metadata;

    // Pointer to frame data. In theory this is redundant as it can be derived
    // from buffer and id, but it's nice for brevity
    uint8_t* const _frame;

    // The calculated layout of the buffer
    struct_layout<hfbField> buffer_layout;

    // NOTE: these need to be defined in a final public block to ensure that they
    // are initialised after the above members.
public:
    /// The number of beams in the data (read only).
    const uint32_t& num_beams;
    /// The number of sub-frequencies in the data (read only).
    const uint32_t& num_subfreq;

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

    /// View of the hyperfine beam data.
    const gsl::span<cfloat> hfb;
    /// View of the weight data.
    const gsl::span<float> weight;
    /// View of the input flags
    const gsl::span<float> flags;
    /// View of the applied gains
    const gsl::span<cfloat> gain;
};
#endif
