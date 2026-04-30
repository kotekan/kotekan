/*****************************************
@file
@brief Code for using the VisFrameView formatted data.
- VisMetadata
- VisFrameView
*****************************************/
#ifndef N2BUFFER_HPP
#define N2BUFFER_HPP

#include "CHORDTelescope.hpp" // for CHORDTelescope
#include "Config.hpp"         // for Config
#include "FrameView.hpp"      // for FrameView
#include "N2FrameDesc.hpp"    // for N2FrameDesc
#include "N2Metadata.hpp"     // for N2Metadata
#include "chordMetadata.hpp"  // for MAX_NUM_RFI_THRESHOLDS
#include "N2Util.hpp"         // for cfloat, get_num_prod
#include "buffer.hpp"         // for Buffer

#include "gsl-lite.hpp"     // for span

#include <algorithm> // for max
#include <exception> // for exception
#include <map>       // for allocator, map
#include <memory>    // for shared_ptr
#include <set>       // for set
#include <stddef.h>  // for size_t
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t, uint64_t
#include <string>    // for basic_string, string
#include <utility>   // for pair, make_pair
#include <vector>    // for vector

using kotekan::N2EigenMethod;
using kotekan::N2Field;

/**
 * @class N2FrameView
 * @brief Provide a structured view of a visn N2k-pipeline visibility buffer.
 *
 * This class inherits from the FrameView base class and sets up a view on a visibility buffer with
 * the ability to interact with the data and metadata.
 *
 **/
class N2FrameView : public FrameView {

public:
    const std::shared_ptr<N2Metadata> _metadata;
    const std::shared_ptr<const kotekan::N2FrameDesc> _desc;

    /// Layout of the visibility matrix
    const N2Layout n2_layout;
    /// Number of elements for data in buffer
    const uint32_t num_elements;
    /// Number of products for data in buffer
    const uint32_t num_prod;
    /// Number of eigenvectors and values calculated
    const uint32_t num_ev;

    kotekan::n2frame_layout_t frame_layout;

    /// ID of the frequency associated with this frame
    const uint32_t& freq_id;
    /// Physical frequency associated with this frame
    const double& freq_MHz;

    /// Absolute time index of frame
    uint64_t& abs_time_idx;

    /// Earth Orientation Paramters
    struct EOP& time_center_eop;
    struct EOP& bin_eop;
    double& bin_start_ERA_deg;
    double& bin_end_ERA_deg;
    double& bin_start_ERAL;
    double& bin_end_ERAL;

    /// The sequence number of the first FPGA frame integrated into this
    /// visibility frame (time<0> in VisFrameView)
    uint64_t& fpga_start_tick;
    /// The time of the start of the integration frame in nanosec (time<1>)
    uint64_t& frame_start_time_ns;
    /// The nominal frame length in FPGA ticks (fpga_seq_length in VisFrameView)
    uint64_t& frame_length_fpga_ticks;
    /// The actual amount of data accumulated in FPGA ticks (fpga_seq_total)
    uint64_t& n_valid_fpga_ticks;
    /// The number of lost samples due to RFI (rfi_total). Might contain Packet Loss as well.
    uint64_t& n_rfi_fpga_ticks;
    /// The number of lost samples due to RFI only
    uint64_t& n_rfi_only_fpga_ticks;
    /// The number of lost samples due to Packet Loss (PL)
    uint64_t& n_pl_fpga_ticks;

    /// Whether second stage RFI excision was applied to this frame
    bool& rfi_frame_excision_enabled;
    /// The number of active RFI excision thresholds.
    int32_t& rfi_frame_excision_num;
    /// The SK thresholds (in sigma) for RFI excision
    std::array<float, MAX_NUM_RFI_THRESHOLDS>& rfi_frame_excision_threshold;
    /// The fraction of samples above threshold that trigger RFI excision.
    std::array<float, MAX_NUM_RFI_THRESHOLDS>& rfi_frame_excision_fraction;

    /// CHIME dataset id tracking updateable config item changes
    dset_id_t& dataset_id;

    /// View of the visibility data.
    const gsl_lite::span<N2::cfloat> vis;
    /// View of the weight data.
    const gsl_lite::span<float> weight;
    /// View of the input flags
    const gsl_lite::span<float> flags;
    /// View of the eigenvalues.
    const gsl_lite::span<float> eval;
    /// View of the eigenvectors (packed as ev,feed).
    const gsl_lite::span<N2::cfloat> evec;
    /// Method used to compute Eigenvalues and Eigenvectors
    N2EigenMethod& emethod;
    /// The RMS of residual visibilities
    float& erms;
    /// Radiometer chi2 statistic for each polarization pair
    const gsl_lite::span<float> radiometer_chi2;
    /// View of the applied gains
    const gsl_lite::span<N2::cfloat> gain;
    /// View of per-element masks (uint8_t per element)
    const gsl_lite::span<uint8_t> mask;

    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created frames.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    N2FrameView(Buffer* buf, int frame_id);

    size_t data_size() const override;
    void zero_frame() override;

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
     * @returns An N2FrameView of the copied frame.
     *
     **/
    static N2FrameView copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                  int frame_id_dest);

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
     * @param  frame_to_copy_from  Frame to copy metadata from.
     * @param  skip_members        Specify a set of data members to *not* copy.
     *
     **/
    void copy_data(N2FrameView frame_to_copy_from, const std::set<N2Field>& skip_members);
};

#endif
