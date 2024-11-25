/*****************************************
@file
@brief Code for using the VisFrameView formatted data.
- VisMetadata
- VisFrameView
*****************************************/
#ifndef N2BUFFER_HPP
#define N2BUFFER_HPP

#include "Config.hpp"       // for Config
#include "FrameView.hpp"    // for FrameView
#include "Telescope.hpp"    // for freq_id_t
#include "buffer.hpp"       // for Buffer
#include "N2Util.hpp"       // for cfloat
#include "N2Metadata.hpp"   // for N2Metadata

#include "gsl-lite.hpp"     // for span

#include <set>              // for set
#include <map>              // for map
#include <utility>          // for pair

/**
 * @brief The fields within the N2FrameView.
 *
 * Use this enum to refer to the fields.
 **/
enum class N2Field { vis, weight, flags, eval, evec, emethod, erms, gain };

/**
 * @brief Eigenvalue and Eigenvector calculation method
 *
 * Use this enum to refer to the method used to compute Eigenvalues and Eigenvectors.
 **/
enum class N2EigenMethod { cheevr, iterative };

/**
 * @class N2FrameView
 * @brief Provide a structured view of a visn N2k-pipeline visibility buffer.
 *
 * This class inherits from the FrameView base class and sets up a view on a visibility buffer with
 * the ability to interact with the data and metadata.
 *
 **/
class N2FrameView :
    public FrameView {

public:
    const std::shared_ptr<N2Metadata> _metadata;
    /// Number of elements for data in buffer
    const uint32_t& num_elements;
    /// Number of products for data in buffer
    const uint32_t& num_prod;
    /// Number of eigenvectors and values calculated
    const uint32_t& num_ev;
    /// Total number of frequencies in the pipeline
    const uint32_t& nfreq;

    std::map<N2Field, std::pair<size_t, size_t>> frame_layout;


    /**** BEGIN visFrameView stuff that we may want to deprecate ****/
    
    /// The sequence number of the first FPGA frame integrated into this
    /// visibility frame (time<0> in VisFrameView)
    uint64_t& fpga_start_tick;
    /// The time of the start of the integration frame in nanosec (time<1>)
    uint64_t& frame_start_time_ns;
    /// The nominal frame length in FPGA ticks (fpga_seq_length in VisFrameView)
    uint64_t& frame_length_fpga_ticks;
    /// The actual amount of data accumulated in FPGA ticks (fpga_seq_total)
    uint64_t& n_valid_fpga_ticks_in_frame;
    /// The number of lost samples due to RFI (rfi_total)
    uint64_t& n_rfi_fpga_ticks;
    /**** END ****/

    /// ID of the frequency bin
    const uint32_t& freq_id;

    /// View of the visibility data.
    const gsl::span<N2::cfloat> vis;
    /// View of the weight data.
    const gsl::span<float> weight;
    /// View of the input flags
    const gsl::span<float> flags;
    /// View of the eigenvalues.
    const gsl::span<float> eval;
    /// View of the eigenvectors (packed as ev,feed).
    const gsl::span<N2::cfloat> evec;
    /// Method used to compute Eigenvalues and Eigenvectors
    N2EigenMethod & emethod;
    /// The RMS of residual visibilities
    float& erms;
    /// View of the applied gains
    const gsl::span<N2::cfloat> gain;

    /**
     * @brief The sizes of the fields in the N2FrameView.
     */
    static std::vector<std::pair<N2Field, size_t>> get_field_sizes(uint32_t num_elements_in, uint32_t num_ev_in) {

        size_t num_prod = N2::get_num_prod(num_elements_in);

        std::vector<std::pair<N2Field, size_t>> field_sizes;
        field_sizes.push_back( { N2Field::vis,      sizeof(N2::cfloat) * num_prod } );
        field_sizes.push_back( { N2Field::weight,   sizeof(float) * num_prod} );
        field_sizes.push_back( { N2Field::flags,    sizeof(float) * num_elements_in} );
        field_sizes.push_back( { N2Field::eval,     sizeof(float) * num_ev_in} );
        field_sizes.push_back( { N2Field::evec,     sizeof(N2::cfloat) * num_ev_in * num_elements_in} );
        field_sizes.push_back( { N2Field::emethod,  sizeof(N2EigenMethod) * 1} );
        field_sizes.push_back( { N2Field::erms,     sizeof(float) * 1} );
        field_sizes.push_back( { N2Field::gain,     sizeof(N2::cfloat) * num_elements_in} );

        return field_sizes;
    }

    /**
     * @brief The layout of the fields within the N2FrameView.
     * 
     * @return A map of the field to the { start, end } of the field in the frame.
     **/
    static std::map<N2Field, std::pair<size_t, size_t>> get_frame_layout(uint32_t num_elements_in, uint32_t num_ev_in)
    {
        std::map<N2Field, std::pair<size_t, size_t>> frame_layout;
        std::vector<std::pair<N2Field, size_t>> field_sizes = get_field_sizes(num_elements_in, num_ev_in);
        
        // build the layout
        size_t offset = 0;
        for (const std::pair<N2Field, size_t> & field : field_sizes) {
            frame_layout[field.first] = std::make_pair(offset, offset + field.second);
            offset += field.second;
        }

        return frame_layout;
    }

    /**
     * @brief Calculate the size of the frame.
     */
    static size_t calculate_frame_size(uint32_t num_elements_in, uint32_t num_ev_in) {
        size_t frame_size = get_frame_layout(num_elements_in, num_ev_in)[N2Field::gain].second;
        return frame_size;
    }

    static size_t calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {

        const int num_elements_in = config.get<int>(unique_name, "num_elements");
        const int num_ev_in = config.get<int>(unique_name, "num_ev");

        return calculate_frame_size(num_elements_in, num_ev_in);
    }


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
    static N2FrameView copy_frame(Buffer* buf_src, int frame_id_src,
                                  Buffer* buf_dest, int frame_id_dest);

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
