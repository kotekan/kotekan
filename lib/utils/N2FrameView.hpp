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
#include "N2Util.hpp"         // for cfloat, get_num_prod
#include "buffer.hpp"         // for Buffer

#include "gsl-lite.hpp" // for span

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
    double& bin_start_LAST;
    double& bin_end_LAST;

    /// The sequence number of the first FPGA frame integrated into this
    /// visibility frame (time<0> in VisFrameView)
    uint64_t& fpga_start_tick;
    /// The time of the start of the integration frame in nanosec (time<1>)
    uint64_t& frame_start_time_ns;
    /// The nominal frame length in FPGA ticks (fpga_seq_length in VisFrameView)
    uint64_t& frame_length_fpga_ticks;
    /// The actual amount of data accumulated in FPGA ticks (fpga_seq_total)
    uint64_t& n_valid_fpga_ticks;
    /// The number of lost samples due to RFI (rfi_total)
    uint64_t& n_rfi_fpga_ticks;

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
    /// View of the applied gains
    const gsl_lite::span<N2::cfloat> gain;

    /**
     * @brief The sizes of the fields in the N2FrameView.
     */
    static std::vector<std::pair<N2Field, size_t>>
    get_field_sizes(uint32_t num_elements_in, uint32_t num_ev_in, size_t num_prod_in) {

        std::vector<std::pair<N2Field, size_t>> field_sizes;
        field_sizes.push_back({N2Field::vis, sizeof(N2::cfloat) * num_prod_in});
        field_sizes.push_back({N2Field::weight, sizeof(float) * num_prod_in});
        field_sizes.push_back({N2Field::flags, sizeof(float) * num_elements_in});
        field_sizes.push_back({N2Field::eval, sizeof(float) * num_ev_in});
        field_sizes.push_back({N2Field::evec, sizeof(N2::cfloat) * num_ev_in * num_elements_in});
        field_sizes.push_back({N2Field::emethod, sizeof(N2EigenMethod) * 1});
        field_sizes.push_back({N2Field::erms, sizeof(float) * 1});
        field_sizes.push_back({N2Field::gain, sizeof(N2::cfloat) * num_elements_in});

        return field_sizes;
    }

    /**
     * @brief The layout of the fields within the N2FrameView.
     *
     * @return A map of the field to the { start, end } of the field in the frame.
     **/
    static std::map<N2Field, std::pair<size_t, size_t>>
    get_frame_layout(uint32_t num_elements_in, uint32_t num_ev_in, size_t num_prod_in) {
        std::map<N2Field, std::pair<size_t, size_t>> frame_layout;
        std::vector<std::pair<N2Field, size_t>> field_sizes =
            get_field_sizes(num_elements_in, num_ev_in, num_prod_in);

        // build the layout
        size_t offset = 0;
        for (const std::pair<N2Field, size_t>& field : field_sizes) {
            frame_layout[field.first] = std::make_pair(offset, offset + field.second);
            offset += size_t(field.second);
        }

        return frame_layout;
    }

    /**
     * @brief Calculate the size of the frame.
     */
    static size_t calculate_frame_size(uint32_t num_elements_in, uint32_t num_ev_in,
                                       uint32_t num_prod_in) {
        size_t frame_size =
            get_frame_layout(num_elements_in, num_ev_in, num_prod_in)[N2Field::gain].second;
        return frame_size;
    }

    static size_t calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {

        const size_t num_elements_in = config.get<size_t>(unique_name, "num_elements");
        const size_t num_ev_in = config.get<size_t>(unique_name, "num_ev");

        const N2Layout vis_layout = config.get<N2Layout>(unique_name, "vis_layout");

        size_t num_prod_in;
        if (vis_layout == N2Layout::GeneralSubset) {
            // Require a list of (i,j) elements to be specified
            num_prod_in = config.get<std::vector<N2::prod_ctype>>(unique_name, "products").size();
        } else if (vis_layout == N2Layout::InputANDMasked
                   || vis_layout == N2Layout::InputORMasked) {
            // Require a list of inputs to be specified
            std::vector<size_t> input_list =
                config.get<std::vector<size_t>>(unique_name, "input_list");
            num_prod_in = get_num_prod(num_elements_in, vis_layout, input_list);
        } else {
            num_prod_in = get_num_prod(num_elements_in, vis_layout);
        }

        return calculate_frame_size(uint32_t(num_elements_in), uint32_t(num_ev_in),
                                    uint32_t(num_prod_in));
    }

    /**
     * @brief Get the number of products in the visibility matrix for the given number of elements
     * and layout.
     *
     * @param   num_elemens_in  number of elements (dishes x polarizations) in the pipeline
     * @param   vis_layout_in       the layout of the visibility matrix in the N2FrameView
     *
     * @throws std::runtime_error    If vis_layout_in is unknown.
     */
    static size_t get_num_prod(size_t num_elements_in, N2Layout vis_layout_in) {
        size_t num_prod_in;

        switch (vis_layout_in) {
            case N2Layout::FullUpperTri:
                num_prod_in = (num_elements_in * (num_elements_in + 1)) / 2;
                break;
            case N2Layout::Autocorrelations:
                num_prod_in = num_elements_in;
                break;
            default:
                throw std::runtime_error(
                    fmt::format("N2FrameView::get_num_prod given unknown N2Layout: {:s}",
                                N2Layout_to_string(vis_layout_in)));
        }

        return num_prod_in;
    }

    /**
     * @brief Get the number of products in the visibility matrix for the given number of elements,
     * layout, and input list.
     *
     * @param   num_elemens_in  number of elements (dishes x polarizations) in the pipeline
     * @param   vis_layout_in       the layout of the visibility matrix in the N2FrameView
     * @param   input_list      list of inputs to consider for masked layouts
     *
     * @throws std::runtime_error    If vis_layout_in is unknown.
     */
    static size_t get_num_prod(size_t num_elements_in, N2Layout vis_layout,
                               const std::vector<size_t>& input_list) {

        size_t num_prod_in;

        switch (vis_layout) {
            case N2Layout::InputANDMasked:
                // "AND" masking: only include products where *both* inputs are in the list.
                // For this, we will effectively have a smaller upper triangular matrix with each
                // dimension equal to the length of input_list.
                num_prod_in = (input_list.size() * (input_list.size() + 1)) / 2;
                break;
            case N2Layout::InputORMasked:
                // "OR" masking: include products where *either* input is in the list.
                // For this, we need to count the number of products that meet this condition.
                num_prod_in = input_list.size() * (2 * num_elements_in - input_list.size() + 1) / 2;
                break;
            default:
                throw std::runtime_error(
                    fmt::format("N2FrameView::get_num_prod given unknown N2Layout: {:s}",
                                N2Layout_to_string(vis_layout)));
        }

        return num_prod_in;
    }

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * FullUpperTri layout.
     *
     * See N2FrameView::fill_prod_maps() for full details.
     *
     * @param   prods           Vector to fill.
     * @param   num_elements_in Number of elements (dishes x polarizations) in the pipeline
     */
    static void fill_prod_maps_FullUpperTri(std::vector<N2::prod_ctype>& prods,
                                            size_t num_elements_in) {

        size_t num_prod_in = (num_elements_in * (num_elements_in + 1)) / 2;

        prods.resize(num_prod_in);

        // Loop over all rows
        size_t p = 0;
        for (size_t i = 0; i < num_elements_in; i++) {
            // Upper Triangular!  Loop over upper columns only;
            for (size_t j = i; j < num_elements_in; j++) {
                prods[p] = {.input_a = static_cast<uint16_t>(i),
                            .input_b = static_cast<uint16_t>(j)};
                p++;
            }
        }
    }

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * Autocorrelations-only (diagonal) layout.
     *
     * See N2FrameView::fill_prod_maps() for full details.
     *
     * @param   prods           Vector to fill.
     * @param   num_elements_in Number of elements (dishes x polarizations) in the pipeline
     */
    static void fill_prod_maps_Autocorrelations(std::vector<N2::prod_ctype>& prods,
                                                size_t num_elements_in) {
        size_t num_prod_in = num_elements_in;

        prods.resize(num_prod_in);

        // Loop over all elements
        for (size_t i = 0; i < num_elements_in; i++) {
            prods[i] = {.input_a = static_cast<uint16_t>(i), .input_b = static_cast<uint16_t>(i)};
        }
    }

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * InputANDMasked layout. Requires a list of inputs to be provided.
     */
    static void fill_prod_maps_InputANDMasked(std::vector<N2::prod_ctype>& prods,
                                              std::vector<size_t> input_list,
                                              size_t num_elements_in) {

        size_t num_prod_in = get_num_prod(num_elements_in, N2Layout::InputANDMasked, input_list);

        prods.resize(num_prod_in);

        // Loop over all rows
        size_t p = 0;
        for (size_t i_idx = 0; i_idx < input_list.size(); i_idx++) {
            size_t i = input_list[i_idx];
            // Upper Triangular!  Loop over upper columns only;
            for (size_t j_idx = i_idx; j_idx < input_list.size(); j_idx++) {
                size_t j = input_list[j_idx];
                prods[p] = {.input_a = static_cast<uint16_t>(i),
                            .input_b = static_cast<uint16_t>(j)};
                p++;
            }
        }
    }

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * InputORMasked layout. Requires a list of inputs to be provided.
     */
    static void fill_prod_maps_InputORMasked(std::vector<N2::prod_ctype>& prods,
                                             std::vector<size_t> input_list,
                                             size_t num_elements_in) {

        size_t num_prod_in = get_num_prod(num_elements_in, N2Layout::InputORMasked, input_list);

        prods.resize(num_prod_in);

        // Loop over all rows
        size_t p = 0;
        for (size_t i = 0; i < num_elements_in; i++) {
            for (size_t j = i; j < num_elements_in; j++) {
                // Check if either input is in the input list
                bool in_list = false;
                for (auto ipt : input_list) {
                    if ((i == ipt) || (j == ipt)) {
                        in_list = true;
                        break;
                    }
                }
                if (in_list) {
                    prods[p] = {.input_a = static_cast<uint16_t>(i),
                                .input_b = static_cast<uint16_t>(j)};
                    p++;
                }
            }
        }
    }

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * GeneralSubset layout. Requires a list of products to be specified in the config.
     *
     * See N2FrameView::fill_prod_maps() for full details.
     *
     * @param   prods           Vector to fill.
     * @param   config         Kotekan configuration object.
     * @param   unique_name    Unique name of this stage in the Kotekan config.
     */
    static void fill_prod_maps_GeneralSubset(std::vector<N2::prod_ctype>& prods,
                                             kotekan::Config& config,
                                             const std::string& unique_name) {
        auto subset_prods = config.get<std::vector<N2::prod_ctype>>(unique_name, "products");
        prods = subset_prods;
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

    void fill_prod_maps(std::vector<N2::prod_ctype>& prods) const;
};

#endif
