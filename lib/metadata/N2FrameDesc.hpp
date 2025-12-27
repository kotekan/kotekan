#ifndef N2_FRAME_DESC_HPP
#define N2_FRAME_DESC_HPP

#include "Config.hpp" // for kotekan::Config
#include "FrameDesc.hpp"
#include "N2Layout.hpp"
#include "N2Util.hpp" // for N2::prod_ctype, N2::cfloat

#include <map>
#include <utility>
#include <vector>

namespace kotekan {

/**
 * @brief The fields within the N2 frame.
 *
 * Use this enum to refer to the fields.
 **/
enum class N2Field { vis, weight, flags, eval, evec, emethod, erms, gain };

/**
 * @brief Eigenvalue and Eigenvector calculation method
 *
 * Use this enum to refer to the method used to compute Eigenvalues and Eigenvectors.
 **/
enum class N2EigenMethod : int32_t { none, cheevr, iterative };

class N2FrameDesc : public FrameDesc {
public:
    N2FrameDesc(uint32_t num_elements, uint32_t num_ev, uint32_t num_products, N2Layout n2_layout);
    virtual ~N2FrameDesc() = default;

    // FrameDesc overrides
    Symbol get_quantity_name() const override;
    void output_metadata(std::ostream& os) const override;
    bool operator==(const FrameDesc& other) const override;
    size_t get_byte_size() const override;

    // Accessors
    uint32_t get_num_elements() const {
        return num_elements;
    }
    uint32_t get_num_ev() const {
        return num_ev;
    }
    uint32_t get_num_products() const {
        return num_products;
    }
    N2Layout get_n2_layout() const {
        return n2_layout;
    }

    // Static helpers (previously in frame view)
    /**
     * @brief Get the number of products in the visibility matrix for the given number of elements
     * and layout.
     *
     * @param   num_elemens_in  number of elements (dishes x polarizations) in the pipeline
     * @param   vis_layout_in       the layout of the visibility matrix in the N2FrameDesc
     *
     * @throws std::runtime_error    If vis_layout_in is unknown.
     */
    static size_t get_num_prod(uint32_t num_elements_in, N2Layout n2_layout_in);

    /**
     * @brief Calculate the size of the frame.
     */
    static size_t calculate_frame_size(uint32_t num_elements_in, uint32_t num_ev_in,
                                       size_t num_prod_in);
    static size_t calculate_frame_size(kotekan::Config& config, const std::string& unique_name);

    /**
     * @brief The layout of data/fields within the frame.
     *
     * @return A map of the field to the { start, end } of the field in the frame.
     **/
    static std::map<N2Field, std::pair<size_t, size_t>>
    get_frame_layout(uint32_t num_elements_in, uint32_t num_ev_in, size_t num_prod_in);

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix.
     *
     * Every product in the frame view is a visibility matrix V_{ab} that was formed from two input
     * elements: a (first, the full vis matrix row index) and b (second, the full vis matrix column
     * index), where 0 <= a, b < num_elements. This function fills the given vector prods with
     * num_prod entries, that is, one entry for each object in vis or weights. Each is a prod_ctype
     * which has the 'a' element index for this product in prod.index_a, and the 'b' element index
     * in prod.index_b.
     *
     * @note The given vector prods is reserved to num_prods size, which potentially performs an
     * allocation of size num_prod * sizeof(prod_ctype).
     *
     * @param   prods   Vector of prod_ctype to fill.
     *
     * @throws  std::runtime_error  If this N2FrameView has an unknown layout.
     */
    void fill_prod_maps(std::vector<N2::prod_ctype>& prods) const;


    // Static helpers for fill_prod_maps

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * FullUpperTri layout.
     *
     * See N2FrameDesc::fill_prod_maps() for full details.
     *
     * @param   prods           Vector to fill.
     * @param   num_elements_in Number of elements (dishes x polarizations) in the pipeline
     */
    static void fill_prod_maps_FullUpperTri(std::vector<N2::prod_ctype>& prods,
                                            uint32_t num_elements_in);
    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * Autocorrelations-only (diagonal) layout.
     *
     * See N2FrameDesc::fill_prod_maps() for full details.
     *
     * @param   prods           Vector to fill.
     * @param   num_elements_in Number of elements (dishes x polarizations) in the pipeline
     */
    static void fill_prod_maps_Autocorrelations(std::vector<N2::prod_ctype>& prods,
                                                uint32_t num_elements_in);

private:
    uint32_t num_elements;
    uint32_t num_ev;
    uint32_t num_products;
    N2Layout n2_layout;
};

} // namespace kotekan

#endif
