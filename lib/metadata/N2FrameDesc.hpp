#ifndef N2_FRAME_DESC_HPP
#define N2_FRAME_DESC_HPP

#include "Config.hpp" // for kotekan::Config
#include "FrameDesc.hpp"
#include "N2Layout.hpp"
#include "N2Util.hpp" // for N2::prod_ctype, N2::cfloat

#include <algorithm>
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
 * @brief Describes the byte range of a field within an N2 frame.
 **/
struct n2field_member_t {
    size_t begin;
    size_t end;
    size_t size() const {
        return end - begin;
    }
};

/**
 * @brief The complete layout of an N2 frame, including field positions and total size.
 **/
struct n2frame_layout_t {
    std::map<N2Field, n2field_member_t> fields;
    size_t total_size() const {
        size_t max_end = 0;
        for (const auto& [_, member] : fields)
            max_end = std::max(max_end, member.end);
        return max_end;
    }
};

/**
 * @brief Eigenvalue and Eigenvector calculation method
 *
 * Use this enum to refer to the method used to compute Eigenvalues and Eigenvectors.
 **/
enum class N2EigenMethod : int32_t { none, cheevr, iterative };

class N2FrameDesc : public FrameDesc {
public:
    /**
     * @brief Construct an N2FrameDesc.
     *
     * @param num_elements  Number of elements (dishes x polarizations)
     * @param num_ev        Number of eigenvalues/eigenvectors
     * @param num_products  Number of products in the visibility matrix
     * @param n2_layout     The layout of the visibility matrix
     * @param product_list  Optional explicit list of products (required for subset layouts)
     *
     * @throws std::runtime_error If product_list is required but not provided, or if
     *         product_list.size() != num_products for layouts that require it.
     */
    N2FrameDesc(uint32_t num_elements, uint32_t num_ev, uint32_t num_products, N2Layout n2_layout,
                std::vector<N2::prod_ctype> product_list = {});
    virtual ~N2FrameDesc() = default;

    // FrameDesc overrides
    Symbol get_quantity_name() const override;
    void output_framedesc(std::ostream& os) const override;
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

    /**
     * @brief Get the explicit product list for this frame descriptor.
     *
     * For FullUpperTri and Autocorrelations layouts, this returns an empty vector
     * (products are computed on-the-fly by fill_prod_maps).
     * For subset layouts (InputANDMasked, InputORMasked, GeneralSubset), this returns
     * the explicit list of products stored in this descriptor.
     *
     * @return Reference to the product list (may be empty for computed layouts)
     */
    const std::vector<N2::prod_ctype>& get_product_list() const {
        return product_list;
    }

    /**
     * @brief Check if this layout requires a product list in the constructor (or implicitly,
     * a product list).
     *
     * Returns true for all subset layouts (InputORMasked, InputANDMasked, GeneralSubset,
     * RedundantBaselineAvg). Note that for InputORMasked/InputANDMasked, the product list
     * is typically generated from an input_list via generate_product_list().
     *
     * @param layout The N2Layout to check
     * @return true if the layout requires an explicit product list
     */
    static bool layout_requires_product_list(N2Layout layout);

    /**
     * @brief Get the name of the config parameter required for a given layout.
     *
     * Layout config requirements:
     * - FullUpperTri, Autocorrelations: "none" (computed from num_elements)
     * - InputORMasked, InputANDMasked: "input_list" (list of element indices)
     * - GeneralSubset, RedundantBaselineAvg: "product_list" (explicit list of products)
     *
     * @param layout The N2Layout to check
     * @return Config parameter name: "none", "input_list", or "product_list"
     */
    static const char* get_required_config_param(N2Layout layout);

    /**
     * @brief Generate a product list from an input list for InputORMasked or InputANDMasked
     * layouts.
     *
     * @param num_elements  Number of elements (dishes x polarizations)
     * @param input_list    List of input indices to filter by
     * @param layout        Must be InputORMasked or InputANDMasked
     *
     * @return Vector of products matching the filter criteria:
     *         - InputORMasked: products where input_a OR input_b is in input_list
     *         - InputANDMasked: products where input_a AND input_b are in input_list
     *
     * @throws std::runtime_error if layout is not InputORMasked or InputANDMasked
     */
    static std::vector<N2::prod_ctype>
    generate_product_list(uint32_t num_elements, const std::vector<uint16_t>& input_list,
                          N2Layout layout);

    /**
     * @brief Create an N2FrameDesc from configuration.
     *
     * Reads num_elements, num_ev, and n2_layout from config, then based on the layout:
     * - FullUpperTri/Autocorrelations: no additional config needed
     * - InputORMasked/InputANDMasked: reads input_list, generates product list
     * - GeneralSubset/RedundantBaselineAvg: reads product_list directly
     *
     * @param config    The configuration object
     * @param location  The config path for this buffer
     *
     * @return Shared pointer to the created N2FrameDesc
     */
    static std::shared_ptr<N2FrameDesc> from_config(kotekan::Config& config,
                                                    const std::string& location);

    // Static helpers (previously in frame view)
    /**
     * @brief Get the number of products in the visibility matrix for the given number of elements
     * and layout.
     *
     * @param   num_elements_in  number of elements (dishes x polarizations) in the pipeline
     * @param   n2_layout_in     the layout of the visibility matrix in the N2FrameDesc
     * @param   product_list     optional explicit product list; if provided and non-empty, its
     *                           size is returned directly (useful for subset layouts)
     *
     * @throws std::runtime_error    If n2_layout_in is unknown or requires a product list but
     *                               none was provided.
     */
    static size_t get_num_prod(uint32_t num_elements_in, N2Layout n2_layout_in,
                               const std::vector<N2::prod_ctype>& product_list = {});

    /**
     * @brief Calculate the size of the frame.
     */
    static size_t calculate_frame_size(uint32_t num_elements_in, uint32_t num_ev_in,
                                       size_t num_prod_in);
    static size_t calculate_frame_size(kotekan::Config& config, const std::string& unique_name);

    /**
     * @brief The layout of data/fields within the frame.
     *
     * @return The frame layout including field positions and total size.
     **/
    static n2frame_layout_t get_frame_layout(uint32_t num_elements_in, uint32_t num_ev_in,
                                             size_t num_prod_in);

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

private:
    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * FullUpperTri layout.
     *
     * See N2FrameDesc::fill_prod_maps() for full details.
     *
     * @param   prods           Vector to fill.
     */
    void fill_prod_maps_FullUpperTri(std::vector<N2::prod_ctype>& prods) const;

    /**
     * @brief Fill the product maps vector for each product in the visibility matrix in the
     * Autocorrelations-only (diagonal) layout.
     *
     * See N2FrameDesc::fill_prod_maps() for full details.
     *
     * @param   prods           Vector to fill.
     */
    void fill_prod_maps_Autocorrelations(std::vector<N2::prod_ctype>& prods) const;

    const uint32_t num_elements;
    const uint32_t num_ev;
    const uint32_t num_products;
    const N2Layout n2_layout;

    /// Explicit product list for subset layouts (empty for FullUpperTri/Autocorrelations)
    const std::vector<N2::prod_ctype> product_list;
};

} // namespace kotekan

#endif
