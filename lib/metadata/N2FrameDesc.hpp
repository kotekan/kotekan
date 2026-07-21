#ifndef N2_FRAME_DESC_HPP
#define N2_FRAME_DESC_HPP

#include <stddef.h>       // for size_t
#include <stdint.h>       // for uint32_t, int32_t, uint16_t
#include <algorithm>      // for max
#include <map>            // for map, operator!=, _Rb_tree_const_iterator
#include <utility>        // for pair
#include <vector>         // for vector
#include <iosfwd>         // for ostream
#include <string>         // for string

#include "Config.hpp"     // for Config
#include "FrameDesc.hpp"  // for FrameDesc
#include "N2Layout.hpp"   // for N2Layout
#include "N2Util.hpp"     // for prod_ctype
#include "Symbol.hpp"     // for Symbol

namespace kotekan {

/**
 * @brief The fields within the N2 frame.
 *
 * Use this enum to refer to the fields.
 **/
enum class N2Field { vis, weight, flags, eval, evec, emethod, erms, radiometer_chi2, gain, mask };

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
enum class N2EigenMethod : int32_t { none, cheevr, iterative, failed_iterative };

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
     * @note Validation failures (product_list missing, sized inconsistently with
     *       num_products, or referencing inputs outside num_elements) are fatal
     *       and shut kotekan down (FATAL_ERROR_NON_OO).
     */
    N2FrameDesc(uint32_t num_elements, uint32_t num_ev, uint32_t num_products, N2Layout n2_layout,
                std::vector<N2::prod_ctype> product_list = {});
    virtual ~N2FrameDesc() = default;

    // FrameDesc overrides
    Symbol get_quantity_name() const override;
    void output_framedesc(std::ostream& os) const override;
    bool operator==(const FrameDesc& other) const override;
    size_t get_byte_size() const override;

    // FrameDesc JSON-serialization override (see FrameDesc::to_json)
    nlohmann::json to_json() const override;
    /// Reconstruct an N2FrameDesc from JSON written by to_json(). Validation
    /// failures are fatal (see the constructor).
    static std::shared_ptr<const FrameDesc> from_json(const nlohmann::json& j);

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
     * @brief Get the product list for this frame descriptor.
     *
     * The product list is populated on construction for all layouts.
     *
     * @return Reference to the product list.
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
    static const char* note_additional_required_config_param(N2Layout layout);

    /**
     * @brief Generate a product list for the given layout.
     *
     * For FullUpperTri: generates all upper-triangular products
     * For Autocorrelations: generates diagonal-only products
     * For InputORMasked: generates products where input_a OR input_b is in input_list
     * For InputANDMasked: generates products where input_a AND input_b are in input_list
     *
     * @param num_elements  Number of elements (dishes x polarizations)
     * @param layout        The N2Layout to generate products for
     * @param input_list    List of input indices (required for InputORMasked/InputANDMasked)
     *
     * @return Vector of products for the given layout
     *
     * @throws std::runtime_error if layout is not supported or if input_list is required
     *         but not provided
     */
    static std::vector<N2::prod_ctype>
    generate_product_list(uint32_t num_elements, N2Layout layout,
                          const std::vector<uint16_t>& input_list = {});

    /**
     * @brief Construct an N2FrameDesc from configuration.
     *
     * Reads num_elements, num_ev, and n2_layout from config, then based on the layout:
     * - FullUpperTri/Autocorrelations: no additional config needed
     * - InputORMasked/InputANDMasked: reads input_list, generates product list
     * - GeneralSubset/RedundantBaselineAvg: reads product_list directly
     *
     * @param config    The configuration object
     * @param location  The config path for this buffer
     */
    N2FrameDesc(kotekan::Config& config, const std::string& location);

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

    /**
     * @brief The layout of data/fields within the frame.
     *
     * @return The frame layout including field positions and total size.
     **/
    static n2frame_layout_t get_frame_layout(uint32_t num_elements_in, uint32_t num_ev_in,
                                             size_t num_prod_in);

private:
    /// Helper for the config constructor: parses config and returns a fully constructed
    /// N2FrameDesc.
    static N2FrameDesc _from_config_impl(kotekan::Config& config, const std::string& location);

    const uint32_t num_elements;
    const uint32_t num_ev;
    const uint32_t num_products;
    const N2Layout n2_layout;

    /// Product list for this frame descriptor (populated for all layouts)
    const std::vector<N2::prod_ctype> product_list;
};

} // namespace kotekan

#endif
