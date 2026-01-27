#include "N2FrameDesc.hpp"

#include "kotekanLogging.hpp"

#include "fmt.hpp"

#include <set>

namespace kotekan {

N2FrameDesc::N2FrameDesc(uint32_t num_elements, uint32_t num_ev, uint32_t num_products,
                         N2Layout n2_layout, std::vector<N2::prod_ctype> product_list) :
    num_elements(num_elements), num_ev(num_ev), num_products(num_products), n2_layout(n2_layout),
    product_list(std::move(product_list)) {

    // Validate product list for layouts that require it
    if (layout_requires_product_list(n2_layout)) {
        if (this->product_list.empty()) {
            throw std::runtime_error(fmt::format(
                "N2FrameDesc: layout {:s} requires {:s} in config (product_list was empty)",
                N2Layout_to_string(n2_layout), get_required_config_param(n2_layout)));
        }
        if (this->product_list.size() != num_products) {
            throw std::runtime_error(fmt::format(
                "N2FrameDesc: product_list size ({:d}) does not match num_products ({:d})",
                this->product_list.size(), num_products));
        }
    }
}

Symbol N2FrameDesc::get_quantity_name() const {
    return Symbol("N2");
}

bool N2FrameDesc::layout_requires_product_list(N2Layout layout) {
    switch (layout) {
        case N2Layout::FullUpperTri:
        case N2Layout::Autocorrelations:
            return false;
        case N2Layout::InputANDMasked:
        case N2Layout::InputORMasked:
        case N2Layout::GeneralSubset:
        case N2Layout::RedundantBaselineAvg:
            return true;
        default:
            return true;
    }
}

const char* N2FrameDesc::get_required_config_param(N2Layout layout) {
    switch (layout) {
        case N2Layout::FullUpperTri:
        case N2Layout::Autocorrelations:
            return "none";
        case N2Layout::InputORMasked:
        case N2Layout::InputANDMasked:
            return "input_list";
        case N2Layout::GeneralSubset:
        case N2Layout::RedundantBaselineAvg:
            return "product_list";
        default:
            return "product_list";
    }
}

std::vector<N2::prod_ctype>
N2FrameDesc::generate_product_list(uint32_t num_elements, const std::vector<uint16_t>& input_list,
                                   N2Layout layout) {
    // Build a set for fast lookup
    std::set<uint16_t> input_set(input_list.begin(), input_list.end());

    std::vector<N2::prod_ctype> products;

    // Iterate over all products in the upper triangle
    for (uint16_t i = 0; i < num_elements; i++) {
        for (uint16_t j = i; j < num_elements; j++) {
            bool i_in = input_set.count(i) > 0;
            bool j_in = input_set.count(j) > 0;

            bool include = false;
            switch (layout) {
                case N2Layout::InputORMasked:
                    // Include if either input is in the list (have_inputs)
                    include = i_in || j_in;
                    break;
                case N2Layout::InputANDMasked:
                    // Include if both inputs are in the list (only_inputs)
                    include = i_in && j_in;
                    break;
                default:
                    throw std::runtime_error(fmt::format(
                        "N2FrameDesc::generate_product_list: layout {:s} requires {:s}, not "
                        "input_list",
                        N2Layout_to_string(layout), get_required_config_param(layout)));
            }

            if (include) {
                products.push_back({.input_a = i, .input_b = j});
            }
        }
    }

    return products;
}

std::shared_ptr<N2FrameDesc> N2FrameDesc::from_config(kotekan::Config& config,
                                                      const std::string& location) {
    const uint32_t num_elements = config.get<uint32_t>(location, "num_elements");
    const uint32_t num_ev = config.get<uint32_t>(location, "num_ev");
    const N2Layout n2_layout = config.get<N2Layout>(location, "n2_layout");

    std::vector<N2::prod_ctype> product_list;

    switch (n2_layout) {
        case N2Layout::FullUpperTri:
        case N2Layout::Autocorrelations:
            // No additional config needed - product count computed from num_elements
            break;

        case N2Layout::InputORMasked:
        case N2Layout::InputANDMasked: {
            // Read input_list and generate product list
            auto input_list = config.get<std::vector<uint16_t>>(location, "input_list");
            product_list = generate_product_list(num_elements, input_list, n2_layout);
            break;
        }

        case N2Layout::GeneralSubset:
        case N2Layout::RedundantBaselineAvg:
            // Read product_list directly
            product_list = config.get<std::vector<N2::prod_ctype>>(location, "product_list");
            break;

        default:
            throw std::runtime_error(fmt::format("N2FrameDesc::from_config: unknown N2Layout {:s}",
                                                 N2Layout_to_string(n2_layout)));
    }

    const uint32_t num_prod = get_num_prod(num_elements, n2_layout, product_list);

    return std::make_shared<N2FrameDesc>(num_elements, num_ev, num_prod, n2_layout,
                                         std::move(product_list));
}

void N2FrameDesc::output_framedesc(std::ostream& os) const {
    os << "N2FrameDesc:\n"
       << "    num_elements: " << num_elements << "\n"
       << "    num_ev:       " << num_ev << "\n"
       << "    num_products: " << num_products << "\n"
       << "    n2_layout:    " << N2Layout_to_string(n2_layout) << "\n";
    if (!product_list.empty()) {
        os << "    product_list: [";
        for (size_t i = 0; i < std::min(product_list.size(), size_t(5)); ++i) {
            if (i > 0)
                os << ", ";
            os << "(" << product_list[i].input_a << "," << product_list[i].input_b << ")";
        }
        if (product_list.size() > 5)
            os << ", ... (" << product_list.size() << " total)";
        os << "]\n";
    }
}

bool N2FrameDesc::operator==(const FrameDesc& other) const {
    const N2FrameDesc* other_ptr = dynamic_cast<const N2FrameDesc*>(&other);
    if (!other_ptr)
        return false;

    if ((num_elements != other_ptr->num_elements) || (num_ev != other_ptr->num_ev)
        || (num_products != other_ptr->num_products) || (n2_layout != other_ptr->n2_layout))
        return false;

    // For layouts with explicit product lists, compare the lists
    if (layout_requires_product_list(n2_layout)) {
        return product_list == other_ptr->product_list;
    }

    return true;
}

size_t N2FrameDesc::get_byte_size() const {
    return calculate_frame_size(num_elements, num_ev, num_products);
}

size_t N2FrameDesc::get_num_prod(uint32_t num_elements_in, N2Layout n2_layout_in,
                                 const std::vector<N2::prod_ctype>& product_list) {
    // If a product list is provided, use its size directly
    if (!product_list.empty()) {
        return product_list.size();
    }

    switch (n2_layout_in) {
        case N2Layout::FullUpperTri:
            return (size_t(num_elements_in) * (num_elements_in + 1)) / 2;
        case N2Layout::Autocorrelations:
            return num_elements_in;
        case N2Layout::InputANDMasked:
        case N2Layout::InputORMasked:
        case N2Layout::GeneralSubset:
        case N2Layout::RedundantBaselineAvg: {
            std::string msg = fmt::format(
                "N2FrameDesc::get_num_prod cannot compute num_prod for layout {:s} "
                "- requires {:s} in config",
                N2Layout_to_string(n2_layout_in), get_required_config_param(n2_layout_in));
            ERROR_NON_OO("{:s}", msg);
            throw std::runtime_error(msg);
        }
        default: {
            std::string msg = fmt::format("N2FrameDesc::get_num_prod given unknown N2Layout: {:s}",
                                          N2Layout_to_string(n2_layout_in));
            ERROR_NON_OO("{:s}", msg);
            throw std::runtime_error(msg);
        }
    }
}

n2frame_layout_t N2FrameDesc::get_frame_layout(uint32_t num_elements_in, uint32_t num_ev_in,
                                               size_t num_prod_in) {
    std::vector<std::pair<N2Field, size_t>> field_sizes;
    field_sizes.push_back({N2Field::vis, sizeof(N2::cfloat) * num_prod_in});
    field_sizes.push_back({N2Field::weight, sizeof(float) * num_prod_in});
    field_sizes.push_back({N2Field::flags, sizeof(float) * num_elements_in});
    field_sizes.push_back({N2Field::eval, sizeof(float) * num_ev_in});
    field_sizes.push_back({N2Field::evec, sizeof(N2::cfloat) * num_ev_in * num_elements_in});
    field_sizes.push_back({N2Field::emethod, sizeof(N2EigenMethod) * 1});
    field_sizes.push_back({N2Field::erms, sizeof(float) * 1});
    field_sizes.push_back({N2Field::gain, sizeof(N2::cfloat) * num_elements_in});

    n2frame_layout_t layout;
    size_t offset = 0;
    for (const auto& field : field_sizes) {
        layout.fields[field.first] = {offset, offset + field.second};
        offset += field.second;
    }
    return layout;
}

size_t N2FrameDesc::calculate_frame_size(uint32_t num_elements_in, uint32_t num_ev_in,
                                         size_t num_prod_in) {
    return get_frame_layout(num_elements_in, num_ev_in, num_prod_in).total_size();
}

size_t N2FrameDesc::calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {
    return from_config(config, unique_name)->get_byte_size();
}

void N2FrameDesc::fill_prod_maps_FullUpperTri(std::vector<N2::prod_ctype>& prods) const {
    size_t num_prod = (size_t(num_elements) * (num_elements + 1)) / 2;
    prods.resize(num_prod);
    size_t p = 0;
    for (uint16_t i = 0; i < num_elements; i++) {
        for (uint16_t j = i; j < num_elements; j++) {
            prods[p] = {.input_a = i, .input_b = j};
            p++;
        }
    }
}

void N2FrameDesc::fill_prod_maps_Autocorrelations(std::vector<N2::prod_ctype>& prods) const {
    prods.resize(num_elements);
    for (uint16_t i = 0; i < num_elements; i++) {
        prods[i] = {.input_a = i, .input_b = i};
    }
}

void N2FrameDesc::fill_prod_maps(std::vector<N2::prod_ctype>& prods) const {
    switch (n2_layout) {
        case N2Layout::FullUpperTri:
            fill_prod_maps_FullUpperTri(prods);
            break;
        case N2Layout::Autocorrelations:
            fill_prod_maps_Autocorrelations(prods);
            break;
        case N2Layout::InputANDMasked:
        case N2Layout::InputORMasked:
        case N2Layout::GeneralSubset:
        case N2Layout::RedundantBaselineAvg:
            // Use stored product list for subset layouts
            prods = product_list;
            break;
        default: {
            std::string msg = fmt::format(
                "N2FrameDesc::fill_prod_maps has not been implemented for N2Layout {:s}",
                N2Layout_to_string(n2_layout));
            FATAL_ERROR_NON_OO("{:s}", msg);
            break;
        }
    }
}

} // namespace kotekan
