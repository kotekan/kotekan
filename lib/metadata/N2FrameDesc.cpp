#include "N2FrameDesc.hpp"

#include "kotekanLogging.hpp"

#include "fmt.hpp"

namespace kotekan {

N2FrameDesc::N2FrameDesc(uint32_t num_elements, uint32_t num_ev, uint32_t num_products,
                         N2Layout n2_layout) :
    num_elements(num_elements), num_ev(num_ev), num_products(num_products), n2_layout(n2_layout) {}

Symbol N2FrameDesc::get_quantity_name() const {
    return Symbol("N2");
}

void N2FrameDesc::output_metadata(std::ostream& os) const {
    os << "N2FrameDesc:\n"
       << "    num_elements: " << num_elements << "\n"
       << "    num_ev:       " << num_ev << "\n"
       << "    num_products: " << num_products << "\n"
       << "    n2_layout:    " << N2Layout_to_string(n2_layout) << "\n";
}

bool N2FrameDesc::operator==(const FrameDesc& other) const {
    const N2FrameDesc* other_ptr = dynamic_cast<const N2FrameDesc*>(&other);
    if (!other_ptr)
        return false;

    return (num_elements == other_ptr->num_elements) && (num_ev == other_ptr->num_ev)
           && (num_products == other_ptr->num_products) && (n2_layout == other_ptr->n2_layout);
}

size_t N2FrameDesc::get_byte_size() const {
    return calculate_frame_size(num_elements, num_ev, num_products);
}

size_t N2FrameDesc::get_num_prod(uint32_t num_elements_in, N2Layout n2_layout_in) {
    size_t num_prod_in;
    switch (n2_layout_in) {
        case N2Layout::FullUpperTri:
            num_prod_in = (size_t(num_elements_in) * (num_elements_in + 1)) / 2;
            break;
        case N2Layout::Autocorrelations:
            num_prod_in = num_elements_in;
            break;
        default:
            std::string msg = fmt::format("N2FrameDesc::get_num_prod given unknown N2Layout: {:s}",
                                          N2Layout_to_string(n2_layout_in));
            ERROR_NON_OO("{:s}", msg);
            throw std::runtime_error(msg);
            break;
    }
    return num_prod_in;
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
    const int num_elements_in = config.get<int>(unique_name, "num_elements");
    const int num_ev_in = config.get<int>(unique_name, "num_ev");
    const N2Layout n2_layout = config.get<N2Layout>(unique_name, "n2_layout");
    const uint64_t num_prod_in = get_num_prod(num_elements_in, n2_layout);
    return calculate_frame_size(num_elements_in, num_ev_in, num_prod_in);
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
        default:
            std::string msg = fmt::format(
                "N2FrameDesc::fill_prod_maps has not been implemented for N2Layout {:s}",
                N2Layout_to_string(n2_layout));
            FATAL_ERROR_NON_OO("{:s}", msg);
            break;
    }
}

} // namespace kotekan