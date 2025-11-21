#include "DataType.hpp" // for operator<<
#include "Symbol.hpp"   // for operator<<, Symbol

#include <NDArray.hpp>
#include <cassert> // for assert
#include <sstream> // for basic_ostringstream
#include <string>  // for operator<<, basic_string, string

namespace kotekan {

namespace {

const char* format_bool(bool b) {
    return b ? "true" : "false";
}

template<typename T>
std::string format_vector(const std::vector<T>& vec) {
    std::ostringstream buf;
    buf << "[";
    bool isfirst = true;
    for (const auto& x : vec)
        buf << (isfirst ? "" : ", ") << x, isfirst = false;
    buf << "]";
    return buf.str();
}

} // namespace

void GenericNDArray::output_metadata(std::ostream& os) const {
    os << "NDArray:\n"
       << "    quantity name:   " << get_quantity_name() << "\n"
       << "    value datatype:  " << get_value_datatype() << "\n"
       << "    value type size: " << get_value_type_size() << "\n"
       << "    rank:            " << get_rank() << "\n"
       << "    extents:         " << format_vector(get_extents()) << "\n"
       << "    empty:           " << format_bool(get_empty()) << "\n"
       << "    size:            " << get_size() << "\n"
       << "    dimnames:        " << format_vector(get_dimnames()) << "\n"
       << "    strides:         " << format_vector(get_strides()) << "\n";
}

// this templated function implemets a reurives 2D loop calling itself until
// my_rank == rank and my_datatype == datatype, then it creates a NDArray of that
// rank and type. This resolves the runtime values into compile time constants.
template<DataType my_datatype, size_t my_rank>
static inline std::shared_ptr<GenericNDArray>
make_NDArray(const DataType datatype, const Symbol quantity_name,
             const std::vector<std::ptrdiff_t>& extents, const std::vector<Symbol>& dimnames,
             void* data) {
    // recurse until datatype matches
    if constexpr (my_datatype == unknown_type) {
        assert(my_datatype != unknown_type);
        return nullptr;
    } else if (datatype == my_datatype) {
        // datatype matches, recurse until rank matches
        if constexpr (my_rank == 0) {
            assert(my_rank != 0);
            return nullptr;
        } else if (int(extents.size()) == my_rank) {
            // both match
            return std::shared_ptr<GenericNDArray>(new NDArray<GetType_t<my_datatype>, my_rank>(
                quantity_name, extents, dimnames, static_cast<GetType_t<my_datatype>*>(data)));
        } else {
            return make_NDArray<my_datatype, my_rank - 1>(datatype, quantity_name, extents,
                                                          dimnames, data);
        }
    } else {
        return make_NDArray<DataType(my_datatype - 1), my_rank>(datatype, quantity_name, extents,
                                                                dimnames, data);
    }
}

std::shared_ptr<GenericNDArray> GenericNDArray::create(const DataType value_datatype,
                                                       const Symbol quantity_name,
                                                       const std::vector<std::ptrdiff_t>& extents,
                                                       const std::vector<Symbol>& dimnames,
                                                       void* data) {
    assert(extents.size() == dimnames.size());
    return make_NDArray<DataType(end_type - 1), max_rank>(value_datatype, quantity_name, extents,
                                                          dimnames, data);
}

} // namespace kotekan
