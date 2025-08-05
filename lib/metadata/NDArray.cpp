#include "DataType.hpp" // for operator<<
#include "Symbol.hpp"   // for operator<<, Symbol

#include <NDArray.hpp>
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
       << "    type:      " << get_value_type() << "\n"
       << "    type size: " << get_value_type_size() << "\n"
       << "    rank:      " << get_rank() << "\n"
       << "    extents:   " << format_vector(get_extents()) << "\n"
       << "    empty:     " << format_bool(get_empty()) << "\n"
       << "    size:      " << get_size() << "\n"
       << "    dimnames:  " << format_vector(get_dimnames()) << "\n"
       << "    strides:   " << format_vector(get_strides()) << "\n";
}

} // namespace kotekan
