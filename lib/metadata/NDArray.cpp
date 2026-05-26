#include <NDArray.hpp>
#include <sys/types.h>    // for ssize_t
#include <cassert>        // for assert
#include <sstream>        // for basic_ostringstream
#include <stdexcept>      // for runtime_error
#include <string>         // for operator<<, basic_string, string
#include <cstring>        // for memcmp

#include "DataType.hpp"   // for DataType, GetType_t, operator<<
#include "Symbol.hpp"     // for Symbol, operator<<, operator!=, operator==
#include "FrameDesc.hpp"  // for FrameDesc

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

void GenericNDArray::output_framedesc(std::ostream& os) const {
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
    assert(extents.size() == dimnames.size() && "Sizes of extents and dimnames must aggree");
    return make_NDArray<DataType(end_type - 1), max_rank>(value_datatype, quantity_name, extents,
                                                          dimnames, data);
}

bool GenericNDArray::operator==(const FrameDesc& other_desc) const {
    const GenericNDArray* other_ptr = dynamic_cast<const GenericNDArray*>(&other_desc);
    if (!other_ptr)
        return false;
    const GenericNDArray& other = *other_ptr;

    if (this->get_value_datatype() != other.get_value_datatype())
        return false;
    if (this->get_quantity_name() != other.get_quantity_name())
        return false;
    if (this->get_rank() != other.get_rank())
        return false;
    if (this->get_extents() != other.get_extents())
        return false;
    if (this->get_dimnames() != other.get_dimnames())
        return false;
    if (this->get_strides() != other.get_strides())
        return false;

    // currently all NDArrays have "simple" strides, ie there are no gaps in
    // memory, so a bulk memory compare will do
    {
        bool is_simple_stride = true;
        auto const& strides = this->get_strides();
        auto const& extents = this->get_extents();
        for (ssize_t d = ssize_t(strides.size()) - 1, simple_stride = 1; d >= 0; --d) {
            if (simple_stride != strides[d]) {
                is_simple_stride = false;
                break;
            }
            simple_stride *= extents[d];
        }
        if (!is_simple_stride) {
            std::ostringstream buf;
            buf << "NDArray " << this->get_quantity_name()
                << " does not have simple stride. Strides " << format_vector(strides)
                << " are not simple for extents " << format_vector(extents)
                << " and cannot be compared.";
            // cannot use ERROR since I don't derive from kotekan_loging
            throw std::runtime_error(buf.str());
        }
    }

    if (this->get_data() == nullptr && other.get_data() == nullptr)
        return true;
    else if (this->get_data() != nullptr && other.get_data() != nullptr)
        return std::memcmp(this->get_data(), other.get_data(),
                           this->get_size() * this->get_value_type_size())
               == 0;
    else
        return false;
}

} // namespace kotekan
