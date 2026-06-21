#include "Config.hpp"   // for Config
#include "DataType.hpp" // for operator<<
#include "Symbol.hpp"   // for operator<<, Symbol

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

FrameDesc::WireType GenericNDArray::wire_type() const {
    return FrameDesc::WireType::generic_ndarray;
}

size_t GenericNDArray::serialized_payload_size() const {
    size_t n = wire::str_size(type_to_string(get_value_datatype()));
    n += sizeof(uint32_t);             // rank
    n += get_rank() * sizeof(int64_t); // extents
    n += wire::str_size(std::string(get_quantity_name()));
    n += sizeof(uint32_t);             // dimnames count
    for (const auto& dimname : get_dimnames())
        n += wire::str_size(std::string(dimname));
    return n;
}

void GenericNDArray::serialize_payload(char* out) const {
    out = wire::put_str(out, type_to_string(get_value_datatype()));
    const auto extents = get_extents();
    out = wire::put_u32(out, static_cast<uint32_t>(extents.size()));
    for (const auto extent : extents)
        out = wire::put_i64(out, static_cast<int64_t>(extent));
    out = wire::put_str(out, std::string(get_quantity_name()));
    const auto dimnames = get_dimnames();
    out = wire::put_u32(out, static_cast<uint32_t>(dimnames.size()));
    for (const auto& dimname : dimnames)
        out = wire::put_str(out, std::string(dimname));
}

std::shared_ptr<const FrameDesc> GenericNDArray::deserialize_payload(const char* bytes,
                                                                     size_t size) {
    wire::Reader r(bytes, size);
    const std::string type_name = r.get_str();
    const DataType value_datatype = string_to_type(type_name);
    if (value_datatype == unknown_type)
        throw std::runtime_error(
            fmt::format(fmt("GenericNDArray::deserialize: unknown value_type '{:s}'"), type_name));

    const uint32_t rank = r.get_u32();
    if (rank < 1 || rank > max_rank)
        throw std::runtime_error(fmt::format(
            fmt("GenericNDArray::deserialize: rank {:d} is outside [1, {:d}]"), rank, max_rank));

    std::vector<std::ptrdiff_t> extents;
    extents.reserve(rank);
    for (uint32_t d = 0; d < rank; ++d) {
        const std::ptrdiff_t extent = static_cast<std::ptrdiff_t>(r.get_i64());
        if (extent <= 0)
            throw std::runtime_error("GenericNDArray::deserialize: non-positive extent");
        extents.push_back(extent);
    }

    const Symbol quantity_name(r.get_str());

    const uint32_t num_dimnames = r.get_u32();
    if (num_dimnames != rank)
        throw std::runtime_error(
            fmt::format(fmt("GenericNDArray::deserialize: dimnames count {:d} does not match rank "
                            "{:d}"),
                        num_dimnames, rank));
    std::vector<Symbol> dimnames;
    dimnames.reserve(num_dimnames);
    for (uint32_t d = 0; d < num_dimnames; ++d)
        dimnames.emplace_back(r.get_str());

    r.require_empty();
    return describe(value_datatype, quantity_name, extents, dimnames);
}

// this templated function implemets a reurives 2D loop calling itself until
// my_rank == rank and my_datatype == datatype, then it creates a NDArray of that
// rank and type. This resolves the runtime values into compile time constants.
template<DataType my_datatype, size_t my_rank>
static inline std::shared_ptr<GenericNDArray>
make_NDArray(const DataType datatype, const Symbol quantity_name,
             const std::vector<std::ptrdiff_t>& extents, const std::vector<Symbol>& dimnames) {
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
                quantity_name, extents, dimnames, static_cast<GetType_t<my_datatype>*>(nullptr)));
        } else {
            return make_NDArray<my_datatype, my_rank - 1>(datatype, quantity_name, extents,
                                                          dimnames);
        }
    } else {
        return make_NDArray<DataType(my_datatype - 1), my_rank>(datatype, quantity_name, extents,
                                                                dimnames);
    }
}

std::shared_ptr<GenericNDArray> GenericNDArray::describe(const DataType value_datatype,
                                                         const Symbol quantity_name,
                                                         const std::vector<std::ptrdiff_t>& extents,
                                                         const std::vector<Symbol>& dimnames) {
    assert(extents.size() == dimnames.size() && "Sizes of extents and dimnames must aggree");
    return make_NDArray<DataType(end_type - 1), max_rank>(value_datatype, quantity_name, extents,
                                                          dimnames);
}

std::shared_ptr<GenericNDArray> GenericNDArray::from_config(const Config& config,
                                                            const std::string& location) {
    const std::string type_name = config.get<std::string>(location, "value_type");
    const DataType value_datatype = string_to_type(type_name);
    if (value_datatype == unknown_type)
        throw std::runtime_error(fmt::format(
            fmt("GenericNDArray: unknown value_type '{:s}' in path {:s}"), type_name, location));

    // `quantity_name` and `dimnames` are semantic labels: they are OPTIONAL in
    // config. When omitted, the descriptor is left with unset (empty) labels and
    // the producing stage fills them in via Buffer::ensure_frame_desc(); when
    // given, the stage validates against them. `value_type` and `extents` are
    // structural (they fix the byte layout the bufferFactory allocates) and are
    // required.
    const Symbol quantity_name(config.get_default<std::string>(location, "quantity_name", ""));

    // Extent entries may be arithmetic expressions referencing other
    // (scoped) config values, so evaluate them individually.
    const auto extents_json = config.get<std::vector<nlohmann::json>>(location, "extents");
    std::vector<std::ptrdiff_t> extents;
    extents.reserve(extents_json.size());
    for (const auto& extent : extents_json)
        extents.push_back(config.eval<std::ptrdiff_t>(location, extent));

    if (extents.empty() || extents.size() > max_rank)
        throw std::runtime_error(
            fmt::format(fmt("GenericNDArray: rank {:d} is outside [1, {:d}] in path {:s}"),
                        extents.size(), max_rank, location));
    for (std::size_t d = 0; d < extents.size(); ++d)
        if (extents[d] <= 0)
            throw std::runtime_error(fmt::format(
                fmt("GenericNDArray: extent {:d} of dimension {:d} is not positive in path {:s}"),
                extents[d], d, location));

    const auto dimname_strings =
        config.get_default<std::vector<std::string>>(location, "dimnames", {});
    std::vector<Symbol> dimnames;
    if (dimname_strings.empty()) {
        // Labels omitted: leave one unset (empty) label per axis for a stage to fill.
        dimnames.assign(extents.size(), Symbol(""));
    } else {
        if (dimname_strings.size() != extents.size())
            throw std::runtime_error(fmt::format(
                fmt("GenericNDArray: extents size ({:d}) does not match dimnames size ({:d}) in "
                    "path {:s}"),
                extents.size(), dimname_strings.size(), location));
        dimnames.reserve(dimname_strings.size());
        for (const auto& dimname : dimname_strings)
            dimnames.emplace_back(dimname);
        for (std::size_t d = 0; d < dimnames.size(); ++d)
            for (std::size_t d1 = 0; d1 < d; ++d1)
                if (dimnames[d] == dimnames[d1])
                    throw std::runtime_error(fmt::format(
                        fmt("GenericNDArray: duplicate dimname '{:s}' in path {:s}"),
                        std::string(dimnames[d]), location));
    }

    return describe(value_datatype, quantity_name, extents, dimnames);
}

std::shared_ptr<GenericNDArray> GenericNDArray::reconcile(const GenericNDArray& a,
                                                          const GenericNDArray& b) {
    // Structural fields must match exactly.
    if (a.get_value_datatype() != b.get_value_datatype())
        throw std::runtime_error(fmt::format(fmt("value type mismatch: {:s} != {:s}"),
                                             type_to_string(a.get_value_datatype()),
                                             type_to_string(b.get_value_datatype())));
    if (a.get_extents() != b.get_extents())
        throw std::runtime_error(fmt::format(fmt("extents do not match: {:s} != {:s}"),
                                             format_vector(a.get_extents()),
                                             format_vector(b.get_extents())));

    // Labels: an unset (empty) label is filled from the other side; labels set
    // on both sides must agree.
    const Symbol unset("");
    auto merge_label = [&unset](const Symbol& x, const Symbol& y, const char* what) -> Symbol {
        if (x == unset)
            return y;
        if (y == unset)
            return x;
        if (x == y)
            return x;
        throw std::runtime_error(fmt::format(fmt("{:s} mismatch: {:s} != {:s}"), what,
                                             std::string(x), std::string(y)));
    };

    const Symbol quantity_name =
        merge_label(a.get_quantity_name(), b.get_quantity_name(), "quantity name");
    const std::vector<Symbol> da = a.get_dimnames();
    const std::vector<Symbol> db = b.get_dimnames();
    std::vector<Symbol> dimnames(da.size());
    for (std::size_t d = 0; d < da.size(); ++d)
        dimnames[d] = merge_label(da[d], db[d], "dimname");

    // `b` contributed no labels that `a` lacked: nothing to complete, so signal
    // "no change" (the caller keeps the existing descriptor).
    if (quantity_name == a.get_quantity_name() && dimnames == da)
        return nullptr;
    return describe(a.get_value_datatype(), quantity_name, a.get_extents(), dimnames);
}

std::string GenericNDArray::structure_mismatch(const GenericNDArray& other) const {
    if (get_value_datatype() != other.get_value_datatype())
        return fmt::format(fmt("value type mismatch: {:s} != {:s}"),
                           type_to_string(get_value_datatype()),
                           type_to_string(other.get_value_datatype()));
    if (get_quantity_name() != other.get_quantity_name())
        return fmt::format(fmt("quantity name mismatch: {:s} != {:s}"), get_quantity_name(),
                           other.get_quantity_name());
    if (get_rank() != other.get_rank())
        return fmt::format(fmt("rank mismatch: {:d} != {:d}"), get_rank(), other.get_rank());
    if (get_extents() != other.get_extents())
        return fmt::format(fmt("extents do not match: {:s} != {:s}"), format_vector(get_extents()),
                           format_vector(other.get_extents()));
    if (get_dimnames() != other.get_dimnames())
        return fmt::format(fmt("dimnames do not match: {:s} != {:s}"),
                           format_vector(get_dimnames()), format_vector(other.get_dimnames()));
    return std::string();
}

std::string GenericNDArray::describe_mismatch(const FrameDesc& other) const {
    const GenericNDArray* nd = dynamic_cast<const GenericNDArray*>(&other);
    if (!nd)
        return FrameDesc::describe_mismatch(other);
    return structure_mismatch(*nd);
}

bool GenericNDArray::operator==(const FrameDesc& other_desc) const {
    const GenericNDArray* other_ptr = dynamic_cast<const GenericNDArray*>(&other_desc);
    if (!other_ptr)
        return false;
    const GenericNDArray& other = *other_ptr;

    if (!structure_mismatch(other).empty())
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
