#include "NDArray.hpp"

#include "Config.hpp"    // for Config
#include "DataType.hpp"  // for DataType, GetType_t, operator<<
#include "FrameDesc.hpp" // for FrameDesc
#include "Symbol.hpp"    // for Symbol, operator<<, operator!=, operator==

#include <cassert>     // for assert
#include <cstring>     // for memcmp
#include <sstream>     // for basic_ostringstream
#include <stdexcept>   // for runtime_error
#include <string>      // for operator<<, basic_string, string
#include <sys/types.h> // for ssize_t

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

// An unset (empty) Symbol has no string; render it as an empty string for logs
// and error messages (get_string() throws on an invalid Symbol).
std::string symbol_str(const Symbol& s) {
    return s.valid() ? s.get_string() : std::string();
}

// Serialize a Symbol to JSON, encoding an unset (invalid) Symbol as `null` so a
// deliberately-unset label is distinguishable from a set one on the wire.
nlohmann::json symbol_to_json(const Symbol& s) {
    return s.valid() ? nlohmann::json(s.get_string()) : nlohmann::json(nullptr);
}

// Inverse of symbol_to_json: `null` becomes an unset (invalid) Symbol.
Symbol symbol_from_json(const nlohmann::json& j) {
    return j.is_null() ? Symbol() : Symbol(j.get<std::string>());
}

// Validate the structural fields and labels of an NDArray descriptor, throwing
// std::runtime_error (with `context` in the message) on any problem. Shared by
// from_config and the JSON deserializer (for descriptors received over
// bufferRecv), so both paths enforce the same rules.
void validate_ndarray_fields(DataType value_datatype, const std::string& value_type_name,
                             const std::vector<std::ptrdiff_t>& extents,
                             const std::vector<Symbol>& dimnames,
                             const std::vector<std::ptrdiff_t>& dimscalings,
                             const std::string& context) {
    if (value_datatype == unknown_type)
        throw std::runtime_error(fmt::format(
            fmt("GenericNDArray ({:s}): unknown value_type '{:s}'"), context, value_type_name));
    if (extents.empty() || extents.size() > GenericNDArray::max_rank)
        throw std::runtime_error(
            fmt::format(fmt("GenericNDArray ({:s}): rank {:d} is outside [1, {:d}]"), context,
                        extents.size(), GenericNDArray::max_rank));
    for (std::size_t d = 0; d < extents.size(); ++d)
        if (extents[d] <= 0)
            throw std::runtime_error(fmt::format(
                fmt("GenericNDArray ({:s}): extent {:d} of dimension {:d} is not positive"),
                context, extents[d], d));
    if (dimnames.size() != extents.size())
        throw std::runtime_error(
            fmt::format(fmt("GenericNDArray ({:s}): dimnames count {:d} does not match rank {:d}"),
                        context, dimnames.size(), extents.size()));
    // Only set (non-empty) labels can collide; unset labels are filled in later.
    for (std::size_t d = 0; d < dimnames.size(); ++d)
        for (std::size_t d1 = 0; d1 < d; ++d1)
            if (dimnames[d].valid() && dimnames[d] == dimnames[d1])
                throw std::runtime_error(
                    fmt::format(fmt("GenericNDArray ({:s}): duplicate dimname '{:s}'"), context,
                                symbol_str(dimnames[d])));
    if (dimscalings.size() != extents.size())
        throw std::runtime_error(fmt::format(
            fmt("GenericNDArray ({:s}): dimscalings count {:d} does not match rank {:d}"), context,
            dimscalings.size(), extents.size()));
    for (std::size_t d = 0; d < dimscalings.size(); ++d)
        if (dimscalings[d] <= 0)
            throw std::runtime_error(fmt::format(
                fmt("GenericNDArray ({:s}): dimscaling {:d} of dimension {:d} is not positive"),
                context, dimscalings[d], d));
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
       << "    dimscalings:     " << format_vector(get_dimscalings()) << "\n"
       << "    strides:         " << format_vector(get_strides()) << "\n";
}

nlohmann::json GenericNDArray::to_json() const {
    nlohmann::json j;
    j["frame_desc_type"] = "ndarray";
    j["value_type"] = type_to_string(get_value_datatype());
    j["extents"] = get_extents();
    j["quantity_name"] = symbol_to_json(get_quantity_name());
    nlohmann::json dimnames = nlohmann::json::array();
    for (const auto& dimname : get_dimnames())
        dimnames.push_back(symbol_to_json(dimname));
    j["dimnames"] = std::move(dimnames);
    j["dimscalings"] = get_dimscalings();
    return j;
}

std::shared_ptr<const FrameDesc> GenericNDArray::from_json(const nlohmann::json& j) {
    const std::string type_name = j.at("value_type").get<std::string>();
    const DataType value_datatype = string_to_type(type_name);

    const auto extents = j.at("extents").get<std::vector<std::ptrdiff_t>>();

    const Symbol quantity_name = symbol_from_json(j.at("quantity_name"));

    std::vector<Symbol> dimnames;
    for (const auto& dimname : j.at("dimnames"))
        dimnames.push_back(symbol_from_json(dimname));

    const auto dimscalings = j.at("dimscalings").get<std::vector<std::ptrdiff_t>>();

    // Full validation, including bounding the rank; describe() then resolves the
    // runtime datatype/rank into a concrete NDArray.
    validate_ndarray_fields(value_datatype, type_name, extents, dimnames, dimscalings,
                            "received frame descriptor");
    return describe(value_datatype, quantity_name, extents, dimnames, dimscalings);
}

// this templated function implemets a reurives 2D loop calling itself until
// my_rank == rank and my_datatype == datatype, then it creates a NDArray of that
// rank and type. This resolves the runtime values into compile time constants.
template<DataType my_datatype, size_t my_rank>
static inline std::shared_ptr<GenericNDArray>
make_NDArray(const DataType datatype, const Symbol quantity_name,
             const std::vector<std::ptrdiff_t>& extents, const std::vector<Symbol>& dimnames,
             const std::vector<std::ptrdiff_t>& dimscalings) {
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
                quantity_name, extents, dimnames, dimscalings,
                static_cast<GetType_t<my_datatype>*>(nullptr)));
        } else {
            return make_NDArray<my_datatype, my_rank - 1>(datatype, quantity_name, extents,
                                                          dimnames, dimscalings);
        }
    } else {
        return make_NDArray<DataType(my_datatype - 1), my_rank>(datatype, quantity_name, extents,
                                                                dimnames, dimscalings);
    }
}

std::shared_ptr<GenericNDArray>
GenericNDArray::describe(const DataType value_datatype, const Symbol quantity_name,
                         const std::vector<std::ptrdiff_t>& extents,
                         const std::vector<Symbol>& dimnames,
                         const std::vector<std::ptrdiff_t>& dimscalings) {
    assert(extents.size() == dimnames.size() && "Sizes of extents and dimnames must aggree");
    assert(extents.size() == dimscalings.size() && "Sizes of extents and dimscalings must aggree");
    return make_NDArray<DataType(end_type - 1), max_rank>(value_datatype, quantity_name, extents,
                                                          dimnames, dimscalings);
}

std::shared_ptr<GenericNDArray> GenericNDArray::from_config(const Config& config,
                                                            const std::string& location) {
    const std::string type_name = config.get<std::string>(location, "value_type");
    const DataType value_datatype = string_to_type(type_name);

    // `quantity_name` and `dimnames` are semantic labels: they are
    // OPTIONAL in config. When omitted, the descriptor is left with
    // unset (empty) labels and the producing stage fills them in via
    // Buffer::ensure_frame_desc(); when given, the stage validates
    // against them. `value_type` and `extents` are structural (they
    // fix the byte layout the bufferFactory allocates) and are
    // required.
    const Symbol quantity_name(config.get_default<std::string>(location, "quantity_name", ""));

    // Extent entries may be arithmetic expressions referencing other
    // (scoped) config values, so evaluate them individually.
    const auto extents_json = config.get<std::vector<nlohmann::json>>(location, "extents");
    std::vector<std::ptrdiff_t> extents;
    extents.reserve(extents_json.size());
    for (const auto& extent : extents_json)
        extents.push_back(config.eval<std::ptrdiff_t>(location, extent));

    const auto dimname_strings =
        config.get_default<std::vector<std::string>>(location, "dimnames", {});
    std::vector<Symbol> dimnames;
    if (dimname_strings.empty()) {
        // Labels omitted: leave one unset (empty) label per axis for a stage to fill.
        dimnames.assign(extents.size(), Symbol(""));
    } else {
        dimnames.reserve(dimname_strings.size());
        for (const auto& dimname : dimname_strings)
            dimnames.emplace_back(dimname);
    }

    // `dimscalings` is an optional structural label like `dimnames`: when
    // omitted, each axis defaults to a scaling of 1 (no downsampling).
    // Entries may be arithmetic expressions referencing other (scoped) config
    // values, so evaluate them individually.
    const auto dimscalings_json =
        config.get_default<std::vector<nlohmann::json>>(location, "dimscalings", {});
    std::vector<std::ptrdiff_t> dimscalings;
    if (dimscalings_json.empty()) {
        // Scalings omitted: default each axis to 1.
        dimscalings.assign(extents.size(), 1);
    } else {
        dimscalings.reserve(dimscalings_json.size());
        for (const auto& dimscaling : dimscalings_json)
            dimscalings.push_back(config.eval<std::ptrdiff_t>(location, dimscaling));
    }

    validate_ndarray_fields(value_datatype, type_name, extents, dimnames, dimscalings,
                            fmt::format(fmt("path {:s}"), location));
    return describe(value_datatype, quantity_name, extents, dimnames, dimscalings);
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
        throw std::runtime_error(
            fmt::format(fmt("{:s} mismatch: {:s} != {:s}"), what, std::string(x), std::string(y)));
    };

    // Dimscalings behave like labels: a scaling of 1 is the "unspecified"
    // default (no downsampling) used when a config buffer declaration omits
    // `dimscalings`, and it yields to an explicit non-1 scaling supplied by the
    // producing stage. Two differing explicit (non-1) scalings are a conflict.
    auto merge_scaling = [](std::ptrdiff_t x, std::ptrdiff_t y) -> std::ptrdiff_t {
        if (x == 1)
            return y;
        if (y == 1)
            return x;
        if (x == y)
            return x;
        throw std::runtime_error(fmt::format(fmt("dimscaling mismatch: {:d} != {:d}"), x, y));
    };

    const Symbol quantity_name =
        merge_label(a.get_quantity_name(), b.get_quantity_name(), "quantity name");
    const std::vector<Symbol> da = a.get_dimnames();
    const std::vector<Symbol> db = b.get_dimnames();
    std::vector<Symbol> dimnames(da.size());
    for (std::size_t d = 0; d < da.size(); ++d)
        dimnames[d] = merge_label(da[d], db[d], "dimname");

    const std::vector<std::ptrdiff_t> sa = a.get_dimscalings();
    const std::vector<std::ptrdiff_t> sb = b.get_dimscalings();
    std::vector<std::ptrdiff_t> dimscalings(sa.size());
    for (std::size_t d = 0; d < sa.size(); ++d)
        dimscalings[d] = merge_scaling(sa[d], sb[d]);

    // `b` contributed nothing that `a` lacked: nothing to complete, so signal
    // "no change" (the caller keeps the existing descriptor).
    if (quantity_name == a.get_quantity_name() && dimnames == da && dimscalings == sa)
        return nullptr;
    return describe(a.get_value_datatype(), quantity_name, a.get_extents(), dimnames, dimscalings);
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
    if (get_dimscalings() != other.get_dimscalings())
        return fmt::format(fmt("dimscalings do not match: {:s} != {:s}"),
                           format_vector(get_dimscalings()),
                           format_vector(other.get_dimscalings()));
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
    if (this->get_dimscalings() != other.get_dimscalings())
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
