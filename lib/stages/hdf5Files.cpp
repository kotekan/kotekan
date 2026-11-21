#include "hdf5Files.hpp"

#include "DataType.hpp" // for DataType, float16_t

#include "fmt.hpp" // for format, format_string

#include <algorithm>                   // for clamp, max
#include <chrono>                      // for system_clock, duration_cast, nanoseconds
#include <cstdint>                     // for uint8_t, int64_t, int16_t, int32_t, int8_t, uint16_t
#include <cstring>                     // for strnlen
#include <ctime>                       // for gmtime_r, tm
#include <highfive/H5DataSet.hpp>      // for DataSet
#include <highfive/H5DataSpace.hpp>    // for DataSpace
#include <highfive/H5Group.hpp>        // for Group
#include <highfive/H5PropertyList.hpp> // for RawPropertyList, PropertyType
#include <highfive/H5Selection.hpp>    // for Selection
#include <iomanip>                     // for operator<<, setfill, setw
#include <sstream>                     // for basic_ostream, operator<<, basic_ostringstream
#include <stdexcept>                   // for runtime_error
#include <string>                      // for string
#include <unistd.h>                    // for gethostname
#include <utility>                     // for move

namespace hdf5 {

HighFive::DataType chord2hdf5(const kotekan::DataType type) {
    switch (type) {
        case kotekan::uint1x8:
            return HighFive::AtomicType<std::uint8_t>(); // 8 bools (packed into a type)
        case kotekan::uint4x2:
            return HighFive::AtomicType<std::uint8_t>(); // 2 unsigned 4-bit integers (packed into a
                                                         // byte)
        case kotekan::uint8:
            return HighFive::AtomicType<std::uint8_t>();
        case kotekan::uint16:
            return HighFive::AtomicType<std::uint16_t>();
        case kotekan::uint32:
            return HighFive::AtomicType<std::uint32_t>();
        case kotekan::uint64:
            return HighFive::AtomicType<std::uint64_t>();
        case kotekan::int4x2:
            return HighFive::AtomicType<std::uint8_t>(); // 2 signed 4-bit integers (packed into a
                                                         // byte)
        case kotekan::int4x2_swapped_withoffset:
            return HighFive::AtomicType<std::uint8_t>(); // offset-encoded (stored is value + 8),
                                                         // low and high values swapped
        case kotekan::int8:
            return HighFive::AtomicType<std::int8_t>();
        case kotekan::int16:
            return HighFive::AtomicType<std::int16_t>();
        case kotekan::int32:
            return HighFive::AtomicType<std::int32_t>();
        case kotekan::int64:
            return HighFive::AtomicType<std::int64_t>();
        case kotekan::float16:
            return HighFive::AtomicType<float16_t>();
            // return HighFive::AtomicType<std::uint16_t>();
        case kotekan::float32:
            return HighFive::AtomicType<float>();
        case kotekan::float64:
            return HighFive::AtomicType<double>();
        default:
            throw std::runtime_error(
                fmt::format("chord2hdf5 given unknown DataType value: {:d} ", (int64_t)type));
    }
}

std::string acq_dir_path(const std::string& base_dir,
                         const std::chrono::system_clock::time_point t) {
    std::string dir = base_dir;
    while (dir.size() > 1 && dir.back() == '/')
        dir.pop_back();

    const auto time_t_t = std::chrono::system_clock::to_time_t(t);
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count()
        % 1'000'000'000LL;

    std::tm tm_buf{};
    gmtime_r(&time_t_t, &tm_buf);

    std::ostringstream oss;
    oss << dir << "/acq_" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_" << std::setfill('0')
        << std::setw(9) << ns;
    return oss.str();
}

std::string file_path_prefix(const std::string& dir, const std::string& file_name,
                             const bool prefix_hostname, const bool prefix_host_rank,
                             const int host_pool_rank) {
    std::ostringstream buf;
    buf << dir << "/";
    if (prefix_hostname) {
        char hostname[256];
        gethostname(hostname, sizeof hostname);
        buf << hostname << "_";
    }
    if (prefix_host_rank)
        buf << "x" << std::setw(4) << std::setfill('0') << host_pool_rank << "_";
    buf << file_name;
    return buf.str();
}

namespace {

using Kind = FrameMetadataColumn::Kind;

// Append the elements of @p j, which must have the shape of @p col from dimension @p d on.
void collect_values(const nlohmann::json& j, const std::string& path, const std::size_t d,
                    FrameMetadataColumn& col) {
    if (d < col.shape.size()) {
        if (!j.is_array() || j.size() != col.shape[d])
            throw std::runtime_error(
                fmt::format("frame metadata \"{:s}\" is a ragged array", path));
        for (const auto& element : j)
            collect_values(element, path, d + 1, col);
        return;
    }
    Kind kind;
    if (j.is_boolean())
        kind = Kind::Bool;
    else if (j.is_string())
        kind = Kind::String;
    else if (j.is_number_float())
        kind = Kind::Float;
    else if (j.is_number_unsigned())
        kind = Kind::UInt;
    else if (j.is_number_integer())
        kind = Kind::Int;
    else
        throw std::runtime_error(
            fmt::format("frame metadata \"{:s}\" holds a {:s}, which cannot be stored per frame",
                        path, j.type_name()));
    if (col.values.empty())
        col.kind = kind;
    else if (kind != col.kind)
        throw std::runtime_error(
            fmt::format("frame metadata \"{:s}\" is an array of mixed kinds", path));
    if (kind == Kind::String)
        col.width = std::max(col.width, j.get_ref<const std::string&>().size());
    col.values.push_back(j);
}

// Add the columns of @p j, the value at @p path. An object gives a column per member.
void add_columns(const nlohmann::json& j, const std::string& path, FrameMetadataColumns& columns) {
    if (j.is_object()) {
        if (j.empty())
            throw std::runtime_error(
                fmt::format("frame metadata \"{:s}\" is an empty object", path));
        for (const auto& item : j.items()) {
            if (item.key().empty() || item.key().find('/') != std::string::npos)
                throw std::runtime_error(
                    fmt::format("frame metadata key \"{:s}\" in \"{:s}\" cannot name a dataset",
                                item.key(), path));
            add_columns(item.value(), path.empty() ? item.key() : path + "/" + item.key(), columns);
        }
        return;
    }
    FrameMetadataColumn col;
    for (const nlohmann::json* p = &j; p->is_array(); p = &p->front()) {
        if (p->empty())
            throw std::runtime_error(
                fmt::format("frame metadata \"{:s}\" holds an empty array", path));
        col.shape.push_back(p->size());
    }
    collect_values(j, path, 0, col);
    columns[path] = std::move(col);
}

HighFive::DataType column_type(const FrameMetadataColumn& col) {
    switch (col.kind) {
        case Kind::Bool:
            return HighFive::create_enum_boolean();
        case Kind::Int:
            return HighFive::AtomicType<std::int64_t>();
        case Kind::UInt:
            return HighFive::AtomicType<std::uint64_t>();
        case Kind::Float:
            return HighFive::AtomicType<double>();
        default:
            return HighFive::FixedLengthStringType(col.width, HighFive::StringPadding::NullPadded);
    }
}

std::string column_path(const std::string& path) {
    return std::string(frame_metadata_group) + "/" + path;
}

template<typename T>
void write_row(HighFive::Selection& row, const HighFive::DataType& type,
               const std::vector<nlohmann::json>& values) {
    std::vector<T> data;
    data.reserve(values.size());
    for (const auto& value : values)
        data.push_back(value.template get<T>());
    row.write_raw(data.data(), type);
}

template<typename T>
std::vector<nlohmann::json> read_column(const HighFive::DataSet& dataset,
                                        const HighFive::DataType& type) {
    std::vector<T> data(dataset.getElementCount());
    dataset.read_raw(data.data(), type);
    return std::vector<nlohmann::json>(data.begin(), data.end());
}

// The value with the shape dims[d...] whose elements start at flat[next].
nlohmann::json nest_values(const std::vector<nlohmann::json>& flat,
                           const std::vector<std::size_t>& dims, const std::size_t d,
                           std::size_t& next) {
    if (d == dims.size())
        return flat[next++];
    nlohmann::json array = nlohmann::json::array();
    for (std::size_t i = 0; i < dims[d]; ++i)
        array.push_back(nest_values(flat, dims, d + 1, next));
    return array;
}

// Read the columns below @p group into @p frames, at the json keys @p keys.
void read_columns(const HighFive::Group& group, const std::vector<std::string>& keys,
                  std::vector<nlohmann::json>& frames, bool& have_frames) {
    for (const std::string& name : group.listObjectNames()) {
        std::vector<std::string> column_keys = keys;
        column_keys.push_back(name);
        if (group.getObjectType(name) == HighFive::ObjectType::Group) {
            read_columns(group.getGroup(name), column_keys, frames, have_frames);
            continue;
        }
        const HighFive::DataSet dataset = group.getDataSet(name);
        const std::vector<std::size_t> dims = dataset.getDimensions();
        if (dims.empty())
            throw std::runtime_error(fmt::format("frame metadata \"{:s}\" has rank 0", name));
        if (!have_frames) {
            frames.assign(dims[0], nlohmann::json::object());
            have_frames = true;
        } else if (dims[0] != frames.size()) {
            throw std::runtime_error(
                fmt::format("frame metadata \"{:s}\" has {:d} frames, other columns have {:d}",
                            name, dims[0], frames.size()));
        }

        const HighFive::DataType type = dataset.getDataType();
        std::vector<nlohmann::json> flat;
        if (type == HighFive::create_enum_boolean()) {
            for (const auto& value : read_column<std::int8_t>(dataset, type))
                flat.push_back(value != 0);
        } else if (type == HighFive::AtomicType<std::int64_t>()) {
            flat = read_column<std::int64_t>(dataset, type);
        } else if (type == HighFive::AtomicType<std::uint64_t>()) {
            flat = read_column<std::uint64_t>(dataset, type);
        } else if (type == HighFive::AtomicType<double>()) {
            flat = read_column<double>(dataset, type);
        } else if (type.isFixedLenStr()) {
            const std::size_t width = type.getSize();
            std::vector<char> data(dataset.getElementCount() * width);
            dataset.read_raw(data.data(), type);
            for (std::size_t i = 0; i < data.size(); i += width)
                flat.push_back(std::string(&data[i], strnlen(&data[i], width)));
        } else {
            throw std::runtime_error(
                fmt::format("frame metadata \"{:s}\" has an unexpected HDF5 type", name));
        }

        std::size_t next = 0;
        for (nlohmann::json& frame : frames) {
            nlohmann::json* node = &frame;
            for (const std::string& key : column_keys)
                node = &(*node)[key];
            *node = nest_values(flat, dims, 1, next);
        }
    }
}

} // namespace

FrameMetadataColumns frame_metadata_columns(const nlohmann::json& metadata) {
    if (!metadata.is_object())
        throw std::runtime_error("frame metadata is not a json object");
    FrameMetadataColumns columns;
    add_columns(metadata, "", columns);
    return columns;
}

FrameMetadataColumns create_frame_metadata(HighFive::File& file, FrameMetadataColumns columns,
                                           const bool use_compression) {
    for (auto& [path, col] : columns) {
        // Leave room for longer strings in later frames.
        if (col.kind == Kind::String)
            col.width = std::max<std::size_t>(col.width, 32);
        const HighFive::DataType type = column_type(col);

        std::vector<std::size_t> dims{0};
        dims.insert(dims.end(), col.shape.begin(), col.shape.end());
        std::vector<std::size_t> max_dims(dims);
        max_dims[0] = HighFive::DataSpace::UNLIMITED;

        // About 64 kB per chunk.
        std::vector<hsize_t> chunk(dims.begin(), dims.end());
        chunk[0] = std::clamp<std::size_t>(65536 / (type.getSize() * col.values.size()), 1, 1024);
        HighFive::RawPropertyList<HighFive::PropertyType::DATASET_CREATE> props;
        props.add(H5Pset_chunk, int(chunk.size()), chunk.data());
        if (use_compression && col.kind != Kind::Bool && col.kind != Kind::String) {
            const std::vector<unsigned int> bitshuffle_flags{BITSHUFFLE_BLOCKSIZE_AUTO,
                                                             BITSHUFFLE_COMPRESS_ZSTD, 9};
            props.add(H5Pset_filter, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY, bitshuffle_flags.size(),
                      bitshuffle_flags.data());
        }
        file.createDataSet(column_path(path), HighFive::DataSpace(dims, max_dims), type, props);
    }
    return columns;
}

bool frame_metadata_fits(const FrameMetadataColumns& file_columns,
                         const FrameMetadataColumns& frame) {
    if (frame.size() != file_columns.size())
        return false;
    for (const auto& [path, col] : frame) {
        const auto it = file_columns.find(path);
        if (it == file_columns.end() || col.kind != it->second.kind || col.shape != it->second.shape
            || col.width > it->second.width)
            return false;
    }
    return true;
}

void append_frame_metadata(HighFive::File& file, const FrameMetadataColumns& file_columns,
                           const FrameMetadataColumns& frame) {
    for (const auto& [path, col] : frame) {
        const FrameMetadataColumn& stored = file_columns.at(path);
        HighFive::DataSet dataset = file.getDataSet(column_path(path));
        std::vector<std::size_t> dims = dataset.getDimensions();
        std::vector<std::size_t> offset(dims.size(), 0);
        std::vector<std::size_t> count(dims);
        offset[0] = dims[0];
        count[0] = 1;
        dims[0] += 1;
        dataset.resize(dims);
        HighFive::Selection row = dataset.select(offset, count);

        const HighFive::DataType type = column_type(stored);
        switch (col.kind) {
            case Kind::Bool:
                write_row<std::int8_t>(row, type, col.values);
                break;
            case Kind::Int:
                write_row<std::int64_t>(row, type, col.values);
                break;
            case Kind::UInt:
                write_row<std::uint64_t>(row, type, col.values);
                break;
            case Kind::Float:
                write_row<double>(row, type, col.values);
                break;
            case Kind::String: {
                std::vector<char> data(col.values.size() * stored.width, '\0');
                for (std::size_t i = 0; i < col.values.size(); ++i) {
                    const std::string& value = col.values[i].get_ref<const std::string&>();
                    value.copy(&data[i * stored.width], stored.width);
                }
                row.write_raw(data.data(), type);
                break;
            }
        }
    }
}

std::vector<nlohmann::json> read_frame_metadata(const HighFive::File& file) {
    std::vector<nlohmann::json> frames;
    bool have_frames = false;
    read_columns(file.getGroup(frame_metadata_group), {}, frames, have_frames);
    return frames;
}

} // namespace hdf5
