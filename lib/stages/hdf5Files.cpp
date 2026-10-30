#include "hdf5Files.hpp"

#include "CHORDTelescope.hpp" // for CHORDTelescope, dishInputFields
#include "DataType.hpp"       // for DataType, float16_t
#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO, WARN_NON_OO

#include "fmt.hpp" // for format, format_string

#include <algorithm> // for max, min
#include <cassert>   // for assert
#include <cstdint>   // for uint8_t, int64_t, int16_t, int32_t, int8_t, uint16_t, uint32_t
#include <iomanip>   // for operator<<, setfill, setw
#include <sstream>   // for basic_ostream, operator<<, basic_ostringstream
#include <stdexcept> // for runtime_error
#include <string>    // for string
#include <unistd.h>  // for gethostname

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


void create_dataset_with_axis(HighFive::File& file, const std::string& name,
                              const std::vector<hsize_t>& dims,
                              const std::vector<std::string>& dim_names,
                              const HighFive::DataType& dtype, HighFive::DataSetCreateProps props,
                              const std::size_t blocksize_f, const std::size_t blocksize_p,
                              const std::size_t blocksize_t) {

    if (file.exist(name)) {
        WARN_NON_OO("Dataset {} already exists in HDF5 file, not creating again.", name);
        return;
    }
    // size of dims should = size of dim_names, and > 0
    assert(dims.size() == dim_names.size());
    assert(dims.size() > 0);

    if (dims.size() == 1) {
        // if only one dimension, assume a simple array, chunking is just that dimension.
        std::vector<hsize_t> chunk = {dims[0]};
        props.add(HighFive::Chunking(chunk));
    } else {
        // chunking is assumed to be ( blocksize_f, ...array dimensions..., blocksize_t )
        std::vector<hsize_t> chunk = dims;

        // If dimension is "time" (usually last dimension), apply blocksize_t
        // If dimension is "frequency" (usually first dimension), apply blocksize_f
        // If dimension is product or element, use blocksize_p
        for (size_t i = 0; i < dim_names.size(); i++) {
            if (dim_names[i] == "time" && blocksize_t > 0) {
                chunk[i] = std::min<hsize_t>(chunk[i], (hsize_t)blocksize_t);
            } else if (dim_names[i].substr(0, 4) == "freq" && blocksize_f > 0) {
                chunk[i] = std::min<hsize_t>(chunk[i], (hsize_t)blocksize_f);
            } else if ((dim_names[i].substr(0, 4) == "prod" || dim_names[i].substr(0, 2) == "el")
                       && blocksize_p > 0) {
                chunk[i] = std::min<hsize_t>(chunk[i], (hsize_t)blocksize_p);
            }
        }

        for (auto& c : chunk)
            c = std::max<hsize_t>(1, c);

        props.add(HighFive::Chunking(chunk));
    }

    // Create dataset
    HighFive::DataSpace space(dims.begin(), dims.end());
    auto dataset = file.createDataSet(name, space, dtype, props);
    dataset.createAttribute("axis", dim_names);
}

void write_index_map(HighFive::File& file, const CHORDTelescope& telescope,
                     const std::vector<std::uint64_t>& rows, const ElementOrder order,
                     const std::size_t blocksize_p) {
    const HighFive::DataSetCreateProps props_empty = HighFive::DataSetCreateProps::Empty();

    // Polarization index 0 is X, 1 is Y.
    static constexpr std::array<char, 2> pol_name = {'X', 'Y'};

    dishInputFields dishes;
    telescope.fill_input_maps(dishes);

    std::vector<int64_t> dish_idx;
    std::vector<int32_t> pol;
    std::vector<dish_index_t> grid_x_idx;
    std::vector<dish_index_t> grid_y_idx;
    std::vector<vec3d_t> feed_pos_disp_m;
    std::vector<double> coelev_disp_deg;
    std::vector<int32_t> type; // DishType enum stored as int32_t
    std::vector<std::string> label;
    for (const uint64_t el : rows) {
        uint64_t dish;
        uint64_t p;
        telescope.decode_station_id(telescope.element_index_to_station_id(el, order), dish, p);
        if (p >= pol_name.size())
            FATAL_ERROR_NON_OO("write_index_map: element {:d} has polarization {:d}, but only "
                               "{:d} polarizations can be named",
                               el, p, pol_name.size());
        dish_idx.push_back(dish);
        pol.push_back(p);
        grid_x_idx.push_back(dishes.grid_x_idx.at(dish));
        grid_y_idx.push_back(dishes.grid_y_idx.at(dish));
        feed_pos_disp_m.push_back(dishes.feed_pos_disp_m.at(dish));
        coelev_disp_deg.push_back(dishes.coelev_disp_deg.at(dish));
        type.push_back(static_cast<int32_t>(dishes.type.at(dish)));
        label.push_back(fmt::format(fmt("{:s}{:c}"), dishes.label.at(dish), pol_name[p]));
    }
    const hsize_t num_el = rows.size();

    create_dataset_with_axis(file, "/index_map/dish_idx", {num_el}, {"element"},
                             HighFive::create_datatype<int64_t>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/dish_idx").write(dish_idx);

    create_dataset_with_axis(file, "/index_map/pol", {num_el}, {"element"},
                             HighFive::create_datatype<int32_t>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/pol").write(pol);

    create_dataset_with_axis(file, "/index_map/grid_x_idx", {num_el}, {"element"},
                             HighFive::create_datatype<int64_t>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/grid_x_idx").write(grid_x_idx);

    create_dataset_with_axis(file, "/index_map/grid_y_idx", {num_el}, {"element"},
                             HighFive::create_datatype<int64_t>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/grid_y_idx").write(grid_y_idx);

    create_dataset_with_axis(file, "/index_map/feed_pos_disp_m", {num_el, 3}, {"element", "xyz"},
                             HighFive::create_datatype<double>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/feed_pos_disp_m").write(feed_pos_disp_m);

    create_dataset_with_axis(file, "/index_map/coelev_disp_deg", {num_el}, {"element"},
                             HighFive::create_datatype<double>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/coelev_disp_deg").write(coelev_disp_deg);

    create_dataset_with_axis(file, "/index_map/type", {num_el}, {"element"},
                             HighFive::create_datatype<int32_t>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/type").write(type);

    create_dataset_with_axis(file, "/index_map/label", {num_el}, {"element"},
                             HighFive::create_datatype<std::string>(), props_empty, 0, blocksize_p,
                             0);
    file.getDataSet("/index_map/label").write(label);

    // Store full dish positions
    const int num_dishes = telescope.get_num_dishes();
    std::vector<std::array<double, 3>> dish_positions(num_dishes);
    for (int i = 0; i < num_dishes; i++) {
        auto pos = telescope.get_dish_position_in_grid_coords(i);
        dish_positions[i] = {pos[0], pos[1], pos[2]};
    }
    create_dataset_with_axis(file, "/index_map/dish_positions_in_grid_coords",
                             {static_cast<hsize_t>(num_dishes), 3}, {"dish", "xyz"},
                             HighFive::create_datatype<double>(), props_empty, 0, blocksize_p, 0);
    file.getDataSet("/index_map/dish_positions_in_grid_coords").write(dish_positions);
}

} // namespace hdf5
