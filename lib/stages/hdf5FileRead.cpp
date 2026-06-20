#include "fmt.hpp" // for compile_string_to_view

#include <Config.hpp>          // for Config
#include <DataType.hpp>        // for string_to_type, DataType
#include <NDArray.hpp>         // for GenericNDArray
#include <Stage.hpp>           // for Stage
#include <StageFactory.hpp>    // for REGISTER_KOTEKAN_STAGE
#include <Symbol.hpp>          // for Symbol
#include <algorithm>           // for copy
#include <array>               // for array
#include <buffer.hpp>          // for Buffer
#include <bufferContainer.hpp> // for bufferContainer
#include <cassert>             // for assert
#include <chordMetadata.hpp>   // for chordMetadata, metadata_is_chord, get_c...
#include <cstddef>             // for ptrdiff_t
#include <cstdint>             // for int64_t, uint8_t
#include <fmt/ranges.h>
#include <functional>                            // for function
#include <hdf5Files.hpp>                         // for chord_metadata_version
#include <highfive/H5Attribute.hpp>              // for Attribute, Attribute::read
#include <highfive/H5DataSet.hpp>                // for DataSet, AnnotateTraits::getAttribute
#include <highfive/H5DataSpace.hpp>              // for DataSpace, DataSpace::getDimensions
#include <highfive/H5Exception.hpp>              // for FileException
#include <highfive/H5File.hpp>                   // for File, File::File, NodeTraits::getDataSet
#include <highfive/bits/H5Slice_traits_misc.hpp> // for SliceTraits::read_raw
#include <iomanip>                               // for operator<<, setfill, setw
#include <kotekanLogging.hpp>                    // for DEBUG, FATAL_ERROR, ERROR
#include <memory>                                // for allocator, shared_ptr, __shared_ptr_access
#include <metadata.hpp>                          // for metadataObject
#include <prometheusMetrics.hpp>                 // for Metrics, Gauge
#include <sstream>                               // for basic_ostream, operator<<, basic_ostrin...
#include <string>                                // for basic_string, char_traits, string, oper...
#include <unistd.h>                              // for gethostname, sleep
#include <vector>                                // for vector
#include <visUtil.hpp>                           // for current_time

using namespace hdf5;
using namespace HighFive;

class hdf5FileRead : public kotekan::Stage {
    const std::string input_dir = config.get<std::string>(unique_name, "input_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    const bool prefix_host_rank = config.get_default<bool>(unique_name, "prefix_host_rank", false);
    const int host_pool_rank = config.get_default<int>(unique_name, "frequency_pool_rank", 0);
    const int host_pool_size = config.get_default<int>(unique_name, "frequency_pool_size", 1);
    const bool do_once = config.get_default<bool>(unique_name, "do_once", false);

    const uint64_t num_polarizations = config.get<uint64_t>(unique_name, "num_polarizations");
    const uint64_t num_dishes = config.get<uint64_t>(unique_name, "num_dishes");

    Buffer* const buffer;

public:
    hdf5FileRead(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("out_buf")) {
        assert(buffer);
        buffer->register_producer(unique_name);
    }

    virtual ~hdf5FileRead() {}

    /**
     * @brief Read and check the telescope metadata
     *
     * @param node   The HDF5 object (file, group, dataset, etc.) where the metadata should be
     *               read from
     */
    template<typename Node>
    void check_telescope_metadata(Node& node) {
        const auto& telescope = Telescope::instance();

        const auto check_string_attr = [&](const std::string& key,
                                           const std::string& expected_value) {
            if (node.hasAttribute(key)) {
                const auto value = node.getAttribute(key).template read<std::string>();
                if (value != expected_value)
                    FATAL_ERROR("Attribute {:s} is \"{}\", expected \"{}\"", key, value,
                                expected_value);
            }
        };

        const auto check_float_attr = [&](const std::string& key, const double expected_value) {
            if (node.hasAttribute(key)) {
                const auto value = node.getAttribute(key).template read<double>();
                if (value != expected_value)
                    FATAL_ERROR("Attribute {:s} is {}, expected {}", key, value, expected_value);
            }
        };

        const auto check_int_attr = [&](const std::string& key, const std::int64_t expected_value) {
            if (node.hasAttribute(key)) {
                const auto value = node.getAttribute(key).template read<std::int64_t>();
                if (value != expected_value)
                    FATAL_ERROR("Attribute {:s} is {}, expected {}", key, value, expected_value);
            }
        };

        const auto check_uint_attr = [&](const std::string& key,
                                         const std::uint64_t expected_value) {
            if (node.hasAttribute(key)) {
                const auto value = node.getAttribute(key).template read<std::uint64_t>();
                if (value != expected_value)
                    FATAL_ERROR("Attribute {:s} is {}, expected {}", key, value, expected_value);
            }
        };

        const auto check_bool_attr = [&](const std::string& key, const bool expected_value) {
            if (node.hasAttribute(key)) {
                const auto value = node.getAttribute(key).template read<bool>();
                if (value != expected_value)
                    FATAL_ERROR("Attribute {:s} is {}, expected {}", key, value, expected_value);
            }
        };

        check_string_attr("telescope_name", telescope.get_name());
        check_uint_attr("seq_length_nsec", telescope.seq_length_nsec());
        check_bool_attr("gps_time_enabled", telescope.gps_time_enabled());
        check_int_attr("num_polarizations", num_polarizations);
        check_int_attr("num_dishes", num_dishes);
        check_float_attr("itrs_lat_deg", telescope.get_itrs_lat_deg());
        check_float_attr("itrs_lon_deg", telescope.get_itrs_lon_deg());
        // node.getAttribute("grid_orientation", telescope.get_grid_orientation());
        check_uint_attr("grid_size_x", telescope.get_grid_size_x());
        check_uint_attr("grid_size_y", telescope.get_grid_size_y());
        check_float_attr("feed_separation_x_m", telescope.get_feed_separation_x_m());
        check_float_attr("feed_separation_y_m", telescope.get_feed_separation_y_m());

        if (node.hasAttribute("dish_grid_indices")) {
            auto attr = node.getAttribute("dish_grid_indices");
            auto space = attr.getSpace();
            auto dims = space.getDimensions(); // {N, 2}
            assert(dims.size() == 2);
            assert(dims.at(1) == 2);
            std::vector<std::array<std::int64_t, 2>> dish_grid_indices(dims.at(0));
            attr.read_raw(reinterpret_cast<std::int64_t*>(dish_grid_indices.data()));
            const auto& expected_dish_grid_indices =
                telescope.get_main_array_grid_indices(num_dishes, ElementOrder::CHORDBeamformer);
            if (dish_grid_indices != expected_dish_grid_indices)
                FATAL_ERROR("Attribute dish_grid_indices is {}, expected {}", dish_grid_indices,
                            expected_dish_grid_indices);
        }

        if (node.hasAttribute("feed_positions_m")) {
            auto attr = node.getAttribute("feed_positions_m");
            auto space = attr.getSpace();
            auto dims = space.getDimensions(); // {N, 3}
            assert(dims.size() == 3);
            assert(dims.at(1) == 3);
            std::vector<std::array<double, 3>> feed_positions_m(dims.at(0));
            attr.read_raw(reinterpret_cast<double*>(feed_positions_m.data()));
            const auto& expected_feed_positions_m =
                telescope.get_feed_positions_m(num_dishes, ElementOrder::CHORDBeamformer);
            if (feed_positions_m != expected_feed_positions_m)
                FATAL_ERROR("Attribute feed_positions_m is {}, expected {}", feed_positions_m,
                            expected_feed_positions_m);
        }
    }

    void main_thread() override {
        auto& read_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_hdf5fileread_read_time_seconds", unique_name);

        for (int frame_index = 0;; ++frame_index) {
            const int frame_id = frame_index % buffer->num_frames;

        wait:

            if (stop_thread)
                break;

            if (do_once && frame_index > 0) {
                sleep(1);
                goto wait;
            }

            // Start timer
            const double t0 = current_time();

            // Define file name
            std::ostringstream buf;
            buf << input_dir << "/";
            if (prefix_hostname) {
                char hostname[256];
                gethostname(hostname, sizeof hostname);
                buf << hostname << "_";
            }
            if (prefix_host_rank) {
                buf << "x" << std::setw(4) << std::setfill('0') << host_pool_rank << "_";
            }
            buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_index << ".h5";
            const std::string full_path = buf.str();

            // Open HDF5 file
            try {
                const File file(full_path, File::ReadOnly);

                // Wait for buffer
                DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
                std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
                if (!frame)
                    break;

                // Read metadata (attributes)
                buffer->allocate_new_metadata_object(frame_id);
                const std::shared_ptr<metadataObject> metadata = buffer->get_metadata(frame_id);
                if (!metadata)
                    FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                                buffer->buffer_name, frame_id);
                assert(metadata);
                if (!metadata_is_chord(metadata))
                    FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD",
                                buffer->buffer_name, frame_id);
                assert(metadata_is_chord(metadata));
                const std::shared_ptr<chordMetadata> meta = get_chord_metadata(metadata);
                assert(meta);

                // Open dataset
                const auto dataset = file.getDataSet(file_name);
                const auto space = dataset.getSpace();
                const auto type = dataset.getDataType();
                const auto dims = space.getDimensions();

                {
                    const auto metadata_version =
                        dataset.getAttribute("chord_metadata_version").read<std::array<int, 2>>();
                    const int major = metadata_version[0];
                    const int minor = metadata_version[1];
                    assert(major >= 0);
                    assert(minor >= 0);
                    assert(major == chord_metadata_version.at(0));
                    assert(minor <= chord_metadata_version.at(1));
                }

                meta->set_name(dataset.getAttribute("name").read<std::string>());
                meta->type =
                    kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
                meta->dims = space.getNumberDimensions();
                assert(meta->dims <= CHORD_META_MAX_DIM);
                const auto dim_names =
                    dataset.getAttribute("dim_names").read<std::vector<std::string>>();
                assert(std::ptrdiff_t(dim_names.size()) == meta->dims);
                for (int d = 0; d < meta->dims; ++d)
                    meta->set_array_dimension(d, dims.at(d), dim_names.at(d));
                {
                    std::ptrdiff_t npoints = 1;
                    for (int d = meta->dims - 1; d >= 0; --d) {
                        meta->stride[d] = npoints;
                        npoints *= meta->dim[d];
                    }
                    assert(std::ptrdiff_t(space.getElementCount()) == npoints);
                }
                meta->offset = 0;

                if (dataset.hasAttribute("fpga_seq_num"))
                    meta->set_fpga_seq_num(
                        dataset.getAttribute("fpga_seq_num").read<std::int64_t>());
                if (dataset.hasAttribute("time_downsampling_fpga"))
                    meta->set_time_downsampling_fpga(
                        dataset.getAttribute("time_downsampling_fpga").read<int>());

                if (dataset.hasAttribute("coarse_freq"))
                    meta->set_coarse_freq(
                        dataset.getAttribute("coarse_freq").read<std::vector<int>>());
                if (dataset.hasAttribute("freq_upchan_factor"))
                    meta->set_freq_upchan_factor(
                        dataset.getAttribute("freq_upchan_factor").read<std::vector<int>>());
                if (dataset.hasAttribute("freq_upchan_index"))
                    meta->set_freq_upchan_index(
                        dataset.getAttribute("freq_upchan_index").read<std::vector<int>>());

                if (dataset.hasAttribute("rfi_frame_excision_enabled"))
                    meta->set_rfi_frame_excision_enabled(
                        dataset.getAttribute("rfi_frame_excision_enabled").read<bool>());
                if (dataset.hasAttribute("rfi_frame_excision_thresholds"))
                    meta->set_rfi_frame_excision_thresholds(
                        dataset.getAttribute("rfi_frame_excision_thresholds")
                            .read<std::vector<std::array<float, 2>>>());

                // Read telescope fields and abort if they don't match
                check_telescope_metadata(dataset);

                {
                    /* new style array description */
                    const kotekan::DataType value_type =
                        kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
                    assert(value_type != kotekan::unknown_type);
                    const std::string name = dataset.getAttribute("name").read<std::string>();

                    std::vector<ptrdiff_t> dimensions(dims.begin(), dims.end());
                    std::vector<kotekan::Symbol> dimnames(dim_names.begin(), dim_names.end());

                    buffer->require_frame_desc(
                        kotekan::GenericNDArray::describe(value_type, name, dimensions, dimnames));
                    /* test that things are consistent */
                    meta->check_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());
                }

                // Read buffer
                DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
                dataset.read_raw(frame, type);

                // Mark buffer as full
                DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
                buffer->mark_frame_full(unique_name, frame_id);

                // Stop timer
                const double t1 = current_time();
                const double elapsed = t1 - t0;
                read_time_metric.set(elapsed);
            } catch (const FileException& ex) {
                if (frame_index == 0)
                    FATAL_ERROR("Could not open HDF5 file {:s}: {:s}", full_path, ex.what());
                else
                    ERROR("Could not open HDF5 file {:s}, terminating reader", full_path);
                break;
            }

        } // while !stop_thread
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileRead);
