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
#include <highfive/H5Selection.hpp>              // for Selection, SliceTraits::select
#include <highfive/bits/H5Selection_misc.hpp>    // for Selection::getSpace
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

/**
 * @class hdf5FileRead
 * @brief Read CHORD-metadata HDF5 files written by @c hdf5FileWrite back into a buffer.
 *
 * This is the replay counterpart of @c hdf5FileWrite: it reconstructs both the frame
 * payload and the CHORD metadata (name, type, dimensions, dimension names and scalings,
 * strides, @c fpga_seq_num, @c time_downsampling_fpga, the frequency mapping, and the RFI
 * excision settings) from the dataset and its attributes, and validates the telescope
 * attributes against the telescope of the current session (a mismatch is fatal).
 *
 * Two on-disk layouts are supported, selected by @c read_single_file, the counterpart of the
 * @c create_single_file option of @c hdf5FileWrite:
 *
 * - Per-frame files (default): one file per frame, named
 *   `<input_dir>/[<hostname>_][x<rank:04d>_]<file_name>.<frame:08d>.h5`. The stage reads
 *   files with consecutive indices starting at 0 until one is missing, then stops
 *   producing frames (the pipeline has to be ended by something else, e.g. a
 *   `testDataCheck*` stage or `num_frames` on a downstream stage).
 * - Single file (`read_single_file: true`): all frames concatenated along axis 0 of one
 *   dataset in `<input_dir>/[<hostname>_][x<rank:04d>_]<file_name>.h5`.
 *
 * @par Reading a single file
 * The single-file format stores the metadata attributes only once (they are taken from the
 * first frame), so neither the per-frame metadata (@c frame_counter,
 * @c first_packet_recv_time, the per-frame @c fpga_seq_num) nor the frame boundary is on
 * disk. The replay therefore reconstructs them:
 * - the axis-0 extent of one frame is the first extent declared for @c out_buf in the
 *   config (all other extents, the value type and the labels must match the file);
 * - `fpga_seq_num(i) = fpga_seq_num + i * extents[0] * time_downsampling_fpga`, using a
 *   downsampling factor of 1 if the attribute is absent;
 * - the number of frames is `axis0 / extents[0]`; a remainder is reported and ignored.
 *
 * @par Sessions
 * The writer must have finished before the reader starts: run the two in separate kotekan
 * sessions. A missing input is fatal at frame index 0 (for a single file: a missing or
 * unreadable file is always fatal), so the first per-frame file, or the whole single file,
 * has to exist and be complete at startup. If a SWMR writer was killed the file may need
 * `h5clear -s FILE.h5` before it can be opened.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill with the frames read from the file(s)
 *         @buffer_format any format, declared as `kotekan_buffer: ndarray`
 *         @buffer_metadata chordMetadata
 *
 * @conf  input_dir             String. Required. Directory holding the input file(s).
 * @conf  file_name             String. Required. Base name of the file(s), and the name of
 *                              the dataset inside them.
 * @conf  prefix_hostname       Bool. Default: true. Expect the hostname and an underscore
 *                              in front of @c file_name.
 * @conf  prefix_host_rank      Bool. Default: false. Expect `x<rank:04d>_` in front of
 *                              @c file_name, using @c frequency_pool_rank.
 * @conf  frequency_pool_rank   Int. Default: 0. The rank used by @c prefix_host_rank.
 * @conf  do_once               Bool. Default: false. Read only the first frame and then
 *                              idle instead of reading the whole input.
 * @conf  read_single_file      Bool. Default: false. Read all frames from a single file
 *                              instead of one file per frame; see above.
 * @conf  num_polarizations     Int. Required. Cross-checked against the file.
 * @conf  num_dishes            Int. Required. Cross-checked against the file.
 *
 * @par Metrics
 * @metric kotekan_hdf5fileread_read_time_seconds Time required to read the last frame.
 */
class hdf5FileRead : public kotekan::Stage {
    const std::string input_dir = config.get<std::string>(unique_name, "input_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    const bool prefix_host_rank = config.get_default<bool>(unique_name, "prefix_host_rank", false);
    const int host_pool_rank = config.get_default<int>(unique_name, "frequency_pool_rank", 0);
    const bool do_once = config.get_default<bool>(unique_name, "do_once", false);
    const bool read_single_file = config.get_default<bool>(unique_name, "read_single_file", false);

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
            const auto& expected_dish_grid_indices = telescope.get_main_array_grid_indices(
                num_dishes, telescope.fiducial_element_order());
            if (dish_grid_indices != expected_dish_grid_indices)
                FATAL_ERROR("Attribute dish_grid_indices is {}, expected {}", dish_grid_indices,
                            expected_dish_grid_indices);
        }

        if (node.hasAttribute("feed_positions_m")) {
            auto attr = node.getAttribute("feed_positions_m");
            auto space = attr.getSpace();
            auto dims = space.getDimensions(); // {N, 3}
            assert(dims.size() == 2);
            assert(dims.at(1) == 3);
            std::vector<std::array<double, 3>> feed_positions_m(dims.at(0));
            attr.read_raw(reinterpret_cast<double*>(feed_positions_m.data()));
            const auto& expected_feed_positions_m =
                telescope.get_feed_positions_m(num_dishes, telescope.fiducial_element_order());
            if (feed_positions_m != expected_feed_positions_m)
                FATAL_ERROR("Attribute feed_positions_m is {}, expected {}", feed_positions_m,
                            expected_feed_positions_m);
        }
    }

    /**
     * @brief The common part of all file names of this stage:
     *        "<input_dir>/[<hostname>_][x<rank:04d>_]<file_name>"
     *
     * The caller appends the suffix, i.e. ".<frame:08d>.h5" for per-frame files and ".h5"
     * for a single file.
     */
    std::string file_path_prefix() const {
        std::ostringstream buf;
        buf << input_dir << "/";
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

    /**
     * @brief Check that this reader understands the metadata format of the dataset
     *
     * The major version has to match, and the file's minor version must not be newer than
     * ours. A mismatch is fatal.
     */
    void check_metadata_version(const DataSet& dataset) const {
        const auto version =
            dataset.getAttribute("chord_metadata_version").read<std::array<int, 2>>();
        if (version[0] != chord_metadata_version[0] || version[1] > chord_metadata_version[1])
            FATAL_ERROR("Dataset \"{:s}\" has chord_metadata_version {:d}.{:d}; this reader "
                        "understands {:d}.{:d}",
                        file_name, version[0], version[1], chord_metadata_version[0],
                        chord_metadata_version[1]);
    }

    /**
     * @brief Fill @p meta from the attributes of @p dataset
     *
     * @param dataset      The dataset that is being read
     * @param meta         The metadata of the frame that is being filled
     * @param frame_dims   The shape of ONE frame: the dataset shape for per-frame files,
     *                     with axis 0 replaced by the buffer's frame extent in single-file
     *                     mode
     * @param frame_index  The index of the frame within the dataset. It advances
     *                     @c fpga_seq_num in single-file mode, where only the value of the
     *                     first frame is stored on disk; it must be 0 for per-frame files.
     */
    void set_metadata(const DataSet& dataset, const std::shared_ptr<chordMetadata>& meta,
                      const std::vector<std::size_t>& frame_dims,
                      const std::int64_t frame_index) const {
        meta->set_name(dataset.getAttribute("name").read<std::string>());
        meta->type = kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
        if (frame_dims.size() > CHORD_META_MAX_DIM)
            FATAL_ERROR("Dataset \"{:s}\" has rank {:d} > CHORD_META_MAX_DIM ({:d})", file_name,
                        frame_dims.size(), std::size_t(CHORD_META_MAX_DIM));
        meta->dims = int(frame_dims.size());
        const auto dim_names = dataset.getAttribute("dim_names").read<std::vector<std::string>>();
        const auto dim_scalings =
            dataset.getAttribute("dim_scalings").read<std::vector<std::ptrdiff_t>>();
        if (dim_names.size() != frame_dims.size() || dim_scalings.size() != frame_dims.size())
            FATAL_ERROR("Dataset \"{:s}\": dim_names ({:d}) and dim_scalings ({:d}) do not match "
                        "rank {:d}",
                        file_name, dim_names.size(), dim_scalings.size(), frame_dims.size());
        for (int d = 0; d < meta->dims; ++d)
            meta->set_array_dimension(d, frame_dims.at(d), dim_names.at(d), dim_scalings.at(d));
        meta->set_strides_simple();
        meta->offset = 0;

        if (dataset.hasAttribute("fpga_seq_num")) {
            std::int64_t fpga_seq_num = dataset.getAttribute("fpga_seq_num").read<std::int64_t>();
            if (frame_index > 0) {
                const int tds = dataset.hasAttribute("time_downsampling_fpga")
                                    ? dataset.getAttribute("time_downsampling_fpga").read<int>()
                                    : 1;
                fpga_seq_num += frame_index * std::int64_t(frame_dims.at(0)) * tds;
            }
            meta->set_fpga_seq_num(fpga_seq_num);
        }
        if (dataset.hasAttribute("time_downsampling_fpga"))
            meta->set_time_downsampling_fpga(
                dataset.getAttribute("time_downsampling_fpga").read<int>());

        if (dataset.hasAttribute("coarse_freq"))
            meta->set_coarse_freq(dataset.getAttribute("coarse_freq").read<std::vector<int>>());
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
    }

    /**
     * @brief Validate the buffer's declared frame descriptor against the file
     *
     * This fills in the labels that the config left unset, and is fatal on a structural or
     * label conflict (see @c Buffer::require_frame_desc).
     *
     * @param dataset     The dataset that is being read
     * @param frame_dims  The shape of ONE frame (see @c set_metadata)
     */
    void require_frame_desc_from(const DataSet& dataset,
                                 const std::vector<std::size_t>& frame_dims) const {
        const kotekan::DataType value_type =
            kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
        if (value_type == kotekan::unknown_type)
            FATAL_ERROR("Dataset \"{:s}\" has unknown data type \"{:s}\"", file_name,
                        dataset.getAttribute("type").read<std::string>());
        const std::string name = dataset.getAttribute("name").read<std::string>();
        const auto dim_names = dataset.getAttribute("dim_names").read<std::vector<std::string>>();
        const auto dim_scalings =
            dataset.getAttribute("dim_scalings").read<std::vector<std::ptrdiff_t>>();
        const std::vector<std::ptrdiff_t> dimensions(frame_dims.begin(), frame_dims.end());
        const std::vector<kotekan::Symbol> dimnames(dim_names.begin(), dim_names.end());
        const std::vector<std::ptrdiff_t> dimscalings(dim_scalings.begin(), dim_scalings.end());
        buffer->require_frame_desc(
            kotekan::GenericNDArray::describe(value_type, name, dimensions, dimnames, dimscalings));
    }

    void main_thread() override {
        if (read_single_file)
            read_single_file_frames();
        else
            read_per_frame_files();
    }

    /**
     * @brief Read one file per frame, with consecutively numbered file names
     *
     * Reading stops when a file is missing; a missing file at index 0 is fatal.
     */
    void read_per_frame_files() {
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
            buf << file_path_prefix() << "." << std::setw(8) << std::setfill('0') << frame_index
                << ".h5";
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

                check_metadata_version(dataset);

                set_metadata(dataset, meta, dims, 0);

                // Read telescope fields and abort if they don't match
                check_telescope_metadata(dataset);

                /* new style array description */
                require_frame_desc_from(dataset, dims);
                /* test that things are consistent */
                meta->check_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());

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

    /**
     * @brief Read all frames from a single file, splitting the dataset along axis 0
     *
     * The per-frame axis-0 extent is taken from the buffer's declared frame descriptor
     * because the file only stores the concatenation of all frames. A missing or
     * unreadable file is always fatal.
     */
    void read_single_file_frames() {
        auto& read_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_hdf5fileread_read_time_seconds", unique_name);
        const std::string full_path = file_path_prefix() + ".h5";
        try {
            const File file(full_path, File::ReadOnly);
            const auto dataset = file.getDataSet(file_name);
            const auto type = dataset.getDataType();
            const auto dims = dataset.getSpace().getDimensions();
            if (dims.empty())
                FATAL_ERROR("Dataset \"{:s}\" in {:s} has rank 0", file_name, full_path);
            check_metadata_version(dataset);

            // One frame's axis-0 extent comes from the destination buffer declaration;
            // the file only records the concatenation of all frames.
            const std::size_t frame_dim0 = std::size_t(
                buffer->require_frame_desc<kotekan::GenericNDArray>()->get_extents().at(0));
            if (frame_dim0 == 0)
                FATAL_ERROR("Buffer {:s} declares a zero-length first axis", buffer->buffer_name);
            std::vector<std::size_t> frame_dims(dims);
            frame_dims.at(0) = frame_dim0;
            require_frame_desc_from(dataset, frame_dims); // type, other extents, labels, bytes
            check_telescope_metadata(dataset);

            const std::int64_t num_frames = std::int64_t(dims.at(0) / frame_dim0);
            if (dims.at(0) % frame_dim0 != 0)
                WARN("{:s}: axis 0 ({:d}) is not a multiple of the frame extent ({:d}); "
                     "ignoring the trailing {:d} samples",
                     full_path, dims.at(0), frame_dim0, dims.at(0) % frame_dim0);
            INFO("Reading {:d} frames of {:d} samples each from {:s}", num_frames, frame_dim0,
                 full_path);

            for (std::int64_t frame_index = 0; frame_index < num_frames; ++frame_index) {
                if (stop_thread)
                    break;
                const int frame_id = int(frame_index % buffer->num_frames);
                const double t0 = current_time();
                DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
                std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
                if (!frame)
                    break;
                buffer->allocate_new_metadata_object(frame_id);
                const std::shared_ptr<metadataObject> metadata = buffer->get_metadata(frame_id);
                if (!metadata || !metadata_is_chord(metadata))
                    FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD",
                                buffer->buffer_name, frame_id);
                const std::shared_ptr<chordMetadata> meta = get_chord_metadata(metadata);
                set_metadata(dataset, meta, frame_dims, frame_index);
                meta->check_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());

                // Read this frame's hyperslab
                DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
                std::vector<std::size_t> offset(dims.size(), 0);
                offset.at(0) = std::size_t(frame_index) * frame_dim0;
                dataset.select(offset, frame_dims).read_raw(frame, type);

                DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
                buffer->mark_frame_full(unique_name, frame_id);
                read_time_metric.set(current_time() - t0);
                if (do_once) {
                    while (!stop_thread)
                        sleep(1);
                    break;
                }
            }
            INFO("Done reading {:d} frames from {:s}", num_frames, full_path);
        } catch (const HighFive::Exception& ex) {
            FATAL_ERROR("Could not read HDF5 file {:s}: {:s}", full_path, ex.what());
        }
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileRead);
