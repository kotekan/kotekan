#include "CHORDTelescope.hpp"      // for EOP
#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for DataType, type_to_string
#include "N2FrameView.hpp"         // for N2FrameView
#include "N2Metadata.hpp"          // for metadata_is_N2
#include "NDArray.hpp"             // for GenericNDArray
#include "Stage.hpp"               // for Stage
#include "StageFactory.hpp"        // for REGISTER_KOTEKAN_STAGE
#include "Symbol.hpp"              // for Symbol
#include "Telescope.hpp"           //
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata, metadata_is_chord, get_c...
#include "errors.h"                // for exit_kotekan, ReturnCode
#include "hdf5Files.hpp"           // for chord2hdf5, BITSHUFFLE_BLOCKSIZE_AUTO
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR, WARN, INFO
#include "metadata.hpp"            // for metadataObject
#include "prometheusMetrics.hpp"   // for Metrics, Gauge
#include "timeUtil.hpp"            //
#include "visUtil.hpp"             // for current_time
#include "waitingForMaxFrames.hpp" // for waiting_for_max_frames

#include <algorithm>             // for max, min
#include <array>                 // for array
#include <atomic>                // for __atomic_base, atomic
#include <cassert>               // for assert
#include <cstddef>               // for size_t, ptrdiff_t
#include <cstdint>               // for uint8_t, int64_t, uint32_t
#include <errno.h>               // for errno, EEXIST, EISDIR
#include <fmt.hpp>               // for compile_string_to_view
#include <functional>            // for function
#include <gsl-lite.hpp>          // for span
#include <highfive/highfive.hpp> //
#include <iomanip>               // for operator<<, setfill, setw
#include <memory>                // for allocator, shared_ptr, __shared_ptr_access
#include <sstream>               // for basic_ostream, operator<<, basic_ostrin...
#include <string.h>              // for strerror
#include <string>                // for basic_string, char_traits, string, oper...
#include <sys/stat.h>            // for mkdir
#include <unistd.h>              // for gethostname
#include <vector>                // for vector


using namespace hdf5;
using namespace HighFive;

/**
 * @class hdf5FileWrite
 * @brief Stream a buffer to disk.
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format Any
 *     @buffer_metadata Any
 *
 * @conf base_dir  String. Directory to write into.
 * @conf file_name String. Base filename to write.
 * @conf prefix_hostname  Bool. Prepend hostname to output file names. Default:
 *       true.
 * @conf max_frames  Int. Stop writing after this many frames, Default 0 = unlimited
 *       frames.
 * @conf skip_writing  Bool. Do not actually write anything. Default:
 *       false.
 *
 * @par Metrics
 * @metric kotekan_hdf5filewrite_write_time_seconds
 *         The write time to write out the last frame.
 *
 * @author Erik Schnetter
 **/
class hdf5FileWrite : public kotekan::Stage {

    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    const bool prefix_host_rank = config.get_default<bool>(unique_name, "prefix_host_rank", false);
    const int host_pool_rank = config.get_default<int>(unique_name, "frequency_pool_rank", 0);
    const int host_pool_size = config.get_default<int>(unique_name, "frequency_pool_size", 1);

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const bool skip_writing = config.get_default<bool>(unique_name, "skip_writing", false);
    const bool create_single_file =
        config.get_default<bool>(unique_name, "create_single_file", false);

    Buffer* const buffer;

    std::shared_ptr<File> the_single_file;

public:
    hdf5FileWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("in_buf")) {

        if (max_frames >= 0)
            ++waiting_for_max_frames;

        buffer->register_consumer(unique_name);
    }

    virtual ~hdf5FileWrite() {}

    // Create the HDF5 file.
    // Determine the file name, and create the containing directory if necessary.
    std::shared_ptr<File> create_file(const std::int64_t frame_counter) const {
        // Define file name
        std::ostringstream buf;
        buf << base_dir << "/";
        if (prefix_hostname) {
            char hostname[256];
            gethostname(hostname, sizeof hostname);
            buf << hostname << "_";
        }
        if (prefix_host_rank) {
            buf << "x" << std::setw(4) << std::setfill('0') << host_pool_rank << "_";
        }
        buf << file_name;
        if (create_single_file) {
            // Do not include the frame counter when all output is written to a single file
            if (frame_counter >= 0)
                FATAL_ERROR("Internal error -- cannot handle a non-negative frame_counter when "
                            "creating a single_file HDF5 file with base name \"{:s}\"",
                            file_name);
        } else {
            if (frame_counter < 0)
                FATAL_ERROR("Internal error -- cannot handle a negative frame_counter when a "
                            "regular (non single_file) HDF5 file with base name \"{:s}\"",
                            file_name);
            buf << "." << std::setw(8) << std::setfill('0') << frame_counter;
        }
        buf << ".h5";
        const std::string full_path = buf.str();

        // Create directory if necessary
        int ierr = mkdir(base_dir.c_str(), 0777);
        if (ierr) {
            if (errno != EEXIST && errno != EISDIR) {
                const char* const msg = strerror(errno);
                FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(), msg);
            }
        }

        // Create HDF5 file
        FileAccessProps fapl;
        // We need at least version 1.10 version for SWMR.
        // (That's the file format version, not the library version.)
        fapl.add(FileVersionBounds(H5F_LIBVER_V110, H5F_LIBVER_V110));
        HighFive::FileCreateProps fprops;
        // Use SWMR. This keeps HDF5 files readable even if they were not properly closed.
        // You may have to use `h5clear -s FILENAME.h5` to "tell" the file that the writer does not
        // exist any more.
        return std::make_unique<File>(full_path, File::Truncate | File::WriteSWMR, fprops, fapl);
    }

    /**
     * @brief Create and write a file for a frame with chordMetadata
     *
     * @param full_path The full path, including file name, for the file.
     * @param frame uint8_t pointer to the frame data
     * @param meta shared pointer to the chordMetadata object for this frame.
     * @param frame_desc shared pointer to the NDarray description for this frame.
     */
    void write_chord(const std::uint8_t* const frame,
                     const std::shared_ptr<const chordMetadata>& meta,
                     const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc,
                     const std::int64_t frame_counter) {
        const auto& telescope = Telescope::instance();

        std::shared_ptr<File> fileptr;
        bool do_create_dataset = false; // Whether to create the dataset and write the attributes
        if (create_single_file) {
            // Create only a single file
            if (!the_single_file) {
                the_single_file = create_file(-1);
                do_create_dataset = true;
            }
            fileptr = the_single_file;
        } else {
            // Create a new file for every frame
            fileptr = create_file(frame_counter);
            do_create_dataset = true;
        }
        auto& file = *fileptr;

        if (do_create_dataset) {

            // Choose dataspace
            const auto extents = frame_desc->get_extents();
            std::vector<std::size_t> initial_shape(extents.begin(), extents.end());
            std::vector<std::size_t> maximum_shape(extents.begin(), extents.end());
            initial_shape.at(0) = 0;
            maximum_shape.at(0) = DataSpace::UNLIMITED;
            const DataSpace space(initial_shape, maximum_shape);
            assert(meta->offset == 0);
            assert(space.getNumberDimensions() == frame_desc->get_rank());

            // Choose datatype
            const DataType type = chord2hdf5(frame_desc->get_value_datatype());

            RawPropertyList<PropertyType::DATASET_CREATE> props;

            // Enable chunking
            std::vector<hsize_t> chunk_dims(extents.begin(), extents.end());
            if (!chunk_dims.empty()) {
                // Choose chunk size
                std::size_t npoints_lo = 1;
                for (std::size_t d = 1; d < chunk_dims.size(); ++d)
                    npoints_lo *= chunk_dims.at(d);
                std::size_t npoints_hi = 8 * 1024 * 1024 / npoints_lo;
                using std::max, std::min;
                npoints_hi = min(std::size_t(chunk_dims.at(0)), max(std::size_t(1), npoints_hi));
                chunk_dims.at(0) = npoints_hi;
            }
            (*(DataSetCreateProps*)&props).add(Chunking(chunk_dims));

            // // Enable compression
            // constexpr int blosc_compression_level = 9;
            // const std::vector<unsigned int> blosc_flags{
            //     blosc_compression_level,
            //     BLOSC_SHUFFLE_BIT,
            //     BLOSC_COMPRESS_ZSTD,
            // };
            // props.add(H5Pset_filter, H5Z_BLOSC, H5Z_FLAG_MANDATORY, blosc_flags.size(),
            //           blosc_flags.data());
            constexpr int bitshuffle_compression_level = 9;
            const std::vector<unsigned int> bitshuffle_flags{
                BITSHUFFLE_BLOCKSIZE_AUTO,
                BITSHUFFLE_COMPRESS_ZSTD,
                bitshuffle_compression_level,
            };
            props.add(H5Pset_filter, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY, bitshuffle_flags.size(),
                      bitshuffle_flags.data());

            // Create dataset
            auto dataset = file.createDataSet(file_name, space, type, props);

            // Write metadata (attributes)

            dataset.createAttribute("telescope_name", telescope.get_name());
            dataset.createAttribute("seq_length_nsec", telescope.seq_length_nsec());
            dataset.createAttribute("gps_time_enabled", telescope.gps_time_enabled());

            dataset.createAttribute("chord_metadata_version", chord_metadata_version);
            dataset.createAttribute("name", meta->get_name());
            dataset.createAttribute("type",
                                    kotekan::type_to_string(frame_desc->get_value_datatype()));
            const auto dimnames = frame_desc->get_dimnames();
            std::vector<std::string> dim_names(dimnames.begin(), dimnames.end());
            dataset.createAttribute("dim_names", dim_names);

            if (meta->has_fpga_seq_num()) {
                dataset.createAttribute("fpga_seq_num", meta->get_fpga_seq_num());
                dataset.createAttribute("fpga_seq_time_nsec",
                                        telescope.to_time_ns(meta->get_fpga_seq_num()));
            }

            if (meta->has_time_downsampling_fpga())
                dataset.createAttribute("time_downsampling_fpga",
                                        meta->get_time_downsampling_fpga());

            if (meta->has_coarse_freq())
                dataset.createAttribute("coarse_freq", meta->get_coarse_freq());

            if (meta->has_freq_upchan_factor())
                dataset.createAttribute("freq_upchan_factor", meta->get_freq_upchan_factor());

            if (meta->has_freq_upchan_index())
                dataset.createAttribute("freq_upchan_index", meta->get_freq_upchan_index());

            if (meta->ndishes >= 0) {
                dataset.createAttribute("ndishes", meta->ndishes);
                // const DataSpace space{std::size_t(meta->n_dish_locations_ns),
                //                       std::size_t(meta->n_dish_locations_ew)};
                // auto attr = dataset.createAttribute<int>("dish_index", space);
                // attr.write(meta->dish_index);
                dataset.createAttribute("n_dish_locations_ns", meta->n_dish_locations_ns);
                dataset.createAttribute("n_dish_locations_ew", meta->n_dish_locations_ew);
                dataset.createAttribute(
                    "dish_index",
                    std::vector<int>(meta->dish_index,
                                     meta->dish_index
                                         + meta->n_dish_locations_ns * meta->n_dish_locations_ew));
            }

            if (create_single_file) {
                // Start SWMR mode
                // (Can only do that after creating the dataset and writing all attributes.)
                file.flush(); // for good measure
                file.startSWMRWrite();
            }

        } // if do_create_dataset

        // Get datatype
        const DataType type = chord2hdf5(frame_desc->get_value_datatype());

        // Get the dataset
        auto dataset = file.getDataSet(file_name);

        // Extend the dataset
        const auto dataspace = dataset.getSpace();
        const auto old_dims = dataspace.getDimensions();
        std::vector<std::size_t> new_dims(old_dims);
        assert(meta->dims > 0);
        new_dims.at(0) = old_dims.at(0) + meta->dim[0];
        dataset.resize(new_dims);

        // Choose hpyerslab
        std::vector<std::size_t> offset(new_dims.size(), 0);
        std::vector<std::size_t> count(new_dims);
        offset.at(0) = old_dims.at(0);
        count.at(0) = meta->dim[0];
        auto hyperslab = dataset.select(offset, count);

        // Write data
        hyperslab.write_raw(frame, type);

        // Only flush when writing to a single file. Otherwise we'll close the file anyway so we
        // don't need to flush.
        if (create_single_file) {
            // dataset.flush();
            file.flush();
        }
    }

    /**
     * @brief Create and write a file for an N2FrameView
     *
     * @param full_path The full path, including file name, for the file.
     * @param frame An N2FrameView object for the frame to output.
     */
    void write_n2(const N2FrameView& frame, const std::int64_t frame_counter) {
        DEBUG("Writing N2 frame seq: {:d}", frame.fpga_start_tick);

        std::shared_ptr<File> fileptr = create_file(frame_counter);
        auto& file = *fileptr;

        const std::vector<size_t> vis_dims({frame.num_prod, 2});
        const std::vector<size_t> weight_dims({frame.num_prod});
        const std::vector<size_t> flags_dims({frame.num_elements});
        const std::vector<size_t> eval_dims({frame.num_ev});
        const std::vector<size_t> evec_dims({frame.num_ev, frame.num_elements, 2});
        const std::vector<size_t> emethod_dims({1});
        const std::vector<size_t> erms_dims({1});
        const std::vector<size_t> gain_dims({frame.num_elements, 2});

        // Create dataspaces
        const DataSpace vis_space(vis_dims);
        const DataSpace weight_space(weight_dims);
        const DataSpace flags_space(flags_dims);
        const DataSpace eval_space(eval_dims);
        const DataSpace evec_space(evec_dims);
        const DataSpace emethod_space(emethod_dims);
        const DataSpace erms_space(erms_dims);
        const DataSpace gain_space(gain_dims);

        // Create datatypes
        const DataType float_type = chord2hdf5(kotekan::float32);
        const DataType int_type = chord2hdf5(kotekan::int32);

        // Make props
        auto vis_props = make_chunked_props(vis_dims);
        auto weight_props = make_chunked_props(weight_dims);
        auto flags_props = make_chunked_props(flags_dims);
        auto eval_props = make_chunked_props(eval_dims);
        auto evec_props = make_chunked_props(evec_dims);
        auto emethod_props = make_chunked_props(emethod_dims);
        auto erms_props = make_chunked_props(erms_dims);
        auto gain_props = make_chunked_props(gain_dims);

        // Create dataset
        auto vis_dset = file.createDataSet("vis", vis_space, float_type, vis_props);
        auto weight_dset = file.createDataSet("weight", weight_space, float_type, weight_props);
        auto flags_dset = file.createDataSet("flags", flags_space, float_type, flags_props);
        auto eval_dset = file.createDataSet("eval", eval_space, float_type, eval_props);
        auto evec_dset = file.createDataSet("evec", evec_space, float_type, evec_props);
        auto emethod_dset = file.createDataSet("emethod", emethod_space, int_type, emethod_props);
        auto erms_dset = file.createDataSet("erms", erms_space, float_type, erms_props);
        auto gain_dset = file.createDataSet("gain", gain_space, float_type, gain_props);

        vis_dset.write_raw(frame.vis.data(), float_type);
        weight_dset.write_raw(frame.weight.data(), float_type);
        flags_dset.write_raw(frame.flags.data(), float_type);
        eval_dset.write_raw(frame.eval.data(), float_type);
        evec_dset.write_raw(frame.evec.data(), float_type);
        emethod_dset.write_raw(&frame.emethod, int_type);
        erms_dset.write_raw(&frame.erms, float_type);
        gain_dset.write_raw(frame.gain.data(), float_type);

        // Set metadata as file-level attributes
        file.createAttribute("num_elements", frame.num_elements);
        file.createAttribute("num_prod", frame.num_prod);
        file.createAttribute("num_ev", frame.num_ev);
        file.createAttribute("freq_id", frame.freq_id);
        file.createAttribute("freq_MHz", frame.freq_MHz);
        file.createAttribute("abs_time_idx", frame.abs_time_idx);
        file.createAttribute("time_center_eop.t_inst_ns", frame.time_center_eop.t_inst_ns);
        file.createAttribute("time_center_eop.t_ut1_ns", frame.time_center_eop.t_ut1_ns);
        file.createAttribute("time_center_eop.delta_UT1_inst",
                             frame.time_center_eop.delta_UT1_inst);
        file.createAttribute("time_center_eop.ERA_deg", frame.time_center_eop.ERA_deg);
        file.createAttribute("time_center_eop.xp_as", frame.time_center_eop.xp_as);
        file.createAttribute("time_center_eop.yp_as", frame.time_center_eop.yp_as);
        file.createAttribute("bin_eop.t_inst_ns", frame.bin_eop.t_inst_ns);
        file.createAttribute("bin_eop.t_ut1_ns", frame.bin_eop.t_ut1_ns);
        file.createAttribute("bin_eop.delta_UT1_inst", frame.bin_eop.delta_UT1_inst);
        file.createAttribute("bin_eop.ERA_deg", frame.bin_eop.ERA_deg);
        file.createAttribute("bin_eop.xp_as", frame.bin_eop.xp_as);
        file.createAttribute("bin_eop.yp_as", frame.bin_eop.yp_as);
        file.createAttribute("bin_start_ERA_deg", frame.bin_start_ERA_deg);
        file.createAttribute("bin_end_ERA_deg", frame.bin_end_ERA_deg);
        file.createAttribute("bin_start_LAST", frame.bin_start_LAST);
        file.createAttribute("bin_end_LAST", frame.bin_end_LAST);
        file.createAttribute("fpga_start_tick", frame.fpga_start_tick);
        file.createAttribute("frame_start_time_ns", frame.frame_start_time_ns);
        file.createAttribute("frame_length_fpga_ticks", frame.frame_length_fpga_ticks);
        file.createAttribute("n_valid_fpga_ticks", frame.n_valid_fpga_ticks);
        file.createAttribute("n_rfi_fpga_ticks", frame.n_rfi_fpga_ticks);
    }

    /**
     * @brief Build a property list for dataset creation that enables compression
     * and chunking.
     *
     * @param dims The dataset dimensions.
     * @return RawPropertyList<PropertyType::DATASET_CREATE> The property list.
     */
    RawPropertyList<PropertyType::DATASET_CREATE>
    make_chunked_props(const std::vector<size_t>& dims) {

        RawPropertyList<PropertyType::DATASET_CREATE> props;

        std::vector<hsize_t> chunk_dims;

        bool dims_nonzero = true;
        for (size_t d = 0; d < dims.size(); d++) {
            chunk_dims.push_back((hsize_t)dims[d]);
            if (dims[d] == 0)
                dims_nonzero = false;
        }

        if (dims_nonzero) {
            // Enable chunking
            if (!chunk_dims.empty()) {
                // Choose chunk size
                std::size_t npoints_lo = 1;
                for (std::size_t d = 1; d < chunk_dims.size(); ++d)
                    npoints_lo *= chunk_dims.at(d);
                std::size_t npoints_hi = 10 * 1000 * 1000 / npoints_lo;
                using std::max, std::min;
                npoints_hi = min(std::size_t(chunk_dims.at(0)), max(std::size_t(1), npoints_hi));
                chunk_dims.at(0) = npoints_hi;
            }
            (*(DataSetCreateProps*)&props).add(Chunking(chunk_dims));

            constexpr int bitshuffle_compression_level = 9;
            const std::vector<unsigned int> bitshuffle_flags{
                BITSHUFFLE_BLOCKSIZE_AUTO,
                BITSHUFFLE_COMPRESS_ZSTD,
                bitshuffle_compression_level,
            };
            props.add(H5Pset_filter, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY, bitshuffle_flags.size(),
                      bitshuffle_flags.data());
        }

        return props;
    }

    /**
     * @brief The main thread function for hdf5FileWrite.
     *
     * This function is responsible for the main logic of the hdf5FileWrite class.
     */
    void main_thread() override {
        auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_hdf5filewrite_write_time_seconds", unique_name);

        const double start_time = current_time();

        for (std::int64_t frame_counter = 0;; ++frame_counter) {
            const std::uint32_t frame_id = frame_counter % buffer->num_frames;

            if (stop_thread)
                break;

            // Wait for the next frame
            DEBUG("wait_for_full_frame: frame_id={}", frame_id);
            const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, frame_id);
            if (!frame)
                break;
            DEBUG("got frame: frame_id={}", frame_id);

            // Start timer
            const double t0 = current_time();

            const double this_time = current_time();
            const double elapsed_time = this_time - start_time;

            INFO("Received buffer {} frame {} (duration {} sec)", unique_name, frame_counter,
                 elapsed_time);

            if (!skip_writing) {
                // Fetch metadata
                const std::shared_ptr<const metadataObject> mc = buffer->get_metadata(frame_id);
                if (!mc)
                    FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                                buffer->buffer_name, frame_id);
                assert(mc);

                // Call one of the writers, depending if metadata is chord or N2
                if (metadata_is_chord(mc)) {
                    assert(metadata_is_chord(mc));
                    const std::shared_ptr<const chordMetadata> meta = get_chord_metadata(mc);
                    const std::shared_ptr<const kotekan::GenericNDArray> frame_desc =
                        buffer->get_ndarray_frame_desc();
                    assert(frame_desc);
                    write_chord(frame, meta, frame_desc, frame_counter);
                } else if (metadata_is_N2(mc)) {
                    assert(metadata_is_N2(mc));
                    N2FrameView frame(buffer, frame_id);
                    write_n2(frame, frame_counter);
                } else {
                    FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD or N2",
                                buffer->buffer_name, frame_id);
                }
            } // if !skip_writing

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            write_time_metric.set(elapsed);

            // Mark frame as done
            DEBUG("mark_frame_empty: frame_id={}", frame_id);
            buffer->mark_frame_empty(unique_name, frame_id);

            if (max_frames >= 0 && frame_counter + 1 >= max_frames) {
                WARN("Processed {} frames", frame_counter + 1);
                break;
            }
        } // for

        // Close file if necessary
        the_single_file.reset();

        if (max_frames >= 0) {
            // Unregister to allow the pipeline to continue, unless I'm the last
            // consumer on this buffer.
            buffer->unregister_consumer(unique_name, true);

            if (--waiting_for_max_frames == 0) {
                WARN("Shutting down Kotekan");
                exit_kotekan(CLEAN_EXIT);
            }
        }

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileWrite);
