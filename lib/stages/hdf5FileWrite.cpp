#include "Config.hpp"   // for Config
#include "DataType.hpp" // for type_to_string
#include "N2FrameView.hpp"
#include "N2Metadata.hpp"
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "hdf5Files.hpp"       // for BITSHUFFLE_BLOCKSIZE_AUTO, BITSHUFFLE_C...
#include "kotekanLogging.hpp"  // for DEBUG, FATAL_ERROR, WARN, INFO
#include "metadata.hpp"        // for metadataObject

#include "fmt.hpp" // for compile_string_to_view

#include <Stage.hpp>                             // for Stage
#include <StageFactory.hpp>                      // for REGISTER_KOTEKAN_STAGE
#include <algorithm>                             // for max, min
#include <array>                                 // for array
#include <atomic>                                // for __atomic_base, atomic
#include <cassert>                               // for assert
#include <chordMetadata.hpp>                     // for chordMetadata, metadata_is_chord, get_c...
#include <cstddef>                               // for size_t, ptrdiff_t
#include <cstdint>                               // for int64_t, uint32_t, uint8_t
#include <errno.h>                               // for errno, EEXIST, EISDIR
#include <errors.h>                              // for exit_kotekan, ReturnCode
#include <functional>                            // for function
#include <highfive/H5DataSet.hpp>                // for DataSet, AnnotateTraits::createAttribute
#include <highfive/H5DataSpace.hpp>              // for DataSpace, DataSpace::DataSpace, DataSp...
#include <highfive/H5DataType.hpp>               // for DataType, DataType::getSize
#include <highfive/H5File.hpp>                   // for File, File::File, NodeTraits::createDat...
#include <highfive/H5Object.hpp>                 // for H5Z_FLAG_MANDATORY, hsize_t
#include <highfive/H5PropertyList.hpp>           // for PropertyType, RawPropertyList, Chunking
#include <highfive/bits/H5PropertyList_misc.hpp> // for PropertyList::_initializeIfNeeded, Chun...
#include <highfive/bits/H5Slice_traits_misc.hpp> // for SliceTraits::write_raw
#include <iomanip>                               // for operator<<, setfill, setw
#include <memory>                                // for shared_ptr, __shared_ptr_access, allocator
#include <prometheusMetrics.hpp>                 // for Metrics, Gauge
#include <sstream>                               // for basic_ostream, operator<<, basic_ostrin...
#include <string.h>                              // for strerror
#include <string>                                // for basic_string, char_traits, string, oper...
#include <sys/stat.h>                            // for mkdir
#include <unistd.h>                              // for gethostname
#include <vector>                                // for vector
#include <visUtil.hpp>                           // for current_time
#include <waitingForMaxFrames.hpp>               // for waiting_for_max_frames

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
 * @conf exit_after_n_frames  Int. Stop writing after this many frames, Default 0 = unlimited
 *       frames.
 * @conf exit_with_n_writers  Int. Exit after this many HDF5 writers finished writing, Default 0 =
 *       unlimited writers.
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

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const bool skip_writing = config.get_default<bool>(unique_name, "skip_writing", false);

    Buffer* const buffer;

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

    /**
     * @brief Create and write a file for a frame with chordMetadata
     *
     * @param full_path The full path, including file name, for the file.
     * @param frame uint8_t pointer to the frame data
     * @param meta shared pointer to the chordMetadata object for this frame.
     */
    void write_chord(const std::string& full_path, const std::uint8_t* const frame,
                     const std::shared_ptr<const chordMetadata> meta) {

        // Create HDF5 file
        File file(full_path, File::Truncate);

        // Choose dataspace
        const DataSpace space(meta->dim, meta->dim + meta->dims);
        {
            [[maybe_unused]] std::ptrdiff_t npoints = 1;
            for (int d = meta->dims - 1; d >= 0; --d) {
                assert(meta->stride[d] == npoints);
                npoints *= meta->dim[d];
            }
            assert(std::ptrdiff_t(space.getElementCount()) == npoints);
            assert(meta->offset == 0);
        }
        assert(std::ptrdiff_t(space.getNumberDimensions()) == meta->dims);

        // Choose datatype
        const DataType type = chord2hdf5(meta->type);

        RawPropertyList<PropertyType::DATASET_CREATE> props;

        // Enable chunking
        std::vector<hsize_t> chunk_dims(meta->dim, meta->dim + meta->dims);
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

        dataset.createAttribute("chord_metadata_version", chord_metadata_version);
        dataset.createAttribute("name", meta->get_name());
        dataset.createAttribute("type", type_to_string(meta->type));
        std::vector<std::string> dim_names;
        for (int d = 0; d < meta->dims; ++d)
            dim_names.push_back(meta->get_dimension_name(d));
        dataset.createAttribute("dim_names", dim_names);

        if (meta->has_fpga_seq_num())
            dataset.createAttribute("fpga_seq_num", meta->get_fpga_seq_num());

        if (meta->has_sample0_offset())
            dataset.createAttribute("sample0_offset", meta->get_sample0_offset());

        if (meta->has_offset_downsampling())
            dataset.createAttribute("offset_downsampling", meta->get_offset_downsampling());

        if (meta->has_coarse_freq())
            dataset.createAttribute("coarse_freq", meta->get_coarse_freq());

        if (meta->has_freq_upchan_factor())
            dataset.createAttribute("freq_upchan_factor", meta->get_freq_upchan_factor());

        if (meta->has_half_fpga_sample0())
            dataset.createAttribute("half_fpga_sample0", meta->get_half_fpga_sample0());

        if (meta->has_time_downsampling_fpga())
            dataset.createAttribute("time_downsampling_fpga",
                                    meta->get_time_downsampling_fpga());

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

        // Write data
        assert(dataset.getElementCount() * dataset.getDataType().getSize() == buffer->frame_size);
        dataset.write_raw(frame, type);
    }

    /**
     * @brief Create and write a file for an N2FrameView
     *
     * @param full_path The full path, including file name, for the file.
     * @param frame An N2FrameView object for the frame to output.
     */
    void write_n2(const std::string& full_path, const N2FrameView& frame) {
        // Create HDF5 file
        File file(full_path, File::Truncate);

        DEBUG("Writing N2 frame seq: {:d} to HDF5 file {:s}", frame.fpga_start_tick, full_path);

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
        file.createAttribute("nfreq", frame.nfreq);
        file.createAttribute("freq_id", frame.freq_id);
        file.createAttribute("freq_Hz", frame.freq_Hz);
        file.createAttribute("eop.t_inst", frame.eop.t_inst);
        file.createAttribute("eop.t_ut1", frame.eop.t_ut1);
        file.createAttribute("eop.delta_UT1_inst", frame.eop.delta_UT1_inst);
        file.createAttribute("eop.ERA_deg", frame.eop.ERA_deg);
        file.createAttribute("eop.xp_as", frame.eop.xp_as);
        file.createAttribute("eop.yp_as", frame.eop.yp_as);
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
            if(dims[d] == 0)
                dims_nonzero = false;
        }
        if(dims_nonzero) {

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
                // Define file name
                std::ostringstream buf;
                buf << base_dir << "/";
                if (prefix_hostname) {
                    char hostname[256];
                    gethostname(hostname, sizeof hostname);
                    buf << hostname << "_";
                }
                buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_counter
                    << ".h5";
                const std::string full_path = buf.str();

                // Create directory if necessary
                int ierr = mkdir(base_dir.c_str(), 0777);
                if (ierr) {
                    if (errno != EEXIST && errno != EISDIR) {
                        const char* const msg = strerror(errno);
                        FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(),
                                    msg);
                    }
                }

                // Fetch metadata
                const std::shared_ptr<const metadataObject> mc = buffer->get_metadata(frame_id);
                if (!mc)
                    FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                                buffer->buffer_name, frame_id);
                assert(mc);
                if (!(metadata_is_chord(mc) || metadata_is_N2(mc)))
                    FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD or N2",
                                buffer->buffer_name, frame_id);

                // Call one of the writers, depending if metadata is chord or N2
                if (metadata_is_chord(mc)) {
                    assert(metadata_is_chord(mc));
                    const std::shared_ptr<const chordMetadata> meta = get_chord_metadata(mc);
                    write_chord(full_path, frame, meta);
                } else if (metadata_is_N2(mc)) {
                    assert(metadata_is_N2(mc));
                    N2FrameView frame(buffer, frame_id);
                    write_n2(full_path, frame);
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

        if (--waiting_for_max_frames == 0) {
            WARN("Shutting down Kotekan");
            exit_kotekan(CLEAN_EXIT);
        }

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileWrite);
