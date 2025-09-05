#include "hdf5Files.hpp"

#include <Stage.hpp>
#include <StageFactory.hpp>
#include <algorithm>
#include <cassert>
#include <chordMetadata.hpp>
#include <complex>
#include <cstdint>
#include <errno.h>
#include <errors.h>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <prometheusMetrics.hpp>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>
#include <visUtil.hpp>

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

        buffer->register_consumer(unique_name);
    }

    virtual ~hdf5FileWrite() {}

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

            // Fetch metadata
            const std::shared_ptr<const metadataObject> mc = buffer->get_metadata(frame_id);
            if (!mc)
                FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                            buffer->buffer_name, frame_id);
            assert(mc);
            if (!metadata_is_chord(mc))
                FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD",
                            buffer->buffer_name, frame_id);
            assert(metadata_is_chord(mc));
            const std::shared_ptr<const chordMetadata> meta = get_chord_metadata(mc);

            const double this_time = current_time();
            const double elapsed_time = this_time - start_time;

            INFO("Received buffer {} frame {} time sample {} (duration {} sec)", unique_name,
                 frame_counter, meta->sample0_offset, elapsed_time);

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

                // Create HDF5 file
                File file(full_path, File::Truncate);

                // Choose dataspace
                const DataSpace space(meta->dim, meta->dim + meta->dims);
                {
                    std::ptrdiff_t npoints = 1;
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
                    npoints_hi =
                        min(std::size_t(chunk_dims.at(0)), max(std::size_t(1), npoints_hi));
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
                props.add(H5Pset_filter, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY,
                          bitshuffle_flags.size(), bitshuffle_flags.data());

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

                if (meta->sample0_offset >= 0)
                    dataset.createAttribute("sample0_offset", meta->sample0_offset);

                if (meta->offset_downsampling >= 0)
                    dataset.createAttribute("offset_downsampling", meta->offset_downsampling);

                if (meta->nfreq >= 0) {
                    dataset.createAttribute("nfreq", meta->nfreq);
                    dataset.createAttribute(
                        "coarse_freq",
                        std::vector<int>(meta->coarse_freq, meta->coarse_freq + meta->nfreq));
                    dataset.createAttribute(
                        "freq_upchan_factor",
                        std::vector<int>(meta->freq_upchan_factor,
                                         meta->freq_upchan_factor + meta->nfreq));
                    dataset.createAttribute(
                        "half_fpga_sample0",
                        std::vector<std::int64_t>(meta->half_fpga_sample0,
                                                  meta->half_fpga_sample0 + meta->nfreq));
                    dataset.createAttribute(
                        "time_downsampling_fpga",
                        std::vector<int>(meta->time_downsampling_fpga,
                                         meta->time_downsampling_fpga + meta->nfreq));
                }

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
                        std::vector<int>(meta->dish_index, meta->dish_index
                                                               + meta->n_dish_locations_ns
                                                                     * meta->n_dish_locations_ew));
                }

                // Write data
                assert(dataset.getElementCount() * dataset.getDataType().getSize()
                       == buffer->frame_size);
                dataset.write_raw(frame, type);

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

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileWrite);
