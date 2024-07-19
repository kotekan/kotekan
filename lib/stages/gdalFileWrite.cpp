#include "gdalFiles.hpp"

#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <complex>
#include <cstdint>
#include <errno.h>
#include <errors.h>
#include <fstream>
#include <gdal.h>
#include <gdal_priv.h>
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

using namespace gdal;

/**
 * @class gdalFileWrite
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
 * @conf exit_with_n_writers  Int. Exit after this many GDAL writers finished writing, Default 0 =
 *       unlimited writers.
 *
 * @par Metrics
 * @metric kotekan_gdalfilewrite_write_time_seconds
 *         The write time to write out the last frame.
 *
 * @author Erik Schnetter
 **/
class gdalFileWrite : public kotekan::Stage {
    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const bool skip_writing = config.get_default<bool>(unique_name, "skip_writing", false);

    Buffer* const buffer;

public:
    gdalFileWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("in_buf")) {

        GDALAllRegister();

        buffer->register_consumer(unique_name);
    }

    virtual ~gdalFileWrite() {}

    void main_thread() override {
        auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_gdalfilewrite_write_time_seconds", unique_name);

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

            // This is not a warning, but it should be displayed even
            // when regular INFO messages are not
            WARN("Received buffer {} frame {} time sample {} (duration {} sec)", unique_name,
                 frame_counter, meta->sample0_offset, elapsed_time);

            if (!skip_writing) {

                // Choose file format (driver)
                const auto driver_manager = GetGDALDriverManager();
                const std::string driver_name = "Zarr";
                const auto driver = driver_manager->GetDriverByName(driver_name.c_str());

                // Define file name
                std::ostringstream buf;
                buf << base_dir << "/";
                if (prefix_hostname) {
                    char hostname[256];
                    gethostname(hostname, sizeof hostname);
                    buf << hostname << "_";
                }
                buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_counter
                    << ".gdal";
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

                // Create GDAL file (dataset)
                const std::vector<std::string> root_group_options{};
                const auto root_group_options_c = convert_to_cstring_list(root_group_options);
                const std::vector<std::string> options{
                    "FORMAT=ZARR_V3",
                };
                const auto options_c = convert_to_cstring_list(options);
                const auto dataset = std::unique_ptr<GDALDataset>(driver->CreateMultiDimensional(
                    full_path.c_str(), root_group_options_c.data(), options_c.data()));
                if (!dataset)
                    FATAL_ERROR("Could not create GDAL file {:s}", full_path);

                const auto group = dataset->GetRootGroup();

                // Write metadata (attributes)

                {
                    const auto chord_metadata_version_attribute =
                        group->CreateAttribute("chord_metadata_version", std::vector<GUInt64>{2},
                                               GDALExtendedDataType::Create(GDT_Int32));
                    const bool success = chord_metadata_version_attribute->Write(
                        chord_metadata_version.data(), sizeof chord_metadata_version);
                    assert(success);
                }

                if (meta->nfreq >= 0) {
                    const auto nfreq = group->CreateAttribute(
                        "nfreq", std::vector<GUInt64>{},
                        GDALExtendedDataType::Create(get_gdal_datatype(meta->nfreq)));
                    const bool success = nfreq->Write(&nfreq, sizeof meta->nfreq);
                    assert(success);
                }

                if (meta->nfreq >= 0) {
                    const auto coarse_freq = group->CreateAttribute(
                        "coarse_freq", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                        GDALExtendedDataType::Create(get_gdal_datatype(*meta->coarse_freq)));
                    const bool success = coarse_freq->Write(
                        meta->coarse_freq, meta->nfreq * sizeof *meta->coarse_freq);
                    assert(success);
                }

                if (meta->nfreq >= 0) {
                    const auto freq_upchan_factor = group->CreateAttribute(
                        "freq_upchan_factor", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                        GDALExtendedDataType::Create(get_gdal_datatype(*meta->freq_upchan_factor)));
                    const bool success = freq_upchan_factor->Write(
                        meta->freq_upchan_factor, meta->nfreq * sizeof *meta->freq_upchan_factor);
                    assert(success);
                }

                if (meta->sample0_offset >= 0) {
                    const auto sample0_offset = group->CreateAttribute(
                        "sample0_offset", std::vector<GUInt64>{},
                        GDALExtendedDataType::Create(get_gdal_datatype(meta->sample0_offset)));
                    const bool success =
                        sample0_offset->Write(&meta->sample0_offset, sizeof meta->sample0_offset);
                    assert(success);
                }

                if (meta->nfreq >= 0) {
                    const auto half_fpga_sample0 = group->CreateAttribute(
                        "half_fpga_sample0", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                        GDALExtendedDataType::Create(get_gdal_datatype(*meta->half_fpga_sample0)));
                    const bool success = half_fpga_sample0->Write(
                        meta->half_fpga_sample0, meta->nfreq * sizeof *meta->half_fpga_sample0);
                    assert(success);
                }

                if (meta->nfreq >= 0) {
                    const auto time_downsampling_fpga = group->CreateAttribute(
                        "time_downsampling_fpga", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                        GDALExtendedDataType::Create(
                            get_gdal_datatype(*meta->time_downsampling_fpga)));
                    const bool success = time_downsampling_fpga->Write(
                        meta->time_downsampling_fpga,
                        meta->nfreq * sizeof *meta->time_downsampling_fpga);
                    assert(success);
                }

                if (meta->ndishes >= 0) {
                    const auto ndishes = group->CreateAttribute(
                        "ndishes", std::vector<GUInt64>{},
                        GDALExtendedDataType::Create(get_gdal_datatype(meta->ndishes)));
                    const bool success = ndishes->Write(&meta->ndishes, sizeof meta->ndishes);
                    assert(success);
                }

                std::shared_ptr<GDALDimension> dishM, dishN;
                if (meta->dish_index) {
                    // const auto dish_index = group->CreateAttribute(
                    //     "dish_index",
                    //     std::vector<GUInt64>{GUInt64(meta->n_dish_locations_ns),
                    //                          GUInt64(meta->n_dish_locations_ew)},
                    //     GDALExtendedDataType::Create(GDT_Int32));
                    // dish_index->Write(meta->dish_index,
                    //                   meta->n_dish_locations_ew * meta->n_dish_locations_ns);
                    const auto datatype =
                        GDALExtendedDataType::Create(get_gdal_datatype(*meta->dish_index));
                    dishM = group->CreateDimension("dishM", "", "", meta->n_dish_locations_ns);
                    assert(dishM);
                    dishN = group->CreateDimension("dishN", "", "", meta->n_dish_locations_ew);
                    assert(dishN);
                    const std::vector<std::shared_ptr<GDALDimension>> dimensions{dishM, dishN};
                    assert(dimensions.at(0));
                    assert(dimensions.at(1));
                    const auto dish_index =
                        group->CreateMDArray("dish_index", dimensions, datatype);
                    assert(dish_index);
                    const std::vector<GUInt64> arrayStart{0, 0};
                    const std::vector<std::size_t> count{std::size_t(meta->n_dish_locations_ns),
                                                         std::size_t(meta->n_dish_locations_ew)};
                    const bool success = dish_index->Write(arrayStart.data(), count.data(), nullptr,
                                                           nullptr, datatype, meta->dish_index);
                    assert(success);
                }

                // Array rank
                const int ndims = meta->dims;

                INFO("name={} type={} typesize={} ndims={} dims={}", meta->get_name(),
                     meta->get_type_string(), chord_datatype_bytes(meta->type), meta->dims,
                     meta->get_dimensions_string());
                for (int d = 0; d < ndims; ++d)
                    INFO("    [{}] name={} size={}", d, meta->get_dimension_name(d), meta->dim[d]);
                INFO("    buffer addr={} size={}", (const void*)frame, buffer->frame_size);

                // Array element type
                const auto datatype = GDALExtendedDataType::Create(chord2gdal(meta->type));
                const std::int64_t datatypesize = GDALGetDataTypeSizeBytes(chord2gdal(meta->type));
                assert(datatypesize == std::int64_t(chord_datatype_bytes(meta->type)));

                // Array size
                std::vector<std::shared_ptr<GDALDimension>> dimensions(ndims);
                for (int d = 0; d < ndims; ++d) {
                    const std::string type;      // unused
                    const std::string direction; // unused
                    if (meta->get_dimension_name(d) == "dishM" && dishM)
                        dimensions.at(d) = dishM;
                    else if (meta->get_dimension_name(d) == "dishN" && dishN)
                        dimensions.at(d) = dishN;
                    else
                        dimensions.at(d) = group->CreateDimension(meta->get_dimension_name(d), type,
                                                                  direction, meta->dim[d]);
                    assert(dimensions.at(d));
                }

                // Choose chunk (block) size
                std::vector<std::int64_t> blocksize(ndims);
                for (int d = 0; d < ndims; ++d)
                    blocksize.at(d) = meta->dim[d];
                std::int64_t size = datatypesize;
                for (int d = 0; d < ndims; ++d)
                    size *= meta->dim[d];
                const std::int64_t maxsize = std::int64_t(1024) * 1024 * 1024; // 1 GByte
                if (size > maxsize) {
                    const std::int64_t ratio = (size + maxsize - 1) / maxsize;
                    assert(blocksize.at(0) >= ratio);
                    blocksize.at(0) /= ratio;
                }

                // Create GDAL array
                std::ostringstream bbuf;
                bbuf << "BLOCKSIZE=";
                for (int d = 0; d < ndims; ++d)
                    bbuf << (d == 0 ? "" : ",") << blocksize.at(d);
                const std::string blocksize_str = bbuf.str();
                const std::vector<std::string> array_options{
                    "COMPRESS=BLOSC",
                    blocksize_str,
                    "BLOSC_CLEVEL=9",
                    "BLOSC_SHUFFLE=BIT",
                };
                const auto array_options_c = convert_to_cstring_list(array_options);
                const auto mdarray = group->CreateMDArray(meta->get_name(), dimensions, datatype,
                                                          array_options_c.data());
                assert(mdarray);

                // Describe datatype
                {
                    const std::string type_value = chord_datatype_string(meta->type);
                    const auto type_datatype =
                        GDALExtendedDataType::CreateString(type_value.size());
                    const auto type =
                        mdarray->CreateAttribute("type", std::vector<GUInt64>{}, type_datatype);
                    assert(type);
                    const bool success = type->Write(type_value.c_str());
                    assert(success);
                }

                // Write data
                {
                    const std::vector<GUInt64> arrayStart(ndims, 0);
                    std::vector<std::size_t> count(ndims);
                    for (int d = 0; d < ndims; ++d)
                        count.at(d) = meta->dim[d];
                    std::vector<GPtrDiff_t> bufferStride(ndims);
                    for (int d = 0; d < ndims; ++d)
                        bufferStride.at(d) = meta->stride[d];

                    const bool success = mdarray->Write(
                        arrayStart.data(), count.data(), nullptr, bufferStride.data(), datatype,
                        frame + datatypesize * meta->offset, frame, buffer->frame_size);
                    assert(success);
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
                WARN("Processed {} frames, shutting down Kotekan", frame_counter);
                exit_kotekan(CLEAN_EXIT);
            }
        } // for

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(gdalFileWrite);
