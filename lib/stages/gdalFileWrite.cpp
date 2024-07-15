#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chordMetadata.hpp>
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
#include <unistd.h>
#include <utility>
#include <vector>
#include <visUtil.hpp>

namespace {
GDALDataType chord2gdal(const chordDataType type) {
    switch (type) {
        case uint4p4:
            return GDT_Byte; // TODO: Define GDAL uint4+4 type
        case uint8:
            return GDT_Byte;
        case uint16:
            return GDT_UInt16;
        case uint32:
            return GDT_UInt32;
        case uint64:
            return GDT_UInt64;
        case int4p4:
            return GDT_Byte; // TODO: Define GDAL int4+4 type
        case int8:
            return GDT_Int8;
        case int16:
            return GDT_Int16;
        case int32:
            return GDT_Int32;
        case int64:
            return GDT_Int64;
        case float16:
            return GDT_UInt16; // TODO: Define GDAL float16 type
        case float32:
            return GDT_Float32;
        case float64:
            return GDT_Float64;
        default:
            assert(0);
    }
}

std::vector<const char*> convert_to_cstring_list(const std::vector<std::string>& strings) {
    std::vector<const char*> result;
    // Convert strings to C strings
    for (const auto& str : strings)
        result.push_back(str.c_str());
    // Add trailing NULL
    result.push_back(nullptr);
    return result;
}

} // namespace

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
                std::ostringstream ibuf;
                ibuf << std::setw(8) << std::setfill('0') << frame_counter;
                const std::string iteration = ibuf.str();
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

                // Create GDAL file (dataset)
                const std::vector<std::string> root_group_options{};
                const auto root_group_options_c = convert_to_cstring_list(root_group_options);
                const std::vector<std::string> options{"FORMAT=ZARR_V3"};
                const auto options_c = convert_to_cstring_list(options);
                const auto dataset = std::unique_ptr<GDALDataset>(driver->CreateMultiDimensional(
                    full_path.c_str(), root_group_options_c.data(), options_c.data()));

                const auto group = dataset->GetRootGroup();

                // Array element type
                const auto datatype = GDALExtendedDataType::Create(chord2gdal(meta->type));

                // Array size
                const int ndims = meta->dims;
                std::vector<std::shared_ptr<GDALDimension>> dimensions(ndims);
                for (int d = 0; d < ndims; ++d) {
                    const std::string type;      // unused
                    const std::string direction; // unused
                    dimensions.at(d) = group->CreateDimension(
                        meta->get_dimension_name(d), type.c_str(), direction.c_str(), meta->dim[d]);
                }

                // Create GDAL array
                std::ostringstream bbuf;
                bbuf << "BLOCKSIZE=";
                for (int d = 0; d < ndims; ++d)
                    bbuf << (d == 0 ? "" : ",") << meta->dim[d];
                const std::string blocksize = bbuf.str();
                const std::vector<std::string> array_options{
                    "COMPRESS=BLOSC",
                    blocksize.c_str(),
                    "BLOSC_CLEVEL=9",
                    "BLOSC_SHUFFLE=BIT",
                };
                const auto array_options_c = convert_to_cstring_list(array_options);
                const auto mdarray = group->CreateMDArray(meta->get_name().c_str(), dimensions,
                                                          datatype, array_options_c.data());

                // Write metadata (attributes)
                const auto coarse_freq = mdarray->CreateAttribute(
                    "coarse_freq", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                    GDALExtendedDataType::Create(GDT_Int32));
                coarse_freq->Write(meta->coarse_freq, meta->nfreq);

                const auto freq_upchan_factor = mdarray->CreateAttribute(
                    "freq_upchan_factor", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                    GDALExtendedDataType::Create(GDT_Int32));
                freq_upchan_factor->Write(meta->freq_upchan_factor, meta->nfreq);

                const auto sample0_offset =
                    mdarray->CreateAttribute("sample0_offset", std::vector<GUInt64>{},
                                             GDALExtendedDataType::Create(GDT_Int64));
                sample0_offset->Write(&meta->sample0_offset, 1);

                const auto half_fpga_sample0 = mdarray->CreateAttribute(
                    "half_fpga_sample0", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                    GDALExtendedDataType::Create(GDT_Int64));
                half_fpga_sample0->Write(meta->half_fpga_sample0, meta->nfreq);

                const auto time_downsampling_fpga = mdarray->CreateAttribute(
                    "time_downsampling_fpga", std::vector<GUInt64>{GUInt64(meta->nfreq)},
                    GDALExtendedDataType::Create(GDT_Int64));
                time_downsampling_fpga->Write(meta->time_downsampling_fpga, meta->nfreq);

                if (meta->ndishes >= 0) {
                    const auto ndishes = mdarray->CreateAttribute(
                        "ndishes", std::vector<GUInt64>{}, GDALExtendedDataType::Create(GDT_Int32));
                    ndishes->Write(&meta->ndishes, 1);
                }

                if (meta->dish_index) {
                    const auto dish_index = mdarray->CreateAttribute(
                        "dish_index",
                        std::vector<GUInt64>{GUInt64(meta->n_dish_locations_ns),
                                             GUInt64(meta->n_dish_locations_ew)},
                        GDALExtendedDataType::Create(GDT_Int32));
                    dish_index->Write(meta->dish_index,
                                      meta->n_dish_locations_ew * meta->n_dish_locations_ns);
                }

                // Write data
                const std::vector<GUInt64> arrayStart(ndims, 0);
                std::vector<std::size_t> count(ndims);
                for (int d = 0; d < ndims; ++d)
                    count.at(d) = meta->dim[d];
                std::vector<GPtrDiff_t> bufferStride(ndims);
                for (int d = 0; d < ndims; ++d)
                    bufferStride.at(d) = meta->stride[d];

                const bool success = mdarray->Write(arrayStart.data(), count.data(), nullptr,
                                                    bufferStride.data(), datatype, frame);
                assert(success);

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
