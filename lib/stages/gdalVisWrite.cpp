#include "gdalFiles.hpp"

#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <N2Metadata.hpp>
#include <N2FrameView.hpp>
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
#include "Telescope.hpp"         // for Telescope

using namespace gdal;

/**
 * @class gdalVisWrite
 * @brief Stream a visibility buffer to disk.
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format VisBuffer
 *     @buffer_metadata VisMetadata
 *
 * @conf base_dir  String. Directory to write into.
 * @conf file_name String. Base filename to write.
 * @conf exit_after_n_frames  Int. Stop writing after this many frames, Default 0 = unlimited
 *       frames.
 * @conf exit_with_n_writers  Int. Exit after this many GDAL writers finished writing, Default 0 =
 *       unlimited writers.
 *
 * @par Metrics
 * @metric kotekan_gdalviswrite_write_time_seconds
 *         The write time to write out the last frame.
 **/
class gdalVisWrite : public kotekan::Stage {

    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const bool skip_writing = config.get_default<bool>(unique_name, "skip_writing", false);

    Buffer* const buffer;

public:
    gdalVisWrite(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("in_buf")) {

        GDALAllRegister();

        buffer->register_consumer(unique_name);
    }

    virtual ~gdalVisWrite() {}

    void main_thread() override {

        auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_gdalviswrite_write_time_seconds", unique_name);

        const double start_time = current_time();

        auto& tel = Telescope::instance();
        const std::uint32_t frames_per_file = 100; // move to config... this sets the transpose size
        N2::frameID in_frame_id(buffer);

        int frame_counter = 0;

        while (!stop_thread) {
            // TODO: Consistency checks
            // For structural metadata, git version, config
            
            // Wait for the next frame
            DEBUG("wait_for_full_frame: frame_id={}", in_frame_id);
            const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, in_frame_id);
            if (!frame)
                break;
            DEBUG("got frame: frame_id={}", in_frame_id);

            // Start timer
            const double t0 = current_time();

            // Fetch metadata
            const std::shared_ptr<const metadataObject> mc = buffer->get_metadata(in_frame_id);
            if (!mc)
                FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                            buffer->buffer_name, in_frame_id);
            assert(mc);
            if (!metadata_is_N2(mc))
                FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type N2",
                            buffer->buffer_name, in_frame_id);
            assert(metadata_is_N2(mc));
            const std::shared_ptr<const N2Metadata> meta = std::static_pointer_cast<const N2Metadata>(mc);

            const double this_time = current_time();
            const double elapsed_time = this_time - start_time;

            INFO("Received buffer {} frame {} (duration {} sec)", unique_name,
                 in_frame_id, elapsed_time);
            frame_counter++;

            if (!skip_writing) {

                // Define file name
                std::ostringstream buf;
                buf << base_dir << "/";
                if (prefix_hostname) {
                    char hostname[256];
                    gethostname(hostname, sizeof hostname);
                    buf << hostname << "_";
                }
                // Get the absolute frame number from fpga ticks
                std::uint64_t frame_len_ns = meta->frame_length_fpga_ticks * tel.seq_length_nsec();
                std::uint64_t file_len_ns = frame_len_ns * frames_per_file;
                std::uint64_t file_start_time_ns = meta->frame_start_time_ns - ( meta->frame_start_time_ns % file_len_ns );
                std::uint16_t frame_n_in_file = ( meta->frame_start_time_ns % file_len_ns ) / frame_len_ns;

                // Use {YYYYMMDD}T{HHMMSS}Z //_{instrument -name}_{type} format
                std::time_t time_t_format = file_start_time_ns / 1'000'000'000; // Convert to seconds
                buf << file_name << "." << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT%H%M%SZ") << ".zarr";
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

                // Open the file if it exists, otherwise create a file.
                GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpen(full_path.c_str(), GA_ReadOnly));

                if (dataset) {
                    // File exists: close and reopen with read-write access
                    GDALClose(dataset);
                    dataset = static_cast<GDALDataset*>(GDALOpen(full_path.c_str(), GA_Update));

                    if (!dataset)
                        FATAL_ERROR("Failed to open the file for updating: \"{:s}\"", full_path);
                    DEBUG("File opened for updating: {:s}", full_path);
                } else {
                    // File does not exist: create a new file
                    // Choose file format (driver)
                    const auto driver_manager = GetGDALDriverManager();
                    const std::string driver_name = "Zarr";
                    const auto driver = driver_manager->GetDriverByName(driver_name.c_str());
                    if (!driver)
                        FATAL_ERROR("GDAL driver not available: {:s}", driver_name);

                    // Create GDAL file (dataset)
                    const std::vector<std::string> root_group_options{};
                    const auto root_group_options_c = convert_to_cstring_list(root_group_options);
                    const std::vector<std::string> options{
                        "FORMAT=ZARR_V2",
                        // "FORMAT=ZARR_V3",
                    };
                    const auto options_c = convert_to_cstring_list(options);
                    const auto dataset = std::unique_ptr<GDALDataset>(driver->CreateMultiDimensional(
                        full_path.c_str(), root_group_options_c.data(), options_c.data()));

                    if (!dataset)
                        FATAL_ERROR("Could not create GDAL file {:s}", full_path);
                    DEBUG("New file created: {:s}", full_path);

                    const auto group = dataset->GetRootGroup();
                    bool success;

                    // Write metadata (attributes)
                    const auto num_elements = group->CreateAttribute(
                        "num_elements", std::vector<GUInt64>{},
                        GDALExtendedDataType::Create(get_gdal_datatype(meta->num_elements)));
                    success = num_elements->Write(&meta->num_elements, sizeof meta->num_elements);
                    assert(success);

                    const auto num_prod = group->CreateAttribute(
                        "num_prod", std::vector<GUInt64>{},
                        GDALExtendedDataType::Create(get_gdal_datatype(meta->num_prod)));
                    success = num_prod->Write(&meta->num_prod, sizeof meta->num_prod);
                    assert(success);

                    const auto num_ev = group->CreateAttribute(
                        "num_ev", std::vector<GUInt64>{},
                        GDALExtendedDataType::Create(get_gdal_datatype(meta->num_ev)));
                    success = num_ev->Write(&meta->num_ev, sizeof meta->num_ev);
                    assert(success);

                    const auto num_freq = group->CreateAttribute(
                        "num_freq", std::vector<GUInt64>{},
                        GDALExtendedDataType::Create(get_gdal_datatype(meta->nfreq)));
                    success = num_freq->Write(&meta->nfreq, sizeof meta->nfreq);
                    assert(success);

                    // metadata["weight_type"] = mstate->get_weight_type();
                    // metadata["archive_version"] = "NT_3.1.0";
                    // metadata["instrument_name"] = instrument_name;
                    // metadata["notes"] = ""; // TODO: connect up notes
                    // metadata["git_version_tag"] = get_git_commit_hash();
                    // metadata["system_user"] = get_username();
                    // metadata["collection_server"] = get_hostname();

                    // Choose dimensions
                    std::vector<std::shared_ptr<GDALDimension>> dimensions = {
                        group->CreateDimension("freqs", "", "", meta->nfreq),
                        group->CreateDimension("products", "", "", meta->num_prod),
                        group->CreateDimension("frames", "", "", frames_per_file),
                    };

                    // Choose chunk (block) size
                    std::vector<std::int64_t> blocksize = {1, meta->num_prod, frames_per_file}; // TODO: optimize?

                    // Create GDAL array
                    std::ostringstream bbuf;
                    bbuf << "BLOCKSIZE=" << blocksize.at(0) << "," << blocksize.at(1)
                        << "," << blocksize.at(2);
                    const std::vector<std::string> array_options{
                        "COMPRESS=BLOSC",
                        bbuf.str(),
                        "BLOSC_CLEVEL=9",
                        "BLOSC_SHUFFLE=BIT",
                    };
                    const auto array_options_c = convert_to_cstring_list(array_options);
                    const auto vis_array = group->CreateMDArray("vis_matrix", dimensions,
                            GDALExtendedDataType::Create(GDT_CFloat32), array_options_c.data());
                    assert(vis_array);
                    const auto weights_array = group->CreateMDArray("weights_matrix", dimensions,
                            GDALExtendedDataType::Create(GDT_Float32), array_options_c.data());
                    assert(weights_array);

                    // Also record hash of config, per frame
                    // const auto hashes = group->CreateMDArray("", dimensions, GDT_Float32,
                    //                                         array_options_c.data());

                }

                const auto group = dataset->GetRootGroup();
                std::shared_ptr<GDALMDArray> vis_array = group->OpenMDArray("vis_array");
                std::shared_ptr<GDALMDArray> weights_array = group->OpenMDArray("weights_array");
                
                // Write data
                {
                    GUInt64 arrayStartIdx[] = {meta->freq_id, 0, frame_n_in_file};
                    size_t count[] = {1, meta->num_prod, 1}; // Number of elements to write in each dimension

                    // Array step for consecutive writing
                    GInt64 arrayStep[] = {1, 1, 1};

                    // Stride for the buffer layout
                    GPtrDiff_t bufferStride[] = {1, meta->num_prod, 1};

                    // Create and populate dataChunk with complex<float> values
                    std::vector<std::complex<float>> dataChunk(meta->num_prod);
                    for (size_t i = 0; i < meta->num_prod; ++i) {
                        dataChunk[i] = std::complex<float>(i * 0.1f, i * 0.2f);
                    }

                    // Write the chunk
                    bool success = vis_array->Write(arrayStartIdx, count, arrayStep, bufferStride,
                                                    GDALExtendedDataType::Create(GDT_CFloat32),
                                                    reinterpret_cast<float*>(dataChunk.data()),
                                                    nullptr, 0);

                    if (!success)
                        FATAL_ERROR("Error writing chunk at F = {:d}, T = {:d}", meta->freq_id, frame_n_in_file );

                }

                // TODO: Close when needed...
                GDALClose(dataset); // ?

            } // if !skip_writing

            // Stop timer
            const double t1 = current_time();
            const double elapsed = t1 - t0;
            write_time_metric.set(elapsed);

            // Mark frame as done
            DEBUG("mark_frame_empty: frame_id={}", in_frame_id);
            buffer->mark_frame_empty(unique_name, in_frame_id++);

            if (max_frames >= 0 && frame_counter + 1 >= max_frames) {
                WARN("Processed {} frames, shutting down Kotekan", frame_counter);
                exit_kotekan(CLEAN_EXIT);
            }
        } // for

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(gdalVisWrite);
