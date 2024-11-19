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
 *
 * @par Metrics
 * @metric kotekan_gdalviswrite_write_time_seconds
 *         The write time to write out the last frame.
 * @metric kotekan_gdalviswrite_n_datasets
 *         The number of datasets currently being held open.
 **/
class gdalVisWrite : public kotekan::Stage {

    const std::string base_dir = config.get<std::string>(unique_name, "base_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const std::string zip_compression = config.get_default<std::string>(unique_name, "zip_compression", "1");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);

    const int max_frames = config.get_default<int>(unique_name, "max_frames", -1);
    const std::uint32_t frames_per_file = config.get_default<std::uint32_t>(unique_name, "frames_per_file", 1000);

    Buffer* const buffer;

private:
    inline std::uint64_t _get_frame_n_in_file(const std::shared_ptr<const N2Metadata> meta)
    {
        auto& tel = Telescope::instance();
        std::uint64_t frame_len_ns = meta->frame_length_fpga_ticks * tel.seq_length_nsec();
        std::uint64_t file_len_ns = frame_len_ns * frames_per_file;
        std::uint64_t frame_n_in_file = ( meta->frame_start_time_ns % file_len_ns ) / frame_len_ns;

        return frame_n_in_file;
    }

    inline std::uint64_t _get_file_start_time_ns(const std::shared_ptr<const N2Metadata> meta)
    {
        auto& tel = Telescope::instance();
        std::uint64_t frame_len_ns = meta->frame_length_fpga_ticks * tel.seq_length_nsec();
        std::uint64_t file_len_ns = frame_len_ns * frames_per_file;
        std::uint64_t file_start_time_ns = meta->frame_start_time_ns - ( meta->frame_start_time_ns % file_len_ns );

        return file_start_time_ns;
    }

    std::string _get_gdal_vis_filename(std::shared_ptr<const N2Metadata> meta)
    {
        // Define file name
        std::ostringstream buf;
        buf << base_dir;
        if (!base_dir.empty() && base_dir.back() != '/') {
            buf << '/';
        }
        if (prefix_hostname) {
            char hostname[256];
            gethostname(hostname, sizeof hostname);
            buf << hostname << "_";
        }
        // Get the absolute frame number from fpga ticks
        const std::uint64_t file_start_time_ns = _get_file_start_time_ns(meta);

        // Use {YYYYMMDD}T{HHMMSS}Z //_{instrument -name}_{type} format
        std::time_t time_t_format = file_start_time_ns / 1'000'000'000; // Convert to seconds
        buf << file_name << "." << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT%H%M%S") << ".zarr.zip";
        const std::string full_path = buf.str();

        return full_path;
    }

    /**
     * Helper function to initialize GDAL file storing visibility data.
     */
    void _initialize_gdal_vis_file(GDALDataset* dataset, std::shared_ptr<const N2Metadata> meta)
    {
        assert(dataset && "Invalid dataset found during file initialization.");
        assert(meta && "Invalid metadata during file initialization.");

        DEBUG("Getting group...");
        const auto root_group = dataset->GetRootGroup();

        DEBUG("Group found for file.");
        if (!root_group) {
            GDALClose(dataset);
            FATAL_ERROR("Failed to get root group during file initialization.");
        }
                
        bool success;

        // Write metadata (attributes)
        const auto num_elements = root_group->CreateAttribute(
            "num_elements", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->num_elements)));
        success = num_elements->Write(&meta->num_elements, sizeof meta->num_elements);
        assert(success);

        const auto num_prod = root_group->CreateAttribute(
            "num_prod", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->num_prod)));
        success = num_prod->Write(&meta->num_prod, sizeof meta->num_prod);
        assert(success);

        const auto num_ev = root_group->CreateAttribute(
            "num_ev", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->num_ev)));
        success = num_ev->Write(&meta->num_ev, sizeof meta->num_ev);
        assert(success);

        const auto num_freq = root_group->CreateAttribute(
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

        DEBUG("Metadata written.");

        // Choose dimensions
        DEBUG("Initializing file with arrays of timensions {} / {} / {}",
            meta->nfreq, meta->num_prod, frames_per_file);
        std::vector<std::shared_ptr<GDALDimension>> dimensions = {
            root_group->CreateDimension("freqs", "", "", meta->nfreq),
            root_group->CreateDimension("products", "", "", meta->num_prod),
            root_group->CreateDimension("frames", "", "", frames_per_file),
        };

        // Create GDAL array
        std::ostringstream bbuf;
        bbuf << "BLOCKSIZE=1," << meta->num_prod << ",1"; // TODO: optimize?
        const std::vector<std::string> array_options{
            "COMPRESS=BLOSC",
            bbuf.str(),
            "BLOSC_CLEVEL=9",
            "BLOSC_SHUFFLE=BIT",
        };
        const auto array_options_c = convert_to_cstring_list(array_options);

        auto vis_array = root_group->CreateMDArray("vis_array", dimensions,
                GDALExtendedDataType::Create(GDT_CFloat32), array_options_c.data());
        assert(vis_array);
        assert(vis_array->GetDimensionCount() == 3);
                
        auto weights_array = root_group->CreateMDArray("weights_array", dimensions,
                GDALExtendedDataType::Create(GDT_Float32), array_options_c.data());
        assert(weights_array);
        assert(weights_array->GetDimensionCount() == 3);

        // Also record hash of config, per frame
        // const auto hashes = root_group->CreateMDArray("", dimensions, GDT_Float32,
        //                                         array_options_c.data());

        DEBUG("Initialized new file.");
    }

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
        auto& n_datasets_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_gdalviswrite_n_datasets", unique_name);

        const double start_time = current_time();

        N2::frameID in_frame_id(buffer);

        int frame_counter = 0;

        // datasets (files) for writing
        std::map<std::string, GDALDataset*> datasets;

        // Choose file format (driver)
        const auto driver_manager = GetGDALDriverManager();
        const std::string driver_name = "Zarr";
        const auto driver = driver_manager->GetDriverByName(driver_name.c_str());
        if (!driver)
            FATAL_ERROR("GDAL driver not available: {:s}", driver_name);

        // Create directory if necessary
        int ierr = mkdir(base_dir.c_str(), 0777);
        if (ierr) {
            if (errno != EEXIST && errno != EISDIR) {
                const char* const msg = strerror(errno);
                FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(),
                            msg);
            }
        }

        const auto cDataType = GDALExtendedDataType::Create(GDT_CFloat32);
        assert(cDataType.GetSize() == sizeof(std::complex<float>));

        while (!stop_thread) {
            // TODO: Consistency checks
            // For structural metadata, git version, config, ... ?
            
            // Wait for the next frame
            DEBUG("wait_for_full_frame: frame_id={}", in_frame_id);
            const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, in_frame_id);
            if (!frame)
                break;
            DEBUG("got frame: frame_id={}", in_frame_id);

            // Fetch metadata and frame vieww
            N2FrameView vis(buffer, in_frame_id);
            const std::shared_ptr<N2Metadata> meta = get_N2_metadata(buffer, in_frame_id);
            assert(meta);

            // Start timer
            const double frame_recv_time = current_time();
            const double total_elapsed_time = frame_recv_time - start_time;
            INFO("Received buffer {} frame {} (duration {} sec)", unique_name,
                 in_frame_id, total_elapsed_time);

            const std::string full_path = _get_gdal_vis_filename(meta);

            // Retrieve the relevant GDAL dataset (stores the array)
            GDALDataset* dataset = nullptr;
            struct stat filecheck_buffer {};
            if (datasets.find(full_path) != datasets.end()) // file/dataset already open
            { 
                dataset = datasets[full_path];
            }
            else if (stat(full_path.c_str(), &filecheck_buffer) == 0) // check if file exists
            {
                // If it does - open the file
                DEBUG("Opening existing file: {:s}", full_path);
                char** open_options = nullptr;
                dataset = static_cast<GDALDataset*>(GDALOpenEx(
                    full_path.c_str(),
                    GDAL_OF_MULTIDIM_RASTER | GDAL_OF_UPDATE,
                    nullptr,
                    const_cast<const char**>(open_options),
                    nullptr));
                // store the dataset as open
                DEBUG("Storing dataset for file {:s}", full_path);
                datasets[full_path] = dataset;
            }
            else // if file and/or dataset do not exist
            {
                // Create GDAL file (dataset)
                char** options = nullptr;
                options = CSLSetNameValue(options, "FORMAT", "ZARR_V2");
                options = CSLSetNameValue(options, "COMPRESS", "DEFLATE"); // zip compression
                options = CSLSetNameValue(options, "LEVEL", zip_compression.c_str());
                options = CSLSetNameValue(options, "STORAGE", "ZIP"); 
                dataset = driver->CreateMultiDimensional(full_path.c_str(), nullptr,
                                                         const_cast<const char**>(options));
                CSLDestroy(options);
                if (!dataset)
                    FATAL_ERROR("Could not initialize GDAL file {:s}", full_path);
                DEBUG("New dataset created for file: {:s}", full_path);
                datasets[full_path] = dataset;
                
                // initialize the dataset with relevant vis meta and arrays
                _initialize_gdal_vis_file(dataset, meta);
            }

            if (!dataset) {
                FATAL_ERROR("Dataset is null. Failed to open dataset.");
                return;
            }

            const auto root_group = dataset->GetRootGroup();
            if (!root_group) {
                GDALClose(dataset);
                FATAL_ERROR("Failed to get root group from existing dataset.");
            }

            auto vis_array = root_group->OpenMDArray("vis_array");
            assert(vis_array);

            auto weights_array = root_group->OpenMDArray("weights_array");
            assert(weights_array);

            const std::uint64_t frame_n_in_file = _get_frame_n_in_file(meta);
            DEBUG("Attempting to write data for {:d} products at f={:d}/{:d}, t={:d}/{:d}", meta->num_prod,
                meta->freq_id, meta->nfreq, frame_n_in_file, frames_per_file);
                
            // Write data
            {
                std::vector<GUInt64> arrayStartIdx = {meta->freq_id, 0, frame_n_in_file};
                std::vector<size_t> count = {1, meta->num_prod, 1}; // write along products dimension only

                bool success = false;
                
                success = vis_array->Write(arrayStartIdx.data(), count.data(), nullptr, nullptr,
                                cDataType, static_cast<const void*>(vis.vis.data()), nullptr, 0);
                if (!success)
                {
                    GDALClose(dataset);
                    FATAL_ERROR("Error writing vis array at F = {:d}, T = {:d}", meta->freq_id, frame_n_in_file );
                }

                success = weights_array->Write(arrayStartIdx.data(), count.data(), nullptr, nullptr,
                                cDataType, static_cast<const void*>(vis.weight.data()), nullptr, 0);
                if (!success)
                {
                    GDALClose(dataset);
                    FATAL_ERROR("Error writing weights array at F = {:d}, T = {:d}", meta->freq_id, frame_n_in_file );
                }
            }

            // Stop timer
            const double currt = current_time();
            const double elapsed_writing_frame = currt - frame_recv_time;
            write_time_metric.set(elapsed_writing_frame);

            // Record current datasets open
            n_datasets_metric.set(datasets.size());

            // Mark frame as done
            DEBUG("mark_frame_empty: frame_id={}", in_frame_id);
            buffer->mark_frame_empty(unique_name, in_frame_id++);

            if (max_frames >= 0 && frame_counter + 1 >= max_frames) {
                WARN("Processed {} frames, shutting down Kotekan", frame_counter);
                exit_kotekan(CLEAN_EXIT);
            }
            frame_counter++;

        } // while !stop_thread

        for (auto& [path, dataset] : datasets) {
            if (dataset) {
                GDALClose(dataset);
            }
        }
        datasets.clear();

        DEBUG("exiting");
    }
};

REGISTER_KOTEKAN_STAGE(gdalVisWrite);
