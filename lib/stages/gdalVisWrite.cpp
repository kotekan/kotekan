#include "gdalVisWrite.hpp"

#include "Telescope.hpp" // for Telescope
#include "gdalFiles.hpp"
#include "util.h" // for mkdir_p

#include "json.hpp"

#include <N2FrameView.hpp>
#include <N2Metadata.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <complex>
#include <configTracker.hpp>
#include <cpl_vsi.h>
#include <cstdint>
#include <cstring>
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

REGISTER_KOTEKAN_STAGE(gdalVisWrite);

gdalVisWrite::gdalVisWrite(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          [](const kotekan::Stage& stage) {
              return const_cast<kotekan::Stage&>(stage).main_thread();
          }),
    base_dir(config.get<std::string>(unique_name, "base_dir")),
    file_name(config.get<std::string>(unique_name, "file_name")),
    prefix_hostname(config.get_default<bool>(unique_name, "prefix_hostname", true)),
    format(config.get_default<std::string>(unique_name, "format", "zarr")),
    compression(config.get_default<std::string>(unique_name, "compression", "none")),
    compression_level(config.get_default<std::uint64_t>(unique_name, "compression_level", 0)),
    use_bitshuffle(config.get_default<bool>(unique_name, "use_bitshuffle", false)),
    blocksize_f(config.get_default<std::uint64_t>(unique_name, "blocksize_f", 0)),
    blocksize_p(config.get_default<std::uint64_t>(unique_name, "blocksize_p", 0)),
    blocksize_t(config.get_default<std::uint64_t>(unique_name, "blocksize_t", 1)),
    file_seconds(config.get_default<std::uint64_t>(unique_name, "file_seconds", 600)),
    late_frame_grace_seconds(
        config.get_default<std::uint64_t>(unique_name, "late_frame_grace_seconds", 60)),
    max_frames(config.get_default<int>(unique_name, "max_frames", -1)),
    buffer(get_buffer("in_buf")), tick_len_ns_override(config.get_default<std::uint64_t>(
                                      unique_name, "seq_length_nsec_override", 0)) {

    GDALAllRegister();
    buffer->register_consumer(unique_name);

    // Validate file window configuration
    if (file_seconds == 0) {
        FATAL_ERROR("file_seconds must be > 0 for gdalVisWrite");
    }
    const std::uint64_t day_seconds = 86400ULL;
    if ((day_seconds % file_seconds) != 0) {
        FATAL_ERROR("file_seconds={} must evenly divide 86400.", file_seconds);
    }
}

gdalVisWrite::~gdalVisWrite() {}


std::uint64_t gdalVisWrite::_get_file_start_time_ns(const std::shared_ptr<const N2Metadata> meta) {
    // Align to UTC midnight and configured file window length.
    const std::uint64_t W_ns = file_seconds * 1'000'000'000ULL;
    const std::uint64_t day_ns = 86'400ULL * 1'000'000'000ULL;
    const std::uint64_t day_start =
        (meta->frame_start_time_ns / day_ns) * day_ns; // round down to previous midnight
    const std::uint64_t offset_in_day =
        meta->frame_start_time_ns - day_start;                 // ns since previous midnight
    const std::uint64_t file_win_index = offset_in_day / W_ns; // 0-based file window index
    return day_start + file_win_index * W_ns;
}

std::string gdalVisWrite::_get_gdal_vis_filename(std::shared_ptr<const N2Metadata> meta) {
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
    const std::uint64_t file_start_time_ns = _get_file_start_time_ns(meta);
    std::time_t time_t_format = file_start_time_ns / 1'000'000'000;      // seconds
    const std::uint64_t ns_part = file_start_time_ns % 1'000'000'000ULL; // sub-second
    buf << file_name << "." << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT%H%M%S");
    // Include nanosecond suffix to avoid collisions for sub-second file windows
    buf << "_" << std::setw(9) << std::setfill('0') << ns_part;
    if (format == std::string("hdf5"))
        buf << ".h5";
    else
        buf << ".zarr";
    return buf.str();
}

std::vector<std::string>
gdalVisWrite::_get_array_create_options(const std::vector<GUInt64>& chunk_dims) const {
    std::vector<std::string> opts;
    if (!chunk_dims.empty()) {
        std::ostringstream b;
        b << "BLOCKSIZE=";
        for (size_t i = 0; i < chunk_dims.size(); ++i) {
            if (i)
                b << ",";
            b << static_cast<unsigned long long>(chunk_dims[i]);
        }
        opts.emplace_back(b.str());
    }

    // Add compression flags. These may be ignored by drivers that do not support them.
    if (compression == std::string("deflate")) {
        opts.emplace_back("COMPRESS=DEFLATE");
        if (compression_level > 0)
            opts.emplace_back("LEVEL=" + std::to_string(compression_level));
        if (use_bitshuffle)
            opts.emplace_back("SHUFFLE=YES");
    } else if (compression == std::string("zstd")) {
        opts.emplace_back("COMPRESS=ZSTD");
        if (compression_level > 0)
            opts.emplace_back("LEVEL=" + std::to_string(compression_level));
        if (use_bitshuffle)
            opts.emplace_back("SHUFFLE=YES");
    } else if (compression == std::string("blosc")) {
        // Approximate with modern ZSTD + SHUFFLE when BLOSC specifics aren't exposed via GDAL.
        opts.emplace_back("COMPRESS=ZSTD");
        if (compression_level > 0)
            opts.emplace_back("LEVEL=" + std::to_string(compression_level));
        if (use_bitshuffle)
            opts.emplace_back("SHUFFLE=YES");
    }

    return opts;
}

void gdalVisWrite::_initialize_gdal_vis_file(GDALDataset* dataset,
                                             std::shared_ptr<const N2Metadata> meta,
                                             std::uint64_t file_nt) {
    assert(dataset && "Invalid dataset found during file initialization.");
    assert(meta && "Invalid metadata during file initialization.");

    const auto root_group = dataset->GetRootGroup();
    if (!root_group) {
        GDALClose(dataset);
        FATAL_ERROR("Failed to get root group during file initialization.");
    }

    bool success;
    // Root attributes
    {
        const auto num_elements = root_group->CreateAttribute(
            "num_elements", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->num_elements)));
        success = num_elements->Write(&meta->num_elements, sizeof meta->num_elements);
        if (!success)
            ERROR("Failed to write num_elements attribute to dataset {}",
                  dataset->GetDescription());

        const auto num_prod = root_group->CreateAttribute(
            "num_prod", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->num_prod)));
        success = num_prod->Write(&meta->num_prod, sizeof meta->num_prod);
        if (!success)
            ERROR("Failed to write num_prod attribute to dataset {}", dataset->GetDescription());

        const auto num_ev = root_group->CreateAttribute(
            "num_ev", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->num_ev)));
        success = num_ev->Write(&meta->num_ev, sizeof meta->num_ev);
        if (!success)
            ERROR("Failed to write num_ev attribute to dataset {}", dataset->GetDescription());

        const auto num_freq = root_group->CreateAttribute(
            "num_freq", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->nfreq)));
        success = num_freq->Write(&meta->nfreq, sizeof meta->nfreq);
        if (!success)
            ERROR("Failed to write num_freq attribute to dataset {}", dataset->GetDescription());

        const auto frame_length_fpga_ticks = root_group->CreateAttribute(
            "frame_length_fpga_ticks", std::vector<GUInt64>{},
            GDALExtendedDataType::Create(get_gdal_datatype(meta->frame_length_fpga_ticks)));
        success = frame_length_fpga_ticks->Write(&meta->frame_length_fpga_ticks,
                                                 sizeof meta->frame_length_fpga_ticks);
        if (!success)
            ERROR("Failed to write frame_length_fpga_ticks attribute to dataset {}",
                  dataset->GetDescription());

        // Add configTracker JSON as attribute
        const nlohmann::json config_json = kotekan::ConfigTracker::instance().getAllConfigsAsJson();
        const std::string cfg_dump = config_json.dump();
        const auto config_attr = root_group->CreateAttribute(
            "config_json", std::vector<GUInt64>{static_cast<GUInt64>(cfg_dump.size())},
            GDALExtendedDataType::Create(GDT_Byte));
        success = config_attr->Write(cfg_dump.data(), cfg_dump.size());
        if (!success)
            ERROR("Failed to write config_json attribute to dataset {}", dataset->GetDescription());
    }

    // GDAL Dimensions
    std::shared_ptr<GDALDimension> dim_freq =
        root_group->CreateDimension("freqs", "", "", meta->nfreq);
    std::shared_ptr<GDALDimension> dim_prod =
        root_group->CreateDimension("products", "", "", meta->num_prod);
    std::shared_ptr<GDALDimension> dim_frames =
        root_group->CreateDimension("frames", "", "", file_nt);
    std::shared_ptr<GDALDimension> dim_inputs =
        root_group->CreateDimension("inputs", "", "", meta->num_elements);
    std::shared_ptr<GDALDimension> dim_ev = root_group->CreateDimension("ev", "", "", meta->num_ev);

    // Build array creation options for common ranks
    const GUInt64 bs_f = (blocksize_f > 0) ? blocksize_f : 1;
    const GUInt64 bs_p = (blocksize_p > 0) ? blocksize_p : meta->num_prod;
    const GUInt64 bs_t = (blocksize_t > 0) ? blocksize_t : file_nt;
    const GUInt64 bs_ev = meta->num_ev;
    const GUInt64 bs_i = meta->num_elements;

    const auto opts_3d_fpt = convert_to_cstring_list(_get_array_create_options({bs_f, bs_p, bs_t}));
    const auto opts_3d_fet =
        convert_to_cstring_list(_get_array_create_options({bs_f, bs_ev, bs_t}));
    const auto opts_4d_feit =
        convert_to_cstring_list(_get_array_create_options({bs_f, bs_ev, bs_i, bs_t}));
    const auto opts_3d_fit = convert_to_cstring_list(_get_array_create_options({bs_f, bs_i, bs_t}));
    const auto opts_2d_ft = convert_to_cstring_list(_get_array_create_options({bs_f, bs_t}));

    // vis and weights: (freqs, products, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_prod, dim_frames};
        auto vis_array = root_group->CreateMDArray(
            "vis_array", dims, GDALExtendedDataType::Create(GDT_CFloat32), opts_3d_fpt.data());
        assert(vis_array && vis_array->GetDimensionCount() == 3);
        auto weights_array = root_group->CreateMDArray(
            "weights_array", dims, GDALExtendedDataType::Create(GDT_Float32), opts_3d_fpt.data());
        assert(weights_array && weights_array->GetDimensionCount() == 3);
    }

    // eval: (freqs, ev, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev, dim_frames};
        auto eval_array = root_group->CreateMDArray(
            "eval_array", dims, GDALExtendedDataType::Create(GDT_Float32), opts_3d_fet.data());
        (void)eval_array;
    }
    // evec: (freqs, ev, inputs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev, dim_inputs, dim_frames};
        auto evec_array = root_group->CreateMDArray(
            "evec_array", dims, GDALExtendedDataType::Create(GDT_CFloat32), opts_4d_feit.data());
        (void)evec_array;
    }
    // erms: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        auto erms_array = root_group->CreateMDArray(
            "erms_array", dims, GDALExtendedDataType::Create(GDT_Float32), opts_2d_ft.data());
        (void)erms_array;
    }
    // gain + flags: (freqs, inputs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_inputs, dim_frames};
        auto gain_array = root_group->CreateMDArray(
            "gain_array", dims, GDALExtendedDataType::Create(GDT_CFloat32), opts_3d_fit.data());
        (void)gain_array;
        auto flags_array = root_group->CreateMDArray(
            "flags_array", dims, GDALExtendedDataType::Create(GDT_Float32), opts_3d_fit.data());
        (void)flags_array;
    }
    // frac_*: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        auto fl_array = root_group->CreateMDArray(
            "frac_lost_array", dims, GDALExtendedDataType::Create(GDT_Float32), opts_2d_ft.data());
        (void)fl_array;
        auto fr_array = root_group->CreateMDArray(
            "frac_rfi_array", dims, GDALExtendedDataType::Create(GDT_Float32), opts_2d_ft.data());
        (void)fr_array;
    }
    // per-time metadata arrays: (frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_frames};
        (void)root_group->CreateMDArray("fpga_start_tick", dims,
                                        GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root_group->CreateMDArray("frame_start_time_ns", dims,
                                        GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root_group->CreateMDArray("frame_length_fpga_ticks", dims,
                                        GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root_group->CreateMDArray("era_deg", dims, GDALExtendedDataType::Create(GDT_Float64),
                                        nullptr);
    }
    // per-(freq,time) metadata arrays: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        (void)root_group->CreateMDArray("n_valid_fpga_ticks", dims,
                                        GDALExtendedDataType::Create(GDT_UInt64),
                                        opts_2d_ft.data());
        (void)root_group->CreateMDArray(
            "n_rfi_fpga_ticks", dims, GDALExtendedDataType::Create(GDT_UInt64), opts_2d_ft.data());
    }
}

void gdalVisWrite::_grace_finalize_datasets(
    std::map<std::string, std::unique_ptr<gdalVisFileData>>& datasets) {
    const double now_s = current_time();
    for (auto ds_it = datasets.begin(); ds_it != datasets.end(); ++ds_it) {
        auto& obj = *ds_it->second;
        if (now_s - obj.last_update_wall_s >= double(late_frame_grace_seconds)) {
            // Grace finalize: flush and rename
            if (obj.gdal_dataset)
                obj.flush();
            if (obj.gdal_dataset) {
                GDALClose(obj.gdal_dataset);
                obj.gdal_dataset = nullptr;
            }
            // Attempt rename from partial to final
            int r = std::rename(obj.partial_path.c_str(), ds_it->first.c_str());
            if (r != 0) {
                const char* msg = strerror(errno);
                ERROR("Failed to rename partial dataset to final: {} -> {}: {}", obj.partial_path,
                      ds_it->first, msg);
            }
            ds_it = datasets.erase(ds_it);
            continue; // to next dataset
        }
    }
}

void gdalVisWrite::main_thread() {
    auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gdalviswrite_write_time_seconds", unique_name);
    double avg_write_time = 0.0;
    auto& n_datasets_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gdalviswrite_n_datasets", unique_name);

    const double start_time = current_time(); // for logging elapsed time
    N2::frameID in_frame_id(buffer);          // Input frame ID tracker
    int frame_counter = 0;                    // Count of frames written
    bool warned_short_file_window = false;    // Warn only once for short file windows

    // datasets (files) for writing (multiple may be open simultaneously)
    std::map<std::string, std::unique_ptr<gdalVisFileData>> datasets;

    // Choose file format (driver)
    const auto driver_manager = GetGDALDriverManager();
    std::string driver_name = std::string("HDF5");
    if (format == std::string("zarr"))
        driver_name = std::string("Zarr");
    const auto driver = driver_manager->GetDriverByName(driver_name.c_str());
    if (!driver)
        FATAL_ERROR("GDAL driver not available: {:s}", driver_name);

    // Create base_dir and partial dir if necessary (recursively)
    {
        if (mkdir_p(base_dir.c_str(), 0777) != 0) {
            const char* const msg = strerror(errno);
            FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(), msg);
        }
        std::string partial_dir = base_dir + "/.partial";
        if (mkdir_p(partial_dir.c_str(), 0777) != 0) {
            const char* const msg = strerror(errno);
            FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", partial_dir.c_str(), msg);
        }
    }

    // Main stage thread
    while (!stop_thread) {

        // Wait for the next frame
        const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, in_frame_id);
        if (!frame)
            break;

        // Fetch metadata and N2 frame view
        N2FrameView fv(buffer, in_frame_id);
        const std::shared_ptr<N2Metadata> meta = get_N2_metadata(buffer, in_frame_id);
        assert(meta);

        // Start timer
        const double frame_recv_time = current_time();
        const double total_elapsed_time = frame_recv_time - start_time;
        INFO("Received buffer {} frame {} (duration_s {})", unique_name, in_frame_id,
             total_elapsed_time);

        const std::string final_path = _get_gdal_vis_filename(meta);
        const std::string file_name_only = _basename(final_path);
        const std::string partial_path = _get_partial_dir() + "/" + file_name_only;

        // Ensure dataset exists/open
        gdalVisFileData* gdalVisFileData_ptr = nullptr;
        auto ds = datasets.find(final_path);
        struct stat filecheck_buffer {};
        if (ds != datasets.end()) {
            // Dataset already open
            gdalVisFileData_ptr = ds->second.get();
        } else if (stat(final_path.c_str(), &filecheck_buffer) == 0) {
            // If final file already exists, drop/ignore this frame (late arrival)
            WARN("Finalized file exists for this frame's file window, dropping late frame: {:s}",
                 final_path);

            // Mark frame as done and continue
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;

            // Before continuing, check for grace-based finalizations on other datasets
            _grace_finalize_datasets(datasets);

            continue;
        } else {
            // Create new dataset (or look for .partial)

            GDALDataset* dataset = nullptr;
            GUInt64 file_nt = 0;
            if (stat(partial_path.c_str(), &filecheck_buffer) == 0) {
                // Open existing partial
                DEBUG("Opening existing partial file: {:s}", partial_path);
                char** open_options = nullptr;
                dataset = static_cast<GDALDataset*>(
                    GDALOpenEx(partial_path.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_UPDATE,
                               nullptr, const_cast<const char**>(open_options), nullptr));
            } else {
                // Create dataset container under .partial
                char** options = nullptr;
                if (format == std::string("zarr")) {
                    options = CSLAddString(options, "FORMAT=ZARR_V2");
                }
                dataset = driver->CreateMultiDimensional(partial_path.c_str(), nullptr,
                                                         const_cast<const char**>(options));
                CSLDestroy(options);
                if (!dataset)
                    FATAL_ERROR("Could not initialize GDAL file {:s}", partial_path);
                DEBUG("New partial dataset created: {:s}", partial_path);

                // Compute per-file-window frames dimension
                const std::uint64_t tick_len_ns = _get_tick_len_ns();
                if (meta->frame_length_fpga_ticks == 0 || tick_len_ns == 0)
                    FATAL_ERROR("Invalid frame_length_fpga_ticks or tick length.");
                const std::uint64_t frame_len_ns = meta->frame_length_fpga_ticks * tick_len_ns;
                const std::uint64_t W_ns = file_seconds * 1'000'000'000ULL;

                // Allow non-integer multiples by rounding up the number of frame slots,
                // so that all frames with start < window_end fit within 0..file_nt-1.
                if ((W_ns % frame_len_ns) != 0) {
                    WARN("Configured file_seconds incompatible with frame cadence: W={} ns, "
                         "frame_len={} ns; using file_nt=ceil(W/frame_len).",
                         W_ns, frame_len_ns);
                }
                file_nt = (W_ns + frame_len_ns - 1) / frame_len_ns;
                _initialize_gdal_vis_file(dataset, meta, file_nt);
            }

            // Create gdalVisFileData object for file
            const std::uint64_t tick_len_ns = _get_tick_len_ns();
            if (meta->frame_length_fpga_ticks == 0 || tick_len_ns == 0)
                FATAL_ERROR("Invalid frame_length_fpga_ticks or tick length.");
            const std::uint64_t frame_len_ns2 = meta->frame_length_fpga_ticks * tick_len_ns;
            const std::uint64_t file_start_ns = _get_file_start_time_ns(meta);
            auto gdalVisFileData_obj = std::make_unique<gdalVisFileData>(
                file_nt, meta->nfreq, meta->num_elements, meta->num_prod, meta->num_ev,
                frame_len_ns2, meta->frame_length_fpga_ticks, file_start_ns, partial_path,
                frame_recv_time);
            gdalVisFileData_obj->gdal_dataset = dataset;
            datasets.emplace(final_path, std::move(gdalVisFileData_obj));
            gdalVisFileData_ptr = datasets.find(final_path)->second.get();

            // Warn (once) if the file time span is less than one second.
            if (!warned_short_file_window && gdalVisFileData_ptr) {
                const std::uint64_t file_len_ns =
                    gdalVisFileData_ptr->num_file_t * gdalVisFileData_ptr->frame_len_ns;
                if (gdalVisFileData_ptr->frame_len_ns > 0 && file_len_ns < 1'000'000'000ULL) {
                    WARN("File window is < 1s ({} * {} ns = {} ns). Ensure downstream tools "
                         "handle sub-second file windows; consider increasing file_seconds or "
                         "cadence.",
                         (unsigned long long)(gdalVisFileData_ptr->num_file_t),
                         gdalVisFileData_ptr->frame_len_ns, file_len_ns);
                    warned_short_file_window = true;
                }
            }
        }

        if (!gdalVisFileData_ptr || !gdalVisFileData_ptr->gdal_dataset) {
            FATAL_ERROR("Dataset is null. Failed to open dataset.");
            return;
        }

        // Validate N2 buffer dimensions consistency for this dataset
        if (meta->nfreq != gdalVisFileData_ptr->num_freq
            || meta->num_elements != gdalVisFileData_ptr->num_input
            || meta->num_prod != gdalVisFileData_ptr->num_prod
            || meta->num_ev != gdalVisFileData_ptr->num_ev) {
            ERROR(
                "Dropping frame due to buffer dimensions mismatch within dataset: nfreq {} vs {}, "
                "num_elements {} vs {}, num_prod {} vs {}, num_ev {} vs {}",
                meta->nfreq, gdalVisFileData_ptr->num_freq, meta->num_elements,
                gdalVisFileData_ptr->num_input, meta->num_prod, gdalVisFileData_ptr->num_prod,
                meta->num_ev, gdalVisFileData_ptr->num_ev);
            // Mark frame as done and skip further processing
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            // Continue to process potential grace finalizations for other datasets
            _grace_finalize_datasets(datasets);
            continue;
        }

        // Check consistent timestamping for this dataset
        const std::uint64_t this_file_start_ns = _get_file_start_time_ns(meta);
        if (gdalVisFileData_ptr->file_start_time_ns != 0
            && gdalVisFileData_ptr->file_start_time_ns != this_file_start_ns) {
            FATAL_ERROR("File start mismatch within dataset context: {} vs {}",
                        gdalVisFileData_ptr->file_start_time_ns, this_file_start_ns);
        }

        // Check frame lengths and compute t-index
        const std::uint64_t tick_len_ns2 = _get_tick_len_ns();
        if (meta->frame_length_fpga_ticks == 0 || tick_len_ns2 == 0)
            FATAL_ERROR("Invalid frame_length_fpga_ticks or tick length.");
        const std::uint64_t frame_len_ns = meta->frame_length_fpga_ticks * tick_len_ns2;
        if (frame_len_ns != gdalVisFileData_ptr->frame_len_ns) {
            FATAL_ERROR("frame_length_fpga_ticks changed within a file window: {} vs {}",
                        frame_len_ns, gdalVisFileData_ptr->frame_len_ns);
        }
        const std::uint64_t t_in_file =
            (meta->frame_start_time_ns - gdalVisFileData_ptr->file_start_time_ns) / frame_len_ns;
        gdalVisFileData_ptr->add_frame(fv, meta, t_in_file);
        gdalVisFileData_ptr->last_update_wall_s = frame_recv_time;

        // If buffer full, flush
        double elapsed_writing_frame = 0.0;
        if (gdalVisFileData_ptr->full()) {
            const double t0 = current_time();
            gdalVisFileData_ptr->flush();
            const double t1 = current_time();
            elapsed_writing_frame = t1 - t0;
            // Close dataset after flush
            gdalVisFileData_ptr->close();
            // Finalize: rename partial to final
            int r = std::rename(gdalVisFileData_ptr->partial_path.c_str(), final_path.c_str());
            if (r != 0) {
                const char* msg = strerror(errno);
                ERROR("Failed to rename partial dataset to final: {} -> {}: {}",
                      gdalVisFileData_ptr->partial_path, final_path, msg);
            }
            datasets.erase(final_path);
        }

        // Stop timer/metrics
        const double currt = current_time();
        if (elapsed_writing_frame <= 0.0)
            elapsed_writing_frame = currt - frame_recv_time;
        write_time_metric.set(elapsed_writing_frame);
        avg_write_time =
            (avg_write_time * frame_counter + elapsed_writing_frame) / double(frame_counter + 1);
        n_datasets_metric.set(datasets.size());

        // Mark frame as done
        buffer->mark_frame_empty(unique_name, in_frame_id);

        if (max_frames >= 0 && frame_counter + 1 >= max_frames) {
            WARN("Processed {} frames with average write time {}, shutting down Kotekan",
                 frame_counter + 1, avg_write_time);
            exit_kotekan(CLEAN_EXIT);
            break;
        }
        frame_counter++;
        in_frame_id++;

        // After handling this frame, scan for grace-based finalizations
        _grace_finalize_datasets(datasets);

    } // while !stop_thread

    // Flush any partially-filled datasets on exit
    for (auto& kv : datasets) {
        const std::string& path = kv.first;
        auto& obj = *kv.second;
        if (obj.gdal_dataset) {
            obj.flush();
        }
        if (obj.gdal_dataset)
            GDALClose(obj.gdal_dataset);
        // Finalize rename
        int r = std::rename(obj.partial_path.c_str(), path.c_str());
        if (r != 0) {
            const char* msg = strerror(errno);
            ERROR("Failed to rename partial dataset to final on exit: {} -> {}: {}",
                  obj.partial_path, path, msg);
        }
    }
    datasets.clear();

    DEBUG("exiting");
}
