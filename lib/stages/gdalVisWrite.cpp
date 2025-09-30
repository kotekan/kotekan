#include "gdalFiles.hpp"
#include "gdalVisWrite.hpp"

#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <N2Metadata.hpp>
#include <N2FrameView.hpp>
#include <complex>
#include <cstdint>
#include <errno.h>
#include <cstring>
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

REGISTER_KOTEKAN_STAGE(gdalVisWrite);

gdalVisWrite::gdalVisWrite(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container)
    : Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
          return const_cast<kotekan::Stage&>(stage).main_thread();
      }),
      base_dir(config.get<std::string>(unique_name, "base_dir")),
      file_name(config.get<std::string>(unique_name, "file_name")),
      prefix_hostname(config.get_default<bool>(unique_name, "prefix_hostname", true)),
      zip_compression(config.get_default<std::uint64_t>(unique_name, "zip_compression", 0)),
      blocksize_f(config.get_default<std::uint64_t>(unique_name, "blocksize_f", 0)),
      blocksize_p(config.get_default<std::uint64_t>(unique_name, "blocksize_p", 0)),
      blocksize_t(config.get_default<std::uint64_t>(unique_name, "blocksize_t", 1)),
      max_frames(config.get_default<int>(unique_name, "max_frames", -1)),
      file_nt(config.get_default<std::uint32_t>(unique_name, "file_nt", 2)),
      buffer(get_buffer("in_buf")) {
    GDALAllRegister();
    buffer->register_consumer(unique_name);
}

gdalVisWrite::~gdalVisWrite() {}

std::uint64_t gdalVisWrite::_get_frame_nt_in_file(const std::shared_ptr<const N2Metadata> meta) {
    auto& tel = Telescope::instance();
    const std::uint64_t tick_len_ns = tel.seq_length_nsec();
    const std::uint64_t frame_len_ticks = meta->frame_length_fpga_ticks;
    if (frame_len_ticks == 0 || tick_len_ns == 0) {
        // Avoid division by zero for malformed/empty metadata; default to first slot.
        return 0;
    }
    const std::uint64_t frame_len_ns = frame_len_ticks * tick_len_ns;
    const std::uint64_t file_len_ns = frame_len_ns * file_nt;
    const std::uint64_t frame_nt_in_file = (meta->frame_start_time_ns % file_len_ns) / frame_len_ns;
    return frame_nt_in_file;
}

std::uint64_t gdalVisWrite::_get_file_start_time_ns(
    const std::shared_ptr<const N2Metadata> meta) {
    auto& tel = Telescope::instance();
    const std::uint64_t tick_len_ns = tel.seq_length_nsec();
    const std::uint64_t frame_len_ticks = meta->frame_length_fpga_ticks;
    if (frame_len_ticks == 0 || tick_len_ns == 0) {
        // Fallback: use the frame start time directly as the file start when unknown.
        return meta->frame_start_time_ns;
    }
    const std::uint64_t frame_len_ns = frame_len_ticks * tick_len_ns;
    const std::uint64_t file_len_ns = frame_len_ns * file_nt;
    const std::uint64_t file_start_time_ns =
        meta->frame_start_time_ns - (meta->frame_start_time_ns % file_len_ns);
    return file_start_time_ns;
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
    std::time_t time_t_format = file_start_time_ns / 1'000'000'000; // seconds
    buf << file_name << "." << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT%H%M%S")
        << ".zarr";
    if (zip_compression > 0)
        buf << ".zip";
    return buf.str();
}

void gdalVisWrite::_initialize_gdal_vis_file(GDALDataset* dataset,
                                             std::shared_ptr<const N2Metadata> meta) {
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
    }

    // Dimensions
    std::shared_ptr<GDALDimension> dim_freq = root_group->CreateDimension("freqs", "", "",
                                                                          meta->nfreq);
    std::shared_ptr<GDALDimension> dim_prod = root_group->CreateDimension("products", "", "",
                                                                          meta->num_prod);
    std::shared_ptr<GDALDimension> dim_frames = root_group->CreateDimension("frames", "", "",
                                                                            file_nt);
    std::shared_ptr<GDALDimension> dim_inputs = root_group->CreateDimension("inputs", "", "",
                                                                            meta->num_elements);
    std::shared_ptr<GDALDimension> dim_ev =
        root_group->CreateDimension("ev", "", "", meta->num_ev);

    // Array creation options (vis/weights only for now)
    std::vector<std::string> array_options;
    {
        std::ostringstream bbuf;
        bbuf << "BLOCKSIZE=" << blocksize_f << "," << meta->num_prod << "," << blocksize_t;
        if (blocksize_f > 0 || blocksize_t > 0) {
            array_options.push_back(bbuf.str());
        }
    }
    const auto array_options_c = convert_to_cstring_list(array_options);

    // vis and weights: (freqs, products, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_prod, dim_frames};
        auto vis_array = root_group->CreateMDArray("vis_array", dims,
                                                  GDALExtendedDataType::Create(GDT_CFloat32),
                                                  array_options_c.data());
        assert(vis_array && vis_array->GetDimensionCount() == 3);
        auto weights_array = root_group->CreateMDArray("weights_array", dims,
                                                      GDALExtendedDataType::Create(GDT_Float32),
                                                      array_options_c.data());
        assert(weights_array && weights_array->GetDimensionCount() == 3);
    }

    // eval: (freqs, ev, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev, dim_frames};
        auto eval_array = root_group->CreateMDArray("eval_array", dims,
                                                   GDALExtendedDataType::Create(GDT_Float32),
                                                   nullptr);
        (void)eval_array;
    }
    // evec: (freqs, ev, inputs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev, dim_inputs, dim_frames};
        auto evec_array = root_group->CreateMDArray("evec_array", dims,
                                                   GDALExtendedDataType::Create(GDT_CFloat32),
                                                   nullptr);
        (void)evec_array;
    }
    // erms: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        auto erms_array = root_group->CreateMDArray("erms_array", dims,
                                                   GDALExtendedDataType::Create(GDT_Float32),
                                                   nullptr);
        (void)erms_array;
    }
    // gain + flags: (freqs, inputs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_inputs, dim_frames};
        auto gain_array = root_group->CreateMDArray("gain_array", dims,
                                                   GDALExtendedDataType::Create(GDT_CFloat32),
                                                   nullptr);
        (void)gain_array;
        auto flags_array = root_group->CreateMDArray("flags_array", dims,
                                                    GDALExtendedDataType::Create(GDT_Float32),
                                                    nullptr);
        (void)flags_array;
    }
    // frac_*: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        auto fl_array = root_group->CreateMDArray("frac_lost_array", dims,
                                                 GDALExtendedDataType::Create(GDT_Float32),
                                                 nullptr);
        (void)fl_array;
        auto fr_array = root_group->CreateMDArray("frac_rfi_array", dims,
                                                 GDALExtendedDataType::Create(GDT_Float32),
                                                 nullptr);
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
        (void)root_group->CreateMDArray("era_deg", dims,
                                        GDALExtendedDataType::Create(GDT_Float64), nullptr);
    }
    // per-(freq,time) metadata arrays: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        (void)root_group->CreateMDArray("n_valid_fpga_ticks", dims,
                                        GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root_group->CreateMDArray("n_rfi_fpga_ticks", dims,
                                        GDALExtendedDataType::Create(GDT_UInt64), nullptr);
    }
}

void gdalVisWrite::main_thread() {
    auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gdalviswrite_write_time_seconds", unique_name);
    double avg_write_time = 0.0;
    auto& n_datasets_metric =
        kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_gdalviswrite_n_datasets", unique_name);

    const double start_time = current_time();
    N2::frameID in_frame_id(buffer);
    int frame_counter = 0;

    // datasets (files) for writing
    std::map<std::string, DatasetCtx> datasets;

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
            FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", base_dir.c_str(), msg);
        }
    }

    auto flush_dataset = [&](const std::string& path, DatasetCtx& ctx) {
        (void)path;
        if (!ctx.ds || !ctx.buf)
            return;
        ctx.buf->write_all_to_dataset(ctx.ds);
    };

    auto close_dataset = [&](DatasetCtx& ctx) {
        if (ctx.ds) {
            GDALClose(ctx.ds);
            ctx.ds = nullptr;
        }
        ctx.buf.reset();
    };

    while (!stop_thread) {
        // Wait for the next frame
        const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, in_frame_id);
        if (!frame)
            break;

        // Fetch metadata and frame view
        N2FrameView fv(buffer, in_frame_id);
        const std::shared_ptr<N2Metadata> meta = get_N2_metadata(buffer, in_frame_id);
        assert(meta);

        // Start timer
        const double frame_recv_time = current_time();
        const double total_elapsed_time = frame_recv_time - start_time;
        INFO("Received buffer {} frame {} (duration_ns {})", unique_name, in_frame_id,
             total_elapsed_time);

        const std::string full_path = _get_gdal_vis_filename(meta);

        // Ensure dataset exists/open
        DatasetCtx* ctx_ptr = nullptr;
        auto it = datasets.find(full_path);
        if (it != datasets.end()) {
            ctx_ptr = &it->second;
        } else {
            // Open existing or create
            GDALDataset* dataset = nullptr;
            struct stat filecheck_buffer {};
            if (stat(full_path.c_str(), &filecheck_buffer) == 0) {
                DEBUG("Opening existing file: {:s}", full_path);
                char** open_options = nullptr;
                dataset = static_cast<GDALDataset*>(
                    GDALOpenEx(full_path.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_UPDATE,
                               nullptr, const_cast<const char**>(open_options), nullptr));
            } else {
                // Create Zarr dataset
                char** options = nullptr;
                options = CSLSetNameValue(options, "FORMAT", "ZARR_V2");
                if (zip_compression > 0) {
                    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
                    options = CSLSetNameValue(options, "LEVEL",
                                              std::to_string(zip_compression).c_str());
                    options = CSLSetNameValue(options, "STORAGE", "ZIP");
                }
                dataset = driver->CreateMultiDimensional(full_path.c_str(), nullptr,
                                                         const_cast<const char**>(options));
                CSLDestroy(options);
                if (!dataset)
                    FATAL_ERROR("Could not initialize GDAL file {:s}", full_path);
                DEBUG("New dataset created for file: {:s}", full_path);

                _initialize_gdal_vis_file(dataset, meta);
            }

            // Store ctx
            DatasetCtx ctx;
            ctx.ds = dataset;
            datasets.emplace(full_path, std::move(ctx));
            ctx_ptr = &datasets.find(full_path)->second;
        }

        if (!ctx_ptr || !ctx_ptr->ds) {
            FATAL_ERROR("Dataset is null. Failed to open dataset.");
            return;
        }

        // Allocate buffer if needed
        if (!ctx_ptr->buf) {
            ctx_ptr->buf = std::make_unique<gdalVisFileData>(
                file_nt, meta->nfreq, meta->num_elements, meta->num_prod, meta->num_ev);
        }

        // Add frame into buffer
        const std::uint64_t t_in_file = _get_frame_nt_in_file(meta);
        ctx_ptr->buf->add_frame(fv, meta, t_in_file);

        // If buffer full, flush
        double elapsed_writing_frame = 0.0;
        if (ctx_ptr->buf->full()) {
            const double t0 = current_time();
            flush_dataset(full_path, *ctx_ptr);
            const double t1 = current_time();
            elapsed_writing_frame = t1 - t0;
            // Close dataset after flush
            close_dataset(*ctx_ptr);
            datasets.erase(full_path);
        }

        // Stop timer/metrics
        const double currt = current_time();
        if (elapsed_writing_frame <= 0.0)
            elapsed_writing_frame = currt - frame_recv_time;
        write_time_metric.set(elapsed_writing_frame);
        avg_write_time = (avg_write_time * frame_counter + elapsed_writing_frame)
                         / double(frame_counter + 1);
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
    } // while !stop_thread

    // Flush any partially-filled datasets on exit
    for (auto& [path, ctx] : datasets) {
        if (ctx.ds && ctx.buf) {
            flush_dataset(path, ctx);
        }
        if (ctx.ds)
            GDALClose(ctx.ds);
    }
    datasets.clear();

    DEBUG("exiting");
}
