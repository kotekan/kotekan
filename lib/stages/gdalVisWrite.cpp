#include "gdalVisWrite.hpp"

#include "Telescope.hpp" // for Telescope
#include "gdalFiles.hpp"
#include "util.h" // for mkdir_p

#include <N2FrameView.hpp>
#include <N2Metadata.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <complex>
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

// Use mkdir_p from util.h for recursive directory creation

gdalVisWrite::gdalVisWrite(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          [](const kotekan::Stage& stage) {
              return const_cast<kotekan::Stage&>(stage).main_thread();
          }),
    base_dir(config.get<std::string>(unique_name, "base_dir")),
    file_name(config.get<std::string>(unique_name, "file_name")),
    prefix_hostname(config.get_default<bool>(unique_name, "prefix_hostname", true)),
    zip_compression(config.get_default<std::uint64_t>(unique_name, "zip_compression", 0)),
    blocksize_f(config.get_default<std::uint64_t>(unique_name, "blocksize_f", 0)),
    blocksize_p(config.get_default<std::uint64_t>(unique_name, "blocksize_p", 0)),
    blocksize_t(config.get_default<std::uint64_t>(unique_name, "blocksize_t", 1)),
    flush_timeout_seconds(
        config.get_default<std::uint64_t>(unique_name, "flush_timeout_seconds", 600)),
    max_frames(config.get_default<int>(unique_name, "max_frames", -1)),
    num_file_t(config.get_default<std::uint32_t>(unique_name, "num_file_t", 2)),
    buffer(get_buffer("in_buf")) {
    GDALAllRegister();
    // override for telescope sequence length (testing)
    const_cast<std::uint64_t&>(tick_len_ns_override) =
        config.get_default<std::uint64_t>(unique_name, "seq_length_nsec_override", 0);
    buffer->register_consumer(unique_name);
}

gdalVisWrite::~gdalVisWrite() {}

std::uint64_t gdalVisWrite::_get_frame_nt_in_file(const std::shared_ptr<const N2Metadata> meta) {
    const std::uint64_t tick_len_ns = _get_tick_len_ns();
    const std::uint64_t frame_len_ticks = meta->frame_length_fpga_ticks;
    if (frame_len_ticks == 0 || tick_len_ns == 0) {
        // Avoid division by zero for malformed/empty metadata; default to first slot.
        return 0;
    }
    const std::uint64_t frame_len_ns = frame_len_ticks * tick_len_ns;
    const std::uint64_t file_len_ns = frame_len_ns * num_file_t;
    const std::uint64_t frame_nt_in_file = (meta->frame_start_time_ns % file_len_ns) / frame_len_ns;
    return frame_nt_in_file;
}

std::uint64_t gdalVisWrite::_get_file_start_time_ns(const std::shared_ptr<const N2Metadata> meta) {
    const std::uint64_t tick_len_ns = _get_tick_len_ns();
    const std::uint64_t frame_len_ticks = meta->frame_length_fpga_ticks;
    if (frame_len_ticks == 0 || tick_len_ns == 0) {
        // Fallback: use the frame start time directly as the file start when unknown.
        return meta->frame_start_time_ns;
    }
    const std::uint64_t frame_len_ns = frame_len_ticks * tick_len_ns;
    const std::uint64_t file_len_ns = frame_len_ns * num_file_t;
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
    const std::uint64_t ns_part = file_start_time_ns % 1'000'000'000ULL; // sub-second
    buf << file_name << "." << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT%H%M%S");
    // Include nanosecond suffix to avoid collisions for sub-second windows
    buf << "_" << std::setw(9) << std::setfill('0') << ns_part;
    buf << ".zarr";
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
    std::shared_ptr<GDALDimension> dim_freq =
        root_group->CreateDimension("freqs", "", "", meta->nfreq);
    std::shared_ptr<GDALDimension> dim_prod =
        root_group->CreateDimension("products", "", "", meta->num_prod);
    std::shared_ptr<GDALDimension> dim_frames =
        root_group->CreateDimension("frames", "", "", num_file_t);
    std::shared_ptr<GDALDimension> dim_inputs =
        root_group->CreateDimension("inputs", "", "", meta->num_elements);
    std::shared_ptr<GDALDimension> dim_ev = root_group->CreateDimension("ev", "", "", meta->num_ev);

    // Array creation options (vis/weights only for now).
    // Ensure positive chunk sizes always.
    std::vector<std::string> array_options;
    {
        const GUInt64 bs_f = (blocksize_f > 0) ? blocksize_f : 1;
        const GUInt64 bs_p = (blocksize_p > 0) ? blocksize_p : meta->num_prod;
        const GUInt64 bs_t = (blocksize_t > 0) ? blocksize_t : num_file_t;
        std::ostringstream bbuf;
        bbuf << "BLOCKSIZE=" << bs_f << "," << bs_p << "," << bs_t;
        array_options.push_back(bbuf.str());
    }
    const auto array_options_c = convert_to_cstring_list(array_options);

    // vis and weights: (freqs, products, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_prod, dim_frames};
        auto vis_array = root_group->CreateMDArray(
            "vis_array", dims, GDALExtendedDataType::Create(GDT_CFloat32), array_options_c.data());
        assert(vis_array && vis_array->GetDimensionCount() == 3);
        auto weights_array = root_group->CreateMDArray("weights_array", dims,
                                                       GDALExtendedDataType::Create(GDT_Float32),
                                                       array_options_c.data());
        assert(weights_array && weights_array->GetDimensionCount() == 3);
    }

    // eval: (freqs, ev, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev, dim_frames};
        auto eval_array = root_group->CreateMDArray(
            "eval_array", dims, GDALExtendedDataType::Create(GDT_Float32), nullptr);
        (void)eval_array;
    }
    // evec: (freqs, ev, inputs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_ev, dim_inputs, dim_frames};
        auto evec_array = root_group->CreateMDArray(
            "evec_array", dims, GDALExtendedDataType::Create(GDT_CFloat32), nullptr);
        (void)evec_array;
    }
    // erms: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        auto erms_array = root_group->CreateMDArray(
            "erms_array", dims, GDALExtendedDataType::Create(GDT_Float32), nullptr);
        (void)erms_array;
    }
    // gain + flags: (freqs, inputs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_inputs, dim_frames};
        auto gain_array = root_group->CreateMDArray(
            "gain_array", dims, GDALExtendedDataType::Create(GDT_CFloat32), nullptr);
        (void)gain_array;
        auto flags_array = root_group->CreateMDArray(
            "flags_array", dims, GDALExtendedDataType::Create(GDT_Float32), nullptr);
        (void)flags_array;
    }
    // frac_*: (freqs, frames)
    {
        std::vector<std::shared_ptr<GDALDimension>> dims{dim_freq, dim_frames};
        auto fl_array = root_group->CreateMDArray(
            "frac_lost_array", dims, GDALExtendedDataType::Create(GDT_Float32), nullptr);
        (void)fl_array;
        auto fr_array = root_group->CreateMDArray(
            "frac_rfi_array", dims, GDALExtendedDataType::Create(GDT_Float32), nullptr);
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
                                        GDALExtendedDataType::Create(GDT_UInt64), nullptr);
        (void)root_group->CreateMDArray("n_rfi_fpga_ticks", dims,
                                        GDALExtendedDataType::Create(GDT_UInt64), nullptr);
    }
}

void gdalVisWrite::main_thread() {
    auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gdalviswrite_write_time_seconds", unique_name);
    double avg_write_time = 0.0;
    auto& n_datasets_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gdalviswrite_n_datasets", unique_name);

    const double start_time = current_time();
    N2::frameID in_frame_id(buffer);
    int frame_counter = 0;
    bool warned_short_window = false;

    // datasets (files) for writing
    std::map<std::string, DatasetState> datasets;

    // Choose file format (driver)
    const auto driver_manager = GetGDALDriverManager();
    const std::string driver_name = "Zarr";
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

    auto flush_dataset = [&](const std::string& path, DatasetState& state) {
        (void)path;
        if (!state.ds || !state.buf)
            return;
        state.buf->write_all_to_dataset(state.ds);
    };

    auto close_dataset = [&](DatasetState& state) {
        if (state.ds) {
            GDALClose(state.ds);
            state.ds = nullptr;
        }
        state.buf.reset();
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
        INFO("Received buffer {} frame {} (duration_s {})", unique_name, in_frame_id,
             total_elapsed_time);

        const std::string final_path = _get_gdal_vis_filename(meta);
        const std::string file_name_only = _basename(final_path);
        const std::string partial_path = _get_partial_dir() + "/" + file_name_only;

        // Ensure dataset exists/open
        DatasetState* state_ptr = nullptr;
        auto it = datasets.find(final_path);
        if (it != datasets.end()) {
            state_ptr = &it->second;
        } else {
            // If final file already exists, drop/ignore this frame (late arrival)
            struct stat filecheck_buffer {};
            if (stat(final_path.c_str(), &filecheck_buffer) == 0) {
                WARN("Finalized file exists for this frame's window, dropping late frame: {:s}",
                     final_path);
                // Mark frame as done and continue
                buffer->mark_frame_empty(unique_name, in_frame_id);
                in_frame_id++;
                // Before continuing, check for timeouts on other datasets
                const double now_s = current_time();
                for (auto it2 = datasets.begin(); it2 != datasets.end();) {
                    auto& state = it2->second;
                    if (now_s - state.last_update_wall_s >= double(flush_timeout_seconds)) {
                        // Timeout finalize: flush and rename
                        if (state.ds && state.buf)
                            state.buf->write_all_to_dataset(state.ds);
                        if (state.ds) {
                            GDALClose(state.ds);
                            state.ds = nullptr;
                        }
                        // Attempt rename from partial to final
                        int r = std::rename(state.partial_path.c_str(), it2->first.c_str());
                        if (r != 0) {
                            const char* msg = strerror(errno);
                            ERROR("Failed to rename partial dataset to final: {} -> {}: {}",
                                  state.partial_path, it2->first, msg);
                        }
                        it2 = datasets.erase(it2);
                        continue;
                    }
                    ++it2;
                }
                continue;
            }

            // Open existing partial or create new under .partial
            GDALDataset* dataset = nullptr;
            if (stat(partial_path.c_str(), &filecheck_buffer) == 0) {
                DEBUG("Opening existing partial file: {:s}", partial_path);
                char** open_options = nullptr;
                dataset = static_cast<GDALDataset*>(
                    GDALOpenEx(partial_path.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_UPDATE,
                               nullptr, const_cast<const char**>(open_options), nullptr));
            } else {
                // Create Zarr dataset under partial path
                char** options = nullptr;
                options = CSLSetNameValue(options, "FORMAT", "ZARR_V2");
                if (zip_compression > 0) {
                    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
                    options =
                        CSLSetNameValue(options, "LEVEL", std::to_string(zip_compression).c_str());
                    options = CSLSetNameValue(options, "STORAGE", "ZIP");
                }
                dataset = driver->CreateMultiDimensional(partial_path.c_str(), nullptr,
                                                         const_cast<const char**>(options));
                CSLDestroy(options);
                if (!dataset)
                    FATAL_ERROR("Could not initialize GDAL file {:s}", partial_path);
                DEBUG("New partial dataset created: {:s}", partial_path);

                _initialize_gdal_vis_file(dataset, meta);
            }

            // Validate arrays and dimensions
            {
                auto root = dataset->GetRootGroup();
                if (!root)
                    FATAL_ERROR("Dataset missing root group: {}", partial_path);
                auto chk = [&](const char* name, int nd, std::initializer_list<GUInt64> sizes) {
                    auto arr = root->OpenMDArray(name);
                    if (!arr)
                        FATAL_ERROR("Missing array '{}' in dataset {}", name, partial_path);
                    if ((int)arr->GetDimensionCount() != nd)
                        FATAL_ERROR("Array '{}' has wrong rank in dataset {}: {} != {}", name,
                                    partial_path, (int)arr->GetDimensionCount(), nd);
                    auto dims = arr->GetDimensions();
                    size_t i = 0;
                    for (auto s : sizes) {
                        if (i >= dims.size())
                            FATAL_ERROR("Array '{}' missing dimension {} in dataset {}", name, i,
                                        partial_path);
                        if (dims[i]->GetSize() != s)
                            FATAL_ERROR("Array '{}' dim {} mismatch in dataset {}: {} != {}", name,
                                        i, partial_path, (unsigned long long)dims[i]->GetSize(),
                                        (unsigned long long)s);
                        ++i;
                    }
                };
                chk("vis_array", 3, {meta->nfreq, meta->num_prod, num_file_t});
                chk("weights_array", 3, {meta->nfreq, meta->num_prod, num_file_t});
                chk("eval_array", 3, {meta->nfreq, meta->num_ev, num_file_t});
                chk("evec_array", 4, {meta->nfreq, meta->num_ev, meta->num_elements, num_file_t});
                chk("erms_array", 2, {meta->nfreq, num_file_t});
                chk("gain_array", 3, {meta->nfreq, meta->num_elements, num_file_t});
                chk("flags_array", 3, {meta->nfreq, meta->num_elements, num_file_t});
                chk("frac_lost_array", 2, {meta->nfreq, num_file_t});
                chk("frac_rfi_array", 2, {meta->nfreq, num_file_t});
                chk("fpga_start_tick", 1, {num_file_t});
                chk("frame_start_time_ns", 1, {num_file_t});
                chk("frame_length_fpga_ticks", 1, {num_file_t});
                chk("era_deg", 1, {num_file_t});
                chk("n_valid_fpga_ticks", 2, {meta->nfreq, num_file_t});
                chk("n_rfi_fpga_ticks", 2, {meta->nfreq, num_file_t});
            }

            // Store state
            DatasetState state;
            state.ds = dataset;
            state.open_wall_s = frame_recv_time;
            state.last_update_wall_s = frame_recv_time;
            state.file_start_time_ns = _get_file_start_time_ns(meta);
            // Compute frame_len_ns for shape tracking
            {
                const std::uint64_t tick_len_ns = _get_tick_len_ns();
                state.frame_len_ns = meta->frame_length_fpga_ticks * tick_len_ns;
            }
            state.partial_path = partial_path;
            datasets.emplace(final_path, std::move(state));
            state_ptr = &datasets.find(final_path)->second;

            // Warn if the file time span is less than one second, as filenames are
            // only second-resolution and will collide across windows.
            if (!warned_short_window) {
                const std::uint64_t file_len_ns = state_ptr->frame_len_ns * num_file_t;
                if (state_ptr->frame_len_ns > 0 && file_len_ns < 1'000'000'000ULL) {
                    WARN("File window too short: num_file_t * frame_len = {} * {} ns = {} ns < 1s. "
                         "This may cause filename collisions. Consider increasing num_file_t or "
                         "cadence.",
                         num_file_t, state_ptr->frame_len_ns, file_len_ns);
                    warned_short_window = true;
                }
            }
        }

        if (!state_ptr || !state_ptr->ds) {
            FATAL_ERROR("Dataset is null. Failed to open dataset.");
            return;
        }

        // Allocate buffer if needed and capture expected geometry
        if (!state_ptr->buf) {
            state_ptr->buf = std::make_unique<gdalVisFileData>(
                num_file_t, meta->nfreq, meta->num_elements, meta->num_prod, meta->num_ev);
            state_ptr->expect_nfreq = meta->nfreq;
            state_ptr->expect_num_elements = meta->num_elements;
            state_ptr->expect_num_prod = meta->num_prod;
            state_ptr->expect_num_ev = meta->num_ev;
        } else {
            // Validate geometry consistency for this dataset
            if (meta->nfreq != state_ptr->expect_nfreq || meta->num_elements != state_ptr->expect_num_elements
                || meta->num_prod != state_ptr->expect_num_prod || meta->num_ev != state_ptr->expect_num_ev) {
                ERROR("Dropping frame due to geometry mismatch within dataset: nfreq {} vs {}, "
                      "num_elements {} vs {}, num_prod {} vs {}, num_ev {} vs {}",
                      meta->nfreq, state_ptr->expect_nfreq, meta->num_elements,
                      state_ptr->expect_num_elements, meta->num_prod, state_ptr->expect_num_prod,
                      meta->num_ev, state_ptr->expect_num_ev);
                // Mark frame as done and skip further processing
                buffer->mark_frame_empty(unique_name, in_frame_id);
                in_frame_id++;
                // Continue to process potential timeouts for other datasets
                const double now_s = current_time();
                for (auto it2 = datasets.begin(); it2 != datasets.end();) {
                    auto& state = it2->second;
                    if (now_s - state.last_update_wall_s >= double(flush_timeout_seconds)) {
                        if (state.ds && state.buf)
                            state.buf->write_all_to_dataset(state.ds);
                        if (state.ds) {
                            GDALClose(state.ds);
                            state.ds = nullptr;
                        }
                        int r = std::rename(state.partial_path.c_str(), it2->first.c_str());
                        if (r != 0) {
                            const char* msg = strerror(errno);
                            ERROR("Failed to rename partial dataset to final (timeout): {} -> {}: {}",
                                  state.partial_path, it2->first, msg);
                        }
                        it2 = datasets.erase(it2);
                        continue;
                    }
                    ++it2;
                }
                continue;
            }
        }

        // Basic validation: consistent time geometry for this dataset
        const std::uint64_t this_file_start_ns = _get_file_start_time_ns(meta);
        if (state_ptr->file_start_time_ns != 0
            && state_ptr->file_start_time_ns != this_file_start_ns) {
            FATAL_ERROR("File start mismatch within dataset context: {} vs {}",
                        state_ptr->file_start_time_ns, this_file_start_ns);
        }
        state_ptr->file_start_time_ns = this_file_start_ns;

        // Add frame into buffer
        const std::uint64_t t_in_file = _get_frame_nt_in_file(meta);
        state_ptr->buf->add_frame(fv, meta, t_in_file);
        state_ptr->last_update_wall_s = frame_recv_time;

        // If buffer full, flush
        double elapsed_writing_frame = 0.0;
        if (state_ptr->buf->full()) {
            const double t0 = current_time();
            flush_dataset(final_path, *state_ptr);
            const double t1 = current_time();
            elapsed_writing_frame = t1 - t0;
            // Close dataset after flush
            close_dataset(*state_ptr);
            // Finalize: rename partial to final
            int r = std::rename(state_ptr->partial_path.c_str(), final_path.c_str());
            if (r != 0) {
                const char* msg = strerror(errno);
                ERROR("Failed to rename partial dataset to final: {} -> {}: {}",
                      state_ptr->partial_path, final_path, msg);
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

        // After handling this frame, scan for timeouts
        const double now_s = current_time();
        for (auto it2 = datasets.begin(); it2 != datasets.end();) {
            auto& state = it2->second;
            if (now_s - state.last_update_wall_s >= double(flush_timeout_seconds)) {
                // Inactivity finalize
                if (state.ds && state.buf)
                    state.buf->write_all_to_dataset(state.ds);
                if (state.ds) {
                    GDALClose(state.ds);
                    state.ds = nullptr;
                }
                int r = std::rename(state.partial_path.c_str(), it2->first.c_str());
                if (r != 0) {
                    const char* msg = strerror(errno);
                    ERROR("Failed to rename partial dataset to final (timeout): {} -> {}: {}",
                          state.partial_path, it2->first, msg);
                }
                it2 = datasets.erase(it2);
                continue;
            }
            ++it2;
        }
    } // while !stop_thread

    // Flush any partially-filled datasets on exit
    for (auto& [path, state] : datasets) {
        if (state.ds && state.buf) {
            flush_dataset(path, state);
        }
        if (state.ds)
            GDALClose(state.ds);
        // Finalize rename
        int r = std::rename(state.partial_path.c_str(), path.c_str());
        if (r != 0) {
            const char* msg = strerror(errno);
            ERROR("Failed to rename partial dataset to final on exit: {} -> {}: {}",
                  state.partial_path, path, msg);
        }
    }
    datasets.clear();

    DEBUG("exiting");
}
