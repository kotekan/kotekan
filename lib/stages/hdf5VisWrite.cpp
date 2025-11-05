#include "hdf5VisWrite.hpp"

#include "Telescope.hpp" // for Telescope
#include "hdf5Files.hpp"
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
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <errors.h>
#include <fstream>
#include <highfive/H5File.hpp>
#include "H5Support.hpp"
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

using namespace HighFive;

REGISTER_KOTEKAN_STAGE(hdf5VisWrite);

hdf5VisWrite::hdf5VisWrite(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          [](const kotekan::Stage& stage) {
              return const_cast<kotekan::Stage&>(stage).main_thread();
          }),
    base_dir(config.get<std::string>(unique_name, "base_dir")),
    file_name(config.get<std::string>(unique_name, "file_name")),
    prefix_hostname(config.get_default<bool>(unique_name, "prefix_hostname", true)),
    format(std::string("hdf5")),
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

    buffer->register_consumer(unique_name);

    // Validate file window configuration
    if (file_seconds == 0) {
        FATAL_ERROR("file_seconds must be > 0 for hdf5VisWrite");
    }
    const std::uint64_t day_seconds = 86400ULL;
    if ((day_seconds % file_seconds) != 0) {
        FATAL_ERROR("file_seconds={} must evenly divide 86400.", file_seconds);
    }
}

hdf5VisWrite::~hdf5VisWrite() {}


std::uint64_t hdf5VisWrite::_get_file_start_time_ns(const std::shared_ptr<const N2Metadata> meta) {
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

std::string hdf5VisWrite::_get_vis_filename(std::shared_ptr<const N2Metadata> meta) {
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
    buf << ".h5";
    return buf.str();
}

std::vector<std::string>
hdf5VisWrite::_get_array_create_options(const std::vector<std::uint64_t>&) const {
    // Not used in HighFive path; kept for interface compatibility.
    return {};
}

// HighFive helpers to create datasets
static void _h5_create_dataset_2d(HighFive::File& f, const std::string& name, size_t dim0,
                                  size_t dim1, const HighFive::DataType& dtype) {
    if (f.exist(name))
        return;
    HighFive::DataSpace space({dim0, dim1});
    (void)f.createDataSet(name, space, dtype);
}

static void _h5_create_dataset_3d(HighFive::File& f, const std::string& name, size_t dim0,
                                  size_t dim1, size_t dim2, const HighFive::DataType& dtype) {
    if (f.exist(name))
        return;
    HighFive::DataSpace space({dim0, dim1, dim2});
    (void)f.createDataSet(name, space, dtype);
}

static void _h5_create_dataset_4d(HighFive::File& f, const std::string& name, size_t dim0,
                                  size_t dim1, size_t dim2, size_t dim3,
                                  const HighFive::DataType& dtype) {
    if (f.exist(name))
        return;
    HighFive::DataSpace space({dim0, dim1, dim2, dim3});
    (void)f.createDataSet(name, space, dtype);
}

static void _initialize_h5(HighFive::File& file, const std::shared_ptr<const N2Metadata> meta,
                           std::uint64_t file_nt) {
    // minimal file-level attributes
    file.createAttribute("num_elements", meta->num_elements);
    file.createAttribute("num_prod", meta->num_prod);
    file.createAttribute("num_ev", meta->num_ev);
    file.createAttribute("num_freq", meta->nfreq);
    file.createAttribute("frame_length_fpga_ticks", meta->frame_length_fpga_ticks);

    _h5_create_dataset_3d(file, "/vis_array", meta->nfreq, meta->num_prod, file_nt,
                          HighFive::create_datatype<cfloat>());
    _h5_create_dataset_3d(file, "/weights_array", meta->nfreq, meta->num_prod, file_nt,
                          HighFive::create_datatype<float>());
    _h5_create_dataset_3d(file, "/eval_array", meta->nfreq, meta->num_ev, file_nt,
                          HighFive::create_datatype<float>());
    _h5_create_dataset_4d(file, "/evec_array", meta->nfreq, meta->num_ev, meta->num_elements,
                          file_nt, HighFive::create_datatype<cfloat>());
    _h5_create_dataset_2d(file, "/erms_array", meta->nfreq, file_nt,
                          HighFive::create_datatype<float>());
    _h5_create_dataset_3d(file, "/gain_array", meta->nfreq, meta->num_elements, file_nt,
                          HighFive::create_datatype<cfloat>());
    _h5_create_dataset_3d(file, "/flags_array", meta->nfreq, meta->num_elements, file_nt,
                          HighFive::create_datatype<float>());
    _h5_create_dataset_2d(file, "/frac_lost_array", meta->nfreq, file_nt,
                          HighFive::create_datatype<float>());
    _h5_create_dataset_2d(file, "/frac_rfi_array", meta->nfreq, file_nt,
                          HighFive::create_datatype<float>());
    _h5_create_dataset_2d(file, "/n_valid_fpga_ticks", meta->nfreq, file_nt,
                          HighFive::create_datatype<uint64_t>());
    _h5_create_dataset_2d(file, "/n_rfi_fpga_ticks", meta->nfreq, file_nt,
                          HighFive::create_datatype<uint64_t>());
    if (!file.exist("/fpga_start_tick"))
        (void)file.createDataSet("/fpga_start_tick", HighFive::DataSpace({file_nt}),
                                 HighFive::create_datatype<uint64_t>());
    if (!file.exist("/frame_start_time_ns"))
        (void)file.createDataSet("/frame_start_time_ns", HighFive::DataSpace({file_nt}),
                                 HighFive::create_datatype<uint64_t>());
    if (!file.exist("/frame_length_fpga_ticks"))
        (void)file.createDataSet("/frame_length_fpga_ticks", HighFive::DataSpace({file_nt}),
                                 HighFive::create_datatype<uint64_t>());
    if (!file.exist("/era_deg"))
        (void)file.createDataSet("/era_deg", HighFive::DataSpace({file_nt}),
                                 HighFive::create_datatype<double>());
}

void hdf5VisWrite::_grace_finalize_datasets(
    std::map<std::string, std::unique_ptr<visFileData>>& datasets) {
    const double now_s = current_time();
    for (auto ds_it = datasets.begin(); ds_it != datasets.end(); ++ds_it) {
        auto& obj = *ds_it->second;
        if (now_s - obj.last_update_wall_s >= double(late_frame_grace_seconds)) {
            // Grace finalize: flush and rename
            obj.flush();
            obj.close();
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

void hdf5VisWrite::main_thread() {
    auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_viswrite_write_time_seconds", unique_name);
    double avg_write_time = 0.0;
    auto& n_datasets_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_viswrite_n_datasets", unique_name);

    const double start_time = current_time(); // for logging elapsed time
    N2::frameID in_frame_id(buffer);          // Input frame ID tracker
    int frame_counter = 0;                    // Count of frames written
    bool warned_short_file_window = false;    // Warn only once for short file windows

    // datasets (files) for writing (multiple may be open simultaneously)
    std::map<std::string, std::unique_ptr<visFileData>> datasets;

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

        const std::string final_path = _get_vis_filename(meta);
        const std::string file_name_only = _basename(final_path);
        const std::string partial_path = _get_partial_dir() + "/" + file_name_only;

        // Ensure dataset exists/open
        visFileData* visFileData_ptr = nullptr;
        auto ds = datasets.find(final_path);
        struct stat filecheck_buffer {};
        if (ds != datasets.end()) {
            // Dataset already open
            visFileData_ptr = ds->second.get();
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
            std::uint64_t file_nt = 0;

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

            const bool exists = (stat(partial_path.c_str(), &filecheck_buffer) == 0);
            if (!exists) {
                DEBUG("New partial dataset created: {:s}", partial_path);
            } else {
                DEBUG("Opening existing partial file: {:s}", partial_path);
            }

            // Create visFileData object for file
            const std::uint64_t frame_len_ns2 = meta->frame_length_fpga_ticks * tick_len_ns;
            const std::uint64_t file_start_ns = _get_file_start_time_ns(meta);
            auto visFileData_obj = std::make_unique<visFileData>(
                file_nt, meta->nfreq, meta->num_elements, meta->num_prod, meta->num_ev,
                frame_len_ns2, meta->frame_length_fpga_ticks, file_start_ns, partial_path,
                frame_recv_time);
            visFileData_obj->h5_file = std::make_unique<HighFive::File>(
                partial_path, HighFive::File::ReadWrite | HighFive::File::Create);
            if (!exists)
                _initialize_h5(*visFileData_obj->h5_file, meta, file_nt);
            datasets.emplace(final_path, std::move(visFileData_obj));
            visFileData_ptr = datasets.find(final_path)->second.get();

            // Warn (once) if the file time span is less than one second.
            if (!warned_short_file_window && visFileData_ptr) {
                const std::uint64_t file_len_ns =
                    visFileData_ptr->num_file_t * visFileData_ptr->frame_len_ns;
                if (visFileData_ptr->frame_len_ns > 0 && file_len_ns < 1'000'000'000ULL) {
                    WARN("File window is < 1s ({} * {} ns = {} ns). Ensure downstream tools "
                         "handle sub-second file windows; consider increasing file_seconds or "
                         "cadence.",
                         (unsigned long long)(visFileData_ptr->num_file_t),
                         visFileData_ptr->frame_len_ns, file_len_ns);
                    warned_short_file_window = true;
                }
            }
        }

        if (!visFileData_ptr || !visFileData_ptr->h5_file) {
            // Avoid stranding the producer frame on failure
            buffer->mark_frame_empty(unique_name, in_frame_id);
            FATAL_ERROR("Dataset is null. Failed to open dataset.");
            return;
        }

        // Validate N2 buffer dimensions consistency for this dataset
        if (meta->nfreq != visFileData_ptr->num_freq
            || meta->num_elements != visFileData_ptr->num_input
            || meta->num_prod != visFileData_ptr->num_prod
            || meta->num_ev != visFileData_ptr->num_ev) {
            ERROR(
                "Dropping frame due to buffer dimensions mismatch within dataset: nfreq {} vs {}, "
                "num_elements {} vs {}, num_prod {} vs {}, num_ev {} vs {}",
                meta->nfreq, visFileData_ptr->num_freq, meta->num_elements,
                visFileData_ptr->num_input, meta->num_prod, visFileData_ptr->num_prod,
                meta->num_ev, visFileData_ptr->num_ev);
            // Mark frame as done and skip further processing
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            // Continue to process potential grace finalizations for other datasets
            _grace_finalize_datasets(datasets);
            continue;
        }

        // Check consistent timestamping for this dataset
        const std::uint64_t this_file_start_ns = _get_file_start_time_ns(meta);
        if (visFileData_ptr->file_start_time_ns != 0
            && visFileData_ptr->file_start_time_ns != this_file_start_ns) {
            FATAL_ERROR("File start mismatch within dataset context: {} vs {}",
                        visFileData_ptr->file_start_time_ns, this_file_start_ns);
        }

        // Check frame lengths and compute t-index
        const std::uint64_t tick_len_ns2 = _get_tick_len_ns();
        if (meta->frame_length_fpga_ticks == 0 || tick_len_ns2 == 0)
            FATAL_ERROR("Invalid frame_length_fpga_ticks or tick length.");
        const std::uint64_t frame_len_ns = meta->frame_length_fpga_ticks * tick_len_ns2;
        if (frame_len_ns != visFileData_ptr->frame_len_ns) {
            FATAL_ERROR("frame_length_fpga_ticks changed within a file window: {} vs {}",
                        frame_len_ns, visFileData_ptr->frame_len_ns);
        }
        const std::uint64_t t_in_file =
            (meta->frame_start_time_ns - visFileData_ptr->file_start_time_ns) / frame_len_ns;
        visFileData_ptr->add_frame(fv, meta, t_in_file);
        visFileData_ptr->last_update_wall_s = frame_recv_time;

        // If buffer full, flush
        double elapsed_writing_frame = 0.0;
        if (visFileData_ptr->full()) {
            const double t0 = current_time();
            visFileData_ptr->flush();
            const double t1 = current_time();
            elapsed_writing_frame = t1 - t0;
            // Close dataset after flush
            visFileData_ptr->close();
            // Finalize: rename partial to final
            int r = std::rename(visFileData_ptr->partial_path.c_str(), final_path.c_str());
            if (r != 0) {
                const char* msg = strerror(errno);
                ERROR("Failed to rename partial dataset to final: {} -> {}: {}",
                      visFileData_ptr->partial_path, final_path, msg);
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
        obj.flush();
        obj.close();
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

// ---------------- HDF5 write helpers (visFileData methods) ----------------

bool visFileData::flush() {
    if (!h5_file)
        return false;
    const size_t nt = num_file_t;

    // Build nested containers and write
    // vis: [nfreq][num_prod][nt]
    {
        std::vector<std::vector<std::vector<cfloat>>> buf(
            num_freq, std::vector<std::vector<cfloat>>(num_prod, std::vector<cfloat>(nt)));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t p = 0; p < num_prod; ++p)
                for (size_t t = 0; t < nt; ++t)
                    buf[f][p][t] = vis[num_file_t * (num_prod * f + p) + t];
        h5_file->getDataSet("/vis_array").write(buf);
    }
    // weights: same shape
    {
        std::vector<std::vector<std::vector<float>>> buf(
            num_freq, std::vector<std::vector<float>>(num_prod, std::vector<float>(nt)));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t p = 0; p < num_prod; ++p)
                for (size_t t = 0; t < nt; ++t)
                    buf[f][p][t] = vis_weight[num_file_t * (num_prod * f + p) + t];
        h5_file->getDataSet("/weights_array").write(buf);
    }
    // eval: [nfreq][num_ev][nt]
    {
        std::vector<std::vector<std::vector<float>>> buf(
            num_freq, std::vector<std::vector<float>>(num_ev, std::vector<float>(nt)));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t e = 0; e < num_ev; ++e)
                for (size_t t = 0; t < nt; ++t)
                    buf[f][e][t] = eval[num_file_t * (num_ev * f + e) + t];
        h5_file->getDataSet("/eval_array").write(buf);
    }
    // evec: [nfreq][num_ev][num_input][nt]
    {
        std::vector<std::vector<std::vector<std::vector<cfloat>>>> buf(
            num_freq, std::vector<std::vector<std::vector<cfloat>>>(
                          num_ev, std::vector<std::vector<cfloat>>(num_input, std::vector<cfloat>(nt))));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t e = 0; e < num_ev; ++e)
                for (size_t i = 0; i < num_input; ++i)
                    for (size_t t = 0; t < nt; ++t)
                        buf[f][e][i][t] = evec[num_file_t * (num_input * (num_ev * f + e) + i) + t];
        h5_file->getDataSet("/evec_array").write(buf);
    }
    // erms: [nfreq][nt]
    {
        std::vector<std::vector<float>> buf(num_freq, std::vector<float>(nt));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t t = 0; t < nt; ++t)
                buf[f][t] = erms[num_file_t * f + t];
        h5_file->getDataSet("/erms_array").write(buf);
    }
    // gain: [nfreq][num_input][nt]
    {
        std::vector<std::vector<std::vector<cfloat>>> buf(
            num_freq, std::vector<std::vector<cfloat>>(num_input, std::vector<cfloat>(nt)));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t i = 0; i < num_input; ++i)
                for (size_t t = 0; t < nt; ++t)
                    buf[f][i][t] = gain[num_file_t * (num_input * f + i) + t];
        h5_file->getDataSet("/gain_array").write(buf);
    }
    // flags: [nfreq][num_input][nt]
    {
        std::vector<std::vector<std::vector<float>>> buf(
            num_freq, std::vector<std::vector<float>>(num_input, std::vector<float>(nt)));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t i = 0; i < num_input; ++i)
                for (size_t t = 0; t < nt; ++t)
                    buf[f][i][t] = flags[num_file_t * (num_input * f + i) + t];
        h5_file->getDataSet("/flags_array").write(buf);
    }
    // frac_lost / frac_rfi: [nfreq][nt]
    {
        std::vector<std::vector<float>> bl(num_freq, std::vector<float>(nt));
        std::vector<std::vector<float>> br(num_freq, std::vector<float>(nt));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t t = 0; t < nt; ++t) {
                const size_t idx = num_file_t * f + t;
                bl[f][t] = frac_lost[idx];
                br[f][t] = frac_rfi[idx];
            }
        h5_file->getDataSet("/frac_lost_array").write(bl);
        h5_file->getDataSet("/frac_rfi_array").write(br);
    }
    // counts: [nfreq][nt]
    {
        std::vector<std::vector<uint64_t>> bn(num_freq, std::vector<uint64_t>(nt));
        std::vector<std::vector<uint64_t>> br(num_freq, std::vector<uint64_t>(nt));
        for (size_t f = 0; f < num_freq; ++f)
            for (size_t t = 0; t < nt; ++t) {
                const size_t idx = num_file_t * f + t;
                bn[f][t] = n_valid_fpga_ticks[idx];
                br[f][t] = n_rfi_fpga_ticks[idx];
            }
        h5_file->getDataSet("/n_valid_fpga_ticks").write(bn);
        h5_file->getDataSet("/n_rfi_fpga_ticks").write(br);
    }
    // per-time arrays
    h5_file->getDataSet("/fpga_start_tick").write(fpga_start_tick);
    h5_file->getDataSet("/frame_start_time_ns").write(frame_start_time_ns);
    std::vector<uint64_t> flft(nt, frame_length_fpga_ticks);
    h5_file->getDataSet("/frame_length_fpga_ticks").write(flft);
    h5_file->getDataSet("/era_deg").write(era_deg);

    return true;
}

void visFileData::close() {
    if (h5_file)
        h5_file.reset();
}
