#include "hdf5VisWrite.hpp"

#include "H5Support.hpp"
#include "Telescope.hpp" // for Telescope
#include "hdf5Files.hpp"
#include "util.h" // for mkdir_p

#include "json.hpp"

#include <N2FrameView.hpp>
#include <N2Metadata.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <algorithm>
#include <cassert>
#include <chordMetadata.hpp>
#include <chrono>
#include <complex>
#include <configTracker.hpp>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <errors.h>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Object.hpp>       // for H5Z_FLAG_MANDATORY, hsize_t
#include <highfive/H5PropertyList.hpp> // for PropertyType, RawPropertyList, Chunking
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

// Monotonic time in seconds
inline double mono_time_s() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    auto dt = clock::now() - t0;
    return std::chrono::duration<double>(dt).count();
}

bool visFileData::add_frame(const N2FrameView& fv, const std::shared_ptr<N2Metadata>& meta, size_t t_index) {
    const size_t f_index = meta->freq_id; // TODO: sync with telescope object. For now assume
                                          // 0..num_freq-1 indexing

    // Make sure frame hasn't been added yet
    size_t check_idx = idx_ft(f_index, t_index);
    if (added_ft[check_idx] != 0) {
        ERROR("visFileData: duplicate frame insertion at (f={}, t={})",
                                f_index, t_index);
        return false;
    }
    if (f_index >= num_freq || t_index >= num_file_t) {
        ERROR("visFileData: index out of bounds: f_index={} >= num_freq={}; t_index={} >= num_file_t={}",
                                f_index, num_freq, t_index, num_file_t);
        return false;
    }
    if(fv.vis.size() != N2::get_num_prod(meta->num_elements) ||
        fv.weight.size() != N2::get_num_prod(meta->num_elements) ||
        fv.eval.size() != meta->num_ev ||
        fv.evec.size() != meta->num_ev * meta->num_elements ||
        fv.gain.size() != meta->num_elements ||
        fv.flags.size() != meta->num_elements ||
        meta->num_elements != num_input ||
        meta->num_prod != num_prod ||
        meta->num_ev != num_ev ||
        fpga_start_tick[t_index] != meta->fpga_start_tick ||
        meta->frame_length_fpga_ticks > 0 ||
        frame_length_fpga_ticks[t_index] != meta->frame_length_fpga_ticks ||
        frame_ut1[t_index] != meta->frame_eop.t_ut1 ||
        bin_ut1[t_index] != meta->bin_eop.t_ut1) {
        ERROR("visFileData: frame information mismatch at (f={}, t={}): "
                "fv.vis.size()={}, fv.weight.size()={}, fv.eval.size()={}, fv.evec.size()={}, "
                "fv.gain.size()={}, fv.flags.size()={}, meta->num_elements={}, meta->num_prod={}, "
                "meta->num_ev={}, fpga_start_tick[t_index]={}, meta->fpga_start_tick={}, "
                "meta->frame_length_fpga_ticks={}, frame_length_fpga_ticks[t_index]={}, "
                "frame_ut1[t_index]={}, meta->frame_eop.t_ut1={}, bin_ut1[t_index]={}, meta->bin_eop.t_ut1={}",
                f_index, t_index,
                fv.vis.size(), fv.weight.size(), fv.eval.size(), fv.evec.size(),
                fv.gain.size(), fv.flags.size(), meta->num_elements, meta->num_prod,
                meta->num_ev, fpga_start_tick[t_index], meta->fpga_start_tick,
                meta->frame_length_fpga_ticks, frame_length_fpga_ticks[t_index],
                frame_ut1[t_index], meta->frame_eop.t_ut1, bin_ut1[t_index], meta->bin_eop.t_ut1);
        return false;
    }

    
    // Store vis + weight
    for (size_t p = 0; p < num_prod; ++p) {
        vis[idx_fpt(f_index, p, t_index)] = fv.vis[p];
        vis_weight[idx_fpt(f_index, p, t_index)] = fv.weight[p];
    }
    // Store eval + evec
    for (size_t e = 0; e < num_ev; ++e) {
        eval[idx_fet(f_index, e, t_index)] = fv.eval[e];
        for (size_t i = 0; i < num_input; ++i) {
            evec[idx_feit(f_index, e, i, t_index)] = fv.evec[num_input * e + i];
        }
    }
    // Store erms, gain, flags
    erms[idx_ft(f_index, t_index)] = fv.erms;
    for (size_t i = 0; i < num_input; ++i) {
        gain[idx_fit(f_index, i, t_index)] = fv.gain[i];
        flags[idx_fit(f_index, i, t_index)] = fv.flags[i];
    }
    // Store fraction lost and RFI
    const uint64_t frame_len_ticks = meta->frame_length_fpga_ticks;
    const uint64_t n_valid = meta->n_valid_fpga_ticks;
    const uint64_t n_rfi = meta->n_rfi_fpga_ticks;
    frac_lost[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (1.0f - float(n_valid) / float(frame_len_ticks)) : 0.0f;
    frac_rfi[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (float(n_rfi) / float(frame_len_ticks)) : 0.0f;
    // Store per-time metadata
    fpga_start_tick[t_index] = meta->fpga_start_tick;
    frame_length_fpga_ticks[t_index] = meta->frame_length_fpga_ticks;
    frame_ut1[t_index] = meta->frame_eop.t_ut1;
    bin_ut1[t_index] = meta->bin_eop.t_ut1;

    // Mark (f, t) as added
    size_t si = idx_ft(f_index, t_index);
    added_ft[si] = 1;
    ++added_count; // increment total number of frames added
    
    return true;
}

std::optional<std::string> visFileData::_get_final_filename() {
    // Get the earliest (non-zero) time in the bin ut1 array
    if (num_file_t == 0)
        return std::nullopt;
    
    std::optional<std::uint64_t> earliest_ut1_ns;
    for (size_t t = 0; t < num_file_t; ++t)
    {
        if (bin_ut1[t] != 0) {
            if (!earliest_ut1_ns.has_value() || bin_ut1[t] < earliest_ut1_ns.value()) {
                earliest_ut1_ns = bin_ut1[t];
            }
        }
    }
    if (!earliest_ut1_ns.has_value())
        return std::nullopt;

    // Construct final filename based on earliest_ut1_ns
    std::ostringstream buf;
    std::time_t time_t_format = earliest_ut1_ns.value() / 1'000'000'000;      // seconds
    const std::uint64_t ns_part = earliest_ut1_ns.value() % 1'000'000'000ULL; // sub-second
    buf << "vis_" << std::to_string(file_start_abs_frame_idx)  << "_" << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT%H%M%S");
    // Include nanosecond suffix to avoid collisions for sub-second file windows
    buf << "_" << std::setw(9) << std::setfill('0') << ns_part;
    buf << ".h5";

    return buf.str();
}

bool visFileData::flush() {
    if (!h5_file)
        return false;

    // Write directly from buffers, use write_raw(ptr) so memspace is applied
    h5_file->getDataSet("/vis")
        .select({0, 0, 0}, {num_freq, num_prod, num_file_t})
        .write_raw(vis.data());
    h5_file->getDataSet("/flags/vis_weight")
        .select({0, 0, 0}, {num_freq, num_prod, num_file_t})
        .write_raw(vis_weight.data());
    h5_file->getDataSet("/eval")
        .select({0, 0, 0}, {num_freq, num_ev, num_file_t})
        .write_raw(eval.data());
    h5_file->getDataSet("/evec")
        .select({0, 0, 0, 0}, {num_freq, num_ev, num_input, num_file_t})
        .write_raw(evec.data());
    h5_file->getDataSet("/erms").select({0, 0}, {num_freq, num_file_t}).write_raw(erms.data());
    h5_file->getDataSet("/flags/frac_lost")
        .select({0, 0}, {num_freq, num_file_t})
        .write_raw(frac_lost.data());
    h5_file->getDataSet("/flags/frac_rfi")
        .select({0, 0}, {num_freq, num_file_t})
        .write_raw(frac_rfi.data());
    h5_file->getDataSet("/gain")
        .select({0, 0, 0}, {num_freq, num_input, num_file_t})
        .write_raw(gain.data());

    h5_file->getDataSet("/fpga_start_tick").write(fpga_start_tick);
    h5_file->getDataSet("/frame_length_fpga_ticks").write(frame_length_fpga_ticks);
    h5_file->getDataSet("/frame_ut1").write(frame_ut1);
    h5_file->getDataSet("/bin_ut1").write(bin_ut1);

    return true;
}

void visFileData::close() {
    if (h5_file)
        h5_file.reset();
}


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
    compression(config.get_default<std::string>(unique_name, "compression", "none")),
    compression_level(config.get_default<std::uint64_t>(unique_name, "compression_level", 0)),
    use_bitshuffle(config.get_default<bool>(unique_name, "use_bitshuffle", false)),
    blocksize_f(config.get_default<std::uint64_t>(unique_name, "blocksize_f", 0)),
    blocksize_p(config.get_default<std::uint64_t>(unique_name, "blocksize_p", 0)),
    blocksize_t(config.get_default<std::uint64_t>(unique_name, "blocksize_t", 1)),
    file_num_t(config.get_default<std::uint64_t>(unique_name, "file_num_t", 10)),
    late_frame_grace_seconds(config.get_default<std::uint64_t>(unique_name, "late_frame_grace_seconds", 60)),
    max_frames(config.get_default<int>(unique_name, "max_frames", -1)),
    buffer(get_buffer("in_buf")),
    tick_len_ns_override(config.get_default<std::uint64_t>(unique_name, "seq_length_nsec_override", 0)) {

    buffer->register_consumer(unique_name);

    // Validate file window configuration
    if (file_num_t == 0) {
        FATAL_ERROR("file_num_t must be > 0 for hdf5VisWrite");
    }
    // Ensure the input buffer is a N2Buffer
    if (buffer->buffer_type != "N2") {
        FATAL_ERROR("Input buffer must be a N2-type buffer.");
    }
}

hdf5VisWrite::~hdf5VisWrite() {}

std::uint64_t hdf5VisWrite::_get_abs_file_idx(const std::shared_ptr<const N2Metadata> meta) const {
    // Get the absolute file index based on the absolute frame index and
    // configured number of time frames per file.

    // Truncate towards zero
    return meta->abs_frame_index / file_num_t;
}

void hdf5VisWrite::_create_dataset(HighFive::File& file, const std::string& name, const std::vector<hsize_t>& dims,
                                       const HighFive::DataType& dtype, HighFive::DataSetCreateProps props) const {

    if (file.exist(name))
    {
        ERROR("Dataset {} already exists in HDF5 file, not creating again.", name);
        return;
    }

    if (dims.size() == 1) {
        // if only one dimension, assume a simple array, chunking is just that dimension.
        std::vector<hsize_t> chunk = {dims[0]};
        props.add(HighFive::Chunking(chunk));
    } else {
        // chunking is ( blocksize_f, ...array dimensions..., blocksize_t )
        std::vector<hsize_t> chunk = dims;
        if (!chunk.empty()) {
            // frequency dimension
            if (blocksize_f > 0)
                chunk[0] = std::min<hsize_t>(chunk[0], (hsize_t)blocksize_f);

            // time dimension
            size_t tdim = chunk.size() - 1;
            if (blocksize_t > 0)
                chunk[tdim] = std::min<hsize_t>(chunk[tdim], (hsize_t)blocksize_t);

            // ensure >= 1
            for (auto& c : chunk)
                c = std::max<hsize_t>(1, c);
        }
        props.add(HighFive::Chunking(chunk));
    }

    HighFive::DataSpace space(dims.begin(), dims.end());
    (void)file.createDataSet(name, space, dtype, props);
};

void hdf5VisWrite::_initialize_h5(HighFive::File& file,
                                  const std::shared_ptr<const N2Metadata> meta,
                                  std::uint64_t file_nt) const {


    HighFive::DataSetCreateProps props_compressed;
    HighFive::DataSetCreateProps props_empty;

    // Compression/filters
    if (use_bitshuffle) {
        // bitshuffle + optional compression backend
        auto level = static_cast<unsigned int>(compression_level > 0 ? compression_level : 9);
        unsigned int comp = hdf5::BITSHUFFLE_COMPRESS_NONE;

        if (compression == "zstd") {
            comp = hdf5::BITSHUFFLE_COMPRESS_ZSTD;
        } else if (compression == "lz4") {
            comp = hdf5::BITSHUFFLE_COMPRESS_LZ4;
        }

        std::vector<unsigned int> bshuf_flags{hdf5::BITSHUFFLE_BLOCKSIZE_AUTO, comp, level};

        props_compressed.add(H5Pset_filter, hdf5::H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY, bshuf_flags.size(),
                  bshuf_flags.data());

    } else if (compression == "deflate") {
        auto level = static_cast<unsigned int>(compression_level > 0 ? compression_level : 4);
        props_compressed.add(H5Pset_deflate, level);
    }

    std::string flags_group_prefix = ""; // "/flags"; // TODO: make config option

    // minimal file-level attributes
    file.createAttribute("num_elements", meta->num_elements);
    file.createAttribute("num_prod", meta->num_prod);
    file.createAttribute("num_ev", meta->num_ev);
    file.createAttribute("num_freq", meta->nfreq);

    _create_dataset(file, "/vis", {meta->nfreq, meta->num_prod, file_nt},
                   HighFive::create_datatype<cfloat>(), props_compressed);
    _create_dataset(file, flags_group_prefix + "/vis_weight", {meta->nfreq, meta->num_prod, file_nt},
                   HighFive::create_datatype<float>(), props_compressed);
    _create_dataset(file, "/eval", {meta->nfreq, meta->num_ev, file_nt},
                   HighFive::create_datatype<float>(), props_compressed);
    _create_dataset(file, "/evec", {meta->nfreq, meta->num_ev, meta->num_elements, file_nt},
                   HighFive::create_datatype<cfloat>(), props_compressed);
    _create_dataset(file, "/erms", {meta->nfreq, file_nt},
                   HighFive::create_datatype<float>(), props_empty);
    _create_dataset(file, "/gain", {meta->nfreq, meta->num_elements, file_nt},
                   HighFive::create_datatype<cfloat>(), props_empty);
    _create_dataset(file, flags_group_prefix + "/frac_lost", {meta->nfreq, file_nt},
                   HighFive::create_datatype<float>(), props_empty);
    _create_dataset(file, flags_group_prefix + "/frac_rfi", {meta->nfreq, file_nt},
                   HighFive::create_datatype<float>(), props_empty);

    _create_dataset(file, "/fpga_start_tick", {file_nt},
                   HighFive::create_datatype<uint64_t>(), props_empty);
    _create_dataset(file, "/frame_length_fpga_ticks", {file_nt},
                   HighFive::create_datatype<uint64_t>(), props_empty);
    _create_dataset(file, "/frame_ut1", {file_nt},
                   HighFive::create_datatype<uint64_t>(), props_empty);
    _create_dataset(file, "/bin_ut1", {file_nt},
                   HighFive::create_datatype<uint64_t>(), props_empty);

}

void hdf5VisWrite::_finalize_dataset(std::unique_ptr<visFileData> dataset) {
    dataset->flush();
    dataset->close();

    // Attempt rename from partial to final
    auto ds_filename = dataset->_get_final_filename();
    int r = std::rename(dataset->partial_filepath.c_str(), ds_filename->c_str());
    if (r != 0) {
        const char* msg = strerror(errno);
        ERROR("Failed to rename partial dataset to final: {} -> {}: {}", dataset->partial_filepath,
                *ds_filename, msg);
    }
}

void hdf5VisWrite::_grace_finalize_datasets(
    std::map<std::string, std::unique_ptr<visFileData>>& datasets,
    const std::string* exclude_path) {
    const double now_s = mono_time_s();
    for (auto ds_it = datasets.begin(); ds_it != datasets.end();) {
        auto& obj = *ds_it->second;
        if (exclude_path && ds_it->first == *exclude_path) {
            ++ds_it;
            continue;
        }
        if (now_s - obj.last_update_wall_s >= double(late_frame_grace_seconds)) {
            _finalize_dataset(ds_it->second);
            ds_it = datasets.erase(ds_it);
        } else {
            ++ds_it;
        }
    }
}

_finalfile_exists(abs_file_idx, base_dir) {
    // look for any files in base_dir that are named vis_<abs_file_idx>_*.h5
    std::string pattern = "vis_" + std::to_string(abs_file_idx) + "_*.h5";
    for (const auto& entry : std::filesystem::directory_iterator(base_dir)) {
        if (entry.is_regular_file()) {
            const std::string filename = entry.path().filename().string();
            if (std::filesystem::path(filename).extension() == ".h5" &&
                filename.find("vis_" + std::to_string(abs_file_idx) + "_") == 0) {
                return true;
            }
        }
    }
    return false;
}

void hdf5VisWrite::main_thread() {
    auto& write_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_viswrite_write_time_seconds", unique_name);
    double avg_write_time = 0.0;
    auto& n_datasets_metric = kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_viswrite_n_datasets", unique_name);

    const double start_time = mono_time_s(); // for logging elapsed time
    N2::frameID in_frame_id(buffer);         // Input frame ID tracker
    int frame_counter = 0;                   // Count of frames written

    /// datasets (files) for writing (multiple may be open simultaneously)
    /// Keyed by absolute file id = absolute frame index / file_num_t
    std::map<size_t, std::unique_ptr<visFileData>> datasets;

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

        // Fetch metadata and create N2 frame view
        N2FrameView fv(buffer, in_frame_id);
        const std::shared_ptr<N2Metadata> meta = get_N2_metadata(buffer, in_frame_id);
        assert(meta);

        // Start timer
        const double frame_recv_time = mono_time_s();
        const double total_elapsed_time = frame_recv_time - start_time;
        INFO("Received buffer {} frame {} (duration_s {})", unique_name, in_frame_id,
             total_elapsed_time);

        auto abs_file_idx = _get_abs_file_idx(meta);
        const std::string partial_dir = base_dir + "/.partial";
        const std::string partial_filename = partial_dir + "/vis_" + std::to_string(abs_file_idx) + ".h5";

        // Ensure dataset exists/open
        visFileData* visFileData_ptr = nullptr;
        auto ds = datasets.find(abs_file_idx);
        if (ds != datasets.end()) {
            // Dataset already open
            visFileData_ptr = ds->second.get();
        } else if (_finalfile_exists(abs_file_idx, base_dir)) {
            // If final file already exists, drop/ignore this frame (late arrival)
            WARN("Finalized file exists for this frame's file window, dropping late frame: {:s}",
                 final_path);

            // Mark frame as done, finalize, and continue
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_datasets(datasets, nullptr);
            continue;
        } else {
            // Create visFileData object for file (also looks for .partial)
            auto visFileData_obj = std::make_unique<visFileData>(
                file_num_t, meta->nfreq, meta->num_elements,
                meta->num_prod, meta->num_ev, frame_recv_time, base_dir);

            datasets.emplace(abs_file_idx, std::move(visFileData_obj));
            visFileData_ptr = datasets.find(abs_file_idx)->second.get();
        }

        if (!visFileData_ptr || !visFileData_ptr->h5_file || ) {
            // Mark frame as done, finalize, and continue
            ERROR("Dataset is null. Failed to open dataset.");
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_datasets(datasets, nullptr);
            continue;
        }

        // Attempt to add frame to dataset
        const std::uint64_t t_in_file = meta->abs_frame_index % file_num_t;
        bool success = visFileData_ptr->add_frame(fv, meta, t_in_file); // performs error checking internally.
        if (!success) {
            // Mark frame as done, finalize, and continue
            ERROR("Failed to add frame to dataset (f={}, t={})", meta->freq_id, t_in_file);
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_datasets(datasets, nullptr);
            continue;
        }
        visFileData_ptr->last_update_wall_s = frame_recv_time;

        // If buffer full, flush
        double elapsed_writing_frame = 0.0;
        if (visFileData_ptr->full()) {
            const double t0 = mono_time_s();
            visFileData_ptr->flush();
            const double t1 = mono_time_s();
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
        const double currt = mono_time_s();
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
        _grace_finalize_datasets(datasets, &final_path);

    } // while !stop_thread

    // Finalize any partially-filled datasets on exit
    for (auto& dset : datasets) {
        _finalize_dataset(dset.second);
    }
    datasets.clear();

    DEBUG("exiting");
}
