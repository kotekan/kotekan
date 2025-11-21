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
#include <cstdio>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <errors.h>
#include <filesystem>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Object.hpp>       // for H5Z_FLAG_MANDATORY, hsize_t
#include <highfive/H5PropertyList.hpp> // for PropertyType, RawPropertyList, Chunking
#include <highfive/bits/H5PropertyList_misc.hpp>
#include <highfive/bits/H5Slice_traits_misc.hpp>
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

template<typename T>
void N2FileData::_check_create_attribute(HighFive::File& file, const std::string& name,
                                         T value) const {
    if (file.hasAttribute(name)) {
        // Attribute exists, check value
        auto attr = file.getAttribute(name);
        T existing_value;
        attr.read(existing_value);
        if (existing_value != value) {
            ERROR_NON_OO("Attribute {} already exists with different value (existing={}, new={})",
                         name, existing_value, value);
        }
        return;
    }

    // Create attribute
    auto attr = file.createAttribute<T>(name, HighFive::DataSpace::From(value));
    attr.write(value);
}

void N2FileData::_check_create_dataset(HighFive::File& file, const std::string& name,
                                       const std::vector<hsize_t>& dims,
                                       const HighFive::DataType& dtype,
                                       HighFive::DataSetCreateProps props) const {

    if (file.exist(name)) {
        WARN_NON_OO("Dataset {} already exists in HDF5 file, not creating again.", name);
        return;
    }

    if (dims.size() == 1) {
        // if only one dimension, assume a simple array, chunking is just that dimension.
        std::vector<hsize_t> chunk = {dims[0]};
        props.add(HighFive::Chunking(chunk));
    } else {
        // chunking is assumed to be ( blocksize_f, ...array dimensions..., blocksize_t )
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

std::unique_ptr<HighFive::File> N2FileData::_open_or_create_file(const std::string& filepath,
                                                                 const uint64_t num_file_t_,
                                                                 const N2FrameView& fv,
                                                                 const FileMode file_mode_) const {

    // 1) Open/create file
    std::unique_ptr<HighFive::File> file;
    if (std::filesystem::exists(filepath)) {
        // Open existing .partial file
        file = std::make_unique<HighFive::File>(filepath, HighFive::File::ReadWrite);
        // TODO: guard against multiple writers?
    } else {
        // Create new .partial file
        file = std::make_unique<HighFive::File>(filepath,
                                                HighFive::File::ReadWrite | HighFive::File::Create);
    }

    // 2) Describe compression/filters
    HighFive::DataSetCreateProps props_compressed = HighFive::DataSetCreateProps::Empty();
    HighFive::DataSetCreateProps props_empty = HighFive::DataSetCreateProps::Empty();

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

        // props_compressed.add(H5Pset_filter, hdf5::H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY,
        //                      bshuf_flags.size(), bshuf_flags.data());
        hid_t dcpl = props_compressed.getId();
        herr_t status =
            H5Pset_filter(dcpl, hdf5::H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY,
                          static_cast<unsigned>(bshuf_flags.size()), bshuf_flags.data());
        if (status < 0) {
            throw std::runtime_error("H5Pset_filter(BITSHUFFLE) failed");
        }

    } else if (compression == "deflate") {
        auto level = static_cast<unsigned int>(compression_level > 0 ? compression_level : 4);
        props_compressed.add(HighFive::Deflate(level));
    }

    std::string flags_group_prefix = file_mode_ == CHIME ? "/flags" : "";
    if (file_mode_ == CHIME && !file->exist(flags_group_prefix)) {
        file->createGroup(flags_group_prefix);
    }

    // 3) Create attributes/datasets, if they don't exist.

    // General File info
    _check_create_attribute(*file, "version", std::string("CHORD_0.0"));
    _check_create_attribute(*file, "file_mode",
                            std::string(file_mode_ == CHIME ? "CHIME" : "CHORD"));
    _check_create_attribute(*file, "abs_file_idx", abs_file_idx);
    _check_create_attribute(*file, "num_file_t", num_file_t);

    // data structure attributes
    _check_create_attribute(*file, "num_elements", fv.num_elements);
    _check_create_attribute(*file, "num_prod", fv.num_prod);
    _check_create_attribute(*file, "num_ev", fv.num_ev);
    _check_create_attribute(*file, "num_freq", fv.nfreq);
    _check_create_attribute(*file, "vis_layout",
                            std::string(fv.vis_layout == N2Layout::FullUpperTri
                                            ? "FullUpperTri"
                                            : "RedundantBaselineAvg"));

    // track which (f,t) frames have been added
    _check_create_dataset(*file, "/frames_added", {fv.nfreq, num_file_t_},
                          HighFive::create_datatype<uint8_t>(), props_empty);

    // JSON config data
    HighFive::DataSetCreateProps json_props = HighFive::DataSetCreateProps::Empty();
    json_props.add(HighFive::Chunking(std::vector<hsize_t>{8}));
    HighFive::DataSpace json_dspace({0}, {HighFive::DataSpace::UNLIMITED});
    if (!file->exist("/config_json"))
        file->createDataSet<std::string>("/config_json", json_dspace, json_props);

    // create datasets
    _check_create_dataset(*file, "/vis", {fv.nfreq, fv.num_prod, num_file_t_},
                          HighFive::create_datatype<cfloat>(), props_compressed);
    _check_create_dataset(*file, flags_group_prefix + "/vis_weight",
                          {fv.nfreq, fv.num_prod, num_file_t_}, HighFive::create_datatype<float>(),
                          props_compressed);
    _check_create_dataset(*file, "/eval", {fv.nfreq, fv.num_ev, num_file_t_},
                          HighFive::create_datatype<float>(), props_compressed);
    _check_create_dataset(*file, "/evec", {fv.nfreq, fv.num_ev, fv.num_elements, num_file_t_},
                          HighFive::create_datatype<cfloat>(), props_compressed);
    _check_create_dataset(*file, "/erms", {fv.nfreq, num_file_t_},
                          HighFive::create_datatype<float>(), props_empty);
    _check_create_dataset(*file, "/gain", {fv.nfreq, fv.num_elements, num_file_t_},
                          HighFive::create_datatype<cfloat>(), props_empty);

    _check_create_dataset(*file, flags_group_prefix + "/flags",
                          {fv.nfreq, fv.num_elements, num_file_t_},
                          HighFive::create_datatype<float>(), props_empty);
    _check_create_dataset(*file, flags_group_prefix + "/frac_lost", {fv.nfreq, num_file_t_},
                          HighFive::create_datatype<float>(), props_empty);
    _check_create_dataset(*file, flags_group_prefix + "/frac_rfi", {fv.nfreq, num_file_t_},
                          HighFive::create_datatype<float>(), props_empty);
    _check_create_dataset(*file, "/fpga_start_tick", {num_file_t_},
                          HighFive::create_datatype<uint64_t>(), props_empty);
    _check_create_dataset(*file, "/frame_length_fpga_ticks", {num_file_t_},
                          HighFive::create_datatype<uint64_t>(), props_empty);

    _check_create_dataset(*file, "/time_center_ut1", {num_file_t_},
                          HighFive::create_datatype<int64_t>(), props_empty);
    _check_create_dataset(*file, "/bin_ut1", {num_file_t_}, HighFive::create_datatype<int64_t>(),
                          props_empty);
    _check_create_dataset(*file, "/bin_start_ERA_deg", {num_file_t_},
                          HighFive::create_datatype<double>(), props_empty);
    _check_create_dataset(*file, "/bin_end_ERA_deg", {num_file_t_},
                          HighFive::create_datatype<double>(), props_empty);
    _check_create_dataset(*file, "/bin_start_LAST", {num_file_t_},
                          HighFive::create_datatype<int64_t>(), props_empty);
    _check_create_dataset(*file, "/bin_end_LAST", {num_file_t_},
                          HighFive::create_datatype<int64_t>(), props_empty);

    return file;
}

bool N2FileData::add_frame(const N2FrameView& fv, size_t t_index) {
    const size_t f_index = fv.freq_id; // TODO: sync with telescope object. For now assume
                                       // 0..num_freq-1 indexing

    // Make sure frame hasn't been added yet
    if (f_index >= num_freq || t_index >= num_file_t) {
        ERROR_NON_OO("N2FileData: index out of bounds: f_index={} >= num_freq={}; t_index={} >= "
                     "num_file_t={}",
                     f_index, num_freq, t_index, num_file_t);
        return false;
    }
    size_t check_idx = idx_ft(f_index, t_index);
    if (added_ft[check_idx] != 0) {
        ERROR_NON_OO("N2FileData: duplicate frame insertion at (f={}, t={})", f_index, t_index);
        return false;
    }

    // Structural data consistency checks
    // TODO: time ints might be off by a bit? Use appx. comparison instead?
    if (fv.vis.size() != N2FrameView::get_num_prod(fv.num_elements, N2Layout::FullUpperTri)
        || fv.weight.size() != N2FrameView::get_num_prod(fv.num_elements, N2Layout::FullUpperTri)
        || fv.eval.size() != fv.num_ev || fv.evec.size() != fv.num_ev * fv.num_elements
        || fv.gain.size() != fv.num_elements || fv.flags.size() != fv.num_elements
        || fv.num_elements != num_elements || fv.num_prod != num_prod || fv.num_ev != num_ev
        || fv.frame_length_fpga_ticks == 0
        || (fpga_start_tick[t_index] > 0 && fpga_start_tick[t_index] != fv.fpga_start_tick)
        || (frame_length_fpga_ticks[t_index] > 0
            && frame_length_fpga_ticks[t_index] != fv.frame_length_fpga_ticks)
        || (time_center_ut1[t_index] > 0 && time_center_ut1[t_index] != fv.time_center_eop.t_ut1)
        || (bin_ut1[t_index] > 0 && bin_ut1[t_index] != fv.bin_eop.t_ut1)
        || (bin_start_ERA_deg[t_index] < 0) || (bin_end_ERA_deg[t_index] < 0)
        || (bin_start_LAST[t_index] > 0 && bin_start_LAST[t_index] != fv.bin_start_LAST)
        || (bin_end_LAST[t_index] > 0 && bin_end_LAST[t_index] != fv.bin_end_LAST)) {
        ERROR_NON_OO(
            "N2FileData: frame information mismatch at (f={}, t={}): "
            "fv.vis.size()={}, fv.weight.size()={}, fv.eval.size()={}, fv.evec.size()={}, "
            "fv.gain.size()={}, fv.flags.size()={}, fv.num_elements={}, fv.num_prod={}, "
            "fv.num_ev={}, fpga_start_tick[t_index]={}, fv.fpga_start_tick={}, "
            "fv.frame_length_fpga_ticks={}, frame_length_fpga_ticks[t_index]={}, "
            "time_center_ut1[t_index]={}, fv.time_center_eop.t_ut1={}, bin_ut1[t_index]={}, "
            "fv.bin_eop.t_ut1={}",
            f_index, t_index, fv.vis.size(), fv.weight.size(), fv.eval.size(), fv.evec.size(),
            fv.gain.size(), fv.flags.size(), fv.num_elements, fv.num_prod, fv.num_ev,
            fpga_start_tick[t_index], fv.fpga_start_tick, fv.frame_length_fpga_ticks,
            frame_length_fpga_ticks[t_index], time_center_ut1[t_index], fv.time_center_eop.t_ut1,
            bin_ut1[t_index], fv.bin_eop.t_ut1);
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
        for (size_t i = 0; i < num_elements; ++i) {
            evec[idx_feit(f_index, e, i, t_index)] = fv.evec[num_elements * e + i];
        }
    }
    // Store erms, gain, flags
    erms[idx_ft(f_index, t_index)] = fv.erms;
    for (size_t i = 0; i < num_elements; ++i) {
        gain[idx_fit(f_index, i, t_index)] = fv.gain[i];
        flags[idx_fit(f_index, i, t_index)] = fv.flags[i];
    }
    // Store fraction lost and RFI
    const uint64_t frame_len_ticks = fv.frame_length_fpga_ticks;
    const uint64_t n_valid = fv.n_valid_fpga_ticks;
    const uint64_t n_rfi = fv.n_rfi_fpga_ticks;
    frac_lost[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (1.0f - float(n_valid) / float(frame_len_ticks)) : 0.0f;
    frac_rfi[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (float(n_rfi) / float(frame_len_ticks)) : 0.0f;
    // Store per-time metadata
    fpga_start_tick[t_index] = fv.fpga_start_tick;
    frame_length_fpga_ticks[t_index] = fv.frame_length_fpga_ticks;
    time_center_ut1[t_index] = fv.time_center_eop.t_ut1;
    bin_ut1[t_index] = fv.bin_eop.t_ut1;
    bin_start_ERA_deg[t_index] = fv.bin_start_ERA_deg;
    bin_end_ERA_deg[t_index] = fv.bin_end_ERA_deg;
    bin_start_LAST[t_index] = fv.bin_start_LAST;
    bin_end_LAST[t_index] = fv.bin_end_LAST;

    // Mark (f, t) as added
    size_t si = idx_ft(f_index, t_index);
    added_ft[si] = 1;
    ++added_count; // increment total number of frames added

    return true;
}

std::optional<std::string> N2FileData::_get_final_filename() {
    // Get the earliest (non-zero) time in the bin ut1 array
    if (num_file_t == 0)
        return std::nullopt;

    std::optional<std::int64_t> earliest_ut1_ns;
    for (size_t t = 0; t < num_file_t; ++t) {
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
    buf << "vis_" << std::to_string(abs_file_idx) << "_"
        << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT%H%M%S");
    // Include nanosecond suffix to avoid collisions for sub-second file windows
    buf << "_" << std::setw(9) << std::setfill('0') << ns_part;
    buf << ".h5";

    const std::string basename = buf.str();
    std::filesystem::path final_path = std::filesystem::path(base_dir) / basename;
    return final_path.string();
}

bool N2FileData::flush_to_disk() {
    if (!h5_file)
        return false;

    std::string flags_group_prefix = file_mode == CHIME ? "/flags" : "";

    // Add and write configs in configTracker
    std::vector<std::string> json_objs = kotekan::ConfigTracker::instance().getAllJSONConfigs();
    HighFive::DataSet json_dset = h5_file->getDataSet("/config_json");
    auto space = json_dset.getSpace();
    auto cur_dims = space.getDimensions();
    std::size_t old_n = cur_dims.empty() ? 0 : cur_dims[0];
    std::size_t extra_n = json_objs.size();
    json_dset.resize({old_n + extra_n});
    json_dset.select({old_n}, {extra_n}).write(json_objs);

    // Write directly from buffers, use write_raw(ptr) so memspace is applied
    h5_file->getDataSet("/frames_added")
        .select({0, 0}, {num_freq, num_file_t})
        .write_raw(added_ft.data());

    h5_file->getDataSet("/vis")
        .select({0, 0, 0}, {num_freq, num_prod, num_file_t})
        .write_raw(vis.data());
    h5_file->getDataSet(flags_group_prefix + "/vis_weight")
        .select({0, 0, 0}, {num_freq, num_prod, num_file_t})
        .write_raw(vis_weight.data());
    h5_file->getDataSet("/eval")
        .select({0, 0, 0}, {num_freq, num_ev, num_file_t})
        .write_raw(eval.data());
    h5_file->getDataSet("/evec")
        .select({0, 0, 0, 0}, {num_freq, num_ev, num_elements, num_file_t})
        .write_raw(evec.data());
    h5_file->getDataSet("/erms").select({0, 0}, {num_freq, num_file_t}).write_raw(erms.data());
    h5_file->getDataSet(flags_group_prefix + "/frac_lost")
        .select({0, 0}, {num_freq, num_file_t})
        .write_raw(frac_lost.data());
    h5_file->getDataSet(flags_group_prefix + "/frac_rfi")
        .select({0, 0}, {num_freq, num_file_t})
        .write_raw(frac_rfi.data());
    h5_file->getDataSet("/gain")
        .select({0, 0, 0}, {num_freq, num_elements, num_file_t})
        .write_raw(gain.data());
    h5_file->getDataSet(flags_group_prefix + "/flags")
        .select({0, 0, 0}, {num_freq, num_elements, num_file_t})
        .write_raw(flags.data());

    h5_file->getDataSet("/fpga_start_tick").write(fpga_start_tick);
    h5_file->getDataSet("/frame_length_fpga_ticks").write(frame_length_fpga_ticks);
    h5_file->getDataSet("/time_center_ut1").write(time_center_ut1);
    h5_file->getDataSet("/bin_ut1").write(bin_ut1);
    h5_file->getDataSet("/bin_start_ERA_deg").write(bin_start_ERA_deg);
    h5_file->getDataSet("/bin_end_ERA_deg").write(bin_end_ERA_deg);
    h5_file->getDataSet("/bin_start_LAST").write(bin_start_LAST);
    h5_file->getDataSet("/bin_end_LAST").write(bin_end_LAST);

    return true;
}

void N2FileData::close() {
    if (h5_file)
        h5_file.reset();
}


REGISTER_KOTEKAN_STAGE(hdf5N2Write);

hdf5N2Write::hdf5N2Write(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          [](const kotekan::Stage& stage) {
              return const_cast<kotekan::Stage&>(stage).main_thread();
          }),
    base_dir(config.get<std::string>(unique_name, "base_dir")),
    file_num_t(config.get_default<std::uint64_t>(unique_name, "file_num_t", 10)),
    late_frame_grace_seconds(
        config.get_default<std::uint64_t>(unique_name, "late_frame_grace_seconds", 60)),
    max_frames(config.get_default<int>(unique_name, "max_frames", -1)),
    buffer(get_buffer("in_buf")),
    write_time_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_viswrite_write_time_seconds", unique_name)),
    n_datasets_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_viswrite_n_datasets", unique_name)) {

    buffer->register_consumer(unique_name);

    // Validate file window configuration
    if (file_num_t == 0) {
        FATAL_ERROR("file_num_t must be > 0 for hdf5N2Write");
    }
    // Ensure the input buffer is an N2Buffer
    if (buffer->buffer_type != "N2") {
        FATAL_ERROR("Input buffer must be a N2-type buffer.");
    }
}

hdf5N2Write::~hdf5N2Write() {}

std::uint64_t hdf5N2Write::_get_abs_file_idx(const N2FrameView& fv) const {
    // Get the absolute file index based on the absolute frame index and
    // configured number of time frames per file.

    // Just truncate towards zero
    return fv.abs_time_idx / file_num_t;
}

void hdf5N2Write::_finalize_file(N2FileData& filedata) {
    filedata.flush_to_disk();
    filedata.close();

    // Attempt rename from partial to final
    auto ds_filename = filedata._get_final_filename();
    if (!ds_filename) {
        WARN("Could not determine final filename for {} (no UT1 found); leaving partial in place.",
             filedata.partial_filepath);
        return;
    }

    int r = std::rename(filedata.partial_filepath.c_str(), ds_filename->c_str());
    if (r != 0) {
        const char* msg = strerror(errno);
        ERROR("Failed to rename partial dataset to final: {} -> {}: {}", filedata.partial_filepath,
              *ds_filename, msg);
    }
}

void hdf5N2Write::_grace_finalize_files(std::map<size_t, std::unique_ptr<N2FileData>>& files,
                                        const size_t* exclude_abs_file_idx) {
    const double now_s = mono_time_s();
    for (auto file_it = files.begin(); file_it != files.end();) {
        auto& obj = *file_it->second;
        if (exclude_abs_file_idx && file_it->first == *exclude_abs_file_idx) {
            ++file_it;
            continue;
        }
        if (now_s - obj.last_update_wall_s >= double(late_frame_grace_seconds)) {
            _finalize_file(*file_it->second);
            file_it = files.erase(file_it);
        } else {
            ++file_it;
        }
    }
}

bool hdf5N2Write::_finalfile_exists(std::uint64_t abs_file_idx,
                                    const std::string& search_dir) const {
    const std::string prefix = "vis_" + std::to_string(abs_file_idx) + "_";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(search_dir)) {
            if (entry.is_regular_file()) {
                const std::string filename = entry.path().filename().string();
                if (std::filesystem::path(filename).extension() == ".h5"
                    && filename.rfind(prefix, 0) == 0) {
                    return true;
                }
            }
        }
    } catch (const std::filesystem::filesystem_error&) {
        return false;
    }
    return false;
}

void hdf5N2Write::main_thread() {
    double avg_write_time = 0.0;

    const double start_time = mono_time_s(); // for logging elapsed time
    N2::frameID in_frame_id(buffer);         // Input frame ID tracker
    int frame_counter = 0;                   // Count of frames written

    /// file data for writing (multiple may be open simultaneously)
    /// Keyed by absolute file id = absolute frame index / file_num_t
    std::map<size_t, std::unique_ptr<N2FileData>> filedata;

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

        // Start timer
        const double frame_recv_time = mono_time_s();
        const double total_elapsed_time = frame_recv_time - start_time;
        INFO("Received buffer {} frame {} (duration_s {})", unique_name, in_frame_id,
             total_elapsed_time);

        auto abs_file_idx = _get_abs_file_idx(fv);
        const std::string partial_dir = base_dir + "/.partial";

        // Ensure dataset exists/open
        N2FileData* N2FileData_ptr = nullptr;
        auto fd = filedata.find(abs_file_idx);
        if (fd != filedata.end()) {
            // Dataset already open
            N2FileData_ptr = fd->second.get();
        } else if (_finalfile_exists(abs_file_idx, base_dir)) {
            // If final file already exists, drop/ignore this frame (late arrival)
            WARN("Finalized file already exists for this frame's file window, dropping frame: "
                 "abs_file_idx={}, frame_id={}",
                 abs_file_idx, in_frame_id);

            // Mark frame as done, finalize, and continue
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_files(filedata, nullptr);
            continue;
        } else {
            // Create N2FileData for file (also looks for .partial)

            auto compression = config.get_default<std::string>(unique_name, "compression", "none");
            auto compression_level =
                config.get_default<std::uint64_t>(unique_name, "compression_level", 0);
            auto use_bitshuffle = config.get_default<bool>(unique_name, "use_bitshuffle", false);
            auto blocksize_f = config.get_default<std::uint64_t>(unique_name, "blocksize_f", 0);
            auto blocksize_t = config.get_default<std::uint64_t>(unique_name, "blocksize_t", 1);

            auto N2FileData_obj = std::make_unique<N2FileData>(
                N2FileData::CHORD, file_num_t, fv, frame_recv_time, abs_file_idx, blocksize_f,
                blocksize_t, compression, compression_level, use_bitshuffle, base_dir);

            filedata.emplace(abs_file_idx, std::move(N2FileData_obj));
            N2FileData_ptr = filedata.find(abs_file_idx)->second.get();
        }

        if (!N2FileData_ptr || !N2FileData_ptr->h5_file) {
            // Mark frame as done, finalize, and continue
            ERROR("Dataset is null. Failed to open dataset.");
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_files(filedata, nullptr);
            continue;
        }

        // Attempt to add frame to dataset
        const std::uint64_t t_in_file = fv.abs_time_idx % file_num_t;
        bool success =
            N2FileData_ptr->add_frame(fv, t_in_file); // performs error checking internally.
        if (!success) {
            // Mark frame as done, finalize, and continue
            ERROR("Failed to add frame to dataset (f={}, t={})", fv.freq_id, t_in_file);
            buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_files(filedata, nullptr);
            continue;
        }
        N2FileData_ptr->last_update_wall_s = frame_recv_time;

        // If buffer full, flush
        double elapsed_writing_frame = 0.0;
        if (N2FileData_ptr->full()) {
            const double t0 = mono_time_s();
            _finalize_file(*N2FileData_ptr);
            const double t1 = mono_time_s();
            elapsed_writing_frame = t1 - t0;
            // Close dataset after flush
            N2FileData_ptr->close();
            filedata.erase(abs_file_idx);
        }

        // Stop timer/metrics
        const double currt = mono_time_s();
        if (elapsed_writing_frame <= 0.0)
            elapsed_writing_frame = currt - frame_recv_time;
        write_time_metric.set(elapsed_writing_frame);
        avg_write_time =
            (avg_write_time * frame_counter + elapsed_writing_frame) / double(frame_counter + 1);
        n_datasets_metric.set(filedata.size());

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
        _grace_finalize_files(filedata, &abs_file_idx);

    } // while !stop_thread

    // Finalize any partially-filled datasets on exit
    for (auto& file : filedata) {
        _finalize_file(*file.second);
    }
    filedata.clear();

    DEBUG("exiting");
}
