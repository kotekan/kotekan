#include "hdf5N2Write.hpp"

#include "H5Support.hpp"
#include "Telescope.hpp"  // for Telescope
#include "restClient.hpp" // for restClient
#include "util.h"         // for mkdir_p

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
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <errors.h>
#include <event2/http.h> // for evhttp_uri_parse
#include <filesystem>
#include <fmt/ranges.h>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5Exception.hpp>
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
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>
#include <waitingForMaxFrames.hpp> // for waiting_for_max_frames

using namespace HighFive;

/// Compute a hash of a file's contents for change detection.
static std::optional<size_t> hash_file_contents(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return std::nullopt;
    std::string contents((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (ifs.bad()) // I/O error during read
        return std::nullopt;
    return std::hash<std::string>{}(contents);
}

/// Fetch `url` over plain HTTP via kotekan's built-in restClient and write the
/// response body verbatim to `dest_path`. FATAL on any parse/fetch/write error.
/// Intended for small files (bytes land in a std::string before hitting disk).
static void download_url_to_file(const std::string& url, const std::string& dest_path) {
    struct evhttp_uri* uri = evhttp_uri_parse(url.c_str());
    if (!uri) {
        FATAL_ERROR_NON_OO("hdf5N2Write: could not parse URL '{}'", url);
        return;
    }

    const char* scheme = evhttp_uri_get_scheme(uri);
    if (!scheme || std::string(scheme) != "http") {
        evhttp_uri_free(uri);
        FATAL_ERROR_NON_OO("hdf5N2Write: baseband_gain_url must use plain http:// (got '{}')", url);
        return;
    }

    const char* host_c = evhttp_uri_get_host(uri);
    if (!host_c || host_c[0] == '\0') {
        evhttp_uri_free(uri);
        FATAL_ERROR_NON_OO("hdf5N2Write: baseband_gain_url has no host: '{}'", url);
        return;
    }
    const std::string host = host_c;

    int port = evhttp_uri_get_port(uri);
    if (port <= 0)
        port = 80;

    const char* path_c = evhttp_uri_get_path(uri);
    std::string path = (path_c && path_c[0] != '\0') ? path_c : "/";
    const char* query_c = evhttp_uri_get_query(uri);
    if (query_c && query_c[0] != '\0') {
        path += "?";
        path += query_c;
    }
    evhttp_uri_free(uri);

    INFO_NON_OO("hdf5N2Write: downloading digial gains file from fpga_master at http://{}:{}{}",
                host, port, path);
    // GET with empty JSON body
    restClient::restReply reply = restClient::instance().make_request_blocking(
        path, nlohmann::json::object(), host, static_cast<unsigned short>(port));
    if (!reply.first) {
        FATAL_ERROR_NON_OO("hdf5N2Write: failed to fetch gains file from '{}'", url);
        return;
    }

    std::ofstream ofs(dest_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        FATAL_ERROR_NON_OO("hdf5N2Write: could not open '{}' for writing downloaded gains file",
                           dest_path);
        return;
    }
    ofs.write(reply.second.data(), static_cast<std::streamsize>(reply.second.size()));
    if (!ofs) {
        FATAL_ERROR_NON_OO("hdf5N2Write: failed to write {} bytes to '{}'", reply.second.size(),
                           dest_path);
        return;
    }
    ofs.close();
    if (!ofs) {
        FATAL_ERROR_NON_OO("hdf5N2Write: failed to finalize writing downloaded gains file to '{}'",
                           dest_path);
        return;
    }

    INFO_NON_OO("hdf5N2Write: wrote {} bytes to '{}'", reply.second.size(), dest_path);
}

namespace {
/// Walk the config tree and collect (unique_name, base_dir) for every
/// hdf5N2Write stage block. Mirrors StageFactory::build_from_tree's notion
/// of unique_name: "/" + slash-joined keys.
void collect_hdf5N2Write_base_dirs(const nlohmann::json& tree, const std::string& path,
                                   std::vector<std::pair<std::string, std::string>>& out) {
    for (auto it = tree.begin(); it != tree.end(); ++it) {
        if (!it.value().is_object())
            continue;
        const std::string sub = path + "/" + it.key();
        if (it.value().value("kotekan_stage", std::string{}) == "hdf5N2Write") {
            const auto bd = it.value().find("base_dir");
            if (bd != it.value().end() && bd->is_string())
                out.emplace_back(sub, bd->get<std::string>());
        }
        collect_hdf5N2Write_base_dirs(it.value(), sub, out);
    }
}

/// Canonicalize a path for collision comparison; falls back to the raw
/// string if weakly_canonical fails (e.g. permission denied on a parent).
std::string normalize_base_dir(const std::string& dir) {
    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(dir, ec);
    return ec ? dir : canonical.string();
}
} // namespace

/// Generate an acquisition base directory path: base_dir/acq_YYYYMMDD_HHMMSS_NNNNNNNNN
static std::string get_acq_base_dir_path(const std::string& base_dir) {
    // Strip trailing slashes
    std::string dir = base_dir;
    while (dir.size() > 1 && dir.back() == '/')
        dir.pop_back();

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count()
              % 1'000'000'000LL;

    std::tm tm_buf{};
    gmtime_r(&time_t_now, &tm_buf);

    std::ostringstream oss;
    oss << dir << "/acq_" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_" << std::setfill('0')
        << std::setw(9) << ns;
    return oss.str();
}

// Monotonic time in seconds
inline double mono_time_s() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    auto dt = clock::now() - t0;
    return std::chrono::duration<double>(dt).count();
}

// Trait to detect containers with a value_type (e.g., std::vector/std::array), but not std::string.
template<typename T, typename = void>
struct has_value_type : std::false_type {};

template<typename T>
struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type {};

template<typename T>
void N2FileData::_check_create_attribute(HighFive::File& file, const std::string& name,
                                         const T& value) const {
    if (file.hasAttribute(name)) {
        // Attribute exists, check value
        auto attr = file.getAttribute(name);
        T existing_value;
        attr.read(existing_value);
        if (existing_value != value) {
            FATAL_ERROR_NON_OO(
                "Attribute {} already exists with different value (existing={}, new={})", name,
                existing_value, value);
        }
        return;
    }

    // Create attribute; container types need an explicit element datatype.
    if constexpr (has_value_type<T>::value && !std::is_same_v<T, std::string>) {
        auto attr = file.createAttribute(name, HighFive::DataSpace::From(value),
                                         HighFive::create_datatype<typename T::value_type>());
        attr.write(value);
    } else {
        auto attr = file.createAttribute<T>(name, HighFive::DataSpace::From(value));
        attr.write(value);
    }
}

void N2FileData::_check_create_dataset(HighFive::File& file, const std::string& name,
                                       const std::vector<hsize_t>& dims,
                                       const std::vector<std::string>& dim_names,
                                       const HighFive::DataType& dtype,
                                       HighFive::DataSetCreateProps props) const {

    if (file.exist(name)) {
        WARN_NON_OO("Dataset {} already exists in HDF5 file, not creating again.", name);
        return;
    }
    // size of dims should = size of dim_names, and > 0
    assert(dims.size() == dim_names.size());
    assert(dims.size() > 0);

    if (dims.size() == 1) {
        // if only one dimension, assume a simple array, chunking is just that dimension.
        std::vector<hsize_t> chunk = {dims[0]};
        props.add(HighFive::Chunking(chunk));
    } else {
        // chunking is assumed to be ( blocksize_f, ...array dimensions..., blocksize_t )
        std::vector<hsize_t> chunk = dims;

        // If dimension is "time" (usually last dimension), apply blocksize_t
        // If dimension is "frequency" (usually first dimension), apply blocksize_f
        // If dimension is product or element, use blocksize_p
        for (size_t i = 0; i < dim_names.size(); i++) {
            if (dim_names[i] == "time" && blocksize_t > 0) {
                chunk[i] = std::min<hsize_t>(chunk[i], (hsize_t)blocksize_t);
            } else if (dim_names[i].substr(0, 4) == "freq" && blocksize_f > 0) {
                chunk[i] = std::min<hsize_t>(chunk[i], (hsize_t)blocksize_f);
            } else if ((dim_names[i].substr(0, 4) == "prod" || dim_names[i].substr(0, 2) == "el")
                       && blocksize_p > 0) {
                chunk[i] = std::min<hsize_t>(chunk[i], (hsize_t)blocksize_p);
            }
        }

        for (auto& c : chunk)
            c = std::max<hsize_t>(1, c);

        props.add(HighFive::Chunking(chunk));
    }

    // Create dataset
    HighFive::DataSpace space(dims.begin(), dims.end());
    auto dataset = file.createDataSet(name, space, dtype, props);
    dataset.createAttribute("axis", dim_names);
};

std::unique_ptr<HighFive::File> N2FileData::_open_or_create_file(const std::string& filepath,
                                                                 const uint64_t num_file_t_,
                                                                 const N2FrameView& fv,
                                                                 const FileMode file_mode_) const {
    // Wrap attempt to create/open file in try/catch (fail gracefully)
    try {

        // 1) Open/create file
        std::unique_ptr<HighFive::File> file;
        if (std::filesystem::exists(filepath)) {
            // Open existing .partial file
            file = std::make_unique<HighFive::File>(filepath, HighFive::File::ReadWrite);
            // TODO: guard against multiple writers?
        } else {
            // Create new .partial file
            file = std::make_unique<HighFive::File>(filepath, HighFive::File::ReadWrite
                                                                  | HighFive::File::Create);
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
        _check_create_attribute(*file, "n2_layout", N2Layout_to_string(fv.n2_layout));

        // Telescope info
        const CHORDTelescope& telescope = Telescope::instance().cast<CHORDTelescope>();
        _check_create_attribute(*file, "instrument_name", Telescope::instance().get_name());
        // _check_create_attribute(*file, "num_stacks", telescope.get_num_stacks());
        _check_create_attribute(*file, "nyquist_zone", telescope.nyquist_zone());
        _check_create_attribute(*file, "gps_time_enabled", telescope.gps_time_enabled());
        _check_create_attribute(*file, "frame0_t_inst_ns", telescope.to_time_ns(0));
        _check_create_attribute(*file, "fpga_seq_length_nsec", telescope.seq_length_nsec());
        _check_create_attribute(*file, "origin_itrs_lon_deg", telescope.get_origin_itrs_lon_deg());
        _check_create_attribute(*file, "origin_itrs_lat_deg", telescope.get_origin_itrs_lat_deg());
        _check_create_attribute(*file, "dish_coelev_deg", telescope.get_dish_coelev_deg());
        _check_create_attribute(*file, "num_dishes", telescope.get_num_dishes());
        _check_create_attribute(*file, "num_file_f",
                                telescope.num_science_freqs()); // "science-case" frequencies only
        _check_create_attribute(*file, "num_telescope_f", telescope.num_freq());

        // Store EOP table.
        {
            std::vector<EOP> eop_table = telescope.get_current_EOP_table();
            const int eop_len = eop_table.size();
            if (eop_len > 0) {
                std::vector<int64_t> eop_t_inst_ns(eop_len);
                std::vector<int64_t> eop_t_ut1_ns(eop_len);
                std::vector<double> eop_delta_UT1_inst(eop_len);
                std::vector<double> eop_ERA_deg(eop_len);
                std::vector<double> eop_xp_as(eop_len);
                std::vector<double> eop_yp_as(eop_len);

                for (int i = 0; i < eop_len; i++) {
                    EOP eop = eop_table[i];
                    eop_t_inst_ns[i] = eop.t_inst_ns;
                    eop_t_ut1_ns[i] = eop.t_ut1_ns;
                    eop_delta_UT1_inst[i] = eop.delta_UT1_inst;
                    eop_ERA_deg[i] = eop.ERA_deg;
                    eop_xp_as[i] = eop.xp_as;
                    eop_yp_as[i] = eop.yp_as;
                }
                _check_create_attribute(*file, "EOP_t_inst_ns", eop_t_inst_ns);
                _check_create_attribute(*file, "EOP_t_ut1_ns", eop_t_ut1_ns);
                _check_create_attribute(*file, "EOP_delta_UT1_inst", eop_delta_UT1_inst);
                _check_create_attribute(*file, "EOP_ERA_deg", eop_ERA_deg);
                _check_create_attribute(*file, "EOP_xp_as", eop_xp_as);
                _check_create_attribute(*file, "EOP_yp_as", eop_yp_as);
            }
        }

        // Store grid orientation (3x3 matrix) and dish orientation (3x3 matrix)
        {
            std::vector<double> grid_orientation(9);
            std::vector<double> dish_orientation(9);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    grid_orientation[i * 3 + j] = telescope.get_grid_orientation_el(i, j);
                    dish_orientation[i * 3 + j] = telescope.get_dish_orientation_el(i, j);
                }
            }
            _check_create_attribute(*file, "grid_orientation", grid_orientation);
            _check_create_attribute(*file, "dish_orientation", dish_orientation);
        }

        // Store dish input info
        {
            // Set up an object to receive the dish input info.
            dishInputFields dish_inputs;
            // Fill the object with the info.
            telescope.fill_input_maps(dish_inputs);

            _check_create_dataset(*file, "/index_map/grid_x_idx", {dish_inputs.grid_x_idx.size()},
                                  {"dish"}, HighFive::create_datatype<int64_t>(), props_empty);
            auto dataset_x = file->getDataSet("/index_map/grid_x_idx");
            dataset_x.write(dish_inputs.grid_x_idx);

            _check_create_dataset(*file, "/index_map/grid_y_idx", {dish_inputs.grid_y_idx.size()},
                                  {"dish"}, HighFive::create_datatype<int64_t>(), props_empty);
            auto dataset_y = file->getDataSet("/index_map/grid_y_idx");
            dataset_y.write(dish_inputs.grid_y_idx);

            _check_create_dataset(*file, "/index_map/feed_pos_disp_m",
                                  {dish_inputs.feed_pos_disp_m.size(), 3}, {"element", "xyz"},
                                  HighFive::create_datatype<double>(), props_empty);
            auto dataset_pos = file->getDataSet("/index_map/feed_pos_disp_m");
            dataset_pos.write(dish_inputs.feed_pos_disp_m);

            _check_create_dataset(*file, "/index_map/coelev_disp_deg",
                                  {dish_inputs.coelev_disp_deg.size()}, {"element"},
                                  HighFive::create_datatype<double>(), props_empty);
            auto dataset_coelev = file->getDataSet("/index_map/coelev_disp_deg");
            dataset_coelev.write(dish_inputs.coelev_disp_deg);

            _check_create_dataset(*file, "/index_map/type", {dish_inputs.type.size()}, {"dish"},
                                  HighFive::create_datatype<int32_t>(), props_empty);
            auto dataset_type = file->getDataSet("/index_map/type");
            // Cast DishType enum to int32_t for storage
            std::vector<int32_t> type_int(dish_inputs.type.size());
            for (size_t i = 0; i < dish_inputs.type.size(); i++) {
                type_int[i] = static_cast<int32_t>(dish_inputs.type[i]);
            }
            dataset_type.write(type_int);
        }

        // Store full dish positions
        {
            const int num_dishes = telescope.get_num_dishes();
            std::vector<std::array<double, 3>> dish_positions(num_dishes);
            for (int i = 0; i < num_dishes; i++) {
                auto pos = telescope.get_dish_position_in_grid_coords(i);
                dish_positions[i] = {pos[0], pos[1], pos[2]};
            }
            _check_create_dataset(*file, "/index_map/dish_positions_in_grid_coords",
                                  {static_cast<hsize_t>(num_dishes), 3}, {"dish", "xyz"},
                                  HighFive::create_datatype<double>(), props_empty);
            auto dataset = file->getDataSet("/index_map/dish_positions_in_grid_coords");
            dataset.write(dish_positions);
        }

        // Store physical frequencies (file frequencies) as a dataset
        {
            std::vector<N2::freq_ctype> freq_chans(num_file_f);
            for (size_t f = 0; f < num_file_f; f++) {
                freq_id_t freq_id = telescope.min_science_freq_id() + f;
                freq_chans[f] = N2::freq_ctype{telescope.to_freq_MHz(freq_id),
                                               telescope.freq_width_MHz(freq_id)};
            }

            auto freq_dtype = HighFive::create_datatype<N2::freq_ctype>();
            _check_create_dataset(*file, "/index_map/freq", {num_file_f}, {"frequency"}, freq_dtype,
                                  props_empty);
            auto dataset = file->getDataSet("/index_map/freq");
            dataset.write(freq_chans);
        }

        // Store product list as a dataset
        {
            const auto& prods = fv._desc->get_product_list();
            auto prod_dtype = HighFive::create_datatype<N2::prod_ctype>();
            _check_create_dataset(*file, "/index_map/prod", {fv.num_prod}, {"product"}, prod_dtype,
                                  props_empty);
            auto dataset = file->getDataSet("/index_map/prod");
            dataset.write(prods);
        }

        // JSON config data (written at file flush time)
        HighFive::DataSetCreateProps json_props = HighFive::DataSetCreateProps::Empty();
        json_props.add(HighFive::Chunking(std::vector<hsize_t>{8}));
        HighFive::DataSpace json_dspace({0}, {HighFive::DataSpace::UNLIMITED});
        if (!file->exist("/config_json"))
            file->createDataSet<std::string>("/config_json", json_dspace, json_props);


        // tracker for which (f,t) frames have been added (written at file flush time)
        _check_create_dataset(*file, "/frames_added", {num_file_f, num_file_t_},
                              {"frequency", "time"}, HighFive::create_datatype<uint8_t>(),
                              props_empty);

        // create datasets (written at file flush time)
        _check_create_dataset(*file, "/vis", {num_file_f, fv.num_prod, num_file_t_},
                              {"frequency", "product", "time"}, HighFive::create_datatype<cfloat>(),
                              props_compressed);
        _check_create_dataset(
            *file, flags_group_prefix + "/vis_weight", {num_file_f, fv.num_prod, num_file_t_},
            {"frequency", "product", "time"}, HighFive::create_datatype<float>(), props_compressed);
        _check_create_dataset(*file, "/eval", {num_file_f, fv.num_ev, num_file_t_},
                              {"frequency", "eigenval", "time"}, HighFive::create_datatype<float>(),
                              props_compressed);
        _check_create_dataset(*file, "/evec", {num_file_f, fv.num_ev, fv.num_elements, num_file_t_},
                              {"frequency", "eigenvec", "element", "time"},
                              HighFive::create_datatype<cfloat>(), props_compressed);
        _check_create_dataset(*file, "/erms", {num_file_f, num_file_t_}, {"frequency", "time"},
                              HighFive::create_datatype<float>(), props_empty);
        _check_create_dataset(*file, "/gain", {num_file_f, fv.num_elements, num_file_t_},
                              {"frequency", "element", "time"}, HighFive::create_datatype<cfloat>(),
                              props_empty);
        _check_create_dataset(*file, "/radiometer_chi2", {num_file_f, num_file_t_, 3},
                              {"frequency", "time", "pol_product"},
                              HighFive::create_datatype<float>(), props_empty);

        _check_create_dataset(
            *file, flags_group_prefix + "/flags", {num_file_f, fv.num_elements, num_file_t_},
            {"frequency", "element", "time"}, HighFive::create_datatype<float>(), props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/valid_fpga_count",
                              {num_file_f, num_file_t_}, {"frequency", "time"},
                              HighFive::create_datatype<uint64_t>(), props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/rfi_fpga_count",
                              {num_file_f, num_file_t_}, {"frequency", "time"},
                              HighFive::create_datatype<uint64_t>(), props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/rfi_only_fpga_count",
                              {num_file_f, num_file_t_}, {"frequency", "time"},
                              HighFive::create_datatype<uint64_t>(), props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/pl_fpga_count",
                              {num_file_f, num_file_t_}, {"frequency", "time"},
                              HighFive::create_datatype<uint64_t>(), props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/frac_lost", {num_file_f, num_file_t_},
                              {"frequency", "time"}, HighFive::create_datatype<float>(),
                              props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/frac_rfi", {num_file_f, num_file_t_},
                              {"frequency", "time"}, HighFive::create_datatype<float>(),
                              props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/frac_rfi_only",
                              {num_file_f, num_file_t_}, {"frequency", "time"},
                              HighFive::create_datatype<float>(), props_empty);
        _check_create_dataset(*file, flags_group_prefix + "/frac_pl", {num_file_f, num_file_t_},
                              {"frequency", "time"}, HighFive::create_datatype<float>(),
                              props_empty);

        _check_create_dataset(*file, "/fpga_start_tick", {num_file_t_}, {"time"},
                              HighFive::create_datatype<uint64_t>(), props_empty);
        _check_create_dataset(*file, "/frame_length_fpga_ticks", {num_file_t_}, {"time"},
                              HighFive::create_datatype<uint64_t>(), props_empty);

        _check_create_dataset(*file, "/time_center_t_inst_ns", {num_file_t_}, {"time"},
                              HighFive::create_datatype<int64_t>(), props_empty);
        _check_create_dataset(*file, "/time_center_ut1_ns", {num_file_t_}, {"time"},
                              HighFive::create_datatype<int64_t>(), props_empty);
        _check_create_dataset(*file, "/bin_t_inst_ns", {num_file_t_}, {"time"},
                              HighFive::create_datatype<int64_t>(), props_empty);
        _check_create_dataset(*file, "/bin_ut1_ns", {num_file_t_}, {"time"},
                              HighFive::create_datatype<int64_t>(), props_empty);
        _check_create_dataset(*file, "/bin_delta_ut1_inst", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/bin_ERA_deg", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/bin_xp_as", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/bin_yp_as", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/bin_start_ERA_deg", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/bin_end_ERA_deg", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/bin_start_ERAL", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/bin_end_ERAL", {num_file_t_}, {"time"},
                              HighFive::create_datatype<double>(), props_empty);
        _check_create_dataset(*file, "/rfi_frame_excision_enabled", {num_file_t_}, {"time"},
                              HighFive::create_datatype<bool>(), props_empty);
        _check_create_dataset(*file, "/rfi_frame_excision_num", {num_file_t_}, {"time"},
                              HighFive::create_datatype<int32_t>(), props_empty);
        _check_create_dataset(*file, "/rfi_frame_excision_threshold",
                              {num_file_t_, MAX_NUM_RFI_THRESHOLDS}, {"time", "threshold"},
                              HighFive::create_datatype<float>(), props_empty);
        _check_create_dataset(*file, "/rfi_frame_excision_fraction",
                              {num_file_t_, MAX_NUM_RFI_THRESHOLDS}, {"time", "threshold"},
                              HighFive::create_datatype<float>(), props_empty);

        // Digital gains: copy entire gains file verbatim into /digital_gains/ group
        if (!baseband_gain_file.empty() && !file->exist("/digital_gains")) {
            if (!std::filesystem::exists(baseband_gain_file)) {
                FATAL_ERROR_NON_OO("Digital gains file {} does not exist.", baseband_gain_file);
            } else {
                try {
                    HighFive::File src(baseband_gain_file, HighFive::File::ReadOnly);

                    // Copy entire gains file root group as /digital_gains
                    herr_t err = H5Ocopy(src.getId(), ".", file->getId(), "digital_gains",
                                         H5P_DEFAULT, H5P_DEFAULT);
                    if (err < 0) {
                        FATAL_ERROR_NON_OO("H5Ocopy failed copying gains file {} into output.",
                                           baseband_gain_file);
                    } else {
                        // Add source file path and selected update index as attributes
                        _check_create_attribute(*file, "digital_gains_source_file",
                                                baseband_gain_file);

                        auto dg_group = file->getGroup("/digital_gains");
                        int selected_idx = baseband_gain_update_idx;
                        if (selected_idx < 0 && dg_group.exist("gain_coeff")) {
                            auto dims = dg_group.getDataSet("gain_coeff").getDimensions();
                            if (!dims.empty() && dims[0] > 0)
                                selected_idx = static_cast<int>(dims[0] - 1);
                        }
                        dg_group.createAttribute("selected_update_idx", selected_idx);

                        INFO_NON_OO(
                            "Copied digital gains from {} into output (selected_update_idx={}).",
                            baseband_gain_file, selected_idx);
                    }
                } catch (const HighFive::Exception& e) {
                    FATAL_ERROR_NON_OO("Failed to copy gains file {} into output: {}",
                                       baseband_gain_file, e.what());
                }
            }
        } else if (baseband_gain_file.empty()) {
            DEBUG_NON_OO("No baseband_gain_file specified, skipping digital gains.");
        }

        return file;

    } catch (const HighFive::Exception& e) {
        FATAL_ERROR_NON_OO("Failed to open or initialize HDF5 file {}: {}", filepath, e.what());
    } catch (const std::exception& e) {
        FATAL_ERROR_NON_OO("Failed to open or initialize HDF5 file {}: {}", filepath, e.what());
    }

    return nullptr;
}

N2FileData::N2FileData(FileMode file_mode_, uint64_t num_file_t_, const N2FrameView& fv,
                       const double open_wall_s_, const uint64_t abs_file_idx_,
                       const size_t blocksize_f_, const size_t blocksize_p_,
                       const size_t blocksize_t_, const std::string compression_,
                       const size_t compression_level_, const bool use_bitshuffle_,
                       const std::string base_dir_, const std::string baseband_gain_file_,
                       const int baseband_gain_update_idx_) :
    num_elements(fv.num_elements), num_prod(fv.num_prod), num_ev(fv.num_ev),
    num_file_f(Telescope::instance().cast<CHORDTelescope>().num_science_freqs()),
    num_file_t(num_file_t_), file_mode(file_mode_), blocksize_f(blocksize_f_),
    blocksize_p(blocksize_p_), blocksize_t(blocksize_t_), compression(compression_),
    compression_level(compression_level_), use_bitshuffle(use_bitshuffle_),
    open_wall_s(open_wall_s_), abs_file_idx(abs_file_idx_), base_dir(std::move(base_dir_)),
    baseband_gain_file(std::move(baseband_gain_file_)),
    baseband_gain_update_idx(baseband_gain_update_idx_),
    partial_filepath(base_dir + "/.partial/" + "vis_" + std::to_string(abs_file_idx_) + ".h5"),
    n2_layout(fv.n2_layout), last_update_wall_s(open_wall_s_),
    h5_file(_open_or_create_file(partial_filepath, num_file_t_, fv, file_mode)) {

    if (!h5_file) {
        FATAL_ERROR_NON_OO("N2FileData: failed to open or create HDF5 file {}", partial_filepath);
    }

    // Record gains file hash for change detection at flush time
    if (!baseband_gain_file.empty() && std::filesystem::exists(baseband_gain_file)) {
        gains_file_hash = hash_file_contents(baseband_gain_file);
    }

    // resize arrays to hold data across (freq, time) blocks
    vis.assign(num_prod * num_file_f * num_file_t, N2::cfloat{0.0f, 0.0f});
    vis_weight.assign(num_prod * num_file_f * num_file_t, 0.0f);
    eval.assign(num_ev * num_file_f * num_file_t, 0.0f);
    evec.assign(num_ev * num_elements * num_file_f * num_file_t, N2::cfloat{0.0f, 0.0f});
    erms.assign(num_file_f * num_file_t, 0.0f);
    gain.assign(num_elements * num_file_f * num_file_t, N2::cfloat{0.0f, 0.0f});
    valid_fpga_count.assign(num_file_f * num_file_t, 0);
    rfi_fpga_count.assign(num_file_f * num_file_t, 0);
    rfi_only_fpga_count.assign(num_file_f * num_file_t, 0);
    pl_fpga_count.assign(num_file_f * num_file_t, 0);
    frac_lost.assign(num_file_f * num_file_t, 1.0f); // match empty frames by default
    frac_rfi.assign(num_file_f * num_file_t, 0.0f);
    frac_rfi_only.assign(num_file_f * num_file_t, 0.0f);
    frac_pl.assign(num_file_f * num_file_t, 0.0f);
    flags.assign(num_elements * num_file_f * num_file_t, 0.0f);
    radiometer_chi2.assign(num_file_f * num_file_t * 3, 0.0f);

    // Additional metadata
    fpga_start_tick.assign(num_file_t, 0);
    frame_length_fpga_ticks.assign(num_file_t, 0);
    time_center_t_inst_ns.assign(num_file_t, 0.0);
    time_center_ut1_ns.assign(num_file_t, 0.0);
    bin_t_inst_ns.assign(num_file_t, 0.0);
    bin_ut1_ns.assign(num_file_t, 0.0);
    bin_delta_ut1_inst.assign(num_file_t, 0.0);
    bin_era_deg.assign(num_file_t, 0.0);
    bin_xp_as.assign(num_file_t, 0.0);
    bin_yp_as.assign(num_file_t, 0.0);
    bin_start_ERA_deg.assign(num_file_t, 0.0);
    bin_end_ERA_deg.assign(num_file_t, 0.0);
    bin_start_ERAL.assign(num_file_t, 0.0);
    bin_end_ERAL.assign(num_file_t, 0.0);
    rfi_frame_excision_enabled.assign(num_file_t, false);
    rfi_frame_excision_num.assign(num_file_t, 0);
    rfi_frame_excision_threshold.assign(num_file_t * MAX_NUM_RFI_THRESHOLDS, 0.0f);
    rfi_frame_excision_fraction.assign(num_file_t * MAX_NUM_RFI_THRESHOLDS, 0.0f);


    added_ft.assign(num_file_f * num_file_t, 0);
}


N2FileData::AddFrameStatus N2FileData::add_frame(const N2FrameView& fv, size_t t_index) {
    const CHORDTelescope& telescope = Telescope::instance().cast<CHORDTelescope>();
    const size_t f_index = fv.freq_id - telescope.min_science_freq_id();

    // Make sure frame hasn't been added yet
    if (f_index >= num_file_f || t_index >= num_file_t
        || fv.freq_id < telescope.min_science_freq_id()) {
        FATAL_ERROR_NON_OO("N2FileData: index out of bounds: (f_index={}, t_index={}). "
                           "Expected f_index < {}, t_index < {}, and freq_id >= {}",
                           f_index, t_index, num_file_f, num_file_t,
                           telescope.min_science_freq_id());
    }
    size_t check_idx = idx_ft(f_index, t_index);
    if (added_ft[check_idx] != 0) {
        FATAL_ERROR_NON_OO("N2FileData: duplicate frame insertion at (f={}, t={})", f_index,
                           t_index);
    }

    // Accept timing differences up to 2 ns (e.g. fuzz on EOP table updates)
    auto ns_close = [](int64_t a, int64_t b, int64_t tol_ns = 2) {
        return std::llabs(a - b) <= tol_ns;
    };
    // Accept timing differences up to 2 ns (e.g. fuzz on EOP table updates)
    auto sec_close = [](double a, double b, double tol_sec = 2e-9) {
        return std::fabs(a - b) <= tol_sec;
    };
    // Accept timing differences up to 2 ns ~ 8.3e-12 deg (e.g. fuzz on EOP table updates)
    auto deg_close = [](double a, double b, double tol_deg = 1e-11) {
        return std::fabs(a - b) <= tol_deg;
    };
    // Accept polar motion drift equivalent to 2 ns of rotation.
    auto arcsec_close = [](double a, double b, double tol_as = 3e-8) {
        return std::fabs(a - b) <= tol_as;
    };

    // Structural data consistency checks
    std::string structural_checks_failures = "";
    auto add_failure = [&](std::string msg) { structural_checks_failures += "\n  - " + msg; };

    if (n2_layout != fv.n2_layout)
        add_failure(fmt::format("n2_layout: {} != {}", N2Layout_to_string(n2_layout),
                                N2Layout_to_string(fv.n2_layout)));
    if (fv.eval.size() != fv.num_ev)
        add_failure(fmt::format("eval.size() != num_ev: {} != {}", fv.eval.size(), fv.num_ev));
    if (fv.evec.size() != fv.num_ev * fv.num_elements)
        add_failure(fmt::format("evec.size() != num_ev * num_elements: {} != {}", fv.evec.size(),
                                fv.num_ev * fv.num_elements));
    if (fv.gain.size() != fv.num_elements)
        add_failure(
            fmt::format("gain.size() != num_elements: {} != {}", fv.gain.size(), fv.num_elements));
    if (fv.flags.size() != fv.num_elements)
        add_failure(fmt::format("flags.size() != num_elements: {} != {}", fv.flags.size(),
                                fv.num_elements));
    if (fv.num_elements != num_elements)
        add_failure(fmt::format("num_elements: {} != {}", fv.num_elements, num_elements));
    if (fv.num_prod != num_prod)
        add_failure(fmt::format("num_prod: {} != {}", fv.num_prod, num_prod));
    if (fv.num_ev != num_ev)
        add_failure(fmt::format("num_ev: {} != {}", fv.num_ev, num_ev));
    if (fv.frame_length_fpga_ticks == 0)
        add_failure(
            fmt::format("frame_length_fpga_ticks must be > 0, got {}", fv.frame_length_fpga_ticks));
    if (fpga_start_tick[t_index] > 0 && fpga_start_tick[t_index] != fv.fpga_start_tick)
        add_failure(fmt::format("fpga_start_tick[t={}] mismatch: stored {} != incoming {}", t_index,
                                fpga_start_tick[t_index], fv.fpga_start_tick));
    if (frame_length_fpga_ticks[t_index] > 0
        && frame_length_fpga_ticks[t_index] != fv.frame_length_fpga_ticks)
        add_failure(fmt::format("frame_length_fpga_ticks[t={}] mismatch: stored {} != incoming {}",
                                t_index, frame_length_fpga_ticks[t_index],
                                fv.frame_length_fpga_ticks));
    if (fv.rfi_frame_excision_num < 0)
        add_failure(fmt::format("rfi_frame_excision_num negative: {}", fv.rfi_frame_excision_num));
    if (fv.rfi_frame_excision_num > MAX_NUM_RFI_THRESHOLDS)
        add_failure(fmt::format("rfi_frame_excision_num exceeds MAX_NUM_RFI_THRESHOLDS: {} > {}",
                                fv.rfi_frame_excision_num, MAX_NUM_RFI_THRESHOLDS));
    if (time_center_t_inst_ns[t_index] > 0
        && !ns_close(time_center_t_inst_ns[t_index], fv.time_center_eop.t_inst_ns))
        add_failure(
            fmt::format("time_center_t_inst_ns[t={}] mismatch: stored {} != incoming {} (ns)",
                        t_index, time_center_t_inst_ns[t_index], fv.time_center_eop.t_inst_ns));
    if (time_center_ut1_ns[t_index] > 0
        && !ns_close(time_center_ut1_ns[t_index], fv.time_center_eop.t_ut1_ns))
        add_failure(fmt::format("time_center_ut1_ns[t={}] mismatch: stored {} != incoming {} (ns)",
                                t_index, time_center_ut1_ns[t_index], fv.time_center_eop.t_ut1_ns));
    if (bin_t_inst_ns[t_index] > 0 && !ns_close(bin_t_inst_ns[t_index], fv.bin_eop.t_inst_ns))
        add_failure(fmt::format("bin_t_inst_ns[t={}] mismatch: stored {} != incoming {} (ns)",
                                t_index, bin_t_inst_ns[t_index], fv.bin_eop.t_inst_ns));
    if (bin_ut1_ns[t_index] > 0 && !ns_close(bin_ut1_ns[t_index], fv.bin_eop.t_ut1_ns))
        add_failure(fmt::format("bin_ut1_ns[t={}] mismatch: stored {} != incoming {} (ns)", t_index,
                                bin_ut1_ns[t_index], fv.bin_eop.t_ut1_ns));
    if (bin_delta_ut1_inst[t_index] > 0
        && !sec_close(bin_delta_ut1_inst[t_index], fv.bin_eop.delta_UT1_inst))
        add_failure(fmt::format("bin_delta_ut1_inst[t={}] mismatch: stored {} != incoming {} (deg)",
                                t_index, bin_delta_ut1_inst[t_index], fv.bin_eop.delta_UT1_inst));
    if (bin_era_deg[t_index] > 0 && !deg_close(bin_era_deg[t_index], fv.bin_eop.ERA_deg))
        add_failure(fmt::format("bin_era_deg[t={}] mismatch: stored {} != incoming {} (deg)",
                                t_index, bin_era_deg[t_index], fv.bin_eop.ERA_deg));
    if (fv.bin_eop.ERA_deg < 0)
        add_failure(fmt::format("bin_eop.ERA_deg < 0: {}", fv.bin_eop.ERA_deg));
    if (fv.bin_eop.ERA_deg >= 360)
        add_failure(fmt::format("bin_eop.ERA_deg >= 360: {}", fv.bin_eop.ERA_deg));
    if (bin_xp_as[t_index] > 0 && !arcsec_close(bin_xp_as[t_index], fv.bin_eop.xp_as))
        add_failure(fmt::format("bin_xp_as[t={}] mismatch: stored {} != incoming {} (arcsec)",
                                t_index, bin_xp_as[t_index], fv.bin_eop.xp_as));
    if (bin_yp_as[t_index] > 0 && !arcsec_close(bin_yp_as[t_index], fv.bin_eop.yp_as))
        add_failure(fmt::format("bin_yp_as[t={}] mismatch: stored {} != incoming {} (arcsec)",
                                t_index, bin_yp_as[t_index], fv.bin_eop.yp_as));
    if (fv.bin_start_ERA_deg < 0)
        add_failure(fmt::format("bin_start_ERA_deg < 0: {}", fv.bin_start_ERA_deg));
    if (fv.bin_start_ERA_deg >= 360)
        add_failure(fmt::format("bin_start_ERA_deg >= 360: {}", fv.bin_start_ERA_deg));
    if (fv.bin_end_ERA_deg < 0)
        add_failure(fmt::format("bin_end_ERA_deg < 0: {}", fv.bin_end_ERA_deg));
    if (fv.bin_end_ERA_deg > 360)
        add_failure(fmt::format("bin_end_ERA_deg > 360: {}", fv.bin_end_ERA_deg));
    // TODO: Don't check these yet, but do when we have ERAL values
    // (bin_start_ERAL[t_index] < 0)
    // (bin_start_ERAL[t_index] > 360)
    // (bin_end_ERAL[t_index] < 0)
    // (bin_end_ERAL[t_index] > 360)

    if (!structural_checks_failures.empty())
        FATAL_ERROR_NON_OO("N2FileData: frame information mismatch or invalid at (f={}, t={}):{}",
                           f_index, t_index, structural_checks_failures);


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
    for (size_t i = 0; i < 3; i++)
        radiometer_chi2[3 * idx_ft(f_index, t_index) + i] = fv.radiometer_chi2[i];
    // Store fraction lost and RFI
    const uint64_t frame_len_ticks = fv.frame_length_fpga_ticks;
    const uint64_t n_valid = fv.n_valid_fpga_ticks;
    const uint64_t n_rfi = fv.n_rfi_fpga_ticks;
    const uint64_t n_rfi_only = fv.n_rfi_only_fpga_ticks;
    const uint64_t n_pl = fv.n_pl_fpga_ticks;
    valid_fpga_count[idx_ft(f_index, t_index)] = n_valid;
    rfi_fpga_count[idx_ft(f_index, t_index)] = n_rfi;
    rfi_only_fpga_count[idx_ft(f_index, t_index)] = n_rfi_only;
    pl_fpga_count[idx_ft(f_index, t_index)] = n_pl;
    frac_lost[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (1.0f - float(n_valid) / float(frame_len_ticks)) : 0.0f;
    frac_rfi[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (float(n_rfi) / float(frame_len_ticks)) : 0.0f;
    frac_rfi_only[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (float(n_rfi_only) / float(frame_len_ticks)) : 0.0f;
    frac_pl[idx_ft(f_index, t_index)] =
        (frame_len_ticks > 0) ? (float(n_pl) / float(frame_len_ticks)) : 0.0f;
    // Store per-time metadata
    fpga_start_tick[t_index] = fv.fpga_start_tick;
    frame_length_fpga_ticks[t_index] = fv.frame_length_fpga_ticks;
    time_center_t_inst_ns[t_index] = fv.time_center_eop.t_inst_ns;
    time_center_ut1_ns[t_index] = fv.time_center_eop.t_ut1_ns;
    bin_t_inst_ns[t_index] = fv.bin_eop.t_inst_ns;
    bin_ut1_ns[t_index] = fv.bin_eop.t_ut1_ns;
    bin_delta_ut1_inst[t_index] = fv.bin_eop.delta_UT1_inst;
    bin_era_deg[t_index] = fv.bin_eop.ERA_deg;
    bin_xp_as[t_index] = fv.bin_eop.xp_as;
    bin_yp_as[t_index] = fv.bin_eop.yp_as;
    bin_start_ERA_deg[t_index] = fv.bin_start_ERA_deg;
    bin_end_ERA_deg[t_index] = fv.bin_end_ERA_deg;
    bin_start_ERAL[t_index] = fv.bin_start_ERAL;
    bin_end_ERAL[t_index] = fv.bin_end_ERAL;
    rfi_frame_excision_enabled[t_index] = fv.rfi_frame_excision_enabled;
    rfi_frame_excision_num[t_index] = fv.rfi_frame_excision_num;
    std::copy(fv.rfi_frame_excision_threshold.begin(), fv.rfi_frame_excision_threshold.end(),
              rfi_frame_excision_threshold.begin() + t_index * MAX_NUM_RFI_THRESHOLDS);
    std::copy(fv.rfi_frame_excision_fraction.begin(), fv.rfi_frame_excision_fraction.end(),
              rfi_frame_excision_fraction.begin() + t_index * MAX_NUM_RFI_THRESHOLDS);

    // Mark (f, t) as added
    size_t si = idx_ft(f_index, t_index);
    added_ft[si] = 1;
    ++added_count; // increment total number of frames added

    return AddFrameStatus::Success;
}

std::optional<std::string> N2FileData::_get_final_filename() {

    // Check we have at least one valid time
    if (num_file_t == 0)
        return std::nullopt;

    // Get the minimum time in the file based on fpga_start_tick,
    std::optional<uint64_t> fpga_start_tick_min = std::nullopt;
    for (size_t t = 0; t < num_file_t; ++t) {
        if (fpga_start_tick[t] != 0) {
            if (!fpga_start_tick_min.has_value()
                || fpga_start_tick[t] < fpga_start_tick_min.value()) {
                fpga_start_tick_min = fpga_start_tick[t];
            }
        }
    }
    if (!fpga_start_tick_min.has_value())
        return std::nullopt;

    timespec earliest_fpga_tick_time = Telescope::instance().to_time(fpga_start_tick_min.value());

    std::ostringstream buf;
    if (file_mode == CHIME) {
        // TODO: This isn't tested, and is a modification of the original CHIME naming scheme.
        // CHIME files are named by an integer indicating the start time of the chunk, relative
        // to the start of the acquisition. Here we modify that slightly, to 9 digits, and
        // set it to be seconds since the start of the instrument.
        timespec instrument_start_time = Telescope::instance().to_time(0);
        auto elapsed_sec = earliest_fpga_tick_time.tv_sec - instrument_start_time.tv_sec;
        buf << std::setw(9) << elapsed_sec; // Raw seconds since , buffered by 9 0's
        // Include frequency info (0000 for full band)
        buf << "_0000.h5";
    } else {
        // Construct final filename based on earliest time and abs_file_idx
        std::time_t time_t_format = earliest_fpga_tick_time.tv_sec;    // seconds
        const std::uint64_t ns_part = earliest_fpga_tick_time.tv_nsec; // sub-second
        const std::string abs_idx_str = fmt::format("{:010}", abs_file_idx);
        buf << "vis_" << abs_idx_str << "_"
            << std::put_time(std::gmtime(&time_t_format), "%Y%m%dT_%H%M%S");
        // Include nanosecond suffix to avoid collisions for sub-second file windows
        buf << "_" << std::setw(9) << std::setfill('0') << ns_part;
        buf << ".h5";
    }

    const std::string basename = buf.str();
    std::filesystem::path final_path = std::filesystem::path(base_dir) / basename;
    return final_path.string();
}

bool N2FileData::flush_to_disk() {
    bool has_error = false;

    if (!h5_file)
        return false;

    DEBUG_NON_OO("hdf5N2Write: Writing to {}", partial_filepath);

    std::string flags_group_prefix = file_mode == CHIME ? "/flags" : "";

    // Add and write configs in configTracker
    try {
        std::vector<std::string> json_objs = kotekan::ConfigTracker::instance().getAllJSONConfigs();
        HighFive::DataSet json_dset = h5_file->getDataSet("/config_json");
        auto space = json_dset.getSpace();
        auto cur_dims = space.getDimensions();
        std::size_t old_n = cur_dims.empty() ? 0 : cur_dims[0];
        std::size_t extra_n = json_objs.size();
        json_dset.resize({old_n + extra_n});
        json_dset.select({old_n}, {extra_n}).write(json_objs);
    } catch (const HighFive::Exception& e) {
        FATAL_ERROR_NON_OO("Failed to write config JSON to HDF5 file {}: {}", partial_filepath,
                           e.what());
        has_error = true;
    } catch (const std::exception& e) {
        FATAL_ERROR_NON_OO("Failed to write config JSON to HDF5 file {}: {}", partial_filepath,
                           e.what());
        has_error = true;
    }

    // Write directly from buffers, use write_raw(ptr) so memspace is applied
    try {
        h5_file->getDataSet("/frames_added")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(added_ft.data());

        h5_file->getDataSet("/vis")
            .select({0, 0, 0}, {num_file_f, num_prod, num_file_t})
            .write_raw(vis.data());
        h5_file->getDataSet(flags_group_prefix + "/vis_weight")
            .select({0, 0, 0}, {num_file_f, num_prod, num_file_t})
            .write_raw(vis_weight.data());
        h5_file->getDataSet("/eval")
            .select({0, 0, 0}, {num_file_f, num_ev, num_file_t})
            .write_raw(eval.data());
        h5_file->getDataSet("/evec")
            .select({0, 0, 0, 0}, {num_file_f, num_ev, num_elements, num_file_t})
            .write_raw(evec.data());
        h5_file->getDataSet("/erms")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(erms.data());
        h5_file->getDataSet(flags_group_prefix + "/valid_fpga_count")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(valid_fpga_count.data());
        h5_file->getDataSet(flags_group_prefix + "/rfi_fpga_count")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(rfi_fpga_count.data());
        h5_file->getDataSet(flags_group_prefix + "/rfi_only_fpga_count")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(rfi_only_fpga_count.data());
        h5_file->getDataSet(flags_group_prefix + "/pl_fpga_count")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(pl_fpga_count.data());
        h5_file->getDataSet(flags_group_prefix + "/frac_lost")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(frac_lost.data());
        h5_file->getDataSet(flags_group_prefix + "/frac_rfi")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(frac_rfi.data());
        h5_file->getDataSet(flags_group_prefix + "/frac_rfi_only")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(frac_rfi_only.data());
        h5_file->getDataSet(flags_group_prefix + "/frac_pl")
            .select({0, 0}, {num_file_f, num_file_t})
            .write_raw(frac_pl.data());
        h5_file->getDataSet("/gain")
            .select({0, 0, 0}, {num_file_f, num_elements, num_file_t})
            .write_raw(gain.data());
        h5_file->getDataSet(flags_group_prefix + "/flags")
            .select({0, 0, 0}, {num_file_f, num_elements, num_file_t})
            .write_raw(flags.data());
        h5_file->getDataSet("/radiometer_chi2")
            .select({0, 0, 0}, {num_file_f, num_file_t, 3})
            .write_raw(radiometer_chi2.data());

        h5_file->getDataSet("/fpga_start_tick").write(fpga_start_tick);
        h5_file->getDataSet("/frame_length_fpga_ticks").write(frame_length_fpga_ticks);
        h5_file->getDataSet("/time_center_t_inst_ns").write(time_center_t_inst_ns);
        h5_file->getDataSet("/time_center_ut1_ns").write(time_center_ut1_ns);
        h5_file->getDataSet("/bin_t_inst_ns").write(bin_t_inst_ns);
        h5_file->getDataSet("/bin_ut1_ns").write(bin_ut1_ns);
        h5_file->getDataSet("/bin_delta_ut1_inst").write(bin_delta_ut1_inst);
        h5_file->getDataSet("/bin_ERA_deg").write(bin_era_deg);
        h5_file->getDataSet("/bin_xp_as").write(bin_xp_as);
        h5_file->getDataSet("/bin_yp_as").write(bin_yp_as);
        h5_file->getDataSet("/bin_start_ERA_deg").write(bin_start_ERA_deg);
        h5_file->getDataSet("/bin_end_ERA_deg").write(bin_end_ERA_deg);
        h5_file->getDataSet("/bin_start_ERAL").write(bin_start_ERAL);
        h5_file->getDataSet("/bin_end_ERAL").write(bin_end_ERAL);
        h5_file->getDataSet("/rfi_frame_excision_enabled").write(rfi_frame_excision_enabled);
        h5_file->getDataSet("/rfi_frame_excision_num").write(rfi_frame_excision_num);
        h5_file->getDataSet("/rfi_frame_excision_threshold")
            .select({0, 0}, {num_file_t, MAX_NUM_RFI_THRESHOLDS})
            .write_raw(rfi_frame_excision_threshold.data());
        h5_file->getDataSet("/rfi_frame_excision_fraction")
            .select({0, 0}, {num_file_t, MAX_NUM_RFI_THRESHOLDS})
            .write_raw(rfi_frame_excision_fraction.data());
    } catch (const HighFive::Exception& e) {
        FATAL_ERROR_NON_OO("Failed to write data to HDF5 file {}: {}", partial_filepath, e.what());
        has_error = true;
    } catch (const std::exception& e) {
        FATAL_ERROR_NON_OO("Failed to write data to HDF5 file {}: {}", partial_filepath, e.what());
        has_error = true;
    }

    // Verify digital gains file hasn't been modified since it was copied at file creation
    if (!baseband_gain_file.empty() && gains_file_hash) {
        std::optional<size_t> current_hash = hash_file_contents(baseband_gain_file);
        if (current_hash && current_hash != gains_file_hash) {
            FATAL_ERROR_NON_OO("Digital gains file {} has been modified since file creation!",
                               baseband_gain_file);
            has_error = true;
            _check_create_attribute(*h5_file, "digital_gains_source_file_modified",
                                    std::string("true"));
        }
    }

    return !has_error;
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
    _base_dir(get_acq_base_dir_path(config.get<std::string>(unique_name, "base_dir"))),
    _baseband_gain_file(config.get_default<std::string>(unique_name, "baseband_gain_file", "")),
    _baseband_gain_url(config.get_default<std::string>(unique_name, "baseband_gain_url", "")),
    _baseband_gain_update_idx(config.get_default<int>(unique_name, "baseband_gain_update_idx", -1)),
    _num_file_t(config.get<std::uint64_t>(unique_name, "num_file_t")),
    _compression(config.get_default<std::string>(unique_name, "compression", "none")),
    _compression_level(config.get_default<std::uint64_t>(unique_name, "compression_level", 0)),
    _use_bitshuffle(config.get_default<bool>(unique_name, "use_bitshuffle", false)),
    _blocksize_f(config.get_default<std::uint64_t>(unique_name, "blocksize_f", 16)),
    _blocksize_p(config.get_default<std::uint64_t>(unique_name, "blocksize_p", 16)),
    _blocksize_t(config.get_default<std::uint64_t>(unique_name, "blocksize_t", _num_file_t)),
    _late_frame_grace_seconds(
        config.get_default<std::uint64_t>(unique_name, "late_frame_grace_seconds", 60)),
    _max_frames(config.get_default<int>(unique_name, "max_frames", -1)),
    _buffer(get_buffer("in_buf")),
    _write_time_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_hdf5N2Write_write_time_seconds", unique_name)),
    _n_datasets_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_hdf5N2Write_n_datasets", unique_name)),
    _open_file_info_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_hdf5N2Write_open_file_info", unique_name,
        {"abs_file_idx", "partial_path", "file_mode"})),
    _open_file_age_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_hdf5N2Write_open_file_age_seconds", unique_name, {"abs_file_idx"})),
    _file_completion_fraction_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_hdf5N2Write_file_completion_fraction", unique_name, {"abs_file_idx"})),
    _add_frame_errors_metric(kotekan::prometheus::Metrics::instance().add_counter(
        "kotekan_hdf5N2Write_add_frame_errors_total", unique_name, {"reason"})),
    _last_add_frame_error_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_hdf5N2Write_last_add_frame_error_seconds", unique_name,
        {"reason", "abs_file_idx", "freq_id", "t_index"})),
    _finalize_failures_metric(kotekan::prometheus::Metrics::instance().add_counter(
        "kotekan_hdf5N2Write_finalize_failures_total", unique_name, {"reason"})),
    _unfinalized_file_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_hdf5N2Write_unfinalized_file", unique_name, {"abs_file_idx", "partial_path"})) {

    _buffer->register_consumer(unique_name);

    // baseband_gain_file and baseband_gain_url are mutually exclusive. If the URL is
    // set, the actual local path is filled in from main_thread() after the .partial
    // directory exists.
    if (!_baseband_gain_file.empty() && !_baseband_gain_url.empty()) {
        FATAL_ERROR("baseband_gain_file and baseband_gain_url are mutually exclusive; set only "
                    "one of them (file='{}', url='{}')",
                    _baseband_gain_file, _baseband_gain_url);
    }

    // Validate file window configuration
    if (_num_file_t == 0) {
        FATAL_ERROR("num_file_t must be > 0 for hdf5N2Write");
    }
    // Ensure the input buffer is an N2Buffer
    if (_buffer->buffer_type != "N2") {
        FATAL_ERROR("Input buffer must be a N2-type buffer.");
    }

    // Reject configs where two hdf5N2Write instances share an output directory.
    {
        std::vector<std::pair<std::string, std::string>> peers;
        collect_hdf5N2Write_base_dirs(config.get_full_config_json(), "", peers);

        const std::string my_dir = config.get<std::string>(unique_name, "base_dir");
        const std::string my_norm = normalize_base_dir(my_dir);
        for (const auto& [peer_name, peer_dir] : peers) {
            if (peer_name == unique_name)
                continue;
            if (normalize_base_dir(peer_dir) == my_norm) {
                FATAL_ERROR("hdf5N2Write[{}]: base_dir '{}' conflicts with stage '{}'. "
                            "Each hdf5N2Write instance must use a unique base_dir.",
                            unique_name, my_dir, peer_name);
            }
        }
    }

    if (_max_frames >= 0) {
        if (_max_frames % _num_file_t != 0)
            WARN("max_frames ({}) is not a multiple of num_file_t ({}). An output file will only "
                 "be partially filled.",
                 _max_frames, _num_file_t);
        ++waiting_for_max_frames;
    }
}

hdf5N2Write::~hdf5N2Write() {}

std::uint64_t hdf5N2Write::_get_abs_file_idx(const N2FrameView& fv) const {
    // Get the absolute file index based on the absolute frame index and
    // configured number of time frames per file.

    // Just truncate towards zero
    return fv.abs_time_idx / _num_file_t;
}

bool hdf5N2Write::_finalize_file(N2FileData& filedata) {
    const std::string abs_idx = std::to_string(filedata.abs_file_idx);
    DEBUG_NON_OO("hdf5N2Write: Flushing and closing file {}...", abs_idx);
    try {
        filedata.flush_to_disk();
    } catch (const HighFive::Exception& e) {
        FATAL_ERROR("Failed to flush dataset {} to disk: {}", filedata.partial_filepath, e.what());
        _finalize_failures_metric.labels({"flush"}).inc();
        _mark_unfinalized(filedata);
        _clear_open_file_metrics(filedata);
        try {
            filedata.close();
        } catch (...) {
        }
        return false;
    } catch (const std::exception& e) {
        FATAL_ERROR("Failed to flush dataset {} to disk: {}", filedata.partial_filepath, e.what());
        _finalize_failures_metric.labels({"flush"}).inc();
        _mark_unfinalized(filedata);
        _clear_open_file_metrics(filedata);
        try {
            filedata.close();
        } catch (...) {
        }
        return false;
    }

    filedata.close();

    // Attempt rename from partial to final
    DEBUG_NON_OO("hdf5N2Write: Renaming file {}...", abs_idx);
    auto ds_filename = filedata._get_final_filename();
    if (!ds_filename) {
        WARN("Could not determine final filename for {} (no UT1 found); leaving partial in place.",
             filedata.partial_filepath);
        _finalize_failures_metric.labels({"missing_final_name"}).inc();
        _mark_unfinalized(filedata);
        _clear_open_file_metrics(filedata);
        return false;
    }

    int r = std::rename(filedata.partial_filepath.c_str(), ds_filename->c_str());
    if (r != 0) {
        const char* msg = strerror(errno);
        FATAL_ERROR("Failed to rename partial dataset to final: {} -> {}: {}",
                    filedata.partial_filepath, *ds_filename, msg);
        _finalize_failures_metric.labels({"rename"}).inc();
        _mark_unfinalized(filedata);
        _clear_open_file_metrics(filedata);
        // Attempt to quarantine the partial file to avoid repeated collisions
        std::filesystem::path ignored = filedata.partial_filepath + ".ignored";
        int attempt = 0;
        while (std::filesystem::exists(ignored) && attempt < 5) {
            ignored = filedata.partial_filepath + ".ignored" + std::to_string(++attempt);
        }
        try {
            std::filesystem::rename(filedata.partial_filepath, ignored);
            FATAL_ERROR("Moved partial dataset to {} after failed rename", ignored.string());
        } catch (const std::exception& ex) {
            FATAL_ERROR("Failed to move partial dataset {} to ignored path {}: {}",
                        filedata.partial_filepath, ignored.string(), ex.what());
        }
        return false;
    }
    _clear_open_file_metrics(filedata);
    _unfinalized_file_metric.labels({abs_idx, filedata.partial_filepath}).set(0.0);

    INFO_NON_OO("hdf5N2Write: Wrote final file {}: {}", abs_idx, *ds_filename);

    return true;
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
        _update_file_metrics(obj);
        if (now_s - obj.last_update_wall_s >= double(_late_frame_grace_seconds)) {
            bool finalized = _finalize_file(*file_it->second);
            (void)finalized;
            file_it = files.erase(file_it);
        } else {
            ++file_it;
        }
    }
}

bool hdf5N2Write::_finalfile_exists(std::uint64_t abs_file_idx,
                                    const std::string& search_dir) const {
    // Check for both padded and legacy unpadded prefixes to avoid duplicating existing files.
    const std::string abs_idx_str_padded = fmt::format("{:010}", abs_file_idx);
    const std::string abs_idx_str_unpadded = std::to_string(abs_file_idx);
    const std::string prefix_padded = "vis_" + abs_idx_str_padded + "_";
    const std::string prefix_unpadded = "vis_" + abs_idx_str_unpadded + "_";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(search_dir)) {
            if (entry.is_regular_file()) {
                const std::string filename = entry.path().filename().string();
                if (std::filesystem::path(filename).extension() == ".h5"
                    && (filename.rfind(prefix_padded, 0) == 0
                        || filename.rfind(prefix_unpadded, 0) == 0)) {
                    return true;
                }
            }
        }
    } catch (const std::filesystem::filesystem_error&) {
        return false;
    }
    return false;
}

void hdf5N2Write::_record_file_open(const N2FileData& filedata) const {
    const std::string abs_idx = std::to_string(filedata.abs_file_idx);
    const std::string mode =
        filedata.file_mode == N2FileData::CHIME ? std::string("CHIME") : std::string("CHORD");
    _open_file_info_metric.labels({abs_idx, filedata.partial_filepath, mode}).set(1.0);
    _open_file_age_metric.labels({abs_idx}).set(mono_time_s() - filedata.open_wall_s);
    _file_completion_fraction_metric.labels({abs_idx}).set(filedata.completion_fraction());
    _unfinalized_file_metric.labels({abs_idx, filedata.partial_filepath}).set(0.0);
}

void hdf5N2Write::_update_file_metrics(const N2FileData& filedata) const {
    const std::string abs_idx = std::to_string(filedata.abs_file_idx);
    _open_file_age_metric.labels({abs_idx}).set(mono_time_s() - filedata.open_wall_s);
    _file_completion_fraction_metric.labels({abs_idx}).set(filedata.completion_fraction());
}

void hdf5N2Write::_clear_open_file_metrics(const N2FileData& filedata) const {
    const std::string abs_idx = std::to_string(filedata.abs_file_idx);
    const std::string mode =
        filedata.file_mode == N2FileData::CHIME ? std::string("CHIME") : std::string("CHORD");
    _open_file_info_metric.labels({abs_idx, filedata.partial_filepath, mode}).set(0.0);
    _open_file_age_metric.labels({abs_idx}).set(0.0);
}

void hdf5N2Write::_mark_unfinalized(const N2FileData& filedata) const {
    const std::string abs_idx = std::to_string(filedata.abs_file_idx);
    _unfinalized_file_metric.labels({abs_idx, filedata.partial_filepath}).set(1.0);
}

void hdf5N2Write::_record_add_frame_error(const std::string& reason, std::uint64_t abs_file_idx,
                                          int32_t freq_id, std::uint64_t t_index) const {
    _add_frame_errors_metric.labels({reason}).inc();
    _last_add_frame_error_metric
        .labels({reason, std::to_string(abs_file_idx), std::to_string(freq_id),
                 std::to_string(t_index)})
        .set(mono_time_s());
}

void hdf5N2Write::main_thread() {
    double avg_write_time = 0.0;

    const double start_time = mono_time_s(); // for logging elapsed time
    N2::frameID in_frame_id(_buffer);        // Input frame ID tracker
    int frame_counter = 0;                   // Count of frames written

    /// file data for writing (multiple may be open simultaneously)
    /// Keyed by absolute file id = absolute frame index / _num_file_t
    std::map<size_t, std::unique_ptr<N2FileData>> filedata;

    // Create acquisition base_dir and partial dir if necessary (recursively)
    INFO("Acquisition directory: {}", _base_dir);
    {
        if (mkdir_p(_base_dir.c_str(), 0777) != 0) {
            const char* const msg = strerror(errno);
            FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", _base_dir.c_str(), msg);
        }
        std::string partial_dir = _base_dir + "/.partial";
        if (mkdir_p(partial_dir.c_str(), 0777) != 0) {
            const char* const msg = strerror(errno);
            FATAL_ERROR("Could not create directory \"{:s}\":\n{:s}", partial_dir.c_str(), msg);
        }

        // If a URL was provided instead of a local path, fetch the gains file
        // once into the partial dir and point _baseband_gain_file at the cached
        // copy. From here on the existing local-file code path is used.
        if (!_baseband_gain_url.empty()) {
            const std::string cached = partial_dir + "/baseband_gains.h5";
            download_url_to_file(_baseband_gain_url, cached);
            _baseband_gain_file = cached;
        }
    }

    // Main stage thread
    while (!stop_thread) {

        // Wait for the next frame
        const std::uint8_t* const frame = _buffer->wait_for_full_frame(unique_name, in_frame_id);
        if (!frame)
            break;

        // Fetch metadata and create N2 frame view
        N2FrameView fv(_buffer, in_frame_id);

        // Start timer
        const double frame_recv_time = mono_time_s();
        const double total_elapsed_time = frame_recv_time - start_time;
        INFO("Received buffer {} frame {} (duration_s {})", _buffer->buffer_name, in_frame_id,
             total_elapsed_time);

        const std::uint64_t file_t_index = fv.abs_time_idx % _num_file_t;
        auto abs_file_idx = _get_abs_file_idx(fv);

        // Ensure dataset exists/open
        N2FileData* N2FileData_ptr = nullptr;
        bool created_new_filedata = false;
        auto fd = filedata.find(abs_file_idx);
        if (fd != filedata.end()) {
            // Dataset already open
            N2FileData_ptr = fd->second.get();
        } else if (_finalfile_exists(abs_file_idx, _base_dir)) {
            // If final file already exists, drop/ignore this frame (late arrival)
            WARN("Finalized file already exists for this frame's file window, dropping frame: "
                 "abs_file_idx={}, frame_id={}",
                 abs_file_idx, in_frame_id);
            _record_add_frame_error("final_exists", abs_file_idx, fv.freq_id, file_t_index);

            // Mark frame as done, finalize, and continue
            _buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_files(filedata, nullptr);
            continue;
        } else {
            // Create N2FileData for file (also looks for .partial)
            auto N2FileData_obj = std::make_unique<N2FileData>(
                N2FileData::CHORD, _num_file_t, fv, frame_recv_time, abs_file_idx, _blocksize_f,
                _blocksize_p, _blocksize_t, _compression, _compression_level, _use_bitshuffle,
                _base_dir, _baseband_gain_file, _baseband_gain_update_idx);

            filedata.emplace(abs_file_idx, std::move(N2FileData_obj));
            N2FileData_ptr = filedata.find(abs_file_idx)->second.get();
            created_new_filedata = true;
        }

        if (!N2FileData_ptr || !N2FileData_ptr->h5_file) {
            // Mark frame as done, finalize, and continue
            ERROR("Dataset is null. Failed to open dataset.");
            _record_add_frame_error("dataset_null", abs_file_idx, fv.freq_id, file_t_index);
            _buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_files(filedata, nullptr);
            continue;
        }

        if (created_new_filedata) {
            _record_file_open(*N2FileData_ptr);
        }

        // Attempt to add frame to dataset
        DEBUG("Adding frame (t_idx={}, freq_id={}) to file {} slot (t={})", fv.abs_time_idx,
              fv.freq_id, abs_file_idx, file_t_index);
        auto add_status =
            N2FileData_ptr->add_frame(fv, file_t_index); // performs error checking internally.
        if (add_status != N2FileData::AddFrameStatus::Success) {
            std::string reason = "unknown";
            switch (add_status) {
                case N2FileData::AddFrameStatus::OutOfBounds:
                    reason = "out_of_bounds";
                    break;
                case N2FileData::AddFrameStatus::Duplicate:
                    reason = "duplicate";
                    break;
                case N2FileData::AddFrameStatus::MetadataMismatch:
                    reason = "metadata_mismatch";
                    break;
                case N2FileData::AddFrameStatus::Success:
                    reason = "success";
                    break;
            }
            // Mark frame as done, finalize, and continue
            ERROR("Failed to add frame to dataset (f={}, t={}), reason={}", fv.freq_id,
                  file_t_index, reason);
            _record_add_frame_error(reason, abs_file_idx, fv.freq_id, file_t_index);
            _buffer->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id++;
            _grace_finalize_files(filedata, nullptr);
            continue;
        }
        DEBUG("File {} has {} of {} slots full.", abs_file_idx, N2FileData_ptr->get_added_count(),
              N2FileData_ptr->get_expected_count());
        N2FileData_ptr->last_update_wall_s = frame_recv_time;
        _update_file_metrics(*N2FileData_ptr);

        // If buffer full, flush
        double elapsed_writing_frame = 0.0;
        if (N2FileData_ptr->full()) {
            const double t0 = mono_time_s();
            bool finalized = _finalize_file(*N2FileData_ptr);
            (void)finalized;
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
        _write_time_metric.set(elapsed_writing_frame);
        avg_write_time =
            (avg_write_time * frame_counter + elapsed_writing_frame) / double(frame_counter + 1);
        _n_datasets_metric.set(filedata.size());

        // Mark frame as done
        _buffer->mark_frame_empty(unique_name, in_frame_id);

        if (_max_frames >= 0 && frame_counter + 1 >= _max_frames) {
            WARN("Processed {} frames with average write time {}, shutting down Kotekan",
                 frame_counter + 1, avg_write_time);
            break;
        }
        frame_counter++;
        in_frame_id++;

        // After handling this frame, scan for grace-based finalizations
        _grace_finalize_files(filedata, &abs_file_idx);

    } // while !stop_thread

    // Finalize any partially-filled datasets on exit
    for (auto& file : filedata) {
        bool finalized = _finalize_file(*file.second);
        (void)finalized;
    }
    filedata.clear();

    if (_max_frames >= 0) {

        // Unregister to allow the pipeline to continue, unless I'm the last
        // consumer on this buffer.
        _buffer->unregister_consumer(unique_name, true);

        if (--waiting_for_max_frames == 0) {
            WARN("Shutting down Kotekan");
            exit_kotekan(CLEAN_EXIT);
        }
    }

    DEBUG("exiting");
}
