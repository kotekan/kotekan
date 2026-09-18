#include "basebandGains.hpp"

#include "kotekanLogging.hpp"
#include "restClient.hpp"

#include "fmt.hpp"
#include "json.hpp"

#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <hdf5.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5Group.hpp>
#include <iterator>
#include <vector>

namespace kotekan {

namespace {

/// FNV-1a over a file's bytes. Chosen over std::hash so that the digest written into one
/// writer's output can be compared with another's, including across rebuilds.
std::optional<std::uint64_t> hash_file_contents(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return std::nullopt;

    std::uint64_t h = 1469598103934665603ULL;
    char buf[64 * 1024];
    while (ifs.read(buf, sizeof(buf)) || ifs.gcount() > 0) {
        const std::streamsize n = ifs.gcount();
        for (std::streamsize i = 0; i < n; i++) {
            h ^= static_cast<unsigned char>(buf[i]);
            h *= 1099511628211ULL;
        }
        if (ifs.eof())
            break;
    }
    if (ifs.bad())
        return std::nullopt;
    return h;
}

/// The archive stores complex64 as an HDF5 compound {float r; float i;}, which is
/// layout-compatible with std::complex<float>.
HighFive::CompoundType complex64_type() {
    return HighFive::CompoundType{
        {"r", HighFive::AtomicType<float>{}},
        {"i", HighFive::AtomicType<float>{}},
    };
}

/// Fetch `path` from `host:port` over plain HTTP via kotekan's built-in restClient and write
/// the response body verbatim to `dest_path`. The archive is small enough that the bytes land
/// in a std::string before hitting disk. FATAL on any fetch/write error.
void download_to_file(const std::string& label, const std::string& host, std::uint16_t port,
                      const std::string& path, const std::string& dest_path) {
    INFO_NON_OO("{:s}: downloading digital gains file from http://{:s}:{:d}{:s}", label, host, port,
                path);
    restClient::restReply reply =
        restClient::instance().make_request_blocking(path, nlohmann::json::object(), host, port);
    if (!reply.first) {
        FATAL_ERROR_NON_OO("{:s}: failed to fetch gains file from http://{:s}:{:d}{:s}", label,
                           host, port, path);
        return;
    }

    // Write to a temporary beside the destination and rename, so that a reader never sees a
    // half-written archive and two writers racing on the same path cannot interleave.
    const std::string tmp_path = dest_path + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        FATAL_ERROR_NON_OO("{:s}: could not open '{:s}' for writing downloaded gains file", label,
                           tmp_path);
        return;
    }
    ofs.write(reply.second.data(), static_cast<std::streamsize>(reply.second.size()));
    ofs.close();
    if (!ofs) {
        FATAL_ERROR_NON_OO("{:s}: failed to write {:d} bytes to '{:s}'", label, reply.second.size(),
                           tmp_path);
        return;
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, dest_path, ec);
    if (ec) {
        FATAL_ERROR_NON_OO("{:s}: failed to move '{:s}' into place at '{:s}': {:s}", label,
                           tmp_path, dest_path, ec.message());
        return;
    }

    INFO_NON_OO("{:s}: wrote {:d} bytes to '{:s}'", label, reply.second.size(), dest_path);
}

} // namespace

BasebandGains::BasebandGains(kotekan::Config& config, const std::string& unique_name,
                             const std::string& stage_label) :
    _stage_label(stage_label),
    _gain_file(config.get_default<std::string>(unique_name, "baseband_gain_file", "")),
    _update_idx(config.get_default<int>(unique_name, "baseband_gain_update_idx", -1)) {

    // baseband_gain_host_info names a config path (e.g. /fpga_controller) to a block exposing
    // host, port, and an optional gains_endpoint. The fetch itself happens in
    // resolve_local_copy(), once the acquisition directory exists.
    const std::string gain_host_ref =
        config.get_default<std::string>(unique_name, "baseband_gain_host_info", "");
    if (!gain_host_ref.empty()) {
        _host = config.get<std::string>(gain_host_ref, "host");
        const int port_int = config.get<int>(gain_host_ref, "port");
        if (port_int <= 0 || port_int > 65535) {
            FATAL_ERROR_NON_OO("{:s}: invalid {:s}.port: {:d}", _stage_label, gain_host_ref,
                               port_int);
        }
        _port = static_cast<std::uint16_t>(port_int);
        _endpoint = config.get_default<std::string>(gain_host_ref, "gains_endpoint",
                                                    "/get-current-gain-file");
    }

    if (!_gain_file.empty() && !gain_host_ref.empty()) {
        FATAL_ERROR_NON_OO("{:s}: baseband_gain_file and baseband_gain_host_info are mutually "
                           "exclusive; set only one of them (file='{:s}', host_info='{:s}')",
                           _stage_label, _gain_file, gain_host_ref);
    }
}

bool BasebandGains::available() const {
    return !_gain_file.empty() && std::filesystem::exists(_gain_file);
}

std::optional<std::uint64_t> BasebandGains::content_hash() const {
    if (_gain_file.empty())
        return std::nullopt;
    if (!_hash_computed) {
        _hash_cache = hash_file_contents(_gain_file);
        _hash_computed = true;
    }
    return _hash_cache;
}

void BasebandGains::resolve_local_copy(const std::string& partial_dir,
                                       const std::string& cache_stem) {
    if (_host.empty())
        return;

    const std::string cached = partial_dir + "/" + cache_stem + ".h5";
    download_to_file(_stage_label, _host, _port, _endpoint, cached);
    _gain_file = cached;
    _hash_computed = false;
}

void BasebandGains::copy_into(HighFive::File& file) const {
    if (_gain_file.empty()) {
        DEBUG_NON_OO("{:s}: no digital gains configured, skipping /digital_gains.", _stage_label);
        return;
    }
    if (file.exist("/digital_gains"))
        return;

    if (!std::filesystem::exists(_gain_file)) {
        FATAL_ERROR_NON_OO("{:s}: digital gains file {:s} does not exist.", _stage_label,
                           _gain_file);
        return;
    }

    try {
        HighFive::File src(_gain_file, HighFive::File::ReadOnly);

        // Copy the archive's entire root group in as /digital_gains.
        herr_t err =
            H5Ocopy(src.getId(), ".", file.getId(), "digital_gains", H5P_DEFAULT, H5P_DEFAULT);
        if (err < 0) {
            FATAL_ERROR_NON_OO("{:s}: H5Ocopy failed copying gains file {:s} into output.",
                               _stage_label, _gain_file);
            return;
        }

        file.createAttribute("digital_gains_source_file", _gain_file);

        // A digest of the archive, so that a tool merging several writers' files can check
        // they all carry the same gains without comparing the archives element by element.
        const auto digest = content_hash();
        if (digest)
            file.createAttribute("digital_gains_hash", *digest);

        auto dg_group = file.getGroup("/digital_gains");
        int selected_idx = _update_idx;
        if (selected_idx < 0 && dg_group.exist("gain_coeff")) {
            const auto dims = dg_group.getDataSet("gain_coeff").getDimensions();
            if (!dims.empty() && dims[0] > 0)
                selected_idx = static_cast<int>(dims[0] - 1);
        }
        dg_group.createAttribute("selected_update_idx", selected_idx);

        INFO_NON_OO("{:s}: copied digital gains from {:s} into output (selected_update_idx={:d}).",
                    _stage_label, _gain_file, selected_idx);
    } catch (const HighFive::Exception& e) {
        FATAL_ERROR_NON_OO("{:s}: failed to copy gains file {:s} into output: {:s}", _stage_label,
                           _gain_file, e.what());
    }
}

void BasebandGains::write_by_dish(HighFive::File& file, const std::vector<int>& gain_input,
                                  std::size_t n_pol, std::size_t n_dish) const {
    if (_gain_file.empty())
        return;
    if (file.exist("/digital_gains/gain_coeff_by_dish"))
        return;

    const std::size_t n_out = n_pol * n_dish;
    if (gain_input.size() != n_out) {
        FATAL_ERROR_NON_OO("{:s}: gain_input has {:d} entries, expected {:d} = {:d} pol x "
                           "{:d} dish",
                           _stage_label, gain_input.size(), n_out, n_pol, n_dish);
        return;
    }

    try {
        HighFive::File src(_gain_file, HighFive::File::ReadOnly);
        if (!src.exist("gain_coeff") || !src.exist("gain_exp")) {
            FATAL_ERROR_NON_OO("{:s}: gains archive {:s} has no gain_coeff/gain_exp", _stage_label,
                               _gain_file);
            return;
        }
        auto ds_coeff = src.getDataSet("gain_coeff");
        auto ds_exp = src.getDataSet("gain_exp");
        const auto cd = ds_coeff.getDimensions();
        const auto ed = ds_exp.getDimensions();
        if (cd.size() != 3 || ed.size() != 2 || ed[0] != cd[0] || ed[1] != cd[2]) {
            FATAL_ERROR_NON_OO("{:s}: unexpected gains archive shape; gain_coeff is rank {:d} "
                               "and gain_exp rank {:d}",
                               _stage_label, cd.size(), ed.size());
            return;
        }
        const std::size_t n_upd = cd[0], n_chan = cd[1], n_in = cd[2];

        for (std::size_t s = 0; s < n_out; s++) {
            const int gi = gain_input[s];
            if (gi < 0 || (std::size_t)gi >= n_in) {
                FATAL_ERROR_NON_OO("{:s}: gain input {:d} for (pol {:d}, dish {:d}) is outside "
                                   "the archive's {:d} inputs",
                                   _stage_label, gi, s / n_dish, s % n_dish, n_in);
                return;
            }
        }

        const auto ctype = complex64_type();
        const auto i32 = HighFive::AtomicType<std::int32_t>{};

        std::vector<std::complex<float>> coeff(n_upd * n_chan * n_in);
        ds_coeff.read_raw(coeff.data(), ctype);
        std::vector<std::int32_t> exps(n_upd * n_in);
        ds_exp.read_raw(exps.data(), i32);

        // Gather along the input axis. The channel axis is the hot one, so the gather
        // indices are hoisted out of it.
        std::vector<std::complex<float>> out_coeff(n_upd * n_chan * n_out);
        std::vector<std::int32_t> out_exp(n_upd * n_out);
        for (std::size_t u = 0; u < n_upd; u++) {
            for (std::size_t s = 0; s < n_out; s++)
                out_exp[u * n_out + s] = exps[u * n_in + (std::size_t)gain_input[s]];
            for (std::size_t c = 0; c < n_chan; c++) {
                const std::complex<float>* in_row = &coeff[(u * n_chan + c) * n_in];
                std::complex<float>* out_row = &out_coeff[(u * n_chan + c) * n_out];
                for (std::size_t s = 0; s < n_out; s++)
                    out_row[s] = in_row[(std::size_t)gain_input[s]];
            }
        }

        auto group = file.exist("/digital_gains") ? file.getGroup("/digital_gains")
                                                  : file.createGroup("/digital_gains");

        auto d_coeff = group.createDataSet(
            "gain_coeff_by_dish", HighFive::DataSpace({n_upd, n_chan, n_pol, n_dish}), ctype);
        d_coeff.write_raw(out_coeff.data(), ctype);
        d_coeff.createAttribute("axis",
                                std::vector<std::string>{"update_time", "freq", "pol", "dish"});

        auto d_exp = group.createDataSet("gain_exp_by_dish",
                                         HighFive::DataSpace({n_upd, n_pol, n_dish}), i32);
        d_exp.write_raw(out_exp.data(), i32);
        d_exp.createAttribute("axis", std::vector<std::string>{"update_time", "pol", "dish"});

        const std::vector<std::int32_t> gi32(gain_input.begin(), gain_input.end());
        auto d_map =
            group.createDataSet("gain_input_index", HighFive::DataSpace({n_pol, n_dish}), i32);
        d_map.write_raw(gi32.data(), i32);
        d_map.createAttribute("axis", std::vector<std::string>{"pol", "dish"});

        INFO_NON_OO("{:s}: wrote gains in dish order, [{:d}][{:d}][{:d}][{:d}] "
                    "(update, channel, pol, dish), from {:d} archive inputs",
                    _stage_label, n_upd, n_chan, n_pol, n_dish, n_in);
    } catch (const HighFive::Exception& e) {
        FATAL_ERROR_NON_OO("{:s}: failed writing dish-ordered gains from {:s}: {:s}", _stage_label,
                           _gain_file, e.what());
    }
}

} // namespace kotekan
