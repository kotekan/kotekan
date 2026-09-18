/*****************************************
@file
@brief Shared handling of the FPGA digital gains archive for the HDF5 writers.
- BasebandGains
*****************************************/

#ifndef BASEBAND_GAINS_HPP
#define BASEBAND_GAINS_HPP

#include "Config.hpp"

#include <cstdint>
#include <highfive/H5File.hpp>
#include <optional>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @brief Locates the FPGA digital gains archive and copies it into output files.
 *
 * The gains archive is a `pychfpga` `DigitalGainArchive` HDF5 file covering the full band
 * and every input. Writers copy it into their output *verbatim* (an `H5Ocopy` of its root
 * group into `/digital_gains`) rather than slicing it, so a file that holds a subset of the
 * frequencies or dishes still carries the whole archive, and the archive's own
 * `index_map/freq` says which channel each gain belongs to.
 *
 * The archive comes either from a path on disk (`baseband_gain_file`) or over HTTP from an
 * FPGA controller (`baseband_gain_host_info`), never both. In the HTTP case the writer calls
 * @c resolve_local_copy once, which fetches the archive into the acquisition's `.partial`
 * directory; every output file is then copied from that cached copy.
 *
 * @conf baseband_gain_file       String. Default "". Path to the gains archive. Mutually
 *                                exclusive with `baseband_gain_host_info`.
 * @conf baseband_gain_host_info  String. Default "". Absolute config path (e.g.
 *                                `/fpga_controller`) of a block giving `host`, `port` and an
 *                                optional `gains_endpoint` (default
 *                                `/get-current-gain-file`) to fetch the archive from.
 * @conf baseband_gain_update_idx Int. Default -1. Index along the archive's `update_time`
 *                                axis to record as current; -1 selects the latest.
 */
class BasebandGains {
public:
    /**
     * @brief Read the gains configuration for a stage.
     *
     * @param config       The kotekan config.
     * @param unique_name  The stage's unique name, for the per-stage config lookups.
     * @param stage_label  Stage name used in log messages, e.g. "hdf5FileWrite".
     */
    BasebandGains(kotekan::Config& config, const std::string& unique_name,
                  const std::string& stage_label);

    /// True if either form of gains configuration was given.
    bool configured() const {
        return !_gain_file.empty() || !_host.empty();
    }

    /// True if the archive is on disk and ready to copy.
    bool available() const;

    /**
     * @brief Make sure a local copy of the archive exists, fetching it if configured over HTTP.
     *
     * Call once, before the first output file is written. A no-op unless
     * `baseband_gain_host_info` was set. The cached copy is named after @p cache_stem so that
     * two writers sharing a `base_dir` (e.g. one per NUMA domain) do not race on it.
     *
     * @param partial_dir  Directory to cache the fetched archive in; must already exist.
     * @param cache_stem   Stem for the cache file name, unique to the calling stage.
     */
    void resolve_local_copy(const std::string& partial_dir, const std::string& cache_stem);

    /**
     * @brief Copy the archive into @p file as the `/digital_gains` group.
     *
     * Also writes the `digital_gains_source_file` attribute, the group's
     * `selected_update_idx`, and `digital_gains_hash` (a stable digest of the archive bytes,
     * so consumers merging several files can check they all hold the same gains). A no-op if
     * gains are not configured or the group already exists. FATAL on a configured but
     * missing or uncopyable archive.
     */
    void copy_into(HighFive::File& file) const;

    /**
     * @brief Write the archive's gains reordered onto the file's own [P][D] dish axes.
     *
     * The archive indexes its inputs by CRS hardware position (`board * 8 + channel`),
     * which is neither the frame's element order nor the telescope's dish order, and it
     * spans the whole array rather than the dishes a subset file holds. @p gain_input
     * supplies, for each (polarization, dish) slot of the output in row-major order, the
     * archive input that slot's gains live at; the datasets written are therefore in the
     * file's own dish order and reduced to the dishes the file actually holds:
     *
     * - `/digital_gains/gain_coeff_by_dish`  [update][channel][P][D]  complex64
     * - `/digital_gains/gain_exp_by_dish`    [update][P][D]           int32
     * - `/digital_gains/gain_input_index`    [P][D]                   int32, the mapping
     *
     * The channel axis is left as the archive's full FPGA band, indexed by coarse channel
     * number. The verbatim archive from @c copy_into stays in place as the provenance
     * record. A no-op if gains are not configured or the datasets already exist; FATAL on a
     * malformed archive or an out-of-range index.
     *
     * @param file        The output file, whose `/digital_gains` group is added to.
     * @param gain_input  Archive input index per (pol, dish) slot, size `n_pol * n_dish`.
     * @param n_pol       Size of the output polarization axis.
     * @param n_dish      Size of the output dish axis.
     */
    void write_by_dish(HighFive::File& file, const std::vector<int>& gain_input, std::size_t n_pol,
                       std::size_t n_dish) const;

    /// Path of the archive currently in use, empty when not configured.
    const std::string& file_path() const {
        return _gain_file;
    }

    /// Stable digest of the archive's bytes, or nullopt if it could not be read.
    std::optional<std::uint64_t> content_hash() const;

private:
    std::string _stage_label;
    std::string _gain_file;
    std::string _host;
    std::uint16_t _port = 0;
    std::string _endpoint;
    int _update_idx;

    /// Lazily computed, then reused: the digest is wanted both for the file-level attribute
    /// and by writers attaching it to their dataset.
    mutable std::optional<std::uint64_t> _hash_cache;
    mutable bool _hash_computed = false;
};

} // namespace kotekan

#endif // BASEBAND_GAINS_HPP
