// NOTE: This header provides the visFileData buffering helper and the
//       declaration for the hdf5VisWrite stage. The stage methods are
//       implemented and registered in hdf5VisWrite.cpp.
#ifndef KOTEKAN_STAGES_HDF5_VIS_WRITE_HPP
#define KOTEKAN_STAGES_HDF5_VIS_WRITE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "errors.h"
#include "hdf5Files.hpp"
#include "kotekanLogging.hpp"

#include "fmt.hpp"

#include <N2FrameView.hpp>
#include <N2Metadata.hpp>
#include <N2Util.hpp>
#include <cassert>
#include <highfive/H5File.hpp>
#include <memory>
#include <string>
#include <vector>
#include <visUtil.hpp>

/**
 * @class visFileData
 * @brief Buffer a full file worth of arrays in memory, then flush to disk
 * all at once.
 *
 * A file contains `num_file_t` time frames across `num_freq` freqs.
 * This class holds all relevant arrays in the target on-disk layout,
 * to facilitate large contiguous writes, rather than writing as individual frames
 * arrive.
 *
 * (Note that num_freq and num_file_t are not necessarily the same as the actual
 * frequency or time BLOCKSIZEs when written to file, which is configurable via the
 * stage parameters, but it may be beneficial to have them match.)
 */
class visFileData {
public:
    // Structural information and fixed sizes
    const size_t num_input;  // number of inputs / elements
    const size_t num_prod;   // number of products
    const size_t num_ev;     // number of eigenvectors/values
    const size_t num_freq;   // number of frequencies
    const size_t num_file_t; // frames ("time" dimension)

    // Per-file bookkeeping owned by this object
    std::unique_ptr<HighFive::File> h5_file; // owning HDF5 file handle
    const std::string partial_path;          // working on-disk location
    const double open_wall_s;                // time opened
    double last_update_wall_s = 0.0;         // last frame receipt
    const uint64_t file_start_time_ns;       // aligned file start time
    const uint64_t frame_len_ns;             // frame length in ns

protected:
    // Datasets to be stored until ready to write
    // f = freq, p = prod, e = eigen, i = input, t = time
    std::vector<N2::cfloat> vis;   // (f, p, t)
    std::vector<float> vis_weight; // (f, p, t)
    std::vector<float> eval;       // (f, e, t)
    std::vector<N2::cfloat> evec;  // (f, e, i, t)
    std::vector<float> erms;       // (f, t)
    std::vector<N2::cfloat> gain;  // (f, i, t)
    std::vector<float> frac_lost;  // (f, t)
    std::vector<float> frac_rfi;   // (f, t)
    std::vector<float> flags;      // (f, i, t)

    /// Earth rotation angle at corresponding times (t)
    std::vector<double> era_deg; // (t)
    // Track initialization of ERA values per time index to avoid using a value sentinel
    std::vector<uint8_t> era_deg_set; // (t), 0 = unset, 1 = set

    // Additional metadata
    std::vector<uint64_t> fpga_start_tick;     // (t)
    std::vector<uint64_t> frame_start_time_ns; // (t)
    std::vector<uint64_t> n_valid_fpga_ticks;  // (f, t)
    std::vector<uint64_t> n_rfi_fpga_ticks;    // (f, t)
    // Should be constant across all frames (set on first add_frame)
    uint64_t frame_length_fpga_ticks;

    // Tracking what (f, t) pairs have been added
    std::vector<uint8_t> added_ft; // size = num_freq * num_file_t
    size_t added_count = 0;        // number of (f, t) frames added

public:
    visFileData(const uint64_t num_file_t_, const uint64_t num_freq_, const uint64_t num_input_,
                const uint64_t num_prod_, const uint64_t num_ev_, const uint64_t frame_len_ns_,
                const uint64_t frame_length_fpga_ticks_, const uint64_t file_start_time_ns_,
                std::string partial_path_, const double open_wall_s_) :
        num_input(num_input_), num_prod(num_prod_), num_ev(num_ev_), num_freq(num_freq_),
        num_file_t(num_file_t_), partial_path(std::move(partial_path_)), open_wall_s(open_wall_s_),
        last_update_wall_s(open_wall_s_), file_start_time_ns(file_start_time_ns_),
        frame_len_ns(frame_len_ns_), frame_length_fpga_ticks(frame_length_fpga_ticks_) {

        // resize arrays to hold data across (freq, time) blocks
        vis.assign(num_prod * num_freq * num_file_t, N2::cfloat{0.0f, 0.0f});
        vis_weight.assign(num_prod * num_freq * num_file_t, 0.0f);
        eval.assign(num_ev * num_freq * num_file_t, 0.0f);
        evec.assign(num_ev * num_input * num_freq * num_file_t, N2::cfloat{0.0f, 0.0f});
        erms.assign(num_freq * num_file_t, 0.0f);
        gain.assign(num_input * num_freq * num_file_t, N2::cfloat{0.0f, 0.0f});
        flags.assign(num_input * num_freq * num_file_t, 0.0f);
        frac_lost.assign(num_freq * num_file_t, 1.0f); // match empty frames by default
        frac_rfi.assign(num_freq * num_file_t, 0.0f);

        // Additional metadata
        fpga_start_tick.assign(num_file_t, 0);
        frame_start_time_ns.assign(num_file_t, 0);
        era_deg.assign(num_file_t, 0.0);
        era_deg_set.assign(num_file_t, 0);
        n_valid_fpga_ticks.assign(num_freq * num_file_t, 0);
        n_rfi_fpga_ticks.assign(num_freq * num_file_t, 0);

        added_ft.assign(num_freq * num_file_t, 0);
    }

    /// Flush buffered data to the associated dataset, always writing the
    /// entire time range [0 .. num_file_t-1] regardless of which frames were
    /// populated. Returns true if a write occurred.
    bool flush();

    /// Close the associated dataset handle if open.
    void close();

    /// Check if all (f, t) pairs have been added
    bool full() const {
        return added_count == num_freq * num_file_t;
    }

    // Accessors for internal storage (index calculation)
    /// Get the index for a (f, p, t) triplet
    inline size_t idx_fpt(size_t f, size_t p, size_t t) const {
        return num_file_t * (num_prod * f + p) + t; // (f, p, t)
    }
    /// Get the index for a (f, e, t) triplet
    inline size_t idx_fet(size_t f, size_t e, size_t t) const {
        return num_file_t * (num_ev * f + e) + t; // (f, e, t)
    }
    /// Get the index for a (f, e, i, t) quadruplet
    inline size_t idx_feit(size_t f, size_t e, size_t i, size_t t) const {
        return num_file_t * (num_input * (num_ev * f + e) + i) + t; // (f, e, i, t)
    }
    /// Get the index for a (f, i, t) triplet
    inline size_t idx_fit(size_t f, size_t i, size_t t) const {
        return num_file_t * (num_input * f + i) + t; // (f, i, t)
    }
    /// Get the index for a (f, t) pair
    inline size_t idx_ft(size_t f, size_t t) const {
        return num_file_t * f + t; // (f, t)
    }

    /**
     * @brief Add a frame of data at the computed time index.
     * @param fv      Frame view containing data.
     * @param meta    N2 metadata for the frame.
     * @param t_index Time index within this file block (0..num_file_t-1).
     */
    void add_frame(const N2FrameView& fv, const std::shared_ptr<N2Metadata>& meta, size_t t_index) {
        const size_t f_index = meta->freq_id; // TODO: sync with telescope object. For now assume
                                              // 0..num_freq-1 indexing

        // TODO: What behavior do we want if these checks don't pass in release builds?

        assert(f_index < num_freq);
        assert(t_index < num_file_t);

        // Make sure frame hasn't been added yet
        size_t check_idx = idx_ft(f_index, t_index);
        if (added_ft[check_idx] != 0) {
            auto msg = fmt::format("visFileData: duplicate frame insertion at (f={}, t={})",
                                   f_index, t_index);
            kotekan::kotekanLogging::internal_logging(LOG_ERR, "", msg);
            kotekan::kotekanLogging::set_error_message(msg);
            exit_kotekan(ReturnCode::FATAL_ERROR);
        }

        // Check structural and metadata properties of incoming frame
        assert(meta->frame_length_fpga_ticks > 0);
        assert(fv.vis.size() == N2::get_num_prod(meta->num_elements));
        assert(fv.weight.size() == N2::get_num_prod(meta->num_elements));
        assert(fv.eval.size() == meta->num_ev);
        assert(fv.evec.size() == meta->num_ev * meta->num_elements);
        assert(fv.gain.size() == meta->num_elements);
        assert(fv.flags.size() == meta->num_elements);
        assert(meta->num_elements == num_input);
        assert(meta->num_prod == num_prod);
        assert(meta->num_ev == num_ev);

        // Check per-time metadata consistency and assignment
        if (fpga_start_tick[t_index] != 0)
            assert(fpga_start_tick[t_index] == meta->fpga_start_tick);
        fpga_start_tick[t_index] = meta->fpga_start_tick;
        if (frame_start_time_ns[t_index] != 0)
            assert(frame_start_time_ns[t_index] == meta->frame_start_time_ns);
        frame_start_time_ns[t_index] = meta->frame_start_time_ns;
        if (frame_length_fpga_ticks != 0)
            assert(frame_length_fpga_ticks == meta->frame_length_fpga_ticks);
        // Initialize constant per-file frame length from first frame
        if (frame_length_fpga_ticks == 0)
            frame_length_fpga_ticks = meta->frame_length_fpga_ticks;

        // Initialize ERA value once per time slot; ignore subsequent changes to avoid
        // relying on a numeric sentinel (e.g., ERA==0) and to ensure consistency.
        if (!era_deg_set[t_index]) {
            era_deg[t_index] = fv.eop.ERA_deg;
            era_deg_set[t_index] = 1;
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

        // Derived fractions and counts (sanitize tick counts to avoid out-of-range values)
        const uint64_t frame_len_ticks = meta->frame_length_fpga_ticks;
        uint64_t n_valid = std::min(meta->n_valid_fpga_ticks, frame_len_ticks);
        uint64_t n_rfi = std::min(meta->n_rfi_fpga_ticks, frame_len_ticks);
        // Ensure counts are non-negative and do not exceed the frame length in total
        if (n_valid > frame_len_ticks)
            n_valid = frame_len_ticks;
        if (n_rfi > frame_len_ticks)
            n_rfi = frame_len_ticks;
        if (n_valid + n_rfi > frame_len_ticks)
            n_rfi = frame_len_ticks - n_valid;
        n_valid_fpga_ticks[idx_ft(f_index, t_index)] = n_valid;
        n_rfi_fpga_ticks[idx_ft(f_index, t_index)] = n_rfi;
        frac_lost[idx_ft(f_index, t_index)] =
            (frame_len_ticks > 0) ? (1.0f - float(n_valid) / float(frame_len_ticks)) : 0.0f;
        frac_rfi[idx_ft(f_index, t_index)] =
            (frame_len_ticks > 0) ? (float(n_rfi) / float(frame_len_ticks)) : 0.0f;

        size_t si = idx_ft(f_index, t_index);
        if (!added_ft[si]) {
            added_ft[si] = 1;
            ++added_count; // number of frames added
        }
    }

    // Some getters for testing and verification
    N2::cfloat get_vis(size_t f, size_t p, size_t t) const {
        return vis[idx_fpt(f, p, t)];
    }
    float get_weight(size_t f, size_t p, size_t t) const {
        return vis_weight[idx_fpt(f, p, t)];
    }
    float get_eval(size_t f, size_t e, size_t t) const {
        return eval[idx_fet(f, e, t)];
    }
    N2::cfloat get_evec(size_t f, size_t e, size_t i, size_t t) const {
        return evec[idx_feit(f, e, i, t)];
    }
    float get_erms(size_t f, size_t t) const {
        return erms[idx_ft(f, t)];
    }
    N2::cfloat get_gain(size_t f, size_t i, size_t t) const {
        return gain[idx_fit(f, i, t)];
    }
    float get_flags(size_t f, size_t i, size_t t) const {
        return flags[idx_fit(f, i, t)];
    }
    float get_frac_lost(size_t f, size_t t) const {
        return frac_lost[idx_ft(f, t)];
    }
    float get_frac_rfi(size_t f, size_t t) const {
        return frac_rfi[idx_ft(f, t)];
    }
    uint64_t get_n_valid(size_t f, size_t t) const {
        return n_valid_fpga_ticks[idx_ft(f, t)];
    }
    uint64_t get_n_rfi(size_t f, size_t t) const {
        return n_rfi_fpga_ticks[idx_ft(f, t)];
    }
    uint64_t get_fpga_start_tick(size_t t) const {
        return fpga_start_tick[t];
    }
    uint64_t get_frame_start_time_ns(size_t t) const {
        return frame_start_time_ns[t];
    }
    uint64_t get_frame_length_fpga_ticks(size_t) const {
        return frame_length_fpga_ticks;
    }
    double get_era_deg(size_t t) const {
        return era_deg[t];
    }
    size_t get_added_count() const {
        return added_count;
    }
    uint8_t get_added(size_t f, size_t t) const {
        return added_ft[idx_ft(f, t)];
    }
};

/**
 * @class hdf5VisWrite
 * @brief Buffered-transpose writer: buffers sequential time frames and writes HDF5 files.
 *
 * This stage groups frames into UTC-midnight-aligned windows of length
 * `file_seconds`, buffers a complete (num_freq * file_nt) block per output file in
 * memory (via visFileData), and when the block is full, writes all arrays to
 * disk in large contiguous slabs. If the block is not full, it is nevertheless finalized
 * after `late_frame_grace_seconds` of inactivity once frames for later file windows begin to
 * arrive. Flushes always write the full time range for the file; any missing frames
 * remain default/zero-valued in the output arrays.
 *
 * @par Buffers
 * @buffer in_buf  Input visibility buffer
 *     @buffer_format VisBuffer
 *     @buffer_metadata N2Metadata
 *
 * @par Config
 * @conf base_dir           String. Directory to write into
 * @conf file_name          String. Base filename to write
 * @conf prefix_hostname    Bool. Prefix files with hostname (default: true)
 * @conf blocksize_f        UInt. Array chunk size along freq (0 = driver default)
 * @conf blocksize_p        UInt. Array chunk size along product (unused currently; 0 = default)
 * @conf blocksize_t        UInt. Array chunk size along time (default: 1)
 *
 * @conf file_seconds              UInt. File length in seconds; must divide 86400
 * @conf late_frame_grace_seconds  UInt. Grace period in seconds for late frames (default: 60)
 * @conf max_frames                Int.  Stop writing after this many frames (-1 = unlimited)
 *
 * @par Metrics
 * @metric kotekan_viswrite_write_time_seconds  Duration to write the last flush
 * @metric kotekan_viswrite_n_datasets          Number of datasets currently open
 *
 * @note User-level documentation lives in docs/sphinx/user/processes/hdf5VisWrite.rst.
 **/
class hdf5VisWrite : public kotekan::Stage {

public:
    hdf5VisWrite(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    virtual ~hdf5VisWrite();

    void main_thread() override;

private:
    // Config (initialized from Config in constructor)
    const std::string base_dir;  /// Base directory to write files into
    const std::string file_name; /// Base filename to write
    const bool prefix_hostname;  /// Prefix files with hostname (default: true)

    // writer options + compression
    const std::string compression; /// "none" | "deflate" | "zstd" (bitshuffle+zstd if enabled)
    const std::uint64_t compression_level; /// compression level (0 = driver default/none)
    const bool use_bitshuffle;             /// use bitshuffle filter if available
    const std::uint64_t blocksize_f;       /// Array chunk size along frequency
    const std::uint64_t
        blocksize_p; /// Array chunk size along product (0 = default = full num products)
    const std::uint64_t blocksize_t; /// Array chunk size along time (default: 1)

    const std::uint64_t file_seconds; /// File length in seconds; must divide 86400
    const std::uint64_t
        late_frame_grace_seconds; /// Grace period in seconds for late frames (default: 60)

    const int max_frames; /// Stop writing after this many frames (-1 = unlimited)

    Buffer* const buffer;

    // Allow override telescope seq length for testing
    const std::uint64_t tick_len_ns_override;
    inline std::uint64_t _get_tick_len_ns() const {
        if (tick_len_ns_override > 0)
            return tick_len_ns_override;
        return Telescope::instance().seq_length_nsec();
    }

    /**
     * @brief Compute the aligned file start time for the given metadata.
     * @param meta  N2Metadata for the file
     * @return      Aligned file start time in nanoseconds since epoch
     * This method computes the UTC-midnight-aligned start time for the
     * output file based on the configured file size. File times are
     * aligned to multiples of `file_seconds` since UTC midnight.
     */
    std::uint64_t _get_file_start_time_ns(const std::shared_ptr<const N2Metadata> meta);

    /**
     * @brief Get the HDF5 filename for the given metadata.
     * @param meta  N2Metadata for the file
     * @return      Full path to the HDF5 file to write
     * This method computes the output filename based on the configured
     * base directory, file name, and whether to prefix with hostname.
     */
    std::string _get_vis_filename(std::shared_ptr<const N2Metadata> meta);

    // Internal: HDF5 initialization handled in implementation file.

    /**
     * @brief Finalize datasets that have been inactive for too long.
     * @param datasets  Map of datasets to finalize
     * @param late_frame_grace_seconds  Grace period in seconds for late frames
     */
    void _grace_finalize_datasets(std::map<std::string, std::unique_ptr<visFileData>>& datasets,
                                  const std::string* exclude_path = nullptr);

    // Initialize HDF5 datasets with chunking/compression options
    void _initialize_h5(HighFive::File& file, const std::shared_ptr<const N2Metadata> meta,
                        std::uint64_t file_nt) const;

    /**
     * @brief Get the partial directory path for temporary files.
     * @return  Path to the partial directory within the base directory
     * This method constructs the path to the ".partial" subdirectory
     * within the configured base directory.
     */
    std::string _get_partial_dir() const {
        return (base_dir + "/.partial");
    }

    /**
     * @brief Extract the basename from a given path.
     * @param path  Full file path
     * @return      Basename of the file
     */
    static std::string _basename(const std::string& path) {
        auto pos = path.find_last_of('/');
        if (pos == std::string::npos)
            return path;
        return path.substr(pos + 1);
    }
};

#endif // KOTEKAN_STAGES_HDF5_VIS_WRITE_HPP
