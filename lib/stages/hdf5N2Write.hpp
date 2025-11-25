#ifndef KOTEKAN_STAGES_HDF5_N2_WRITE_HPP
#define KOTEKAN_STAGES_HDF5_N2_WRITE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "errors.h"
#include "hdf5Files.hpp"
#include "kotekanLogging.hpp"
#include "prometheusMetrics.hpp"

#include "fmt.hpp"

#include <N2FrameView.hpp>
#include <N2Metadata.hpp>
#include <N2Util.hpp>
#include <cassert>
#include <filesystem>
#include <highfive/H5File.hpp>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <visUtil.hpp>

/**
 * @class N2FileData
 * @brief Buffer a full file worth of arrays in memory, then flush to disk
 * all at once.
 *
 * A file contains `num_file_t` time frames across `num_freq` freqs.
 * This class holds all relevant arrays in their target on-disk layout,
 * "transposed" so the time-axis is fastest-varying, to facilitate large
 * contiguous writes, rather than writing as individual frames arrive.
 *
 * (Note that num_freq and num_file_t are not necessarily the same as the actual
 * frequency or time BLOCKSIZEs when written to file, which is configurable via the
 * stage parameters, but it may be beneficial to have them match.)
 *
 * Additional bookkeeping regarding the file (HDF5 handle, paths, open time) is
 * also stored here. The file start time (and therefore final file name) is determined
 * by the timestamp of the earliest frame in the file, so it cannot be set until
 * all frames have arrived, since frames of different frequencies may arrive out
 * of temporal order.
 */
class N2FileData {
public:
    enum FileMode { CHORD, CHIME };

    // Structural information and fixed sizes
    const size_t num_elements; // number of inputs / elements
    const size_t num_prod;     // number of products
    const size_t num_ev;       // number of eigenvectors/values
    const size_t num_freq;     // number of frequencies
    const size_t num_file_t;   // frames ("time" dimension)

    // file bookkeeping owned by this object
    FileMode file_mode;                      // CHORD or CHIME (or other?)-type file
    const size_t blocksize_f;                // frequency block size for chunking
    const size_t blocksize_p;                // product/element block size for chunking
    const size_t blocksize_t;                // time block size for chunking
    const std::string compression;           // compression type
    const size_t compression_level;          // gzip compression level
    const bool use_bitshuffle;               // whether to use bitshuffle filter
    const double open_wall_s;                // time opened
    double last_update_wall_s;               // last frame receipt
    const uint64_t abs_file_idx;             // absolute file index (abs_time_idx / num_file_t)
    const std::string base_dir;              // base output directory (without /.partial)
    const std::string partial_filepath;      // working on-disk location
    std::unique_ptr<HighFive::File> h5_file; // Working on-disk HDF5 file handle

protected:
    // Datasets to be stored until ready to write
    // f = freq, p = prod, e = eigen, i = input, t = time
    std::vector<N2::cfloat> vis;   // (f, p, t)
    std::vector<float> vis_weight; // (f, p, t)
    std::vector<float> eval;       // (f, e, t)
    std::vector<N2::cfloat> evec;  // (f, e, i, t)
    std::vector<float> erms;       // (f, t)
    std::vector<N2::cfloat> gain;  // (f, i, t)
    std::vector<float> frac_lost;  // (f, t) ; uses n_valid_fpga_ticks
    std::vector<float> frac_rfi;   // (f, t) ; uses n_rfi_fpga_ticks
    std::vector<float> flags;      // (f, i, t)

    // t-dependent metadata
    std::vector<uint64_t> fpga_start_tick;         // (t)
    std::vector<uint64_t> frame_length_fpga_ticks; // (t)
    std::vector<int64_t> time_center_ut1;          // (t)
    std::vector<int64_t> bin_ut1;                  // (t)
    std::vector<double> bin_start_ERA_deg;         // (t)
    std::vector<double> bin_end_ERA_deg;           // (t)
    std::vector<int64_t> bin_start_LAST;           // (t)
    std::vector<int64_t> bin_end_LAST;             // (t)

    // Tracking what (f, t) pairs have been added
    std::vector<uint8_t> added_ft; // size = num_freq * num_file_t
    size_t added_count = 0;        // number of (f, t) frames added

private:
    /// Create (or check existence of) an attribute
    template<typename T>
    void _check_create_attribute(HighFive::File& file, const std::string& name,
                                 const T& value) const;

    /// Create (or check existence of) a dataset
    void _check_create_dataset(HighFive::File& file, const std::string& name,
                               const std::vector<hsize_t>& dims,
                               const std::vector<std::string>& dim_names,
                               const HighFive::DataType& dtype,
                               HighFive::DataSetCreateProps props) const;

    /// Open/create/init datasets in h5 file
    std::unique_ptr<HighFive::File> _open_or_create_file(const std::string& filepath,
                                                         const uint64_t num_file_t_,
                                                         const N2FrameView& fv,
                                                         const FileMode file_mode) const;

public:
    N2FileData(FileMode file_mode_, uint64_t num_file_t_, const N2FrameView& fv,
               const double open_wall_s_, const uint64_t abs_file_idx_, const size_t blocksize_f_,
               const size_t blocksize_p_, const size_t blocksize_t_, const std::string compression_,
               const size_t compression_level_, const bool use_bitshuffle_,
               const std::string base_dir_) :
        num_elements(fv.num_elements), num_prod(fv.num_prod), num_ev(fv.num_ev), num_freq(fv.nfreq),
        num_file_t(num_file_t_), file_mode(file_mode_), blocksize_f(blocksize_f_),
        blocksize_p(blocksize_p_), blocksize_t(blocksize_t_), compression(compression_),
        compression_level(compression_level_), use_bitshuffle(use_bitshuffle_),
        open_wall_s(open_wall_s_), last_update_wall_s(open_wall_s_), abs_file_idx(abs_file_idx_),
        base_dir(std::move(base_dir_)),
        partial_filepath(base_dir + "/.partial/" + "vis_" + std::to_string(abs_file_idx_) + ".h5"),
        h5_file(_open_or_create_file(partial_filepath, num_file_t_, fv, file_mode)) {

        // resize arrays to hold data across (freq, time) blocks
        vis.assign(num_prod * num_freq * num_file_t, N2::cfloat{0.0f, 0.0f});
        vis_weight.assign(num_prod * num_freq * num_file_t, 0.0f);
        eval.assign(num_ev * num_freq * num_file_t, 0.0f);
        evec.assign(num_ev * num_elements * num_freq * num_file_t, N2::cfloat{0.0f, 0.0f});
        erms.assign(num_freq * num_file_t, 0.0f);
        gain.assign(num_elements * num_freq * num_file_t, N2::cfloat{0.0f, 0.0f});
        frac_lost.assign(num_freq * num_file_t, 1.0f); // match empty frames by default
        frac_rfi.assign(num_freq * num_file_t, 0.0f);
        flags.assign(num_elements * num_freq * num_file_t, 0.0f);

        // Additional metadata
        fpga_start_tick.assign(num_file_t, 0);
        frame_length_fpga_ticks.assign(num_file_t, 0);
        time_center_ut1.assign(num_file_t, 0.0);
        bin_ut1.assign(num_file_t, 0);
        bin_start_ERA_deg.assign(num_file_t, 0.0);
        bin_end_ERA_deg.assign(num_file_t, 0.0);
        bin_start_LAST.assign(num_file_t, 0);
        bin_end_LAST.assign(num_file_t, 0);

        added_ft.assign(num_freq * num_file_t, 0);
    }

    /**
     * @brief Add a frame of data at the computed time index.
     * @param fv      Frame view containing data.
     * @param meta    N2 metadata for the frame.
     * @param t_index Time index within this file block (0..num_file_t-1).
     *
     * @returns success status (false if frame could not be added).
     */
    bool add_frame(const N2FrameView& fv, size_t t_index);

    /// Get the final filename for this file based on earliest frame time
    /// May return std::nullopt if no frames have been added yet,
    /// so earliest time is unknown. If earlier frames are later added,
    /// this method may return a different filename.
    std::optional<std::string> _get_final_filename();

    /// Flush buffered data to the associated dataset, always writing the
    /// entire time range [0 .. num_file_t-1] regardless of which frames were
    /// populated. Returns true if a write occurred.
    bool flush_to_disk();

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
        return num_file_t * (num_elements * (num_ev * f + e) + i) + t; // (f, e, i, t)
    }
    /// Get the index for a (f, i, t) triplet
    inline size_t idx_fit(size_t f, size_t i, size_t t) const {
        return num_file_t * (num_elements * f + i) + t; // (f, i, t)
    }
    /// Get the index for a (f, t) pair
    inline size_t idx_ft(size_t f, size_t t) const {
        return num_file_t * f + t; // (f, t)
    }
};

/**
 * @class hdf5N2Write
 * @brief Buffered-transpose writer: buffers sequential time frames and writes HDF5 files.
 *
 * This stage groups frames into UTC-midnight-aligned windows of length
 * `file_seconds`, buffers a complete (num_freq * file_nt) block per output file in
 * memory (via N2FileData), and when the block is full, writes all arrays to
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
 * @conf num_file_t         UInt. Number of time frames per file
 * @conf late_frame_grace_seconds  UInt. Grace period in seconds for late frames (default: 60)
 * @conf max_frames                Int.  Stop writing after this many frames (-1 = unlimited)
 *
 * @par Metrics
 * @metric kotekan_N2write_write_time_seconds  Duration to write the last flush
 * @metric kotekan_N2write_n_datasets          Number of datasets currently open
 *
 * @note User-level documentation lives in docs/sphinx/user/processes/hdf5N2Write.rst.
 **/
class hdf5N2Write : public kotekan::Stage {

public:
    hdf5N2Write(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~hdf5N2Write();

    void main_thread() override;

private:
    // Config settings (initialized from Config in constructor)
    const std::string base_dir;     /// Base directory to write files into
    const std::uint64_t num_file_t; /// Number of incoming time frames per file, as indexed by the
                                    /// absolute frame index
    const std::string compression;
    const std::uint64_t compression_level;
    const bool use_bitshuffle;
    const std::uint64_t blocksize_f;
    const std::uint64_t blocksize_p;
    const std::uint64_t blocksize_t;
    const std::uint64_t
        late_frame_grace_seconds; /// Grace period in seconds for late frames (default: 60)
    const int max_frames;         /// Stop writing after this many frames (-1 = unlimited)

    Buffer* const buffer;

    kotekan::prometheus::Gauge& write_time_metric;
    kotekan::prometheus::Gauge& n_datasets_metric;

    /**
     * @brief Get an absolute file number (index) for given metadata.
     * @param meta  N2Metadata for the frame
     * @return      Absolute file index
     * This method computes which file a given frame belongs to,
     * based on its absolute frame index and the configured
     * number of time frames per file.
     */
    std::uint64_t _get_abs_file_idx(const N2FrameView& fv) const;

    /**
     * @brief Finalize a file: close and rename from .partial to final name.
     * @param filedata  File to finalize
     */
    void _finalize_file(N2FileData& filedata);

    /**
     * @brief Finalize N2FileData objects that have been inactive for too long.
     * @param files  Map of abs_idx-N2FileData objects to finalize
     * @param exclude_abs_file_idx  Optional file index to skip while finalizing
     */
    void _grace_finalize_files(std::map<size_t, std::unique_ptr<N2FileData>>& files,
                               const size_t* exclude_abs_file_idx = nullptr);

    bool _finalfile_exists(std::uint64_t abs_file_idx, const std::string& search_dir) const;
};

#endif // KOTEKAN_STAGES_HDF5_N2_WRITE_HPP
