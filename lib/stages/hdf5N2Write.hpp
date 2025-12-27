#ifndef KOTEKAN_STAGES_HDF5_N2_WRITE_HPP
#define KOTEKAN_STAGES_HDF5_N2_WRITE_HPP

#include "Config.hpp"
#include "N2Metadata.hpp"
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

/**
 * @class N2FileData
 * @brief Buffer a full file worth of arrays in memory, then flush to disk
 * all at once.
 *
 * A file contains `num_file_t` time frames across `num_file_f` freqs. This class
 * holds all relevant arrays in their target on-disk layout, "transposed" so the
 * time-axis is fastest-varying, to facilitate large contiguous writes, rather
 * than writing as individual frames arrive. Additional bookkeeping regarding
 * the file (HDF5 handle, paths, open time) is also stored here. The file start
 * time (and therefore final file name) is determined by the timestamp of the
 * earliest frame in the file, so it cannot be set until all frames have arrived,
 * since frames of different frequencies may arrive out of temporal order.
 *
 * @par File layout
 * - Attributes: version, file_mode (CHORD/CHIME), abs_file_idx,
 *   num_file_t, num_elements, num_prod, num_ev, num_freq, n2_layout, telescope
 *   geometry (origin, orientations, dish maps), EOP tables, num_file_f.
 * - Index maps: /index_map/freq (MHz + width per file frequency), /index_map/prod,
 *   /index_map/grid_x_idx, /index_map/grid_y_idx, /index_map/feed_pos_disp_m,
 *   /index_map/coelev_disp_deg, /index_map/type, /index_map/dish_positions_in_grid_coords.
 * - Per-(f, p, t)/(f, t) datasets: /vis, /eval, /evec, /erms, /gain, /frames_added;
 *   vis_weight, flags, frac_lost, and frac_rfi live at the root (CHORD) or under
 *   /flags/{vis_weight, flags, frac_lost, frac_rfi} when file_mode == CHIME.
 * - Per-time metadata: /fpga_start_tick, /frame_length_fpga_ticks,
 *   /time_center_ut1_ns, /bin_ut1_ns, /bin_start_ERA_deg, /bin_end_ERA_deg,
 *   /bin_start_LAST, /bin_end_LAST.
 * - /config_json grows on flush with snapshots from configTracker.
 *
 * @par Chunking and compression
 * - Chunk caps: blocksize_f (frequency), blocksize_p (product/element),
 *   blocksize_t (time). A value of 0 leaves the dimension size unchanged.
 * - Compression: compression="none" (default) or "deflate" (zlib; default level 4
 *   when compression_level=0).
 * - use_bitshuffle=true adds the bitshuffle filter and uses compression as the
 *   backend ("none", "zstd", "lz4"); compression_level=0 maps to level 9.
 *   Required HDF5 plugins must be available at runtime.
 */
class N2FileData {
public:
    enum FileMode { CHORD, CHIME };

    // Structural information and fixed sizes
    const size_t num_elements; // number of inputs / elements
    const size_t num_prod;     // number of products
    const size_t num_ev;       // number of eigenvectors/values
    const size_t num_file_f;   // number of frequencies
    const size_t num_file_t;   // frames ("time" dimension)

    // file bookkeeping owned by this object
    const FileMode file_mode;           // CHORD or CHIME (or other?)-type file
    const size_t blocksize_f;           // frequency block size for chunking
    const size_t blocksize_p;           // product/element block size for chunking
    const size_t blocksize_t;           // time block size for chunking
    const std::string compression;      // compression type
    const size_t compression_level;     // gzip compression level
    const bool use_bitshuffle;          // whether to use bitshuffle filter
    const double open_wall_s;           // time opened
    const uint64_t abs_file_idx;        // absolute file index (abs_time_idx / num_file_t)
    const std::string base_dir;         // base output directory (without /.partial)
    const std::string partial_filepath; // working on-disk location
    const N2Layout n2_layout;           // visibility (N2) layout

    double last_update_wall_s;               // last frame receipt
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
    std::vector<double> bin_start_LAST;            // (t)
    std::vector<double> bin_end_LAST;              // (t)

    // Tracking what (f, t) pairs have been added
    std::vector<uint8_t> added_ft; // size = num_file_f * num_file_t
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
    enum class AddFrameStatus { Success, OutOfBounds, Duplicate, MetadataMismatch };

    N2FileData(FileMode file_mode_, uint64_t num_file_t_, const N2FrameView& fv,
               const double open_wall_s_, const uint64_t abs_file_idx_, const size_t blocksize_f_,
               const size_t blocksize_p_, const size_t blocksize_t_, const std::string compression_,
               const size_t compression_level_, const bool use_bitshuffle_,
               const std::string base_dir_);

    /**
     * @brief Add a frame of data at the computed time index.
     * @param fv      Frame view containing data.
     * @param meta    N2 metadata for the frame.
     * @param t_index Time index within this file block (0..num_file_t-1).
     *
     * @returns status describing whether the frame was accepted.
     */
    AddFrameStatus add_frame(const N2FrameView& fv, size_t t_index);

    /// Get the final filename for this file based on earliest fpga tick in the frame.
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
        return added_count == num_file_f * num_file_t;
    }

    double completion_fraction() const {
        const size_t expected = num_file_f * num_file_t;
        return expected > 0 ? static_cast<double>(added_count) / static_cast<double>(expected)
                            : 0.0;
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

    /// Get the number of added frames (ie. (f, t) pairs)
    inline size_t get_added_count() const {
        return added_count;
    }

    /// Get the number of added frames (ie. (f, t) pairs)
    inline size_t get_expected_count() const {
        return num_file_t * num_file_f;
    }
};

/**
 * @class hdf5N2Write
 * @brief Buffered-transpose writer: buffers sequential time frames and writes HDF5 files.
 *
 * This stage groups frames into windows of fixed-number-of-frames `num_file_t`
 * based on their absolute time index, buffers a complete (num_file_f *
 * num_file_t) block per output file in memory (via N2FileData), and writes all
 * arrays to disk in large contiguous slabs. Missing frames stay zero-filled in
 * the output; /frames_added records which (f, t) pairs were present and
 * frac_lost remains 1.0 where frames were absent. Files are first written to
 * `<base_dir>/.partial/vis_<abs_idx>.h5` and renamed to
 * `vis_<abs_idx>_<YYYYMMDDThhmmss>_<nsec>.h5` based on the earliest
 * fpga_start_tick in the file; if that finalized name already exists when a new
 * window is opened, late frames are dropped. Partial files are finalized when
 * full, after `late_frame_grace_seconds` of inactivity once later windows
 * arrive, or during shutdown.
 *
 * @par Buffers
 * @buffer in_buf  Input visibility buffer
 *     @buffer_format VisBuffer
 *     @buffer_metadata N2Metadata
 *
 * @par Configuration
 * @conf in_buf                   String. N2 buffer supplying frames (`buffer_type` must be "N2").
 * @conf base_dir                 String. Output directory (absolute or relative to the process
 *                                working directory where kotekan was invoked); `<base_dir>` and
 *                                `<base_dir>/.partial` are created.
 * @conf num_file_t               UInt. Number of time frames per file (`t_index = abs_time_idx %
 *num_file_t`).
 * @conf blocksize_f              UInt. Chunk cap for the frequency dimension (default: 16).
 * @conf blocksize_p              UInt. Chunk cap for product/element dimensions (default: 16).
 * @conf blocksize_t              UInt. Chunk cap for the time dimension (default: num_file_t).
 * @conf compression              String. "none" (default) or "deflate" for zlib; also "zstd" or
 *                                "lz4" when paired with bitshuffle.
 * @conf compression_level        UInt. Codec level (0 picks 4 for deflate, 9 for bitshuffle).
 * @conf use_bitshuffle           Bool. Enable bitshuffle with the selected backend codec (default:
 *                                false).
 * @conf late_frame_grace_seconds UInt. Grace period in seconds for late frames (default: 60).
 * @conf max_frames               Int. Stop writing after this many frames (-1 = unlimited).
 *
 * @par Metrics
 * @metric kotekan_hdf5N2Write_write_time_seconds        Duration to write the last flush
 * @metric kotekan_hdf5N2Write_n_datasets                Number of datasets currently open
 * @metric kotekan_hdf5N2Write_open_file_info            Gauge=1 for each open file {abs_file_idx,
 *                                partial_path, file_mode}
 * @metric kotekan_hdf5N2Write_open_file_age_seconds     Wall time since file open {abs_file_idx}
 * @metric kotekan_hdf5N2Write_file_completion_fraction  Added (f,t) pairs / expected per file
 *                                {abs_file_idx}
 * @metric kotekan_hdf5N2Write_add_frame_errors_total    Counter of add_frame failures {reason}
 * @metric kotekan_hdf5N2Write_last_add_frame_error_seconds Timestamp of last add_frame failure
 *                                {reason, abs_file_idx, freq_id, t_index}
 * @metric kotekan_hdf5N2Write_finalize_failures_total   Counter of finalize failures {reason}
 * @metric kotekan_hdf5N2Write_unfinalized_file          Gauge=1 for files left partial/quarantined
 *                                {abs_file_idx, partial_path}
 *
 * @par Example
 * @code{.yaml}
 * hdf5_vis_write:
 *     kotekan_stage: hdf5N2Write
 *     in_buf: n2_merge_buffer      # Input N2-type buffer
 *     base_dir: ./vis_data         # Output directory root
 *     num_file_t: 10               # Frames per file window
 *     blocksize_f: 4               # Chunk cap (frequency)
 *     blocksize_p: 8               # Chunk cap (products/elements)
 *     blocksize_t: 5               # Chunk cap (time)
 *     compression: deflate         # none|deflate|zstd|lz4 (with bitshuffle)
 *     compression_level: 4         # Codec level (0=stage default)
 *     use_bitshuffle: true         # Enable bitshuffle filter
 *     late_frame_grace_seconds: 30 # Grace before finalizing partials
 *     max_frames: -1               # Stop after N frames (-1 disables)
 * @endcode
 *
 * @note N2FileData documents the per-file layout, chunking, and compression details.
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
    const std::string _base_dir;     /// Base directory to write files into
    const std::uint64_t _num_file_t; /// Number of incoming time frames per file, as indexed by the
                                     /// absolute frame index
    const std::string _compression;
    const std::uint64_t _compression_level;
    const bool _use_bitshuffle;
    const std::uint64_t _blocksize_f;
    const std::uint64_t _blocksize_p;
    const std::uint64_t _blocksize_t;
    const std::uint64_t
        _late_frame_grace_seconds; /// Grace period in seconds for late frames (default: 60)
    const int _max_frames;         /// Stop writing after this many frames (-1 = unlimited)

    Buffer* const _buffer;

    kotekan::prometheus::Gauge& _write_time_metric;
    kotekan::prometheus::Gauge& _n_datasets_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& _open_file_info_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& _open_file_age_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& _file_completion_fraction_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& _add_frame_errors_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& _last_add_frame_error_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& _finalize_failures_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& _unfinalized_file_metric;

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
    bool _finalize_file(N2FileData& filedata);

    /**
     * @brief Finalize N2FileData objects that have been inactive for too long.
     * @param files  Map of abs_idx-N2FileData objects to finalize
     * @param exclude_abs_file_idx  Optional file index to skip while finalizing
     */
    void _grace_finalize_files(std::map<size_t, std::unique_ptr<N2FileData>>& files,
                               const size_t* exclude_abs_file_idx = nullptr);

    /**
     * @brief Check if the final file (of the form vis_<abs_file_idx>_*.h5) already exists on disk.
     * @param abs_file_idx  Absolute file index to check
     * @param search_dir    Directory to search in
     * @return              True if the final file exists, false otherwise
     */
    bool _finalfile_exists(std::uint64_t abs_file_idx, const std::string& search_dir) const;

    void _record_file_open(const N2FileData& filedata) const;
    void _update_file_metrics(const N2FileData& filedata) const;
    void _clear_open_file_metrics(const N2FileData& filedata) const;
    void _mark_unfinalized(const N2FileData& filedata) const;
    void _record_add_frame_error(const std::string& reason, std::uint64_t abs_file_idx,
                                 int32_t freq_id, std::uint64_t t_index) const;
};

#endif // KOTEKAN_STAGES_HDF5_N2_WRITE_HPP
