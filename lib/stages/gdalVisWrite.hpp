// NOTE: This header provides the gdalVisFileData buffering helper and the
//       declaration for the gdalVisWrite stage. The stage methods are
//       implemented and registered in gdalVisWrite.cpp.
#ifndef KOTEKAN_STAGES_GDAL_VIS_WRITE_HPP
#define KOTEKAN_STAGES_GDAL_VIS_WRITE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gdalFiles.hpp"

#include <N2FrameView.hpp>
#include <N2Metadata.hpp>
#include <N2Util.hpp>
#include <cassert>
#include <gdal.h>
#include <gdal_priv.h>
#include <memory>
#include <vector>
#include <visUtil.hpp>

/**
 * @class gdalVisFileData
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
class gdalVisFileData {
public:
    // Structural information and fixed sizes
    const size_t num_input;  // number of inputs / elements
    const size_t num_prod;   // number of products
    const size_t num_ev;     // number of eigenvectors/values
    const size_t num_freq;   // number of frequencies
    const size_t num_file_t; // frames ("time" dimension)

    // Per-file bookkeeping owned by this object
    GDALDataset* gdal_dataset = nullptr; // non-owning GDAL dataset handle
    const std::string partial_path;      // working on-disk location
    const double open_wall_s;            // time opened
    double last_update_wall_s = 0.0;     // last frame receipt
    const uint64_t file_start_time_ns;   // aligned file start time
    const uint64_t frame_len_ns;         // frame length in ns

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
    gdalVisFileData(const uint64_t num_file_t_, const uint64_t num_freq_, const uint64_t num_input_,
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
        n_valid_fpga_ticks.assign(num_freq * num_file_t, 0);
        n_rfi_fpga_ticks.assign(num_freq * num_file_t, 0);

        added_ft.assign(num_freq * num_file_t, 0);
    }

    /// Flush buffered data to the associated dataset handle, always writing the
    /// entire time range [0 .. num_file_t-1] regardless of which frames were
    /// populated. Returns true if a write occurred.
    bool flush() {
        if (!gdal_dataset)
            return false;

        // Inline previous _write_arrays() logic here
        const size_t nt = num_file_t;
        assert(gdal_dataset);
        const auto root_group = gdal_dataset->GetRootGroup();
        assert(root_group);

        auto vis_array = root_group->OpenMDArray("vis_array");
        auto weights_array = root_group->OpenMDArray("weights_array");
        auto eval_array = root_group->OpenMDArray("eval_array");
        auto evec_array = root_group->OpenMDArray("evec_array");
        auto erms_array = root_group->OpenMDArray("erms_array");
        auto gain_array = root_group->OpenMDArray("gain_array");
        auto flags_array = root_group->OpenMDArray("flags_array");
        auto frac_lost_array = root_group->OpenMDArray("frac_lost_array");
        auto frac_rfi_array = root_group->OpenMDArray("frac_rfi_array");
        auto fpga_start_tick_array = root_group->OpenMDArray("fpga_start_tick");
        auto frame_start_time_ns_array = root_group->OpenMDArray("frame_start_time_ns");
        auto era_deg_array = root_group->OpenMDArray("era_deg");
        auto frame_length_fpga_ticks_array = root_group->OpenMDArray("frame_length_fpga_ticks");
        auto n_valid_array = root_group->OpenMDArray("n_valid_fpga_ticks");
        auto n_rfi_array = root_group->OpenMDArray("n_rfi_fpga_ticks");

        const auto c32Type = GDALExtendedDataType::Create(GDT_CFloat32);
        const auto f32Type = GDALExtendedDataType::Create(GDT_Float32);
        const auto f64Type = GDALExtendedDataType::Create(GDT_Float64);
        const auto u64Type = GDALExtendedDataType::Create(GDT_UInt64);

        // vis/weights: (freqs, products, nt)
        {
            std::vector<GUInt64> start_v = {0, 0, 0};
            std::vector<size_t> count_v = {num_freq, num_prod, nt};
            bool ok = vis_array->Write(start_v.data(), count_v.data(), nullptr, nullptr, c32Type,
                                       reinterpret_cast<const void*>(vis.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write vis_array to dataset {}", partial_path);
            ok = weights_array->Write(start_v.data(), count_v.data(), nullptr, nullptr, f32Type,
                                      reinterpret_cast<const void*>(vis_weight.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write weights_array to dataset {}", partial_path);
        }

        // eval: (freqs, ev, nt)
        {
            std::vector<GUInt64> start_e = {0, 0, 0};
            std::vector<size_t> count_e = {num_freq, num_ev, nt};
            bool ok = eval_array->Write(start_e.data(), count_e.data(), nullptr, nullptr, f32Type,
                                        reinterpret_cast<const void*>(eval.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write eval_array to dataset {}", partial_path);
        }

        // evec: (freqs, ev, inputs, nt)
        {
            std::vector<GUInt64> start_ev = {0, 0, 0, 0};
            std::vector<size_t> count_ev = {num_freq, num_ev, num_input, nt};
            bool ok = evec_array->Write(start_ev.data(), count_ev.data(), nullptr, nullptr, c32Type,
                                        reinterpret_cast<const void*>(evec.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write evec_array to dataset {}", partial_path);
        }

        // erms: (freqs, nt); gain/flags: (freqs, inputs, nt); frac_* and counts: (freqs, nt)
        {
            std::vector<GUInt64> start_et = {0, 0};
            std::vector<size_t> count_et = {num_freq, nt};
            bool ok = erms_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, f32Type,
                                        reinterpret_cast<const void*>(erms.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write erms_array to dataset {}", partial_path);

            std::vector<GUInt64> start_g = {0, 0, 0};
            std::vector<size_t> count_g = {num_freq, num_input, nt};
            ok = gain_array->Write(start_g.data(), count_g.data(), nullptr, nullptr, c32Type,
                                   reinterpret_cast<const void*>(gain.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write gain_array to dataset {}", partial_path);
            ok = flags_array->Write(start_g.data(), count_g.data(), nullptr, nullptr, f32Type,
                                    reinterpret_cast<const void*>(flags.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write flags_array to dataset {}", partial_path);

            ok =
                frac_lost_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, f32Type,
                                       reinterpret_cast<const void*>(frac_lost.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write frac_lost_array to dataset {}", partial_path);
            ok = frac_rfi_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, f32Type,
                                       reinterpret_cast<const void*>(frac_rfi.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write frac_rfi_array to dataset {}", partial_path);
            ok = n_valid_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, u64Type,
                                      reinterpret_cast<const void*>(n_valid_fpga_ticks.data()),
                                      nullptr, 0);
            if(!ok)
                ERROR("Failed to write n_valid_array to dataset {}", partial_path);
            ok = n_rfi_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, u64Type,
                                    reinterpret_cast<const void*>(n_rfi_fpga_ticks.data()), nullptr,
                                    0);
            if(!ok)
                ERROR("Failed to write n_rfi_array to dataset {}", partial_path);
        }

        // per-time arrays: (:)
        {
            std::vector<GUInt64> start = {0};
            std::vector<size_t> count = {nt};
            bool ok = fpga_start_tick_array->Write(
                start.data(), count.data(), nullptr, nullptr, u64Type,
                reinterpret_cast<const void*>(fpga_start_tick.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write fpga_start_tick_array to dataset {}", partial_path);
            ok = frame_start_time_ns_array->Write(
                start.data(), count.data(), nullptr, nullptr, u64Type,
                reinterpret_cast<const void*>(frame_start_time_ns.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write frame_start_time_ns_array to dataset {}", partial_path);
            // Write frame_length_fpga_ticks across full file dimension (constant per file)
            std::vector<GUInt64> start_full = {0};
            std::vector<size_t> count_full = {num_file_t};
            std::vector<uint64_t> flft(num_file_t, frame_length_fpga_ticks);
            ok = frame_length_fpga_ticks_array->Write(
                start_full.data(), count_full.data(), nullptr, nullptr, u64Type,
                reinterpret_cast<const void*>(flft.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write frame_length_fpga_ticks_array to dataset {}", partial_path);
            ok = era_deg_array->Write(start.data(), count.data(), nullptr, nullptr, f64Type,
                                      reinterpret_cast<const void*>(era_deg.data()), nullptr, 0);
            if(!ok)
                ERROR("Failed to write era_deg_array to dataset {}", partial_path);
        }

        return true;
    }

    /// Close the associated dataset handle if open.
    void close() {
        if (gdal_dataset) {
            GDALClose(gdal_dataset);
            gdal_dataset = nullptr;
        }
    }

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
        assert(f_index < num_freq);
        assert(t_index < num_file_t);

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
        // meta->num_elements already asserted above

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
        if (era_deg[t_index] == 0)
            era_deg[t_index] = fv.eop.ERA_deg; // No consistency check for ERA (float comparison)

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
        const uint64_t n_valid = std::min(meta->n_valid_fpga_ticks, frame_len_ticks);
        const uint64_t n_rfi = std::min(meta->n_rfi_fpga_ticks, frame_len_ticks);
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
};

/**
 * @class gdalVisWrite
 * @brief Buffered-transpose writer: buffers sequential time frames and writes GDAL Zarr files.
 *
 * This stage groups frames into UTC-midnight-aligned windows of length
 * `window_seconds`, buffers a complete (num_freq * file_nt) block per output file in
 * memory (via gdalVisFileData), and when the block is full, writes all arrays to
 * disk in large contiguous slabs. If the block is not full, it is nevertheless finalized
 * after `late_frame_grace_seconds` of inactivity once frames for later windows begin to
 * arrive. Flushes always write the full time range for the window; any missing frames
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
 * @conf zip_compression    UInt. 0 disables ZIP; >0 enables ZIP STORAGE with given DEFLATE level
 *(default: 0)
 * @conf blocksize_f        UInt. Array chunk size along freq (0 = driver default)
 * @conf blocksize_p        UInt. Array chunk size along product (unused currently; 0 = default)
 * @conf blocksize_t        UInt. Array chunk size along time (default: 1)
 *
 * @conf window_seconds            UInt. Window length in seconds; must divide 86400
 * @conf late_frame_grace_seconds  UInt. Grace period in seconds for late frames (default: 60)
 * @conf max_frames                Int.  Stop writing after this many frames (-1 = unlimited)
 *
 * @par Metrics
 * @metric kotekan_gdalviswrite_write_time_seconds  Duration to write the last flush
 * @metric kotekan_gdalviswrite_n_datasets          Number of datasets currently open
 **/
class gdalVisWrite : public kotekan::Stage {

public:
    gdalVisWrite(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    virtual ~gdalVisWrite();

    void main_thread() override;

private:
    // Config (initialized from Config in constructor)
    const std::string base_dir;  /// Base directory to write files into
    const std::string file_name; /// Base filename to write
    const bool prefix_hostname;  /// Prefix files with hostname (default: true)

    const std::uint64_t zip_compression; /// ZIP compression level (0 = disabled)
    const std::uint64_t blocksize_f;     /// Array chunk size along frequency
    const std::uint64_t
        blocksize_p; /// Array chunk size along product (0 = default = full num products)
    const std::uint64_t blocksize_t;    /// Array chunk size along time (default: 1)
    const std::uint64_t window_seconds; /// Window length in seconds; must divide 86400
    const std::uint64_t
        late_frame_grace_seconds; /// Grace period in seconds for late frames (default: 60)

    const int max_frames; /// Stop writing after this many frames (-1 = unlimited)

    Buffer* const buffer;

private:
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
     * output file based on the configured window size. File times are
     * aligned to multiples of `window_seconds` since UTC midnight.
     */
    std::uint64_t _get_file_start_time_ns(const std::shared_ptr<const N2Metadata> meta);

    /**
     * @brief Get the GDAL Zarr filename for the given metadata.
     * @param meta  N2Metadata for the file
     * @return      Full path to the GDAL Zarr file to write
     * This method computes the output filename based on the configured
     * base directory, file name, and whether to prefix with hostname.
     */
    std::string _get_gdal_vis_filename(std::shared_ptr<const N2Metadata> meta);

    /**
     * @brief Initialize a GDALDataset for the given file and metadata.
     * @param dataset   GDALDataset pointer to initialize
     * @param meta      N2Metadata for the file
     * @param file_nt   Number of time frames in the file
     *
     * This method sets up the GDALDataset with the appropriate arrays,
     * dimensions, chunking, and compression based on the configuration
     * parameters.
     */
    void _initialize_gdal_vis_file(GDALDataset* dataset, std::shared_ptr<const N2Metadata> meta,
                                   std::uint64_t file_nt);

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

#endif // KOTEKAN_STAGES_GDAL_VIS_WRITE_HPP
