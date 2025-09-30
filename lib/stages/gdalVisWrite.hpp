// NOTE: This header provides the gdalVisFileData buffering helper and the
//       declaration for the gdalVisWrite stage. The stage methods are
//       implemented and registered in gdalVisWrite.cpp.
#ifndef KOTEKAN_STAGES_GDAL_VIS_WRITE_HPP
#define KOTEKAN_STAGES_GDAL_VIS_WRITE_HPP

#include "gdalFiles.hpp"

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <N2FrameView.hpp>
#include <N2Metadata.hpp>
#include <N2Util.hpp>
#include <visUtil.hpp>

#include <cassert>
#include <gdal.h>
#include <gdal_priv.h>
#include <memory>
#include <vector>

/**
 * @class gdalVisFileData
 * @brief Buffer a full file block worth of arrays in memory, then flush once.
 *
 * A file block contains `file_nt` time frames across all `num_freq` freqs.
 * This class holds all relevant arrays in the target on-disk layout to enable
 * large contiguous writes (per-frequency slab) rather than per-frame writes.
 */
class gdalVisFileData {
protected:
    // Structural information -- all frames that get added should agree with these
    const size_t num_input;    // number of inputs / elements
    const size_t num_prod;     // number of products
    const size_t num_ev;       // number of eigenvectors/values
    const size_t num_freq;     // number of frequencies
    const size_t file_nt;      // frames per file (time dimension)

    // Datasets to be stored until ready to write
    // f = freq, p = prod, e = eigen, i = input, t = time
    std::vector<N2::cfloat> vis;       // (f, p, t)
    std::vector<float> vis_weight;     // (f, p, t)
    std::vector<float> eval;           // (f, e, t)
    std::vector<N2::cfloat> evec;      // (f, e, i, t)
    std::vector<float> erms;           // (f, t)
    std::vector<N2::cfloat> gain;      // (f, i, t)
    std::vector<float> frac_lost;      // (f, t)
    std::vector<float> frac_rfi;       // (f, t)
    std::vector<float> flags;          // (f, i, t)

    /// Earth rotation angle at corresponding times (t)
    std::vector<double> era_deg;       // (t)

    // Additional metadata
    std::vector<uint64_t> fpga_start_tick;         // (t)
    std::vector<uint64_t> frame_start_time_ns;     // (t)
    std::vector<uint64_t> frame_length_fpga_ticks; // (t)
    std::vector<uint64_t> n_valid_fpga_ticks;      // (f, t)
    std::vector<uint64_t> n_rfi_fpga_ticks;        // (f, t)

    // Tracking what (f, t) pairs have been seen
    std::vector<uint8_t> seen; // size = num_freq * file_nt
    size_t seen_count;

public:
    gdalVisFileData(const uint64_t file_nt_, const uint64_t num_freq_, const uint64_t num_input_,
                    const uint64_t num_prod_, const uint64_t num_ev_)
        : num_input(num_input_), num_prod(num_prod_), num_ev(num_ev_), num_freq(num_freq_),
          file_nt(file_nt_), seen_count(0) {
        // resize arrays to hold data across (freq, time) blocks
        vis.assign(num_prod * num_freq * file_nt, N2::cfloat{0.0f, 0.0f});
        vis_weight.assign(num_prod * num_freq * file_nt, 0.0f);
        eval.assign(num_ev * num_freq * file_nt, 0.0f);
        evec.assign(num_ev * num_input * num_freq * file_nt, N2::cfloat{0.0f, 0.0f});
        erms.assign(num_freq * file_nt, 0.0f);
        gain.assign(num_input * num_freq * file_nt, N2::cfloat{0.0f, 0.0f});
        flags.assign(num_input * num_freq * file_nt, 0.0f);
        frac_lost.assign(num_freq * file_nt, 1.0f); // match empty frames by default
        frac_rfi.assign(num_freq * file_nt, 0.0f);

        // Additional metadata
        fpga_start_tick.assign(file_nt, 0);
        frame_start_time_ns.assign(file_nt, 0);
        frame_length_fpga_ticks.assign(file_nt, 0);
        era_deg.assign(file_nt, 0.0);
        n_valid_fpga_ticks.assign(num_freq * file_nt, 0);
        n_rfi_fpga_ticks.assign(num_freq * file_nt, 0);

        seen.assign(num_freq * file_nt, 0);


    }

public:
    inline size_t idx_fpt(size_t f, size_t p, size_t t) const {
        return file_nt * (num_prod * f + p) + t; // (f, p, t)
    }
    inline size_t idx_fet(size_t f, size_t e, size_t t) const {
        return file_nt * (num_ev * f + e) + t; // (f, e, t)
    }
    inline size_t idx_feit(size_t f, size_t e, size_t i, size_t t) const {
        return file_nt * (num_input * (num_ev * f + e) + i) + t; // (f, e, i, t)
    }
    inline size_t idx_fit(size_t f, size_t i, size_t t) const {
        return file_nt * (num_input * f + i) + t; // (f, i, t)
    }
    inline size_t idx_ft(size_t f, size_t t) const { return file_nt * f + t; } // (f, t)

    inline size_t idx_seen(size_t f, size_t t) const { return file_nt * f + t; }

    /**
     * @brief Add a frame of data at the computed time index.
     * @param fv      Frame view containing data.
     * @param meta    N2 metadata for the frame.
     * @param t_index Time index within this file block (0..file_nt-1).
     */
    void add_frame(const N2FrameView& fv, const std::shared_ptr<N2Metadata>& meta, size_t t_index) {
        const size_t f_index = meta->freq_id; // assumes local 0..nfreq-1 indexing
        assert(f_index < num_freq);
        assert(t_index < file_nt);

        // Per-time metadata consistency and assignment
        if (fpga_start_tick[t_index] != 0)
            assert(fpga_start_tick[t_index] == meta->fpga_start_tick);
        fpga_start_tick[t_index] = meta->fpga_start_tick;
        if (frame_start_time_ns[t_index] != 0)
            assert(frame_start_time_ns[t_index] == meta->frame_start_time_ns);
        frame_start_time_ns[t_index] = meta->frame_start_time_ns;
        if (frame_length_fpga_ticks[t_index] != 0)
            assert(frame_length_fpga_ticks[t_index] == meta->frame_length_fpga_ticks);
        frame_length_fpga_ticks[t_index] = meta->frame_length_fpga_ticks;
        era_deg[t_index] = fv.eop.ERA_deg;

        // vis + weight
        for (size_t p = 0; p < num_prod; ++p) {
            vis[idx_fpt(f_index, p, t_index)] = fv.vis[p];
            vis_weight[idx_fpt(f_index, p, t_index)] = fv.weight[p];
        }
        // eval + evec
        for (size_t e = 0; e < num_ev; ++e) {
            eval[idx_fet(f_index, e, t_index)] = fv.eval[e];
            for (size_t i = 0; i < num_input; ++i) {
                evec[idx_feit(f_index, e, i, t_index)] = fv.evec[num_input * e + i];
            }
        }
        // erms, gain, flags
        erms[idx_ft(f_index, t_index)] = fv.erms;
        for (size_t i = 0; i < num_input; ++i) {
            gain[idx_fit(f_index, i, t_index)] = fv.gain[i];
            flags[idx_fit(f_index, i, t_index)] = fv.flags[i];
        }

        // Derived fractions and counts
        n_valid_fpga_ticks[idx_ft(f_index, t_index)] = meta->n_valid_fpga_ticks;
        n_rfi_fpga_ticks[idx_ft(f_index, t_index)] = meta->n_rfi_fpga_ticks;
        if (meta->frame_length_fpga_ticks == 0) {
            frac_lost[idx_ft(f_index, t_index)] = 1.0f;
            frac_rfi[idx_ft(f_index, t_index)] = 0.0f;
        } else {
            frac_lost[idx_ft(f_index, t_index)] =
                1.0f - float(meta->n_valid_fpga_ticks) / float(meta->frame_length_fpga_ticks);
            frac_rfi[idx_ft(f_index, t_index)] =
                float(meta->n_rfi_fpga_ticks) / float(meta->frame_length_fpga_ticks);
        }

        size_t si = idx_seen(f_index, t_index);
        if (!seen[si]) {
            seen[si] = 1;
            ++seen_count;
        }
    }

    bool full() const { return seen_count == num_freq * file_nt; }

    /**
     * @brief Flush the entire buffered block to an open GDAL dataset.
     * The dataset is expected to already contain all arrays created by the stage.
     */
    void write_all_to_dataset(GDALDataset* dataset) const {
        assert(dataset);
        const auto root_group = dataset->GetRootGroup();
        assert(root_group);

        // Open arrays
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
        auto frame_length_fpga_ticks_array = root_group->OpenMDArray("frame_length_fpga_ticks");
        auto era_deg_array = root_group->OpenMDArray("era_deg");
        auto n_valid_array = root_group->OpenMDArray("n_valid_fpga_ticks");
        auto n_rfi_array = root_group->OpenMDArray("n_rfi_fpga_ticks");

        const auto c32Type = GDALExtendedDataType::Create(GDT_CFloat32);
        const auto f32Type = GDALExtendedDataType::Create(GDT_Float32);
        const auto f64Type = GDALExtendedDataType::Create(GDT_Float64);
        const auto u64Type = GDALExtendedDataType::Create(GDT_UInt64);

        // Per-frequency slabs for arrays with freq dimension
        for (GUInt64 f = 0; f < num_freq; ++f) {
            // vis/weights: (f, :, :)
            std::vector<GUInt64> start_v = {f, 0, 0};
            std::vector<size_t> count_v = {1, num_prod, file_nt};
            bool ok = vis_array->Write(start_v.data(), count_v.data(), nullptr, nullptr, c32Type,
                                       reinterpret_cast<const void*>(&vis[idx_fpt(f, 0, 0)]), nullptr,
                                       0);
            assert(ok);
            ok = weights_array->Write(start_v.data(), count_v.data(), nullptr, nullptr, f32Type,
                                      reinterpret_cast<const void*>(&vis_weight[idx_fpt(f, 0, 0)]),
                                      nullptr, 0);
            assert(ok);

            // eval: (f, :, :)
            std::vector<GUInt64> start_e = {f, 0, 0};
            std::vector<size_t> count_e = {1, num_ev, file_nt};
            ok = eval_array->Write(start_e.data(), count_e.data(), nullptr, nullptr, f32Type,
                                   reinterpret_cast<const void*>(&eval[idx_fet(f, 0, 0)]), nullptr,
                                   0);
            assert(ok);

            // evec: (f, :, :, :)
            std::vector<GUInt64> start_ev = {f, 0, 0, 0};
            std::vector<size_t> count_ev = {1, num_ev, num_input, file_nt};
            ok = evec_array->Write(start_ev.data(), count_ev.data(), nullptr, nullptr, c32Type,
                                   reinterpret_cast<const void*>(&evec[idx_feit(f, 0, 0, 0)]),
                                   nullptr, 0);
            assert(ok);

            // erms: (f, :)
            std::vector<GUInt64> start_et = {f, 0};
            std::vector<size_t> count_et = {1, file_nt};
            ok = erms_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, f32Type,
                                   reinterpret_cast<const void*>(&erms[idx_ft(f, 0)]), nullptr, 0);
            assert(ok);

            // gain: (f, :, :)
            std::vector<GUInt64> start_g = {f, 0, 0};
            std::vector<size_t> count_g = {1, num_input, file_nt};
            ok = gain_array->Write(start_g.data(), count_g.data(), nullptr, nullptr, c32Type,
                                   reinterpret_cast<const void*>(&gain[idx_fit(f, 0, 0)]), nullptr,
                                   0);
            assert(ok);

            // flags: (f, :, :)
            ok = flags_array->Write(start_g.data(), count_g.data(), nullptr, nullptr, f32Type,
                                    reinterpret_cast<const void*>(&flags[idx_fit(f, 0, 0)]), nullptr,
                                    0);
            assert(ok);

            // frac_* and counts: (f, :)
            ok = frac_lost_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, f32Type,
                                        reinterpret_cast<const void*>(&frac_lost[idx_ft(f, 0)]),
                                        nullptr, 0);
            assert(ok);
            ok = frac_rfi_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, f32Type,
                                       reinterpret_cast<const void*>(&frac_rfi[idx_ft(f, 0)]), nullptr,
                                       0);
            assert(ok);
            ok = n_valid_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, u64Type,
                                      reinterpret_cast<const void*>(&n_valid_fpga_ticks[idx_ft(f, 0)]),
                                      nullptr, 0);
            assert(ok);
            ok = n_rfi_array->Write(start_et.data(), count_et.data(), nullptr, nullptr, u64Type,
                                    reinterpret_cast<const void*>(&n_rfi_fpga_ticks[idx_ft(f, 0)]),
                                    nullptr, 0);
            assert(ok);
        }

        // per-time arrays: (:)
        {
            std::vector<GUInt64> start = {0};
            std::vector<size_t> count = {file_nt};
            bool ok = fpga_start_tick_array->Write(start.data(), count.data(), nullptr, nullptr,
                                                   u64Type,
                                                   reinterpret_cast<const void*>(
                                                       fpga_start_tick.data()),
                                                   nullptr, 0);
            assert(ok);
            ok = frame_start_time_ns_array->Write(start.data(), count.data(), nullptr, nullptr,
                                                  u64Type,
                                                  reinterpret_cast<const void*>(
                                                      frame_start_time_ns.data()),
                                                  nullptr, 0);
            assert(ok);
            ok = frame_length_fpga_ticks_array->Write(start.data(), count.data(), nullptr, nullptr,
                                                      u64Type,
                                                      reinterpret_cast<const void*>(
                                                          frame_length_fpga_ticks.data()),
                                                      nullptr, 0);
            assert(ok);
            ok = era_deg_array->Write(start.data(), count.data(), nullptr, nullptr, f64Type,
                                      reinterpret_cast<const void*>(era_deg.data()), nullptr, 0);
            assert(ok);
        }
    }
};

/**
 * @class gdalVisWrite
 * @brief Buffered-transpose writer: buffers sequential time frames and writes full GDAL Zarr blocks.
 *
 * Frames arrive sequentially in time. The stage buffers a complete (nfreq × file_nt) block
 * per output file in memory (via gdalVisFileData), and when the block is complete, writes
 * all arrays to disk in large contiguous slabs.
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
 * @conf zip_compression    UInt. 0 disables ZIP; >0 enables ZIP STORAGE with given DEFLATE level (default: 0)
 * @conf blocksize_f        UInt. Array chunk size along freq (0 = driver default)
 * @conf blocksize_p        UInt. Array chunk size along product (unused currently; 0 = default)
 * @conf blocksize_t        UInt. Array chunk size along time (default: 1)
 * @conf file_nt            UInt. Frames per file in time dimension (default: 2)
 * @conf max_frames         Int.  Stop after this many frames (-1 = unlimited)
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
    const std::string base_dir;
    const std::string file_name;
    const bool prefix_hostname;

    const std::uint64_t zip_compression;
    const std::uint64_t blocksize_f;
    const std::uint64_t blocksize_p;
    const std::uint64_t blocksize_t;

    const int max_frames;
    const std::uint32_t file_nt;

    Buffer* const buffer;

private:
    std::uint64_t _get_frame_nt_in_file(const std::shared_ptr<const N2Metadata> meta);
    std::uint64_t _get_file_start_time_ns(const std::shared_ptr<const N2Metadata> meta);
    std::string _get_gdal_vis_filename(std::shared_ptr<const N2Metadata> meta);
    void _initialize_gdal_vis_file(GDALDataset* dataset, std::shared_ptr<const N2Metadata> meta);

    struct DatasetCtx {
        GDALDataset* ds = nullptr;
        std::unique_ptr<gdalVisFileData> buf;
    };
};

#endif // KOTEKAN_STAGES_GDAL_VIS_WRITE_HPP
