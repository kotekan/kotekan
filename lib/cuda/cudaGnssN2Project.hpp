/**
 * @file
 * @brief The N^2 lane projection on the GPU, in place on the science correlator's frame.
 *  - cudaGnssN2Project : public cudaCommand
 */
#ifndef CUDA_GNSS_N2_PROJECT_HPP
#define CUDA_GNSS_N2_PROJECT_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "prometheusMetrics.hpp"

#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

class chordMetadata;

/// What the instances of one cudaGnssN2Project share.
class cudaGnssN2ProjectState : public cudaCommandState {
public:
    cudaGnssN2ProjectState(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaGnssN2ProjectState() override;

    /// The newest basis record taken from in_basis_buf; empty before the first.
    std::vector<uint8_t> newest;
    int basis_in_id = 0; ///< next frame of in_basis_buf
    int live_id = 0;     ///< next frame of live_buf
    /// Resolved on the first frame from its coarse frequencies: per slot, the local channel and
    /// its freq_id, in local channel order (GnssN2Project's slot order).
    bool resolved = false;
    std::vector<int> slot_chan, slot_fid;
    int* d_chan = nullptr;
    float2 *d_v = nullptr, *d_q = nullptr, *d_w = nullptr, *d_m = nullptr;
    cublasHandle_t blas = nullptr;
    kotekan::prometheus::Gauge& lag;
    kotekan::prometheus::Counter& dropped;
    uint64_t frames = 0, drops = 0, records = 0;
};

/**
 * @class cudaGnssN2Project
 * @brief Project each live channel of the correlation frame with the newest lane basis
 *        GnssN2Project has built, in place, and hand the host the live channels' raw and
 *        projected blocks and the record of what was applied.
 *
 * Runs in the science correlator's cudaProcess, after cudaCorrelator and before cudaSyncOutput,
 * on its kernel stream: the frame cudaOutputData then copies out is the projected one, and
 * N2Accumulate reads it directly. Per frame:
 *  1. the newest record of in_basis_buf (GnssN2Project in gpu mode writes one per frame it
 *     processes, built from that frame's raw blocks) is taken in wait_on_precondition without
 *     waiting; before the first, k is 0 on every slot. The record becomes the frame's applied
 *     record, gpu_mem_basis, with its seq set to this frame's and its source frame's seq at
 *     byte 40 (gnssN2ProjectMath.hpp), for cudaOutputData to copy out to N2Accumulate: the
 *     record is the bytes the kernels used;
 *  2. the raw blocks of the live channels are gathered, the frame is projected in FP32
 *     (cudaGnssN2ProjectKernel.hpp), and the projected blocks gathered after them;
 *  3. both go to live_buf if GnssN2Project has an empty frame, else the frame's are dropped
 *     (kotekan_gnss_n2project_gpu_live_dropped_total): the correlator never waits for the lane
 *     builder.
 * The live-block frame (gnss_n2proj::live_*): a header (seq, the applied record's source seq,
 * the slot count, the words per slot and the applied k per slot), then the raw and the
 * projected blocks, each [slot][36][16][16][2] int32 in the n2k layout. Its metadata carries
 * the seq and, as coarse frequencies, the slots' freq_ids.
 *
 * kotekan_gnss_n2project_gpu_basis_lag_frames: frames between the applied record's source frame
 * and this one (1 when GnssN2Project keeps up; -1 before the first record).
 *
 * @conf num_elements, num_local_freq, samples_per_data_set, sub_integration_ntime, buffer_depth
 *       As the correlator: 128 inputs, one integration per frame.
 * @conf n2k_correlation_name String. cudaCorrelator's output (its device region <name>_buffer).
 * @conf stations             Int list. GnssN2Project's stations; checked against each record.
 * @conf live_freq_ids        Int list. GnssN2Project's live_freq_ids: the slots, resolved
 *                            against the first frame's coarse frequencies.
 * @conf k_lanes_max          Int. GnssN2Project's k_lanes_max: the record's directions per slot.
 * @conf in_basis_buf         String. The records GnssN2Project builds (its basis_buf).
 * @conf live_buf             String. The live-block frames, for GnssN2Project's in_buf.
 * @conf gpu_mem_basis        String. The applied record's device region, for cudaOutputData.
 */
class cudaGnssN2Project : public cudaCommand {
public:
    cudaGnssN2Project(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                      int instance_num, const std::shared_ptr<cudaCommandState>& state);
    ~cudaGnssN2Project() override;
    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    cudaGnssN2ProjectState& state();
    void resolve(const chordMetadata& meta);
    /// FATAL unless @p rec is a record for this command's stations, slots and k_lanes_max.
    void check_record(const uint8_t* rec);
    /// A record with k 0 on every slot into the staging.
    void write_empty();

    const int num_elements, num_local_freq, buffer_depth, samples_per_data_set;
    const std::vector<int> stations;
    const std::vector<int> live_freq_ids;
    const int k_slot;
    const std::string corr_mem, basis_mem, live_mem;
    Buffer* in_basis;
    Buffer* live;
    const size_t corr_bytes, rec_bytes;

    // This instance's frame
    uint8_t* staging = nullptr; ///< pinned: the record uploaded for the frame
    int live_frame = -1;        ///< live_buf frame taken for it, -1 if dropped
    int64_t seq = 0, src_seq = -1, tdf = 0;
    std::vector<int32_t> k_applied;
};

#endif
