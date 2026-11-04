/**
 * @file
 * @brief Coherent beam per selected channel from the voltage ring, written for offline study.
 *  - cudaGnssBeamDump : public cudaCommand
 */

#ifndef CUDA_GNSS_BEAM_DUMP_HPP
#define CUDA_GNSS_BEAM_DUMP_HPP

#include "DataType.hpp"          // for int4x2_swapped_withoffset_t
#include "NDArrayBuffer.hpp"     // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp" // for NDArrayRingBuffer
#include "cudaCommand.hpp"
#include "cudaGnssBeamDumpKernel.hpp" // for BeamDumpLayout
#include "gnssElemSteer.hpp"          // for gnss::ElemSteer
#include "gnssProjSubspace.hpp"       // for gnss::ProjEntry

#include <complex>
#include <map>
#include <string>
#include <vector>

/**
 * @class cudaGnssBeamDump
 * @brief A self-calibrated beam toward the dominant source, per selected channel, per frame,
 *        and a beam per tracked satellite steered by its chain's lane direction.
 *
 * For each selected channel the command forms the covariance of the 128 inputs over the frame,
 * whitens each input by its own power, takes the top eigenvector (power iterations warm-started
 * from the previous frame), references its phase to one input and sums the inputs with those
 * weights, so the beam reads in the reference input's units. The frame it writes carries the
 * weights, the per-input powers and the two largest eigenvalues beside the samples
 * (cudaGnssBeamDumpKernel.hpp). It is a consumer of the voltage ring like cudaCorrelatorDual,
 * in the same process and on the same stream.
 *
 * Target beams (beam_lane_owner). The chain's assembler publishes, per channel, each tracked
 * PRN's element direction on the process-wide ProjBoard (gnssLaneDirection.hpp: the averaged
 * despread row, held through a bright pass by elem_dir_hold_s). Per frame the command takes
 * that chain's lanes on the selected channels, keeps the strongest beam_targets PRNs by
 * coherent power (coherence x row power), re-steers each direction to the frame's mid-time
 * along the lane's track
 * (gnssFleetLane.hpp: a LEO moves the far elements' phase by tens of degrees in one frame)
 * and forms a beam with those weights beside the eigen beam. A target is kept while its lane
 * is on the board above beam_target_min_coh, and replaced when another lane out-scores the
 * weakest target by beam_target_gain for beam_target_hold_s. The frame carries, per target and
 * channel, the lane used, the weights and the beam (gnss_cuda::BeamDumpTarget). The weights are the
 * direction itself: the despread row is the element response conjugated, in the ring's own
 * convention, so sum_i q_i x_i is the coherent sum. beam_lane_conj flips that if the sky says
 * otherwise (beam_dump_read.py info prints the overlap with the eigen weights).
 *
 * The output is an NDArrayBuffer named `beam_name`, on the device as `<beam_name>_buffer`, for
 * cudaOutputData to copy out; its metadata is the ring's with fpga_seq_num set to the frame's
 * first hop. The frame's clock for FrameWindowGate is the int64 seq0 at byte 0.
 *
 * @conf As the ring consumers (buffer_depth, num_times, num_elements, num_local_freq,
 *       voltage_name), and:
 * @conf  beam_name             String. Output NDArrayBuffer name, e.g. gnss0_beam.
 * @conf  beam_local_channels   List of Int. Local frame channel of each selected channel.
 * @conf  beam_freq_ids         List of Int. Their global freq_ids, same order; checked against
 *                              the ring metadata's coarse frequencies on the first frame.
 * @conf  beam_ref_input        Int. The phase-reference input (a live, unflagged input).
 * @conf  beam_eig_iters        Int, default 4. Power iterations per frame per component.
 * @conf  beam_lane_owner       String, default "". The assembler whose lanes are the targets
 *                              (its unique_name, e.g. /gnss0_1202_n2assemble). "" = no targets.
 * @conf  beam_targets          Int, default 2. Target beams per channel.
 * @conf  elem_positions_enu    Double list [128][3]. The assembler's element positions, metres
 *                              ENU. Required with targets.
 * @conf  elem_steer_sign       Double, default 1.0. The assembler's steering sign.
 * @conf  beam_lane_max_age_s   Double, default 2.0. A board entry older than this is ignored.
 * @conf  beam_target_hold_s    Double, default 2.0. See beam_target_gain.
 * @conf  beam_target_gain      Double, default 1.5. A lane replaces the weakest target once its
 *                              coherent power has exceeded the target's by this factor for
 *                              beam_target_hold_s.
 * @conf  beam_target_min_coh   Double, default 0.15. A lane whose coherence is below this on
 *                              every selected channel is not a target (the assembler
 *                              publishes held rows from 0.08).
 * @conf  beam_lane_conj        Bool, default false. Conjugate the lane direction into the
 *                              weights.
 * @conf  frame0_utc            Double, default 0 (the telescope's GPS time0). The record epoch
 *                              override cudaGnssChordTrack takes: the frame times that place
 *                              the re-steer must share the assembler's axis.
 */
class cudaGnssBeamDump : public cudaCommand {
public:
    cudaGnssBeamDump(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                     int instance_num);
    ~cudaGnssBeamDump();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    /// UTC of hop @p seq on the assembler's record axis.
    double frame_utc(long long seq) const;
    /// The best-scoring lane that is not a target; 0 if none.
    int strongest_outsider(const std::map<int, double>& score) const;
    /// Update _target_prn from this frame's lanes (coherent power per PRN) at steady time
    /// @p now_s.
    void choose_targets(const std::map<int, double>& score, double now_s);
    /// Fill _tgt_stage (the target records and weights) for the frame starting at @p seq0.
    void build_targets(long long seq0);

    const int _buffer_depth;
    const int _num_times;
    const int _num_elements;
    const int _num_local_freq;
    const std::string _voltage_name;
    const std::string _beam_name;
    const std::vector<std::int32_t> _local_channels;
    const std::vector<std::int32_t> _freq_ids;
    const int _ref_input;
    const int _eig_iters;
    const std::string _lane_owner;
    const int _n_targets;
    const gnss_cuda::BeamDumpLayout _layout;

    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    NDArrayBuffer<unsigned char, 2> beam;

    std::vector<unsigned char> _hdr_stage; // header + freq_ids, per frame
    bool _uploaded_static = false;
    bool _checked_freqs = false;
    long _frames = 0;

    // Targets (beam_lane_owner).
    std::string _lane_prefix; ///< "<owner>/lane/"
    std::string _lane_group;  ///< the board group: the owner up to its first '_'
    double _lane_max_age_s = 2.0, _target_hold_s = 2.0, _target_gain = 1.5;
    double _target_min_coh = 0.15;
    bool _lane_conj = false;
    double _frame0_offset_s = 0.0; ///< config frame0_utc minus the telescope's epoch (0 = none)
    bool _time_ok = false;         ///< a record epoch exists (frame0_utc or the GPS time0)
    gnss::ElemSteer _geo;
    std::vector<int> _target_prn;   ///< [n_targets]; 0 = empty slot
    int _challenger_prn = 0;        ///< the outsider out-powering the weakest target
    double _challenger_since = 0.0; ///< steady seconds since which it has
    std::map<int, std::vector<gnss::ProjEntry>> _lanes; ///< this frame's lanes per PRN, per
                                                        ///< selected channel (k 0 = none)
    std::vector<unsigned char> _tgt_stage;  ///< the target records and weights, per frame
    std::vector<gnss::ElemSteer::cf> _sh;   ///< fleet_lane_direction scratch
    std::vector<std::complex<double>> _dir; ///< [BEAM_INPUTS]
    bool _warned_n_elem = false;
};

#endif
