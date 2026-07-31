/**
 * @file
 * @brief CHORD N-antenna GNSS despread command: waveform generation + N x M correlation.
 *  - cudaGnssChordTrackState : public cudaCommandState
 *  - cudaGnssChordTrack : public cudaCommand
 */

#ifndef CUDA_GNSS_CHORD_TRACK_HPP
#define CUDA_GNSS_CHORD_TRACK_HPP

#include "Config.hpp"
#include "GnssCudaDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "restServer.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class cudaGnssChordTrackState
 * @brief Broker seeds + per-PRN control, shared across this command's instances.
 */
class cudaGnssChordTrackState : public cudaCommandState {
public:
    cudaGnssChordTrackState(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaGnssChordTrackState();

    /// Broker seed contract, identical to cudaGnssTrack's /set_seeds: a JSON array of
    /// {prn, doppler_hz, code_phase_chips, code_phase_rate, doppler_rate_hz_s,
    ///  carrier_trim_hz, ref_hop}. Deliberately the SAME contract so one broker can drive
    /// either chain and the airspy tooling works unchanged against a CHORD node.
    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// GET endpoint: per-PRN code-trim state {prn, trim_chips, disc, quality}, for the
    /// broker/scripts to watch the in-tracker DLL without touching the record stream.
    void get_trim_callback(kotekan::connectionInstance& conn);

    // Geometry (immutable after construction).
    std::vector<int> prns;
    int n_prn = 0, n_chan = 0, n_elem = 0;
    int elem_stride = 0, frame_chan_stride = 0;
    int hops_per_record = 0, n_hops_frame = 0, fft_len = 16384;
    double sample_rate = 3.2e9, f_offset_hz = 0.0, dll_spacing = 0.5;
    bool _conjugate = false; ///< F-engine conjugation (see DespreadParams::conj_data)
    double doppler_margin_hz = 5000.0;

    /// One PRN's live seed. Model-primary: the broker owns these and refreshes them every
    /// cycle, so there is no frozen-seed state to age or unfreeze here.
    struct Seed {
        bool have = false;
        double doppler_hz = 0.0;
        double cp_chips = 0.0;    ///< prompt code phase at ref_hop
        double cp_rate = 0.0;     ///< chips per hop (the broker's measured l-a residual)
        double dop_rate = 0.0;    ///< Hz/s
        double ctrim_hz = 0.0;    ///< broker carrier trim
        long long ref_hop = 0;    ///< absolute hop the cp is referenced to
    };
    std::mutex seed_mtx;
    std::vector<Seed> seeds;

    // ---- In-tracker DLL code trim (config `code_trim`, default false = today's behaviour).
    //
    // WHY IT EXISTS (2026-07-31): the CHORD clock chain breathes ~+-1 chip (+-98 ns) with a
    // ~20 s period (see docs/CHORD_GNSS_STATE.md 5h). Airspy closes its code loop in the
    // BROKER from the combiner's windowed E/L powers (~1 Hz), which that clock tolerates;
    // CHORD's +-0.2 chips/s slew defeats any seconds-cadence external loop (REST latency
    // 1.5-3 s, measured). So the same discriminator/leaky-integrator math runs HERE, once per
    // GPU frame (~24 Hz), on this command's own E/P/L rows: each execute() enqueues a small
    // D2H copy of its correlator rows (reference element only) with an event, and a later
    // execute() consumes whatever has completed -- one frame (~42 ms) of loop latency against
    // a 20 s oscillation. The trim is added to the broker's model cp at the single
    // Spec-construction point; the broker's own DLL (3c) then sees disc ~ 0 and stays quiet.
    bool trim_enable = false;
    double trim_gain = 0.15;        ///< integrator gain per update
    double trim_leak = 0.002;       ///< leaky-integrator leak per update (noise can't walk it)
    double trim_clamp = 3.0;        ///< |trim| bound, chips
    double trim_quality_min = 1.8;  ///< q = 2|P|^2/(|E|^2+|L|^2) gate: ~1 noise, ~4 locked
    int trim_ref_elem = 0;          ///< element the loop listens to (match the assembler's)
    std::mutex trim_mtx;            ///< guards the four vectors below (REST getter thread)
    std::vector<double> trim;       ///< per-PRN cp trim, chips (applied cp = model + trim)
    std::vector<double> trim_disc;  ///< last applied discriminator, diagnostics
    std::vector<double> trim_q;     ///< last quality, diagnostics
    std::vector<long long> trim_n;  ///< updates applied, diagnostics
    uint64_t trim_frames = 0;       ///< frames processed (rate-limits the log line)

    /// One in-flight E/P/L readback: host landing zone + the event that says it is real.
    /// One slot per GPU frame slot; the pipeline depth guarantees a slot's previous use has
    /// completed long before it is reused.
    struct TrimSlot {
        cudaEvent_t ev = nullptr;
        double* host = nullptr;    ///< pinned, [rows][n_chan] x (re, im) of the ref element
        std::vector<int> job0;     ///< [n_rec*n_prn] global row base this frame, -1 if idle
        int n_rows = 0;
        bool pending = false;
    };
    std::vector<TrimSlot> trim_slots; ///< lazily sized by the first command instance
    std::mutex trim_slot_mtx;

    std::unique_ptr<gnss::ChannelizedReplicaBank> replica;
    std::unique_ptr<GnssCudaDespread> despread;
    std::vector<int> covering; ///< local channel indices this signal occupies (0..n_chan-1)
};

/**
 * @class cudaGnssChordTrack
 * @brief Despread N antennas against M references, per record window.
 *
 * ⚠️ RELATIONSHIP TO cudaGnssTrack -- READ BEFORE MERGING GNSS-SIDE WORK.
 *
 * This is a SEPARATE command, not a mode of @ref cudaGnssTrack, by explicit decision. It
 * duplicates part of that stage's role, so the two can drift, and a fix to one may need
 * mirroring in the other. What is shared and what is not:
 *
 *   SHARED (single implementation -- changes propagate for free):
 *     * replica synthesis          cudaGnssReplicaDevice.cuh
 *     * job construction           GnssCudaDespread::build_jobs
 *     * the record schema          gnssRecord.hpp
 *     * the output frame layout    gnssGpuChain.hpp
 *     * the broker seed contract   /set_seeds, same JSON as cudaGnssTrack
 *
 *   DUPLICATED (must be mirrored by hand if changed there):
 *     * per-record seed -> Spec construction and code-phase extrapolation
 *     * the PrnCtl / FrameHdr control block the assembler reads
 *
 *   PORTED KNOWINGLY (2026-07-31):
 *     * The DLL CODE TRIM (config `code_trim`) -- but IN-TRACKER, not the broker's version.
 *       Airspy's code loop lives in the broker at ~1 Hz off the combiner's E/L; CHORD's clock
 *       chain breathes +-1 chip / ~20 s and only per-frame closure follows it. Same
 *       discriminator and leaky-integrator math, run here on this command's own E/P/L rows.
 *
 *   DELIBERATELY ABSENT here, and why:
 *     * The FROZEN-SEED machinery (hold-on-lock, anchor ageing, the snap-to-model fence and
 *       the code-currency f_ref re-pin). The airspy node retired all of it in favour of
 *       model-primary seeding (`--dop-continuous`), where the seed follows the BRDC model
 *       every cycle and the re-pin is free. Starting there rather than reproducing the state
 *       machine it replaced is the point; if CHORD ever needs the fence, port it knowingly.
 *     * The VOLTAGE PEEL. Deferred until acq/track and beam mapping are proven -- a single-PRN
 *       replica is likely sub-quantization at 4+4b. rows_spec is therefore always 4.
 *     * The channel-major device RING. The CHORD correlator reads the tap's frame in its native
 *       [hop][chan][elem] order, so there is nothing to transpose into.
 *
 * RECORD LENGTH. hops_per_record defaults to 2048 (10.49 ms at CHORD's 5.12 us hop), which
 * divides the tap's 8192-hop frame exactly 4 ways. TWO LESSONS bought on sky (2026-07-31),
 * both consequences of records NOT being code-period aligned (airspy's are: 1 ms exactly):
 *   1. The extrapolation must add the NOMINAL code advance (52.3776 chips/hop) between
 *      records -- airspy's residual-only formula is correct there ONLY because its nominal
 *      advance mod L is zero per record. See the ABSOLUTE EXTRAPOLATION comment in the cpp.
 *   2. A 10.49 ms record spans ~10 NH20 overlay chips (1 ms EACH -- the 20 ms figure is the
 *      SEQUENCE period, not the transition spacing), whose +-1 partial sums mostly cancel:
 *      the bare-primary record of a snr-40 satellite despreads to noise. Trackers must use
 *      GPS_L5_Q_NH (overlay baked into a 204600-chip code, 20 ms period) -- with which the
 *      code-period boundary IS the overlay boundary, one per record at most, and the
 *      P_HEAD/m_head machinery handles exactly that case as designed.
 *
 * @conf prns, n_channels, n_elements, elem_stride, frame_chan_stride
 * @conf hops_per_record   default 2048
 * @conf signal            gnssSignal.hpp name, e.g. GPS_L5_Q
 * @conf sample_rate       pre-channelization, 3.2e9 on CHORD
 * @conf seed_endpoint     default "/chord_track/set_seeds"
 * @conf code_trim         default false: in-tracker DLL code trim (see the state class)
 * @conf trim_gain, trim_leak, trim_clamp, trim_quality_min, trim_ref_elem
 * @conf trim_endpoint     default "/chord_track/get_trim"
 */
class cudaGnssChordTrack : public cudaCommand {
public:
    cudaGnssChordTrack(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                       int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssChordTrack() override = default;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    cudaGnssChordTrackState* st();

    std::string _gpu_mem_input, _gpu_mem_output;
    std::string _mem_jobs, _mem_wave, _mem_scale, _mem_chanids;
    size_t _in_frame_len = 0, _out_frame_len = 0;
    std::vector<char> _ctl_stage; ///< host staging for the FrameHdr + PrnCtl control block
    bool _uploaded_static = false;
};

#endif // CUDA_GNSS_CHORD_TRACK_HPP
