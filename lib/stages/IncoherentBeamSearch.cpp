#include "Config.hpp"
#include "IncoherentSearch.hpp"
#include "NDArray.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"
#include "prometheusMetrics.hpp"
#include "timeUtil.hpp"
#include "visUtil.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <json.hpp>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using kotekan::prometheus::Metrics;

/**
 * @class IncoherentBeamSearch
 * @brief Search the incoherent-beam power of all X-engine halves for dispersed pulses.
 *
 * An input frame is the power of one half (IncoherentBeamPower) for a run of time samples at
 * that half's frequencies. The frames of all halves arrive in one buffer, in any order. Their
 * samples are filed by FPGA time and frequency id, and a time sample goes to the search
 * (IncoherentSearch) once it is assembly_delay_s older than the newest sample received. A
 * channel that has not arrived by then is missing in that sample, and a frame that arrives
 * later is dropped, as is a frame off the first frame's sample grid.
 *
 * A frame more than restart_gap_s from the samples being assembled is dropped too. A run of
 * such frames means the F-engine or every sender has restarted, and restarts the search at
 * the newest of them; frames of one sender with a wrong time never make such a run.
 *
 * Each candidate is logged, and appended to candidate_file as a line of JSON: the FPGA sequence
 * number of its arrival at the top channel (and its UNIX time, with GPS time), its S/N, DM,
 * width in samples, the number of samples above threshold merged into it, the S/N in each
 * of 16 blocks of channels along its dispersion curve, and the number of channels with data
 * there. A candidate is written once the highest DM has been searched for its arrival: the
 * sweep of dm_max after it reached the top channel.
 *
 * The line's "flags" list every reason the candidate is not one to act on, and is empty when
 * there is none:
 *  - "undispersed": it is one of several over a wide range of DM that fit one arrival without
 *    dispersion, which is interference in part of the band ("partners" is the number of the
 *    others). Such a candidate is in the file but is not counted or logged as one.
 *  - "low_dm": its DM is below trigger_min_dm.
 *  - "low_snr": its S/N is below trigger_min_snr.
 *  - "one_band": it is 8 samples or wider and one of 16 blocks of channels has more than 0.8
 *    times its S/N, which is how interference a few MHz wide looks. A wide burst in a narrow
 *    band looks the same.
 *
 * "part_snr" is the candidate's highest S/N in a contiguous part of the band and "part_ratio"
 * its ratio to the whole band (IncoherentCandidate::part_snr): about 1 for noise, below 1 for
 * interference a few MHz wide, above 1 for a pulse that fills part of the band.
 *
 * Nothing acts on a candidate yet.
 *
 * With out_buf, every search pass also writes its dedispersed sums (IncoherentSearch::last_sums)
 * as one frame, for tests: 13 MB/s with the default DMs. Row k of a frame holds, for each trial
 * DM, the sum along the dispersion curve that reaches the bottom channel at the frame's
 * fpga_seq_num + k time samples. dm_file lists the trial DMs and each one's sweep from the top
 * channel to the bottom, which gives the time at the top. The search waits for out_buf.
 *
 * With snr_buf, every search pass also writes an image for monitoring: the highest S/N of each
 * trial DM (IncoherentSearch::last_max_snr) in each of the last passes, one row per pass, the
 * newest last. The buffer is meant to be peeked (peek_hold), not consumed.
 *
 * With cutout_buf, a candidate without a flag, or one of at least cutout_flagged_min_snr that is
 * not "undispersed", also gets a frame of the samples around it as
 * the search saw them (IncoherentSearch::cutout): each channel's power less its running mean,
 * in units of 1/32 of its sigma, less the average of the channels at each time, 0 where the
 * channel was masked. The channels are shifted by the candidate's DM, so that a pulse is a
 * column in the middle of the frame, and the frame's fpga_seq_num is the first column at the
 * top channel. The candidate's line gets "cutout", the number of its frame since the stage
 * started, which is the number hdf5FileWrite puts in the file name. The search does not wait
 * for cutout_buf: when it is full the candidate has no cutout.
 *
 * With power_buf, the power of every time sample also goes out as the search received it, before
 * any mask, in frames of 256 samples: 2 bytes per channel and sample, 10 MB/s for 6656 channels.
 * The search does not wait for power_buf: when it is full the frame is left out, which leaves a
 * gap in the stream.
 *
 * With waterfall_buf, every search pass also writes an image for a live display: the samples
 * as the search saw them (IncoherentSearch::waterfall), added up in bins of waterfall_time_bin
 * samples by waterfall_freq_bin channels, one row per time bin over the last seconds, the newest
 * last. A value is in units of 1/16 sigma of its bin and -128 marks a bin with nothing in use.
 * The newest row ends one pass before the newest sample: a channel's mask also takes the samples
 * shortly before it, so the last pass can still change. The frame's fpga_seq_num is its first
 * row, so a reader that asks more often than the image is long can put the frames together
 * without a gap. The buffer is meant to be peeked (peek_hold), not consumed.
 *
 * The search runs in this stage's thread and its OpenMP threads, between frames. in_buf must
 * hold the frames that arrive during one search pass (0.3 s).
 *
 * @par Buffers
 * @buffer in_buf   Power frames of all halves.
 *     @buffer_format   NDArray uint16 [Trfi, F]; 0 marks a missing value.
 *     @buffer_metadata chordMetadata
 * @buffer snr_buf  Optional. Highest S/N per trial DM in the last search passes.
 *     @buffer_format   NDArray float32 [Tpass, DM]: any number of passes (0.34 s each) by the
 *                      number of trial DMs; 0 where there is no value.
 *     @buffer_metadata chordMetadata
 * @buffer out_buf  Optional. Dedispersed sums, one frame per search pass.
 *     @buffer_format   NDArray float32 [Trfi, DM], 256 time samples by the number of trial DMs
 *                      (logged at start); 0 where there is no sum.
 *     @buffer_metadata chordMetadata
 * @buffer cutout_buf  Optional. The samples around each candidate without a flag, and around
 *                     those of cutout_flagged_min_snr.
 *     @buffer_format   NDArray int8 [F, Trfi]: the channels searched by 256 time samples.
 *     @buffer_metadata chordMetadata
 * @buffer power_buf  Optional. The power of every time sample, one frame per 256 samples.
 *     @buffer_format   NDArray uint16 [Trfi, F]: 256 time samples by the channels searched;
 *                      0 marks a missing value.
 *     @buffer_metadata chordMetadata
 * @buffer waterfall_buf  Optional. The samples of the last seconds, binned for a display.
 *     @buffer_format   NDArray int8 [Trfi, F]: any number of passes, 256 / waterfall_time_bin
 *                      rows each, by the channels searched over waterfall_freq_bin, rounded up;
 *                      -128 where there is no value.
 *     @buffer_metadata chordMetadata
 *
 * @conf freq_id_base     Int. FPGA frequency id of the lowest channel searched.
 * @conf num_local_freq    Int. Number of channels searched, upward from freq_id_base.
 * @conf num_feeds         Int. Feeds summed in the power; sets the expected noise.
 * @conf dm_min            Double. Lowest trial DM, pc/cm^3. Default 100.
 * @conf dm_max            Double. Highest trial DM. Default 1500.
 * @conf dm_knee           Double. The step between trial DMs is dm_step_min up to this DM and
 *                         grows above it. Default dm_min.
 * @conf dm_step_min       Double. Step between trial DMs up to dm_knee. Default 0.12.
 * @conf dm_step_max       Double. Step at dm_max; from dm_knee it grows as a power of the DM.
 *                         Default 0.56. With 1.31 ms samples the defaults keep 94 % of the
 *                         S/N of a 1 ms pulse on average.
 * @conf dm_file           String. File that gets the trial DMs and their sweeps in samples, as
 *                         JSON: {"dm": [...], "sweep_samples": [...]}. Default none.
 * @conf mask_bands_mhz    List of [low, high]. Frequency ranges that are never searched.
 * @conf mask_freq_ids     List of int. FPGA frequency ids that are never searched.
 * @conf threshold         Double. S/N above which a candidate is reported. Default 8.
 * @conf trigger_min_snr   Double. A candidate below this S/N gets the flag "low_snr".
 *                         Default 7.5.
 * @conf trigger_min_dm    Double. A candidate below this DM gets the flag "low_dm".
 *                         Default 100.
 * @conf cutout_flagged_min_snr  Double. A candidate of at least this S/N gets a cutout with any
 *                         flag but "undispersed". Default 0: none does.
 * @conf cutout_exclude    List of [DM low, DM high, ERA low, ERA high]. No cutout for a candidate
 *                         with its DM in the range and the Earth rotation angle of its arrival
 *                         (degrees, with UT1 taken as UTC) in the range, which may pass through
 *                         360: a known pulsar in the beam. Needs GPS time. Default none.
 * @conf waterfall_time_bin  Int. Time samples per row of waterfall_buf; divides 256. Default 32.
 * @conf waterfall_freq_bin  Int. Channels per column of waterfall_buf. Default 8.
 * @conf holdoff_s         Double. No candidates for this long after a (re)start. Default 120.
 * @conf tau_s             Double. Time constant of the running statistics. Default 5.
 * @conf num_threads       Int. OpenMP threads of the search. Default 4.
 * @conf assembly_delay_s  Double. How long a time sample waits for all halves. Default 1.
 * @conf restart_gap_s     Double. Frames this far from the assembled samples are dropped, and
 *                         a run of them restarts the search. Default 10.
 * @conf candidate_file    String. File the candidates are appended to. Default none.
 *
 * @metric kotekan_incoherent_beam_search_candidates_total   Candidates reported.
 * @metric kotekan_incoherent_beam_search_undispersed_total   Candidates left out as undispersed.
 * @metric kotekan_incoherent_beam_search_cutouts_skipped_total   Candidates without a cutout
 *         because cutout_buf was full.
 * @metric kotekan_incoherent_beam_search_power_skipped_total   Frames left out because power_buf
 *         was full.
 * @metric kotekan_incoherent_beam_search_max_snr  Highest S/N in the last 32 search passes.
 * @metric kotekan_incoherent_beam_search_channels_used   Channels used in the newest sample.
 * @metric kotekan_incoherent_beam_search_dropped_frames_total   Frames late, far or off the grid.
 * @metric kotekan_incoherent_beam_search_restarts_total   Restarts of the search.
 */
class IncoherentBeamSearch : public kotekan::Stage {
public:
    IncoherentBeamSearch(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffers) :
        Stage(config, unique_name, buffers, std::bind(&IncoherentBeamSearch::main_thread, this)),
        in_buf(get_buffer("in_buf")),
        out_buf(config.exists(unique_name, "out_buf") ? get_buffer("out_buf") : nullptr),
        snr_buf(config.exists(unique_name, "snr_buf") ? get_buffer("snr_buf") : nullptr),
        cutout_buf(config.exists(unique_name, "cutout_buf") ? get_buffer("cutout_buf") : nullptr),
        power_buf(config.exists(unique_name, "power_buf") ? get_buffer("power_buf") : nullptr),
        waterfall_buf(config.exists(unique_name, "waterfall_buf") ? get_buffer("waterfall_buf")
                                                                  : nullptr),
        waterfall_time_bin(config.get_default<int>(unique_name, "waterfall_time_bin", 32)),
        waterfall_freq_bin(config.get_default<int>(unique_name, "waterfall_freq_bin", 8)),
        freq_id_base(config.get<int>(unique_name, "freq_id_base")),
        candidates_counter(Metrics::instance().add_counter(
            "kotekan_incoherent_beam_search_candidates_total", unique_name)),
        undispersed_counter(Metrics::instance().add_counter(
            "kotekan_incoherent_beam_search_undispersed_total", unique_name)),
        cutouts_skipped_counter(Metrics::instance().add_counter(
            "kotekan_incoherent_beam_search_cutouts_skipped_total", unique_name)),
        power_skipped_counter(Metrics::instance().add_counter(
            "kotekan_incoherent_beam_search_power_skipped_total", unique_name)),
        max_snr_gauge(
            Metrics::instance().add_gauge("kotekan_incoherent_beam_search_max_snr", unique_name)),
        channels_used_gauge(Metrics::instance().add_gauge(
            "kotekan_incoherent_beam_search_channels_used", unique_name)),
        dropped_frames_counter(Metrics::instance().add_counter(
            "kotekan_incoherent_beam_search_dropped_frames_total", unique_name)),
        restarts_counter(Metrics::instance().add_counter(
            "kotekan_incoherent_beam_search_restarts_total", unique_name)) {
        in_buf->register_consumer(unique_name);

        in_desc = in_buf->require_frame_desc<kotekan::GenericNDArray>();
        const std::vector<std::ptrdiff_t> extents = in_desc->get_extents();
        if (in_desc->get_value_datatype() != kotekan::uint16 || extents.size() != 2)
            FATAL_ERROR("IncoherentBeamSearch: in_buf {:s} must be uint16 [Trfi, F]",
                        in_buf->buffer_name);
        frame_samples = extents[0];
        frame_freqs = extents[1];
        period = in_desc->get_dimscaling(0);
        if (period <= 0)
            FATAL_ERROR("IncoherentBeamSearch: dimension 0 of in_buf {:s} has no time scaling",
                        in_buf->buffer_name);

        const Telescope& tel = Telescope::instance();
        params.num_freq = config.get<int>(unique_name, "num_local_freq");
        params.freq0_mhz = tel.to_freq_MHz((freq_id_t)freq_id_base);
        params.dfreq_mhz = tel.to_freq_MHz((freq_id_t)(freq_id_base + 1)) - params.freq0_mhz;
        if (params.dfreq_mhz <= 0)
            FATAL_ERROR("IncoherentBeamSearch: frequency must rise with the frequency id");
        params.sample_s = period * tel.seq_length_nsec() * 1e-9;
        params.thermal = 1 / std::sqrt((double)period * config.get<int>(unique_name, "num_feeds"));
        const double dm_min = config.get_default<double>(unique_name, "dm_min", 100.0);
        const double dm_max = config.get_default<double>(unique_name, "dm_max", 1500.0);
        const double step_min = config.get_default<double>(unique_name, "dm_step_min", 0.12);
        const double step_max = config.get_default<double>(unique_name, "dm_step_max", 0.56);
        const double dm_knee = config.get_default<double>(unique_name, "dm_knee", dm_min);
        if (dm_min <= 0 || dm_knee < dm_min || dm_max <= dm_knee || step_min <= 0
            || step_max < step_min)
            FATAL_ERROR("IncoherentBeamSearch: need 0 < dm_min <= dm_knee < dm_max and 0 < "
                        "dm_step_min <= dm_step_max");
        // The step is dm_step_min up to dm_knee, and grows from there as a power of the DM to
        // dm_step_max at dm_max.
        const double growth = std::log(step_max / step_min) / std::log(dm_max / dm_knee);
        for (double dm = dm_min; dm <= dm_max;
             dm += step_min * std::pow(std::max(dm / dm_knee, 1.0), growth))
            params.dms.push_back(dm);
        INFO("IncoherentBeamSearch: {:d} trial DMs", params.dms.size());
        if (out_buf != nullptr) {
            out_buf->register_producer(unique_name);
            out_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::float32, "incoherent_beam_sums",
                {IncoherentSearch::chunk, (std::ptrdiff_t)params.dms.size()}, {"Trfi", "DM"},
                {period, 1}));
            out_desc = out_buf->get_frame_desc<kotekan::GenericNDArray>();
            params.keep_sums = true;
        }
        if (snr_buf != nullptr) {
            // The frame holds as many passes as it has room for.
            const size_t row_bytes = params.dms.size() * sizeof(float);
            if (snr_buf->frame_size == 0 || snr_buf->frame_size % row_bytes != 0)
                FATAL_ERROR("IncoherentBeamSearch: snr_buf {:s} must hold whole rows of {:d} "
                            "float32, one per trial DM",
                            snr_buf->buffer_name, params.dms.size());
            snr_buf->register_producer(unique_name);
            snr_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::float32, "incoherent_beam_snr",
                {(std::ptrdiff_t)(snr_buf->frame_size / row_bytes),
                 (std::ptrdiff_t)params.dms.size()},
                {"Tpass", "DM"}, {IncoherentSearch::chunk * period, 1}));
            snr_desc = snr_buf->get_frame_desc<kotekan::GenericNDArray>();
            snr_image.assign(snr_buf->frame_size / sizeof(float), 0.0f);
        }
        if (cutout_buf != nullptr) {
            cutout_buf->register_producer(unique_name);
            cutout_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::int8, "incoherent_beam_cutout",
                {params.num_freq, IncoherentSearch::cutout_samples}, {"F", "Trfi"}, {1, period}));
            cutout_desc = cutout_buf->get_frame_desc<kotekan::GenericNDArray>();
        }
        if (power_buf != nullptr) {
            power_buf->register_producer(unique_name);
            power_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::uint16, "incoherent_beam_power",
                {IncoherentSearch::chunk, params.num_freq}, {"Trfi", "F"}, {period, 1}));
            power_desc = power_buf->get_frame_desc<kotekan::GenericNDArray>();
            power_frame.assign((size_t)IncoherentSearch::chunk * params.num_freq, 0);
        }
        if (waterfall_buf != nullptr) {
            if (waterfall_time_bin < 1 || IncoherentSearch::chunk % waterfall_time_bin != 0
                || waterfall_freq_bin < 1)
                FATAL_ERROR("IncoherentBeamSearch: waterfall_time_bin must divide {:d} and "
                            "waterfall_freq_bin must be at least 1",
                            IncoherentSearch::chunk);
            // The frame holds as many passes as it has room for.
            waterfall_columns = (params.num_freq + waterfall_freq_bin - 1) / waterfall_freq_bin;
            const size_t pass_bytes =
                (size_t)(IncoherentSearch::chunk / waterfall_time_bin) * waterfall_columns;
            if (waterfall_buf->frame_size == 0 || waterfall_buf->frame_size % pass_bytes != 0)
                FATAL_ERROR("IncoherentBeamSearch: waterfall_buf {:s} must hold whole passes of "
                            "{:d} rows by {:d} int8",
                            waterfall_buf->buffer_name,
                            IncoherentSearch::chunk / waterfall_time_bin, waterfall_columns);
            waterfall_buf->register_producer(unique_name);
            waterfall_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::int8, "incoherent_waterfall",
                {(std::ptrdiff_t)(waterfall_buf->frame_size / waterfall_columns),
                 (std::ptrdiff_t)waterfall_columns},
                {"Trfi", "F"}, {waterfall_time_bin * period, waterfall_freq_bin}));
            waterfall_desc = waterfall_buf->get_frame_desc<kotekan::GenericNDArray>();
            waterfall_image.assign(waterfall_buf->frame_size, IncoherentSearch::waterfall_no_data);
        }
        params.mask_bands_mhz = config.get_default<std::vector<std::pair<double, double>>>(
            unique_name, "mask_bands_mhz", {});
        for (int id : config.get_default<std::vector<int>>(unique_name, "mask_freq_ids", {})) {
            const double f = tel.to_freq_MHz((freq_id_t)id);
            params.mask_bands_mhz.push_back({f - params.dfreq_mhz / 4, f + params.dfreq_mhz / 4});
        }
        params.threshold = config.get_default<double>(unique_name, "threshold", params.threshold);
        trigger_min_snr = config.get_default<double>(unique_name, "trigger_min_snr", 7.5);
        trigger_min_dm = config.get_default<double>(unique_name, "trigger_min_dm", 100.0);
        cutout_flagged_min_snr =
            config.get_default<double>(unique_name, "cutout_flagged_min_snr", 0.0);
        cutout_exclude =
            config.get_default<std::vector<std::vector<double>>>(unique_name, "cutout_exclude", {});
        for (const std::vector<double>& e : cutout_exclude)
            if (e.size() != 4 || e[0] > e[1])
                FATAL_ERROR("IncoherentBeamSearch: an entry of cutout_exclude must be [DM low, "
                            "DM high, ERA low, ERA high]");
        params.holdoff_s = config.get_default<double>(unique_name, "holdoff_s", params.holdoff_s);
        params.tau_s = config.get_default<double>(unique_name, "tau_s", params.tau_s);
        params.num_threads =
            config.get_default<int>(unique_name, "num_threads", params.num_threads);

        delay_samples = std::llround(
            config.get_default<double>(unique_name, "assembly_delay_s", 1.0) / params.sample_s);
        restart_gap = std::llround(config.get_default<double>(unique_name, "restart_gap_s", 10.0)
                                   / params.sample_s);
        if (delay_samples < frame_samples || restart_gap <= delay_samples)
            FATAL_ERROR("IncoherentBeamSearch: assembly_delay_s must cover a frame and be below "
                        "restart_gap_s");
        ring_len = delay_samples + 2 * frame_samples;
        ring.assign((size_t)ring_len * params.num_freq, 0);
        power.assign(params.num_freq, 0.0f);
        try {
            search = std::make_unique<IncoherentSearch>(params);
        } catch (const std::invalid_argument& e) {
            FATAL_ERROR("IncoherentBeamSearch: {:s}", e.what());
        }

        const std::string path = config.get_default<std::string>(unique_name, "candidate_file", "");
        if (!path.empty()) {
            std::error_code ignored; // a failure shows when the file is opened
            std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ignored);
            candidate_file.open(path, std::ios::app);
            if (!candidate_file)
                FATAL_ERROR("IncoherentBeamSearch: cannot open candidate_file {:s}", path);
        }

        const std::string dm_path = config.get_default<std::string>(unique_name, "dm_file", "");
        if (!dm_path.empty()) {
            std::vector<int64_t> sweeps;
            for (int d = 0; d < (int)params.dms.size(); ++d)
                sweeps.push_back(search->sweep_samples(d));
            std::error_code ignored;
            std::filesystem::create_directories(std::filesystem::path(dm_path).parent_path(),
                                                ignored);
            std::ofstream dm_file(dm_path);
            dm_file << nlohmann::json({{"dm", params.dms}, {"sweep_samples", sweeps}}).dump()
                    << std::endl;
            if (!dm_file)
                FATAL_ERROR("IncoherentBeamSearch: cannot write dm_file {:s}", dm_path);
        }
    }

    void main_thread() override {
        frameID in_id(in_buf);
        bool started = false;
        int far_run = 0;

        while (!stop_thread) {
            const uint8_t* frame = in_buf->wait_for_full_frame(unique_name, in_id);
            if (frame == nullptr)
                break;
            const auto meta = get_chord_metadata(in_buf, in_id);
            if (!meta)
                FATAL_ERROR("IncoherentBeamSearch: {:s}[{:d}] has no chordMetadata",
                            in_buf->buffer_name, (int)in_id);
            meta->check_frame_desc(in_desc);
            if (!meta->has_fpga_seq_num() || !meta->has_coarse_freq())
                FATAL_ERROR("IncoherentBeamSearch: {:s} frames need fpga_seq_num and coarse_freq",
                            in_buf->buffer_name);
            const std::vector<int> freq_ids = meta->get_coarse_freq();
            if ((std::ptrdiff_t)freq_ids.size() != frame_freqs)
                FATAL_ERROR("IncoherentBeamSearch: {:s} frame has {:d} coarse_freq for {:d} "
                            "frequencies",
                            in_buf->buffer_name, freq_ids.size(), frame_freqs);
            const int64_t seq = meta->get_fpga_seq_num();
            const int64_t first = seq / period;

            const bool far = started
                             && (first >= next_sample + restart_gap
                                 || first + frame_samples <= next_sample - restart_gap);
            far_run = far ? far_run + 1 : 0;
            if (!started || far_run >= far_frames_to_restart) {
                if (started) {
                    WARN("IncoherentBeamSearch: frames at FPGA seq {:d} are far from the samples "
                         "being assembled at {:d}; restarting the search",
                         seq, next_sample * period + phase);
                    restarts_counter.inc();
                }
                phase = seq % period;
                start_search(first);
                started = true;
                far_run = 0;
            } else if (far || seq % period != phase || first + frame_samples <= next_sample) {
                // Far, off the sample grid, or already searched.
                dropped_frames_counter.inc();
                in_buf->mark_frame_empty(unique_name, in_id++);
                continue;
            }

            // Make room, file the samples that are still open, and search what is old enough.
            while (first + frame_samples > next_sample + ring_len)
                search_next_sample();
            const uint16_t* const values = (const uint16_t*)frame;
            for (int64_t t = std::max<int64_t>(0, next_sample - first); t < frame_samples; ++t) {
                uint16_t* const row = &ring[(size_t)((first + t) % ring_len) * params.num_freq];
                for (std::ptrdiff_t f = 0; f < frame_freqs; ++f) {
                    const int c = freq_ids[f] - freq_id_base;
                    if (c >= 0 && c < params.num_freq)
                        row[c] = values[t * frame_freqs + f];
                }
            }
            newest_end = std::max(newest_end, first + frame_samples);
            in_buf->mark_frame_empty(unique_name, in_id++);
            while (next_sample < newest_end - delay_samples)
                search_next_sample();
        }
    }

private:
    // A run of this many frames far from the assembled samples restarts the search: several
    // frames from each of 16 halves.
    static constexpr int far_frames_to_restart = 64;

    // Makes `sample` the first time sample of a search without history.
    void start_search(int64_t sample) {
        if (search->num_samples() > 0) {
            search.reset(); // free the buffers before allocating them again
            search = std::make_unique<IncoherentSearch>(params);
        }
        origin = next_sample = newest_end = sample;
        power_rows = 0;
        std::fill(ring.begin(), ring.end(), 0);
        std::fill(recent_max.begin(), recent_max.end(), 0.0);
        std::fill(snr_image.begin(), snr_image.end(), 0.0f);
        std::fill(waterfall_image.begin(), waterfall_image.end(),
                  IncoherentSearch::waterfall_no_data);
    }

    void search_next_sample() {
        uint16_t* const row = &ring[(size_t)(next_sample % ring_len) * params.num_freq];
        for (int c = 0; c < params.num_freq; ++c)
            power[c] = row[c];
        if (power_buf != nullptr)
            std::copy(row, row + params.num_freq,
                      &power_frame[(size_t)power_rows * params.num_freq]);
        std::fill(row, row + params.num_freq, 0);
        ++next_sample;
        if (power_buf != nullptr && ++power_rows == IncoherentSearch::chunk)
            write_power();
        if (search->add_sample(power.data()))
            report();
    }

    // Writes the IncoherentSearch::chunk samples before next_sample to power_buf, unless it is
    // full.
    void write_power() {
        power_rows = 0;
        if (!power_buf->is_frame_empty(power_id)) {
            power_skipped_counter.inc();
            return;
        }
        uint8_t* const frame = power_buf->wait_for_empty_frame(unique_name, power_id);
        if (frame == nullptr)
            return;
        std::copy(power_frame.begin(), power_frame.end(), (uint16_t*)frame);

        power_buf->allocate_new_metadata_object(power_id);
        const auto meta = get_chord_metadata(power_buf, power_id);
        meta->set_from_frame_desc(power_desc);
        meta->set_fpga_seq_num((next_sample - IncoherentSearch::chunk) * period + phase);
        meta->set_time_downsampling_fpga(period);
        std::vector<int> freq_ids(params.num_freq);
        std::iota(freq_ids.begin(), freq_ids.end(), freq_id_base);
        meta->set_coarse_freq(freq_ids);
        power_buf->mark_frame_full(unique_name, power_id);
        power_id = (power_id + 1) % power_buf->num_frames;
    }

    // Writes the sums of the search pass that just ran to out_buf, as [time][DM].
    void write_sums() {
        uint8_t* const frame = out_buf->wait_for_empty_frame(unique_name, out_id);
        if (frame == nullptr)
            return;
        const std::vector<float>& sums = search->last_sums(); // [DM][time]
        const size_t num_dm = params.dms.size();
        float* const out = (float*)frame;
        for (size_t d = 0; d < num_dm; ++d)
            for (size_t k = 0; k < IncoherentSearch::chunk; ++k)
                out[k * num_dm + d] = sums[d * IncoherentSearch::chunk + k];

        out_buf->allocate_new_metadata_object(out_id);
        const auto meta = get_chord_metadata(out_buf, out_id);
        meta->set_from_frame_desc(out_desc);
        const int64_t first = origin + search->num_samples() - IncoherentSearch::chunk;
        meta->set_fpga_seq_num(first * period + phase);
        meta->set_time_downsampling_fpga(period);
        out_buf->mark_frame_full(unique_name, out_id);
        out_id = (out_id + 1) % out_buf->num_frames;
    }

    // Appends the search pass that just ran to the image of recent passes and writes the image
    // to snr_buf.
    void write_snr() {
        const std::vector<float>& newest = search->last_max_snr();
        std::copy(snr_image.begin() + newest.size(), snr_image.end(), snr_image.begin());
        std::copy(newest.begin(), newest.end(), snr_image.end() - newest.size());

        uint8_t* const frame = snr_buf->wait_for_empty_frame(unique_name, snr_id);
        if (frame == nullptr)
            return;
        std::copy(snr_image.begin(), snr_image.end(), (float*)frame);
        snr_buf->allocate_new_metadata_object(snr_id);
        const auto meta = get_chord_metadata(snr_buf, snr_id);
        meta->set_from_frame_desc(snr_desc);
        // The first row is the pass that ended this many passes ago.
        const int64_t rows = snr_image.size() / newest.size();
        const int64_t first = origin + search->num_samples() - rows * IncoherentSearch::chunk;
        meta->set_fpga_seq_num(first * period + phase);
        meta->set_time_downsampling_fpga(IncoherentSearch::chunk * period);
        snr_buf->mark_frame_full(unique_name, snr_id);
        snr_id = (snr_id + 1) % snr_buf->num_frames;
    }

    // Appends the pass before the one that just ran to the image of recent samples and writes
    // the image to waterfall_buf.
    void write_waterfall() {
        const int64_t end = search->num_samples() - IncoherentSearch::chunk;
        if (end < IncoherentSearch::chunk)
            return;
        const int rows = IncoherentSearch::chunk / waterfall_time_bin;
        const size_t pass_bytes = (size_t)rows * waterfall_columns;
        std::copy(waterfall_image.begin() + pass_bytes, waterfall_image.end(),
                  waterfall_image.begin());
        search->waterfall(end - IncoherentSearch::chunk, rows, waterfall_time_bin,
                          waterfall_freq_bin,
                          &waterfall_image[waterfall_image.size() - pass_bytes]);

        uint8_t* const frame = waterfall_buf->wait_for_empty_frame(unique_name, waterfall_id);
        if (frame == nullptr)
            return;
        std::copy(waterfall_image.begin(), waterfall_image.end(), (int8_t*)frame);
        waterfall_buf->allocate_new_metadata_object(waterfall_id);
        const auto meta = get_chord_metadata(waterfall_buf, waterfall_id);
        meta->set_from_frame_desc(waterfall_desc);
        const int64_t image_rows = waterfall_image.size() / waterfall_columns;
        const int64_t first = origin + end - image_rows * waterfall_time_bin;
        meta->set_fpga_seq_num(first * period + phase);
        meta->set_time_downsampling_fpga(waterfall_time_bin * period);
        waterfall_buf->mark_frame_full(unique_name, waterfall_id);
        waterfall_id = (waterfall_id + 1) % waterfall_buf->num_frames;
    }

    // True when an entry of cutout_exclude holds the DM and the Earth rotation angle of the
    // arrival at FPGA sequence number seq.
    bool cutout_excluded(double dm, int64_t seq) const {
        const Telescope& tel = Telescope::instance();
        if (cutout_exclude.empty() || !tel.gps_time_enabled())
            return false;
        const double era = get_ERA_from_time(tel.to_time(seq), 0.0);
        for (const std::vector<double>& e : cutout_exclude) {
            const bool in_era =
                e[2] <= e[3] ? era >= e[2] && era <= e[3] : era >= e[2] || era <= e[3];
            if (dm >= e[0] && dm <= e[1] && in_era)
                return true;
        }
        return false;
    }

    // Writes the samples around a candidate to cutout_buf and returns the number of the frame
    // since the stage started, or -1 when the buffer is full.
    int64_t write_cutout(const IncoherentCandidate& c) {
        if (!cutout_buf->is_frame_empty(cutout_id)) {
            cutouts_skipped_counter.inc();
            return -1;
        }
        uint8_t* const frame = cutout_buf->wait_for_empty_frame(unique_name, cutout_id);
        if (frame == nullptr)
            return -1;
        search->cutout(c, (int8_t*)frame);

        cutout_buf->allocate_new_metadata_object(cutout_id);
        const auto meta = get_chord_metadata(cutout_buf, cutout_id);
        meta->set_from_frame_desc(cutout_desc);
        const int64_t first = origin + c.sample - IncoherentSearch::cutout_samples / 2;
        meta->set_fpga_seq_num(first * period + phase);
        meta->set_time_downsampling_fpga(period);
        std::vector<int> freq_ids(params.num_freq);
        std::iota(freq_ids.begin(), freq_ids.end(), freq_id_base);
        meta->set_coarse_freq(freq_ids);
        cutout_buf->mark_frame_full(unique_name, cutout_id);
        cutout_id = (cutout_id + 1) % cutout_buf->num_frames;
        return num_cutouts++;
    }

    // Called after each search pass.
    void report() {
        if (out_buf != nullptr)
            write_sums();
        if (snr_buf != nullptr)
            write_snr();
        if (waterfall_buf != nullptr)
            write_waterfall();
        recent_max[num_reports++ % recent_max.size()] = search->take_max_snr();
        max_snr_gauge.set(*std::max_element(recent_max.begin(), recent_max.end()));
        channels_used_gauge.set(search->channels_used());

        const Telescope& tel = Telescope::instance();
        for (const IncoherentCandidate& c : search->take_candidates()) {
            const int64_t seq = (origin + c.sample) * period + phase;
            if (c.undispersed) {
                undispersed_counter.inc();
            } else {
                candidates_counter.inc();
                INFO("IncoherentBeamSearch: candidate S/N {:.1f}, DM {:.1f}, width {:d} samples, "
                     "FPGA seq {:d}",
                     c.snr, c.dm, c.width, seq);
            }
            std::vector<std::string> flags;
            if (c.undispersed)
                flags.push_back("undispersed");
            if (c.dm < trigger_min_dm)
                flags.push_back("low_dm");
            if (c.snr < trigger_min_snr)
                flags.push_back("low_snr");
            if (c.one_band)
                flags.push_back("one_band");
            const bool bright_flagged =
                cutout_flagged_min_snr > 0 && c.snr >= cutout_flagged_min_snr && !c.undispersed;
            const int64_t cutout = cutout_buf != nullptr && (flags.empty() || bright_flagged)
                                           && !cutout_excluded(c.dm, seq)
                                       ? write_cutout(c)
                                       : -1;
            if (!candidate_file.is_open())
                continue;
            nlohmann::json line = {{"fpga_seq", seq},
                                   {"snr", c.snr},
                                   {"dm", c.dm},
                                   {"width", c.width},
                                   {"num_hits", c.num_hits},
                                   {"block_snr", c.block_snr},
                                   {"num_channels", c.num_channels},
                                   {"flags", flags},
                                   {"partners", c.num_partners},
                                   {"one_band", c.one_band},
                                   {"part_snr", c.part_snr},
                                   {"part_ratio", c.part_ratio},
                                   {"reported_unix", current_time()}};
            if (cutout >= 0)
                line["cutout"] = cutout;
            if (tel.gps_time_enabled())
                line["unix_time"] = ts_to_double(tel.to_time(seq));
            candidate_file << line.dump() << std::endl;
        }
    }

    Buffer* const in_buf;
    Buffer* const out_buf; // nullptr when the sums are not written
    Buffer* const snr_buf; // nullptr when the image of recent passes is not written
    std::shared_ptr<const kotekan::GenericNDArray> snr_desc;
    std::vector<float> snr_image; // [pass][DM], the newest pass last
    int snr_id = 0;
    Buffer* const cutout_buf; // nullptr when no cutouts are written
    std::shared_ptr<const kotekan::GenericNDArray> cutout_desc;
    int cutout_id = 0;
    int64_t num_cutouts = 0;
    Buffer* const power_buf; // nullptr when the power is not written
    std::shared_ptr<const kotekan::GenericNDArray> power_desc;
    std::vector<uint16_t> power_frame; // [time][channel]: the samples since the last frame
    int power_rows = 0;
    int power_id = 0;
    Buffer* const waterfall_buf; // nullptr when the image of recent samples is not written
    const int waterfall_time_bin;
    const int waterfall_freq_bin;
    int waterfall_columns = 0;
    std::shared_ptr<const kotekan::GenericNDArray> waterfall_desc;
    std::vector<int8_t> waterfall_image; // [time bin][column], the newest row last
    int waterfall_id = 0;
    const int freq_id_base;
    std::shared_ptr<const kotekan::GenericNDArray> in_desc;
    std::shared_ptr<const kotekan::GenericNDArray> out_desc;
    int out_id = 0;
    std::ptrdiff_t frame_samples = 0;
    std::ptrdiff_t frame_freqs = 0;
    int64_t period = 0; // FPGA samples per time sample
    IncoherentSearchParams params;
    std::unique_ptr<IncoherentSearch> search;

    // Time samples are numbered fpga_seq / period. Rows of `ring` hold the samples from
    // next_sample on, by sample number modulo ring_len; 0 marks a missing value.
    std::vector<uint16_t> ring;
    std::vector<float> power;
    int64_t ring_len = 0;
    int64_t delay_samples = 0;
    int64_t restart_gap = 0;
    double trigger_min_snr = 0;
    double trigger_min_dm = 0;
    double cutout_flagged_min_snr = 0;
    std::vector<std::vector<double>> cutout_exclude;
    int64_t phase = 0;       // fpga_seq modulo period of the sample grid
    int64_t origin = 0;      // the search's first sample
    int64_t next_sample = 0; // the next sample to search
    int64_t newest_end = 0;  // one past the newest sample received
    std::vector<double> recent_max = std::vector<double>(32, 0.0);
    size_t num_reports = 0;

    std::ofstream candidate_file;
    kotekan::prometheus::Counter& candidates_counter;
    kotekan::prometheus::Counter& undispersed_counter;
    kotekan::prometheus::Counter& cutouts_skipped_counter;
    kotekan::prometheus::Counter& power_skipped_counter;
    kotekan::prometheus::Gauge& max_snr_gauge;
    kotekan::prometheus::Gauge& channels_used_gauge;
    kotekan::prometheus::Counter& dropped_frames_counter;
    kotekan::prometheus::Counter& restarts_counter;
};

REGISTER_KOTEKAN_STAGE(IncoherentBeamSearch);
