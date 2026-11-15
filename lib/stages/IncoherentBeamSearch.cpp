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
#include "visUtil.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <json.hpp>
#include <memory>
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
 * width in samples, the number of samples above threshold merged into it, and the S/N in each
 * of 16 blocks of channels along its dispersion curve.
 *
 * The search runs in this stage's thread and its OpenMP threads, between frames. in_buf must
 * hold the frames that arrive during one search pass (0.3 s).
 *
 * @par Buffers
 * @buffer in_buf   Power frames of all halves.
 *     @buffer_format   NDArray uint16 [Trfi, F]; 0 marks a missing value.
 *     @buffer_metadata chordMetadata
 *
 * @conf freq_id_base      Int. FPGA frequency id of the lowest channel searched.
 * @conf num_local_freq    Int. Number of channels searched, upward from freq_id_base.
 * @conf num_feeds         Int. Feeds summed in the power; sets the expected noise.
 * @conf dm_min            Double. Lowest trial DM, pc/cm^3. Default 100.
 * @conf dm_max            Double. Highest trial DM. Default 1500.
 * @conf dm_step           Double. Step between trial DMs as a fraction of the DM. Default 0.001.
 * @conf mask_bands_mhz    List of [low, high]. Frequency ranges that are never searched.
 * @conf mask_freq_ids     List of int. FPGA frequency ids that are never searched.
 * @conf threshold         Double. S/N above which a candidate is reported. Default 8.
 * @conf holdoff_s         Double. No candidates for this long after a (re)start. Default 120.
 * @conf tau_s             Double. Time constant of the running statistics. Default 5.
 * @conf num_threads       Int. OpenMP threads of the search. Default 4.
 * @conf assembly_delay_s  Double. How long a time sample waits for all halves. Default 1.
 * @conf restart_gap_s     Double. Frames this far from the assembled samples are dropped, and
 *                         a run of them restarts the search. Default 10.
 * @conf candidate_file    String. File the candidates are appended to. Default none.
 *
 * @metric kotekan_incoherent_beam_search_candidates_total   Candidates reported.
 * @metric kotekan_incoherent_beam_search_max_snr   Highest S/N in the last 32 search passes.
 * @metric kotekan_incoherent_beam_search_channels_used   Channels used in the newest sample.
 * @metric kotekan_incoherent_beam_search_dropped_frames_total   Frames late, far or off the grid.
 * @metric kotekan_incoherent_beam_search_restarts_total   Restarts of the search.
 */
class IncoherentBeamSearch : public kotekan::Stage {
public:
    IncoherentBeamSearch(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffers) :
        Stage(config, unique_name, buffers, std::bind(&IncoherentBeamSearch::main_thread, this)),
        in_buf(get_buffer("in_buf")), freq_id_base(config.get<int>(unique_name, "freq_id_base")),
        candidates_counter(Metrics::instance().add_counter(
            "kotekan_incoherent_beam_search_candidates_total", unique_name)),
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
        const double dm_step = config.get_default<double>(unique_name, "dm_step", 0.001);
        if (dm_min <= 0 || dm_max < dm_min || dm_step <= 0)
            FATAL_ERROR("IncoherentBeamSearch: need 0 < dm_min <= dm_max and dm_step > 0");
        for (double dm = dm_min; dm <= dm_max; dm *= 1 + dm_step)
            params.dms.push_back(dm);
        params.mask_bands_mhz = config.get_default<std::vector<std::pair<double, double>>>(
            unique_name, "mask_bands_mhz", {});
        for (int id : config.get_default<std::vector<int>>(unique_name, "mask_freq_ids", {})) {
            const double f = tel.to_freq_MHz((freq_id_t)id);
            params.mask_bands_mhz.push_back({f - params.dfreq_mhz / 4, f + params.dfreq_mhz / 4});
        }
        params.threshold = config.get_default<double>(unique_name, "threshold", params.threshold);
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
        std::fill(ring.begin(), ring.end(), 0);
        std::fill(recent_max.begin(), recent_max.end(), 0.0);
    }

    void search_next_sample() {
        uint16_t* const row = &ring[(size_t)(next_sample % ring_len) * params.num_freq];
        for (int c = 0; c < params.num_freq; ++c)
            power[c] = row[c];
        std::fill(row, row + params.num_freq, 0);
        ++next_sample;
        if (search->add_sample(power.data()))
            report();
    }

    // Called after each search pass.
    void report() {
        recent_max[num_reports++ % recent_max.size()] = search->take_max_snr();
        max_snr_gauge.set(*std::max_element(recent_max.begin(), recent_max.end()));
        channels_used_gauge.set(search->channels_used());

        const Telescope& tel = Telescope::instance();
        for (const IncoherentCandidate& c : search->take_candidates()) {
            const int64_t seq = (origin + c.sample) * period + phase;
            candidates_counter.inc();
            INFO("IncoherentBeamSearch: candidate S/N {:.1f}, DM {:.1f}, width {:d} samples, "
                 "FPGA seq {:d}",
                 c.snr, c.dm, c.width, seq);
            if (!candidate_file.is_open())
                continue;
            nlohmann::json line = {{"fpga_seq", seq},
                                   {"snr", c.snr},
                                   {"dm", c.dm},
                                   {"width", c.width},
                                   {"num_hits", c.num_hits},
                                   {"block_snr", c.block_snr},
                                   {"reported_unix", current_time()}};
            if (tel.gps_time_enabled())
                line["unix_time"] = ts_to_double(tel.to_time(seq));
            candidate_file << line.dump() << std::endl;
        }
    }

    Buffer* const in_buf;
    const int freq_id_base;
    std::shared_ptr<const kotekan::GenericNDArray> in_desc;
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
    int64_t phase = 0;       // fpga_seq modulo period of the sample grid
    int64_t origin = 0;      // the search's first sample
    int64_t next_sample = 0; // the next sample to search
    int64_t newest_end = 0;  // one past the newest sample received
    std::vector<double> recent_max = std::vector<double>(32, 0.0);
    size_t num_reports = 0;

    std::ofstream candidate_file;
    kotekan::prometheus::Counter& candidates_counter;
    kotekan::prometheus::Gauge& max_snr_gauge;
    kotekan::prometheus::Gauge& channels_used_gauge;
    kotekan::prometheus::Counter& dropped_frames_counter;
    kotekan::prometheus::Counter& restarts_counter;
};

REGISTER_KOTEKAN_STAGE(IncoherentBeamSearch);
