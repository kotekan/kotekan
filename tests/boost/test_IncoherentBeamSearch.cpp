#define BOOST_TEST_MODULE "test_IncoherentBeamSearch"

#include "Config.hpp"
#include "NDArray.hpp"
#include "StageFactory.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "configUpdater.hpp"
#include "kotekanLoggingFixture.hpp"
#include "prometheusMetrics.hpp"
#include "test_utils.hpp"
#include "timeUtil.hpp"
#include "visUtil.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

static bool metric_has(const std::string& line) {
    return kotekan::prometheus::Metrics::instance().serialize().find(line) != std::string::npos;
}

// The trial DMs the stage makes from the configuration below: steps of 1 from 50 to the knee at
// 100, growing from there to 3 at 150.
static std::vector<double> trial_dms() {
    std::vector<double> dms;
    const double growth = std::log(3.0) / std::log(150.0 / 100.0);
    for (double dm = 50; dm <= 150; dm += std::pow(std::max(dm / 100.0, 1.0), growth))
        dms.push_back(dm);
    return dms;
}

// Four halves of 16 channels each, interleaved in frequency as the X-engine halves are, send
// frames of 32 time samples. 64 channels from 400 MHz: DM 100 sweeps 116 samples.
struct SearchPipeline {
    static constexpr int halves = 4, T = 32, F = 16, min_freq_id = 2048, period = 256;
    static constexpr int frames = 8, sum_samples = 256, snr_passes = 32, cutout_samples = 256;
    // The image of recent samples: 4 passes of 8 rows, 8 channels to a column.
    static constexpr int waterfall_rows = 32, waterfall_time_bin = 32, waterfall_columns = 8;
    const std::string path, dm_path, cutout_dir;
    const std::vector<double> dms = trial_dms();
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer input, sums, snr, cutout, power, waterfall;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;
    std::unique_ptr<kotekan::Stage> writer; // hdf5FileWrite of the cutouts, when kotekan has it
    std::mt19937 rng{1};
    int next_slot = 0;
    std::vector<uint16_t> first_sent; // [T][halves * F]: the first samples sent, by channel
    bool sent_any = false;

    // The largest dedispersed sum the stage has written, kept by sums_thread.
    std::thread sums_thread;
    std::mutex sums_mutex;
    float largest_sum = 0;
    int64_t largest_seq = 0; // FPGA sequence number of its row
    int largest_dm = 0;      // index of its trial DM
    int64_t sum_frames = 0;

    // extra: further configuration of the search stage. Two pipelines in one process need
    // different names: a stage's metrics outlive it.
    explicit SearchPipeline(const nlohmann::json& extra = nlohmann::json::object(),
                            const std::string& name = "search") :
        path("/tmp/test_IncoherentBeamSearch." + std::to_string(getpid()) + ".jsonl"),
        dm_path("/tmp/test_IncoherentBeamSearch." + std::to_string(getpid()) + ".dms.json"),
        cutout_dir("/tmp/test_IncoherentBeamSearch." + std::to_string(getpid()) + ".cutouts"),
        pool(metadataPool::create(32, sizeof(chordMetadata), "pool", "chordMetadata")),
        input(frames, T * F * sizeof(uint16_t), pool, "input", "standard", 0, false, false,
              std::vector<int>{}, true),
        sums(4, sum_samples * dms.size() * sizeof(float), pool, "sums", "standard", 0, false, false,
             std::vector<int>{}, true),
        snr(4, snr_passes * dms.size() * sizeof(float), pool, "snr", "standard", 0, false, false,
            std::vector<int>{}, true),
        cutout(4, halves * F * cutout_samples, pool, "cutout", "standard", 0, false, false,
               std::vector<int>{}, true),
        power(2, sum_samples * halves * F * sizeof(uint16_t), pool, "power", "standard", 0, false,
              false, std::vector<int>{}, true),
        waterfall(4, waterfall_rows * waterfall_columns, pool, "waterfall", "standard", 0, false,
                  false, std::vector<int>{}, true) {
        // Force metadata registration from the static library.
        chordMetadata metadata;
        std::remove(path.c_str());
        nlohmann::json cfg = {{"cpu_affinity", {0}},
                              {"log_level", "ERROR"},
                              {"search",
                               {{"in_buf", "input"},
                                {"out_buf", "sums"},
                                {"snr_buf", "snr"},
                                {"cutout_buf", "cutout"},
                                {"power_buf", "power"},
                                {"waterfall_buf", "waterfall"},
                                {"freq_id_base", min_freq_id},
                                {"num_local_freq", halves * F},
                                {"num_feeds", 128},
                                {"dm_min", 50},
                                {"dm_knee", 100},
                                {"dm_max", 150},
                                {"dm_step_min", 1},
                                {"dm_step_max", 3},
                                {"holdoff_s", 3},
                                {"trigger_min_dm", 60},
                                {"num_threads", 2},
                                {"candidate_file", path},
                                {"dm_file", dm_path}}},
                              {"write_cutouts",
                               {{"in_buf", "cutout"},
                                {"base_dir", cutout_dir},
                                {"file_name", "cutout"},
                                {"time_in_file_name", true},
                                {"prefix_hostname", false}}}};
        cfg["search"].update(extra);
        if (name != "search") {
            cfg[name] = cfg["search"];
            cfg.erase("search");
            cfg["write_" + name] = cfg["write_cutouts"];
            cfg.erase("write_cutouts");
        }
        add_test_telescope_config(cfg);
        cfg["gps_time"] = {{"frame0_nano", 1767225600000000000}};
        cfg["num_polarizations"] = 2;
        cfg["num_dishes"] = 2;
        cfg["num_elements"] = 4;
        config.update_config(cfg);
        kotekan::configUpdater::instance().apply_config(config);
        Telescope::instance(config);
        input.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint16, "incoherent_beam_power", {T, F}, {"Trfi", "F"}, {period, 1}));
        buffers.add_buffer("input", &input);
        buffers.add_buffer("sums", &sums);
        buffers.add_buffer("snr", &snr);
        buffers.add_buffer("cutout", &cutout);
        buffers.add_buffer("power", &power);
        buffers.add_buffer("waterfall", &waterfall);
        input.register_producer("test");
        sums.register_consumer("test");
        cutout.register_consumer("test");
        power.register_consumer("test");
        // As on the telescope, nothing consumes the image of recent passes: it is peeked.
        snr.enable_peek_hold();
        waterfall.enable_peek_hold();
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("IncoherentBeamSearch")->create(config, "/" + name, buffers));
        if (makers.count("hdf5FileWrite")) {
            writer.reset(makers.at("hdf5FileWrite")
                             ->create(config,
                                      name == "search" ? "/write_cutouts" : "/write_" + name,
                                      buffers));
            writer->start();
        }
        sums_thread = std::thread(&SearchPipeline::read_sums, this);
    }

    ~SearchPipeline() {
        input.send_shutdown_signal();
        sums.send_shutdown_signal();
        cutout.send_shutdown_signal();
        power.send_shutdown_signal();
        stage->stop();
        stage->join();
        if (writer) {
            writer->stop();
            writer->join();
        }
        sums_thread.join();
        std::filesystem::remove_all(cutout_dir);
        std::remove(path.c_str());
        std::remove(dm_path.c_str());
    }

    // Takes the frames of sums, [time][DM], as the stage writes them.
    void read_sums() {
        for (int slot = 0;; slot = (slot + 1) % sums.num_frames) {
            const float* const frame = (const float*)sums.wait_for_full_frame("test", slot);
            if (frame == nullptr)
                return;
            const int64_t seq = get_chord_metadata(&sums, slot)->get_fpga_seq_num();
            {
                std::lock_guard<std::mutex> lock(sums_mutex);
                ++sum_frames;
                for (int k = 0; k < sum_samples; ++k)
                    for (size_t d = 0; d < dms.size(); ++d)
                        if (frame[k * dms.size() + d] > largest_sum) {
                            largest_sum = frame[k * dms.size() + d];
                            largest_seq = seq + (int64_t)k * period;
                            largest_dm = (int)d;
                        }
            }
            sums.mark_frame_empty("test", slot);
        }
    }

    // Sends num_samples of noise from every half, starting at FPGA sequence number seq0. A
    // pulse of 2 sigma per channel with DM 100 reaches the top channel at pulse_sample.
    void send(int64_t seq0, int num_samples, int pulse_sample) {
        const double width_mhz = 3200.0 / 16384, mean = 1870, sigma = mean / std::sqrt(256 * 128.);
        const double nu_top = (min_freq_id + halves * F - 1) * width_mhz / 1e3;
        std::normal_distribution<double> gauss(0, 1);
        for (int first = 0; first < num_samples; first += T) {
            // Halves in descending order: frames need not arrive in frequency order.
            for (int half = halves - 1; half >= 0; --half) {
                const int slot = next_slot++ % frames;
                uint16_t* const frame = (uint16_t*)input.wait_for_empty_frame("test", slot);
                BOOST_REQUIRE(frame);
                input.allocate_new_metadata_object(slot);
                auto meta = get_chord_metadata(&input, slot);
                meta->set_from_frame_desc(input.get_frame_desc<kotekan::GenericNDArray>());
                meta->set_fpga_seq_num(seq0 + (int64_t)first * period);
                meta->set_time_downsampling_fpga(period);
                std::vector<int> freq_ids(F);
                for (int f = 0; f < F; ++f) {
                    freq_ids[f] = min_freq_id + half + halves * f;
                    const double nu = freq_ids[f] * width_mhz / 1e3;
                    const int delay =
                        (int)std::lround(4.148808 * 100 * (1 / (nu * nu) - 1 / (nu_top * nu_top))
                                         / (period * 5.12e-3));
                    for (int t = 0; t < T; ++t) {
                        const bool on = first + t == pulse_sample + delay;
                        frame[t * F + f] =
                            (uint16_t)std::lround(mean + sigma * (gauss(rng) + (on ? 2 : 0)));
                    }
                }
                if (!sent_any && first == 0) {
                    first_sent.resize((size_t)T * halves * F);
                    for (int t = 0; t < T; ++t)
                        for (int f = 0; f < F; ++f)
                            first_sent[(t * F + f) * halves + half] = frame[t * F + f];
                }
                meta->set_coarse_freq(freq_ids);
                input.mark_frame_full("test", slot);
            }
        }
        sent_any = true;
    }

    // Reads the candidate file once the stage has taken every frame and written `expected`
    // candidates, or after 20 s.
    std::vector<nlohmann::json> candidates(size_t expected) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        std::vector<nlohmann::json> out;
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            out.clear();
            std::ifstream file(path);
            for (std::string line; std::getline(file, line);)
                out.push_back(nlohmann::json::parse(line));
        } while ((out.size() < expected || input.get_num_full_frames() > 0)
                 && std::chrono::steady_clock::now() < deadline);
        return out;
    }
};

BOOST_AUTO_TEST_CASE(a_bright_flagged_candidate_gets_a_cutout) {
    // The pulse of DM 100 is below trigger_min_dm.
    SearchPipeline p({{"trigger_min_dm", 120}, {"cutout_flagged_min_snr", 8}}, "search_flagged");
    p.stage->start();
    p.send(int64_t(1) << 34, 12000, 8000);
    const std::vector<nlohmann::json> cands = p.candidates(1);
    BOOST_REQUIRE_EQUAL(cands.size(), 1);
    BOOST_CHECK_EQUAL(cands[0]["flags"].dump(), R"(["low_dm"])");
    BOOST_CHECK_EQUAL(cands[0].value("cutout", -1), 0);
}

BOOST_AUTO_TEST_CASE(a_candidate_in_cutout_exclude_gets_no_cutout) {
    // The pulse arrives 8000 samples after 2^34 FPGA samples of 5.12 us from frame0_nano,
    // 2026-01-02 00:26:11 UTC.
    const int64_t seq0 = int64_t(1) << 34;
    const double unix_s = 1767225600 + (seq0 + int64_t(8000) * SearchPipeline::period) * 5.12e-6;
    const double era = get_ERA_from_time(double_to_ts(unix_s), 0.0);
    BOOST_CHECK_CLOSE(era, 107.8788, 1e-3);
    SearchPipeline p({{"cutout_exclude", {{90.0, 110.0, era - 1, era + 1}}}}, "search_excluded");
    p.stage->start();
    p.send(seq0, 12000, 8000);
    const std::vector<nlohmann::json> cands = p.candidates(1);
    BOOST_REQUIRE_EQUAL(cands.size(), 1);
    BOOST_CHECK(cands[0]["flags"].empty());
    BOOST_CHECK(!cands[0].contains("cutout"));
}

BOOST_AUTO_TEST_CASE(finds_a_pulse_and_restarts_after_a_jump) {
    SearchPipeline p;
    p.stage->start();

    // 2 sigma in each of 64 channels is 16 sigma in the sum. The sequence numbers are off the
    // multiples of the period by 7.
    const int64_t seq0 = (int64_t(1) << 34) + 7;
    p.send(seq0, 12000, 8000);
    std::vector<nlohmann::json> cands = p.candidates(1);
    BOOST_REQUIRE_EQUAL(cands.size(), 1);
    BOOST_CHECK_GT(cands[0]["snr"].get<double>(), 11);
    BOOST_CHECK_LT(cands[0]["snr"].get<double>(), 21);
    BOOST_CHECK_CLOSE(cands[0]["dm"].get<double>(), 100, 5);
    BOOST_CHECK_SMALL((double)(cands[0]["fpga_seq"].get<int64_t>()
                               - (seq0 + int64_t(8000) * SearchPipeline::period)),
                      3.0 * SearchPipeline::period);
    BOOST_CHECK_EQUAL(cands[0]["block_snr"].size(), 16);
    // All 64 channels had data (a sample that rounds to 0 does not count), and nothing speaks
    // against the pulse: its DM is above trigger_min_dm and its S/N above the default
    // trigger_min_snr.
    BOOST_CHECK_GT(cands[0]["num_channels"].get<int>(), 56);
    BOOST_CHECK(cands[0]["flags"].empty());
    BOOST_CHECK_EQUAL(cands[0]["partners"].get<int>(), 0);
    BOOST_CHECK_EQUAL(cands[0]["one_band"].get<bool>(), false);
    BOOST_CHECK(cands[0].contains("part_ratio"));
    // GPS time is on: the arrival time follows from the sequence number at 5.12 us per sample.
    BOOST_CHECK_SMALL(cands[0]["unix_time"].get<double>()
                          - (1767225600 + cands[0]["fpga_seq"].get<int64_t>() * 5.12e-6),
                      1e-3);

    // The power went out as it came in, by time and channel, 256 samples to a frame. Nothing
    // takes the frames here: the search went on with power_buf full.
    for (int k = 0; k < p.power.num_frames; ++k) {
        const int num_freq = SearchPipeline::halves * SearchPipeline::F;
        const uint16_t* const frame = (const uint16_t*)p.power.wait_for_full_frame("test", k);
        BOOST_REQUIRE(frame);
        const auto meta = get_chord_metadata(&p.power, k);
        BOOST_CHECK_EQUAL(meta->get_fpga_seq_num(),
                          seq0 + int64_t(k) * SearchPipeline::sum_samples * SearchPipeline::period);
        BOOST_CHECK_EQUAL(meta->get_coarse_freq().size(), num_freq);
        BOOST_CHECK_EQUAL(meta->get_coarse_freq()[0], SearchPipeline::min_freq_id);
        if (k == 0)
            BOOST_CHECK(std::equal(p.first_sent.begin(), p.first_sent.end(), frame));
    }
    BOOST_CHECK(!metric_has(
        "kotekan_incoherent_beam_search_power_skipped_total{stage_name=\"/search\"} 0"));

    // The candidate has no flag, so it has a cutout: the first one. The pulse is the middle
    // column, and the frame starts half a cutout before the candidate at the top channel.
    BOOST_CHECK_EQUAL(cands[0]["cutout"].get<int64_t>(), 0);
    {
        const int T = SearchPipeline::cutout_samples;
        const int num_freq = SearchPipeline::halves * SearchPipeline::F;
        const int8_t* const frame = (const int8_t*)p.cutout.wait_for_full_frame("test", 0);
        BOOST_REQUIRE(frame);
        const auto meta = get_chord_metadata(&p.cutout, 0);
        BOOST_CHECK_EQUAL(meta->get_fpga_seq_num(), cands[0]["fpga_seq"].get<int64_t>()
                                                        - int64_t(T / 2) * SearchPipeline::period);
        BOOST_CHECK_EQUAL(meta->get_coarse_freq().size(), num_freq);
        BOOST_CHECK_EQUAL(meta->get_coarse_freq()[0], SearchPipeline::min_freq_id);
        int brightest = 0;
        double brightest_sum = 0;
        for (int k = 0; k < T; ++k) {
            double sum = 0;
            for (int c = 0; c < num_freq; ++c)
                sum += frame[c * T + k] / 32.0;
            if (sum > brightest_sum) {
                brightest_sum = sum;
                brightest = k;
            }
        }
        BOOST_CHECK_EQUAL(brightest, T / 2);
        BOOST_CHECK_GT(brightest_sum, 80); // 2 sigma in each of 64 channels
        p.cutout.mark_frame_empty("test", 0);
    }
    // hdf5FileWrite numbers its files as the stage numbers the cutouts, after the UTC time of
    // the cutout's first sample: 5.12 us per FPGA sample from frame0_nano.
    if (p.writer) {
        const int64_t first_seq =
            cands[0]["fpga_seq"].get<int64_t>()
            - int64_t(SearchPipeline::cutout_samples / 2) * SearchPipeline::period;
        const std::time_t seconds = 1767225600 + first_seq * 5120 / 1000000000;
        std::tm utc{};
        gmtime_r(&seconds, &utc);
        char time_tag[32];
        std::strftime(time_tag, sizeof(time_tag), "%Y%m%dT%H%M%S", &utc);
        const std::string file = p.cutout_dir + "/cutout." + time_tag + ".00000000.h5";
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!std::filesystem::exists(file) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        BOOST_CHECK(std::filesystem::exists(file));
    }

    // dm_file has the trial DMs. The largest dedispersed sum in the output is the pulse: at
    // DM 100, in the row of its arrival at the bottom channel, one sweep after the top.
    std::ifstream dm_file(p.dm_path);
    const nlohmann::json grid = nlohmann::json::parse(dm_file);
    BOOST_REQUIRE_EQUAL(grid["dm"].size(), p.dms.size());
    BOOST_CHECK_CLOSE(grid["dm"].back().get<double>(), p.dms.back(), 1e-9);
    BOOST_CHECK_EQUAL(grid["dm"][50].get<double>(), 100);
    {
        std::lock_guard<std::mutex> lock(p.sums_mutex);
        const int64_t sweep = grid["sweep_samples"][50].get<int64_t>();
        BOOST_CHECK_EQUAL(sweep, 116);
        BOOST_CHECK_GT(p.sum_frames, 12000 / SearchPipeline::sum_samples - 6);
        BOOST_CHECK_GT(p.largest_sum, 80); // 2 sigma in each of 64 channels
        BOOST_CHECK_EQUAL(p.largest_dm, 50);
        BOOST_CHECK_EQUAL(p.largest_seq, seq0 + (8000 + sweep) * SearchPipeline::period);
    }

    // The image of recent passes, one row per pass and the newest last, has the pulse at
    // DM 100. Its first row starts snr_passes passes before the end of the newest pass.
    std::vector<uint8_t> image_bytes;
    std::shared_ptr<metadataObject> image_meta;
    BOOST_REQUIRE_GE(p.snr.peek_newest_full_frame(image_bytes, SIZE_MAX, image_meta), 0);
    const size_t num_dm = p.dms.size();
    BOOST_REQUIRE_EQUAL(image_bytes.size(), SearchPipeline::snr_passes * num_dm * sizeof(float));
    const float* const image = (const float*)image_bytes.data();
    size_t brightest = 0;
    for (size_t i = 0; i < SearchPipeline::snr_passes * num_dm; ++i)
        if (image[i] > image[brightest])
            brightest = i;
    BOOST_CHECK_GT(image[brightest], 11);
    BOOST_CHECK_EQUAL(brightest % num_dm, 50);
    const int64_t pass = SearchPipeline::sum_samples * SearchPipeline::period;
    const int64_t image_seq =
        std::static_pointer_cast<chordMetadata>(image_meta)->get_fpga_seq_num();
    const int64_t row_end = image_seq + (int64_t)(brightest / num_dm + 1) * pass;
    const int64_t found = seq0 + int64_t(8000) * SearchPipeline::period;
    // The row whose pass searched the pulse ends within two passes after its arrival.
    BOOST_CHECK_GT(row_end, found);
    BOOST_CHECK_LE(row_end, found + 3 * pass);

    // The image of recent samples: rows of 32 samples counted from the search's first sample,
    // 64 channels in 8 columns, every bin in use and at about 1 sigma (16 in the image).
    std::vector<uint8_t> waterfall_bytes;
    std::shared_ptr<metadataObject> waterfall_meta;
    BOOST_REQUIRE_GE(p.waterfall.peek_newest_full_frame(waterfall_bytes, SIZE_MAX, waterfall_meta),
                     0);
    BOOST_REQUIRE_EQUAL(waterfall_bytes.size(),
                        SearchPipeline::waterfall_rows * SearchPipeline::waterfall_columns);
    const int64_t row_step = SearchPipeline::waterfall_time_bin * SearchPipeline::period;
    const auto waterfall_chord = std::static_pointer_cast<chordMetadata>(waterfall_meta);
    BOOST_CHECK_GT(waterfall_chord->get_fpga_seq_num(), seq0);
    BOOST_CHECK_EQUAL((waterfall_chord->get_fpga_seq_num() - seq0) % row_step, 0);
    BOOST_CHECK_EQUAL(waterfall_chord->get_time_downsampling_fpga(), row_step);
    double waterfall_square = 0;
    for (uint8_t byte : waterfall_bytes) {
        BOOST_CHECK_NE((int8_t)byte, -128);
        waterfall_square += (double)(int8_t)byte * (int8_t)byte;
    }
    const double waterfall_rms = std::sqrt(waterfall_square / waterfall_bytes.size()) / 16;
    BOOST_CHECK_GT(waterfall_rms, 0.7);
    BOOST_CHECK_LT(waterfall_rms, 1.2);

    // One round of frames 200 s ahead (a sender with a wrong time) is dropped, and the stream
    // goes on.
    const int no_pulse = -100000;
    p.send(seq0 + int64_t(152588) * SearchPipeline::period, SearchPipeline::T, no_pulse);
    p.send(seq0 + int64_t(12000) * SearchPipeline::period, 640, no_pulse);
    BOOST_CHECK_EQUAL(p.candidates(1).size(), 1);
    BOOST_CHECK(metric_has(
        "kotekan_incoherent_beam_search_dropped_frames_total{stage_name=\"/search\"} 4"));
    BOOST_CHECK(
        !metric_has("kotekan_incoherent_beam_search_restarts_total{stage_name=\"/search\"} 1"));

    // The whole stream jumps ahead by 100 s: the search starts again and finds the next pulse.
    const int64_t seq1 = seq0 + int64_t(76294) * SearchPipeline::period;
    p.send(seq1, 12000, 8000);
    cands = p.candidates(2);
    BOOST_CHECK(
        metric_has("kotekan_incoherent_beam_search_restarts_total{stage_name=\"/search\"} 1"));
    BOOST_REQUIRE_EQUAL(cands.size(), 2);
    BOOST_CHECK_CLOSE(cands[1]["dm"].get<double>(), 100, 5);
    BOOST_CHECK_EQUAL(cands[1]["cutout"].get<int64_t>(), 1);
    BOOST_CHECK_SMALL((double)(cands[1]["fpga_seq"].get<int64_t>()
                               - (seq1 + int64_t(8000) * SearchPipeline::period)),
                      3.0 * SearchPipeline::period);
}
