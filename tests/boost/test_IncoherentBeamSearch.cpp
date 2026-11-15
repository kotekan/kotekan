#define BOOST_TEST_MODULE "test_IncoherentBeamSearch"

#include "Config.hpp"
#include "NDArray.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLoggingFixture.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <json.hpp>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

// Four halves of 16 channels each, interleaved in frequency as the X-engine halves are, send
// frames of 32 time samples. 64 channels from 400 MHz: DM 100 sweeps 116 samples.
struct SearchPipeline {
    static constexpr int halves = 4, T = 32, F = 16, min_freq_id = 2048, period = 256;
    static constexpr int frames = 8;
    const std::string path;
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer input;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;
    std::mt19937 rng{1};
    int next_slot = 0;

    SearchPipeline() :
        path("/tmp/test_IncoherentBeamSearch." + std::to_string(getpid()) + ".jsonl"),
        pool(metadataPool::create(32, sizeof(chordMetadata), "pool", "chordMetadata")),
        input(frames, T * F * sizeof(uint16_t), pool, "input", "standard", 0, false, false,
              std::vector<int>{}, true) {
        // Force metadata registration from the static library.
        chordMetadata metadata;
        std::remove(path.c_str());
        config.update_config({{"cpu_affinity", {0}},
                              {"log_level", "ERROR"},
                              {"search",
                               {{"in_buf", "input"},
                                {"min_freq_id", min_freq_id},
                                {"num_freq", halves * F},
                                {"num_feeds", 128},
                                {"dm_min", 50},
                                {"dm_max", 150},
                                {"dm_step", 0.02},
                                {"holdoff_s", 3},
                                {"num_threads", 2},
                                {"candidate_file", path}}}});
        input.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint16, "incoherent_beam_power", {T, F}, {"Trfi", "F"}, {period, 1}));
        buffers.add_buffer("input", &input);
        input.register_producer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("IncoherentBeamSearch")->create(config, "/search", buffers));
    }

    ~SearchPipeline() {
        input.send_shutdown_signal();
        stage->stop();
        stage->join();
        std::remove(path.c_str());
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
                meta->set_coarse_freq(freq_ids);
                input.mark_frame_full("test", slot);
            }
        }
    }

    std::vector<nlohmann::json> candidates() {
        // Let the stage take what was sent, then finish its pass.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (input.get_num_full_frames() > 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::vector<nlohmann::json> out;
        std::ifstream file(path);
        for (std::string line; std::getline(file, line);)
            out.push_back(nlohmann::json::parse(line));
        return out;
    }
};

BOOST_AUTO_TEST_CASE(finds_a_pulse_and_restarts_after_a_jump) {
    SearchPipeline p;
    p.stage->start();

    // 2 sigma in each of 64 channels is 16 sigma in the sum. The sequence numbers are off the
    // multiples of the period by 7.
    const int64_t seq0 = (int64_t(1) << 34) + 7;
    p.send(seq0, 12000, 8000);
    std::vector<nlohmann::json> cands = p.candidates();
    BOOST_REQUIRE_EQUAL(cands.size(), 1);
    BOOST_CHECK_GT(cands[0]["snr"].get<double>(), 11);
    BOOST_CHECK_LT(cands[0]["snr"].get<double>(), 21);
    BOOST_CHECK_CLOSE(cands[0]["dm"].get<double>(), 100, 5);
    BOOST_CHECK_SMALL((double)(cands[0]["fpga_seq"].get<int64_t>()
                               - (seq0 + int64_t(8000) * SearchPipeline::period)),
                      3.0 * SearchPipeline::period);
    BOOST_CHECK_EQUAL(cands[0]["block_snr"].size(), 16);

    // The stream jumps ahead by 100 s: the search starts again and finds the next pulse.
    const int64_t seq1 = seq0 + int64_t(76294) * SearchPipeline::period;
    p.send(seq1, 12000, 8000);
    cands = p.candidates();
    BOOST_REQUIRE_EQUAL(cands.size(), 2);
    BOOST_CHECK_CLOSE(cands[1]["dm"].get<double>(), 100, 5);
    BOOST_CHECK_SMALL((double)(cands[1]["fpga_seq"].get<int64_t>()
                               - (seq1 + int64_t(8000) * SearchPipeline::period)),
                      3.0 * SearchPipeline::period);
}
