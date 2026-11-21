#define BOOST_TEST_MODULE "test_fEngineSimulate"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "crsPacket.hpp" // for pack12
#include "fftWindowJob.hpp"
#include "kotekanLoggingFixture.hpp"
#include "metadata.hpp"
#include "prometheusMetrics.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <math.h> // for cos, lround, M_PI
#include <memory>
#include <string>
#include <thread>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

namespace {

constexpr uint32_t fft_length = 2048;
constexpr uint32_t samples_per_packet = 512;
constexpr uint32_t num_packets = fft_length / samples_per_packet;
constexpr size_t payload_slot = samples_per_packet * 3 / 2;
constexpr uint32_t num_bins = fft_length / 2;
// One cycle per packet lands on output bin fft_length / samples_per_packet = 4.
constexpr int tone_bin = 1;
constexpr uint32_t out_bin = tone_bin * fft_length / samples_per_packet;
// Amplitude 1024 ADC counts is half scale: |X[out_bin]| = 0.5.
constexpr double amplitude = 1024.0;

struct FEnginePipeline {
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer job, payload, noise, out;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    FEnginePipeline(float gain) :
        pool(metadataPool::create(4, 1, "pool", "none")),
        job(2, fft_window_job_size(num_packets), pool, "job", "standard", 0, false, false,
            std::vector<int>{}, true),
        payload(1, num_packets * payload_slot, pool, "payload", "standard", 0, false, false,
                std::vector<int>{}, true),
        noise(1, payload_slot, pool, "noise", "standard", 0, false, false, std::vector<int>{},
              true),
        out(2, num_bins, pool, "out", "standard", 0, false, false, std::vector<int>{}, true) {
        nlohmann::json cfg = {{"cpu_affinity", {0}},
                              {"log_level", "ERROR"},
                              {"fengine",
                               {{"job_buf", "job"},
                                {"payload_buf", "payload"},
                                {"noise_buf", "noise"},
                                {"out_payload_buf", "out"},
                                {"fft_length", fft_length},
                                {"samples_per_packet", samples_per_packet},
                                {"bins_per_packet", num_bins},
                                {"gain", gain}}}};
        config.update_config(cfg);
        for (auto* b : {&job, &payload, &noise, &out})
            buffers.add_buffer(b->buffer_name, b);
        for (auto* b : {&job, &payload, &noise})
            b->register_producer("test");
        out.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("fEngineSimulate")->create(config, "/fengine", buffers));
    }

    ~FEnginePipeline() {
        for (auto* b : {&job, &payload, &noise, &out})
            b->send_shutdown_signal();
        stage->stop();
        stage->join();
        kotekan::prometheus::Metrics::instance().remove_stage_metrics("/fengine");
    }

    // One window: num_packets identical packets of a tone_bin-cycle tone, the noise frame the
    // stage takes at startup, and a job whose lookup table points at the payload slots.
    void fill() {
        BOOST_REQUIRE(noise.wait_for_empty_frame("test", 0));
        noise.mark_frame_full("test", 0);

        uint8_t* slots = payload.wait_for_empty_frame("test", 0);
        BOOST_REQUIRE(slots);
        for (uint32_t i = 0; i < num_packets; i++)
            for (uint32_t k = 0; k < samples_per_packet; k++)
                pack12(slots + i * payload_slot, k,
                       (int16_t)lround(amplitude
                                       * cos(2.0 * M_PI * tone_bin * k / samples_per_packet)));
        payload.mark_frame_full("test", 0);

        auto* j = reinterpret_cast<fft_window_job*>(job.wait_for_empty_frame("test", 0));
        BOOST_REQUIRE(j);
        j->start_seq = 0;
        j->window_index = 0;
        j->num_packets = num_packets;
        j->lost_packets = 0;
        j->acquire_up_to = 1;
        j->release_up_to = 1;
        j->flags = FFT_WINDOW_REALIGNED;
        j->reserved = 0;
        uint64_t* lut = fft_window_lut(j);
        for (uint32_t i = 0; i < num_packets; i++)
            lut[i] = (uint64_t)(uintptr_t)(slots + i * payload_slot);
        job.mark_frame_full("test", 0);
    }

    const uint8_t* read_out() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (out.get_num_full_frames() == 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        BOOST_REQUIRE_GT(out.get_num_full_frames(), 0);
        return out.wait_for_full_frame("test", 0);
    }
};

} // namespace

BOOST_AUTO_TEST_CASE(quantized_tone) {
    FEnginePipeline p(10.0f);
    p.stage->start();
    p.fill();
    const uint8_t* bins = p.read_out();
    BOOST_REQUIRE(bins);
    // gain 10 * 0.5 = 5: real 5+8 in the high nibble, imag 0+8 in the low.
    BOOST_CHECK_EQUAL((int)bins[out_bin], 0xd8);
    for (uint32_t b = 0; b < num_bins; b++)
        if (b != out_bin)
            BOOST_CHECK_EQUAL((int)bins[b], 0x88);
    // The payload frame and the job are released before the out frame is marked full.
    BOOST_CHECK_EQUAL(p.payload.get_num_full_frames(), 0);
    BOOST_CHECK_EQUAL(p.job.get_num_full_frames(), 0);
}

BOOST_AUTO_TEST_CASE(clamped_tone) {
    FEnginePipeline p(100.0f);
    p.stage->start();
    p.fill();
    const uint8_t* bins = p.read_out();
    BOOST_REQUIRE(bins);
    // gain 100 * 0.5 = 50 clamps to 7.
    BOOST_CHECK_EQUAL((int)bins[out_bin], 0xf8);
}
