#define BOOST_TEST_MODULE "test_fftwEngine"

#include "Config.hpp"
#include "NDArray.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "kotekanLoggingFixture.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

static constexpr int SPECTRUM_LENGTH = 64;
static constexpr int SAMPLES = 4 * SPECTRUM_LENGTH; // complex samples per input frame
static constexpr int NUM_FRAMES = 3;

/// One fftwEngine fed by the test: `input_type` complex (cint16) or complex_int8.
struct Engine {
    const bool int8;
    kotekan::Config config;
    Buffer in, out;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    Engine(bool int8, int num_taps) :
        int8(int8), in(2, SAMPLES * (int8 ? 2 : 4), nullptr, "in", "standard", 0, false, false,
                       std::vector<int>{}, true),
        out(2, SAMPLES * 8, nullptr, "out", "standard", 0, false, false, std::vector<int>{}, true) {
        config.update_config({{"cpu_affinity", {0}},
                              {"log_level", "ERROR"},
                              {"fft",
                               {{"in_buf", "in"},
                                {"out_buf", "out"},
                                {"input_type", int8 ? "complex_int8" : "complex"},
                                {"spectrum_length", SPECTRUM_LENGTH},
                                {"num_taps", num_taps}}}});
        out.ensure_frame_desc(kotekan::GenericNDArray::describe(kotekan::cfloat32, "spectrum",
                                                                {SAMPLES}, {"F"}, {1}));
        buffers.add_buffer("in", &in);
        buffers.add_buffer("out", &out);
        in.register_producer("test");
        out.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("fftwEngine")->create(config, "/fft", buffers));
        stage->start();
    }

    ~Engine() {
        in.send_shutdown_signal();
        out.send_shutdown_signal();
        stage->stop();
        stage->join();
    }

    /// Feed one frame of I/Q values and return the spectra it produces.
    std::vector<std::complex<float>> run(int frame, const std::vector<int8_t>& iq) {
        const int slot = frame % 2;
        uint8_t* dst = in.wait_for_empty_frame("test", slot);
        BOOST_REQUIRE(dst);
        for (int k = 0; k < 2 * SAMPLES; k++) {
            if (int8)
                reinterpret_cast<int8_t*>(dst)[k] = iq[k];
            else
                reinterpret_cast<int16_t*>(dst)[k] = iq[k];
        }
        in.mark_frame_full("test", slot);
        auto* src = reinterpret_cast<std::complex<float>*>(out.wait_for_full_frame("test", slot));
        BOOST_REQUIRE(src);
        std::vector<std::complex<float>> spectra(src, src + SAMPLES);
        out.mark_frame_empty("test", slot);
        return spectra;
    }
};

// The same I/Q values as cint8 and as cint16 give the same spectra, with and without the PFB
// (whose history spans frames).
BOOST_AUTO_TEST_CASE(int8_input_matches_int16) {
    for (int num_taps : {1, 4}) {
        Engine e16(false, num_taps), e8(true, num_taps);
        uint32_t state = 12345;
        for (int frame = 0; frame < NUM_FRAMES; frame++) {
            std::vector<int8_t> iq(2 * SAMPLES);
            for (auto& v : iq) {
                state = state * 1664525u + 1013904223u;
                v = (int8_t)((state >> 24) - 128);
            }
            const auto a = e16.run(frame, iq), b = e8.run(frame, iq);
            float peak = 0.0f;
            for (const auto& v : a)
                peak = std::max(peak, std::abs(v));
            BOOST_REQUIRE_GT(peak, 0.0f);
            for (int k = 0; k < SAMPLES; k++)
                BOOST_CHECK_SMALL(std::abs(a[k] - b[k]), 1e-5f * peak);
        }
    }
}
