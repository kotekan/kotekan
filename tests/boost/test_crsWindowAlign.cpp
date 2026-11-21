#define BOOST_TEST_MODULE "test_crsWindowAlign"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "crsPacket.hpp"
#include "fftWindowJob.hpp"
#include "kotekanLoggingFixture.hpp"
#include "prometheusMetrics.hpp"
#include "udpPacket.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

namespace {

constexpr uint32_t sp = 512;          // samples_per_packet
constexpr uint32_t fft_length = 4096; // 8 packets per window
constexpr size_t header_size = 64;
constexpr size_t header_slot = 128;
constexpr size_t payload_size = 768; // sp 12-bit samples
constexpr size_t payload_slot = 768;
constexpr uint32_t noise_slots = 16;

// Value of a series in the Prometheus output, or -1 if it is not exported
int64_t metric_value(const std::string& name, const std::string& stage_name) {
    const std::string series = name + "{stage_name=\"" + stage_name + "\"} ";
    std::istringstream lines(kotekan::prometheus::Metrics::instance().serialize());
    std::string line;
    while (std::getline(lines, line))
        if (line.rfind(series, 0) == 0)
            return std::stoll(line.substr(series.size()));
    return -1;
}

// count seq_numbers, one packet apart, starting at first
std::vector<uint64_t> seqs(uint64_t first, int count) {
    std::vector<uint64_t> s(count);
    for (int i = 0; i < count; i++)
        s[i] = first + (uint64_t)i * sp;
    return s;
}

void check_job(const fft_window_job* job, uint64_t start_seq, uint32_t lost_packets,
               uint64_t acquire_up_to, uint64_t release_up_to, uint32_t flags) {
    BOOST_CHECK_EQUAL(job->start_seq, start_seq);
    BOOST_CHECK_EQUAL(job->window_index, start_seq / fft_length);
    BOOST_CHECK_EQUAL(job->num_packets, fft_length / sp);
    BOOST_CHECK_EQUAL(job->lost_packets, lost_packets);
    BOOST_CHECK_EQUAL(job->acquire_up_to, acquire_up_to);
    BOOST_CHECK_EQUAL(job->release_up_to, release_up_to);
    BOOST_CHECK_EQUAL(job->flags, flags);
}

// One aligner fed hand-crafted header frames of ppf slots. The test produces header_buf and
// consumes job_buf; payload_buf is never written (the stage only takes its frame addresses)
// and noise_buf has no consumer (its one frame recycles at once, the address stays valid).
struct AlignPipeline {
    kotekan::Config config;
    size_t ppf;
    std::string stage_name;
    Buffer header, payload, job, noise;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    AlignPipeline(const std::string& name, size_t _ppf, uint32_t queue_depth) :
        ppf(_ppf), stage_name("/" + name),
        header(8, _ppf * header_slot, nullptr, "header_buf", "standard", 0, false, false, {}, true),
        payload(8, _ppf * payload_slot, nullptr, "payload_buf", "standard", 0, false, false, {},
                true),
        job(4, fft_window_job_size(fft_length / sp), nullptr, "job_buf", "standard", 0, false,
            false, {}, true),
        noise(1, noise_slots * payload_slot, nullptr, "noise_buf", "standard", 0, false, false, {},
              true) {
        nlohmann::json cfg = {{"cpu_affinity", {0}}, {"log_level", "ERROR"}};
        cfg[name] = {{"header_buf", "header_buf"},
                     {"payload_buf", "payload_buf"},
                     {"job_buf", "job_buf"},
                     {"noise_buf", "noise_buf"},
                     {"fft_length", fft_length},
                     {"samples_per_packet", sp},
                     {"adc_id", 0},
                     {"noise_slots", noise_slots},
                     {"queue_depth", queue_depth},
                     {"header_size", header_size},
                     {"header_slot", header_slot},
                     {"payload_size", payload_size},
                     {"payload_slot", payload_slot}};
        config.update_config(cfg);
        for (auto* b : {&header, &payload, &job, &noise})
            buffers.add_buffer(b->buffer_name, b);
        header.register_producer("test");
        job.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("crsWindowAlign")->create(config, stage_name, buffers));
    }

    ~AlignPipeline() {
        for (auto* b : {&header, &payload, &job, &noise})
            b->send_shutdown_signal();
        stage->stop();
        stage->join();
        kotekan::prometheus::Metrics::instance().remove_stage_metrics(stage_name);
    }

    // Fill the first seq_numbers.size() slots of header frame `frame` with good CRS headers;
    // any remaining slots stay zeroed and scan as bad.
    void feed(int frame, const std::vector<uint64_t>& seq_numbers) {
        uint8_t* f = header.wait_for_empty_frame("test", frame);
        BOOST_REQUIRE(f);
        BOOST_REQUIRE_LE(seq_numbers.size(), ppf);
        std::memset(f, 0, header.frame_size);
        for (size_t j = 0; j < seq_numbers.size(); j++) {
            uint8_t* slot = f + j * header_slot;
            crs_packet_header* crs = crs_header_of(slot);
            crs->cookie = CRS_PACKET_COOKIE;
            crs->stream_id = 0;
            crs->seq_number = seq_numbers[j];
            udp_slot_tail* tail = udp_slot_tail_of(slot, header_slot);
            tail->wire_len = header_size + payload_size;
            tail->flags = UDP_SLOT_CSUM_OK;
        }
        header.mark_frame_full("test", frame);
    }

    // Wait until the stage has consumed every fed header frame, so jobs and counters are final.
    void drain() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (header.get_num_full_frames() > 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        BOOST_REQUIRE_EQUAL(header.get_num_full_frames(), 0);
    }

    const fft_window_job* read_job(int id) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (job.get_num_full_frames() == 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        BOOST_REQUIRE_GT(job.get_num_full_frames(), 0);
        auto* j = reinterpret_cast<const fft_window_job*>(job.wait_for_full_frame("test", id));
        BOOST_REQUIRE(j);
        return j;
    }

    // Address of payload slot `slot` of frame `frame`, as the lookup table stores it
    uint64_t pay(int frame, int slot) const {
        return (uint64_t)(uintptr_t)(payload.frames[frame] + slot * payload_slot);
    }

    bool in_noise(uint64_t addr) const {
        const uint64_t base = (uint64_t)(uintptr_t)noise.frames[0];
        return addr >= base && addr < base + noise_slots * payload_slot;
    }

    int64_t metric(const std::string& name) const {
        return metric_value("kotekan_crs_window_align_" + name, stage_name);
    }
};

} // namespace

// Two aligned frames, one window each
BOOST_AUTO_TEST_CASE(clean_windows) {
    AlignPipeline p("align1", 8, 8);
    p.stage->start();
    p.feed(0, seqs(8192, 8));
    p.feed(1, seqs(8192 + 8 * sp, 8));
    p.drain();

    const fft_window_job* j0 = p.read_job(0);
    check_job(j0, 8192, 0, 1, 1, FFT_WINDOW_REALIGNED);
    const uint64_t* lut = fft_window_lut(j0);
    for (int i = 0; i < 8; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(0, i));
    p.job.mark_frame_empty("test", 0);

    const fft_window_job* j1 = p.read_job(1);
    check_job(j1, 12288, 0, 2, 2, 0);
    lut = fft_window_lut(j1);
    for (int i = 0; i < 8; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(1, i));

    BOOST_CHECK_EQUAL(p.metric("windows_total"), 2);
    BOOST_CHECK_EQUAL(p.metric("packets_total"), 16);
    // a never-incremented counter is not exported
    BOOST_CHECK_EQUAL(p.metric("lost_packets_total"), -1);
    BOOST_CHECK_EQUAL(p.metric("bad_packets_total"), -1);
    BOOST_CHECK_EQUAL(p.metric("discarded_packets_total"), -1);
}

// A first seq off the window grid: everything below the next boundary is discarded
BOOST_AUTO_TEST_CASE(misaligned_start) {
    AlignPipeline p("align2", 8, 8);
    p.stage->start();
    p.feed(0, seqs(8704, 8));          // 8704 .. 12288
    p.feed(1, seqs(8704 + 8 * sp, 7)); // 12800 .. 15872; slot 7 stays zeroed and scans bad
    p.drain();

    const fft_window_job* j0 = p.read_job(0);
    check_job(j0, 12288, 0, 2, 1, FFT_WINDOW_REALIGNED);
    const uint64_t* lut = fft_window_lut(j0);
    BOOST_CHECK_EQUAL(lut[0], p.pay(0, 7));
    for (int i = 1; i < 8; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(1, i - 1));

    BOOST_CHECK_EQUAL(p.metric("discarded_packets_total"), 7);
    BOOST_CHECK_EQUAL(p.metric("bad_packets_total"), 1);
    BOOST_CHECK_EQUAL(p.metric("windows_total"), 1);
}

// One packet missing inside the window: its lookup entry points into the noise buffer
BOOST_AUTO_TEST_CASE(interior_gap) {
    AlignPipeline p("align3", 8, 8);
    p.stage->start();
    std::vector<uint64_t> s;
    for (int i : {0, 1, 2, 4, 5, 6, 7, 8})
        s.push_back(8192 + (uint64_t)i * sp);
    p.feed(0, s);
    p.drain();

    const fft_window_job* j0 = p.read_job(0);
    check_job(j0, 8192, 1, 1, 0, FFT_WINDOW_REALIGNED);
    const uint64_t* lut = fft_window_lut(j0);
    for (int i = 0; i < 3; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(0, i));
    BOOST_CHECK(p.in_noise(lut[3]));
    for (int i = 4; i < 8; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(0, i - 1));

    BOOST_CHECK_EQUAL(p.metric("lost_packets_total"), 1);
    BOOST_CHECK_EQUAL(p.metric("windows_total"), 1);
}

// A gap past max_gap_windows: one all-noise window, then abandon and re-align
BOOST_AUTO_TEST_CASE(gap_realigns) {
    AlignPipeline p("align4", 8, 8);
    p.stage->start();
    p.feed(0, seqs(8192, 8));
    p.feed(1, seqs(28672, 8)); // jump of four windows
    p.drain();

    const fft_window_job* j0 = p.read_job(0);
    check_job(j0, 8192, 0, 1, 1, FFT_WINDOW_REALIGNED);
    p.job.mark_frame_empty("test", 0);

    // The gap synthesizes one whole window of noise ...
    const fft_window_job* j1 = p.read_job(1);
    check_job(j1, 12288, 8, 1, 1, 0);
    const uint64_t* lut = fft_window_lut(j1);
    for (int i = 0; i < 8; i++)
        BOOST_CHECK(p.in_noise(lut[i]));
    p.job.mark_frame_empty("test", 1);

    // ... then re-aligns at the window holding seq 28672, which completes from frame 1
    const fft_window_job* j2 = p.read_job(2);
    check_job(j2, 28672, 0, 2, 2, FFT_WINDOW_REALIGNED);
    lut = fft_window_lut(j2);
    for (int i = 0; i < 8; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(1, i));

    BOOST_CHECK_EQUAL(p.metric("realigns_total"), 1);
    BOOST_CHECK_EQUAL(p.metric("lost_packets_total"), 8);
    BOOST_CHECK_EQUAL(p.metric("windows_total"), 3);
}

// A repeated seq counts as reordered and is skipped; the window still completes losslessly
BOOST_AUTO_TEST_CASE(duplicate_packet) {
    AlignPipeline p("align5", 8, 8);
    p.stage->start();
    std::vector<uint64_t> s;
    for (int i : {0, 1, 1, 2, 3, 4, 5, 6})
        s.push_back(8192 + (uint64_t)i * sp);
    p.feed(0, s);
    p.feed(1, seqs(8192 + 7 * sp, 8));
    p.drain();

    const fft_window_job* j0 = p.read_job(0);
    check_job(j0, 8192, 0, 2, 1, FFT_WINDOW_REALIGNED);
    const uint64_t* lut = fft_window_lut(j0);
    BOOST_CHECK_EQUAL(lut[0], p.pay(0, 0));
    BOOST_CHECK_EQUAL(lut[1], p.pay(0, 1));
    // the duplicate in slot 2 is skipped; slots 3..7 carry samples 2..6
    for (int i = 2; i < 7; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(0, i + 1));
    BOOST_CHECK_EQUAL(lut[7], p.pay(1, 0));

    BOOST_CHECK_EQUAL(p.metric("reordered_packets_total"), 1);
    BOOST_CHECK_EQUAL(p.metric("lost_packets_total"), -1); // never incremented, not exported
    BOOST_CHECK_EQUAL(p.metric("windows_total"), 1);
}

// Two windows in one frame: the first closes mid-frame and releases nothing
BOOST_AUTO_TEST_CASE(windows_straddle_frame) {
    AlignPipeline p("align6", 16, 16);
    p.stage->start();
    p.feed(0, seqs(8192, 16));
    p.drain();

    const fft_window_job* j0 = p.read_job(0);
    check_job(j0, 8192, 0, 1, 0, FFT_WINDOW_REALIGNED);
    const uint64_t* lut = fft_window_lut(j0);
    for (int i = 0; i < 8; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(0, i));
    p.job.mark_frame_empty("test", 0);

    const fft_window_job* j1 = p.read_job(1);
    check_job(j1, 12288, 0, 1, 1, 0);
    lut = fft_window_lut(j1);
    for (int i = 0; i < 8; i++)
        BOOST_CHECK_EQUAL(lut[i], p.pay(0, 8 + i));
}

// One window spanning two frames: the closing packet acquires and releases both
BOOST_AUTO_TEST_CASE(window_spans_frames) {
    AlignPipeline p("align7", 4, 4);
    p.stage->start();
    p.feed(0, seqs(8192, 4));
    p.feed(1, seqs(8192 + 4 * sp, 4));
    p.drain();

    const fft_window_job* j0 = p.read_job(0);
    check_job(j0, 8192, 0, 2, 2, FFT_WINDOW_REALIGNED);
    const uint64_t* lut = fft_window_lut(j0);
    for (int i = 0; i < 4; i++) {
        BOOST_CHECK_EQUAL(lut[i], p.pay(0, i));
        BOOST_CHECK_EQUAL(lut[4 + i], p.pay(1, i));
    }
}
