#define BOOST_TEST_MODULE "test_ProcessPacketMask"

#include "Config.hpp"
#include "NDArray.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLoggingFixture.hpp"
#include "prometheusMetrics.hpp"

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

constexpr uint32_t time_long = 8;
constexpr uint32_t num_frequency = 384;
constexpr uint32_t element_long = 2;
constexpr uint32_t num_source_ids = 16;
constexpr int64_t E_DIV_8 = element_long * 8 / 8;

// Value of a series in the Prometheus output, or -1 if it is not exported
int64_t metric_value(const std::string& name, const std::string& labels = "") {
    const std::string series = name + "{stage_name=\"/mask\"" + labels + "} ";
    std::istringstream lines(kotekan::prometheus::Metrics::instance().serialize());
    std::string line;
    while (std::getline(lines, line))
        if (line.rfind(series, 0) == 0)
            return std::stoll(line.substr(series.size()));
    return -1;
}

int64_t board_lost_packets(int source_id) {
    return metric_value("kotekan_processpacketmask_crs_board_lost_packets_total",
                        ",source_id=\"" + std::to_string(source_id) + "\"");
}

int64_t board_expected_packets(int source_id) {
    return metric_value("kotekan_processpacketmask_crs_board_expected_packets_total",
                        ",source_id=\"" + std::to_string(source_id) + "\"");
}

struct MaskPipeline {
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer voltage, bitmap_lo, bitmap_hi, combined, pl_mask;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    MaskPipeline() :
        pool(metadataPool::create(16, sizeof(chordMetadata), "pool", "chordMetadata")),
        voltage(2, time_long * num_frequency * element_long * 16 * 8, pool, "voltage", "standard",
                0, false, false, std::vector<int>{}, true),
        bitmap_lo(2, time_long * num_source_ids, pool, "bitmap_lo", "standard", 0, false, false,
                  std::vector<int>{}, true),
        bitmap_hi(2, time_long * num_source_ids, pool, "bitmap_hi", "standard", 0, false, false,
                  std::vector<int>{}, true),
        combined(2, time_long * num_source_ids, pool, "combined", "standard", 0, false, false,
                 std::vector<int>{}, true),
        pl_mask(2, time_long * 16 / 64 * num_frequency * E_DIV_8 * sizeof(uint64_t), pool,
                "pl_mask", "standard", 0, false, false, std::vector<int>{}, true) {
        // Force metadata registration from the static library.
        chordMetadata metadata;
        nlohmann::json cfg = {
            {"cpu_affinity", {0}},
            {"log_level", "ERROR"},
            {"mask",
             {{"voltage_buf", "voltage"},
              {"receipt_bitmap_bufs", {"bitmap_lo", "bitmap_hi"}},
              {"combined_receipt_bitmap_buf", "combined"},
              {"pl_mask_buf", "pl_mask"},
              {"time_long", time_long},
              {"num_frequency", num_frequency},
              {"element_long", element_long},
              {"num_source_ids", num_source_ids},
              {"missing_source_ids", {5}},
              {"crs_board_remap", {0, 2, 4, 6, 8, 10, 12, 15, 1, 3, 5, 7, 9, 11, 13, 14}}}}};
        config.update_config(cfg);
        pl_mask.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint1x8, "pl_mask_exp",
            {time_long * 16 / 64, num_frequency, 2, E_DIV_8 / 2, 8},
            {"Thi64", "F", "P", "D8", "Tlo64"}, {64, 1, 1, 8, 8}));
        for (auto* b : {&voltage, &bitmap_lo, &bitmap_hi, &combined, &pl_mask})
            buffers.add_buffer(b->buffer_name, b);
        for (auto* b : {&voltage, &bitmap_lo, &bitmap_hi})
            b->register_producer("test");
        combined.register_consumer("test");
        pl_mask.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("ProcessPacketMask")->create(config, "/mask", buffers));
    }

    ~MaskPipeline() {
        for (auto* b : {&voltage, &bitmap_lo, &bitmap_hi, &combined, &pl_mask})
            b->send_shutdown_signal();
        stage->stop();
        stage->join();
        kotekan::prometheus::Metrics::instance().remove_stage_metrics("/mask");
    }

    // Each capture worker reports 4 of the 8 streams per source_id byte; clearing a bit loses
    // that packet. Slots are output slots, after crs_board_remap.
    void fill(int index) {
        const int slot = index % 2;
        for (auto* b : {&voltage, &bitmap_lo, &bitmap_hi})
            BOOST_REQUIRE(b->wait_for_empty_frame("test", slot));
        voltage.allocate_new_metadata_object(slot);
        auto meta = get_chord_metadata(&voltage, slot);
        meta->set_fpga_seq_num(int64_t(index) * time_long * 16);
        meta->set_coarse_freq(std::vector<int>(num_frequency, 0));
        meta->set_freq_upchan_factor(std::vector<int>(num_frequency, 1));

        uint8_t* lo = bitmap_lo.frames[slot];
        uint8_t* hi = bitmap_hi.frames[slot];
        std::memset(lo, 0x0f, bitmap_lo.frame_size);
        std::memset(hi, 0xf0, bitmap_hi.frame_size);
        lo[0 * num_source_ids + 0] &= ~0x02; // t 0, slot 0 (source 0): stream 1
        hi[0 * num_source_ids + 9] &= ~0x20; // t 0, slot 9 (source 3): stream 5
        lo[0 * num_source_ids + 1] &= ~0x04; // t 0, slot 1 (source 2): stream 2
        hi[3 * num_source_ids + 7] &= ~0x50; // t 3, slot 7 (source 15): streams 4, 6
        lo[5 * num_source_ids + 5] = hi[5 * num_source_ids + 5] = 0;   // t 5, missing slot 5
        lo[7 * num_source_ids + 15] = hi[7 * num_source_ids + 15] = 0; // t 7, slot 15 (source 14)

        for (auto* b : {&voltage, &bitmap_lo, &bitmap_hi})
            b->mark_frame_full("test", slot);
    }
};

} // namespace

BOOST_AUTO_TEST_CASE(packet_loss_metrics) {
    MaskPipeline p;
    p.stage->start();
    for (int index = 0; index < 3; ++index) {
        p.fill(index);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (p.pl_mask.get_num_full_frames() == 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        BOOST_REQUIRE_GT(p.pl_mask.get_num_full_frames(), 0);
        const int slot = index % 2;
        auto* mask = reinterpret_cast<uint64_t*>(p.pl_mask.wait_for_full_frame("test", slot));
        BOOST_REQUIRE(mask);
        BOOST_REQUIRE(p.combined.wait_for_full_frame("test", slot));

        // Every element group of the mask is flagged alike, so count lost (time, frequency)
        // samples from the zero bits of the whole mask.
        int64_t zero_bits = 0;
        for (size_t w = 0; w < p.pl_mask.frame_size / sizeof(uint64_t); ++w)
            zero_bits += __builtin_popcountll(~mask[w]);
        BOOST_CHECK_EQUAL(zero_bits % E_DIV_8, 0);
        // t 0 loses both 4-stream groups, t 3 one group and t 7 both: 20 streams.
        BOOST_CHECK_EQUAL(zero_bits / E_DIV_8, 20 * 48 * 16);

        const int frames = index + 1;
        BOOST_CHECK_EQUAL(metric_value("kotekan_processpacketmask_lost_samples_total"),
                          frames * zero_bits / E_DIV_8);
        BOOST_CHECK_EQUAL(metric_value("kotekan_processpacketmask_lost_packets_total"),
                          frames * 13);
        BOOST_CHECK_EQUAL(board_lost_packets(0), frames * 1);
        BOOST_CHECK_EQUAL(board_lost_packets(3), frames * 1);
        BOOST_CHECK_EQUAL(board_lost_packets(2), frames * 1);
        BOOST_CHECK_EQUAL(board_lost_packets(15), frames * 2);
        BOOST_CHECK_EQUAL(board_lost_packets(14), frames * 8);
        BOOST_CHECK_EQUAL(board_lost_packets(1), 0);
        // Slot 5 holds source 10, which is listed as missing and not exported.
        BOOST_CHECK_EQUAL(board_lost_packets(10), -1);

        // 8 streams per time_long from each of the 15 boards not listed as missing
        BOOST_CHECK_EQUAL(metric_value("kotekan_processpacketmask_expected_packets_total"),
                          frames * time_long * 15 * 8);
        BOOST_CHECK_EQUAL(metric_value("kotekan_processpacketmask_samples_total"),
                          frames * time_long * 16 * num_frequency);
        for (int source_id : {0, 1, 14, 15})
            BOOST_CHECK_EQUAL(board_expected_packets(source_id), frames * time_long * 8);
        BOOST_CHECK_EQUAL(board_expected_packets(10), -1);

        p.pl_mask.mark_frame_empty("test", slot);
        p.combined.mark_frame_empty("test", slot);
    }
}
