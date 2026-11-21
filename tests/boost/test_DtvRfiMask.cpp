#define BOOST_TEST_MODULE "test_DtvRfiMask"

#include "Config.hpp"
#include "NDArray.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLoggingFixture.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

/// Write a pilot_profiles.json with one calibrated channel, qualified or not, beside a
/// control and a pending channel, which never need the record.
static std::string write_profiles(bool qualified) {
    nlohmann::json provenance = {{"note", "test"}};
    if (qualified)
        provenance["mask_qualification"] = {{"test_report_sha256", std::string(64, 'a')},
                                            {"reviewer", "test"}};
    const nlohmann::json bundle = {
        {"profiles",
         {{{"physical_channel", 30},
           {"fine_calibration", {{"status", "calibrated"}, {"provenance", provenance}}}},
          {{"physical_channel", 31}, {"fine_calibration", {{"status", "pending_campaign"}}}},
          {{"physical_channel", 37}, {"fine_calibration", {{"status", "control"}}}}}}};
    const std::string path = "/tmp/test_DtvRfiMask_" + std::to_string(getpid())
                             + (qualified ? "_qualified" : "_detect_only") + ".json";
    std::ofstream(path) << bundle.dump();
    return path;
}

struct MaskPipeline {
    static constexpr int F = 4;
    const int samples;
    const bool has_fine;
    std::shared_ptr<kotekan::GenericNDArray> dtv_desc, support_desc;
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer rfi, dtv, support, output;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    MaskPipeline(int samples, bool fine = true, bool time_axis = false, bool declared = true,
                 bool qualified = true) :
        samples(samples), has_fine(fine),
        pool(metadataPool::create(32, sizeof(chordMetadata), "pool", "chordMetadata")),
        rfi(2, samples / 8 * F, pool, "rfi", "standard", 0, false, false, std::vector<int>{}, true),
        dtv(2, F, pool, "dtv", "standard", 0, false, false, std::vector<int>{}, true),
        support(2, F * 2 * sizeof(int32_t), pool, "support", "standard", 0, false, false,
                std::vector<int>{}, true),
        output(2, samples / 8 * F, pool, "output", "standard", 0, false, false, std::vector<int>{},
               true) {
        // Force metadata registration from the static library.
        chordMetadata metadata;
        nlohmann::json cfg = {
            {"cpu_affinity", {0}},
            {"log_level", "ERROR"},
            {"num_local_freq", F},
            {"num_times", samples},
            {"mask",
             {{"pilot_profiles_path", write_profiles(qualified)},
              {"rfi_buf", "rfi"},
              {"dtv_buf", "dtv"},
              {"out_buf", "output"},
              {"permanent_mask_freq_ids", {12}},
              {"require_fine_freq_ids", fine ? nlohmann::json{13} : nlohmann::json::array()}}}};
        if (fine)
            cfg["mask"]["fine_support_buf"] = "support";
        config.update_config(cfg);
        auto mask_desc =
            kotekan::GenericNDArray::describe(kotekan::uint1x8, "RFImask", {samples / 1024, F, 128},
                                              {"T8hi128", "F", "T8lo128"}, {1024, 1, 8});
        rfi.ensure_frame_desc(mask_desc);
        output.ensure_frame_desc(mask_desc);
        dtv_desc = time_axis ? kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask", {1, F},
                                                                 {"T", "F"}, {samples * 4, 1})
                             : kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask", {F},
                                                                 {"F"}, {1});
        if (fine)
            support_desc =
                time_axis ? kotekan::GenericNDArray::describe(kotekan::int32, "dtv_fine_support",
                                                              {1, F, 2}, {"T", "F", "S"},
                                                              {samples * 4, 1, 1})
                          : kotekan::GenericNDArray::describe(kotekan::int32, "dtv_fine_support",
                                                              {F, 2}, {"F", "S"}, {1, 1});
        if (declared) {
            dtv.ensure_frame_desc(dtv_desc);
            if (fine)
                support.ensure_frame_desc(support_desc);
        }
        for (auto* b : {&rfi, &dtv, &support, &output})
            buffers.add_buffer(b->buffer_name, b);
        rfi.register_producer("test");
        dtv.register_producer("test");
        support.register_producer("test");
        output.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("DtvRfiMask")->create(config, "/mask", buffers));
    }

    ~MaskPipeline() {
        for (auto* b : {&rfi, &dtv, &support, &output})
            b->send_shutdown_signal();
        stage->stop();
        stage->join();
    }

    void fill(int index, bool frequency_index = true) {
        const int slot = index % 2;
        for (auto* b : {&rfi, &dtv, &support}) {
            if (b == &support && !has_fine)
                continue;
            BOOST_REQUIRE(b->wait_for_empty_frame("test", slot));
            b->allocate_new_metadata_object(slot);
            auto meta = get_chord_metadata(b, slot);
            meta->set_from_frame_desc(b == &dtv ? dtv_desc
                                      : b == &support
                                          ? support_desc
                                          : b->get_frame_desc<kotekan::GenericNDArray>());
            meta->set_fpga_seq_num((int64_t(1) << 34) + int64_t(index) * samples * 4);
            meta->set_time_downsampling_fpga(b == &rfi ? 4096 : samples * 4);
            meta->set_coarse_freq({10, 11, 12, 13});
            meta->set_freq_upchan_factor({1, 1, 1, 1});
            if (frequency_index)
                meta->set_freq_upchan_index({0, 0, 0, 0});
        }
        for (size_t j = 0; j < rfi.frame_size; ++j)
            rfi.frames[slot][j] = (j + index) % 251;
        const uint8_t reject[F] = {0, 1, 0, 0};
        const int32_t fine[F * 2] = {-1, -1, 1, 124, -1, -1, index % 2, 124};
        std::memcpy(dtv.frames[slot], reject, sizeof(reject));
        std::memcpy(support.frames[slot], fine, sizeof(fine));
        for (auto* b : {&rfi, &dtv, &support})
            if (b != &support || has_fine)
                b->mark_frame_full("test", slot);
    }
};

BOOST_AUTO_TEST_CASE(mask_and_metadata_across_wrap) {
    for (int samples : {8192, 16384})
        for (bool fine : {false, true})
            for (bool time_axis : {false, true})
                for (bool declared : {false, true}) {
                    MaskPipeline p(samples, fine, time_axis, declared);
                    p.stage->start();
                    for (int index = 0; index < 8; ++index) {
                        p.fill(index, index % 2 != 0);
                        const auto deadline =
                            std::chrono::steady_clock::now() + std::chrono::seconds(5);
                        while (p.output.get_num_full_frames() == 0
                               && std::chrono::steady_clock::now() < deadline)
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        BOOST_REQUIRE_GT(p.output.get_num_full_frames(), 0);
                        const int slot = index % 2;
                        auto* data = p.output.wait_for_full_frame("test", slot);
                        BOOST_REQUIRE(data);
                        for (size_t j = 0; j < p.output.frame_size; ++j) {
                            const int f = (j / 128) % MaskPipeline::F;
                            const bool excluded =
                                f == 1 || f == 2 || (fine && f == 3 && index % 2 == 0);
                            BOOST_CHECK_EQUAL(data[j], excluded ? 0 : (j + index) % 251);
                        }
                        auto meta = get_chord_metadata(&p.output, slot);
                        BOOST_CHECK_EQUAL(meta->get_fpga_seq_num(),
                                          (int64_t(1) << 34) + int64_t(index) * samples * 4);
                        BOOST_CHECK_EQUAL(meta->get_time_downsampling_fpga(), 4096);
                        BOOST_CHECK(meta->get_coarse_freq() == std::vector<int>({10, 11, 12, 13}));
                        BOOST_CHECK_EQUAL(meta->has_freq_upchan_index(), index % 2 != 0);
                        p.output.mark_frame_empty("test", slot);
                    }
                }
}

BOOST_AUTO_TEST_CASE(reject_malformed_input) {
    for (bool time_axis : {false, true})
        for (int fault = 0; fault < (time_axis ? 10 : 6); ++fault) {
            MaskPipeline p(8192, true, time_axis);
            p.fill(0);
            switch (fault) {
                case 0:
                    p.dtv.frames[0][0] = 2;
                    break;
                case 1:
                    get_chord_metadata(&p.dtv, 0)->set_fpga_seq_num(1);
                    break;
                case 2:
                    get_chord_metadata(&p.dtv, 0)->set_coarse_freq({11, 10, 12, 13});
                    break;
                case 3:
                    reinterpret_cast<int32_t*>(p.support.frames[0])[6] = 2;
                    break;
                case 4:
                    reinterpret_cast<int32_t*>(p.support.frames[0])[6] = -1;
                    reinterpret_cast<int32_t*>(p.support.frames[0])[7] = -1;
                    break;
                case 5:
                    get_chord_metadata(&p.support, 0)->set_time_downsampling_fpga(1);
                    break;
                case 6:
                    get_chord_metadata(&p.dtv, 0)->dim[0] = 2;
                    break;
                case 7:
                    get_chord_metadata(&p.support, 0)->set_dimension_name(0, "F");
                    break;
                case 8:
                    get_chord_metadata(&p.dtv, 0)->dim_scaling[0] = 1;
                    break;
                case 9:
                    get_chord_metadata(&p.support, 0)->dim_scaling[0] = 1;
                    break;
            }
            BOOST_CHECK_THROW(p.stage->main_thread(), FatalError);
        }
}

BOOST_AUTO_TEST_CASE(reject_overflowing_time_metadata) {
    for (int samples : {8192, 16384})
        for (bool time_axis : {false, true})
            for (bool overflow_period : {false, true}) {
                MaskPipeline p(samples, true, time_axis);
                p.fill(0);
                p.fill(1);
                get_chord_metadata(&p.dtv, 1)->set_fpga_seq_num(0);
                if (overflow_period) {
                    // Multiplying this period by the mask rows wraps to samples * 4.
                    const int64_t period = (int64_t(1) << (samples == 8192 ? 61 : 60)) + 4096;
                    get_chord_metadata(&p.rfi, 0)->set_time_downsampling_fpga(period);
                } else {
                    const int64_t seq = std::numeric_limits<int64_t>::max() - samples * 4 + 1;
                    for (auto* buffer : {&p.rfi, &p.dtv, &p.support})
                        get_chord_metadata(buffer, 0)->set_fpga_seq_num(seq);
                }
                BOOST_CHECK_THROW(p.stage->main_thread(), FatalError);
                BOOST_CHECK_EQUAL(p.output.get_num_full_frames(), 0);
            }
}

BOOST_AUTO_TEST_CASE(accept_last_representable_frame) {
    for (int samples : {8192, 16384})
        for (bool time_axis : {false, true}) {
            MaskPipeline p(samples, true, time_axis);
            p.fill(0);
            p.fill(1);
            const int64_t seq = std::numeric_limits<int64_t>::max() - samples * 4;
            for (auto* buffer : {&p.rfi, &p.dtv, &p.support})
                get_chord_metadata(buffer, 0)->set_fpga_seq_num(seq);
            // The first frame fits; the next frame has a discontinuous sequence.
            BOOST_CHECK_THROW(p.stage->main_thread(), FatalError);
            BOOST_REQUIRE_EQUAL(p.output.get_num_full_frames(), 1);
            BOOST_CHECK_EQUAL(get_chord_metadata(&p.output, 0)->get_fpga_seq_num(), seq);
        }
}

BOOST_AUTO_TEST_CASE(refuse_detect_only_bundle_when_always_on) {
    // without enabled_updatable_config the mask is always on
    BOOST_CHECK_THROW(MaskPipeline(8192, true, false, true, false), FatalError);
}

/// Write a qualified fine_decision_v3 bundle: frequency 10 with the default word only,
/// 11 with a power-limit word above its default word, and 13 as the control.
static std::string write_v3_profiles() {
    const nlohmann::json provenance = {
        {"mask_qualification",
         {{"test_report_sha256", std::string(64, 'a')}, {"reviewer", "test"}}}};
    const auto calibrated = [&](int id, int channel, nlohmann::json words) {
        nlohmann::json calibration = {{"status", channel == 37 ? "control" : "calibrated"},
                                      {"decision_version", "fine_decision_v3"},
                                      {"provenance", provenance}};
        calibration.update(words);
        return nlohmann::json{{"physical_channel", channel},
                              {"receiver_channel_id", id},
                              {"fine_calibration", calibration}};
    };
    const nlohmann::json bundle = {
        {"profiles",
         {calibrated(10, 20, {{"cfar_multiplier_q16", 65536}}),
          calibrated(11, 21,
                     {{"cfar_multiplier_q16", 65536}, {"power_limit_multiplier_q16", 4 * 65536}}),
          calibrated(13, 37, {{"cfar_multiplier_q16", 65536}})}}};
    const std::string path = "/tmp/test_DtvRfiMask_" + std::to_string(getpid()) + "_v3.json";
    std::ofstream(path) << bundle.dump();
    return path;
}

struct V3Pipeline {
    static constexpr int F = 4, samples = 8192;
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer rfi, dtv, stat, output;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    explicit V3Pipeline(bool mask_control, bool with_stat = true) :
        pool(metadataPool::create(32, sizeof(chordMetadata), "pool", "chordMetadata")),
        rfi(2, samples / 8 * F, pool, "rfi", "standard", 0, false, false, std::vector<int>{}, true),
        dtv(2, F, pool, "dtv", "standard", 0, false, false, std::vector<int>{}, true),
        stat(2, F * 2 * sizeof(uint64_t), pool, "stat", "standard", 0, false, false,
             std::vector<int>{}, true),
        output(2, samples / 8 * F, pool, "output", "standard", 0, false, false, std::vector<int>{},
               true) {
        chordMetadata metadata;
        nlohmann::json cfg = {{"cpu_affinity", {0}},
                              {"log_level", "ERROR"},
                              {"num_local_freq", F},
                              {"num_times", samples},
                              {"mask",
                               {{"pilot_profiles_path", write_v3_profiles()},
                                {"rfi_buf", "rfi"},
                                {"dtv_buf", "dtv"},
                                {"out_buf", "output"},
                                {"mask_control", mask_control},
                                {"require_fine_freq_ids", {11}}}}};
        if (with_stat)
            cfg["mask"]["stat_buf"] = "stat";
        config.update_config(cfg);
        auto mask_desc =
            kotekan::GenericNDArray::describe(kotekan::uint1x8, "RFImask", {samples / 1024, F, 128},
                                              {"T8hi128", "F", "T8lo128"}, {1024, 1, 8});
        rfi.ensure_frame_desc(mask_desc);
        output.ensure_frame_desc(mask_desc);
        dtv.ensure_frame_desc(
            kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask", {F}, {"F"}, {1}));
        for (auto* b : {&rfi, &dtv, &stat, &output})
            buffers.add_buffer(b->buffer_name, b);
        for (auto* b : {&rfi, &dtv, &stat})
            b->register_producer("test");
        output.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        stage.reset(makers.at("DtvRfiMask")->create(config, "/mask", buffers));
    }

    ~V3Pipeline() {
        for (auto* b : {&rfi, &dtv, &stat, &output})
            b->send_shutdown_signal();
        stage->stop();
        stage->join();
    }

    void fill(int index, const std::vector<uint64_t>& pairs) {
        const int slot = index % 2;
        for (auto* b : {&rfi, &dtv, &stat}) {
            BOOST_REQUIRE(b->wait_for_empty_frame("test", slot));
            b->allocate_new_metadata_object(slot);
            auto meta = get_chord_metadata(b, slot);
            if (b != &stat)
                meta->set_from_frame_desc(b->get_frame_desc<kotekan::GenericNDArray>());
            meta->set_fpga_seq_num((int64_t(1) << 34) + int64_t(index) * samples * 4);
            meta->set_time_downsampling_fpga(b == &rfi ? 4096 : samples * 4);
            meta->set_coarse_freq({10, 11, 12, 13});
            meta->set_freq_upchan_factor({1, 1, 1, 1});
        }
        std::memset(rfi.frames[slot], 0x5a, rfi.frame_size);
        std::memset(dtv.frames[slot], 0, F); // a v3 channel's byte is not read
        std::memcpy(stat.frames[slot], pairs.data(), F * 2 * sizeof(uint64_t));
        for (auto* b : {&rfi, &dtv, &stat})
            b->mark_frame_full("test", slot);
    }

    /// Return the frequencies the frame rejected.
    std::vector<int> rejected(int index) {
        const int slot = index % 2;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (output.get_num_full_frames() == 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        BOOST_REQUIRE_GT(output.get_num_full_frames(), 0);
        auto* data = output.wait_for_full_frame("test", slot);
        BOOST_REQUIRE(data);
        std::vector<int> result;
        for (int f = 0; f < F; ++f)
            if (data[f * 128] == 0)
                result.push_back(f);
        output.mark_frame_empty("test", slot);
        return result;
    }
};

BOOST_AUTO_TEST_CASE(v3_channels_are_decided_with_their_word) {
    const uint64_t not_evaluated = std::numeric_limits<uint64_t>::max();
    for (bool mask_control : {false, true}) {
        V3Pipeline p(mask_control);
        p.stage->start();
        // 10: 3/2 above its word 1; 11: 3/1 below its power-limit word 4; 13: control at 100
        p.fill(0, {3, 2, 3, 1, 0, 0, 100, 1});
        std::vector<int> want = {0};
        if (mask_control)
            want.push_back(3);
        BOOST_CHECK(p.rejected(0) == want);
        // at the boundary n 2^16 = word d the frame is kept; d_rho = 0 has no decision and is
        // rejected; [0, 2^64 - 1] was not evaluated and is kept
        p.fill(1, {1, 1, 0, 0, 0, 0, 0, not_evaluated});
        BOOST_CHECK(p.rejected(1) == std::vector<int>({1}));
    }
}

BOOST_AUTO_TEST_CASE(v3_bundle_needs_the_statistic) {
    BOOST_CHECK_THROW(V3Pipeline(false, false), FatalError);
}

BOOST_AUTO_TEST_CASE(v3_required_channel_must_be_evaluated) {
    V3Pipeline p(false);
    p.fill(0, {3, 2, 0, std::numeric_limits<uint64_t>::max(), 0, 0, 0, 1});
    BOOST_CHECK_THROW(p.stage->main_thread(), FatalError);
}
