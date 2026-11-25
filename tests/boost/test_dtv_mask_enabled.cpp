// Boost tests for the /dtv_mask_enabled switch, which DtvRfiMask and N2Accumulate share

#define BOOST_TEST_MODULE "test_dtv_mask_enabled"

#include "CHIMETelescope.hpp"
#include "Config.hpp"
#include "N2FrameDesc.hpp"
#include "N2Metadata.hpp"
#include "N2Util.hpp"
#include "NDArray.hpp"
#include "StageFactory.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "configUpdater.hpp"
#include "kotekanLoggingFixture.hpp"
#include "prometheusMetrics.hpp"
#include "restClient.hpp"
#include "restServer.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

BOOST_AUTO_TEST_CASE(switch_time_rounds_up_to_whole_5_s) {
    const int64_t whole = 1'700'000'000'000'000'000, now = whole + 2'000'000'000;
    const int64_t next = whole + 5'000'000'000;
    BOOST_CHECK_EQUAL(N2::switch_time_ns(0, now), next);
    BOOST_CHECK_EQUAL(N2::switch_time_ns(now - 60'000'000'000, now), next);
    BOOST_CHECK_EQUAL(N2::switch_time_ns(now, now), next);
    BOOST_CHECK_EQUAL(N2::switch_time_ns(0, whole), next);
    BOOST_CHECK_EQUAL(N2::switch_time_ns(now + 1, now), now + 1);
    // Less than 1 s before a whole 5 s, the next one is used.
    const int64_t early = next - 1'000'000'001, late = next - 999'999'999;
    BOOST_CHECK_EQUAL(N2::switch_time_ns(0, early), next);
    BOOST_CHECK_EQUAL(N2::switch_time_ns(0, late), next + 5'000'000'000);
    // A second stage reading its clock later keeps the time the first one wrote back.
    BOOST_CHECK_EQUAL(N2::switch_time_ns(N2::switch_time_ns(0, early), late), next);
}

/// DtvRfiMask and N2Accumulate on one /dtv_mask_enabled block, with frames of the same span.
struct SwitchPipeline {
    // DtvRfiMask frames of 8192 samples span 4 * 8192 FPGA ticks; N2Accumulate takes two
    // correlations of half that per frame and one frame per bin.
    static constexpr int64_t span = 32768;
    static constexpr int samples = 8192, elements = 64;
    const std::string profiles_path =
        "/tmp/test_dtv_mask_enabled_" + std::to_string(getpid()) + ".json";
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool, n2_pool;
    std::shared_ptr<kotekan::GenericNDArray> rfi_desc, dtv_desc;
    Buffer rfi, dtv, mask_out, corr, counts, rficounts, plcounts, framemask, vis;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> mask, accumulate;
    // Set by wait_for_whole_5_s, which runs between the two stages' callbacks
    int boundary_calls = 0;
    int64_t passed_time_ns = -1, before_wait_ns = -1, after_wait_ns = -1;

    /// The config value and its valid_at_time_ns, and whether the bundle may mask.
    SwitchPipeline(bool enabled = false, int64_t valid_at_time_ns = 0, bool qualified = true) :
        pool(metadataPool::create(64, sizeof(chordMetadata), "pool", "chordMetadata")),
        n2_pool(metadataPool::create(8, sizeof(N2Metadata), "n2_pool", "N2Metadata")),
        rfi(4, samples / 8, pool, "rfi", "standard", 0, false, false, std::vector<int>{}, true),
        dtv(4, 1, pool, "dtv", "standard", 0, false, false, std::vector<int>{}, true),
        mask_out(4, samples / 8, pool, "mask_out", "standard", 0, false, false, std::vector<int>{},
                 true),
        corr(4, 2 * 10 * 16 * 16 * 2 * sizeof(int32_t), pool, "corr", "standard", 0, false, false,
             std::vector<int>{}, true),
        counts(4, 2 * 8 * 8 * sizeof(int32_t), pool, "counts", "standard", 0, false, false,
               std::vector<int>{}, true),
        rficounts(4, 2 * sizeof(int32_t), pool, "rficounts", "standard", 0, false, false,
                  std::vector<int>{}, true),
        plcounts(4, 2 * sizeof(int32_t), pool, "plcounts", "standard", 0, false, false,
                 std::vector<int>{}, true),
        framemask(4, 2, pool, "framemask", "standard", 0, false, false, std::vector<int>{}, true),
        vis(4,
            kotekan::N2FrameDesc::calculate_frame_size(
                elements, 0, kotekan::N2FrameDesc::get_num_prod(elements, N2Layout::FullUpperTri)),
            n2_pool, "vis", "N2", 0, false, false, std::vector<int>{}, true) {
        // Force metadata registration from the static library.
        chordMetadata chord_metadata;
        N2Metadata n2_metadata;
        nlohmann::json provenance = nlohmann::json::object();
        if (qualified)
            provenance["mask_qualification"] = {{"test_report_sha256", std::string(64, 'a')},
                                                {"reviewer", "test"}};
        std::ofstream(profiles_path)
            << nlohmann::json({{"profiles",
                                {{{"physical_channel", 30},
                                  {"fine_calibration",
                                   {{"status", "calibrated"}, {"provenance", provenance}}}}}}})
                   .dump();
        // Seq 0 an hour ago, so that now is well inside the stream.
        const int64_t time0_ns = N2::current_system_time_ns() - 3'600'000'000'000;
        nlohmann::json cfg = {
            {"cpu_affinity", {0}},
            {"log_level", "WARN"},
            {"num_polarizations", 2},
            {"num_dishes", elements / 2},
            {"telescope",
             {{"name", "CHIMETelescope"},
              {"sampling_rate_MHz", 800.0},
              {"fft_length", 2048},
              {"nyquist_zone", 2},
              {"inst_lat", 50.0},
              {"inst_long", -120.0}}},
            {"gps_time", {{"frame0_nano", time0_ns}}},
            {"dtv_mask_enabled",
             {{"kotekan_update_endpoint", "json"}, {"enabled", false}, {"valid_at_time_ns", 0}}},
            {"mask",
             {{"num_local_freq", 1},
              {"num_times", samples},
              {"pilot_profiles_path", profiles_path},
              {"rfi_buf", "rfi"},
              {"dtv_buf", "dtv"},
              {"out_buf", "mask_out"},
              {"enabled_updatable_config", "/dtv_mask_enabled"}}},
            {"accumulate",
             {{"num_freq_per_n2k_frame", 1},
              {"num_subintegrations_per_bin", 2},
              {"packet_loss_is_scalar", true},
              {"samples_per_data_set", span},
              {"sub_integration_ntime", span / 2},
              {"variance_mode", "EvenOddPosDef"},
              {"input_order", "CHIMEBeamformer"},
              {"in_buf", "corr"},
              {"in_counts_buf", "counts"},
              {"in_rficounts_buf", "rficounts"},
              {"in_plcounts_buf", "plcounts"},
              {"in_rfiframemask_buf", "framemask"},
              {"out_buf", "vis"},
              {"dtv_mask_enabled_updatable_config", "/dtv_mask_enabled"}}}};
        cfg["dtv_mask_enabled"]["enabled"] = enabled;
        cfg["dtv_mask_enabled"]["valid_at_time_ns"] = valid_at_time_ns;
        config.update_config(cfg);
        kotekan::configUpdater::instance().apply_config(config);
        Telescope::instance(config);
        BOOST_REQUIRE(dynamic_cast<const CHIMETelescope*>(&Telescope::instance()));

        rfi_desc =
            kotekan::GenericNDArray::describe(kotekan::uint1x8, "RFImask", {samples / 1024, 1, 128},
                                              {"T8hi128", "F", "T8lo128"}, {1024, 1, 8});
        dtv_desc = kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask", {1}, {"F"}, {1});
        rfi.ensure_frame_desc(rfi_desc);
        mask_out.ensure_frame_desc(rfi_desc);
        dtv.ensure_frame_desc(dtv_desc);
        corr.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int32, "n2k_correlation", {2, 1, 10, 16, 16, 2},
            {"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"}, {span / 2, 1, 16, 1, 1, 1}));
        counts.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int32, "n2k_counts", {2, 1, 1, 8, 8}, {"Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"},
            {span / 2, 1, 64, 8, 8}));
        rficounts.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int32, "RFImask_counts", {2, 1}, {"Tc", "F"}, {span / 2, 1}));
        plcounts.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int32, "pl_lost_counts_scalar", {2, 1}, {"Tc", "F"}, {span / 2, 1}));
        framemask.ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint8, "RFIFrameMask", {2, 1}, {"Tc", "F"}, {span / 2, 1}));
        vis.ensure_frame_desc(std::make_shared<kotekan::N2FrameDesc>(
            elements, 0, kotekan::N2FrameDesc::get_num_prod(elements, N2Layout::FullUpperTri),
            N2Layout::FullUpperTri));
        for (auto* b : inputs())
            b->register_producer("test");
        for (auto* b :
             {&rfi, &dtv, &mask_out, &corr, &counts, &rficounts, &plcounts, &framemask, &vis})
            buffers.add_buffer(b->buffer_name, b);
        mask_out.register_consumer("test");
        vis.register_consumer("test");
        auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
        mask.reset(makers.at("DtvRfiMask")->create(config, "/mask", buffers));
        kotekan::configUpdater::instance().subscribe(
            "/dtv_mask_enabled",
            std::bind(&SwitchPipeline::wait_for_whole_5_s, this, std::placeholders::_1));
        accumulate.reset(makers.at("N2Accumulate")->create(config, "/accumulate", buffers));
    }

    /// On a POST, record the time DtvRfiMask passed on, then wait until just after the next
    /// whole 5 s, so N2Accumulate reads its clock on the other side of it.
    bool wait_for_whole_5_s(nlohmann::json& update) {
        if (boundary_calls++ == 0)
            return true;
        const int64_t step = 5'000'000'000;
        passed_time_ns = update.at("valid_at_time_ns").get<int64_t>();
        before_wait_ns = N2::current_system_time_ns();
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(step - before_wait_ns % step + 100'000'000));
        after_wait_ns = N2::current_system_time_ns();
        return true;
    }

    ~SwitchPipeline() {
        for (auto* b : inputs())
            b->send_shutdown_signal();
        mask_out.send_shutdown_signal();
        vis.send_shutdown_signal();
        for (auto* stage : {mask.get(), accumulate.get()}) {
            stage->stop();
            stage->join();
            kotekan::prometheus::Metrics::instance().remove_stage_metrics(stage->get_unique_name());
        }
        std::remove(profiles_path.c_str());
    }

    std::vector<Buffer*> inputs() {
        return {&rfi, &dtv, &corr, &counts, &rficounts, &plcounts, &framemask};
    }

    /// Fill a frame starting at seq, with a fresh metadata object.
    std::shared_ptr<chordMetadata> start_frame(Buffer& buffer, int index, int64_t seq,
                                               int64_t period) {
        BOOST_REQUIRE(buffer.wait_for_empty_frame("test", index));
        buffer.allocate_new_metadata_object(index);
        auto meta = get_chord_metadata(&buffer, index);
        meta->set_fpga_seq_num(seq);
        meta->set_time_downsampling_fpga(period);
        meta->set_coarse_freq({100});
        meta->set_freq_upchan_factor({1});
        return meta;
    }

    /// Give DtvRfiMask one frame, rejected by the detector, and return whether it was masked.
    bool mask_frame(int index, int64_t seq) {
        start_frame(rfi, index, seq, 4096)->set_from_frame_desc(rfi_desc);
        start_frame(dtv, index, seq, span)->set_from_frame_desc(dtv_desc);
        std::memset(rfi.frames[index], 0xff, rfi.frame_size);
        dtv.frames[index][0] = 1;
        rfi.mark_frame_full("test", index);
        dtv.mark_frame_full("test", index);
        const uint8_t* out = mask_out.wait_for_full_frame("test", index);
        BOOST_REQUIRE(out);
        const bool masked = out[0] == 0;
        mask_out.mark_frame_empty("test", index);
        return masked;
    }

    /// Give N2Accumulate one zeroed frame of two correlations.
    void accumulate_frame(int index, int64_t seq) {
        for (auto* b : {&corr, &counts, &rficounts, &plcounts, &framemask}) {
            auto meta = start_frame(*b, index, seq, span / 2);
            if (b == &framemask) {
                meta->set_rfi_frame_excision_enabled(false);
                meta->set_rfi_frame_excision_thresholds({{3.0f, 0.1f}});
            }
            std::memset(b->frames[index], 0, b->frame_size);
            b->mark_frame_full("test", index);
        }
    }
};

BOOST_AUTO_TEST_CASE(both_stages_switch_at_the_same_frame) {
    kotekan::restServer::instance().start("127.0.0.1", 0);
    SwitchPipeline p;
    p.mask->start();
    p.accumulate->start();

    // Post 0.2 to 0.4 s before a whole 5 s, so DtvRfiMask reads its clock before it, and
    // N2Accumulate, after wait_for_whole_5_s, after it.
    const int64_t step = 5'000'000'000, offset = 4'600'000'000;
    int64_t now = N2::current_system_time_ns();
    std::this_thread::sleep_for(std::chrono::nanoseconds((offset - now % step + step) % step));
    now = N2::current_system_time_ns();
    BOOST_REQUIRE_LT(now % step, offset + 200'000'000);
    // The next whole 5 s at least 1 s ahead
    const int64_t switch_ns = (now / step + 2) * step;
    const int64_t switch_seq = Telescope::instance().to_seq(switch_ns);
    // Frames 0 and 1 start before the switch, frame 3 after it.
    const int64_t seq0 = (switch_seq / p.span - 2) * p.span;

    // DtvRfiMask is ahead of N2Accumulate in the stream when the POST arrives.
    BOOST_CHECK(!p.mask_frame(0, seq0));
    const auto reply = restClient::instance().make_request_blocking(
        "/dtv_mask_enabled", nlohmann::json{{"enabled", true}, {"valid_at_time_ns", 0}},
        "127.0.0.1", kotekan::restServer::instance().port());
    BOOST_REQUIRE_MESSAGE(reply.first, reply.second);
    BOOST_REQUIRE_EQUAL(p.before_wait_ns / step, now / step);
    BOOST_REQUIRE_EQUAL(p.after_wait_ns / step, now / step + 1);
    // DtvRfiMask passes the resolved time on; the config keeps the posted value.
    BOOST_CHECK_EQUAL(p.passed_time_ns, switch_ns);
    BOOST_CHECK_EQUAL(p.config.get<int64_t>("/dtv_mask_enabled", "valid_at_time_ns"), 0);

    for (int k = 1; k < 4; ++k)
        BOOST_CHECK_EQUAL(p.mask_frame(k, seq0 + k * p.span), seq0 + k * p.span >= switch_seq);
    for (int k = 0; k < 4; ++k) {
        p.accumulate_frame(k, seq0 + k * p.span);
        BOOST_REQUIRE(p.vis.wait_for_full_frame("test", k));
        auto meta = get_N2_metadata(&p.vis, k);
        BOOST_CHECK_EQUAL(meta->fpga_start_tick, seq0 + k * p.span);
        BOOST_CHECK_EQUAL(meta->n_dtv_mask_enabled_fpga_ticks,
                          seq0 + k * p.span >= switch_seq ? p.span : 0);
        p.vis.mark_frame_empty("test", k);
    }
}

BOOST_AUTO_TEST_CASE(detect_only_bundle_refuses_the_mask_on) {
    BOOST_CHECK_THROW(SwitchPipeline(true, 0, false), std::runtime_error);
    std::remove(("/tmp/test_dtv_mask_enabled_" + std::to_string(getpid()) + ".json").c_str());
}

BOOST_AUTO_TEST_CASE(config_value_applies_from_startup) {
    // On from an hour ahead, and off from before the stream start with a detect-only bundle
    const int64_t later = N2::current_system_time_ns() + 3'600'000'000'000;
    for (const bool enabled : {true, false}) {
        SwitchPipeline p(enabled, enabled ? later : 1, enabled);
        p.mask->start();
        p.accumulate->start();
        BOOST_CHECK_EQUAL(p.mask_frame(0, 0), enabled);
        p.accumulate_frame(0, 0);
        BOOST_REQUIRE(p.vis.wait_for_full_frame("test", 0));
        BOOST_CHECK_EQUAL(get_N2_metadata(&p.vis, 0)->n_dtv_mask_enabled_fpga_ticks,
                          enabled ? p.span : 0);
        p.vis.mark_frame_empty("test", 0);
    }
}
