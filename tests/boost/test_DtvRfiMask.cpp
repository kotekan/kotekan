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
#include <memory>
#include <thread>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

struct MaskPipeline {
    static constexpr int F = 4;
    const int samples;
    kotekan::Config config;
    std::shared_ptr<metadataPool> pool;
    Buffer rfi, dtv, support, output;
    kotekan::bufferContainer buffers;
    std::unique_ptr<kotekan::Stage> stage;

    MaskPipeline(int samples, bool fine = true) :
        samples(samples),
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
             {{"rfi_buf", "rfi"},
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
        dtv.ensure_frame_desc(
            kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask", {F}, {"F"}, {1}));
        if (fine)
            support.ensure_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::int32, "dtv_fine_support", {F, 2}, {"F", "S"}, {1, 1}));
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

    void fill(int index) {
        const int slot = index % 2;
        for (auto* b : {&rfi, &dtv, &support}) {
            if (b == &support && !support.get_frame_desc())
                continue;
            BOOST_REQUIRE(b->wait_for_empty_frame("test", slot));
            b->allocate_new_metadata_object(slot);
            auto meta = get_chord_metadata(b, slot);
            if (b != &support || support.get_frame_desc())
                meta->set_from_frame_desc(b->get_frame_desc<kotekan::GenericNDArray>());
            meta->set_fpga_seq_num((int64_t(1) << 34) + int64_t(index) * samples * 4);
            meta->set_time_downsampling_fpga(b == &rfi ? 4096 : samples * 4);
            meta->set_coarse_freq({10, 11, 12, 13});
            meta->set_freq_upchan_factor({1, 1, 1, 1});
            meta->set_freq_upchan_index({0, 0, 0, 0});
        }
        for (size_t j = 0; j < rfi.frame_size; ++j)
            rfi.frames[slot][j] = (j + index) % 251;
        const uint8_t reject[F] = {0, 1, 0, 0};
        const int32_t fine[F * 2] = {-1, -1, 1, 124, -1, -1, index % 2, 124};
        std::memcpy(dtv.frames[slot], reject, sizeof(reject));
        std::memcpy(support.frames[slot], fine, sizeof(fine));
        for (auto* b : {&rfi, &dtv, &support})
            if (b != &support || support.get_frame_desc())
                b->mark_frame_full("test", slot);
    }
};

BOOST_AUTO_TEST_CASE(mask_and_metadata_across_wrap) {
    for (int samples : {8192, 16384})
        for (bool fine : {false, true}) {
            MaskPipeline p(samples, fine);
            p.stage->start();
            for (int index = 0; index < 8; ++index) {
                p.fill(index);
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (p.output.get_num_full_frames() == 0
                       && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                BOOST_REQUIRE_GT(p.output.get_num_full_frames(), 0);
                const int slot = index % 2;
                auto* data = p.output.wait_for_full_frame("test", slot);
                BOOST_REQUIRE(data);
                for (size_t j = 0; j < p.output.frame_size; ++j) {
                    const int f = (j / 128) % MaskPipeline::F;
                    const bool excluded = f == 1 || f == 2 || (fine && f == 3 && index % 2 == 0);
                    BOOST_CHECK_EQUAL(data[j], excluded ? 0 : (j + index) % 251);
                }
                auto meta = get_chord_metadata(&p.output, slot);
                BOOST_CHECK_EQUAL(meta->get_fpga_seq_num(),
                                  (int64_t(1) << 34) + int64_t(index) * samples * 4);
                BOOST_CHECK_EQUAL(meta->get_time_downsampling_fpga(), 4096);
                BOOST_CHECK(meta->get_coarse_freq() == std::vector<int>({10, 11, 12, 13}));
                p.output.mark_frame_empty("test", slot);
            }
        }
}

BOOST_AUTO_TEST_CASE(reject_malformed_input) {
    for (int fault = 0; fault < 6; ++fault) {
        MaskPipeline p(8192);
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
        }
        BOOST_CHECK_THROW(p.stage->main_thread(), FatalError);
    }
}
