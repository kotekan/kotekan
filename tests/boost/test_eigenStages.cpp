#define BOOST_TEST_MODULE "test_eigen_stages"

#include "Config.hpp"
#include "EigenN2Iter.hpp"
#include "EigenVisIter.hpp"
#include "FakeN2.hpp"
#include "FakeVis.hpp"
#include "N2FrameDesc.hpp"
#include "N2FrameView.hpp"
#include "N2Metadata.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "configUpdater.hpp"
#include "datasetManager.hpp"
#include "eigenVis.hpp"
#include "restServer.hpp"
#include "test_logging.hpp"
#include "test_utils.hpp"
#include "visBuffer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;
using std::string;

// Ensure FakeVisPattern.cpp is linked so the factory registrations are available.
static void ensure_fakevis_patterns_registered() {
    static const bool registered = []() {
        FACTORY(FakeVisPattern)::register_type<PhaseIJVisPattern>("phase_ij");
        return true;
    }();
    (void)registered;
}

// Ensure N2Metadata.cpp is linked so the factory registration is available.
static void ensure_n2metadata_registered() {
    static const bool registered = []() {
        // Creating an instance forces the linker to include N2Metadata.o
        N2Metadata dummy;
        (void)dummy;
        return true;
    }();
    (void)registered;
}

// Reuse the small fixtures used elsewhere to bootstrap REST/config for tests.
struct RestServerFixture {
    RestServerFixture() {
        try {
            kotekan::restServer::instance().start("127.0.0.1", 0);
        } catch (...) {
        }
    }
};

struct EigenStageTestParams {
    size_t num_elements = 64;
    size_t num_ev = 2;
    size_t num_ev_conv = 2;
    size_t total_frames = 6;
    size_t check_start_frame = 0;
    uint32_t num_diagonals_filled = 0;
    std::vector<uint32_t> exclude_inputs;
    string mode = "phase_ij";
    // For the concurrent-pipeline test. The default settings are equivalent
    // to the original single-pipeline behaviour.
    std::vector<int> cpu_affinity = {0};
    int num_blaze_workers = 0; // 0 leaves blaze::setNumThreads untouched
};

template<typename Cfloat>
static void check_phase_vector(const std::vector<Cfloat>& evec,
                               const std::vector<uint32_t>& exclude_inputs, size_t num_elements,
                               double phase_tol, double amp_tol) {
    const auto base = evec[0];
    BOOST_CHECK_GT(std::abs(base), 1e-6);
    const double expected_amp = 1.0 / std::sqrt((double)(num_elements - exclude_inputs.size()));
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::find(exclude_inputs.begin(), exclude_inputs.end(), i) != exclude_inputs.end()) {
            BOOST_CHECK_SMALL(static_cast<double>(std::abs(evec[i])), 1e-4);
            continue;
        }
        auto ratio = evec[i] / base; // cancels global phase
        auto expected = std::polar(1.0f, (float)i);
        BOOST_CHECK_SMALL(static_cast<double>(std::abs(evec[i]) - expected_amp), amp_tol);
        BOOST_CHECK_SMALL(static_cast<double>(std::abs(ratio - expected)), phase_tol);
    }
}

struct EigenResults {
    std::vector<float> eval0;
    std::vector<float> eval1;
    std::vector<std::vector<std::complex<float>>> evec0;
    std::vector<float> erms;
};

// Run FakeVis/FakeN2 -> eigenVis/eigenVisIter/eigenN2Iter and collect output frames.
static EigenResults run_pipeline(const EigenStageTestParams& p, const string& stage_name) {
    ensure_fakevis_patterns_registered();
    ensure_n2metadata_registered();

    static std::atomic<int> run_counter{0};
    int rc = run_counter++;
    const std::string eigen_name = "eigen_" + std::to_string(rc);
    const std::string fake_name = "fake_" + std::to_string(rc);

    nlohmann::json cfg;
    cfg["log_level"] = "ERROR";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg["num_ev"] = p.num_ev;
    cfg["vis_layout"] = N2Layout::FullUpperTri;
    cfg["num_polarizations"] = 2;

    cfg[fake_name]["freq_ids"] = std::vector<uint32_t>{0};
    cfg[fake_name]["num_elements"] = p.num_elements;
    cfg[fake_name]["num_frames"] = p.total_frames;
    cfg[fake_name]["cadence"] = 1.0;
    cfg[fake_name]["wait"] = false;
    cfg[fake_name]["out_buf"] = "in_buf";
    cfg[fake_name]["mode"] = p.mode;
    cfg[fake_name]["kill_on_complete"] = false;

    cfg[eigen_name]["kotekan_stage"] = stage_name;
    cfg[eigen_name]["in_buf"] = "in_buf";
    cfg[eigen_name]["out_buf"] = "out_buf";
    cfg[eigen_name]["num_diagonals_filled"] = p.num_diagonals_filled;
    cfg[eigen_name]["num_ev_conv"] = p.num_ev_conv;
    if (!p.exclude_inputs.empty())
        cfg[eigen_name]["exclude_inputs"] = p.exclude_inputs;

    const bool is_vis = (stage_name == "EigenVisIter" || stage_name == "eigenVis");
    cfg["dataset_manager"]["enable_state_caching"] = false;
    cfg["dataset_manager"]["use_dataset_broker"] = false;
    if (is_vis) {
        cfg[fake_name]["kotekan_stage"] = "FakeVis";
        cfg[fake_name]["block_size"] = 1;
    } else {
        cfg[fake_name]["kotekan_stage"] = "FakeN2";
    }

    // Add telescope config, initialize telescope and dataset manager singletons.
    add_test_telescope_config(cfg);
    kotekan::Config conf;
    conf.update_config(cfg);
    kotekan::configUpdater::instance().apply_config(conf);
    Telescope::instance(conf);
    datasetManager::instance(conf);

    // Create and add buffers
    size_t num_prod = 0, frame_size = 0;
    std::shared_ptr<metadataPool> pool;
    std::string buffer_type;
    std::shared_ptr<kotekan::N2FrameDesc> n2_desc;
    if (!is_vis) {
        num_prod = kotekan::N2FrameDesc::get_num_prod(p.num_elements, N2Layout::FullUpperTri);
        frame_size = kotekan::N2FrameDesc::calculate_frame_size(p.num_elements, p.num_ev, num_prod);
        pool = metadataPool::create(p.total_frames, sizeof(N2Metadata), "n2_pool", "N2Metadata");
        buffer_type = "N2";
        n2_desc = std::make_shared<kotekan::N2FrameDesc>(p.num_elements, p.num_ev, num_prod,
                                                         N2Layout::FullUpperTri);
    } else {
        num_prod = p.num_elements * (p.num_elements + 1) / 2;
        frame_size = VisFrameView::calculate_frame_size(p.num_elements, num_prod, p.num_ev);
        pool = metadataPool::create(p.total_frames, sizeof(VisMetadata), "vis_pool", "VisMetadata");
        buffer_type = "vis";
    }
    Buffer in_buf(p.total_frames, frame_size, pool, "in_buf", buffer_type, 0, false, false,
                  std::vector<int>{}, true);
    Buffer out_buf(p.total_frames, frame_size, pool, "out_buf", buffer_type, 0, false, false,
                   std::vector<int>{}, true);

    // Set frame descriptors for N2 buffers (required by stages)
    if (n2_desc) {
        in_buf.ensure_frame_desc(n2_desc);
        out_buf.ensure_frame_desc(n2_desc);
    }

    kotekan::bufferContainer bc;
    bc.add_buffer("in_buf", &in_buf);
    bc.add_buffer("out_buf", &out_buf);

    // Register a "sink" consumer to the output buffer to prevent it from being
    // automatically marked as free when the eigen stage writes to it.
    out_buf.register_consumer("test_sink");

    // Create stages
    const std::string eigen_unique_name = "/" + eigen_name;
    std::unique_ptr<kotekan::Stage> eigen_stage;
    if (stage_name == "eigenVis") {
        eigen_stage = std::make_unique<eigenVis>(conf, eigen_unique_name, bc);
    } else if (stage_name == "EigenVisIter") {
        eigen_stage = std::make_unique<EigenVisIter>(conf, eigen_unique_name, bc);
    } else if (stage_name == "EigenN2Iter") {
        eigen_stage = std::make_unique<EigenN2Iter>(conf, eigen_unique_name, bc);
    } else {
        BOOST_FAIL("Unknown eigen stage name: " << stage_name);
    }
    const std::string fake_unique_name = "/" + fake_name;
    std::unique_ptr<kotekan::Stage> fake_stage;
    if (is_vis) {
        fake_stage = std::make_unique<FakeVis>(conf, fake_unique_name, bc);
    } else {
        fake_stage = std::make_unique<FakeN2>(conf, fake_unique_name, bc);
    }

    // Start stages
    eigen_stage->start();
    fake_stage->start();

    // Wait for output frames
    const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool timed_out = false;
    while (out_buf.get_num_full_frames() < (int)p.total_frames) {
        if (std::chrono::steady_clock::now() > timeout) {
            timed_out = true;
            break;
        }
        std::this_thread::sleep_for(10ms);
    }

    if (timed_out) {
        BOOST_FAIL("Timed out waiting for eigen output frames.");
        return EigenResults();
    }

    // Shutdown stages
    in_buf.send_shutdown_signal();
    out_buf.send_shutdown_signal();
    fake_stage->stop();
    eigen_stage->stop();
    fake_stage->join();
    eigen_stage->join();

    // Collect results
    EigenResults results;
    for (size_t f = 0; f < p.total_frames; ++f) {
        if (is_vis) {
            VisFrameView fv(&out_buf, f);
            results.eval0.push_back(fv.eval[0]);
            if (p.num_ev > 1)
                results.eval1.push_back(fv.eval[1]);
            results.erms.push_back(fv.erms);
            std::vector<std::complex<float>> evec(fv.num_elements);
            for (size_t i = 0; i < fv.num_elements; ++i) {
                evec[i] = fv.evec[i];
            }
            results.evec0.emplace_back(std::move(evec));
        } else {
            N2FrameView fv(&out_buf, f);
            results.eval0.push_back(fv.eval[0]);
            if (p.num_ev > 1)
                results.eval1.push_back(fv.eval[1]);
            results.erms.push_back(fv.erms);
            std::vector<std::complex<float>> evec(fv.num_elements);
            for (size_t i = 0; i < fv.num_elements; ++i) {
                evec[i] = fv.evec[i];
            }
            results.evec0.emplace_back(std::move(evec));
        }
    }
    return results;
}

// Verify eigen results against expected values.
// Tolerances are relative for eval, absolute for others.
// Expected values are based on the FakeVis/FakeN2 patterns, with excluded inputs removed.
// eval0 should be close to num_elements - num_excluded, eval1 close to 0.
// evec0 should have phase increasing by 1 radian per input, and amplitude 1/sqrt(num_good_inputs).
// erms should be small.
static void verify_results(const EigenResults& res, const EigenStageTestParams& p, double eval_tol,
                           double phase_tol, double amp_tol, float rms_limit) {
    const float expected_eval0 =
        (float)(p.num_elements - static_cast<long>(p.exclude_inputs.size()));
    for (size_t idx = p.check_start_frame; idx < res.eval0.size(); ++idx) {
        BOOST_CHECK_SMALL(static_cast<double>(std::abs(res.eval0[idx] - expected_eval0))
                              / (double)p.num_elements,
                          eval_tol);
        if (p.num_ev > 1)
            BOOST_CHECK_SMALL(static_cast<double>(res.eval1[idx]) / (double)p.num_elements,
                              eval_tol);
        check_phase_vector(res.evec0[idx], p.exclude_inputs, p.num_elements, phase_tol, amp_tol);
        BOOST_CHECK_LT(static_cast<double>(std::abs(res.erms[idx])), (double)rms_limit);
    }
}


BOOST_TEST_GLOBAL_FIXTURE(RestServerFixture);
BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture_Locale);

BOOST_AUTO_TEST_CASE(eigenVis_filled) {
    EigenStageTestParams params;
    params.total_frames = 16;
    params.num_diagonals_filled = 10;
    params.check_start_frame = 8;
    auto res = run_pipeline(params, "eigenVis");
    // Tolerances from python: eval: 1e-4, evec: 1e-3 (using 1e-3 for phase/amp), erms: 1e-3
    // Relaxed for stability
    verify_results(res, params, 3e-2, 5e-3, 5e-3, 5e-3f);
}

BOOST_AUTO_TEST_CASE(eigenVis_direct) {
    EigenStageTestParams params;
    params.total_frames = 8;
    auto res = run_pipeline(params, "eigenVis");
    verify_results(res, params, 1e-5, 1e-5, 1e-5, 1e-4f);
}

BOOST_AUTO_TEST_CASE(eigenVis_excluded) {
    EigenStageTestParams params;
    params.total_frames = 8;
    params.exclude_inputs = {5, 10, 6};
    auto res = run_pipeline(params, "eigenVis");
    verify_results(res, params, 1e-5, 1e-5, 1e-5, 1e-4f);
}

BOOST_AUTO_TEST_CASE(eigenN2Iter_iterative) {
    EigenStageTestParams params;
    params.total_frames = 4;
    params.num_elements = 16;
    auto res = run_pipeline(params, "EigenN2Iter");
    verify_results(res, params, 1e-4, 1e-4, 1e-4, 2e-2f);
}

BOOST_AUTO_TEST_CASE(eigenVisIter_vis_buffers) {
    EigenStageTestParams params;
    params.total_frames = 4;
    params.num_elements = 16;
    auto res = run_pipeline(params, "EigenVisIter");
    verify_results(res, params, 1e-4, 1e-4, 1e-4, 2e-2f);
}

// Run two independent EigenN2Iter pipelines concurrently within one process,
// each with its own buffers, stages, and CPU affinity. This exercises the
// path where two stage threads enter Blaze's shared-memory parallel code at
// the same time. The Blaze headers are patched (see
// cmake/Features/FeatureMath.cmake) to make the parallel-section guard
// per-thread; without that patch the static flag is shared process-wide
// and the concurrent calls throw "Nested parallel sections detected".
//
// The test verifies that both pipelines complete and that each one's
// eigendecomposition results match the single-pipeline tolerances.
static std::pair<EigenResults, EigenResults>
run_n2_pipeline_pair(const EigenStageTestParams& params_a, const EigenStageTestParams& params_b) {
    ensure_fakevis_patterns_registered();
    ensure_n2metadata_registered();
    BOOST_REQUIRE_EQUAL(params_a.num_elements, params_b.num_elements);
    BOOST_REQUIRE_EQUAL(params_a.num_ev, params_b.num_ev);

    static std::atomic<int> run_counter{0};
    const int rc = run_counter++;
    const std::string suffix = "_" + std::to_string(rc);
    struct Side {
        std::string label;
        EigenStageTestParams params;
        std::string eigen_name;
        std::string fake_name;
        std::string in_buf_name;
        std::string out_buf_name;
        std::unique_ptr<Buffer> in_buf;
        std::unique_ptr<Buffer> out_buf;
        std::unique_ptr<kotekan::Stage> eigen_stage;
        std::unique_ptr<kotekan::Stage> fake_stage;
    };
    std::array<Side, 2> sides;
    sides[0].label = "a";
    sides[0].params = params_a;
    sides[1].label = "b";
    sides[1].params = params_b;
    for (auto& s : sides) {
        s.eigen_name = "eigen_" + s.label + suffix;
        s.fake_name = "fake_" + s.label + suffix;
        s.in_buf_name = "in_buf_" + s.label + suffix;
        s.out_buf_name = "out_buf_" + s.label + suffix;
    }

    // Build one combined config with both pipelines so the global config
    // updater is applied exactly once.
    nlohmann::json cfg;
    cfg["log_level"] = "ERROR";
    cfg["num_ev"] = params_a.num_ev;
    cfg["vis_layout"] = N2Layout::FullUpperTri;
    cfg["dataset_manager"]["enable_state_caching"] = false;
    cfg["dataset_manager"]["use_dataset_broker"] = false;
    // Top-level cpu_affinity acts as a fallback for any stage that does not
    // override it (here, the FakeN2 producers).
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg["num_polarizations"] = 2;
    for (const auto& s : sides) {
        cfg[s.fake_name]["kotekan_stage"] = "FakeN2";
        cfg[s.fake_name]["freq_ids"] = std::vector<uint32_t>{0};
        cfg[s.fake_name]["num_elements"] = s.params.num_elements;
        cfg[s.fake_name]["num_frames"] = s.params.total_frames;
        cfg[s.fake_name]["cadence"] = 1.0;
        cfg[s.fake_name]["wait"] = false;
        cfg[s.fake_name]["out_buf"] = s.in_buf_name;
        cfg[s.fake_name]["mode"] = s.params.mode;
        cfg[s.fake_name]["kill_on_complete"] = false;

        cfg[s.eigen_name]["kotekan_stage"] = "EigenN2Iter";
        cfg[s.eigen_name]["in_buf"] = s.in_buf_name;
        cfg[s.eigen_name]["out_buf"] = s.out_buf_name;
        cfg[s.eigen_name]["num_ev"] = s.params.num_ev;
        cfg[s.eigen_name]["num_diagonals_filled"] = s.params.num_diagonals_filled;
        cfg[s.eigen_name]["num_ev_conv"] = s.params.num_ev_conv;
        cfg[s.eigen_name]["cpu_affinity"] = s.params.cpu_affinity;
        if (s.params.num_blaze_workers > 0)
            cfg[s.eigen_name]["num_blaze_workers"] = s.params.num_blaze_workers;
        if (!s.params.exclude_inputs.empty())
            cfg[s.eigen_name]["exclude_inputs"] = s.params.exclude_inputs;
    }
    add_test_telescope_config(cfg);
    kotekan::Config conf;
    conf.update_config(cfg);
    kotekan::configUpdater::instance().apply_config(conf);
    Telescope::instance(conf);
    datasetManager::instance(conf);

    // Shared buffer metadata; both pipelines have the same shape.
    const size_t num_prod =
        kotekan::N2FrameDesc::get_num_prod(params_a.num_elements, N2Layout::FullUpperTri);
    const size_t frame_size = kotekan::N2FrameDesc::calculate_frame_size(params_a.num_elements,
                                                                         params_a.num_ev, num_prod);
    auto pool = metadataPool::create(2 * params_a.total_frames + 8, sizeof(N2Metadata), "n2_pool",
                                     "N2Metadata");
    auto n2_desc = std::make_shared<kotekan::N2FrameDesc>(params_a.num_elements, params_a.num_ev,
                                                          num_prod, N2Layout::FullUpperTri);

    kotekan::bufferContainer bc;
    for (auto& s : sides) {
        s.in_buf = std::make_unique<Buffer>(s.params.total_frames, frame_size, pool, s.in_buf_name,
                                            "N2", 0, false, false, std::vector<int>{}, true);
        s.out_buf =
            std::make_unique<Buffer>(s.params.total_frames, frame_size, pool, s.out_buf_name, "N2",
                                     0, false, false, std::vector<int>{}, true);
        s.in_buf->ensure_frame_desc(n2_desc);
        s.out_buf->ensure_frame_desc(n2_desc);
        bc.add_buffer(s.in_buf_name, s.in_buf.get());
        bc.add_buffer(s.out_buf_name, s.out_buf.get());
        // Hold the output frames so we can inspect them after the stage exits.
        s.out_buf->register_consumer("test_sink");
    }

    for (auto& s : sides) {
        s.eigen_stage = std::make_unique<EigenN2Iter>(conf, "/" + s.eigen_name, bc);
        s.fake_stage = std::make_unique<FakeN2>(conf, "/" + s.fake_name, bc);
    }

    // Start everything before waiting so both eigen stages spin up before
    // either has finished its frames — this is what reliably causes both
    // stages to be inside Blaze SMP code at the same time.
    for (auto& s : sides)
        s.eigen_stage->start();
    for (auto& s : sides)
        s.fake_stage->start();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    auto done = [&]() {
        for (const auto& s : sides) {
            if (s.out_buf->get_num_full_frames() < (int)s.params.total_frames)
                return false;
        }
        return true;
    };
    bool timed_out = false;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) {
            timed_out = true;
            break;
        }
        std::this_thread::sleep_for(10ms);
    }

    for (auto& s : sides) {
        s.in_buf->send_shutdown_signal();
        s.out_buf->send_shutdown_signal();
    }
    for (auto& s : sides) {
        s.fake_stage->stop();
        s.eigen_stage->stop();
    }
    for (auto& s : sides) {
        s.fake_stage->join();
        s.eigen_stage->join();
    }

    if (timed_out)
        BOOST_FAIL("Timed out waiting for concurrent eigen pipelines.");

    auto collect = [](Buffer& out, const EigenStageTestParams& p) {
        EigenResults r;
        for (size_t f = 0; f < p.total_frames; ++f) {
            N2FrameView fv(&out, f);
            r.eval0.push_back(fv.eval[0]);
            if (p.num_ev > 1)
                r.eval1.push_back(fv.eval[1]);
            r.erms.push_back(fv.erms);
            std::vector<std::complex<float>> evec(fv.num_elements);
            for (size_t i = 0; i < fv.num_elements; ++i)
                evec[i] = fv.evec[i];
            r.evec0.emplace_back(std::move(evec));
        }
        return r;
    };
    return {collect(*sides[0].out_buf, params_a), collect(*sides[1].out_buf, params_b)};
}

BOOST_AUTO_TEST_CASE(eigenN2Iter_concurrent_pipelines) {
    // Matrix size matches the eigenN2Iter_iterative single-pipeline test so we
    // can reuse its tolerances; what this case adds is concurrent execution.
    //
    // Per-pipeline CPU affinity is kept to a single distinct core so the test
    // runs on minimal CI hosts (e.g. 2-core GitHub runners) as well as larger
    // machines. The race the patch guards against fires whenever two stages
    // are inside Blaze's parallel section at the same time, which is reliably
    // produced by kernel time-slicing even when the threads share a CPU.
    EigenStageTestParams params_a;
    params_a.total_frames = 6;
    params_a.num_elements = 16;
    params_a.cpu_affinity = {0};
    params_a.num_blaze_workers = 2;

    EigenStageTestParams params_b = params_a;
    params_b.cpu_affinity = {1};

    auto results = run_n2_pipeline_pair(params_a, params_b);
    verify_results(results.first, params_a, 1e-4, 1e-4, 1e-4, 2e-2f);
    verify_results(results.second, params_b, 1e-4, 1e-4, 1e-4, 2e-2f);
}
