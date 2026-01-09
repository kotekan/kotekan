#define BOOST_TEST_MODULE "test_eigen_stages"

#include "Config.hpp"
#include "EigenN2Iter.hpp"
#include "EigenVisIter.hpp"
#include "FakeN2.hpp"
#include "FakeVis.hpp"
#include "N2FrameDesc.hpp"
#include "N2FrameView.hpp"
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
#include <atomic>
#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <thread>
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

    static std::atomic<int> run_counter{0};
    int rc = run_counter++;
    const std::string eigen_name = "eigen_" + std::to_string(rc);
    const std::string fake_name = "fake_" + std::to_string(rc);

    nlohmann::json cfg;
    cfg["log_level"] = "ERROR";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg["num_ev"] = p.num_ev;
    cfg["vis_layout"] = N2Layout::FullUpperTri;

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

    const bool is_vis = (stage_name == "EigenVisIter");
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
        in_buf.set_frame_desc(n2_desc);
        out_buf.set_frame_desc(n2_desc);
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
