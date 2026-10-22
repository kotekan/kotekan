#define BOOST_TEST_MODULE "test_N2Accumulate"

#include "Config.hpp"
#include "N2Accumulate.hpp"
#include "N2Metadata.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "configUpdater.hpp"
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "metadata.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kotekan;
using json = nlohmann::json;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

BOOST_AUTO_TEST_CASE(_fringestop_update_needs_phase_center) {
    json cfg;
    cfg["log_level"] = "WARN";
    cfg["num_polarizations"] = 2;
    cfg["num_dishes"] = 2;
    // No dish_coelev_deg, so the phase center is not set.
    cfg["telescope"] = {{"name", "CHORDTelescope"},
                        {"dish_grid_size_x", 2},
                        {"dish_grid_size_y", 1},
                        {"dish_inputs", json::array()},
                        {"require_gps", false}};

    json fringestop;
    fringestop["kotekan_update_endpoint"] = "json";
    fringestop["enabled"] = true;
    fringestop["valid_at_time_ns"] = 0;
    cfg["fringestop_enabled"] = fringestop;

    const std::vector<std::string> buf_names = {
        "in_buf",          "in_counts_buf",       "in_rficounts_buf",
        "in_plcounts_buf", "in_rfiframemask_buf", "out_buf"};
    json stage;
    stage["kotekan_stage"] = "N2Accumulate";
    stage["log_level"] = "WARN";
    stage["cpu_affinity"] = std::vector<int>{0};
    stage["num_freq_per_n2k_frame"] = 1;
    stage["packet_loss_is_scalar"] = true;
    stage["samples_per_data_set"] = 256;
    stage["sub_integration_ntime"] = 128;
    stage["variance_mode"] = "EvenOddPosDef";
    stage["fringestop_updatable_config"] = "/fringestop_enabled";
    for (const auto& name : buf_names)
        stage[name] = name;
    cfg["accumulate"] = stage;

    Config conf;
    conf.update_config(cfg);
    configUpdater::instance().apply_config(conf);
    Telescope::instance(conf);
    BOOST_REQUIRE(!Telescope::instance().phase_center_is_set());

    auto pool =
        metadataPool::create(2 * buf_names.size(), sizeof(N2Metadata), "test_pool", "N2Metadata");
    std::vector<std::unique_ptr<Buffer>> bufs;
    bufferContainer bc;
    for (const auto& name : buf_names) {
        bufs.push_back(std::make_unique<Buffer>(2, 1024, pool, name,
                                                name == "out_buf" ? "N2" : "standard", 0, false,
                                                false, std::vector<int>{}, true));
        bc.add_buffer(name, bufs.back().get());
    }

    // configUpdater throws when the subscriber refuses the initial value.
    BOOST_CHECK_EXCEPTION(N2Accumulate(conf, "/accumulate", bc), std::runtime_error,
                          [](const std::runtime_error& e) {
                              return std::string(e.what()).find("initial value at endpoint "
                                                                "'/fringestop_enabled'")
                                     != std::string::npos;
                          });
}
