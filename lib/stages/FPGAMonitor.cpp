#include "FPGAMonitor.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "configTracker.hpp"  // for ConfigTracker
#include "kotekanLogging.hpp" // for FATAL_ERROR, INFO, WARN, ERROR, DEBUG

#include "fmt.hpp" // for compile_string_to_view

#include <chrono>     // for duration, steady_clock, milliseconds
#include <functional> // for bind
#include <thread>     // for sleep_for

using kotekan::Config;
using kotekan::ConfigTracker;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(FPGAMonitor);

FPGAMonitor::FPGAMonitor(Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&FPGAMonitor::main_thread, this)),
    _matches_record_metric(
        Metrics::instance().add_gauge("kotekan_fpga_monitor_matches_record", unique_name)),
    _polls_metric(Metrics::instance().add_counter("kotekan_fpga_monitor_polls_total", unique_name,
                                                  {"result"})),
    _last_ok_metric(Metrics::instance().add_gauge("kotekan_fpga_monitor_last_ok_timestamp_seconds",
                                                  unique_name)) {

    _poll_interval_seconds = config.get_default<double>(unique_name, "poll_interval_seconds", 5.0);
    if (_poll_interval_seconds <= 0.0)
        FATAL_ERROR("poll_interval_seconds must be positive, got {}", _poll_interval_seconds);

    _fetch_timeout_seconds = config.get_default<int>(unique_name, "fetch_timeout_seconds", 5);
    _fetch_retries = config.get_default<int>(unique_name, "fetch_retries", 0);
    _fatal_on_change = config.get_default<bool>(unique_name, "fatal_on_change", true);
    _fatal_on_unreachable = config.get_default<bool>(unique_name, "fatal_on_unreachable", false);
    _max_consecutive_failures = config.get_default<int>(unique_name, "max_consecutive_failures", 3);
    if (_max_consecutive_failures < 1)
        FATAL_ERROR("max_consecutive_failures must be at least 1, got {}",
                    _max_consecutive_failures);

    if (!ConfigTracker::instance().is_enabled())
        FATAL_ERROR("FPGAMonitor needs the config tracker: add a /config_tracker block naming the "
                    "controller with `fpga_host_info`.");

    // Registered by ConfigTracker::applyConfig, which kotekanMode runs before
    // it builds any stage, so the endpoint is already known here.
    auto endpoint = ConfigTracker::instance().getFpgaEndpoint();
    if (!endpoint.has_value())
        FATAL_ERROR("FPGAMonitor has no FPGA controller to poll: set "
                    "/config_tracker/fpga_host_info to the controller's host block.");
    _endpoint = *endpoint;

    INFO("Monitoring FPGA controller {}:{} ({} and {}) every {}s", _endpoint.host, _endpoint.port,
         _endpoint.config_endpoint, _endpoint.timing_endpoint, _poll_interval_seconds);

    // Start out matching: the startup fetch is what defines the record.
    _matches_record_metric.set(1.0);
}

FPGAMonitor::~FPGAMonitor() {}

bool FPGAMonitor::poll_once() {
    const ConfigTracker::FpgaCheckResult result =
        ConfigTracker::instance().checkFpgaTracking(_fetch_retries, _fetch_timeout_seconds);

    switch (result.status) {
        case ConfigTracker::FpgaCheckStatus::ok:
            _polls_metric.labels({"ok"}).inc();
            _last_ok_metric.set(
                std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            if (_consecutive_failures > 0) {
                INFO("FPGA controller {}:{} readable again after {} failed poll(s); still matches "
                     "the tracker record ({})",
                     _endpoint.host, _endpoint.port, _consecutive_failures, result.recorded_hash);
                _consecutive_failures = 0;
            }
            DEBUG("FPGA controller {}:{} matches the tracker record ({})", _endpoint.host,
                  _endpoint.port, result.recorded_hash);
            return true;

        case ConfigTracker::FpgaCheckStatus::unreachable: {
            _polls_metric.labels({"unreachable"}).inc();
            _consecutive_failures++;
            if (_consecutive_failures < _max_consecutive_failures) {
                WARN("Could not read FPGA controller {}:{} ({} in a row): {}", _endpoint.host,
                     _endpoint.port, _consecutive_failures, result.detail);
                return true;
            }
            // Past the threshold: the controller has been unreadable long
            // enough that the tracker's record is no longer being confirmed
            // by anything.
            if (_fatal_on_unreachable) {
                FATAL_ERROR("FPGA controller {}:{} unreadable for {} consecutive polls: {}",
                            _endpoint.host, _endpoint.port, _consecutive_failures, result.detail);
                return false;
            }
            ERROR("FPGA controller {:s}:{:d} unreadable for {:d} consecutive polls: {:s}",
                  _endpoint.host, _endpoint.port, _consecutive_failures, result.detail);
            return true;
        }

        case ConfigTracker::FpgaCheckStatus::changed:
            _polls_metric.labels({"changed"}).inc();
            _consecutive_failures = 0;
            _matches_record_metric.set(0.0);
            if (_fatal_on_change) {
                FATAL_ERROR("FPGA controller {}:{} no longer matches the config tracker record "
                            "({}): recorded hash {}, now {}. Every config already handed "
                            "downstream describes the old controller state.",
                            _endpoint.host, _endpoint.port, result.detail, result.recorded_hash,
                            result.observed_hash);
                return false;
            }
            // Report the deviation once. It will not clear on its own: the
            // tracker's record is fixed at startup, so every later poll would
            // repeat this.
            if (!_deviation_reported) {
                _deviation_reported = true;
                ERROR("FPGA controller {:s}:{:d} no longer matches the config tracker record "
                      "({:s}): recorded hash {:s}, now {:s}. Continuing because fatal_on_change is "
                      "false; tracked configs are now stale.",
                      _endpoint.host, _endpoint.port, result.detail, result.recorded_hash,
                      result.observed_hash);
            }
            return true;

        case ConfigTracker::FpgaCheckStatus::not_tracked:
            // The constructor established that an endpoint exists, and nothing
            // clears it while the pipeline runs.
            FATAL_ERROR("FPGA controller registration disappeared from the config tracker.");
            return false;
    }
    return true;
}

void FPGAMonitor::main_thread() {
    using namespace std::chrono;
    using namespace std::chrono_literals;

    const auto interval =
        duration_cast<steady_clock::duration>(duration<double>(_poll_interval_seconds));

    // First poll one interval in, so a controller that is still settling at
    // pipeline start isn't read the instant after the tracker read it.
    auto next_poll = steady_clock::now() + interval;

    while (!stop_thread) {
        // Sleep in slices rather than for the whole interval, so shutdown does
        // not have to wait out a poll period.
        if (steady_clock::now() < next_poll) {
            std::this_thread::sleep_for(100ms);
            continue;
        }
        next_poll = steady_clock::now() + interval;

        if (!poll_once())
            break;
    }

    INFO("FPGAMonitor: exiting main thread");
}
