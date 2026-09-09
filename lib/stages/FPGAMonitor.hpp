/**
 * @file
 * @brief Stage that polls the upstream FPGA controller for drift.
 *
 * The ConfigTracker fetches the FPGA controller's config and timing once at
 * startup and treats them as fixed for the life of the pipeline: every
 * downstream node is handed that snapshot, and data written to disk is
 * labelled with it. Nothing re-reads the controller after that, so a
 * controller that is reprogrammed or resynced mid-acquisition leaves the whole
 * pipeline quietly describing data it no longer produced.
 *
 * This stage closes that gap: it re-reads the same two endpoints on an
 * interval and compares them against the tracker's record.
 *
 * - FPGAMonitor : public kotekan::Stage
 */
#ifndef FPGA_MONITOR_HPP
#define FPGA_MONITOR_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "bufferContainer.hpp"
#include "configTracker.hpp"
#include "prometheusMetrics.hpp"

#include <stdint.h>
#include <string>

/**
 * @class FPGAMonitor
 * @brief Periodically checks that the FPGA controller still matches what the
 *        ConfigTracker recorded at startup.
 *
 * Requires the config tracker to be enabled and to have registered an FPGA
 * controller (i.e. ``/config_tracker/fpga_host_info`` is set); it polls the
 * controller registered there, at the address the tracker resolved, using the
 * same ``config_endpoint`` and ``timing_endpoint``.
 *
 * A poll has three outcomes. It matches; the controller could not be read; or
 * the controller answered with something other than what was recorded. Only
 * the last is a real deviation, and by default it is fatal, because every
 * config the tracker has already handed downstream is now wrong. Failing to
 * read the controller is tolerated by default (a REST server restart or a
 * blip should not take down an acquisition), but is escalated to an error
 * after `max_consecutive_failures` polls in a row.
 *
 * @par Metrics
 * @metric kotekan_fpga_monitor_matches_record  1 while the controller matches
 *         the tracker's record, 0 once a deviation has been seen.
 * @metric kotekan_fpga_monitor_polls_total     Polls, labelled by `result`
 *         (`ok`, `changed`, `unreachable`).
 * @metric kotekan_fpga_monitor_last_ok_timestamp_seconds
 *         Monotonic time of the last poll that matched, on the same clock as
 *         kotekan_config_tracker_last_change_timestamp_seconds.
 *
 * @conf poll_interval_seconds     Double. Default 5. Seconds between polls.
 * @conf fetch_timeout_seconds     Int. Default 5. HTTP timeout per request.
 *                                 Keep it below poll_interval_seconds.
 * @conf fetch_retries             Int. Default 0. Retries per request. A poll
 *                                 that misses is retried on the next tick
 *                                 anyway, so retrying inside one is rarely
 *                                 what you want.
 * @conf fatal_on_change           Bool. Default true. Exit kotekan when the
 *                                 controller no longer matches the record.
 * @conf fatal_on_unreachable      Bool. Default false. Exit kotekan once the
 *                                 controller has been unreadable for
 *                                 `max_consecutive_failures` polls.
 * @conf max_consecutive_failures  Int. Default 3. Failed polls in a row before
 *                                 the stage escalates from a warning to an
 *                                 error (and, if `fatal_on_unreachable`, exits).
 *
 * @author James Mertens
 */
class FPGAMonitor : public kotekan::Stage {
public:
    FPGAMonitor(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~FPGAMonitor() override;

    void main_thread() override;

private:
    /// Run one poll and act on the outcome. Returns false if the stage should stop.
    bool poll_once();

    /// Seconds between polls.
    double _poll_interval_seconds;
    /// Per-request HTTP timeout, in seconds.
    int _fetch_timeout_seconds;
    /// Per-request retry count.
    int _fetch_retries;
    /// Whether a deviation from the record is fatal.
    bool _fatal_on_change;
    /// Whether a persistently unreachable controller is fatal.
    bool _fatal_on_unreachable;
    /// Consecutive failed polls tolerated before escalating.
    int _max_consecutive_failures;

    /// The controller being polled, as the tracker resolved it.
    kotekan::ConfigTracker::FpgaEndpoint _endpoint;

    /// Consecutive polls that failed to read the controller.
    int _consecutive_failures = 0;
    /// Set once a deviation has been reported, so it is logged once, not every poll.
    bool _deviation_reported = false;

    kotekan::prometheus::Gauge& _matches_record_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& _polls_metric;
    kotekan::prometheus::Gauge& _last_ok_metric;
};

#endif // FPGA_MONITOR_HPP
