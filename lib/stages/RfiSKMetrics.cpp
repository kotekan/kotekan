#include "RfiSKMetrics.hpp"

#include "N2Util.hpp"       // for frameID
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "restServer.hpp"   // for restServer, connectionInstance

#include <algorithm>  // for fill
#include <cmath>      // for isnan
#include <functional> // for bind
#include <json.hpp>   // for json
#include <limits>     // for numeric_limits
#include <mutex>      // for lock_guard, mutex

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::restServer;
using kotekan::Stage;
using kotekan::prometheus::Metrics;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(RfiSKMetrics);

RfiSKMetrics::RfiSKMetrics(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RfiSKMetrics::main_thread, this)),
    num_freq(config.get<size_t>(unique_name, "num_local_freq")),
    num_times(config.get<size_t>(unique_name, "rfi_num_times_bar")),
    num_elements(config.get<size_t>(unique_name, "num_polarizations")
                 * config.get<size_t>(unique_name, "num_dishes")),
    ema_frames(config.get_default<size_t>(unique_name, "ema_frames", 256)),
    ema_alpha(1.0 / (double)ema_frames),
    ema_sk(num_elements, std::numeric_limits<double>::quiet_NaN()),
    ema_valid_frac(num_elements, 0.0) {

    // labels() scans the family under a lock, so look each gauge up once here
    // and keep the references (stable: metrics live in a deque, never removed).
    auto& sk_metric =
        Metrics::instance().add_gauge("kotekan_rfi_sk_per_feed", unique_name, {"element"});
    auto& valid_frac_metric = Metrics::instance().add_gauge("kotekan_rfi_sk_per_feed_valid_frac",
                                                            unique_name, {"element"});
    for (size_t e = 0; e < num_elements; ++e) {
        sk_gauges.push_back(&sk_metric.labels({std::to_string(e)}));
        valid_frac_gauges.push_back(&valid_frac_metric.labels({std::to_string(e)}));
    }

    restServer::instance().register_get_callback(
        unique_name + "/sk", std::bind(&RfiSKMetrics::send_sk, this, std::placeholders::_1));

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
}

RfiSKMetrics::~RfiSKMetrics() {
    // Must manually remove GET callbacks
    restServer::instance().remove_get_callback(unique_name + "/sk");
}

void RfiSKMetrics::send_sk(connectionInstance& conn) {
    nlohmann::json reply;
    reply["num_elements"] = num_elements;
    reply["ema_frames"] = ema_frames;
    nlohmann::json sk = nlohmann::json::array();
    nlohmann::json valid_frac = nlohmann::json::array();
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (size_t e = 0; e < num_elements; e++) {
            if (std::isnan(ema_sk[e]))
                sk.push_back(nullptr); // no valid measurement yet
            else
                sk.push_back(ema_sk[e]);
            valid_frac.push_back(ema_valid_frac[e]);
        }
    }
    reply["sk"] = sk;
    reply["valid_frac"] = valid_frac;
    conn.send_json_reply(reply);
}

void RfiSKMetrics::main_thread() {
    frameID frame_id(in_buf);

    std::vector<double> sum(num_elements);
    std::vector<size_t> count(num_elements);

    while (!stop_thread) {
        const float* frame = (const float*)in_buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);

        // Frame layout: [num_times][num_freq][3 = {SK, b, sigma}][num_elements].
        for (size_t tf = 0; tf < num_times * num_freq; tf++) {
            const float* sk = frame + (tf * 3 + 0) * num_elements;
            const float* sigma = frame + (tf * 3 + 2) * num_elements;
            for (size_t e = 0; e < num_elements; e++) {
                if (sigma[e] >= 0) { // a negative sigma marks an invalid (masked) cell
                    sum[e] += sk[e];
                    count[e]++;
                }
            }
        }

        // Per-element EMA over frames. (N2::movingAverage is integer-only —
        // add_sample(uint64_t) — so it cannot hold these fractional values.)
        const double ncells = (double)(num_times * num_freq);
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (size_t e = 0; e < num_elements; e++) {
                ema_valid_frac[e] += ema_alpha * (count[e] / ncells - ema_valid_frac[e]);
                if (count[e] > 0) {
                    double mean = sum[e] / count[e];
                    // seed with the first valid measurement, then track
                    ema_sk[e] =
                        std::isnan(ema_sk[e]) ? mean : ema_sk[e] + ema_alpha * (mean - ema_sk[e]);
                }
            }
        }
        for (size_t e = 0; e < num_elements; e++) {
            if (!std::isnan(ema_sk[e]))
                sk_gauges[e]->set(ema_sk[e]);
            valid_frac_gauges[e]->set(ema_valid_frac[e]);
        }

        in_buf->mark_frame_empty(unique_name, frame_id);
        frame_id++;
    }
}
