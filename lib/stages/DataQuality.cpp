#include "DataQuality.hpp"

#include "Hash.hpp"              // for operator<
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "datasetManager.hpp"    // for fingerprint_t, datasetManager, dset_id_t
#include "datasetState.hpp"      // for stackState
#include "kotekanLogging.hpp"    // for FATAL_ERROR
#include "prometheusMetrics.hpp" // for Gauge, Metrics, MetricFamily
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for frameID, modulo

#include <algorithm>    // for copy, copy_backward, equal, max
#include <atomic>       // for atomic_bool
#include <deque>        // for deque
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <gsl-lite.hpp> // for span
#include <math.h>       // for pow
#include <stdexcept>    // for out_of_range
#include <stdint.h>     // for uint32_t
#include <string.h>     // for size_t
#include <string>       // for string, to_string
#include <vector>       // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(DataQuality);

DataQuality::DataQuality(Config& config_, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&DataQuality::main_thread, this)),
    data_quality_metric(Metrics::instance().add_gauge("kotekan_dataquality_dataquality",
                                                      unique_name, {"freq_id"})) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

DataQuality::~DataQuality() {}

void DataQuality::calc_alpha_coeffs(fingerprint_t fprint, dset_id_t ds_id) {

    auto& dm = datasetManager::instance();
    const stackState* ss = dm.dataset_state<stackState>(ds_id);

    if (ss == nullptr) {
        FATAL_ERROR("Couldn't find stackState ancestor of dataset "
                    "{}. Make sure there is a stage upstream in the config, that adds a "
                    "freqState.\nExiting...",
                    ds_id);
    }

    auto ns = ss->get_num_stack();
    std::vector<size_t> counts(ns, 0);

    // Calculate the no. of visibilities averaged into each stack
    for (auto [ind, conj] : ss->get_rstack_map()) {
        (void)conj;
        if (ind >= ns)
            continue;

        counts[ind]++;
    }

    double total = 0.0;
    for (auto count : counts)
        total += count;

    // Compute alpha coefficients
    std::vector<double> alpha(ns, 0);

    for (uint32_t i = 0; i < ns; i++) {
        alpha[i] = pow(counts[i] / total, 2);
    }

    // Insert alpha coefficients into map
    fprint_map[fprint] = alpha;
}

void DataQuality::main_thread() {

    frameID input_frame_id(in_buf);
    auto& dm = datasetManager::instance();

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        auto frame = VisFrameView(in_buf, input_frame_id);
        dset_id_t ds_id = frame.dataset_id;

        // Store datasets and fingerprints in a map
        if (dset_id_map.count(ds_id) == 0) {
            dset_id_map[ds_id] = dm.fingerprint(ds_id, {"stack"});
        }

        auto fprint = dset_id_map[ds_id];

        // If the dataset has changed construct a new vector of alpha coefficients
        if (fprint_map.count(fprint) == 0) {
            calc_alpha_coeffs(fprint, ds_id);
        }

        // Get correct set of alpha coefficients
        const std::vector<double>& alpha = fprint_map.at(fprint);

        // Compute sensitivity
        double sensitivity = 0;

        for (uint32_t i = 0; i < frame.num_prod; i++) {
            auto var = (frame.weight[i] == 0 ? 0.0 : 1.0 / frame.weight[i]);
            sensitivity += alpha.at(i) * var;
        }

        std::vector<std::string> labels = {std::to_string(frame.freq_id)};
        data_quality_metric.labels(labels).set(sensitivity);

        // Finish up iteration.
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id++);

    } // end stop thread
}
