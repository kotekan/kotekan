#include "DataQuality.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"         // for mark_frame_empty, Buffer, register_consumer, wait_for...
#include "chimeMetadata.h"
#include "datasetManager.hpp"    // for state_id_t, datasetManager, dset_id_t
#include "kotekanLogging.hpp"    // for DEBUG, DEBUG2
#include "prometheusMetrics.hpp" // for Gauge, Metrics, MetricFamily
#include "version.h"             // for get_git_commit_hash
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for freq_ctype

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <future>      // for vector
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string.h>    // for memcpy
#include <string>      // for string
#include <sys/types.h> // for uint
#include <vector>      // for vector

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

    // Apply config.
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

DataQuality::~DataQuality() {}

void DataQuality::calc_alpha_coeffs(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();

    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, ds_id);
    const stackState* ss = sstate_fut.get();

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

    // Compute alpha coefficients
    std::vector<double> alpha(ns, 0);

    for (uint32_t i = 0; i < ns; i++) {
        alpha[i] = pow(counts[i] / _num_elements, 2);
    }

    dset_id_map[ds_id] = alpha;
}

void DataQuality::main_thread() {

    frameID input_frame_id(in_buf);

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        auto frame = VisFrameView(in_buf, input_frame_id);
        dset_id_t ds_id = frame.dataset_id;

        // If the dataset has changed construct a new vector of alpha coefficients
        if (dset_id_map.count(ds_id) == 0) {
            calc_alpha_coeffs(ds_id);
        }

        // Get correct set of alpha coefficients
        const std::vector<double>& alpha = dset_id_map.at(ds_id);

        // Compute sensitivity
        double sensitivity = 0;

        for (uint32_t i = 0; i < frame.num_prod; i++) {
            auto var = (frame.weight[i] == 0 ? 0.0 : frame.weight[i]);
            sensitivity += alpha.at(i) * var;
        }

        std::vector<std::string> labels = {std::to_string(frame.freq_id)};
        data_quality_metric.labels(labels).set(sensitivity);

        // Finish up iteration.
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id++);

    } // end stop thread
}
