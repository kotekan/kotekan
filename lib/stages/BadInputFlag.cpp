#include "BadInputFlag.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for operator!=, operator==, Hash, operator<
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"              // for mark_frame_empty, mark_frame_full, register_consumer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, datasetManager, fingerprint_t
#include "kotekanLogging.hpp"    // for FATAL_ERROR
#include "prometheusMetrics.hpp" // for Counter, MetricFamily, Metrics
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for frameID, modulo

#include "gsl-lite.hpp" // for span

#include <algorithm>  // for copy, copy_backward, equal, max
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cmath>      // for isinf, isnan
#include <deque>      // for deque
#include <functional> // for _Bind_helper<>::type, bind, function
#include <future>     // for future
#include <optional>   // for optional
#include <stddef.h>   // for size_t


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(BadInputFlag);

BadInputFlag::BadInputFlag(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&BadInputFlag::main_thread, this)),
    bad_input_counter(Metrics::instance().add_counter("kotekan_badinputflag_frames_total",
                                                      unique_name, {"input_id", "type"})) {

    // Get buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void BadInputFlag::check_dataset_state(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();

    // If any of these states have changed the interpretation of the vis/weight elements
    // will have changed, in that case we need to exit as things will end up being
    // inconsistent
    // TODO: this limitation could be fixed by doing a bunch of work interpreting the
    // products and matching up to specific inputs, but that's for another time
    auto fprint = dm.fingerprint(ds_id, {"inputs", "products", "stack"});
    if (fingerprint == fingerprint_t::null) {

        // Check that this is not a stacked dataset, if it is we can't look at
        // individual autocorrelations and so this stage won't work
        if (dm.closest_dataset_of_type(ds_id, "stack")) {
            FATAL_ERROR("This dataset {} has a stackState, but we need the full triangle.", ds_id);
        }

        fingerprint = fprint;
    }

    if (fprint != fingerprint) {
        FATAL_ERROR("This dataset {} has a different fingerprint {} from expected.", ds_id, fprint,
                    fingerprint);
    }

    dset_id_set.insert(ds_id);
}

void BadInputFlag::main_thread() {


    frameID input_frame_id(in_buf);
    frameID output_frame_id(out_buf);

    std::future<void> change_dset_fut;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Wait for the output buffer frame to be free
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = VisFrameView::copy_frame(in_buf, input_frame_id, out_buf, output_frame_id);

        // check if the input dataset has changed
        if (dset_id_set.count(frame.dataset_id) == 0) {
            check_dataset_state(frame.dataset_id);
        }

        size_t num_elements = frame.num_elements;
        size_t num_prod = frame.num_prod;

        assert(num_prod == num_elements * (num_elements + 1) / 2);

        // Iterate over autocorrelations
        size_t auto_ind = 0;
        for (size_t i = 0; i < num_elements; i++) {

            // If the input is listed as valid, check to see if the corresponding
            // weights are bad (i.e. infinite or NaN). If so, flag the input and export
            // it to Prometheus
            if (frame.flags[i] != 0) {
                if (std::isinf(frame.weight[auto_ind])) {
                    bad_input_counter.labels({std::to_string(i), "Inf"}).inc();
                    // TODO: post dataset state changes and turn this on
                    // frame.flags[i] = 0;

                } else if (std::isnan(frame.weight[auto_ind])) {
                    bad_input_counter.labels({std::to_string(i), "NaN"}).inc();
                    // TODO: post dataset state changes and turn this on
                    // frame.flags[i] = 0;
                }
            }

            auto_ind += (num_elements - i);
        }

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id++);
    }
}
