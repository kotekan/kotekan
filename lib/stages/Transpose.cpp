#include "Transpose.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator!=
#include "SystemInterface.hpp"   // for get_hostname, get_username
#include "buffer.hpp"            // for wait_for_full_frame, mark_frame_empty, register_consumer
#include "bufferContainer.hpp"   // for bufferContainer
#include "dataset.hpp"           // for dataset
#include "datasetManager.hpp"    // for dset_id_t, datasetManager
#include "datasetState.hpp"      // for metadataState, stackState, acqDatasetIdState, eigenvalu...
#include "errors.h"              // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"    // for DEBUG, FATAL_ERROR, logLevel, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "version.h"             // for get_git_commit_hash

#include "fmt.hpp" // for format

#include <algorithm>    // for max, fill, min
#include <atomic>       // for atomic_bool
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <future>       // for async, future
#include <map>          // for map
#include <memory>       // for allocator_traits<>::value_type
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for out_of_range, invalid_argument
#include <stdint.h>     // for uint32_t, uint64_t
#include <system_error> // for system_error


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

Transpose::Transpose(Config& config, const std::string& unique_name,
                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&Transpose::main_thread, this)),
    in_buf(get_buffer("in_buf")), frame_id(in_buf) {

    in_buf->register_consumer(unique_name);

    // Get file path to write to
    filename = config.get<std::string>(unique_name, "outfile");

    // Get a timeout for communication with broker
    timeout =
        std::chrono::duration<float>(config.get_default<float>(unique_name, "comet_timeout", 60.));

    // Collect some metadata. The rest is requested from the datasetManager,
    // once we received the first frame.
    metadata["notes"] = "";
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = get_username();
    metadata["collection_server"] = get_hostname();
}

void Transpose::main_thread() {

    uint32_t frames_so_far = 0;
    // frequency and time indices within chunk
    uint32_t fi = 0, ti = 0;

    // Flags to indicate incomplete chunks
    bool t_edge = false, f_edge = false;
    size_t f_ind = 0, t_ind = 0;

    // The dataset ID we read from the frame
    dset_id_t ds_id;
    // The dataset ID of the state without the time axis
    dset_id_t base_ds_id;
    // String formatted dataset ID to be written to the file
    std::string ds_id_str;

    size_t frame_size = 0;

    // wait for a non-empty frame to get dataset ID from
    uint32_t num_empty_skip = 0;
    while (true) {
        // Wait for a frame in the input buffer in order to get the dataset ID
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            return;
        }

        // Get the frame size to publish metrics later
        uint64_t fpga_seq_total;
        dset_id_t frame_ds_id;
        std::tie(frame_size, fpga_seq_total, frame_ds_id) = get_frame_data();

        // If the frame is empty, release the buffer and continue
        if (fpga_seq_total == 0 && frame_ds_id == dset_id_t::null) {
            DEBUG("Got empty frame ({:d}).", frame_id);
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);
            num_empty_skip++;
        } else {
            ds_id = frame_ds_id;
            break;
        }
    }

    if (num_empty_skip > 0) {
        INFO("Found {:d} empty frames at the start of the file.", num_empty_skip);
    }

    if (!get_dataset_state(ds_id)) {
        FATAL_ERROR("Couldn't find ancestor of dataset {}. "
                    "Make sure there is a stage upstream in the config, that adds the dataset "
                    "states.\nExiting...",
                    ds_id);
        return;
    }

    // Get the original dataset ID (before adding time axis)
    base_ds_id = base_dset(ds_id);

    found_flags = std::vector<bool>(write_t, false);

    // Create HDF5 file
    create_hdf5_file();

    auto& transposed_bytes_metric =
        Metrics::instance().add_counter("kotekan_transpose_data_transposed_bytes", unique_name);

    while (!stop_thread) {
        if (num_empty_skip > 0) {
            // Write out empty frames that were skipped at start
            // All arrays are initialised to zero, so we just need to move through them
            num_empty_skip--;
        } else {
            // Wait for a full frame in the input buffer
            if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
                break;
            }

            // Collect frames until a chunk is filled
            // Time-transpose as frames come in
            // Fastest varying is time (needs to be consistent with reader!)
            copy_frame_data(fi, ti);

            auto [frame_size, fpga_seq_total, frame_ds_id] = get_frame_data();
            (void)frame_size;

            // Offset into dset_id list
            uint32_t offset = fi * write_t;

            // Parse the dataset ID
            if (fpga_seq_total == 0 && frame_ds_id == dset_id_t::null) {
                DEBUG2("Got an empty frame.");
                // Empty frames have a null dataset ID
                ds_id_str = fmt::format("{}", dset_id_t::null);
            } else if (frame_ds_id != ds_id) {
                // TODO assuming that dataset ID changes here never change dataset dimensions
                DEBUG("Dataset ID has changed from {} to {}.", ds_id, frame_ds_id);
                // Update the dataset ID we are writing out
                ds_id = frame_ds_id;
                // Store original dataset ID (before adding time axis)
                base_ds_id = base_dset(ds_id);
                ds_id_str = fmt::format("{}", base_ds_id);
            } else {
                // Dataset ID hasn't changed
                ds_id_str = fmt::format("{}", base_ds_id);
            }
            if (ds_id_str.length() != DSET_ID_LEN - 1) {
                FATAL_ERROR("Formatted dataset ID string does not have expected length.");
                return;
            }
            std::copy(ds_id_str.c_str(), ds_id_str.c_str() + DSET_ID_LEN,
                      dset_id[offset + ti].hash);

            // Only copy flags if we haven't already
            copy_flags(ti);

            // move to next frame
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);
        }

        // Increment within read chunk
        // within a chunk, frequency is the fastest varying index
        fi = (fi + 1) % write_f;
        if (fi == 0)
            ti++;
        if (ti == write_t) {
            // chunk is complete
            write_chunk(t_ind, f_ind);
            // increment between chunks
            increment_chunk(t_ind, f_ind, t_edge, f_edge);
            fi = 0;
            ti = 0;

            // export prometheus metric
            transposed_bytes_metric.inc(frame_size);
        }

        frames_so_far++;
        // Exit when all frames have been written
        if (frames_so_far == num_time * num_freq) {
            INFO("Done. Exiting.");
            exit_kotekan(ReturnCode::CLEAN_EXIT);
            return;
        }
    }
}

dset_id_t Transpose::base_dset(dset_id_t ds_id) {

    datasetManager& dm = datasetManager::instance();

    try {
        return dm.datasets().at(ds_id).base_dset();
    } catch (std::out_of_range& e) {
        DEBUG("Fetching metadata state...");
        // fetch a metadata state just to ensure we have a copy of that dataset
        auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>, &dm, ds_id);
        auto ready = mstate_fut.wait_for(timeout);
        if (ready == std::future_status::timeout) {
            ERROR("Communication with dataset broker timed out for datatset id {}.", ds_id);
            dm.stop();
            exit_kotekan(ReturnCode::DATASET_MANAGER_FAILURE);
            return ds_id;
        }
        const metadataState* mstate = mstate_fut.get();
        (void)mstate;
        try {
            return dm.datasets().at(ds_id).base_dset();
        } catch (std::out_of_range& e) {
            FATAL_ERROR("Failed to get base dataset of dataset with ID {}. {}", ds_id, e.what());
            return ds_id;
        }
    }
}
