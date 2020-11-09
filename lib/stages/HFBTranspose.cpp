#include "HFBTranspose.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator!=
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for wait_for_full_frame, mark_frame_empty, register_consumer
#include "bufferContainer.hpp"   // for bufferContainer
#include "dataset.hpp"           // for dataset
#include "datasetManager.hpp"    // for dset_id_t, datasetManager
#include "datasetState.hpp"      // for metadataState, stackState, acqDatasetIdState, eigenvalu...
#include "errors.h"              // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"    // for DEBUG, FATAL_ERROR, logLevel, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "version.h"             // for get_git_commit_hash
#include "HFBFrameView.hpp"      // for HFBFrameView
#include "HFBFileArchive.hpp"    // for HFBFileArchive

#include "fmt.hpp"      // for format
#include "gsl-lite.hpp" // for span

#include <algorithm>    // for max, fill, min
#include <atomic>       // for atomic_bool
#include <complex>      // for complex
#include <cstdint>      // for uint32_t
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <future>       // for async, future
#include <iterator>     // for make_move_iterator, move_iterator, operator!=
#include <map>          // for map
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for out_of_range, invalid_argument
#include <stdint.h>     // for uint32_t, uint64_t
#include <sys/types.h>  // for uint
#include <system_error> // for system_error
#include <unistd.h>     // for gethostname, getlogin_r
#include <utility>      // for move, pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(HFBTranspose);

HFBTranspose::HFBTranspose(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&HFBTranspose::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // get chunk dimensions for write from config file
    chunk = config.get<std::vector<int>>(unique_name, "chunk_size");
    if (chunk.size() != 5)
        throw std::invalid_argument("Chunk size needs exactly five elements "
                                    "(has "
                                    + std::to_string(chunk.size()) + ").");
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1 || chunk[3] < 1 || chunk[4] < 1)
        throw std::invalid_argument("HFBTranspose: Config: Chunk size needs "
                                    "to be equal to or greater than one.");
    chunk_t = chunk[2];
    chunk_f = chunk[0];

    // Get file path to write to
    filename = config.get<std::string>(unique_name, "outfile");

    // Get a timeout for communication with broker
    timeout =
        std::chrono::duration<float>(config.get_default<float>(unique_name, "comet_timeout", 60.));

    // Collect some metadata. The rest is requested from the datasetManager,
    // once we received the first frame.
    metadata["archive_version"] = "3.1.0";
    metadata["notes"] = "";
    metadata["git_version_tag"] = get_git_commit_hash();
    char temp[256];
    std::string username = (getlogin_r(temp, 256) == 0) ? temp : "unknown";
    metadata["system_user"] = username;
    gethostname(temp, 256);
    std::string hostname = temp;
    metadata["collection_server"] = hostname;
}

bool HFBTranspose::get_dataset_state(dset_id_t ds_id) {

    datasetManager& dm = datasetManager::instance();

    // Get the states synchronously.
    auto tstate_fut = std::async(&datasetManager::dataset_state<timeState>, &dm, ds_id);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    auto bstate_fut = std::async(&datasetManager::dataset_state<beamState>, &dm, ds_id);
    auto sfstate_fut = std::async(&datasetManager::dataset_state<subfreqState>, &dm, ds_id);
    auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>, &dm, ds_id);

    const metadataState* mstate;
    const timeState* tstate;
    const freqState* fstate;
    const beamState* bstate;
    const subfreqState* sfstate;
    bool timed_out = fstate_fut.wait_for(timeout) == std::future_status::timeout;
    timed_out = timed_out || (mstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        mstate = mstate_fut.get();
    timed_out = timed_out || (tstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        tstate = tstate_fut.get();
    timed_out = timed_out || (fstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        fstate = fstate_fut.get();
    timed_out = timed_out || (bstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        bstate = bstate_fut.get();
    timed_out = timed_out || (sfstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        sfstate = sfstate_fut.get();
    if (timed_out) {
        ERROR("Communication with dataset broker timed out for datatset id {}.", ds_id);
        dm.stop();
        exit_kotekan(ReturnCode::DATASET_MANAGER_FAILURE);
        return false;
    }


    if (mstate == nullptr || tstate == nullptr || bstate == nullptr || fstate == nullptr
        || sfstate == nullptr) {
        FATAL_ERROR("One of the dataset states is null.");
        return false;
    }

    // TODO split instrument_name up into the real instrument name,
    // registered by HFBAccumulate (?) and a data type, registered where
    // data is written to file the first time
    metadata["instrument_name"] = mstate->get_instrument_name();
    metadata["weight_type"] = mstate->get_weight_type();

    std::string git_commit_hash_dataset = mstate->get_git_version_tag();

    // TODO: enforce this if build type == release?
    if (git_commit_hash_dataset != metadata["git_version_tag"].get<std::string>())
        INFO("Git version tags don't match: dataset {} has tag {:s},"
             "while the local git version tag is {:s}",
             ds_id, git_commit_hash_dataset, metadata["git_version_tag"].get<std::string>());

    times = tstate->get_times();
    beams = bstate->get_beams();
    sub_freqs = sfstate->get_subfreqs();

    // unzip the vector of pairs in freqState
    auto freq_pairs = fstate->get_freqs();
    for (auto it = std::make_move_iterator(freq_pairs.begin()),
              end = std::make_move_iterator(freq_pairs.end());
         it != end; ++it) {
        freqs.push_back(std::move(it->second));
    }

    num_time = times.size();
    num_freq = freqs.size();
    num_beams = beams.size();
    num_subfreq = sub_freqs.size();

    // the dimension of the visibilities is different for stacked data
    eff_data_dim = num_beams * 128; // num_subfreq???

    DEBUG("Dataset {} has {:d} times, {:d} frequencies, {:d} beams, {:d} sub-frequencies", ds_id, num_time, num_freq, num_beams, num_subfreq, eff_data_dim);

    // Ensure chunk_size not too large
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

    // Allocate memory for collecting frames
    hfb.resize(chunk_t * chunk_f * eff_data_dim, 0.);
    hfb_weight.resize(chunk_t * chunk_f * eff_data_dim, 0.);
    // init frac_lost to 1.0 to match empty frames
    frac_lost.resize(chunk_t * chunk_f, 1.);
    frac_rfi.resize(chunk_t * chunk_f, 0.);
    dset_id.resize(chunk_t * chunk_f);

    // Initialise dataset ID array with null IDs
    std::string null_ds_id = fmt::format("{}", dset_id_t::null);
    for (auto& ds : dset_id) {
        std::copy(null_ds_id.c_str(), null_ds_id.c_str() + DSET_ID_LEN, ds.hash);
    }

    return true;
}

void HFBTranspose::main_thread() {

    frameID frame_id(in_buf);
    uint32_t frames_so_far = 0;
    // frequency and time indices within chunk
    uint32_t fi = 0;
    uint32_t ti = 0;
    // offset for copying into buffer
    uint32_t offset = 0;

    // The dataset ID we read from the frame
    dset_id_t ds_id;
    // The dataset ID of the state without the time axis
    dset_id_t base_ds_id;
    // String formatted dataset ID to be written to the file
    std::string ds_id_str;

    uint64_t frame_size = 0;

    // wait for a non-empty frame to get dataset ID from
    uint32_t num_empty_skip = 0;
    while (true) {
        // Wait for a frame in the input buffer in order to get the dataset ID
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            return;
        }
        auto frame = HFBFrameView(in_buf, frame_id);

        // Get the frame size to publish metrics later
        if (frame_size == 0)
            frame_size = frame.calculate_buffer_layout(num_beams, num_subfreq).first;

        // If the frame is empty, release the buffer and continue
        if (frame.fpga_seq_total == 0 && frame.dataset_id == dset_id_t::null) {
            DEBUG("Got empty frame ({:d}).", frame_id);
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);
            num_empty_skip++;
        } else {
            ds_id = frame.dataset_id;
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

    // Once the async get_dataset_state() is done, we have all the metadata to
    // create a file.

    found_flags = std::vector<bool>(write_t, false);

    // Create HDF5 file
    //if (stack.size() > 0) {
    //    file = std::unique_ptr<HFBFileArchive>(
    //        new HFBFileArchive(filename, metadata, times, freqs, beams, sub_freqs, stack,
    //                           reverse_stack, num_ev, chunk, kotekan::logLevel(_member_log_level)));
    //} else {
    file = std::unique_ptr<HFBFileArchive>(
        new HFBFileArchive(filename, metadata, times, freqs, beams, sub_freqs, chunk,
                           kotekan::logLevel(_member_log_level)));
    //}

    // TODO: it seems like this should be a Counter?
    auto& transposed_bytes_metric =
        Metrics::instance().add_gauge("kotekan_hfbtranspose_data_transposed_bytes", unique_name);

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
            auto frame = HFBFrameView(in_buf, frame_id);

            // Collect frames until a chunk is filled
            // Time-transpose as frames come in
            // Fastest varying is time (needs to be consistent with reader!)
            offset = fi * write_t;
            strided_copy(frame.hfb.data(), hfb.data(), offset * eff_data_dim + ti, write_t,
                         eff_data_dim);
            strided_copy(frame.weight.data(), hfb_weight.data(), offset * eff_data_dim + ti,
                         write_t, eff_data_dim);
            frac_lost[offset + ti] = frame.fpga_seq_length == 0
                                         ? 1.
                                         : 1. - float(frame.fpga_seq_total) / frame.fpga_seq_length;

            // Parse the dataset ID
            if (frame.fpga_seq_total == 0 && frame.dataset_id == dset_id_t::null) {
                DEBUG("Got an empty frame.");
                // Empty frames have a null dataset ID
                ds_id_str = fmt::format("{}", dset_id_t::null);
            } else if (frame.dataset_id != ds_id) {
                // TODO assuming that dataset ID changes here never change dataset dimensions
                DEBUG("Dataset ID has changed from {} to {}.", ds_id, frame.dataset_id);
                // Update the dataset ID we are writing out
                ds_id = frame.dataset_id;
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
            write();
            // increment between chunks
            increment_chunk();
            fi = 0;
            ti = 0;

            // export prometheus metric
            transposed_bytes_metric.set(frame_size * frames_so_far);
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

void HFBTranspose::write() {
    DEBUG("Writing at freq {:d} and time {:d}", f_ind, t_ind);
    DEBUG("Writing block of {:d} freqs and {:d} times", write_f, write_t);

    file->write_block("hfb", f_ind, t_ind, write_f, write_t, hfb.data());
    file->write_block("hfb_weight", f_ind, t_ind, write_f, write_t, hfb_weight.data());
    //file->write_block("flags/dataset_id", f_ind, t_ind, write_f, write_t, dset_id.data());
}

// increment between chunks
// chunks come in (time, freq) order
// WARNING: This order must be consistent with how HFBRawReader
//      implements chunked reads. The mechanism for avoiding
//      overwriting flags also relies on this ordering.
void HFBTranspose::increment_chunk() {
    // Figure out where the next chunk starts
    f_ind = f_edge ? 0 : (f_ind + chunk_f) % num_freq;
    if (f_ind == 0) {
        // set incomplete chunk flag
        f_edge = (num_freq < chunk_f);
        t_ind += chunk_t;
        if (num_time - t_ind < chunk_t) {
            // Reached an incomplete chunk
            t_edge = true;
        }
    } else if (num_freq - f_ind < chunk_f) {
        // Reached an incomplete chunk
        f_edge = true;
    }
    // Determine size of next chunk
    write_f = f_edge ? num_freq - f_ind : chunk_f;
    write_t = t_edge ? num_time - t_ind : chunk_t;
}

dset_id_t HFBTranspose::base_dset(dset_id_t ds_id) {

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
