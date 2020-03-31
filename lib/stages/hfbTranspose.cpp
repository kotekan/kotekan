#include "hfbTranspose.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator!=
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for wait_for_full_frame, mark_frame_empty, register_consumer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, datasetManager
#include "datasetState.hpp"      // for metadataState, stackState, acqDatasetIdState, eigenvalu...
#include "errors.h"              // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"    // for DEBUG, FATAL_ERROR, logLevel, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "version.h"             // for get_git_commit_hash
#include "HfbFrameView.hpp"      // for HfbFrameView
#include "hfbFileArchive.hpp"    // for hfbFileArchive

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

REGISTER_KOTEKAN_STAGE(hfbTranspose);

hfbTranspose::hfbTranspose(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&hfbTranspose::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // get chunk dimensions for write from config file
    chunk = config.get<std::vector<int>>(unique_name, "chunk_size");
    if (chunk.size() != 3)
        throw std::invalid_argument("Chunk size needs exactly three elements "
                                    "(has "
                                    + std::to_string(chunk.size()) + ").");
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1)
        throw std::invalid_argument("hfbTranspose: Config: Chunk size needs "
                                    "to be equal to or greater than one.");
    chunk_t = chunk[2];
    chunk_f = chunk[0];

    // Get file path to write to
    filename = config.get<std::string>(unique_name, "outfile");

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

bool hfbTranspose::get_dataset_state(dset_id_t ds_id) {

    datasetManager& dm = datasetManager::instance();

    // Get the states synchronously.
    // Set by hfbFileRaw.cpp
    auto tstate_fut = std::async(&datasetManager::dataset_state<timeState>, &dm, ds_id);

    // Set by integrateHFBData.cpp
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    auto bstate_fut = std::async(&datasetManager::dataset_state<beamState>, &dm, ds_id);
    auto sfstate_fut = std::async(&datasetManager::dataset_state<subfreqState>, &dm, ds_id);
    auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>, &dm, ds_id);
    
    // Set by visCompression.cpp???
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, ds_id);
    
    // Set by hfbRawReader.cpp???
    auto idstate_fut = std::async(&datasetManager::dataset_state<acqDatasetIdState>, &dm, ds_id);

    const stackState* sstate = sstate_fut.get();
    const metadataState* mstate = mstate_fut.get();
    const timeState* tstate = tstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const beamState* bstate = bstate_fut.get();
    const subfreqState* sfstate = sfstate_fut.get();
    const acqDatasetIdState* idstate = idstate_fut.get();


    if (mstate == nullptr || tstate == nullptr || fstate == nullptr || 
        idstate == nullptr)
        return false;

    // TODO split instrument_name up into the real instrument name,
    // registered by visAccumulate (?) and a data type, registered where
    // data is written to file the first time
    metadata["instrument_name"] = mstate->get_instrument_name();
    metadata["weight_type"] = mstate->get_weight_type();
    metadata["dataset_id"] = fmt::format("{}", idstate->get_id());

    std::string git_commit_hash_dataset = mstate->get_git_version_tag();

    // TODO: enforce this if build type == release?
    if (git_commit_hash_dataset != metadata["git_version_tag"].get<std::string>())
        INFO("Git version tags don't match: dataset {} has tag {:s},"
             "while the local git version tag is {:s}",
             ds_id, git_commit_hash_dataset, metadata["git_version_tag"].get<std::string>());

    times = tstate->get_times();
    beams = bstate->get_beams();

    // unzip the vector of pairs in freqState
    auto freq_pairs = fstate->get_freqs();
    for (auto it = std::make_move_iterator(freq_pairs.begin()),
              end = std::make_move_iterator(freq_pairs.end());
         it != end; ++it) {
        freqs.push_back(std::move(it->second));
    }
    
    // unzip the vector of pairs in subfreqState
    auto subfreq_pairs = sfstate->get_subfreqs();
    for (auto it = std::make_move_iterator(subfreq_pairs.begin()),
              end = std::make_move_iterator(subfreq_pairs.end());
         it != end; ++it) {
        sub_freqs.push_back(std::move(it->second));
    }

    // Check if this is baseline-stacked data
    if (sstate) {
        stack = sstate->get_stack_map();
        // TODO: verify this is where it gets stored
        reverse_stack = sstate->get_rstack_map();
    }

    num_time = times.size();
    num_freq = freqs.size();
    num_beams = beams.size();
    num_subfreq = sub_freqs.size();

    // the dimension of the visibilities is different for stacked data
    eff_data_dim = (stack.size() > 0) ? stack.size() : num_beams * 128;

    DEBUG("Dataset {} has {:d} times, {:d} frequencies, {:d} beams, {:d} sub-frequencies", ds_id, num_time, num_freq,
          num_beams, num_subfreq);

    // Ensure chunk_size not too large
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

    // Allocate memory for collecting frames
    hfb.resize(chunk_t * chunk_f * eff_data_dim);

    return true;
}

void hfbTranspose::main_thread() {

    uint32_t frame_id = 0;
    uint32_t frames_so_far = 0;
    // frequency and time indices within chunk
    uint32_t fi = 0;
    uint32_t ti = 0;
    // offset for copying into buffer
    uint32_t offset = 0;

    uint64_t frame_size = 0;

    // Wait for a frame in the input buffer in order to get the dataset ID
    if ((wait_for_full_frame(in_buf, unique_name.c_str(), 0)) == nullptr) {
        return;
    }
    auto frame = HfbFrameView(in_buf, 0);
    dset_id_t ds_id = frame.dataset_id;
    auto future_ds_state = std::async(&hfbTranspose::get_dataset_state, this, ds_id);

    if (!future_ds_state.get()) {
        FATAL_ERROR("Set to not use dataset_broker and couldn't find ancestor of dataset {}. "
                    "Make sure there is a stage upstream in the config, that adds the dataset "
                    "states.\nExiting...",
                    ds_id);
    }

    // Once the async get_dataset_state() is done, we have all the metadata to
    // create a file.

    found_flags = std::vector<bool>(write_t, false);

    // Create HDF5 file
    if (stack.size() > 0) {
        file = std::unique_ptr<hfbFileArchive>(
            new hfbFileArchive(filename, metadata, times, freqs, beams, sub_freqs, stack,
                               reverse_stack, chunk, kotekan::logLevel(_member_log_level)));
    } else {
        file = std::unique_ptr<hfbFileArchive>(
            new hfbFileArchive(filename, metadata, times, freqs, beams, sub_freqs, chunk,
                               kotekan::logLevel(_member_log_level)));
    }

    // TODO: it seems like this should be a Counter?
    auto& transposed_bytes_metric =
        Metrics::instance().add_gauge("kotekan_hfbtranspose_data_transposed_bytes", unique_name);

    while (!stop_thread) {
        // Wait for a full frame in the input buffer
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }
        auto frame = HfbFrameView(in_buf, frame_id);

        if (frame.dataset_id != ds_id) {
            FATAL_ERROR("Dataset ID of incoming frames changed from {} to {}. Changing  ID "
                        "not supported, exiting...",
                        ds_id, frame.dataset_id);
        }

        // Collect frames until a chunk is filled
        // Time-transpose as frames come in
        // Fastest varying is time (needs to be consistent with reader!)
        offset = fi * write_t;
        strided_copy(frame.hfb.data(), hfb.data(), offset * eff_data_dim + ti, write_t,
                     eff_data_dim);

        // Increment within read chunk
        ti = (ti + 1) % write_t;
        if (ti == 0)
            fi++;
        if (fi == write_f) {
            // chunk is complete
            write();
            // increment between chunks
            increment_chunk();
            fi = 0;
            ti = 0;

            // export prometheus metric
            if (frame_size == 0)
                //TODO: num_subfreq should be 128 when passed to calc_frame_size()
                frame_size = HfbFrameView::calculate_frame_size(num_beams, num_subfreq);
            transposed_bytes_metric.set(frame_size * frames_so_far);
        }

        frames_so_far++;
        // Exit when all frames have been written
        if (frames_so_far == num_time * num_freq)
            exit_kotekan(ReturnCode::CLEAN_EXIT);

        // move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}

void hfbTranspose::write() {
    DEBUG("Writing at freq {:d} and time {:d}", f_ind, t_ind);
    DEBUG("Writing block of {:d} freqs and {:d} times", write_f, write_t);

    file->write_block("hfb", f_ind, t_ind, write_f, write_t, hfb.data());
}

// increment between chunks
// cycle through all times before incrementing the frequency
// WARNING: This order must be consistent with how hfbRawReader
//      implements chunked reads. The mechanism for avoiding
//      overwriting flags also relies on this ordering.
void hfbTranspose::increment_chunk() {
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
