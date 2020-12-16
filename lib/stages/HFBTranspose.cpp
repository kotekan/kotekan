#include "HFBTranspose.hpp"

#include "Config.hpp"            // for Config
#include "HFBFileArchive.hpp"    // for HFBFileArchive
#include "HFBFrameView.hpp"      // for HFBFrameView
#include "Hash.hpp"              // for Hash, operator!=
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, datasetManager
#include "datasetState.hpp"      // for metadataState, stackState, acqDatasetIdState, eigenvalu...
#include "dset_id.hpp"           // for dset_id_str, DSET_ID_LEN
#include "errors.h"              // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"    // for DEBUG, FATAL_ERROR, logLevel, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge

#include "fmt.hpp"      // for format
#include "gsl-lite.hpp" // for span
#include "json.hpp"     // for basic_json<>::object_t, json, basic_json,

#include <algorithm>    // for max, fill, min
#include <cstdint>      // for uint32_t
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <future>       // for async, future
#include <iterator>     // for make_move_iterator, move_iterator, operator!=
#include <stdexcept>    // for out_of_range, invalid_argument
#include <stdint.h>     // for uint32_t, uint64_t
#include <system_error> // for system_error
#include <utility>      // for move, pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(HFBTranspose);

HFBTranspose::HFBTranspose(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Transpose(config, unique_name, buffer_container) {

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
    bool timed_out = mstate_fut.wait_for(timeout) == std::future_status::timeout;
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

    // Ensure chunk_size not too large
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

    DEBUG("Dataset {} has {:d} times, {:d} frequencies, {:d} beams and {:d} sub-frequencies.\n "
          "Data dimension: {:d} bytes, time chunk: {:d}, freq chunk: {:d}",
          ds_id, num_time, num_freq, num_beams, num_subfreq, eff_data_dim, chunk_t, chunk_f);

    // Allocate memory for collecting frames
    hfb.resize(chunk_t * chunk_f * eff_data_dim, 0.);
    hfb_weight.resize(chunk_t * chunk_f * eff_data_dim, 0.);
    // init frac_lost to 1.0 to match empty frames
    // frac_lost.resize(chunk_t * chunk_f, 1.);
    // frac_rfi.resize(chunk_t * chunk_f, 0.);
    dset_id.resize(chunk_t * chunk_f);

    // Initialise dataset ID array with null IDs
    std::string null_ds_id = fmt::format("{}", dset_id_t::null);
    for (auto& ds : dset_id) {
        std::copy(null_ds_id.c_str(), null_ds_id.c_str() + DSET_ID_LEN, ds.hash);
    }

    return true;
}

std::tuple<size_t, uint64_t, dset_id_t> HFBTranspose::get_frame_data() {

    auto frame = HFBFrameView(in_buf, frame_id);
    return std::make_tuple(frame.calculate_frame_size(num_beams, num_subfreq), frame.fpga_seq_total,
                           frame.dataset_id);
}

void HFBTranspose::write_chunk() {
    DEBUG("Writing at freq {:d} and time {:d}", f_ind, t_ind);
    DEBUG("Writing block of {:d} freqs and {:d} times. data: {}...{}...{}", write_f, write_t,
          hfb[0], hfb[write_t], hfb[write_t * 2]);

    file->write_block("hfb", f_ind, t_ind, write_f, write_t, hfb.data());
    file->write_block("hfb_weight", f_ind, t_ind, write_f, write_t, hfb_weight.data());
    file->write_block("flags/dataset_id", f_ind, t_ind, write_f, write_t, dset_id.data());
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

void HFBTranspose::create_hdf5_file() {
    // Create HDF5 file
    file = std::unique_ptr<HFBFileArchive>(
        new HFBFileArchive(filename, metadata, times, freqs, beams, sub_freqs, chunk,
                           kotekan::logLevel(_member_log_level)));
}

void HFBTranspose::copy_frame_data(uint32_t freq_index, uint32_t time_index) {

    auto frame = HFBFrameView(in_buf, frame_id);

    // Collect frames until a chunk is filled
    // Time-transpose as frames come in
    // Fastest varying is time (needs to be consistent with reader!)
    uint32_t offset = freq_index * write_t;
    strided_copy(frame.hfb.data(), hfb.data(), offset * eff_data_dim + time_index, write_t,
                 eff_data_dim);
    strided_copy(frame.weight.data(), hfb_weight.data(), offset * eff_data_dim + time_index,
                 write_t, eff_data_dim);
    // frac_lost[offset + time_index] = frame.fpga_seq_length == 0
    //                             ? 1.
    //                             : 1. - float(frame.fpga_seq_total) / frame.fpga_seq_length;
}

void HFBTranspose::copy_flags(uint32_t time_index) {
    (void)time_index;
}
