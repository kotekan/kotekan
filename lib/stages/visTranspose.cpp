#include "visTranspose.hpp"

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
#include "visBuffer.hpp"         // for VisFrameView
#include "visFileArchive.hpp"    // for visFileArchive

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

REGISTER_KOTEKAN_STAGE(visTranspose);

visTranspose::visTranspose(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Transpose(config, unique_name, buffer_container) {

    // get chunk dimensions for write from config file
    chunk = config.get<std::vector<int>>(unique_name, "chunk_size");
    if (chunk.size() != 3)
        throw std::invalid_argument("Chunk size needs exactly three elements "
                                    "(has "
                                    + std::to_string(chunk.size()) + ").");
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1)
        throw std::invalid_argument("visTranspose: Config: Chunk size needs "
                                    "to be equal to or greater than one.");
    chunk_t = chunk[2];
    chunk_f = chunk[0];

}

bool visTranspose::get_dataset_state(dset_id_t ds_id) {

    datasetManager& dm = datasetManager::instance();

    // Get the states synchronously.
    auto tstate_fut = std::async(&datasetManager::dataset_state<timeState>, &dm, ds_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, ds_id);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>, &dm, ds_id);
    auto evstate_fut = std::async(&datasetManager::dataset_state<eigenvalueState>, &dm, ds_id);
    auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>, &dm, ds_id);
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, ds_id);

    const stackState* sstate;
    const metadataState* mstate;
    const eigenvalueState* evstate;
    const timeState* tstate;
    const prodState* pstate;
    const freqState* fstate;
    const inputState* istate;
    bool timed_out = sstate_fut.wait_for(timeout) == std::future_status::timeout;
    if (!timed_out)
        sstate = sstate_fut.get();
    timed_out = timed_out || (mstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        mstate = mstate_fut.get();
    timed_out = timed_out || (evstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        evstate = evstate_fut.get();
    timed_out = timed_out || (tstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        tstate = tstate_fut.get();
    timed_out = timed_out || (pstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        pstate = pstate_fut.get();
    timed_out = timed_out || (fstate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        fstate = fstate_fut.get();
    timed_out = timed_out || (istate_fut.wait_for(timeout) == std::future_status::timeout);
    if (!timed_out)
        istate = istate_fut.get();
    if (timed_out) {
        ERROR("Communication with dataset broker timed out for datatset id {}.", ds_id);
        dm.stop();
        exit_kotekan(ReturnCode::DATASET_MANAGER_FAILURE);
        return false;
    }


    if (mstate == nullptr || tstate == nullptr || pstate == nullptr || fstate == nullptr
        || istate == nullptr || evstate == nullptr) {
        FATAL_ERROR("One of the dataset states is null.");
        return false;
    }

    // TODO split instrument_name up into the real instrument name,
    // registered by visAccumulate (?) and a data type, registered where
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
    inputs = istate->get_inputs();
    prods = pstate->get_prods();
    ev = evstate->get_ev();

    // unzip the vector of pairs in freqState
    auto freq_pairs = fstate->get_freqs();
    for (auto it = std::make_move_iterator(freq_pairs.begin()),
              end = std::make_move_iterator(freq_pairs.end());
         it != end; ++it) {
        freqs.push_back(std::move(it->second));
    }

    // Check if this is baseline-stacked data
    if (sstate) {
        stack = sstate->get_stack_map();
        // TODO: verify this is where it gets stored
        reverse_stack = sstate->get_rstack_map();
    }

    num_time = times.size();
    num_freq = freqs.size();
    num_input = inputs.size();
    num_prod = prods.size();
    num_ev = ev.size();

    // the dimension of the visibilities is different for stacked data
    eff_data_dim = (stack.size() > 0) ? stack.size() : num_prod;

    DEBUG("Dataset {} has {:d} times, {:d} frequencies, {:d} products", ds_id, num_time, num_freq,
          eff_data_dim);

    // Ensure chunk_size not too large
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

    // Allocate memory for collecting frames
    vis.resize(chunk_t * chunk_f * eff_data_dim, 0.);
    vis_weight.resize(chunk_t * chunk_f * eff_data_dim, 0.);
    eval.resize(chunk_t * chunk_f * num_ev, 0.);
    evec.resize(chunk_t * chunk_f * num_ev * num_input, 0.);
    erms.resize(chunk_t * chunk_f, 0.);
    gain.resize(chunk_t * chunk_f * num_input, 0.);
    // init frac_lost to 1.0 to match empty frames
    frac_lost.resize(chunk_t * chunk_f, 1.);
    frac_rfi.resize(chunk_t * chunk_f, 0.);
    input_flags.resize(chunk_t * num_input, 0.);
    dset_id.resize(chunk_t * chunk_f);

    // Initialise dataset ID array with null IDs
    std::string null_ds_id = fmt::format("{}", dset_id_t::null);
    for (auto& ds : dset_id) {
        std::copy(null_ds_id.c_str(), null_ds_id.c_str() + DSET_ID_LEN, ds.hash);
    }

    return true;
}

std::tuple<size_t, uint64_t, dset_id_t> visTranspose::get_frame_data() {

  auto frame = VisFrameView(in_buf, frame_id);
  return std::make_tuple(frame.calculate_frame_size(num_input, num_prod, num_ev),
                         frame.fpga_seq_total, frame.dataset_id);
}

void visTranspose::write_chunk() {
    DEBUG("Writing at freq {:d} and time {:d}", f_ind, t_ind);
    DEBUG("Writing block of {:d} freqs and {:d} times", write_f, write_t);

    file->write_block("vis", f_ind, t_ind, write_f, write_t, vis.data());

    file->write_block("vis_weight", f_ind, t_ind, write_f, write_t, vis_weight.data());

    if (num_ev > 0) {
        file->write_block("eval", f_ind, t_ind, write_f, write_t, eval.data());
        file->write_block("evec", f_ind, t_ind, write_f, write_t, evec.data());
        file->write_block("erms", f_ind, t_ind, write_f, write_t, erms.data());
    }

    file->write_block("gain", f_ind, t_ind, write_f, write_t, gain.data());

    file->write_block("flags/frac_lost", f_ind, t_ind, write_f, write_t, frac_lost.data());

    file->write_block("flags/frac_rfi", f_ind, t_ind, write_f, write_t, frac_rfi.data());

    file->write_block("flags/inputs", f_ind, t_ind, write_f, write_t, input_flags.data());

    file->write_block("flags/dataset_id", f_ind, t_ind, write_f, write_t, dset_id.data());
}

// increment between chunks
// chunks come in (time, freq) order
// WARNING: This order must be consistent with how VisRawReader
//      implements chunked reads. The mechanism for avoiding
//      overwriting flags also relies on this ordering.
void visTranspose::increment_chunk() {
    // Figure out where the next chunk starts
    f_ind = f_edge ? 0 : (f_ind + chunk_f) % num_freq;
    if (f_ind == 0) {
        // set incomplete chunk flag
        f_edge = (num_freq < chunk_f);
        t_ind += chunk_t;
        // clear flags buffer for next time chunk
        std::fill(input_flags.begin(), input_flags.end(), 0.);
        std::fill(found_flags.begin(), found_flags.end(), false);
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

void visTranspose::create_hdf5_file() {
  // Create HDF5 file
  if (stack.size() > 0) {
    file = std::unique_ptr<visFileArchive>(
        new visFileArchive(filename, metadata, times, freqs, inputs, prods, stack,
          reverse_stack, num_ev, chunk, kotekan::logLevel(_member_log_level)));
  } else {
    file = std::unique_ptr<visFileArchive>(
        new visFileArchive(filename, metadata, times, freqs, inputs, prods, num_ev, chunk,
          kotekan::logLevel(_member_log_level)));
  }
}

void visTranspose::copy_frame_data(uint32_t freq_index, uint32_t time_index) {

  auto frame = VisFrameView(in_buf, frame_id);

  // Collect frames until a chunk is filled
  // Time-transpose as frames come in
  // Fastest varying is time (needs to be consistent with reader!)
  uint32_t offset = freq_index * write_t;
  strided_copy(frame.vis.data(), vis.data(), offset * eff_data_dim + time_index, write_t,
               eff_data_dim);
  strided_copy(frame.weight.data(), vis_weight.data(), offset * eff_data_dim + time_index,
               write_t, eff_data_dim);

  strided_copy(frame.eval.data(), eval.data(), freq_index * num_ev * write_t + time_index, write_t,
               num_ev);

  strided_copy(frame.evec.data(), evec.data(), freq_index * num_ev * num_input * write_t + time_index,
               write_t, num_ev * num_input);

  erms[offset + time_index] = frame.erms;
  frac_lost[offset + time_index] = frame.fpga_seq_length == 0
                               ? 1.
                               : 1. - float(frame.fpga_seq_total) / frame.fpga_seq_length;
  frac_rfi[offset + time_index] =
      frame.fpga_seq_length == 0 ? 0. : float(frame.rfi_total) / frame.fpga_seq_length;
  strided_copy(frame.gain.data(), gain.data(), offset * num_input + time_index, write_t,
               num_input);
}

void visTranspose::copy_flags(uint32_t time_index) {

  auto frame = VisFrameView(in_buf, frame_id);

  // Only copy flags if we haven't already
  if (!found_flags[time_index]) {
    // Only update flags if they are non-zero
    bool nz_flags = false;
    for (uint i = 0; i < num_input; i++) {
      if (frame.flags[i] != 0.) {
        nz_flags = true;
        break;
      }
    }
    if (nz_flags) {
      // Copy flags into the buffer. These will not be overwritten until
      // the chunks increment in time
      strided_copy(frame.flags.data(), input_flags.data(), time_index, write_t, num_input);
      found_flags[time_index] = true;
    }
  }
}
