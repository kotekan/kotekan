#include "visTranspose.hpp"

#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "processFactory.hpp"
#include "prometheusMetrics.hpp"
#include "version.h"
#include "visBuffer.hpp"
#include "visUtil.hpp"

#include "gsl-lite.hpp"

#include <algorithm>
#include <atomic>
#include <complex>
#include <csignal>
#include <cstdint>
#include <cxxabi.h>
#include <exception>
#include <functional>
#include <future>
#include <inttypes.h>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

REGISTER_KOTEKAN_PROCESS(visTranspose);

visTranspose::visTranspose(Config& config, const string& unique_name,
                           bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTranspose::main_thread, this)) {

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
        throw std::invalid_argument("visTranspose: Config: Chunk size needs "
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

    const stackState* sstate = sstate_fut.get();
    const metadataState* mstate = mstate_fut.get();
    const eigenvalueState* evstate = evstate_fut.get();
    const timeState* tstate = tstate_fut.get();
    const prodState* pstate = pstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const inputState* istate = istate_fut.get();


    if (mstate == nullptr || tstate == nullptr || pstate == nullptr || fstate == nullptr
        || istate == nullptr || evstate == nullptr)
        return false;

    // TODO split instrument_name up into the real instrument name,
    // registered by visAccumulate (?) and a data type, registered where
    // data is written to file the first time
    metadata["instrument_name"] = mstate->get_instrument_name();
    metadata["weight_type"] = mstate->get_weight_type();

    std::string git_commit_hash_dataset = mstate->get_git_version_tag();

    // TODO: enforce this if build type == release?
    if (git_commit_hash_dataset != metadata["git_version_tag"].get<std::string>())
        INFO("Git version tags don't match: dataset 0x%" PRIx64 " has tag %s,"
             "while the local git version tag is %s",
             ds_id, git_commit_hash_dataset.c_str(),
             metadata["git_version_tag"].get<std::string>().c_str());

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
    eff_prod_dim = (stack.size() > 0) ? stack.size() : num_prod;

    DEBUG("Dataset 0x%" PRIx64 " has %d times, %d frequencies, %d products", ds_id, num_time,
          num_freq, eff_prod_dim);

    // Ensure chunk_size not too large
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

    // Allocate memory for collecting frames
    vis.resize(chunk_t * chunk_f * eff_prod_dim);
    vis_weight.resize(chunk_t * chunk_f * eff_prod_dim);
    eval.resize(chunk_t * chunk_f * num_ev);
    evec.resize(chunk_t * chunk_f * num_ev * num_input);
    erms.resize(chunk_t * chunk_f);
    gain.resize(chunk_t * chunk_f * num_input);
    frac_lost.resize(chunk_t * chunk_f);
    input_flags.resize(chunk_t * num_input);
    std::fill(input_flags.begin(), input_flags.end(), 0.);

    return true;
}

void visTranspose::main_thread() {

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
    auto frame = visFrameView(in_buf, 0);
    dset_id_t ds_id = frame.dataset_id;
    auto future_ds_state = std::async(&visTranspose::get_dataset_state, this, ds_id);

    if (!future_ds_state.get()) {
        ERROR("Set to not use dataset_broker and couldn't find "
              "ancestor of dataset 0x%" PRIx64 ". Make sure there is a process"
              " upstream in the config, that adds the dataset states."
              "\nExiting...",
              ds_id);
        raise(SIGINT);
    }

    // Once the async get_dataset_state() is done, we have all the metadata to
    // create a file.

    found_flags = vector<bool>(write_t, false);

    // Create HDF5 file
    if (stack.size() > 0) {
        file = std::unique_ptr<visFileArchive>(new visFileArchive(
            filename, metadata, times, freqs, inputs, prods, stack, reverse_stack, num_ev, chunk));
    } else {
        file = std::unique_ptr<visFileArchive>(
            new visFileArchive(filename, metadata, times, freqs, inputs, prods, num_ev, chunk));
    }

    while (!stop_thread) {
        // Wait for a full frame in the input buffer
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);

        if (frame.dataset_id != ds_id) {
            ERROR("Dataset ID of incoming frames changed from 0x%" PRIx64 " to"
                  "0x%" PRIx64 ". "
                  "Not supported, exiting...",
                  ds_id, frame.dataset_id);
            raise(SIGINT);
        }

        // Collect frames until a chunk is filled
        // Time-transpose as frames come in
        // Fastest varying is time (needs to be consistent with reader!)
        offset = fi * write_t;
        strided_copy(frame.vis.data(), vis.data(), offset * eff_prod_dim + ti, write_t,
                     eff_prod_dim);
        strided_copy(frame.weight.data(), vis_weight.data(), offset * eff_prod_dim + ti, write_t,
                     eff_prod_dim);
        strided_copy(frame.eval.data(), eval.data(), fi * num_ev * write_t + ti, write_t, num_ev);
        strided_copy(frame.evec.data(), evec.data(), fi * num_ev * num_input * write_t + ti,
                     write_t, num_ev * num_input);
        erms[offset + ti] = frame.erms;
        frac_lost[offset + ti] = frame.fpga_seq_length == 0
                                     ? 1.
                                     : 1. - float(frame.fpga_seq_total) / frame.fpga_seq_length;
        strided_copy(frame.gain.data(), gain.data(), offset * num_input + ti, write_t, num_input);

        // Only copy flags if we haven't already
        if (!found_flags[ti]) {
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
                strided_copy(frame.flags.data(), input_flags.data(), ti, write_t, num_input);
                found_flags[ti] = true;
            }
        }

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
                frame_size = frame.calculate_buffer_layout(num_input, num_prod, num_ev).first;
            prometheusMetrics::instance().add_process_metric(
                "kotekan_vistranspose_data_transposed_bytes", unique_name,
                frame_size * frames_so_far);
        }

        frames_so_far++;
        // Exit when all frames have been written
        if (frames_so_far == num_time * num_freq)
            std::raise(SIGINT);

        // move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}

void visTranspose::write() {
    DEBUG("Writing at freq %d and time %d", f_ind, t_ind);
    DEBUG("Writing block of %d freqs and %d times", write_f, write_t);

    file->write_block("vis", f_ind, t_ind, write_f, write_t, vis.data());

    file->write_block("vis_weight", f_ind, t_ind, write_f, write_t, vis_weight.data());

    if (num_ev > 0) {
        file->write_block("eval", f_ind, t_ind, write_f, write_t, eval.data());
        file->write_block("evec", f_ind, t_ind, write_f, write_t, evec.data());
        file->write_block("erms", f_ind, t_ind, write_f, write_t, erms.data());
    }

    file->write_block("gain", f_ind, t_ind, write_f, write_t, gain.data());

    file->write_block("flags/frac_lost", f_ind, t_ind, write_f, write_t, frac_lost.data());

    file->write_block("flags/inputs", f_ind, t_ind, write_f, write_t, input_flags.data());
}

// increment between chunks
// cycle through all times before incrementing the frequency
// WARNING: This order must be consistent with how visRawReader
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
