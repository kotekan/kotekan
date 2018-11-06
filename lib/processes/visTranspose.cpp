#include <algorithm>
#include <sys/stat.h>
#include <fstream>
#include <csignal>
#include <stdexcept>

#include "errors.h"
#include "visBuffer.hpp"
#include "fmt.hpp"
#include "visUtil.hpp"
#include "visTranspose.hpp"
#include "prometheusMetrics.hpp"
#include "datasetManager.hpp"
#include "version.h"

REGISTER_KOTEKAN_PROCESS(visTranspose);

visTranspose::visTranspose(Config &config, const string& unique_name,
        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
            std::bind(&visTranspose::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // get chunk dimensions for write from config file
    chunk = config.get<std::vector<int>>(unique_name, "chunk_size");
    if (chunk.size() != 3)
        throw std::invalid_argument("Chunk size needs exactly three elements " \
                "(has " + std::to_string(chunk.size()) + ").");
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1)
        throw std::invalid_argument("visTranspose: Config: Chunk size needs " \
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

    _use_dataset_manager = config.get_default<bool>(
                unique_name, "use_dataset_manager", false);
    if (!_use_dataset_manager) {
        // Read the metadata from file like in the old times
        // TODO: remove this option completely
        std::string md_filename = config.get<std::string>(unique_name, "infile")
                + ".meta";

        INFO("Reading metadata file: %s", md_filename.c_str());
        struct stat st;
        if (stat(md_filename.c_str(), &st) == -1)
            throw std::ios_base::failure("visTranspose: Error reading from " \
                                    "metadata file: " + md_filename);
        size_t filesize = st.st_size;
        std::vector<uint8_t> packed_json(filesize);
        std::string version;

        std::ifstream metadata_file(md_filename, std::ios::binary);
        if (metadata_file) // read only if no error
            metadata_file.read((char *)&packed_json[0], filesize);
        if (!metadata_file) // check if open and read successful
            throw std::ios_base::failure("visTranspose: Error reading from " \
                                    "metadata file: " + md_filename);
        json _t = json::from_msgpack(packed_json);
        metadata_file.close();

        // Extract the attributes and index maps from metadata

        // change archive version: remove "NT_" prefix (not transposed)
        metadata = _t["attributes"];
        metadata["archive_version"] = "3.1.0";

        times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
        freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
        inputs = _t["index_map"]["input"].get<std::vector<input_ctype>>();
        prods = _t["index_map"]["prod"].get<std::vector<prod_ctype>>();
        ev = _t["index_map"]["ev"].get<std::vector<uint32_t>>();

        // Check if this is baseline-stacked data
        if (_t["index_map"].find("stack") != _t["index_map"].end()) {
            stack = _t["index_map"]["stack"].get<std::vector<stack_ctype>>();
            // TODO: verify this is where it gets stored
            reverse_stack =
                    _t["reverse_map"]["stack"].get<std::vector<rstack_ctype>>();
        }
    }
}

void visTranspose::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void visTranspose::gather_metadata() {
    dset_id_t ds_id = 0;

    if (_use_dataset_manager) {
        // Wait for the first frame in the input buffer
        if((wait_for_full_frame(in_buf, unique_name.c_str(), 0)) == nullptr) {
            return;
        }
        auto frame = visFrameView(in_buf, 0);
        ds_id = frame.dataset_id;

        // Get metadata from dataset states
        datasetManager& dm = datasetManager::instance();
        const metadataState* mstate = nullptr;
        const timeState* tstate = nullptr;
        const prodState* pstate = nullptr;
        const freqState* fstate = nullptr;
        const inputState* istate = nullptr;
        const eigenvalueState* evstate = nullptr;
        const stackState* sstate = nullptr;
        try {
            // TODO: get the states synchronously (?)
            mstate = dm.dataset_state<metadataState>(ds_id);
            tstate = dm.dataset_state<timeState>(ds_id);
            pstate = dm.dataset_state<prodState>(ds_id);
            fstate = dm.dataset_state<freqState>(ds_id);
            istate = dm.dataset_state<inputState>(ds_id);
            evstate = dm.dataset_state<eigenvalueState>(ds_id);
            sstate = dm.dataset_state<stackState>(ds_id);
        } catch (std::runtime_error& e) {
            // Crash if anything goes wrong. This process is processing data
            // from a file, so should be restarted after fixing the problem.
            ERROR("Failure in datasetManager: %s.");
            ERROR("Exiting...");
            raise(SIGINT);
        }

        if (mstate == nullptr || tstate == nullptr || pstate == nullptr ||
            fstate == nullptr || istate == nullptr || evstate == nullptr ||
            sstate == nullptr) {

            // Also crash here, if a state is missing, there is something wrong.
            // Problem should be fixed and restarted.
            ERROR("Could not find all dataset states.");
            ERROR("One of those is a nullptr: stack %d, meta %d, time %d, " \
                  "prod %d, freq %d, input %d, ev %d",
                  sstate, mstate, tstate, pstate, fstate, istate, evstate);
            ERROR("Exiting...");
            raise(SIGINT);
        }

        // TODO split instrument_name up into the real instrument name,
        // registered by visAccumulate (?) and a data type, registered where
        // data is written to file the first time
        metadata["instrument_name"] = mstate->get_instrument_name();
        metadata["weight_type"] = mstate->get_weight_type();
        std::string git_commit_hash_dataset = mstate->get_git_version_tag();

        //TODO: enforce this if build type == release?
        if (git_commit_hash_dataset
                    != metadata["git_version_tag"].get<std::string>())
            INFO("Git version tags don't match: dataset %zu has tag %s, while "\
                 "the local git version tag is %s", ds_id,
                 git_commit_hash_dataset.c_str(),
                 metadata["git_version_tag"].get<std::string>().c_str());

        times = tstate->get_times();
        inputs = istate->get_inputs();
        prods = pstate->get_prods();
        ev = evstate->get_ev();

        // unzip the vector of pairs in freqState
        auto freq_pairs = fstate->get_freqs();
        for (auto it = std::make_move_iterator(freq_pairs.begin()),
                 end = std::make_move_iterator(freq_pairs.end());
             it != end; ++it)
        {
            freqs.push_back(std::move(it->second));
        }

        // Check if this is baseline-stacked data
        if (sstate->is_stacked()) {
            stack = sstate->get_stack_map();
            // TODO: verify this is where it gets stored
            reverse_stack = sstate->get_rstack_map();
        }
    }

    num_time = times.size();
    num_freq = freqs.size();
    num_input = inputs.size();
    num_prod = prods.size();
    num_ev = ev.size();

    // the dimension of the visibilities is different for stacked data
    eff_prod_dim = (stack.size() > 0) ? stack.size() : num_prod;

    if (_use_dataset_manager) {
        DEBUG("Dataset %zu has %d times, %d frequencies, %d products",
              ds_id, num_time, num_freq, eff_prod_dim);
    } else
        DEBUG("File has %d times, %d frequencies, %d products",
              num_time, num_freq, eff_prod_dim);

    // Ensure chunk_size not too large
    chunk_t = std::min(chunk_t, num_time);
    write_t = chunk_t;
    chunk_f = std::min(chunk_f, num_freq);
    write_f = chunk_f;

    // Allocate memory for collecting frames
    vis.resize(chunk_t*chunk_f*eff_prod_dim);
    vis_weight.resize(chunk_t*chunk_f*eff_prod_dim);
    eval.resize(chunk_t*chunk_f*num_ev);
    evec.resize(chunk_t*chunk_f*num_ev*num_input);
    erms.resize(chunk_t*chunk_f);
    gain.resize(chunk_t*chunk_f*num_input);
    frac_lost.resize(chunk_t*chunk_f);
    input_flags.resize(chunk_t*num_input);
    std::fill(input_flags.begin(), input_flags.end(), 0.);
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

    // Get the dataset states and prepare all metadata
    gather_metadata();

    found_flags = vector<bool>(write_t, false);

    // Create HDF5 file
    if (stack.size() > 0) {
        file = std::unique_ptr<visFileArchive>(new visFileArchive(filename,
                    metadata, times, freqs, inputs, prods,
                    stack, reverse_stack, num_ev, chunk)
        );
    } else {
        file = std::unique_ptr<visFileArchive>(new visFileArchive(filename,
                    metadata, times, freqs, inputs, prods, num_ev, chunk)
        );
    }

    while (!stop_thread) {
        // Wait for a full frame in the input buffer
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                        frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);

        // Collect frames until a chunk is filled
        // Time-transpose as frames come in
        // Fastest varying is time (needs to be consistent with reader!)
        offset = fi * write_t;
        strided_copy(frame.vis.data(), vis.data(), offset*eff_prod_dim + ti,
                write_t, eff_prod_dim);
        strided_copy(frame.weight.data(), vis_weight.data(),
                offset*eff_prod_dim + ti, write_t, eff_prod_dim);
        strided_copy(frame.eval.data(), eval.data(), fi*num_ev*write_t + ti,
                write_t, num_ev);
        strided_copy(frame.evec.data(), evec.data(),
                fi*num_ev*num_input*write_t + ti, write_t, num_ev*num_input);
        erms[offset + ti] = frame.erms;
        frac_lost[offset + ti] = frame.fpga_seq_length == 0 ?
                1. : 1. - float(frame.fpga_seq_total) / frame.fpga_seq_length;
        strided_copy(frame.gain.data(), gain.data(), offset*num_input + ti,
                write_t, num_input);

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
                strided_copy(frame.flags.data(), input_flags.data(), ti,
                        write_t, num_input);
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
                frame_size = frame.calculate_buffer_layout(num_input, num_prod,
                        num_ev)["_struct"].second;
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

    file->write_block("vis_weight", f_ind, t_ind, write_f, write_t,
            vis_weight.data());

    if (num_ev > 0) {
        file->write_block("eval", f_ind, t_ind, write_f, write_t, eval.data());
        file->write_block("evec", f_ind, t_ind, write_f, write_t, evec.data());
        file->write_block("erms", f_ind, t_ind, write_f, write_t, erms.data());
    }

    file->write_block("gain", f_ind, t_ind, write_f, write_t,
            gain.data());

    file->write_block("flags/frac_lost", f_ind, t_ind, write_f, write_t,
            frac_lost.data());

    file->write_block("flags/inputs", f_ind, t_ind, write_f, write_t,
            input_flags.data());
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
