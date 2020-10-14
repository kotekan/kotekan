
#include "visFileRaw.hpp"

#include "Hash.hpp"           // for Hash
#include "datasetManager.hpp" // for datasetManager, dset_id_t
#include "datasetState.hpp"   // for stackState, eigenvalueState, freqState, gatingState, input...
#include "visBuffer.hpp"      // for VisFrameView, VisMetadata

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value_type, json

#include <cstdio>       // for remove
#include <cxxabi.h>     // for __forced_unwind
#include <errno.h>      // for errno
#include <exception>    // for exception
#include <fcntl.h>      // for fallocate, sync_file_range, open, posix_fadvise, FALLOC_FL...
#include <fstream>      // for ofstream, basic_ostream::write, ios
#include <future>       // for async, future
#include <stdexcept>    // for out_of_range, runtime_error
#include <string.h>     // for strerror
#include <sys/stat.h>   // for S_IRGRP, S_IROTH, S_IRUSR, S_IWGRP, S_IWUSR
#include <system_error> // for system_error
#include <unistd.h>     // for close, pwrite, TEMP_FAILURE_RETRY
#include <utility>      // for pair


// Register the raw file writer
REGISTER_VIS_FILE("raw", visFileRaw);

//
// Implementation of raw visibility data file
//
visFileRaw::visFileRaw(const std::string& name, const kotekan::logLevel log_level,
                       const std::map<std::string, std::string>& metadata, dset_id_t dataset,
                       size_t max_time, int oflags) :
    _name(name) {
    set_log_level(log_level);

    INFO("Creating new output file {:s}", name);

    // Get properties of stream from datasetManager
    auto& dm = datasetManager::instance();
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, dataset);
    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>, &dm, dataset);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, dataset);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, dataset);
    auto evstate_fut = std::async(&datasetManager::dataset_state<eigenvalueState>, &dm, dataset);
    auto gstate_fut = std::async(&datasetManager::dataset_state<gatingState>, &dm, dataset);

    const inputState* istate = istate_fut.get();
    const prodState* pstate = pstate_fut.get();
    const freqState* fstate = fstate_fut.get();

    if (!istate || !pstate || !fstate) {
        ERROR("Required datasetState not found for dataset ID {}\nThe following required states "
              "were found:\ninputState - {:p}\nprodState - {:p}\nfreqState - {:p}\n",
              dataset, (void*)istate, (void*)pstate, (void*)fstate);
        throw std::runtime_error("Could not create file.");
    }

    // Set the axis metadata
    file_metadata["attributes"] = metadata;
    file_metadata["index_map"]["freq"] = unzip(fstate->get_freqs()).second;
    file_metadata["index_map"]["input"] = istate->get_inputs();
    file_metadata["index_map"]["prod"] = pstate->get_prods();

    // Create and add eigenvalue index
    const eigenvalueState* evstate = evstate_fut.get();
    if (evstate) {
        file_metadata["index_map"]["ev"] = evstate->get_ev();
        num_ev = evstate->get_num_ev();
    } else {
        num_ev = 0;
    }

    const stackState* sstate = sstate_fut.get();
    if (sstate) {
        file_metadata["index_map"]["stack"] = sstate->get_stack_map();
        file_metadata["reverse_map"]["stack"] = sstate->get_rstack_map();
        file_metadata["structure"]["num_stack"] = sstate->get_num_stack();
    }

    const gatingState* gstate = gstate_fut.get();
    if (gstate) {
        file_metadata["gating_type"] = gstate->gating_type;
        file_metadata["gating_data"] = gstate->gating_data;
    }

    // Calculate the file structure
    nfreq = fstate->get_freqs().size();
    size_t ninput = istate->get_inputs().size();
    size_t nvis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();

    // Set the alignment (in kB)
    // TODO: find some way of getting this from config
    alignment = 4; // Align on page boundaries

    // Calculate the file structure
    data_size = VisFrameView::calculate_frame_size(ninput, nvis, num_ev);

    metadata_size = sizeof(VisMetadata);
    frame_size = _member_alignment(data_size + metadata_size + 1, alignment * 1024);

    // Write the structure into the file for decoding
    file_metadata["structure"]["metadata_size"] = metadata_size;
    file_metadata["structure"]["data_size"] = data_size;
    file_metadata["structure"]["frame_size"] = frame_size;
    file_metadata["structure"]["nfreq"] = nfreq;


    // Create lock file and then open the other files
    lock_filename = create_lockfile(_name);
    metadata_file = std::ofstream(_name + ".meta", std::ios::binary);
    if ((fd = open((_name + ".data").c_str(), oflags,
                   S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH))
        == -1) {
        throw std::runtime_error(
            fmt::format(fmt("Failed to open file {:s}.data: {:s}."), _name, strerror(errno)));
    }

    // Preallocate data file (without increasing the length)
#ifdef __linux__
    // Note not all versions of linux support this feature, and they don't
    // include the macro FALLOC_FL_KEEP_SIZE in that case
#ifdef FALLOC_FL_KEEP_SIZE
    fallocate(fd, FALLOC_FL_KEEP_SIZE, 0, frame_size * nfreq * max_time);
#else
    (void)max_time; // Suppress warning
    WARN("fallocate not supported on this system!");
#endif
#else
    (void)max_time; // Suppress warning
#endif
}

visFileRaw::~visFileRaw() {

    // Finalize the metadata file
    file_metadata["structure"]["ntime"] = num_time();
    file_metadata["index_map"]["time"] = times;
    std::vector<uint8_t> t = nlohmann::json::to_msgpack(file_metadata);
    metadata_file.write((const char*)&t[0], t.size());
    metadata_file.close();

    // TODO: final sync of data file.
    close(fd);

    std::remove(lock_filename.c_str());
}

size_t visFileRaw::num_time() {
    return times.size();
}

void visFileRaw::flush_raw_async(int ind) {
#ifdef __linux__
    size_t n = nfreq * frame_size;
    sync_file_range(fd, ind * n, n, SYNC_FILE_RANGE_WRITE);
#else
    (void)ind;      // Suppress warning
#endif
}

void visFileRaw::flush_raw_sync(int ind) {
#ifdef __linux__
    size_t n = nfreq * frame_size;
    sync_file_range(fd, ind * n, n,
                    SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE
                        | SYNC_FILE_RANGE_WAIT_AFTER);
    posix_fadvise(fd, ind * n, n, POSIX_FADV_DONTNEED);
#else
    (void)ind;      // Suppress warning
#endif
}

uint32_t visFileRaw::extend_time(time_ctype new_time) {

    size_t ntime = num_time();

    // Start to flush out older dataset regions
    uint delta_async = 2;
    if (ntime > delta_async) {
        flush_raw_async(ntime - delta_async);
    }

    // Flush and clear out any really old parts of the datasets
    uint delta_sync = 4;
    if (ntime > delta_sync) {
        flush_raw_sync(ntime - delta_sync);
    }

    times.push_back(new_time);

    // Extend the file length for the new time
#ifdef __linux__
    fallocate(fd, 0, 0, frame_size * nfreq * num_time());
#else
    ftruncate(fd, frame_size * nfreq * num_time());
#endif

    return num_time() - 1;
}

void visFileRaw::deactivate_time(uint32_t time_ind) {
    flush_raw_sync(time_ind);
}


bool visFileRaw::write_raw(off_t offset, size_t nb, const void* data) {

    // Write in a retry macro loop incase the write was interrupted by a signal
    int nbytes = TEMP_FAILURE_RETRY(pwrite(fd, data, nb, offset));

    if (nbytes < 0) {
        ERROR("Write error attempting to write {:d} bytes at offset {:d} into file {:s}: {:s}", nb,
              offset, _name, strerror(errno));
        return false;
    }

    return true;
}

void visFileRaw::write_sample(uint32_t time_ind, uint32_t freq_ind, const FrameView& frame_view) {

    const VisFrameView& frame = static_cast<const VisFrameView&>(frame_view);

    // TODO: consider adding checks for all dims
    if (frame.num_ev != num_ev) {
        throw std::runtime_error(fmt::format(fmt("Number of eigenvalues don't match for write (got "
                                                 "{:d}, expected {:d})"),
                                             frame.num_ev, num_ev));
    }

    const uint8_t ONE = 1;

    // Write out data to the right place
    off_t offset = (time_ind * nfreq + freq_ind) * frame_size;

    write_raw(offset, 1, &ONE);
    write_raw(offset + 1, metadata_size, frame.metadata());
    write_raw(offset + 1 + metadata_size, data_size, frame.data());
}
