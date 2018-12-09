
#include "visFileRaw.hpp"
#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "visCompression.hpp"

#include "json.hpp"

#include <cxxabi.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <exception>
#include <fmt.hpp>
#include <fstream>
#include <future>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <inttypes.h>


// Register the raw file writer
REGISTER_VIS_FILE("raw", visFileRaw);

//
// Implementation of raw visibility data file
//
void visFileRaw::create_file(
    const std::string& name,
    const std::map<std::string, std::string>& metadata,
    dset_id_t dataset, size_t max_time)
{
    INFO("Creating new output file %s", name.c_str());

    // Get properties of stream from datasetManager
    auto& dm = datasetManager::instance();
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>,
                                 &dm, dataset);
    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>,
                                 &dm, dataset);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>,
                                 &dm, dataset);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>,
                                 &dm, dataset);
    auto evstate_fut = std::async(
        &datasetManager::dataset_state<eigenvalueState>, &dm, dataset
    );

    const stackState* sstate = sstate_fut.get();
    const inputState* istate = istate_fut.get();
    const prodState* pstate = pstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const eigenvalueState* evstate = evstate_fut.get();

    if (!istate || !pstate || !fstate) {
        ERROR("Required datasetState not found for dataset ID " \
              "0x%" PRIx64 "\nThe following required states were found:\n" \
              "inputState - %d\nprodState - %d\nfreqState - %d\n",
              dataset, istate, pstate, fstate);
        throw std::runtime_error("Could not create file.");
    }

    // Set the axis metadata
    file_metadata["attributes"] = metadata;
    file_metadata["index_map"]["freq"] = unzip(fstate->get_freqs()).second;
    file_metadata["index_map"]["input"] = istate->get_inputs();
    file_metadata["index_map"]["prod"] = pstate->get_prods();

    // Create and add eigenvalue index
    if (evstate) {
        file_metadata["index_map"]["ev"] = evstate->get_ev();
        num_ev = evstate->get_num_ev();
    }
    else {
        num_ev = 0;
    }

    if (sstate) {
        file_metadata["index_map"]["stack"] = sstate->get_stack_map();
        file_metadata["reverse_map"]["stack"] = sstate->get_rstack_map();
        file_metadata["structure"]["num_stack"] = sstate->get_num_stack();
    }


    // Calculate the file structure
    nfreq = fstate->get_freqs().size();
    size_t ninput = istate->get_inputs().size();
    size_t nvis = sstate ?
                sstate->get_num_stack() : pstate->get_prods().size();

    // Set the alignment (in kB)
    // TODO: find some way of getting this from config
    alignment = 4;  // Align on page boundaries

    // Calculate the file structure
    auto layout = visFrameView::calculate_buffer_layout(
        ninput, nvis, num_ev
    );
    data_size = layout.first;
    metadata_size = sizeof(visMetadata);
    frame_size = _member_alignment(data_size + metadata_size + 1,
                                   alignment * 1024);

    // Write the structure into the file for decoding
    file_metadata["structure"]["metadata_size"] = metadata_size;
    file_metadata["structure"]["data_size"] = data_size;
    file_metadata["structure"]["frame_size"] = frame_size;
    file_metadata["structure"]["nfreq"] = nfreq;


    // Create lock file and then open the other files
    _name = name;
    lock_filename = create_lockfile(name);
    metadata_file = std::ofstream(name + ".meta", std::ios::binary);
    if((fd = open((name + ".data").c_str(), oflags,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) {
        std::runtime_error(fmt::format("Failed to open file {}: {}.",
                                       name + ".data", strerror(errno)));
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
    std::vector<uint8_t> t = json::to_msgpack(file_metadata);
    metadata_file.write((const char *)&t[0], t.size());
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
    (void)ind; // Suppress warning
#endif
}

void visFileRaw::flush_raw_sync(int ind) {
#ifdef __linux__
    size_t n = nfreq * frame_size;
    sync_file_range(fd, ind * n, n,
                    SYNC_FILE_RANGE_WAIT_BEFORE |
                    SYNC_FILE_RANGE_WRITE |
                    SYNC_FILE_RANGE_WAIT_AFTER);
    posix_fadvise(fd, ind * n, n, POSIX_FADV_DONTNEED);
#else
    (void)ind; // Suppress warning
#endif
}

uint32_t visFileRaw::extend_time(time_ctype new_time) {

    size_t ntime = num_time();

    // Start to flush out older dataset regions
    uint delta_async = 2;
    if(ntime > delta_async) {
        flush_raw_async(ntime - delta_async);
    }

    // Flush and clear out any really old parts of the datasets
    uint delta_sync = 4;
    if(ntime > delta_sync) {
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
    int nbytes = TEMP_FAILURE_RETRY(
        pwrite(fd, data, nb, offset)
    );

    if(nbytes < 0) {
        ERROR("Write error attempting to write %i bytes at offset %llu into file %s: %s",
              nb, offset, _name.c_str(), strerror(errno));
        return false;
    }

    return true;
}

void visFileRaw::write_sample(
    uint32_t time_ind, uint32_t freq_ind, const visFrameView& frame
)
{
    // TODO: consider adding checks for all dims
    if (frame.num_ev != num_ev) {
        std::string msg = fmt::format(
            "Number of eigenvalues don't match for write (got {}, expected {})",
            frame.num_ev, num_ev
        );
        throw std::runtime_error(msg);
    }

    const uint8_t ONE = 1;

    // Write out data to the right place
    off_t offset = (time_ind * nfreq + freq_ind) * frame_size;

    write_raw(offset, 1, &ONE);
    write_raw(offset + 1, metadata_size, frame.metadata());
    write_raw(offset + 1 + metadata_size, data_size, frame.data());
}
