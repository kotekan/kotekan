
#include "hfbFileRaw.hpp"

#include "Hash.hpp"           // for Hash
#include "HfbFrameView.hpp"   // for HfbFrameView, hfbMetadata
#include "datasetManager.hpp" // for datasetManager, dset_id_t
#include "datasetState.hpp"   // for stackState, eigenvalueState, freqState, gatingState, input...

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value_type, json

#include <algorithm>    // for max
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
REGISTER_VIS_FILE("hfbraw", hfbFileRaw);

//
// Implementation of raw visibility data file
//
hfbFileRaw::hfbFileRaw(const std::string& name, const kotekan::logLevel log_level,
                       const std::map<std::string, std::string>& metadata, dset_id_t dataset,
                       size_t max_time, int oflags) :
    _name(name) {
    set_log_level(log_level);
    (void)dataset;

    INFO("Creating new output file {:s}", name);

    // Set the axis metadata
    file_metadata["attributes"] = metadata;

    // Calculate the file structure
    nfreq = 1024;
    num_beams = std::stoi(metadata.at("num_beams"));
    num_subfreq = std::stoi(metadata.at("num_sub_freqs"));

    // Set the alignment (in kB)
    // TODO: find some way of getting this from config
    alignment = 4; // Align on page boundaries

    // Calculate the file structure
    data_size = HfbFrameView::calculate_frame_size(num_beams, num_subfreq);

    metadata_size = sizeof(hfbMetadata);
    frame_size = _member_alignment(data_size + metadata_size + 1, alignment * 1024);

    // Write the structure into the file for decoding
    file_metadata["structure"]["metadata_size"] = metadata_size;
    file_metadata["structure"]["data_size"] = data_size;
    file_metadata["structure"]["frame_size"] = frame_size;
    file_metadata["structure"]["nfreq"] = nfreq;
    file_metadata["structure"]["num_beams"] = num_beams;
    file_metadata["structure"]["num_subfreq"] = num_subfreq;

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

hfbFileRaw::~hfbFileRaw() {

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

size_t hfbFileRaw::num_time() {
    return times.size();
}

void hfbFileRaw::flush_raw_async(int ind) {
#ifdef __linux__
    size_t n = nfreq * frame_size;
    sync_file_range(fd, ind * n, n, SYNC_FILE_RANGE_WRITE);
#else
    (void)ind;      // Suppress warning
#endif
}

void hfbFileRaw::flush_raw_sync(int ind) {
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

uint32_t hfbFileRaw::extend_time(time_ctype new_time) {

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

void hfbFileRaw::deactivate_time(uint32_t time_ind) {
    flush_raw_sync(time_ind);
}


bool hfbFileRaw::write_raw(off_t offset, size_t nb, const void* data) {

    // Write in a retry macro loop incase the write was interrupted by a signal
    int nbytes = TEMP_FAILURE_RETRY(pwrite(fd, data, nb, offset));

    if (nbytes < 0) {
        ERROR("Write error attempting to write {:d} bytes at offset {:d} into file {:s}: {:s}", nb,
              offset, _name, strerror(errno));
        return false;
    }

    return true;
}

void hfbFileRaw::write_sample(uint32_t time_ind, uint32_t freq_ind, const FrameView& frame_view) {

    const HfbFrameView& frame = static_cast<const HfbFrameView&>(frame_view);

    // TODO: consider adding checks for all dims
    // if (frame.num_ev != num_ev) {
    //    throw std::runtime_error(fmt::format(fmt("Number of eigenvalues don't match for write (got
    //    "
    //                                             "{:d}, expected {:d})"),
    //                                         frame.num_ev, num_ev));
    //}

    const uint8_t ONE = 1;

    // Write out data to the right place
    off_t offset = (time_ind * nfreq + freq_ind) * frame_size;

    write_raw(offset, 1, &ONE);
    write_raw(offset + 1, metadata_size, frame.metadata());
    write_raw(offset + 1 + metadata_size, data_size, frame.data());
}
