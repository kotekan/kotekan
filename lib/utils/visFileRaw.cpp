
#include "visFileRaw.hpp"
#include "errors.h"
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <libgen.h>
#include <errno.h>
#include "fmt.hpp"


// Register the HDF5 file writers
REGISTER_VIS_FILE("raw", visFileRaw);

//
// Implementation of standard HDF5 visibility data file
//

void visFileRaw::create_file(const std::string& name,
                             const std::map<std::string, std::string>& metadata,
                             const std::vector<freq_ctype>& freqs,
                             const std::vector<input_ctype>& inputs,
                             const std::vector<prod_ctype>& prods,
                             size_t num_ev, size_t num_time) {

    INFO("Creating new output file %s", name.c_str());

    file_metadata["attributes"] = metadata;
    file_metadata["index_map"]["freq"] = freqs;
    file_metadata["index_map"]["input"] = inputs;
    file_metadata["index_map"]["prod"] = prods;

    // Create and add eigenvalue index
    std::vector<int> eval_index(num_ev);
    std::iota(eval_index.begin(), eval_index.end(), 0);
    file_metadata["index_map"]["ev"] = eval_index;

    nfreq = freqs.size();
    size_t ninput = inputs.size(), nprod = prods.size();

    // Set the alignment
    // TODO: find some way of getting this from config
    alignment = 4;  // Align on page boundaries

    // Calculate the file structure
    auto layout = visFrameView::calculate_buffer_layout(
        ninput, nprod, num_ev
    );
    data_size = layout["_struct"].second;
    metadata_size = sizeof(visMetadata);
    frame_size = _member_alignment(data_size + metadata_size + 1,
                                   alignment * 1024);

    // Write the structure into the file for decoding
    file_metadata["structure"]["metadata_size"] = metadata_size;
    file_metadata["structure"]["data_size"] = data_size;
    file_metadata["structure"]["frame_size"] = frame_size;
    file_metadata["structure"]["nfreq"] = nfreq;

    // Create lock file and then open the other files
    lock_filename = create_lockfile(name);
    metadata_file = std::ofstream(name + ".meta", std::ios::binary);
    if((fd = open((name + ".data").c_str(), O_CREAT | O_EXCL | O_WRONLY,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) {
        std::runtime_error(fmt::format("Failed to open file {}: {}.",
                                       name + ".data", strerror(errno)));
    }
    std::cout << (name + ".data ") << fd << std::endl;

    // TODO: Preallocate data file (without increasing the length)
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


uint32_t visFileRaw::extend_time(time_ctype new_time) {

    times.push_back(new_time);

    ftruncate(fd, frame_size * nfreq * num_time());

    // Extend the file length
    return num_time() - 1;
}


bool visFileRaw::write_raw(off_t offset, size_t nb, const void* data) {

    // Write in a retry macro loop incase the write was interrupted by a signal
    int nbytes = TEMP_FAILURE_RETRY( 
        pwrite(fd, data, nb, offset)
    );

    if(nbytes < 0) {
        ERROR("Write error attempting to write %i bytes at offset %i: %s",
              nb, offset, strerror(errno));
        return false;
    }

    return true;
}

void visFileRaw::write_sample(
    uint32_t time_ind, uint32_t freq_ind, const visFrameView& frame
) {

    const uint8_t ONE = 1;

    // Write out data to the right place
    off_t offset = (time_ind * nfreq + freq_ind) * frame_size;

    write_raw(offset, 1, &ONE);
    write_raw(offset + 1, metadata_size, frame.metadata());
    write_raw(offset + 1 + metadata_size, data_size, frame.data());
}