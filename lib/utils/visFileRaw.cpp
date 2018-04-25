
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
                        
    std::string data_filename = name + ".raw";

    lock_filename = create_lockfile(name);

    // Determine whether to write the eigensector or not...
    write_ev = (num_ev > 0);

    INFO("Creating new output file %s", name.c_str());

    file_metadata["attributes"] = metadata;
    file_metadata["index_map"]["freq"] = freqs;
    file_metadata["index_map"]["inputs"] = inputs;
    file_metadata["index_map"]["prods"] = prods;

    std::vector<int> eval_index;
    std::iota(eval_index.begin(), eval_index.end(), 0);
    file_metadata["index_map"]["ev"] = eval_index;

    nfreq = freqs.size();

    // Open data file

    // Preallocate data file (without increasing the length)
}

visFileRaw::~visFileRaw() {

    file_metadata["index_map"]["time"] = times;
    // Close up file and maybe sync
    // Write out metadata

    std::remove(lock_filename.c_str());
}

size_t visFileRaw::num_time() {
    return times.size();
}


uint32_t visFileRaw::extend_time(time_ctype new_time) {

    times.push_back(new_time);

    // Extend the file length
    return num_time() - 1;
}


void visFileRaw::write_sample(
    uint32_t time_ind, uint32_t freq_ind, std::vector<cfloat> new_vis,
    std::vector<float> new_weight, std::vector<cfloat> new_gcoeff,
    std::vector<int32_t> new_gexp, std::vector<float> new_eval,
    std::vector<cfloat> new_evec, float new_erms
) {

    // Write out data to the right place

    size_t nb = n * sizeof(T);
    off_t offset = dset_base + ind * nb;

    // Write in a retry macro loop incase the write was interrupted by a signal
    int nbytes = TEMP_FAILURE_RETRY( 
        pwrite(fd, (const void *)data, nb, offset)
    );

    if(nbytes < 0) {
        ERROR("Write error attempting to write %i bytes at offset %i: %s",
              nb, offset, strerror(errno));
        return false;
    }

    return true;
}