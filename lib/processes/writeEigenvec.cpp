#include "writeEigenvec.hpp"
#include "visBuffer.hpp"
#include "chimeMetadata.h"
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

writeEigenvec::writeEigenvec(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&writeEigenvec::main_thread, this)) {

    // get buffer
    // open file for writing
    //      need to deal with existing files

}

writeEigenvec::~writeEigenvec() {

    file->flush();
    // TODO: delete file?
    // TODO: should be part of a eigenvecFile class
}

void writeEigenvec:::apply_config(uint64_t fpga_seq) {

}

void writeEigenvec:::main_thread() {

    // loop over input frames
    // write to file
    //      need to keep track of 

}


evFile::evFile(const std:string & fname,
               const uint16_t & num_eigenvectors,
               const size_t & num_times,
               const std::vector<freq_ctype> & freqs,
               const std::vector<input_ctype> & inputs) {

    size_t ninput = inputs.size();
    size_t nfreq = freqs.size()

    // Create file
    // TODO: Should be robust to existing file
    INFO("Creating new eigenvectors file %s", fname.c_str());
    file = File(fname, File::SwmrCreate);

    // Create eigenvector dataset
    std::vector<size_t> dims = {num_times, nfreq, num_eigenvectors, ninput};
    std::vector<std::string> axes = {"time", "freq", "eigenvector", "input"};
    DataSpace ev_space = DataSpace(dims);
    DataSet ev = file->createDataSet(
        "eigenvector", ev_space, create_datatype<std::complex<float>>(), dims
    );
    ev.createAttribute<std::string>("axis", DataSpace::From(axes)).write(axes);

    // Create index map
    Group indexmap = file->createGroup("index_map");

    DataSet time_imap = indexmap.createDataSet(
      "time", DataSpace(num_times), create_datatype<time_ctype>()
    );

    DataSet freq_imap = indexmap.createDataSet<freq_ctype>("freq", DataSpace(nfreq));
    freq_imap.write(freqs);

    DataSet input_imap = indexmap.createDataSet<input_ctype>("input", DataSpace(ninput));
    input_imap.write(inputs);

    // Start writing at top of file
    curr_ind = 0;

}

evFile::~evFile() {
    file->flush();
    // TODO: delete file?
}

evFile::write_eigenvector(time_ctype new_time, uint32_t freq_ind,
                          std::complex<float> eigenvector) {
    // write eigenvectors
    // TODO: check dimensions work out
    ev.select({curr_ind, freq_ind, 0, 0}, {1, 1, num_eigenvectors, ninput}).write(eigenvector);
    // write time
    time_imap.select({curr_ind}, {1}).write(new_time);
    // increment file position, going back to top after reaching end of file
    curr_ind = (curr_ind + 1) % num_times;

}
