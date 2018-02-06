#ifndef WRITE_EIGENVEC
#define WRITE_EIGENVEC

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <highfive/H5File.hpp>

// TODO: Should define a class for the eignevector file

class writeEigenvec : public KotekanProcess {

public:
    writeEigenvec(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:
    /// Acquisition parameters saved from the config files
    size_t num_elements, num_eigenvectors;

    /// File to write to
    std::string fname;
    File file;

    /// Number of frames to hold in file
    uint32_t file_len;

    /// Input buffer
    Buffer * in_buf;

};

class evFile {

public:
    evFile(const std:string & fname,
           const std:string & path,
           const uint16_t & num_eigenvectors,
           const std::vector<freq_ctype> & freqs,
           const std::vector<input_ctype> & inputs,
           const std::vector<prod_ctype> & prods);

    ~evFile();

    write_eigenvectors(time_ctype new_time, uint32_t freq_ind,
                       std::complex<float> eigenvector)

private:

    // file datasets
    DataSet ev;
    DataSet time_imap;
    DataSet freq_imap;
    DataSet input_imap;

    // current position in file
    size_t curr_ind;

};

#endif
