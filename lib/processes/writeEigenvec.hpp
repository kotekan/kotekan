#ifndef WRITE_EIGENVEC
#define WRITE_EIGENVEC

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>


class evFile {

public:
    evFile(const std::string & fname,
           const uint16_t & num_eigenvectors,
           const size_t & num_times,
           const std::vector<freq_ctype> & freqs,
           const std::vector<input_ctype> & inputs);

    ~evFile();

    void flush();

    /// Write a set of eigenvectors/values to file for a given time and frequency
    void write_eigenvectors(time_ctype new_time, uint32_t freq_ind,
                            std::vector<std::complex<float>> eigenvectors,
                            std::vector<float> eigenvalues, float rms);

    /// Access datasets
    HighFive::DataSet evec();
    HighFive::DataSet eval();
    HighFive::DataSet rms();
    HighFive::DataSet time();
    HighFive::DataSet freq();
    HighFive::DataSet prod();

private:

    // length of file
    size_t ntimes;
    size_t ninput;
    size_t nfreq;
    size_t nev;
    // current timestamps held in file
    std::vector<uint64_t> curr_times;
    // position of the 'end' of the ring buffer
    size_t eof_ind;

    std::unique_ptr<HighFive::File> file;

};


class writeEigenvec : public KotekanProcess {

public:
    writeEigenvec(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);

    ~writeEigenvec();

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:
    /// Number of eigenvectors that will be provided
    size_t num_eigenvectors;
    /// File path to write to
    std::string ev_fname;
    /// Number of frames to hold in file
    size_t ev_file_len;
    /// Which half of the band this receiver node holds
    int freq_half;

    /// Vectors of frequencies and inputs in the data
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;

    /// File to write to
    std::unique_ptr<evFile> file;

    /// Input buffer
    Buffer * in_buf;

};


#endif
