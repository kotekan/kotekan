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

#endif
