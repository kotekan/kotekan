#ifndef HDF5_WRITER_HPP
#define HDF5_WRITER_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <unistd.h>
#include <highfive/H5File.hpp>

/*
 * Displays CHIME metedata for a given buffer
 * "buf": String with the name of the buffer display metadata info for.
 */

class hdf5Writer : public KotekanProcess {
public:
    hdf5Writer(Config &config,
               const string& unique_name,
               bufferContainer &buffer_container);
    ~hdf5Writer();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf;
    int32_t len;
    int32_t offset;

    HighFive::File * current_file;
};

#endif
