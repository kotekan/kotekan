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

