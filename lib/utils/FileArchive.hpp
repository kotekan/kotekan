#ifndef FILE_ARCHIVE_HPP
#define FILE_ARCHIVE_HPP

#include "kotekanLogging.hpp" // for logLevel, kotekanLogging

#include <highfive/H5PropertyList.hpp> // for H5Pcreate, H5Pset_chunk, H5Pset_filter, H5P_DATAS...

/** @brief A Bitshuffle header file.
 *
 * Header file to store common Bitshuffle constants.
 *
 * @author James Willis
 **/
class FileArchive : public kotekan::kotekanLogging {

protected:
    // Bitshuffle parameters
    H5Z_filter_t H5Z_BITSHUFFLE = 32008;
    unsigned int BSHUF_H5_COMPRESS_LZ4 = 2;
    unsigned int BSHUF_BLOCK = 0; // let bitshuffle choose

    const std::vector<unsigned int> BSHUF_CD = {BSHUF_BLOCK, BSHUF_H5_COMPRESS_LZ4};
};

#endif
