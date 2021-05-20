#include "FileArchive.hpp"

// Assign compression algorithm and input arguments
FileArchive::FileArchive(const std::string& comp_alg, const int zstd_comp_lv) {
    if (comp_alg == "lz4") {
        BSHUF_CD.push_back(BSHUF_H5_COMPRESS_LZ4);
    } else if (comp_alg == "zstd") {
        BSHUF_CD.push_back(BSHUF_H5_COMPRESS_ZSTD);
        BSHUF_CD.push_back(zstd_comp_lv);
    } else {
        throw std::runtime_error(
            fmt::format(fmt("The compression algorithm selected: '{:s}' is not supported. Choose "
                            "from: ['lz4', 'zstd'] instead."),
                        comp_alg));
    }
}

FileArchive::~FileArchive() {}
