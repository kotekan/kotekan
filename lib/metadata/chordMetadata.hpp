#ifndef CHORD_METADATA
#define CHORD_METADATA

#include "Telescope.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "datasetManager.hpp"
#include "metadata.h"

#include <sys/time.h>
#include <vector>

#pragma pack()

enum chordDataType {
    int4p4, int8, float16, float32
};

// Maximum number of frequencies in metadata array
const int CHORD_META_MAX_FREQ = 1024;

// Maximum number of dimensions for arrays
const int CHORD_META_MAX_DIM = 10;

struct chordMetadata {
    struct chimeMetadata chime;
    int frame_counter;
    // TODO -- make this a static array because metadata objects get c-copied around.
    //std::vector<int> hotIndices;

    //cudaDataType_t type;
    //std::string type;
    //char type[16];
    chordDataType type;

    int dims;
    int dim[CHORD_META_MAX_DIM];
    // array strides / layouts?
    char dim_names[CHORD_META_MAX_DIM]; // 'F', 'T', 'D', etc

    // One-hot arrays?
    int n_one_hot;
    char onehot_name[CHORD_META_MAX_DIM];
    int onehot_index[CHORD_META_MAX_DIM];

    // Per-frequency arrays
    int nfreq;

    // frequencies -- integer (0-2047) identifier for FPGA coarse frequencies
    int coarse_freq[CHORD_META_MAX_FREQ];

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    int freq_upchan_factor[CHORD_META_MAX_FREQ];

    // Time sampling -- for each coarse frequency channel, 2x the FPGA
    // sample number of the first sample.  The 2x is there to handle
    // the upchannelization case, where 2 or more samples may get
    // averaged, producing a new sample that is effectively halfway in
    // between them, ie, at a half-FPGAsample time.
    int64_t half_fpga_sample0[CHORD_META_MAX_FREQ];

    // Time sampling -- for each coarse frequency channel, the factor
    // by which the time samples have been downsampled relative to
    // FPGA samples.
    int time_downsampling_fpga[CHORD_META_MAX_FREQ];

    std::string get_dimensions_string();

    std::string get_onehot_string();

    void set_array_dimension(int dim, int size, char name);

    void set_onehot_dimension(int dim, int size, char name);

};

inline void chord_metadata_copy(struct chordMetadata* out, const struct chordMetadata* in) {
    memcpy(out, in, sizeof(struct chordMetadata));
}

#endif
