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
const int CHORD_META_MAX_FREQ = 16;

// Maximum number of dimensions for arrays
const int CHORD_META_MAX_DIM = 10;

// Maximum length of dimension names for arrays
const int CHORD_META_MAX_DIMNAME = 16;

struct chordMetadata {
    struct chimeMetadata chime;
    int frame_counter;

    //cudaDataType_t type;
    chordDataType type;

    int dims;
    int dim[CHORD_META_MAX_DIM];
    // array strides / layouts?
    char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME]; // 'F', 'T', 'D', etc

    // One-hot arrays?
    int n_one_hot;
    char onehot_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME];
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

    std::string get_dimension_name(size_t i) {
        return std::string(dim_name[i], strnlen(dim_name[i], CHORD_META_MAX_DIMNAME));
    }

    std::string get_dimensions_string() {
        std::string s;
        for (int i=0; i<this->dims; i++) {
            if (i)
                s += " x ";
            s += get_dimension_name(i) + "(";
            s += std::to_string(this->dim[i]) + ")";
        }
        return s;
    }

    std::string get_onehot_name(size_t i) {
        return std::string(onehot_name[i], strnlen(onehot_name[i], CHORD_META_MAX_DIMNAME));
    }

    std::string get_onehot_string() {
        std::string s;
        for (int i=0; i<this->n_one_hot; i++) {
            if (i)
                s += ", ";
            s += get_onehot_name(i) + "=";
            s += std::to_string(this->onehot_index[i]);
        }
        return s;
    }

    void set_array_dimension(int dim, int size, std::string name) {
        assert(dim < CHORD_META_MAX_DIM);
        this->dim[dim] = size;
        strncpy(this->dim_name[dim], name.c_str(), CHORD_META_MAX_DIMNAME);
    }

    void set_onehot_dimension(int dim, int i, std::string name) {
        assert(dim < CHORD_META_MAX_DIM);
        this->onehot_index[dim] = i;
        strncpy(this->onehot_name[dim], name.c_str(), CHORD_META_MAX_DIMNAME);
    }

};

inline void chord_metadata_init(struct chordMetadata* c) {
    bzero(c, sizeof(struct chordMetadata));
}

inline void chord_metadata_copy(struct chordMetadata* out, const struct chordMetadata* in) {
    memcpy(out, in, sizeof(struct chordMetadata));
}

inline bool metadata_is_chord(struct Buffer* buf, int) {
    return strcmp(buf->metadata_pool->type_name, "chordMetadata") == 0;
}

inline bool metadata_container_is_chord(struct metadataContainer* mc) {
    return strcmp(mc->parent_pool->type_name, "chordMetadata") == 0;
}

inline struct chordMetadata* get_chord_metadata(struct Buffer* buf, int frame_id) {
    return (struct chordMetadata*)buf->metadata[frame_id]->metadata;
}

#endif
