#ifndef CHORD_METADATA
#define CHORD_METADATA

#include "Telescope.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "datasetManager.hpp"
#include "metadata.h"

#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

#pragma pack()

enum chordDataType { unknown_type, int4p4, int8, int16, int32, int64, float16, float32, float64 };

constexpr std::size_t chord_datatype_bytes(chordDataType type) {
    switch (type) {
        case int4p4:
            return 1;
        case int8:
            return 1;
        case int16:
            return 2;
        case int32:
            return 4;
        case int64:
            return 8;
        case float16:
            return 2;
        case float32:
            return 4;
        case float64:
            return 8;
        case unknown_type:
        default:
            return -1;
    }
}

const char* chord_datatype_string(chordDataType type);

// Maximum number of frequencies in metadata array
const int CHORD_META_MAX_FREQ = 16;

// Maximum number of dimensions for arrays
const int CHORD_META_MAX_DIM = 10;

// Maximum length of dimension names for arrays
const int CHORD_META_MAX_DIMNAME = 16;

struct chordMetadata {
    chimeMetadata chime;
    int frame_counter;

    // cudaDataType_t type;
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

    std::string get_dimension_name(size_t i) const {
        return std::string(dim_name[i], strnlen(dim_name[i], CHORD_META_MAX_DIMNAME));
    }

    std::string get_type_string() const {
        return std::string(chord_datatype_string(type));
    }

    std::string get_dimensions_string() const {
        std::ostringstream s;
        for (int i = 0; i < this->dims; i++) {
            if (i)
                s << " x ";
            s << get_dimension_name(i) << "(" << dim[i] << ")";
        }
        return s.str();
    }

    std::string get_onehot_name(size_t i) const {
        return std::string(onehot_name[i], strnlen(onehot_name[i], CHORD_META_MAX_DIMNAME));
    }

    std::string get_onehot_string() const {
        std::ostringstream s;
        for (int i = 0; i < this->n_one_hot; i++) {
            if (i)
                s << ", ";
            s << get_onehot_name(i) << "=" << onehot_index[i];
        }
        return s.str();
    }

    void set_array_dimension(int dim, int size, const std::string& name) {
        assert(dim < CHORD_META_MAX_DIM);
        this->dim[dim] = size;
        strncpy(this->dim_name[dim], name.c_str(), CHORD_META_MAX_DIMNAME);
    }

    void set_onehot_dimension(int dim, int i, const std::string& name) {
        assert(dim < CHORD_META_MAX_DIM);
        this->onehot_index[dim] = i;
        strncpy(this->onehot_name[dim], name.c_str(), CHORD_META_MAX_DIMNAME);
    }
};

inline void chord_metadata_init(chordMetadata* c) {
    bzero(c, sizeof(chordMetadata));
}

inline void chord_metadata_copy(chordMetadata* out, const chordMetadata* in) {
    memcpy(out, in, sizeof(chordMetadata));
}

inline bool metadata_is_chord(Buffer* buf, int) {
    return strcmp(buf->metadata_pool->type_name, "chordMetadata") == 0;
}

inline bool metadata_container_is_chord(const metadataContainer* mc) {
    return strcmp(mc->parent_pool->type_name, "chordMetadata") == 0;
}

inline const chordMetadata* get_chord_metadata(const Buffer* buf, int frame_id) {
    return (const chordMetadata*)buf->metadata[frame_id]->metadata;
}

inline chordMetadata* get_chord_metadata(Buffer* buf, int frame_id) {
    return (chordMetadata*)buf->metadata[frame_id]->metadata;
}

inline const chordMetadata* get_chord_metadata(const metadataContainer* mc) {
    if (!mc)
        return nullptr;
    if (strcmp(mc->parent_pool->type_name, "chordMetadata")) {
        WARN_NON_OO("Expected metadata to be type \"chordMetadata\", got \"{:s}\".",
                    mc->parent_pool->type_name);
        return nullptr;
    }
    return static_cast<const chordMetadata*>(mc->metadata);
}

inline chordMetadata* get_chord_metadata(metadataContainer* mc) {
    if (!mc)
        return nullptr;
    if (strcmp(mc->parent_pool->type_name, "chordMetadata")) {
        WARN_NON_OO("Expected metadata to be type \"chordMetadata\", got \"{:s}\".",
                    mc->parent_pool->type_name);
        return nullptr;
    }
    return static_cast<chordMetadata*>(mc->metadata);
}

#endif
