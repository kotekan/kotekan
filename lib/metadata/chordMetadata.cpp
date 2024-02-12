#include "chordMetadata.hpp"

REGISTER_TYPE_WITH_FACTORY(metadataObject, chordMetadata);

const char* chord_datatype_string(chordDataType type) {
    switch (type) {
        case int4p4:
            return "int4p4";
        case int8:
            return "int8";
        case int16:
            return "int16";
        case int32:
            return "int32";
        case int64:
            return "int64";
        case float16:
            return "float16";
        case float32:
            return "float32";
        case float64:
            return "float64";
        case unknown_type:
        default:
            return "<unknown-type>";
    }
}

chordMetadata::chordMetadata() :
    chimeMetadata(), frame_counter(-1), type(unknown_type), dims(-1), offset(0), n_one_hot(-1),
    nfreq(-1), ndishes(-1), n_dish_locations_ew(-1), n_dish_locations_ns(-1), dish_index(nullptr) {
    for (int d = 0; d < CHORD_META_MAX_DIM; ++d) {
        dim[d] = -1;
        dim_name[d][0] = '\0';
        strides[d] = -1;
        onehot_name[d][0] = '\0';
        onehot_index[d] = -1;
    }
    for (int f = 0; f < CHORD_META_MAX_FREQ; ++f) {
        coarse_freq[f] = -1;
        freq_upchan_factor[f] = -1;
        half_fpga_sample0[f] = -1;
        time_downsampling_fpga[f] = -1;
    }
}

struct chordMetadataFormat {
    int32_t max_dim;
    int32_t max_dimname;
    int32_t max_freq;

    int32_t frame_counter;

    // chordDataType type;
    int32_t type;

    int32_t dims;
    int32_t dim[CHORD_META_MAX_DIM];
    char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME]; // "F", "Tbar", "D", etc
    int64_t strides[CHORD_META_MAX_DIM];
    int64_t offset;

    // One-hot arrays?
    int32_t n_one_hot;
    char onehot_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME];
    int32_t onehot_index[CHORD_META_MAX_DIM];

    // Per-frequency arrays
    int32_t nfreq;

    // frequencies -- integer (0-2047) identifier for FPGA coarse frequencies
    int32_t coarse_freq[CHORD_META_MAX_FREQ];

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    int32_t freq_upchan_factor[CHORD_META_MAX_FREQ];

    // Time sampling -- for each coarse frequency channel, 2x the FPGA
    // sample number of the first sample.  The 2x is there to handle
    // the upchannelization case, where 2 or more samples may get
    // averaged, producing a new sample that is effectively halfway in
    // between them, ie, at a half-FPGAsample time.
    int64_t half_fpga_sample0[CHORD_META_MAX_FREQ];

    // Time sampling -- for each coarse frequency channel, the factor
    // by which the time samples have been downsampled relative to
    // FPGA samples.
    int32_t time_downsampling_fpga[CHORD_META_MAX_FREQ];
};

size_t chordMetadata::get_serialized_size() {
    return chimeMetadata::get_serialized_size() + sizeof(chordMetadataFormat);
}

size_t chordMetadata::set_from_bytes(const char* bytes, size_t length) {
    assert(length >= get_serialized_size());
    size_t offset = chimeMetadata::set_from_bytes(bytes, length);
    bytes += offset;
    length -= offset;
    assert(length >= sizeof(chordMetadataFormat));

    const chordMetadataFormat* fmt = reinterpret_cast<const chordMetadataFormat*>(bytes);

    frame_counter = fmt->frame_counter;
    type = (chordDataType)fmt->type;
    assert(CHORD_META_MAX_DIM == fmt->max_dim);
    assert(CHORD_META_MAX_DIMNAME == fmt->max_dimname);
    assert(CHORD_META_MAX_FREQ == fmt->max_freq);
    dims = fmt->dims;
    assert(dims < CHORD_META_MAX_DIM);
    for (int i = 0; i < dims; i++) {
        dim[i] = fmt->dim[i];
        for (int j = 0; j < CHORD_META_MAX_DIMNAME; j++) {
            dim_name[i][j] = fmt->dim_name[i][j];
            onehot_name[i][j] = fmt->onehot_name[i][j];
        }
        strides[i] = fmt->strides[i];
        onehot_index[i] = fmt->onehot_index[i];
    }
    offset = fmt->offset;
    n_one_hot = fmt->n_one_hot;
    nfreq = fmt->nfreq;
    assert(nfreq < CHORD_META_MAX_FREQ);
    for (int i = 0; i < nfreq; i++) {
        coarse_freq[i] = fmt->coarse_freq[i];
        freq_upchan_factor[i] = fmt->freq_upchan_factor[i];
        half_fpga_sample0[i] = fmt->half_fpga_sample0[i];
        time_downsampling_fpga[i] = fmt->time_downsampling_fpga[i];
    }
    return offset + sizeof(chordMetadataFormat);
}

size_t chordMetadata::serialize(char* bytes) {
    size_t offset = chimeMetadata::serialize(bytes);
    bytes += offset;

    chordMetadataFormat* fmt = reinterpret_cast<chordMetadataFormat*>(bytes);
    memset(fmt, 0, sizeof(chordMetadataFormat));

    fmt->max_dim = CHORD_META_MAX_DIM;
    fmt->max_dimname = CHORD_META_MAX_DIMNAME;
    fmt->max_freq = CHORD_META_MAX_FREQ;

    fmt->frame_counter = frame_counter;
    fmt->type = (int32_t)fmt->type;
    fmt->dims = dims;
    for (int i = 0; i < dims; i++) {
        fmt->dim[i] = dim[i];
        for (int j = 0; j < CHORD_META_MAX_DIMNAME; j++) {
            fmt->dim_name[i][j] = dim_name[i][j];
            fmt->onehot_name[i][j] = onehot_name[i][j];
        }
        fmt->strides[i] = strides[i];
        fmt->onehot_index[i] = onehot_index[i];
    }
    fmt->offset = offset;
    fmt->n_one_hot = n_one_hot;
    fmt->nfreq = nfreq;
    assert(nfreq < CHORD_META_MAX_FREQ);
    for (int i = 0; i < nfreq; i++) {
        fmt->coarse_freq[i] = coarse_freq[i];
        fmt->freq_upchan_factor[i] = freq_upchan_factor[i];
        fmt->half_fpga_sample0[i] = half_fpga_sample0[i];
        fmt->time_downsampling_fpga[i] = time_downsampling_fpga[i];
    }
    return offset + sizeof(chordMetadataFormat);
}
