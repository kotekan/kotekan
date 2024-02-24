#include "chordMetadata.hpp"

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
    chime(), frame_counter(-1), type(unknown_type), dims(-1), offset(0), n_one_hot(-1), nfreq(-1) {
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
        for (int v = 0; v < CHORD_META_MAX_VIS_SAMPLES; ++v) {
            lost_fpga_samples[f][v] = 0;
            rfi_flagged_samples[f][v] = 0;
        }
    }
}
