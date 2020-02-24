#ifndef HFB_METADATA
#define HFB_METADATA

#include "buffer.h"
#include "metadata.h"

#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack()

struct hfbMetadata {
    /// The ICEBoard sequence number
    int64_t fpga_seq_num;
    /// The GPS time of @c fpga_seq_num.
    struct timespec gps_time;
    /// Is GPS time accurate?
    uint32_t gps_time_flag;
    /// Frequency bin
    uint32_t freq_bin_num;
    /// Normalisation fraction
    float norm_frac;
    /// Number of samples integrated
    uint32_t num_samples_integrated;
    /// Number of samples expected
    uint32_t num_samples_expected;
    /// Data size of frame after compression
    uint32_t compressed_data_size;
    /// The number of beams in the data.
    uint32_t num_beams;
    /// The number of sub-frequencies in the data.
    uint32_t num_subfreq;
    /// ID of the dataset
    dset_id_t dataset_id;
};

// Helper functions to save lots of pointer work

inline int64_t get_fpga_seq_num_hfb(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->fpga_seq_num;
}

inline uint32_t get_gps_time_flag(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->gps_time_flag;
}

inline struct timespec get_gps_time_hfb(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->gps_time;
}

inline uint32_t get_freq_bin_num(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->freq_bin_num;
}

inline float get_norm_frac(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->norm_frac;
}

inline uint32_t get_num_samples_integrated(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->num_samples_integrated;
}

inline uint32_t get_num_samples_expected(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->num_samples_expected;
}

inline int64_t get_compressed_data_size_hfb(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->compressed_data_size;
}

inline uint32_t get_num_beams(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->num_beams;
}

inline uint32_t get_num_subfreq(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->num_subfreq;
}

inline dset_id_t get_dataset_id(struct Buffer* buf, int ID) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->dataset_id;
}

// Setting functions

inline void set_fpga_seq_num_hfb(struct Buffer* buf, int ID, int64_t seq) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->fpga_seq_num = seq;
}

inline void set_gps_time_flag(struct Buffer* buf, int ID, uint32_t flag) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->gps_time_flag = flag;
}

inline void set_gps_time_hfb(struct Buffer* buf, int ID, struct timespec time) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->gps_time = time;
}

inline void set_freq_bin_num(struct Buffer* buf, int ID, uint32_t bin_num) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->freq_bin_num = bin_num;
}

inline void set_norm_frac(struct Buffer* buf, int ID, float frac) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->norm_frac = frac;
}

inline void set_num_samples_integrated(struct Buffer* buf, int ID, uint32_t num_samples) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->num_samples_integrated = num_samples;
}

inline void set_num_samples_expected(struct Buffer* buf, int ID, uint32_t num_samples) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->num_samples_expected = num_samples;
}

inline void set_compressed_data_size_hfb(struct Buffer* buf, int ID, uint32_t compressed_size) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->compressed_data_size = compressed_size;
}

inline void set_num_beams(struct Buffer* buf, int ID, uint32_t num_beams) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->num_beams = num_beams;
}

inline void set_num_subfreq(struct Buffer* buf, int ID, uint32_t num_subfreq) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->num_subfreq = num_subfreq;
}

inline void set_dataset_id(struct Buffer* buf, int ID, dset_id_t dataset_id) {
    struct hfbMetadata* hfb_metadata = (struct hfbMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->dataset_id = dataset_id;
}
#ifdef __cplusplus
}
#endif

#endif
