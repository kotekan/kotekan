#ifndef HFB_METADATA
#define HFB_METADATA

#include "Telescope.hpp"
#include "buffer.h"
#include "dataset.hpp" // for dset_id_t
#include "metadata.h"

#include <sys/time.h>

struct HFBMetadata {
    /// The ICEBoard sequence number
    int64_t fpga_seq_start;
    /// The GPS time of @c fpga_seq_start.
    struct timespec ctime;
    /// Frequency bin
    freq_id_t freq_id;
    /// Number of samples integrated
    uint64_t fpga_seq_total;
    /// Number of samples expected
    uint64_t fpga_seq_length;
    /// The number of beams in the data.
    uint32_t num_beams;
    /// The number of sub-frequencies in the data.
    uint32_t num_subfreq;
    /// ID of the dataset
    dset_id_t dataset_id;
};

// Helper functions to save lots of pointer work

inline int64_t get_fpga_seq_start_hfb(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->fpga_seq_start;
}

inline struct timespec get_ctime_hfb(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->ctime;
}

inline freq_id_t get_freq_id(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->freq_id;
}

inline uint64_t get_fpga_seq_total(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->fpga_seq_total;
}

inline uint64_t get_fpga_seq_length(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->fpga_seq_length;
}

inline uint32_t get_num_beams(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->num_beams;
}

inline uint32_t get_num_subfreq(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->num_subfreq;
}

inline dset_id_t get_dataset_id_hfb(struct Buffer* buf, int ID) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    return hfb_metadata->dataset_id;
}

// Setting functions

inline void set_fpga_seq_start_hfb(struct Buffer* buf, int ID, int64_t seq) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->fpga_seq_start = seq;
}

inline void set_ctime_hfb(struct Buffer* buf, int ID, struct timespec time) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->ctime = time;
}

inline void set_freq_id(struct Buffer* buf, int ID, freq_id_t freq_id) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->freq_id = freq_id;
}

inline void set_fpga_seq_total(struct Buffer* buf, int ID, uint64_t num_samples) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->fpga_seq_total = num_samples;
}

inline void set_fpga_seq_length(struct Buffer* buf, int ID, uint64_t num_samples) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->fpga_seq_length = num_samples;
}

inline void set_num_beams(struct Buffer* buf, int ID, uint32_t num_beams) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->num_beams = num_beams;
}

inline void set_num_subfreq(struct Buffer* buf, int ID, uint32_t num_subfreq) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->num_subfreq = num_subfreq;
}

inline void set_dataset_id_hfb(struct Buffer* buf, int ID, dset_id_t dataset_id) {
    struct HFBMetadata* hfb_metadata = (struct HFBMetadata*)buf->metadata[ID]->metadata;
    hfb_metadata->dataset_id = dataset_id;
}

#endif
