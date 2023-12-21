#ifndef HFB_METADATA
#define HFB_METADATA

#include "Telescope.hpp"
#include "buffer.hpp"
#include "dataset.hpp" // for dset_id_t
#include "metadata.hpp"

#include <sys/time.h>

class HFBMetadata : public metadataObject {
public:
    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;

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

inline int64_t get_fpga_seq_start_hfb(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->fpga_seq_start;
}

inline struct timespec get_ctime_hfb(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->ctime;
}

inline freq_id_t get_freq_id(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->freq_id;
}

inline uint64_t get_fpga_seq_total(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->fpga_seq_total;
}

inline uint64_t get_fpga_seq_length(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->fpga_seq_length;
}

inline uint32_t get_num_beams(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->num_beams;
}

inline uint32_t get_num_subfreq(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->num_subfreq;
}

inline dset_id_t get_dataset_id_hfb(Buffer* buf, int ID) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    return hfb_metadata->dataset_id;
}

// Setting functions

inline void set_fpga_seq_start_hfb(Buffer* buf, int ID, int64_t seq) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->fpga_seq_start = seq;
}

inline void set_ctime_hfb(Buffer* buf, int ID, struct timespec time) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->ctime = time;
}

inline void set_freq_id(Buffer* buf, int ID, freq_id_t freq_id) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->freq_id = freq_id;
}

inline void set_fpga_seq_total(Buffer* buf, int ID, uint64_t num_samples) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->fpga_seq_total = num_samples;
}

inline void set_fpga_seq_length(Buffer* buf, int ID, uint64_t num_samples) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->fpga_seq_length = num_samples;
}

inline void set_num_beams(Buffer* buf, int ID, uint32_t num_beams) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->num_beams = num_beams;
}

inline void set_num_subfreq(Buffer* buf, int ID, uint32_t num_subfreq) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->num_subfreq = num_subfreq;
}

inline void set_dataset_id_hfb(Buffer* buf, int ID, dset_id_t dataset_id) {
    HFBMetadata* hfb_metadata = (HFBMetadata*)buf->metadata[ID].get();
    hfb_metadata->dataset_id = dataset_id;
}

#endif
