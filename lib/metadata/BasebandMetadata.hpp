#ifndef BASEBAND_METADATA_HPP
#define BASEBAND_METADATA_HPP

#include "Telescope.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "metadata.hpp"

class BasebandMetadata : public metadataObject {
public:
    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;

    /// event and frequency ID
    uint64_t event_id;
    uint64_t freq_id;

    //@{
    /// Event start and end fpga seq at this frequency
    uint64_t event_start_fpga;
    uint64_t event_end_fpga;
    //@}

    //@{
    /// Timestamp of the first captured sample
    uint64_t time0_fpga;
    double time0_ctime;
    double time0_ctime_offset;
    //@}

    /// Time of arrival of the packet containing the first sample
    double first_packet_recv_time;

    /// FPGA seq of the first sample in this frame
    int64_t frame_fpga_seq;

    /// Number of valid samples in this frame
    int64_t valid_to;

    /// The time of FPGA frame=0
    uint64_t fpga0_ns;

    /// Number of inputs per sample
    int32_t num_elements;

    /// Future expansion+padding
    int32_t reserved;
};

#endif // BASEBAND_METADATA_HPP
