#ifndef BEAMMETADATA_HPP
#define BEAMMETADATA_HPP

#include "Telescope.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "dataset.hpp" // for dset_id_t
#include "metadata.hpp"

class BeamMetadata : public metadataObject {
public:
    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;

    nlohmann::json to_json() override;

    /// The ICEBoard sequence number
    int64_t fpga_seq_start;
    /// The GPS time of @c fpga_seq_start.
    timespec ctime;
    /// Stream identifier
    stream_t stream_id;
    /// ID of the dataset
    dset_id_t dataset_id;
    /// Beam number (e.g. which of the tracking beams is in this stream)
    uint32_t beam_number;
    /// Right ascension of the beam
    float ra;
    /// Declination of the beam
    float dec;
    /// Scaling factor applied to the beam ( typically: raw_beam/(scaling + .5) )
    uint32_t scaling;
};

void to_json(nlohmann::json& j, const BeamMetadata& m);
void from_json(const nlohmann::json& j, BeamMetadata& m);

#endif // BEAMMETADATA_HPP
