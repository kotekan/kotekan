#include "chimeMetadata.hpp"

#include "BasebandMetadata.hpp"
#include "HFBMetadata.hpp"
#include "factory.hpp"
#include "metadata.hpp"

REGISTER_TYPE_WITH_FACTORY(metadataObject, chimeMetadata);

// chimeMetadata::chimeMetadata() :
//     fpga_seq_num(0), lost_timesamples(0), rfi_flagged_samples(0), rfi_zeroed(0),
//     rfi_num_bad_inputs(0), stream_ID(-1) {}
// 
// chimeMetadata::chimeMetadata(const chimeMetadata& other) {
//     // boil them plates
//     fpga_seq_num = other.fpga_seq_num;
//     first_packet_recv_time = other.first_packet_recv_time;
//     gps_time = other.gps_time;
//     // the compiler couldn't *possibly* be trusted to auto-generate this...
//     lost_timesamples.store(other.lost_timesamples.load());
//     rfi_flagged_samples.store(other.rfi_flagged_samples.load());
//     // /so hard
//     rfi_zeroed = other.rfi_zeroed;
//     rfi_num_bad_inputs = other.rfi_num_bad_inputs;
//     stream_ID = other.stream_ID;
//     dataset_id = other.dataset_id;
//     beam_coord = other.beam_coord;
// }

chimeMetadata& chimeMetadata::operator=(const chimeMetadata& other) {
    // boil them plates
    fpga_seq_num = other.fpga_seq_num;
    first_packet_recv_time = other.first_packet_recv_time;
    gps_time = other.gps_time;
    // the compiler couldn't *possibly* be trusted to auto-generate this...
    lost_timesamples.store(other.lost_timesamples.load());
    rfi_flagged_samples.store(other.rfi_flagged_samples.load());
    // /so hard
    rfi_zeroed = other.rfi_zeroed;
    rfi_num_bad_inputs = other.rfi_num_bad_inputs;
    stream_ID = other.stream_ID;
    dataset_id = other.dataset_id;
    beam_coord = other.beam_coord;
    return *this;
}

// Ugh this is annoying, define the serialization format via plain-old-data struct...
struct chimeMetadataFormat {
    int64_t fpga_seq_num;
    struct timeval first_packet_recv_time;
    struct timespec gps_time;
    int32_t lost_timesamples;
    int32_t rfi_flagged_samples;
    uint32_t rfi_zeroed;
    uint32_t rfi_num_bad_inputs;
    uint16_t stream_ID;
    dset_id_t dataset_id;
    struct beamCoord beam_coord;
};

size_t chimeMetadata::get_serialized_size() {
    return sizeof(chimeMetadataFormat);
}

size_t chimeMetadata::set_from_bytes(const char* bytes, size_t length) {
    size_t sz = get_serialized_size();
    assert(length >= sz);
    // chimeMetadataFormat fmt;
    // memcpy(&fmt, bytes, sz);
    const chimeMetadataFormat* fmt = reinterpret_cast<const chimeMetadataFormat*>(bytes);
    fpga_seq_num = fmt->fpga_seq_num;
    first_packet_recv_time = fmt->first_packet_recv_time;
    gps_time = fmt->gps_time;
    lost_timesamples.store(fmt->lost_timesamples);
    rfi_flagged_samples.store(fmt->rfi_flagged_samples);
    rfi_zeroed = fmt->rfi_zeroed;
    rfi_num_bad_inputs = fmt->rfi_num_bad_inputs;
    stream_ID = fmt->stream_ID;
    dataset_id = fmt->dataset_id;
    beam_coord = fmt->beam_coord;
    return sz;
}

size_t chimeMetadata::serialize(char* bytes) {
    size_t sz = get_serialized_size();
    chimeMetadataFormat* fmt = reinterpret_cast<chimeMetadataFormat*>(bytes);
    fmt->fpga_seq_num = fpga_seq_num;
    fmt->first_packet_recv_time = first_packet_recv_time;
    fmt->gps_time = gps_time;
    fmt->lost_timesamples = lost_timesamples;
    fmt->rfi_flagged_samples = rfi_flagged_samples;
    fmt->rfi_zeroed = rfi_zeroed;
    fmt->rfi_num_bad_inputs = rfi_num_bad_inputs;
    fmt->stream_ID = stream_ID;
    fmt->dataset_id = dataset_id;
    fmt->beam_coord = beam_coord;

    return sz;
}
