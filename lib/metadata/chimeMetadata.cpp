#include "chimeMetadata.hpp"
#include "factory.hpp"
#include "metadata.hpp"
#include "BasebandMetadata.hpp"
#include "HFBMetadata.hpp"

REGISTER_TYPE_WITH_FACTORY(metadataObject, chimeMetadata);

// and since we don't have anywhere else to do this...
REGISTER_TYPE_WITH_FACTORY(metadataObject, BasebandMetadata);
REGISTER_TYPE_WITH_FACTORY(metadataObject, HFBMetadata);

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
