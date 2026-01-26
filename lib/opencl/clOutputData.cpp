#include "clOutputData.hpp"
#include "BeamMetadata.hpp"   // for BeamMetadata
#include "BasebandMetadata.hpp"   // for BeamMetadata

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clOutputData);

clOutputData::clOutputData(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, clDeviceInterface& device,
                           int instance_num) :
    clCommand(config, unique_name, host_buffers, device, instance_num, no_cl_command_state, "clOutputData", "") {

    _gpu_memory = config.get_default<std::string>(unique_name, "gpu_memory", "output");
    std::cout<<"clOutputData constructed"<<"\n";
    std::cout<<unique_name<<"\n";
    // Get the output buffers and register as a producer if we are the first instance.
    out_bufs = get_buffer_array("out_bufs", instance_num == 0, true);
    // Get the input buffers and register as a consumer if we are the first instance.
    in_bufs = get_buffer_array("in_bufs", instance_num == 0, false);

    command_type = gpuCommandType::COPY_OUT;
}

clOutputData::~clOutputData() {}

int clOutputData::wait_on_precondition() {
    // Wait for there to be data in the input (network) buffer.
    Buffer* in_buf = in_bufs[gpu_frame_id % in_bufs.size()];
    int in_frame_index = gpu_frame_id / in_bufs.size() % in_buf->num_frames;
    uint8_t* in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_index);
    if (in_frame == nullptr)
        return -1;
    
    Buffer* out_buf = out_bufs[gpu_frame_id % out_bufs.size()];
    int out_frame_index = gpu_frame_id / out_bufs.size() % out_buf->num_frames;
    uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_index);
    if (out_frame == nullptr)
        return -1;

    return 0;
}


cl_event clOutputData::execute(cl_event pre_event) {
    pre_execute();

    Buffer * out_buf = out_bufs[gpu_frame_id % out_bufs.size()];
    int out_frame_index = gpu_frame_id / out_bufs.size() % out_buf->num_frames;

    void* host_output_frame = (void*)out_buf->frames[out_frame_index];

    assert(pre_event != nullptr);

    // Transfer data from the device to host
    device.async_copy_gpu_to_host(_gpu_memory, gpu_frame_id, host_output_frame,
                                  out_buf->aligned_frame_size, pre_event, post_event);

    return post_event;
}

void clOutputData::finalize_frame() {
    clCommand::finalize_frame();

    Buffer * in_buf = in_bufs[gpu_frame_id % in_bufs.size()];
    int in_frame_index = gpu_frame_id / in_bufs.size() % in_buf->num_frames;

    Buffer * out_buf = out_bufs[gpu_frame_id % out_bufs.size()];
    int out_frame_index = gpu_frame_id / out_bufs.size() % out_buf->num_frames;

    //pass_metadata(in_buf, in_frame_index, out_buf, out_frame_index);
    // Copy over the relevant metadata
    allocate_new_metadata_object(out_buf, out_frame_index);

    BasebandMetadata* in_metadata = (BasebandMetadata*)get_metadata(in_buf, in_frame_index);
    BeamMetadata* out_metadata = (BeamMetadata*)get_metadata(out_buf, out_frame_index);

    out_metadata->time0_ctime = in_metadata->time0_ctime;
    out_metadata->time0_ctime_offset = in_metadata->time0_ctime_offset;
    out_metadata->event_id = in_metadata->event_id;
    out_metadata->freq_id = in_metadata->freq_id;
    // Copy the base dataset ID from the GPU data.
    // @TODO we will likely want to add a unique dataset ID state
    // for each beam for systems that like to track things with the dataset ID
    // however at the moment all that data should be contained in this metadata
    // plus the root dataset ID.
    
    //out_metadata->ra = in_metadata->beam_coord.ra[_extract_beam];
    //out_metadata->dec = in_metadata->beam_coord.dec[_extract_beam];
    
    mark_frame_empty(in_buf, unique_name.c_str(), in_frame_index);
    mark_frame_full(out_buf, unique_name.c_str(), out_frame_index);
}
