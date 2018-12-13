#include "hsaOutputData.hpp"
#include "visUtil.hpp"
#include "gpsTime.h"

REGISTER_HSA_COMMAND(hsaOutputData);

hsaOutputData::hsaOutputData(Config& config, const string &unique_name,
                            bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name,
                                 host_buffers, device, "hsaOutputData",""){
    command_type = gpuCommandType::COPY_OUT;

    network_buffer = host_buffers.get_buffer("network_buf");
    register_consumer(network_buffer, unique_name.c_str());

    output_buffer = host_buffers.get_buffer("output_buf");
    lost_samples_buf = host_buffers.get_buffer("lost_samples_buf");
    // Each of the command objects in a subframe set outputs is only doing
    // one of every _num_sub_frame frames.  So we only register one consumer
    // and one producer name which in this case is ok to be static.
    static_unique_name = "hsa_output_static_" + std::to_string(device.get_gpu_id());
    if (_sub_frame_index == 0) {
        register_producer(output_buffer, static_unique_name.c_str());
        register_consumer(lost_samples_buf, static_unique_name.c_str());
    }

    network_buffer_id = 0;

    output_buffer_id = _sub_frame_index;
    output_buffer_precondition_id = _sub_frame_index;
    output_buffer_excute_id = _sub_frame_index;

    lost_samples_buf_id = 0;
}

hsaOutputData::~hsaOutputData() {
}

int hsaOutputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // We want to make sure we have some space to put our results.
    uint8_t * frame = wait_for_empty_frame(output_buffer,
                          unique_name.c_str(), output_buffer_precondition_id);
    if (frame == NULL) return -1;
    output_buffer_precondition_id = (output_buffer_precondition_id + _num_sub_frames) %
                                    output_buffer->num_frames;
    return 0;
}

hsa_signal_t hsaOutputData::execute(int gpu_frame_id,
                                    hsa_signal_t precede_signal) {

    void * gpu_output_ptr =
        device.get_gpu_memory_array("corr_" + std::to_string(_sub_frame_index),
                                    gpu_frame_id, output_buffer->frame_size);

    void * host_output_ptr = (void *)output_buffer->frames[output_buffer_excute_id];

    device.async_copy_gpu_to_host(host_output_ptr,
            gpu_output_ptr, output_buffer->frame_size,
            precede_signal, signals[gpu_frame_id]);

    output_buffer_excute_id = (output_buffer_excute_id + _num_sub_frames) %
                              output_buffer->num_frames;

    return signals[gpu_frame_id];
}


void hsaOutputData::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    allocate_new_metadata_object(output_buffer, output_buffer_id);

    // We make a new copy of the metadata since there are now
    // _num_sub_frames output frames for each input frame.
    copy_metadata(network_buffer, network_buffer_id,
                  output_buffer, output_buffer_id);

    // Adjust the time stamps

    // Subframe updated fpga_seq
    uint64_t fpga_seq_num = get_fpga_seq_num(network_buffer, network_buffer_id);
    fpga_seq_num += _sub_frame_index * _sub_frame_samples;
    set_fpga_seq_num(output_buffer, output_buffer_id, fpga_seq_num);

    // Subframe updated GPS time
    struct timespec new_gps_time = compute_gps_time(fpga_seq_num);
    set_gps_time(output_buffer, output_buffer_id, new_gps_time);

    // Subframe updated system_time
    struct timeval sys_time = get_first_packet_recv_time(network_buffer, network_buffer_id);
    double sys_time_d = tv_to_double(sys_time);
    sys_time_d += _sub_frame_index * _sub_frame_samples * 2.56 / 1000 / 1000;
    sys_time = double_to_tv(sys_time_d);
    set_first_packet_recv_time(output_buffer, output_buffer_id, sys_time);

    // Add up the number of lost samples (from packet loss/packet errors)
    uint8_t * frame = lost_samples_buf->frames[lost_samples_buf_id];

    uint32_t num_sum_frame_lost_samples = 0;
    for (uint32_t i = _sub_frame_samples * _sub_frame_index;
         i < (_sub_frame_samples * (_sub_frame_index + 1));
         ++i) {
        if (frame[i] == 1) {
            num_sum_frame_lost_samples++;
        }
    }
    zero_lost_samples(output_buffer, output_buffer_id);
    atomic_add_lost_timesamples(output_buffer, output_buffer_id, num_sum_frame_lost_samples);

    // Mark the input buffer as "empty" so that it can be reused.
    mark_frame_empty(network_buffer, unique_name.c_str(), network_buffer_id);

    // Mark the output buffer as full, so it can be processed.
    mark_frame_full(output_buffer, static_unique_name.c_str(), output_buffer_id);

    if ((_sub_frame_index + 1) == _num_sub_frames) {
        mark_frame_empty(lost_samples_buf, static_unique_name.c_str(), lost_samples_buf_id);
    }

    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;
    output_buffer_id = (output_buffer_id + _num_sub_frames) % output_buffer->num_frames;
    lost_samples_buf_id = (lost_samples_buf_id + 1) % lost_samples_buf->num_frames;
}
