#include "hsaOutputData.hpp"

#include "Telescope.hpp"
#include "buffer.hpp"             // for Buffer, mark_frame_empty, register_consumer, wait_for_...
#include "bufferContainer.hpp"    // for bufferContainer
#include "chimeMetadata.hpp"      // for atomic_add_lost_timesamples, get_first_packet_recv_time
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::COPY_OUT
#include "hsaCommand.hpp"         // for REGISTER_HSA_COMMAND, _factory_aliashsaCommand, hsaCom...
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "visUtil.hpp"            // for double_to_tv, tv_to_double

#include "fmt.hpp" // for format, fmt

#include <sys/time.h> // for timeval
#include <time.h>     // for timespec

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaOutputData);

hsaOutputData::hsaOutputData(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name, host_buffers, device, "hsaOutputData", "") {
    command_type = gpuCommandType::COPY_OUT;

    network_buffer = host_buffers.get_buffer("network_buf");
    output_buffer = host_buffers.get_buffer("output_buf");
    lost_samples_buf = host_buffers.get_buffer("lost_samples_buf");
    // Each of the command objects in a subframe set outputs is only doing
    // one of every _num_sub_frame frames.  So we only register one consumer
    // and one producer name which in this case is ok to be static.
    static_unique_name = fmt::format(fmt("hsa_output_static_{:d}"), device.get_gpu_id());

    if (_sub_frame_index == 0) {
        network_buffer->register_consumer(static_unique_name);
        output_buffer->register_producer(static_unique_name);
        lost_samples_buf->register_consumer(static_unique_name);
    }

    network_buffer_id = 0;
    network_buffer_precondition_id = 0;

    output_buffer_id = _sub_frame_index;
    output_buffer_precondition_id = _sub_frame_index;
    output_buffer_excute_id = _sub_frame_index;

    lost_samples_buf_id = 0;
    lost_samples_buf_precondition_id = 0;
}

hsaOutputData::~hsaOutputData() {}

int hsaOutputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    // We want to make sure we have some space to put our results.
    uint8_t* frame =
        output_buffer->wait_for_empty_frame(static_unique_name, output_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    output_buffer_precondition_id =
        (output_buffer_precondition_id + _num_sub_frames) % output_buffer->num_frames;
    if (_sub_frame_index == 0) {
        frame =
            network_buffer->wait_for_full_frame(static_unique_name, network_buffer_precondition_id);
        if (frame == nullptr)
            return -1;
        frame = lost_samples_buf->wait_for_full_frame(static_unique_name,
                                                      lost_samples_buf_precondition_id);
        if (frame == nullptr)
            return -1;
        network_buffer_precondition_id =
            (network_buffer_precondition_id + 1) % network_buffer->num_frames;
        lost_samples_buf_precondition_id =
            (lost_samples_buf_precondition_id + 1) % lost_samples_buf->num_frames;
    }

    return 0;
}

hsa_signal_t hsaOutputData::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    void* gpu_output_ptr =
        device.get_gpu_memory_array(fmt::format(fmt("corr_{:d}"), _sub_frame_index), gpu_frame_id,
                                    _gpu_buffer_depth, output_buffer->frame_size);

    void* host_output_ptr = (void*)output_buffer->frames[output_buffer_excute_id];

    device.async_copy_gpu_to_host(host_output_ptr, gpu_output_ptr, output_buffer->frame_size,
                                  precede_signal, signals[gpu_frame_id]);

    output_buffer_excute_id =
        (output_buffer_excute_id + _num_sub_frames) % output_buffer->num_frames;

    return signals[gpu_frame_id];
}


void hsaOutputData::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);

    auto& tel = Telescope::instance();

    output_buffer->allocate_new_metadata_object(output_buffer_id);

    // We make a new copy of the metadata since there are now
    // _num_sub_frames output frames for each input frame.
    network_buffer->copy_metadata(network_buffer_id, output_buffer, output_buffer_id);

    // Adjust the time stamps

    // Subframe updated fpga_seq
    uint64_t fpga_seq_num = get_fpga_seq_num(network_buffer, network_buffer_id);
    fpga_seq_num += _sub_frame_index * _sub_frame_samples;
    set_fpga_seq_num(output_buffer, output_buffer_id, fpga_seq_num);

    // Subframe updated GPS time
    auto new_gps_time = tel.to_time(fpga_seq_num);
    set_gps_time(output_buffer, output_buffer_id, new_gps_time);

    // Subframe updated system_time
    struct timeval sys_time = get_first_packet_recv_time(network_buffer, network_buffer_id);
    double sys_time_d = tv_to_double(sys_time);
    sys_time_d += _sub_frame_index * _sub_frame_samples * tel.seq_length_nsec() * 1e-9;
    sys_time = double_to_tv(sys_time_d);
    set_first_packet_recv_time(output_buffer, output_buffer_id, sys_time);

    // Add up the number of lost samples (from packet loss/packet errors)
    uint8_t* frame = lost_samples_buf->frames[lost_samples_buf_id];

    uint32_t num_sum_frame_lost_samples = 0;
    for (uint32_t i = _sub_frame_samples * _sub_frame_index;
         i < (_sub_frame_samples * (_sub_frame_index + 1)); ++i) {
        if (frame[i] == 1) {
            num_sum_frame_lost_samples++;
        }
    }
    zero_lost_samples(output_buffer, output_buffer_id);
    atomic_add_lost_timesamples(output_buffer, output_buffer_id, num_sum_frame_lost_samples);

    // Mark the output buffer as full, so it can be processed.
    output_buffer->mark_frame_full(static_unique_name, output_buffer_id);

    if ((_sub_frame_index + 1) == _num_sub_frames) {
        // Mark the input buffer as "empty" so that it can be reused.
        network_buffer->mark_frame_empty(static_unique_name, network_buffer_id);
        lost_samples_buf->mark_frame_empty(static_unique_name, lost_samples_buf_id);
    }

    network_buffer_id = (network_buffer_id + 1) % network_buffer->num_frames;
    output_buffer_id = (output_buffer_id + _num_sub_frames) % output_buffer->num_frames;
    lost_samples_buf_id = (lost_samples_buf_id + 1) % lost_samples_buf->num_frames;
}

std::string hsaOutputData::get_unique_name() const {
    return static_unique_name;
}
