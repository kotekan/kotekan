#include "hsaPreseedKernel.hpp"

#include <unistd.h>

// What is this?
#define N_PRESUM 1024

hsaPreseedKernel::hsaPreseedKernel(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers, const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name){
    apply_config(0);
}

void hsaPreseedKernel::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    presum_len = _num_elements * _num_local_freq * 2 * sizeof (int32_t);
}

hsaPreseedKernel::~hsaPreseedKernel() {

}

void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
    __atomic_store_n(packet, header | (rest << 16),   __ATOMIC_RELEASE);
}

uint16_t kernel_dispatch_setup() {
    return 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
}

uint16_t header(hsa_packet_type_t type) {
    uint16_t header = type << HSA_PACKET_HEADER_TYPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    return header;
}

hsa_signal_t hsaPreseedKernel::execute(int frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t {
        void *input_buffer;
        void *mystery;
        int constant;
        void *presum_buffer;
    } args;

    

    memset(&args, 0, sizeof(args));

    args.input_buffer = device.get_gpu_memory_array("input", frame_id, input_frame_len);
    args.mystery = NULL;
    args.constant = _num_elements/4;//global_x size
    args.presum_buffer = device.get_gpu_memory_array("presum", frame_id, presum_len);
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[frame_id], &args, sizeof(args));

    // NEW

    uint64_t packet_id = hsa_queue_add_write_index_screlease(device.get_queue(), 1);

    // Should never hit this condition, but lets be safe.
    while (packet_id - hsa_queue_load_read_index_scacquire(device.get_queue()) >= device.get_queue()->size);

    INFO("hsaPreseedKernel gpu[%d]: input_buffer: %p, presum_buffer: %p, args_ptr: %p, size: %lu, packet_id: %d, queue size: %d",
            device.get_gpu_id(), args.input_buffer, args.presum_buffer, kernel_args[frame_id], sizeof(args), packet_id, device.get_queue()->size);

    hsa_kernel_dispatch_packet_t* packet = (hsa_kernel_dispatch_packet_t*) device.get_queue()->base_address + (packet_id % device.get_queue()->size);

    // Set basic packet details.
    memset(((uint8_t*) packet) + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);

    packet->workgroup_size_x = (uint16_t)64;
    packet->workgroup_size_y = (uint16_t)1;
    packet->workgroup_size_z = (uint16_t)1;
    packet->grid_size_x = (uint32_t)_num_elements/4;
    packet->grid_size_y = (uint32_t)_samples_per_data_set/N_PRESUM;
    packet->grid_size_z = (uint32_t)1;

    packet->kernel_object = this->kernel_object;

    packet->kernarg_address = (void*) kernel_args[frame_id];

    hsa_signal_create(1, 0, NULL, &packet->completion_signal);
    signals[frame_id] = packet->completion_signal;

    packet_store_release((uint32_t*) packet, header(HSA_PACKET_TYPE_KERNEL_DISPATCH), kernel_dispatch_setup());
    hsa_signal_store_screlease(device.get_queue()->doorbell_signal, packet_id);

    INFO("Waiting for gpu[%d]: input_buffer: %p, presum_buffer: %p, args_ptr: %p, size: %lu, packet_id: %d, queue size: %d",
            device.get_gpu_id(), args.input_buffer, args.presum_buffer, kernel_args[frame_id], sizeof(args), packet_id, device.get_queue()->size);

    //usleep(10);
    //while (hsa_signal_wait_scacquire(packet->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0);

    //hsa_signal_destroy(packet->completion_signal);

    return signals[frame_id];

    // OLD
/*
    hsa_status_t hsa_status = hsa_signal_create(1, 0, NULL, &signals[frame_id]);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Obtain the current queue write index and add one to it.
    // TODO Is it safe to increment this before filling the packet??
    uint64_t index = hsa_queue_load_write_index_acquire(device.get_queue());
    hsa_kernel_dispatch_packet_t* dispatch_packet = (hsa_kernel_dispatch_packet_t*)device.get_queue()->base_address +
                                                            (index % device.get_queue()->size);
    INFO("hsaPreseedKernel gpu[%d]: got write index: %" PRIu64 ", packet_address %p, post_signal %lu", device.get_gpu_id(), index, dispatch_packet, signals[frame_id].handle);
    dispatch_packet->setup  |= 2 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    dispatch_packet->workgroup_size_x = (uint16_t)64;
    dispatch_packet->workgroup_size_y = (uint16_t)1;
    dispatch_packet->workgroup_size_z = (uint16_t)1;
    dispatch_packet->grid_size_x = (uint32_t)_num_elements/4;
    dispatch_packet->grid_size_y = (uint32_t)_samples_per_data_set/N_PRESUM;
    dispatch_packet->grid_size_z = (uint32_t)1;
    dispatch_packet->completion_signal = signals[frame_id];
    dispatch_packet->kernel_object = this->kernel_object;
    dispatch_packet->kernarg_address = (void*) kernel_args[frame_id];
    dispatch_packet->private_segment_size = 0;
    dispatch_packet->group_segment_size = 0;
    dispatch_packet-> header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    hsa_queue_add_write_index_acquire(device.get_queue(), 1);
    hsa_signal_store_relaxed(device.get_queue()->doorbell_signal, index);

    return signals[frame_id]; */
}
