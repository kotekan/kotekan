
#include "hsaCorrelatorKernel.hpp"
#include "hsaBase.h"

// What does this mean?
#define N_INTG 16384

hsaCorrelatorKernel::hsaCorrelatorKernel(const string& kernel_name, const string& kernel_file_name,
                            gpuHSADeviceInterface& device, Config& config,
                            bufferContainer& host_buffers) :
    gpuHSAcommand(kernel_name, kernel_file_name, device, config, host_buffers) {

    apply_config(0);

    // Allocate and copy the block map
    host_block_map = (uint32_t *)hsa_host_malloc(block_map_len);
    int block_id = 0;
    for (int y = 0; block_id < _num_blocks; y++) {
        for (int x = y; x < _num_elements/32; x++) {
            host_block_map[2*block_id+0] = x;
            host_block_map[2*block_id+1] = y;
            block_id++;
        }
    }

    // Copy it to the GPU
    void * device_block_map = device.get_gpu_memory("block_map", block_map_len);
    device.sync_copy_host_to_gpu(device_block_map, host_block_map, block_map_len);

    // Create the extra kernel args object.
    host_kernel_args = (corr_kernel_config_t *)hsa_host_malloc(sizeof(corr_kernel_config_t));
    host_kernel_args->n_elem = _num_elements;
    host_kernel_args->n_intg = N_INTG;
    host_kernel_args->n_iter = _samples_per_data_set;
    host_kernel_args->n_blk = _num_blocks;

    void * device_kernel_args = device.get_gpu_memory("corr_kernel_config", sizeof(corr_kernel_config_t));
    device.sync_copy_host_to_gpu(device_kernel_args, host_kernel_args, sizeof(corr_kernel_config_t));

}

void hsaCorrelatorKernel::apply_config(const uint64_t& fpga_seq) {
    gpuHSAcommand::apply_config(fpga_seq);

    _num_elements = config.get_int("/processing/num_elements");
    _num_local_freq = config.get_int("/processing/num_local_freq");
    _samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    _num_blocks = config.get_int("/gpu/num_blocks");
    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    presum_len = _num_elements * _num_local_freq * 2 * sizeof (int32_t);
    // I don't really like this way of getting to correlator output size (AR)
    corr_frame_len = _num_blocks * 32 * 32 * 2 * sizeof(int32_t);
    block_map_len = _num_blocks * 2 * sizeof(uint32_t);
}

hsaCorrelatorKernel::~hsaCorrelatorKernel() {
    hsa_host_free((void *)host_block_map);
    hsa_host_free((void *)host_kernel_args);
}

hsa_signal_t hsaCorrelatorKernel::execute(int gpu_frame_id,
                        const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t {
        void *input_buffer;
        void *presum_buffer;
        void *corr_buffer;
        void *blk_map;
        void *config;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.presum_buffer = device.get_gpu_memory_array("presum", gpu_frame_id, presum_len);
    args.corr_buffer = device.get_gpu_memory_array("corr", gpu_frame_id, corr_frame_len);
    args.blk_map = device.get_gpu_memory("block_map", block_map_len);
    args.config = device.get_gpu_memory("corr_kernel_config", sizeof(corr_kernel_config_t));
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    INFO("correlatorKernel: gpu[%d][%d], input_buffer: %p, presum_buffer: %p, corr_buffer: %p, blk_map: %p, config: %p, sizeof(args) = %d, kernels_args[%d] = %p",
            device.get_gpu_id(), gpu_frame_id, args.input_buffer, args.presum_buffer, args.corr_buffer, args.blk_map, args.config,
            (int)sizeof(args), gpu_frame_id, kernel_args[gpu_frame_id]);

    INFO("correlatorKernel: gpu[%d][%d], wgx %d, wgy %d, wgz %d, gsx %d, gsy %d, gsz %d",
            device.get_gpu_id(), gpu_frame_id, 16, 4, 1, 16, 4*_samples_per_data_set/N_INTG, _num_blocks);

    hsa_status_t hsa_status = hsa_signal_create(1, 0, NULL, &signals[gpu_frame_id]);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Obtain the current queue write index.
    uint64_t index = hsa_queue_load_write_index_acquire(device.get_queue());

    hsa_kernel_dispatch_packet_t* dispatch_packet = (hsa_kernel_dispatch_packet_t*)device.get_queue()->base_address +
                                                            (index % device.get_queue()->size);
    INFO("hsaCorrelatorKernel got write index: %" PRIu64 ", packet_address: %p, post_signal: %lu", index, dispatch_packet, signals[gpu_frame_id].handle);

    dispatch_packet->setup  |= 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    dispatch_packet->workgroup_size_x = (uint16_t)16;
    dispatch_packet->workgroup_size_y = (uint16_t)4;
    dispatch_packet->workgroup_size_z = (uint16_t)1;
    dispatch_packet->grid_size_x = (uint32_t)16;
    dispatch_packet->grid_size_y = (uint32_t)4*_samples_per_data_set/N_INTG;
    dispatch_packet->grid_size_z = (uint32_t)_num_blocks;
    dispatch_packet->completion_signal = signals[gpu_frame_id];
    dispatch_packet->kernel_object = kernel_object;
    dispatch_packet->kernarg_address = (void*) kernel_args[gpu_frame_id];
    dispatch_packet->private_segment_size = 0;
    dispatch_packet->group_segment_size = 0;
    dispatch_packet-> header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    hsa_queue_add_write_index_acquire(device.get_queue(), 1);
    hsa_signal_store_relaxed(device.get_queue()->doorbell_signal, index);

    return signals[gpu_frame_id];
}
