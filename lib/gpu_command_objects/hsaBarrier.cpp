#include "hsaBarrier.hpp"

hsaBarrier::~hsaBarrier() {

}

// Not needed.
void hsaBarrier::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
}

hsa_signal_t hsaBarrier::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    // Get the current write index.
    uint64_t index = hsa_queue_add_write_index_acquire(device.get_queue(), 1);
    hsa_barrier_and_packet_t* barrier_and_packet =
            (hsa_barrier_and_packet_t*)device.get_queue()->base_address +
                                            (index % device.get_queue()->size);

    // Set the packet details, including the preceded signal to wait on.
    memset(((uint8_t*) barrier_and_packet) + 4, 0, sizeof(*barrier_and_packet) - 4);
    barrier_and_packet->dep_signal[0] = precede_signal;
    barrier_and_packet->header = HSA_PACKET_TYPE_BARRIER_AND;
    hsa_signal_store_screlease(device.get_queue()->doorbell_signal, index);

    // Does not generate a completion signal at this time, although it could.
    hsa_signal_t empty_signal;
    empty_signal.handle = 0;
    return empty_signal;
}

// If a completion signal was created this would clean it.
void hsaBarrier::finalize_frame(int frame_id) {
    gpuHSAcommand::finalize_frame(frame_id);
}

