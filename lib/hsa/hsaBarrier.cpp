#include "hsaBarrier.hpp"

#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::BARRIER
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for uint32_t, uint8_t, uint64_t
#include <string.h> // for memset

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBarrier);

hsaBarrier::hsaBarrier(Config& config, const string& unique_name, bufferContainer& host_buffers,
                       hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaBarrier", "") {
    command_type = gpuCommandType::BARRIER;
}


hsaBarrier::~hsaBarrier() {}

hsa_signal_t hsaBarrier::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Get the queue index
    uint64_t index = hsa_queue_add_write_index_scacquire(device.get_queue(), 1);

    // Make sure the queue isn't full
    // Should never hit this condition, but lets be safe.
    // See the HSA docs for details.
    while (index - hsa_queue_load_read_index_relaxed(device.get_queue())
           >= device.get_queue()->size)
        ;

    // Get the packet address.
    hsa_barrier_and_packet_t* barrier_and_packet =
        (hsa_barrier_and_packet_t*)device.get_queue()->base_address
        + (index % device.get_queue()->size);
    // INFO("hsaBarrier got write index: {:d}, packet_address: {:p}, precede_signal: {:d}",
    // index, barrier_and_packet, precede_signal.handle);

    // Set the packet details, including the preceded signal to wait on.
    //    barrier_and_packet->header = HSA_PACKET_TYPE_INVALID;
    packet_store_release((uint32_t*)barrier_and_packet, header(HSA_PACKET_TYPE_INVALID), 0);
    memset(((uint8_t*)barrier_and_packet) + 4, 0, sizeof(*barrier_and_packet) - 4);
    barrier_and_packet->dep_signal[0] = precede_signal;
    barrier_and_packet->completion_signal = signals[gpu_frame_id];

    while (0 < hsa_signal_cas_screlease(signals[gpu_frame_id], 0, 1))
        ;


    // Set packet header after packet body.
    packet_store_release((uint32_t*)barrier_and_packet, header(HSA_PACKET_TYPE_BARRIER_AND), 0);

    // Signal doorbell after packet header.
    hsa_signal_store_screlease(device.get_queue()->doorbell_signal, index);

    // Does not generate a completion signal at this time, although it could.
    return signals[gpu_frame_id];
}

// If a completion signal was created this would clean it.
void hsaBarrier::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
}
