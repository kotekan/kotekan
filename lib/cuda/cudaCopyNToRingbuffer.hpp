// Copyright (c) 2025 Kotekan Project
#pragma once

#include "Config.hpp"              // for Config
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "driver_types.h"          // for cudaEvent_t
#include "ringbuffer.hpp"          // for RingBuffer

#include <stddef.h> // for size_t
#include <string>   // for string, basic_string
#include <vector>   // for vector

/**
 * @class cudaCopyNToRingbuffer
 * @brief Merges frames from multiple input buffers into a single GPU ringbuffer.
 *
 * This is _very_ spesific to the CHIME (N=2048 + ICEBoard) layout. The core idea is
 * we merge frames with an layout of (F, T_sample, E) from N different input buffers
 * into a single ringbuffer with layout (T_frame, F, T_sample, E)
 * Where each input frame has F=1, and each output frame has F=16.
 *
 * @conf in_bufs            List of input buffer names
 * @conf gpu_mem_output     Name of the GPU memory region to use for the output
 * @conf signal_buf         Name of the ringbuffer to manage the GPU memory
 */
class cudaCopyNToRingbuffer : public cudaCommand {
public:
    cudaCopyNToRingbuffer(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                          int instance_num);
    virtual ~cudaCopyNToRingbuffer();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    /// The array of input buffers
    std::vector<Buffer*> in_buffers;

    /// The size of all combined input buffer frames
    size_t total_input_size;

    /// The name of the gpu output memory region
    std::string _gpu_mem_output;

    /// The current location in the ring
    size_t output_cursor;

    /// The ring buffer managing the GPU memory region
    RingBuffer* signal_buffer;

    /// Indicates if this is the first time the command is being executed
    bool first_time;
};
