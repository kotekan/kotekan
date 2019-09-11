#include "hsaAsyncCopyGain.hpp"

#include "utils/util.h"
#include "fmt.hpp"
#include <utils/visUtil.hpp>

#include <random>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaAsyncCopyGain);

hsaAsyncCopyGain::hsaAsyncCopyGain(Config& config, const string& unique_name, bufferContainer& host_buffers,
                           hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaAsyncCopyGain", "") {
    command_type = gpuCommandType::COPY_IN;

    gain_len = 2 * 2048 * sizeof(float);

    gain_buf = host_buffers.get_buffer("gain_buf");
    register_consumer(gain_buf, unique_name.c_str());
    gain_buf_precondition_id = 0;
}


hsaAsyncCopyGain::~hsaAsyncCopyGain() {}

hsa_signal_t hsaAsyncCopyGain::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

  //Check for new gains
  auto timeout = double_to_ts(0);
  //Not too sure if the ID should be gain_buf_precondition_id or gpu_frame_id
  int status = wait_for_full_frame_timeout(gain_buf, unique_name.c_str(),  gain_buf_precondition_id, timeout);
  INFO("[CHECK] status of gain_buf={:d} ==================", status);
  //Not sure how to include the continue/break statement
  /*if (status == 1)
    continue; // Timed out, try next buffer
  if (status == -1)
    break; // Got shutdown signal
  */
    
  //async copy in
  void* device_gain = device.get_gpu_memory_array("beamform_gain", gpu_frame_id, gain_len);
  void* host_gain = (void*)gain_buf->frames[gain_buf_precondition_id];
  device.async_copy_host_to_gpu(device_gain, host_gain, gain_len, precede_signal, signals[gpu_frame_id]);
  return signals[gpu_frame_id];
}

void hsaAsyncCopyGain::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    mark_frame_empty(gain_buf, unique_name.c_str(), gain_buf_precondition_id);
    gain_buf_precondition_id = (gain_buf_precondition_id + 1) % gain_buf->num_frames;
}
