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
    gain_buf_finalize_id = 0;
    gain_buf_id = 0;
    frame_to_fill = 0;

    first_pass = true;
}


hsaAsyncCopyGain::~hsaAsyncCopyGain() {}

int hsaAsyncCopyGain::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    
    //Check for new gains
    INFO("[AsyncCopyPrecon] waiting for gain_buf_id={:d} to be full; gpu_frame_id={:d}", gain_buf_id, gpu_frame_id);
    if (first_pass) {
      uint8_t* frame = wait_for_full_frame(gain_buf, unique_name.c_str(),  gain_buf_id);
      first_pass = false;
      frame_to_fill = gain_buf->num_frames;      
      if (frame == NULL)
	return -1;
    } else {
      //Check for new gains only if filled all gpu frames
      if (frame_to_fill == 0) {
	auto timeout = double_to_ts(0);
	//Do I really need gin_buf_id here, it seems to be always 0
	int status = wait_for_full_frame_timeout(gain_buf, unique_name.c_str(), gain_buf_id, timeout);
	INFO("[AsyncCopyProcon] status of gain_buf_id[{:d}]={:d} ==(0=ready 1=not)================", gain_buf_id, status);
	if (status == 0)
	  frame_to_fill = gain_buf->num_frames;
	if (status == -1)
	  return -1;
      }
    }
    INFO("[AsyncCopyPrecon] leaving with gain_buf_id={:d} frame_to_fill={:d}", gain_buf_id, frame_to_fill);
    return 0;
}


hsa_signal_t hsaAsyncCopyGain::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

  //async copy in
  //Set a counter so it doesnt do anything if there is no new data (don't check for new gain unless its zero)
  if (frame_to_fill > 0) {
    INFO("[AsyncCopyExe] going to async copy gain_buf_id={:d} gpu_frame_id={:d}", gain_buf_id, gpu_frame_id);
    void* device_gain = device.get_gpu_memory_array("beamform_gain", gpu_frame_id, gain_len);
    void* host_gain = (void*)gain_buf->frames[gain_buf_id];
    device.async_copy_host_to_gpu(device_gain, host_gain, gain_len, precede_signal, signals[gpu_frame_id]);
    frame_to_fill = frame_to_fill -1;
    INFO("[AsyncCopyEx] frame left to be filled={:d} gain_buf_id={:d}", frame_to_fill, gain_buf_id);
  }
  return signals[gpu_frame_id];
      
}

void hsaAsyncCopyGain::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    INFO("[AsyncCopyFin] finalize_frame for frame_id={:d} mark gain_bu_id={:d} empty", frame_id, gain_buf_id);
    mark_frame_empty(gain_buf, unique_name.c_str(), gain_buf_id);
    gain_buf_id = (gain_buf_id + 1) % gain_buf->num_frames;
}
