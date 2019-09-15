#include "ReadGain.hpp"

#include "buffer.h"
#include "bufferContainer.hpp"
#include "chimeMetadata.h"
#include "configUpdater.hpp"
#include "errors.h"
#include <utils/visUtil.hpp>
#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(ReadGain);

ReadGain::ReadGain(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
  Stage(config, unique_name, buffer_container, std::bind(&ReadGain::main_thread, this)) {
  //Apply config.
  _num_elements = config.get<uint32_t>(unique_name, "num_elements");
  scaling = config.get_default<float>(unique_name, "frb_scaling", 1.0);
  vector<float> dg = {0.0, 0.0}; // re,im
  default_gains = config.get_default<std::vector<float>>(unique_name, "frb_missing_gains", dg);

  metadata_buf = get_buffer("network_buf");
  register_consumer(metadata_buf, unique_name.c_str());
  metadata_buffer_id = 0;
  metadata_buffer_precondition_id = 0;
  freq_idx = -1;
  freq_MHz = -1;
    
  gain_buf = get_buffer("gain_buf");
  register_producer(gain_buf, unique_name.c_str());
  //gain_buf_precondition_id = 0;
  
  update_gains = true;
  first_pass = true;
  
  using namespace std::placeholders;

  // listen for gain updates
  _gain_dir = config.get_default<std::string>(unique_name, "updatable_config/gain_frb", "");
  if (_gain_dir.length() > 0)
    configUpdater::instance().subscribe(
					config.get<std::string>(unique_name, "updatable_config/gain_frb"),
					std::bind(&ReadGain::update_gains_callback, this, _1));
}

bool ReadGain::update_gains_callback(nlohmann::json& json) {
    {
      std::lock_guard<std::mutex> lock(mux);
      update_gains = true;
    }
    cond_var.notify_all();
    
    try {
        _gain_dir = json.at("frb_gain_dir");
    } catch (std::exception& e) {
        WARN("[FRB] Fail to read gain_dir {:s}", e.what());
        return false;
    }
    INFO("[ReadGain] updated gain with {:s}============update_gains={:d}", _gain_dir, update_gains);
    return true;

}


void ReadGain::main_thread() {

  if (first_pass) {
    first_pass = false;
    uint8_t* frame = wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == NULL)
      goto end_loop;
    stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
    freq_idx = bin_number_chime(&stream_id);
    freq_MHz = freq_from_bin(freq_idx);
    //INFO("[CHECK ReadGain] freq_idx {:d} {:.2f}", freq_idx, freq_MHz);
    metadata_buffer_precondition_id = (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;
  }
  mark_frame_empty(metadata_buf, unique_name.c_str(), metadata_buffer_id);
  metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;
    
  while (!stop_thread) {
    INFO("[ReadGain] update_gains={:d}-(0=false 1=true)-------------", update_gains);
    {
      std::unique_lock<std::mutex> lock(mux);
      while (!update_gains) {
	cond_var.wait(lock);//, [&]{return update_gains = true;});
      }
    }
    INFO("[ReadGain] Going to update gain+++++++++++++++++update_gains={:d}", update_gains);

    //Need to read it in to all gpu frames.
    for (int f = 0; f < gain_buf->num_frames; f++) {
      INFO("[ReadGain] start of main_thread: frame now={:d} total nframe={:d}", f, gain_buf->num_frames);
      float* out_frame = (float*)wait_for_empty_frame(gain_buf, unique_name.c_str(), f);
      if (out_frame == NULL) {
	INFO("[ReadGain] waiting for frame={:d} but it is not empty", f);
	goto end_loop;
      }

      double start_time = current_time();
      FILE* ptr_myfile;
      char filename[256];
      snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin", _gain_dir.c_str(), freq_idx);
      INFO("[ReadGain] Loading gains from {:s}", filename);
      ptr_myfile = fopen(filename, "rb");
      if (ptr_myfile == NULL) {
	WARN("GPU Cannot open gain file {:s}", filename);
	for (int i = 0; i < 2048; i++) {
	  out_frame[i * 2] = default_gains[0] * scaling;
	  out_frame[i * 2 + 1] = default_gains[1] * scaling;
	}
      }
      else {
	if (_num_elements != fread(out_frame, sizeof(float) * 2, _num_elements, ptr_myfile)) {
	  WARN("Gain file ({:s}) wasn't long enough! Something went wrong, breaking...", filename);
	  for (int i = 0; i < 2048; i++) {
	    out_frame[i * 2] = default_gains[0] * scaling;
	    out_frame[i * 2 + 1] = default_gains[1] * scaling;
	  }
	}
	fclose(ptr_myfile);
      }
      mark_frame_full(gain_buf, unique_name.c_str(), f);
      INFO("[ReadGain] maked gain_buf frame {:d} full", f);
      INFO("[ReadGain] Time required to load FRB gains: {:f}", current_time() - start_time);
      INFO("[ReadGain] gain_buf: {:.2f} {:.2f} {:.2f} ", out_frame[0], out_frame[1], out_frame[2]);
    }

    update_gains = false;
  } //end stop thread
 end_loop:;
}
	
