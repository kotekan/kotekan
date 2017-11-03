#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <string>

using std::string;

// TODO Where do these live?
# define likely(x)      __builtin_expect(!!(x), 1)
# define unlikely(x)    __builtin_expect(!!(x), 0)

#include "buffer.h"
#include "frbBufferWrite.hpp"
#include "Config.hpp"
#include "util.h"
#include "errors.h"
#include "chimeMetadata.h"
#include "fpga_header_functions.h"

frbBufferWrite::frbBufferWrite(Config& config_, 
const string& unique_name, bufferContainer &buffer_container) :
KotekanProcess(config_, unique_name, buffer_container,
std::bind(&frbBufferWrite::main_thread, this))
{
  
  frb_buf = get_buffer("frb_out_buf");
  register_producer(frb_buf, unique_name.c_str());
  
}

frbBufferWrite::~frbBufferWrite()
{
  free(frb_buf);
}


void frbBufferWrite::apply_config(uint64_t fpga_seq) 
{
  /*
  _num_gpus = config.get_int(unique_name, "num_gpus");
  _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
  _nfreq_coarse = config.get_int(unique_name, "num_gpus"); //4
  _downsample_time = config.get_int(unique_name, "downsample_time");
  _factor_upchan = config.get_int(unique_name, "factor_upchan");
  _factor_upchan_out = config.get_int(unique_name, "factor_upchan_out");
  _nbeams = config.get_int(unique_name, "num_beams");
  _timesamples_per_frb_packet = config.get_int(unique_name, "timesamples_per_frb_packet");
  _udp_packet_size = config.get_int(unique_name, "udp_packet_size");
  _udp_header_size = config.get_int(unique_name, "udp_header_size");
  _freq_array = config.get_int_array(unique_name, "freq_array");

  _fpga_counts_per_sample = _downsample_time * _factor_upchan;
  */
}


void frbBufferWrite::main_thread() 
{
  int frame_id = 0;
  uint8_t * frame = NULL;
  int frame_size = 0;
  int udp_packet_size = 4264/2;
  int staic_header = 28;

  uint16_t *buffer = (uint16_t*) malloc(frb_buf->frame_size);
  
  for(uint16_t i=0; i<256*8; i++)
  {
    buffer[i*udp_packet_size+14] = i*4;
  }
  
  while(!stop_thread)
  {
    frame = wait_for_empty_frame(frb_buf, unique_name.c_str(), frame_id);
    if(frame==NULL)
      break;
    memcpy(frame,buffer,frb_buf->frame_size);
    mark_frame_full(frb_buf, unique_name.c_str(), frame_id);
    frame_id = ( frame_id + 1 ) % frb_buf->num_frames;
    INFO("Written %d frame_size %d",frame_id,frb_buf->frame_size);
  }
  return;
}


