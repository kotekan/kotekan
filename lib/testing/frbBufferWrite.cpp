#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <string>
#include <unistd.h>
using std::string;

// TODO Where do these live?
# define likely(x)      __builtin_expect(!!(x), 1)
# define unlikely(x)    __builtin_expect(!!(x), 0)

#include "buffer.h"
#include "frbBufferWrite.hpp"
#include "Config.hpp"
#include "util.h"
#include "errors.h"
#include "frb_functions.h" 
#include "chimeMetadata.h"
#include "fpga_header_functions.h"
#include <chrono>
#include <random>
#include "tx_utils.hpp"

REGISTER_KOTEKAN_PROCESS(frbBufferWrite);


frbBufferWrite::frbBufferWrite(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container) :
        KotekanProcess(config_, unique_name, buffer_container,
                       std::bind(&frbBufferWrite::main_thread, this)){

    apply_config(0);

    frb_buf = get_buffer("frb_out_buf");
    register_producer(frb_buf, unique_name.c_str());

    //Dynamic header
    frb_header_beam_ids = new uint16_t[_nbeams];
    frb_header_coarse_freq_ids = new uint16_t[_nfreq_coarse];
    frb_header_scale = new float[_nbeams * _nfreq_coarse];
    frb_header_offset = new float[_nbeams * _nfreq_coarse];
    my_host_name = (char*) malloc(sizeof(char)*100);
}

frbBufferWrite::~frbBufferWrite() {
    free(in_buf);

    free(frb_header_beam_ids);
    free(frb_header_coarse_freq_ids);
    free(frb_header_scale);
    free(frb_header_offset);

}

int frbBufferWrite::random_generator(float *intensity,int size)
{
  std::random_device rd;
  std::mt19937 rng(rd());
  std::normal_distribution<float> dist;
  dist(rng);

  //std::random_device rd;
  //unsigned int seed = rd();
  //std::ranlux48_base rando(seed);

  //float mean = 40;
  //float stddev = 4;
  //float r0 = mean - 2.*stddev;
  //float scale = 4.*stddev / (rando.max() - rando.min());

  for (int i = 0; i < size; i++)
    //intensity[i] = r0 + scale * (float)rando();
    intensity[i] = 100+dist(rng);
  return 0;
}

template <class TYPE>
void copy_array(unsigned char *buf, TYPE* data, int num_elements)
{
  TYPE *packet = reinterpret_cast<TYPE*>(buf);
  for(int i=0;i<num_elements;i++)
   {
      packet[i] = data[i];
   }
}


void frbBufferWrite::fill_headers(unsigned char * out_buf,
                  struct FRBHeader * frb_header,
                  const uint64_t fpga_seq_num,
		  const uint16_t num_L1_streams, //256
		  uint16_t * frb_header_coarse_freq_ids,
		  float * frb_header_scale,
		  float * frb_header_offset){  
    // Populate the headers
    //assert(sizeof(struct FRBHeader) == _udp_header_size);
    
    for (int j=0;j<num_L1_streams; ++j) { //256 streams
      for (int k=0;k<_nbeams;++k) { //the 4 beams are supposed to be consecutive
	frb_header_beam_ids[k] = j*_nbeams+k;
      }
      for (int i = 0; i < 8; ++i) {  //8 frames in a stream
	frb_header->fpga_count = fpga_seq_num + 16*_fpga_counts_per_sample * i;
	memcpy(&out_buf[j*8*_udp_frb_packet_size + i*_udp_frb_packet_size], frb_header, sizeof(struct FRBHeader));
  
	int static_header_size = 24;
	//copy in the dynamic header
  /*	
  memcpy(&out_buf[j*8*_udp_frb_packet_size + i*_udp_frb_packet_size + static_header_size], 
	       frb_header_beam_ids, 8);
         	
  memcpy(&out_buf[j*8*_udp_frb_packet_size + i*_udp_frb_packet_size + static_header_size + 8], 
	       frb_header_coarse_freq_ids, 8);
  */
   copy_array(&out_buf[j*8*_udp_frb_packet_size + i*_udp_frb_packet_size + static_header_size],
                   frb_header_beam_ids, _nbeams);
   copy_array(&out_buf[j*8*_udp_frb_packet_size + i*_udp_frb_packet_size + static_header_size + 2*_nbeams],
                   frb_header_coarse_freq_ids, _nfreq_coarse);
   copy_array(&out_buf[j*8*_udp_frb_packet_size + i*_udp_frb_packet_size + static_header_size + 2*_nbeams + 2*_nfreq_coarse],
                   frb_header_scale, _nbeams*_nfreq_coarse);
   copy_array(&out_buf[j*8*_udp_frb_packet_size + i*_udp_frb_packet_size + static_header_size + 2*_nbeams + 2*_nfreq_coarse + 4*_nbeams*_nfreq_coarse],
                   frb_header_offset, _nbeams*_nfreq_coarse);


  
      }
      

    }
}

void frbBufferWrite::apply_config(uint64_t fpga_seq) {
    //if (!config.update_needed(fpga_seq))
    //    return;

    _num_gpus = config.get_int(unique_name, "num_gpus");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _nfreq_coarse = config.get_int(unique_name, "num_gpus"); //4
    _downsample_time = config.get_int(unique_name, "downsample_time");
    _factor_upchan = config.get_int(unique_name, "factor_upchan");
    _factor_upchan_out = config.get_int(unique_name, "factor_upchan_out"); 
    _nbeams = config.get_int(unique_name, "num_beams");
    _timesamples_per_frb_packet = config.get_int(unique_name, "timesamples_per_frb_packet");
    _udp_frb_packet_size = config.get_int(unique_name, "udp_frb_packet_size");
    _udp_frb_header_size = config.get_int(unique_name, "udp_frb_header_size");
    _freq_array = config.get_int_array(unique_name, "freq_array");

    _fpga_counts_per_sample = _downsample_time * _factor_upchan;
      
}


void frbBufferWrite::main_thread() {
    
    struct timespec t0;
    t0.tv_sec = 0;
    t0.tv_nsec = 0; /*  nanoseconds */
    int rack,node,nos;
    parse_host_name(rack, node, nos, my_node_id);
      
    struct FRBHeader frb_header;
    frb_header.protocol_version = 1;     
    frb_header.data_nbytes =  _udp_frb_packet_size - _udp_frb_header_size;
    frb_header.fpga_counts_per_sample =  _fpga_counts_per_sample;
    frb_header.fpga_count = 0 ;  //to be updated in fill_header
    frb_header.nbeams = _nbeams;  //4
    frb_header.nfreq_coarse = _nfreq_coarse; //4
    frb_header.nupfreq = _factor_upchan_out;
    frb_header.ntsamp = _timesamples_per_frb_packet;

    for (int ii=0;ii<_nbeams;++ii){
      frb_header_beam_ids[ii] = 7; //To be overwritten in fill_header
    }
    for (int ii=0;ii<_nfreq_coarse;++ii){
      frb_header_coarse_freq_ids[ii] = (uint16_t) (4*my_node_id+ii);//_freq_array[ii] 
    }
    for (int ii =0; ii<_nbeams * _nfreq_coarse;++ii){
      frb_header_scale[ii] = 1.; 
      frb_header_offset[ii] = 0;
    }

    //int frame = 0;
    //int in_frame_location = 0; //goes from 0 to 16
    uint64_t fpga_seq_num = 0;

    int num_L1_streams = 1024/_nbeams;
    uint64_t count = 0;
    
    int frame_id = 0;
    unsigned char* out_frame;
    uint16_t *packet;
    
    while(!stop_thread) {
        INFO("started thread %d",frame_id);  
       // Fill the first output buffer headers
	      if(count==0)
              {
               clock_gettime(CLOCK_REALTIME, &t0);
               long double time = 100*(t0.tv_sec+(t0.tv_nsec/1e9));
               time = 390625*(time-151190506200.0)/100.0;
               fpga_seq_num = (long)time;
               fpga_seq_num -= fpga_seq_num%(384*16);
              }
              else
              fpga_seq_num += 16*_fpga_counts_per_sample * 8;
        
        INFO("waiting for buffer %d",frame_id);
	      
        out_frame = wait_for_empty_frame(frb_buf, unique_name.c_str(), frame_id);  
        
        //random_generator(reinterpret_cast<float*>(out_frame),2*256*_udp_frb_packet_size); 
        INFO("Frame Size %d ",frb_buf->frame_size);    
        
        INFO("got the buffer %d",frame_id);
         
    
        fill_headers((unsigned char*)out_frame,
			   &frb_header,
			   fpga_seq_num,
			   num_L1_streams,
			   (uint16_t*)frb_header_coarse_freq_ids,
			   (float*)frb_header_scale,
			   (float*)frb_header_offset);
        packet = reinterpret_cast<uint16_t*>(out_frame);        
        INFO("marking full buffer %d",packet[(_udp_frb_packet_size/2)*254*8+12]);
        mark_frame_full(frb_buf, unique_name.c_str(), frame_id);
        
        INFO("marked the buffer %d time: %ld",frame_id,t0.tv_sec); 
        frame_id = ( frame_id + 1 ) % frb_buf->num_frames;
        
        // Get a new output buffer
        count++; 

        }
}
