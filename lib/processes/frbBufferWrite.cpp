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
	memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size], frb_header, sizeof(struct FRBHeader));
  
	int static_header_size = 24;
	//copy in the dynamic header
	
  memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size], 
	       frb_header_beam_ids, sizeof(uint16_t)*_nbeams);
	
  memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size + 2*_nbeams], 
	       frb_header_coarse_freq_ids, sizeof(uint16_t)*_nfreq_coarse);
	
  memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size + 2*_nbeams + 2*_nfreq_coarse], 
	       frb_header_scale, sizeof(float)*_nbeams*_nfreq_coarse);
	
  memcpy(&out_buf[j*8*_udp_packet_size + i*_udp_packet_size + static_header_size + 2*_nbeams + 2*_nfreq_coarse + 4*_nbeams*_nfreq_coarse], 
	       frb_header_offset, sizeof(float)*_nbeams*_nfreq_coarse);

  
      }
    }
}

void frbBufferWrite::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

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
      
}


void frbBufferWrite::parse_host_name()
{
  int rack=0,node=0,nos=0;
  std::stringstream temp_ip;

  gethostname(my_host_name, sizeof(my_host_name));

  if(my_host_name[0] != 'c' && my_host_name[3] != 'g')
  {
    INFO("Not a valid name \n");
    exit(0);
  } 
    
    
  if(my_host_name[1] == 'n') 
  {
    nos =0;
    my_node_id = 0;
  }
  else if(my_host_name[1] == 's') 
  {
    nos =100;
    my_node_id  = 128;
  }
  else 
  {
    INFO("Not a valid name \n");
    exit(0);
  }
          
  switch(my_host_name[2])
  {
    case '0': rack=0; break;
    case '1': rack=1; break;
    case '2': rack=2; break;
    case '3': rack=3; break;
    case '4': rack=4; break;
    case '5': rack=5; break;
    case '6': rack=6; break;
    case '7': rack=7; break;
    case '8': rack=8; break;
    case '9': rack=9; break;
    case 'A': rack=10; break;
    case 'B': rack=11; break;
    case 'C': rack=12; break;
    case 'D': rack=13; break;
    default: INFO("Not a valid name \n"); exit(0);
  }
  
  switch(my_host_name[4])
  {
    case '0': node=0; break;
    case '1': node=1; break;
    case '2': node=2; break;
    case '3': node=3; break;
    case '4': node=4; break;
    case '5': node=5; break;
    case '6': node=6; break;
    case '7': node=7; break;
    case '8': node=8; break;
    case '9': node=9; break;
    default: INFO("Not a valid name \n"); exit(0);

  }

  temp_ip<<"10.1."<<nos+rack<<"."<<node;
  my_ip_address = temp_ip.str();
  my_node_id += node;
  
}

void frbBufferWrite::main_thread() {

    parse_host_name();
  
    struct FRBHeader frb_header;
    frb_header.protocol_version = 1;     
    frb_header.data_nbytes =  _udp_packet_size - _udp_header_size;
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

    while(!stop_thread) {
         // Fill the first output buffer headers
	      fpga_seq_num = count*16*_fpga_counts_per_sample * 8;;
	      
        out_frame = wait_for_empty_frame(frb_buf, unique_name.c_str(), frame_id);  
        //INFO("Frame Size %d ",frb_buf->frame_size);    
        
        
        fill_headers((unsigned char*)out_frame,
			   &frb_header,
			   fpga_seq_num,
			   num_L1_streams,
			   (uint16_t*)frb_header_coarse_freq_ids,
			   (float*)frb_header_scale,
			   (float*)frb_header_offset);
        
        mark_frame_full(frb_buf, unique_name.c_str(), frame_id);
        frame_id = ( frame_id + 1 ) % frb_buf->num_frames;
        
                        // Get a new output buffer
        

        }
}
