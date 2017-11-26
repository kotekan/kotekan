#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <string>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include<arpa/inet.h>
#include<sys/socket.h>
#include <netinet/in.h>
#include <cmath>

using std::string;

#include "buffer.h"
#include "clear_frb_out.hpp"
#include "Config.hpp"
#include "util.h"
#include "errors.h"
#include "chimeMetadata.h"
#include "fpga_header_functions.h"

clear_frb_out::clear_frb_out(Config& config_, 
const string& unique_name, bufferContainer &buffer_container) :
KotekanProcess(config_, unique_name, buffer_container,
std::bind(&clear_frb_out::main_thread, this))
{
  
  frb_buf = get_buffer("frb_out_buf");
  register_consumer(frb_buf, unique_name.c_str());
  apply_config(0); 
}

clear_frb_out::~clear_frb_out()
{
  
}


void clear_frb_out::apply_config(uint64_t fpga_seq) 
{
  
}



void clear_frb_out::main_thread() 
{
  int frame_id = 0;
  uint8_t *packet_buffer; 
  long count=0;
  long double temp = 0.5583761708352805;
  packet_buffer = wait_for_full_frame(frb_buf, unique_name.c_str(), frame_id);

  mark_frame_empty(frb_buf, unique_name.c_str(), frame_id);
  frame_id = ( frame_id + 1 ) % frb_buf->num_frames;
  std::chrono::system_clock::time_point start = std::chrono::high_resolution_clock::now();
  while(!stop_thread)
  {
    count++;
    //INFO("Waiting for the buffer \n\n\n\n\n\n\n\n\n\n\n");
    packet_buffer = wait_for_full_frame(frb_buf, unique_name.c_str(), frame_id);
  
    mark_frame_empty(frb_buf, unique_name.c_str(), frame_id);
    frame_id = ( frame_id + 1 ) % frb_buf->num_frames;
    
    long nsec = std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::high_resolution_clock::now() - start).count();
    INFO("Data rate: %Lf %Lf",count*8*8794112.0/(long double)nsec, temp);
  }
  return;
}

