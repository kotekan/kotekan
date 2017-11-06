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
#include "frbNetworkProcess.hpp"
#include "Config.hpp"
#include "util.h"
#include "errors.h"
#include "chimeMetadata.h"
#include "fpga_header_functions.h"

frbNetworkProcess::frbNetworkProcess(Config& config_, 
const string& unique_name, bufferContainer &buffer_container) :
KotekanProcess(config_, unique_name, buffer_container,
std::bind(&frbNetworkProcess::main_thread, this))
{
  
  frb_buf = get_buffer("frb_out_buf");
  register_consumer(frb_buf, unique_name.c_str());
  apply_config(0); 
}

frbNetworkProcess::~frbNetworkProcess()
{
  
}


void frbNetworkProcess::apply_config(uint64_t fpga_seq) 
{
  udp_packet_size = config.get_int(unique_name, "udp_packet_size");
  udp_port_number = config.get_int(unique_name, "udp_port_number");
  file_name = config.get_string(unique_name, "file_name");
  my_ip_address = config.get_string(unique_name, "my_ip_address");
  number_of_nodes = config.get_int(unique_name, "number_of_nodes");
  packets_per_stream = config.get_int(unique_name, "packets_per_stream");
  my_node_id = config.get_int(unique_name, "my_node_id");

}



void frbNetworkProcess::main_thread() 
{
  int frame_id = 0;
  uint8_t * packet_buffer = NULL;
  
  string link_ip[number_of_nodes];
  std::ifstream fp(file_name.c_str());
  if(!fp.is_open()) exit(0);  

  for(int i=0;i<number_of_nodes;i++)
  {  
    std::getline(fp,link_ip[i]);
  }
  
  int sock_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
 
  if (sock_fd < 0)
  {
    std::cout << "network thread: socket() failed: " <<
    strerror(errno) << std::endl;
    exit(0);
  }
  
  struct sockaddr_in server_address[number_of_nodes], myaddr;

  
  std::memset((char *)&myaddr, 0, sizeof(myaddr));

  myaddr.sin_family = AF_INET;
  inet_pton(AF_INET, my_ip_address.c_str(), &myaddr.sin_addr);

  myaddr.sin_port = htons(udp_port_number);

  // Binding port to the socket
  if (bind(sock_fd, (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
       INFO("port binding failed");
       exit(0);
    }
  
  
  
  
  
  
  for(int i=0;i<number_of_nodes;i++)
  {
    memset(&server_address[i], 0, sizeof(server_address[i]));
    server_address[i].sin_family = AF_INET;
    inet_pton(AF_INET, link_ip[i].c_str(), &server_address[i].sin_addr);
    server_address[i].sin_port = htons(udp_port_number);
  }
  
  int n = 256* 1024 * 1024;
  if (setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF,(void *) &n, sizeof(n))  < 0)
  {
    std::cout << "network thread: setsockopt() failed: " <<  strerror(errno) << std::endl;
    exit(0);
  }
  
  struct timespec t0,t1;
  t0.tv_sec = 0;
  t0.tv_nsec = 0; /*  nanoseconds */
  
  unsigned long time_interval = 125995520; //time per buffer frame in ns

   
  long count=0;
  while(!stop_thread)
  {
    
    
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned long abs_ns = t0.tv_sec*1e9 + t0.tv_nsec;
    unsigned long reminder = (abs_ns%time_interval);
    unsigned long wait_ns = time_interval-reminder + my_node_id*240; // analytically it must be 240.3173828125
 
 
    //INFO("sec: %ld nsec: %ld",t0.tv_sec,t0.tv_nsec);
 
    t0.tv_nsec += wait_ns;
    if(t0.tv_nsec>=1000000000)
    {
      t0.tv_sec += 1;
      t0.tv_nsec -= 1000000000;
    }
 
    t1.tv_sec = t0.tv_sec;
    t1.tv_nsec = t0.tv_nsec;
    
     
    
    /*
    t0.tv_nsec += time_interval;
    if(t0.tv_nsec>=1000000000)
    {
      t0.tv_sec += 1;
      t0.tv_nsec -= 1000000000;
    }
   
    t1.tv_sec = t0.tv_sec;
    t1.tv_nsec = t0.tv_nsec;
    */

    packet_buffer = wait_for_full_frame(frb_buf, unique_name.c_str(), frame_id);
    if(packet_buffer==NULL)
      break;
    
    

    for(int frame=0; frame<packets_per_stream; frame++)
    {
      for(int stream=0; stream<number_of_nodes; stream++)
      {
        int e_stream = my_node_id + stream;
        if(e_stream>number_of_nodes-1) e_stream -= number_of_nodes;
        
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t1, NULL);

         

         sendto(sock_fd, &packet_buffer[(e_stream*packets_per_stream+frame)*udp_packet_size], 
                   udp_packet_size , 0 , (struct sockaddr *) &server_address[stream] , sizeof(server_address[stream])); 
         
         
         long wait_per_packet = (long)(61440*(frame*number_of_nodes+stream)); 
         
         //61521.25 is the theoritical seperation of packets in ns 
         // I have used 61440 for convinence and also hope this will take care for
         // any clock glitches.

         t1.tv_nsec += t0.tv_nsec+wait_per_packet;
         if(t1.tv_nsec>=1000000000)
         {
           t1.tv_sec = t0.tv_sec + 1;
           t1.tv_nsec = t0.tv_nsec -1000000000;
         }
         

      }
    }

    mark_frame_empty(frb_buf, unique_name.c_str(), frame_id);
    frame_id = ( frame_id + 1 ) % frb_buf->num_frames;
    count++;
  }
  return;
}


