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
#include "pulsarNetworkProcess.hpp"
#include "Config.hpp"
#include "util.h"
#include "errors.h"
#include "chimeMetadata.h"
#include "fpga_header_functions.h"
#include "tx_utils.hpp"

REGISTER_KOTEKAN_PROCESS(pulsarNetworkProcess);

pulsarNetworkProcess::pulsarNetworkProcess(Config& config_, 
const string& unique_name, bufferContainer &buffer_container) :
KotekanProcess(config_, unique_name, buffer_container,
std::bind(&pulsarNetworkProcess::main_thread, this))
{
  
  in_buf = get_buffer("pulsar_out_buf");
  register_consumer(in_buf, unique_name.c_str());
  apply_config(0);
  my_host_name = (char*) malloc(sizeof(char)*100);
  CHECK_MEM(my_host_name);
}

pulsarNetworkProcess::~pulsarNetworkProcess()
{
  free(my_host_name);
}


void pulsarNetworkProcess::apply_config(uint64_t fpga_seq) 
{
  udp_pulsar_packet_size = config.get_int(unique_name, "udp_pulsar_packet_size");
  udp_pulsar_port_number = config.get_int(unique_name, "udp_pulsar_port_number");
  number_of_nodes = config.get_int(unique_name, "number_of_nodes");
  //packets_per_stream = config.get_int(unique_name, "packets_per_stream");
  number_of_subnets = config.get_int(unique_name, "number_of_subnets");
}

void pulsarNetworkProcess::main_thread() 
{
  //parsing the host name
   
  int rack,node,nos,my_node_id;
  std::stringstream temp_ip[number_of_subnets];


  //parsing the host name

  parse_host_name(rack, node, nos, my_node_id);
  for(int i=0;i<number_of_subnets;i++)
  { 
    temp_ip[i]<<"10."<<i+15<<"."<<nos+rack<<".1"<<node;
    my_ip_address[i] = temp_ip[i].str();
    INFO("%s ",my_ip_address[i].c_str());
  }
 
  
  int frame_id = 0;
  uint8_t * packet_buffer = NULL;
  
  std::vector<std::string> link_ip = config.get_string_array(unique_name, "pulsar_node_ips");
  int number_of_pulsar_links = link_ip.size();
  INFO("number_of_pulsar_links: %d",number_of_pulsar_links);  
    

  int *sock_fd = new int[number_of_subnets];

  for(int i=0;i<number_of_subnets;i++) 
  {
    sock_fd[i] = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
 
    if (sock_fd[i] < 0)
    {
      std::cout << "network thread: socket() failed: " <<
      strerror(errno) << std::endl;
      exit(0);
    }
  }

  struct sockaddr_in server_address[number_of_pulsar_links], myaddr[number_of_subnets];

  
  for(int i=0;i<number_of_subnets;i++) 
  {
    std::memset((char *)&myaddr[i], 0, sizeof(myaddr[i]));

    myaddr[i].sin_family = AF_INET;
    inet_pton(AF_INET, my_ip_address[i].c_str(), &myaddr[i].sin_addr);
  
    myaddr[i].sin_port = htons(udp_pulsar_port_number);

    // Binding port to the socket
    if (bind(sock_fd[i], (struct sockaddr *)&myaddr[i], sizeof(myaddr[i])) < 0) {
         INFO("port binding failed");
         exit(0);
      }
  }
  
  
  for(int i=0;i<number_of_pulsar_links;i++)
  {
    memset(&server_address[i], 0, sizeof(server_address[i]));
    server_address[i].sin_family = AF_INET;
    inet_pton(AF_INET, link_ip[i].c_str(), &server_address[i].sin_addr);
    server_address[i].sin_port = htons(udp_pulsar_port_number);
  }
  
  int n = 256* 1024 * 1024;
  for(int i=0;i<number_of_subnets;i++)
  {  
    if (setsockopt(sock_fd[i], SOL_SOCKET, SO_SNDBUF,(void *) &n, sizeof(n))  < 0)
    {
      std::cout << "network thread: setsockopt() failed: " <<  strerror(errno) << std::endl;
      exit(0);
    }
  }

  struct timespec t0,t1,temp;
  t0.tv_sec = 0;
  t0.tv_nsec = 0; /*  nanoseconds */
  
  unsigned long time_interval = 128000000; //time per buffer frame in ns

   
  long count=0;

  int my_sequence_id = (int)(my_node_id/128) + 2*((my_node_id%128)/8) + 32*(my_node_id%8);
  
  
  packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
  mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
  frame_id = ( frame_id + 1 ) % in_buf->num_frames;
    
  /* 
  clock_gettime(CLOCK_REALTIME, &t0);
  
  t0.tv_nsec += 8*time_interval;
  if(t0.tv_nsec>=1000000000)
  {
    t0.tv_sec += 1;
    t0.tv_nsec -= 1000000000;
  }
  clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &t0, NULL);
  */
  
  while(!stop_thread)
  {
    
    long lock_miss=0; 
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned long abs_ns = t0.tv_sec*1e9 + t0.tv_nsec;
    unsigned long reminder = (abs_ns%time_interval);
    unsigned long wait_ns = time_interval-reminder + my_sequence_id*600; // analytically it must be 781.25
 
    
    add_nsec(t0,wait_ns); 
        
    // Checking with the NTP server    
    if(count==0)
    {
      temp.tv_sec = t0.tv_sec;
      temp.tv_nsec = t0.tv_nsec;
    }
    else
    {
      add_nsec(temp,time_interval);
            
      long sec = (long)temp.tv_sec - (long)t0.tv_sec;
      long nsec = (long)temp.tv_nsec - (long)t0.tv_nsec;
      nsec = sec*1e9+nsec;

      if (abs(nsec)%time_interval==0 && abs(nsec)!=0)
      {
        INFO("Buffers are too slow %d \n\n\n\n\n\n\n\n",abs(nsec));
        add_nsec(t0,-1*nsec);
        temp=t0;
        lock_miss++;
      }
      else if(abs(nsec)!=0)
      {
        //INFO("Not locked with NTP %d\n",abs(nsec));
        //exit(0);
        lock_miss++;
      }
    }
    
    //INFO("Host name %s ip: %s node: %d",my_host_name,my_ip_address.c_str(),my_node_id);


    t1.tv_sec = t0.tv_sec;
    t1.tv_nsec = t0.tv_nsec;
   
    packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
    if(packet_buffer==NULL)
      break;
    //uint16_t *packet = reinterpret_cast<uint16_t*>(packet_buffer);
    INFO("Host name %s ip: %s node: %d sequence_id: %d lock_miss: %ld",my_host_name,my_ip_address[1].c_str(),my_node_id,my_sequence_id,lock_miss);
    

    for(int frame=0; frame<80; frame++)
    {
      for(int beam=0; beam<10; beam++)
      {
        int e_beam = my_sequence_id + beam;
        e_beam =  e_beam%10;
        
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t1, NULL);

        if(e_beam<number_of_pulsar_links)
        {
          int i = e_beam%number_of_subnets;
          sendto(sock_fd[i], &packet_buffer[(e_beam)*80*udp_pulsar_packet_size + frame*udp_pulsar_packet_size], 
                   udp_pulsar_packet_size , 0 , (struct sockaddr *) &server_address[e_beam] , sizeof(server_address[e_beam])); 
             
        }
         
        long wait_per_packet = (long)(153600); 
        
        //61521.25 is the theoritical seperation of packets in ns 
        // I have used 61440 for convinence and also hope this will take care for
        // any clock glitches.
        add_nsec(t1,wait_per_packet);
      }
    }
    
    
    mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
    frame_id = ( frame_id + 1 ) % in_buf->num_frames;
    count++;
    
  }
  return;
}

