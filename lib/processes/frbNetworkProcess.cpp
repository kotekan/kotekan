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
  my_host_name = (char*) malloc(sizeof(char)*100); 
}

frbNetworkProcess::~frbNetworkProcess()
{
  free(my_host_name);
}


void frbNetworkProcess::apply_config(uint64_t fpga_seq) 
{
  udp_packet_size = config.get_int(unique_name, "udp_packet_size");
  udp_port_number = config.get_int(unique_name, "udp_port_number");
  number_of_nodes = config.get_int(unique_name, "number_of_nodes");
  packets_per_stream = config.get_int(unique_name, "packets_per_stream");
}

void frbNetworkProcess::parse_host_name()
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
  my_node_id += rack*10+node;
}


void frbNetworkProcess::main_thread() 
{
  //parsing the host name
  parse_host_name(); 
  
  
  int frame_id = 0;
  uint8_t * packet_buffer = NULL;
  
  std::vector<std::string> link_ip = config.get_string_array(unique_name, "L1_node_ips");
  int number_of_l1_links = link_ip.size();
  INFO("number_of_l1_links: %d",number_of_l1_links);  
    

  

  int sock_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
 
  if (sock_fd < 0)
  {
    std::cout << "network thread: socket() failed: " <<
    strerror(errno) << std::endl;
    exit(0);
  }
  
  struct sockaddr_in server_address[number_of_l1_links], myaddr;

  
  std::memset((char *)&myaddr, 0, sizeof(myaddr));

  myaddr.sin_family = AF_INET;
  inet_pton(AF_INET, my_ip_address.c_str(), &myaddr.sin_addr);
  
  // Comment the above line and uncomment the line below to run it on McGill nodes
  //inet_pton(AF_INET, "22.22.0.2", &myaddr.sin_addr);

  myaddr.sin_port = htons(udp_port_number);

  // Binding port to the socket
  if (bind(sock_fd, (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
       INFO("port binding failed");
       exit(0);
    }
  
  
  
  for(int i=0;i<number_of_l1_links;i++)
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
  
  struct timespec t0,t1,temp;
  t0.tv_sec = 0;
  t0.tv_nsec = 0; /*  nanoseconds */
  
  unsigned long time_interval = 125995520; //time per buffer frame in ns

   
  long count=0;

  while(!stop_thread)
  {
    
    
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned long abs_ns = t0.tv_sec*1e9 + t0.tv_nsec;
    unsigned long reminder = (abs_ns%time_interval);
    unsigned long wait_ns = time_interval-reminder + my_node_id*240*(256/number_of_l1_links); // analytically it must be 240.3173828125
 
 
    t0.tv_nsec += wait_ns;
    if(t0.tv_nsec>=1000000000)
    {
      t0.tv_sec += 1;
      t0.tv_nsec -= 1000000000;
    }
    
    // Checking with the NTP server    
    if(count==0)
    {
      temp.tv_sec = t0.tv_sec;
      temp.tv_nsec = t0.tv_nsec;
    }
    else
    {
      temp.tv_nsec += 125995520;
      if(temp.tv_nsec>=1000000000)
      {
        temp.tv_sec += 1;
        temp.tv_nsec -= 1000000000;
      }
      
      long sec = (long)temp.tv_sec - (long)t0.tv_sec;
      long nsec = (long)temp.tv_nsec - (long)t0.tv_nsec;
      nsec = sec*1e9+nsec;

      if(abs(nsec)<50000000) temp = t0; 
      else INFO("Not locked with NTP \n");

    }
    
    //INFO("Host name %s ip: %s node: %d",my_host_name,my_ip_address.c_str(),my_node_id);


    t1.tv_sec = t0.tv_sec;
    t1.tv_nsec = t0.tv_nsec;
   
    packet_buffer = wait_for_full_frame(frb_buf, unique_name.c_str(), frame_id);
    if(packet_buffer==NULL)
      break;
    
    INFO("Host name %s ip: %s node: %d",my_host_name,my_ip_address.c_str(),my_node_id);
    

    for(int frame=0; frame<packets_per_stream; frame++)
    {
      for(int stream=0; stream<number_of_l1_links; stream++)
      {
        int e_stream = my_node_id + stream;
        if(e_stream>number_of_nodes-1) e_stream -= number_of_nodes;
        
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t1, NULL);

         

         sendto(sock_fd, &packet_buffer[(e_stream*packets_per_stream+frame)*udp_packet_size], 
                   udp_packet_size , 0 , (struct sockaddr *) &server_address[stream] , sizeof(server_address[stream])); 
         
         
         long wait_per_packet = (long)(61440*(256/number_of_l1_links)); 
         
         //61521.25 is the theoritical seperation of packets in ns 
         // I have used 61440 for convinence and also hope this will take care for
         // any clock glitches.

         t1.tv_nsec = t1.tv_nsec+wait_per_packet;
         if(t1.tv_nsec>=1000000000)
         {
           t1.tv_sec = t1.tv_sec + 1;
           t1.tv_nsec = t1.tv_nsec -1000000000;
         }
         
         
      }
    }
    
    
    mark_frame_empty(frb_buf, unique_name.c_str(), frame_id);
    frame_id = ( frame_id + 1 ) % frb_buf->num_frames;
    count++;
    
  }
  return;
}


