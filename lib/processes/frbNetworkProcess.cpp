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

//Update beam_offset parameter with:
//curl localhost:12048/frb/update_beam_offset -X POST -H 'Content-Type: application/json' -d '{"beam_offset":108}'

REGISTER_KOTEKAN_PROCESS(frbNetworkProcess);

frbNetworkProcess::frbNetworkProcess(Config& config_, 
const string& unique_name, bufferContainer &buffer_container) :
KotekanProcess(config_, unique_name, buffer_container,
std::bind(&frbNetworkProcess::main_thread, this))
{
  in_buf = get_buffer("in_buf");
  register_consumer(in_buf, unique_name.c_str());
  apply_config(0);
  my_host_name = (char*) malloc(sizeof(char)*100); 
  CHECK_MEM(my_host_name);
}

frbNetworkProcess::~frbNetworkProcess()
{
  free(my_host_name);
}

void frbNetworkProcess::update_offset_callback(connectionInstance& conn, json& json_request) {
    DEBUG("Got a callback!");
    std::lock_guard<std::mutex> lock(_packet_frame_lock);
    try {
        int bo = json_request["beam_offset"];
        if (bo > 512) throw std::invalid_argument("beam_offset can be >512!");
        if (bo < 0) throw std::invalid_argument("Negative beam_offset!");
        beam_offset = bo;
    } catch (...) {
        conn.send_error("could not parse/find num_packets parameter", STATUS_BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(STATUS_OK);
}

void frbNetworkProcess::apply_config(uint64_t fpga_seq) 
{
  udp_frb_packet_size = config.get_int_default(unique_name, "udp_frb_packet_size", 4264);
  udp_frb_port_number = config.get_int_default(unique_name, "udp_frb_port_number", 1313);
  number_of_nodes = config.get_int_default(unique_name, "number_of_nodes", 256);
  number_of_subnets = config.get_int_default(unique_name, "number_of_subnets",4);
  packets_per_stream = config.get_int_default(unique_name, "packets_per_stream",8);
  beam_offset = config.get_int_default(unique_name, "beam_offset",0);
  time_interval = config.get_uint64_default(unique_name, "time_interval",125829120);
  column_mode = config.get_bool_default(unique_name, "column_mode", false);
}

void frbNetworkProcess::parse_host_name()
{
  int rack=0,node=0,nos=0;
  std::stringstream temp_ip[number_of_subnets];

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
    //case '7': rack=7; break;
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

  for(int i=0;i<number_of_subnets;i++) 
  {
    temp_ip[i]<<"10."<<i+6<<"."<<nos+rack<<".1"<<node;
    my_ip_address[i] = temp_ip[i].str();
    INFO("%s ",my_ip_address[i].c_str());
  }
  if(rack>7)my_node_id += rack*10+(9-node); //fix for the arrangment of nodes in the racks
  if(rack>7) my_node_id += (rack-1)*10+(9-node);
}


void frbNetworkProcess::main_thread() 
{
  //parsing the host name
  parse_host_name(); 

  using namespace std::placeholders;
  restServer * rest_server = get_rest_server();
  string endpoint = "/frb/update_beam_offset";
  rest_server->register_json_callback(endpoint,
      std::bind(&frbNetworkProcess::update_offset_callback, this, _1, _2));


  int frame_id = 0;
  uint8_t * packet_buffer = NULL;

  std::vector<std::string> link_ip = config.get_string_array(unique_name, "L1_node_ips");
  int number_of_l1_links = link_ip.size();
  INFO("number_of_l1_links: %d",number_of_l1_links);  

  int *sock_fd = new int[number_of_subnets];

  for(int i=0;i<number_of_subnets;i++) 
  {
    sock_fd[i] = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    if (sock_fd[i] < 0)
    {
      ERROR("Network Thread: socket() failed: %s", strerror(errno));
      exit(0);
    }
  }

  struct sockaddr_in server_address[number_of_l1_links], myaddr[number_of_subnets];

  
  for(int i=0;i<number_of_subnets;i++) 
  {
    std::memset((char *)&myaddr[i], 0, sizeof(myaddr[i]));

    myaddr[i].sin_family = AF_INET;
    inet_pton(AF_INET, my_ip_address[i].c_str(), &myaddr[i].sin_addr);
  
    myaddr[i].sin_port = htons(udp_frb_port_number);

    // Binding port to the socket
    if (bind(sock_fd[i], (struct sockaddr *)&myaddr[i], sizeof(myaddr[i])) < 0) {
         ERROR("port binding failed");
         exit(0);
      }
  }
  
  
  for(int i=0;i<number_of_l1_links;i++)
  {
    memset(&server_address[i], 0, sizeof(server_address[i]));
    server_address[i].sin_family = AF_INET;
    inet_pton(AF_INET, link_ip[i].c_str(), &server_address[i].sin_addr);
    server_address[i].sin_port = htons(udp_frb_port_number);
  }
  
  int n = 256* 1024 * 1024;
  for(int i=0;i<number_of_subnets;i++)
  {  
    if (setsockopt(sock_fd[i], SOL_SOCKET, SO_SNDBUF,(void *) &n, sizeof(n))  < 0)
    {
      ERROR("Network Thread: setsockopt() failed: %s ", strerror(errno));
      exit(0);
    }
  }

  struct timespec t0,t1,temp;
  t0.tv_sec = 0;
  t0.tv_nsec = 0; /*  nanoseconds */

  //unsigned long time_interval = 125829120; //time per buffer frame in ns

  long count=0;

  int my_sequence_id = (int)(my_node_id/128) + 2*((my_node_id%128)/8) + 32*(my_node_id%8);

  packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
  mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
  frame_id = ( frame_id + 1 ) % in_buf->num_frames;

  clock_gettime(CLOCK_MONOTONIC, &t0);

  t0.tv_nsec += 2*time_interval;
  if(t0.tv_nsec>=1000000000)
  {
    t0.tv_sec += 1;
    t0.tv_nsec -= 1000000000;
  }
  clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t0, NULL);

  while(!stop_thread)
  {
    long lock_miss=0; 
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned long abs_ns = t0.tv_sec*1e9 + t0.tv_nsec;
    unsigned long reminder = (abs_ns%time_interval);
    unsigned long wait_ns = time_interval-reminder + my_sequence_id*230; // analytically it must be 240.3173828125

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
      temp.tv_nsec += time_interval;
      if(temp.tv_nsec>=1000000000)
      {
        temp.tv_sec += 1;
        temp.tv_nsec -= 1000000000;
      }
      
      long sec = (long)temp.tv_sec - (long)t0.tv_sec;
      long nsec = (long)temp.tv_nsec - (long)t0.tv_nsec;
      nsec = sec*1e9+nsec;

      if (abs(nsec)==time_interval && abs(nsec)!=0)
      {
        WARN("Buffers are too slow %d \n\n\n\n\n\n\n\n",abs(nsec));
        t0.tv_nsec -= nsec;
        if(t0.tv_nsec>=1000000000)
        {
         t0.tv_sec += 1;
         t0.tv_nsec -= 1000000000;
        }
        else if(t0.tv_nsec<0)
        {
          t0.tv_sec -= 1;
          t0.tv_nsec += 1000000000;
        }
        temp=t0;
        lock_miss++;
      }
      else if(abs(nsec)!=0)
      {
        lock_miss++;
        temp=t0;  
      }
    }

    t1.tv_sec = t0.tv_sec;
    t1.tv_nsec = t0.tv_nsec;
   
    packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
    if(packet_buffer==NULL)
      break;
    uint16_t *packet = reinterpret_cast<uint16_t*>(packet_buffer);
    INFO("Host name %s ip: %s node: %d sequence_id: %d beam_id %d lock_miss: %ld",my_host_name,my_ip_address[2].c_str(),my_node_id,my_sequence_id,packet[udp_frb_packet_size*4*253+12],lock_miss);

    std::lock_guard<std::mutex> lock(_packet_frame_lock);
    DEBUG("Beam offset: %i",beam_offset);
    for(int frame=0; frame<packets_per_stream; frame++)
    {
      for(int stream=0; stream<256; stream++)
      {
        int e_stream = my_sequence_id + stream;
        if(e_stream>255) e_stream -= 256;
        
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t1, NULL);
        
         //if(e_stream==(beam_offset/4)+link/4+(link%4)*64)
         //{
         //  int i = link%2;
         //  sendto(sock_fd[i], &packet_buffer[(e_stream*packets_per_stream+frame)*udp_frb_packet_size], 
         //          udp_frb_packet_size , 0 , (struct sockaddr *) &server_address[e_stream] , sizeof(server_address[e_stream])); 
         //
         //  link++;
         //  if(link==number_of_l1_links) link=0;
         //}
/*
         for(int link=0;link<number_of_l1_links;link++)
         {
           //Block Mode
           //if(e_stream==(int)(beam_offset/4)+(int)(link/4)+(int)(link%4)*64)
           //RA Mode
           if (e_stream==beam_offset/4+link)
           {
             int i = link%2;
             sendto(sock_fd[i], &packet_buffer[(e_stream*packets_per_stream+frame)*udp_frb_packet_size],
                    udp_frb_packet_size , 0 , (struct sockaddr *) &server_address[link] , sizeof(server_address[link])); 
           }
         }
*/
         for(int link=0;link<number_of_l1_links;link++)
         {    
           //if (((column_mode) && (e_stream==beam_offset/4+link)) || ((!column_mode) && (e_stream==(int)(beam_offset/4)+(int)(link/4)+(int)(link%4)*64))) 
           if (e_stream==beam_offset/4+link) 
           {
           //}
           //if(e_stream==(int)(beam_offset/4)+(int)(link/4)+(int)(link%4)*64)
           //if(e_stream==beam_offset/4+link)
           //{

             int i = link%2;
             sendto(sock_fd[i], &packet_buffer[(e_stream*packets_per_stream+frame)*udp_frb_packet_size], 
                     udp_frb_packet_size , 0 , (struct sockaddr *) &server_address[link] , sizeof(server_address[link])); 

             //link++;
             //if(link==number_of_l1_links) link=0;
           }
         }
         long wait_per_packet = (long)(58880); 
         
         //61521.25 is the theoretical seperation of packets in ns 
         // I have used 58880 for convinence and also hope this will take care for
         // any clock glitches.

         t1.tv_nsec = t1.tv_nsec+wait_per_packet;
         if(t1.tv_nsec>=1000000000)
         {
           t1.tv_sec = t1.tv_sec + 1;
           t1.tv_nsec = t1.tv_nsec -1000000000;
         }
      }
    }

    mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
    frame_id = ( frame_id + 1 ) % in_buf->num_frames;
    count++;
  }
  return;
}

