#include "parse_host_name.hpp"

void parse_host_name(int &my_rack, int &my_node, int &my_nos, int &my_node_id)
{
  int rack=0,node=0,nos=0;
  //std::stringstream temp_ip[number_of_subnets];
  char* my_host_name = (char*) malloc(sizeof(char)*100);
  //CHECK_MEM(my_host_name);
  gethostname(my_host_name, sizeof(my_host_name));

  if(my_host_name[0] != 'c' && my_host_name[3] != 'g')
  {
    //INFO("Not a valid name \n");
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
    //INFO("Not a valid name \n");
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
    default: exit(0);
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
    default: exit(0);

  }

  if(rack<7)my_node_id += rack*10+(9-node); //fix for the arrangment of nodes in the racks
  if(rack>7) my_node_id += (rack-1)*10+(9-node);
  my_rack = rack;
  my_node = node;
  my_nos = nos;
}


