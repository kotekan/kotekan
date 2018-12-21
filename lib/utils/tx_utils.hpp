/*****************************************
@file
@brief Utils for the transmission code.
*****************************************/

#ifndef TX_UTILS_HPP
#define TX_UTILS_HPP


#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>

/** @brief parse the gethostname() return string to the IP address of the node
 *
 *  @reference my_rack rack number of the node
 *  @reference my_node node number on the rack
 *  @reference my_nos  north hut or south hut
 *  @reference my_node_id ID of the node derived from the rack, node and nos.
 **/

void parse_chime_host_name(int& my_rack, int& my_node, int& my_nos, int& my_node_id);


/** @brief adds nsec to a give timespec structure.
 *
 *  @param temp timespec structure which must be incremented.
 *  @param nsec nanoseconds to be added to the temp.
 **/

void add_nsec(struct timespec& temp, long nsec);


/** @brief gets the vlan from the IP address
 *
 *  @param ip_address IP addrees for frb or pulsar network.
 **/
int get_vlan_from_ip(const char* ip_address);


#ifdef MAC_OSX
void osx_clock_abs_nanosleep(clockid_t clock, struct timespec ts);
#define CLOCK_ABS_NANOSLEEP(clock, ts) osx_clock_abs_nanosleep(clock, ts)
#else
#define CLOCK_ABS_NANOSLEEP(clock, ts) clock_nanosleep(clock, TIMER_ABSTIME, &ts, NULL)
#endif

#endif
