/*****************************************
@file
@brief Utils for the transmission code.
*****************************************/

#ifndef TX_UTILS_HPP
#define TX_UTILS_HPP

#ifdef MAC_OSX
#include <chrono> // for clockid_t
#endif

#include <arpa/inet.h>


/** @brief get primary IPv4 address from gethostname()
 *
 *  @return scokaddr_in IPv4 address of node
 **/

struct sockaddr_in get_my_addr(const in_addr_t netaddr, const in_addr_t netmask);


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
#define CLOCK_ABS_NANOSLEEP(clock, ts) clock_nanosleep(clock, TIMER_ABSTIME, &ts, nullptr)
#endif

#endif
