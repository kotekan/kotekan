#ifndef NETWORK_UTILITY_FUNCTIONS_H
#define NETWORK_UTILITY_FUNCTIONS_H
#include <netinet/ip.h>


/*****************************************
@file
@brief Utility functions for dealing with low-level sockets APIs
*****************************************/

/**
 * @brief Send a ping packet to the destination using source socket @a s
 *
 * @param s socket to use for communication
 * @param dst destination address to which to send
 *
 * @return @c true if ping was sent successfully
 */
bool send_ping(int s, const sockaddr_in& dst);

/**
 * @brief Receive a ping response on socket @p s, returning `true` if all OK and putting the
 * sender's address into @a from
 *
 * @param[in] s socket to use for communication
 * @param[out] from the response sender's address
 *
 * @return @c true if a ping response was received successfully
 */
bool receive_ping(int s, sockaddr_in& from);

/**
 * @brief Calculate internet packet checksum (based on BSD networking code)
 *
 * @param addr
 * @param length

 * @return valid IP packet checksum
 */
int in_cksum(uint16_t* addr, const int length);


#endif // NETWORK_UTILITY_FUNCTIONS_H
