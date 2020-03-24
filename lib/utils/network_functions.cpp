#include "network_functions.hpp"

#include "kotekanLogging.hpp" // for DEBUG_NON_OO

#include <netinet/in.h>      // for sockaddr_in, htons, ntohs
#include <netinet/ip.h>      // for ip, IP_MAXPACKET
#include <netinet/ip_icmp.h> // for icmp, icmp_id, icmp_seq, ICMP_ECHO, ICMP_ECHOREPLY, ICMP_M...
#include <sys/socket.h>      // for recvfrom, sendto
#include <unistd.h>          // for getpid, socklen_t


bool send_ping(int s, const sockaddr_in& dst, const uint16_t seq_no) {
    uint8_t outpackhdr[IP_MAXPACKET];
    uint8_t* outpack = outpackhdr + sizeof(struct ip);
    int cc = (64 - 8); // data length - icmp echo header len, excluding time
    struct icmp* icp = (struct icmp*)outpack;
    icp->icmp_type = ICMP_ECHO;
    icp->icmp_code = 0;
    icp->icmp_cksum = 0;
    icp->icmp_id = getpid();

    icp->icmp_seq = htons(seq_no);
    icp->icmp_cksum = in_cksum((uint16_t*)icp, cc);
    int rc = sendto(s, (char*)outpack, cc, 0, (struct sockaddr*)&dst, sizeof(struct sockaddr_in));
    return rc == cc;
}


int receive_ping(int s, sockaddr_in& from) {
    socklen_t from_length = sizeof(from);
    uint8_t packet[4096];
    int packet_length = sizeof(packet);
    int rc = recvfrom(s, packet, packet_length, 0, (sockaddr*)&from, &from_length);

    // Check the IP header
    int hlen = sizeof(struct ip);
    if (rc < (hlen + ICMP_MINLEN)) {
        DEBUG_NON_OO("Packet too short. Ignoring.");
        return -1;
    }
    // Now the icmp part
    struct icmp* icp = (struct icmp*)(packet + hlen);
    if (icp->icmp_type == ICMP_ECHOREPLY) {
        if (icp->icmp_id != getpid()) {
            DEBUG_NON_OO("Wrong ICMP id: {}", icp->icmp_id);
            return -1;
        }
        return ntohs(icp->icmp_seq);
    }
    return -1;
}


// source: https://github.com/openbsd/src/blob/master/sbin/ping/ping.c
int in_cksum(uint16_t* addr, int len) {
    int nleft = len;
    uint16_t* w = addr;
    int sum = 0;
    uint16_t answer = 0;

    /*
     * Our algorithm is simple, using a 32 bit accumulator (sum), we add
     * sequential 16 bit words to it, and at the end, fold back all the
     * carry bits from the top 16 bits into the lower 16 bits.
     */
    while (nleft > 1) {
        sum += *w++;
        nleft -= 2;
    }

    /* mop up an odd byte, if necessary */
    if (nleft == 1) {
        *(uint8_t*)(&answer) = *(uint8_t*)w;
        sum += answer;
    }

    /* add back carry outs from top 16 bits to low 16 bits */
    sum = (sum >> 16) + (sum & 0xffff); /* add hi 16 to low 16 */
    sum += (sum >> 16);                 /* add carry */
    answer = ~sum;                      /* truncate to 16 bits */
    return answer;
}
