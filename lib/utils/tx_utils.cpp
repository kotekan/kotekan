#include "tx_utils.hpp"

#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

#include "fmt.hpp" // for compile_string_to_view

#include <cassert>      // for assert
#include <cstring>      // for strlen
#include <netdb.h>      // for addrinfo, freeaddrinfo, gai_strerror, getaddrinfo
#include <netinet/in.h> // for sockaddr_in, AF_INET
#include <stdexcept>    // for runtime_error
#include <stdio.h>      // for sscanf
#include <stdlib.h>     // for malloc, size_t
#include <time.h>       // for timespec
#include <unistd.h>     // for gethostname

struct sockaddr_in get_my_addr(const in_addr_t netaddr, const in_addr_t netmask) {
    // std::stringstream temp_ip[number_of_subnets];
    char* my_host_name = (char*)malloc(sizeof(char) * 100);
    // CHECK_MEM(my_host_name);
    gethostname(my_host_name, sizeof(my_host_name));

    // Use getaddrinfo to resolve the address (works for both IPs and hostnames)
    struct addrinfo hints, *result = nullptr;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;       // Only IPv4
    hints.ai_socktype = SOCK_STREAM; // TCP socket
    hints.ai_protocol = IPPROTO_TCP; // TCP once more
    hints.ai_flags = AI_ADDRCONFIG;

    const int status = getaddrinfo(my_host_name, nullptr, &hints, &result);
    if (status != 0) {
        FATAL_ERROR_NON_OO("get_my_ip: getaddrinfo failed for {:s}: {:s}", my_host_name,
                           gai_strerror(status));
    }

    struct sockaddr_in retval = {};
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        // all CHIME x-engine nodes have ip addresses 10.1.X.Y
        // similar for other telescopes given appropriate netmask and netaddr
        if (rp->ai_family == AF_INET) {
            struct sockaddr_in const* const addr_in =
                reinterpret_cast<const sockaddr_in*>(rp->ai_addr);
            // NOTE: these are all in network order (big endian)
            if ((addr_in->sin_addr.s_addr & netmask) == netaddr) {
                retval.sin_family = static_cast<unsigned short int>(rp->ai_family);
                retval.sin_port = reinterpret_cast<const sockaddr_in*>(rp->ai_addr)->sin_port;
                retval.sin_addr.s_addr =
                    reinterpret_cast<const sockaddr_in*>(rp->ai_addr)->sin_addr.s_addr;
            }
            break;
        }
    }
    if (retval.sin_addr.s_addr == 0) {
        FATAL_ERROR_NON_OO(
            "Could not find primary address matching netmask {:x} and netaddr {:x} for node {:s}",
            netmask, netaddr, my_host_name);
    }
    freeaddrinfo(result);
    return retval;
}

void parse_chime_host_name(int& my_rack, int& my_node, int& my_nos, int& my_node_id) {
    int rack = 0, node = 0, nos = 0;
    // std::stringstream temp_ip[number_of_subnets];
    char* my_host_name = (char*)malloc(sizeof(char) * 100);
    // CHECK_MEM(my_host_name);
    gethostname(my_host_name, sizeof(my_host_name));

    if (my_host_name[0] != 'c' && my_host_name[3] != 'g') {
        // INFO_NON_OO("Not a valid name \n");
        throw std::runtime_error("Invalid host name");
    }


    if (my_host_name[1] == 'n') {
        nos = 0;
        my_node_id = 0;
    } else if (my_host_name[1] == 's') {
        nos = 100;
        my_node_id = 128;
    } else {
        throw std::runtime_error("Invalid host name");
    }

    switch (my_host_name[2]) {
        case '0':
            rack = 0;
            break;
        case '1':
            rack = 1;
            break;
        case '2':
            rack = 2;
            break;
        case '3':
            rack = 3;
            break;
        case '4':
            rack = 4;
            break;
        case '5':
            rack = 5;
            break;
        case '6':
            rack = 6;
            break;
        // case '7': rack=7; break;
        case '8':
            rack = 8;
            break;
        case '9':
            rack = 9;
            break;
        case 'A':
            rack = 10;
            break;
        case 'B':
            rack = 11;
            break;
        case 'C':
            rack = 12;
            break;
        case 'D':
            rack = 13;
            break;
        default:
            throw std::runtime_error("Invalid host name");
    }

    switch (my_host_name[4]) {
        case '0':
            node = 0;
            break;
        case '1':
            node = 1;
            break;
        case '2':
            node = 2;
            break;
        case '3':
            node = 3;
            break;
        case '4':
            node = 4;
            break;
        case '5':
            node = 5;
            break;
        case '6':
            node = 6;
            break;
        case '7':
            node = 7;
            break;
        case '8':
            node = 8;
            break;
        case '9':
            node = 9;
            break;
        default:
            throw std::runtime_error("Invalid host name");
    }

    if (rack < 7)
        my_node_id += rack * 10 + (9 - node); // fix for the arrangment of nodes in the racks
    if (rack > 7)
        my_node_id += (rack - 1) * 10 + (9 - node);
    my_rack = rack;
    my_node = node;
    my_nos = nos;
}

void add_nsec(struct timespec& temp, long nsec) {
    temp.tv_nsec += nsec;
    if (temp.tv_nsec >= 1000000000) {
        long sec = temp.tv_nsec / 1000000000;

        temp.tv_sec += sec;
        temp.tv_nsec -= sec * 1000000000;
    } else if (temp.tv_nsec < 0) {
        long sec = temp.tv_nsec / 1000000000;
        sec -= 1;
        temp.tv_nsec -= sec * 1000000000;
        temp.tv_sec += sec;
    }
}

int get_vlan_from_ip(const char* ip_address) {
    int a, b, c, d, n;
    const int fields_parsed = sscanf(ip_address, "%d.%d.%d.%d%n", &a, &b, &c, &d, &n);
    if (fields_parsed != 4 || (size_t)n != strlen(ip_address)) {
        FATAL_ERROR_NON_OO("Could not parse {:s} as an IPv4 address", ip_address);
    }
    return b;
}


#ifdef MAC_OSX

void osx_clock_abs_nanosleep(clockid_t clock, struct timespec ts) {
    timespec t0;
    clock_gettime(clock, &t0);
    long sec = ts.tv_sec - t0.tv_sec;
    long nsec = ts.tv_nsec - t0.tv_nsec;
    if (nsec < 0) {
        nsec += 1000000000;
        sec -= 1;
    }
    if (sec < 0) {
        return;
    } else {
        timespec waittime;
        waittime.tv_sec = sec;
        waittime.tv_nsec = nsec;
        nanosleep(&waittime, nullptr);
    }
    return;
}

#endif
