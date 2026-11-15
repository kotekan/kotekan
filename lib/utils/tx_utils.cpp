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
