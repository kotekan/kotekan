#include "simple_udp_cap.h"

#include "network.h"
#include "buffers.h"
#include "errors.h"

#include <sys/epoll.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>

#define MAX_EVENTS 100

#define UDP_PACKETSIZE 5032

void simple_udp_cap(void * arg) {

    fprintf(stderr, "Starting udp cap");

    struct udpCapArgs * args;
    args = (struct udpCapArgs *) arg;

    struct epoll_event ev;
    struct epoll_event events[MAX_EVENTS];
    int epoll_fd;
    int num_events;

    // Create the EPOLL file handle
    epoll_fd = epoll_create(10);
    if (epoll_fd == -1) {
        printf("epoll_create; errno %d", errno);
        exit(EXIT_FAILURE);
    }

    // Setup the UDP server.

    int udp_socket;
    struct sockaddr_in server_address;

    udp_socket = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK , IPPROTO_UDP);

    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    inet_pton(AF_INET, args->ip_address, &(server_address.sin_addr));
    server_address.sin_port=htons(args->port_number);
    if ( bind(udp_socket,(struct sockaddr *)&server_address,sizeof(server_address)) == -1) {
        printf("Socket bind; errno %d", errno);
    }

    int n = 256 * 1024 * 1024;
    if (setsockopt(udp_socket, SOL_SOCKET, SO_RCVBUF, (void *) &n, sizeof(n)) == -1) {
        printf("Error in set socket rcvbuf size; errno %d", errno);
    }

    ev.events = EPOLLIN;
    ev.data.fd = udp_socket;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, udp_socket, &ev) == -1) {
        printf("epoll_ctl: listen_sock; errno %d", errno);
        exit(EXIT_FAILURE);
    }

    // Done network setup
    printf("Network thread started");

    int buffer_ID = 0;
    int buffer_location = 0;

    wait_for_empty_buffer(args->buf, buffer_ID);

    for (;;) {

        num_events = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (num_events == -1) {
            printf("epoll_pwait; errno %d", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < num_events; i++) {
            if (events[i].data.fd != udp_socket) {
                continue;
            }

            assert(buffer_location < args->buf->buffer_size);
            ssize_t bytes_read = read(udp_socket, (void*) &args->buf->data[buffer_ID][buffer_location], UDP_PACKETSIZE);

            if (bytes_read != UDP_PACKETSIZE) {
                fprintf(stderr, "Packet with incorrect size received!");
                continue;
            }
            buffer_location += UDP_PACKETSIZE;

            if (buffer_location == args->buf->buffer_size) {
                fprintf(stderr, "filled buffer");
                mark_buffer_full(args->buf, buffer_ID);
                buffer_ID = (buffer_ID + 1) % args->buf->num_buffers;
                wait_for_empty_buffer(args->buf, buffer_ID);
                buffer_location = 0;
            }
        }
    }

}