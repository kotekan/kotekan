/*
inspect_pkt_server.c

Adapted by JF Cliche from code from Andre Recnik and Kevin Bandura.

Implements a server that receives packet capture requests on TCP port 5001,
captures the specied number of packets from the specified DNA interface data,
and sends the data back with a header.

Packet requests are single strings in the format "dnaX, n=N" where X is the
DNA port number and N is the number of requested packets. (This format came
when test code was quickly hacked together. Really, this should be a JSON
string.)

Replies consists of N JSON-parsable headers followed each by the packet data
in binary. The header format is {"packet_number": N, "packet_length": L,
"error": S}\n" where N is the sequence number of the packet that follows, L is
the length of that packet, and S is either null (no error) or a string
describing the error. The N packets are then sent in binary format right after
the header.

The program handles timeout errors by sending the appropriate error field in the reply header.

When there is an error, no binary data is sent.

Only one client can connect to the server at a time. The PFRING driver is opened and closed on each request.

Notes:

1) The server must be run on the node as a superuser in order to be able to acces the PFRING drivers.
   > sudo ./inspeck_pkt_server

2) The note must be configured to accept connections on TCP port 5001.
   > sydo iptables -nL   # Lists all rules with their rule number
   > sudo iptables -I INPUT x -p TCP --dport 5001 -J ACCEPT   # where x is the rule number where we want to insert the new rule

3) To compile:
   > cmake inspect_packet  # where inspect_packet is the *folder* containing this program. This generates the Makefile to build the program.
   > make inspect_pkt_server # to compile and link the program.

4) We can ask for a large number of packets, but after an initial burst we
   start missing a few packets now and then. Decode the packets timestamps to
   see what is missing.

Suggested improvements:

- Make the packet capture request a JSON string
- Allow readout of multiple DNAs at the same time
- Pre-allocate large RAM buffers or do whatever is necessary to not drop packets on moderately large requests
- Maybe even extend the packet capture command to capture and stream to files
  on multiple disks to allow sustained throughput. This way we can have a
  single packet capture program to do everything.
- Maybe add command line options to also operate this program as the original screen dump program

*/

#include "pfring.h"


/*Required Headers*/

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>


int main(int argc, unsigned char ** argv)
{

    char str[100];
    int sock_fd, comm_fd;
    int port = 5001;
    struct sockaddr_in servaddr;
    int so_reuseaddr_value = 1;
    int n;
    int dna_number, number_of_packets;
    char dna_name[10];
    if ((sock_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      perror("ERROR opening socket");
      exit(1);
      }

    if ((setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &so_reuseaddr_value, sizeof(so_reuseaddr_value))) < 0) {
        perror("Error while setting socket options");
        exit(1);
        }

    bzero( &servaddr, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htons(INADDR_ANY);
    servaddr.sin_port = htons(port);

    if (bind(sock_fd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0) {
        perror("Error while binding socket");
        exit(1);
        }

    listen(sock_fd, 10);

    printf("Waiting for TCP connections on port %i\n", port);

    while(1) {
        // comm_fd = accept(sock_fd, (struct sockaddr*) NULL, NULL);
        if ((comm_fd = accept(sock_fd, (struct sockaddr*) NULL, NULL)) < 0) {
            perror("Error while accepting connection");
            exit(1);
            }
        printf("Connection opened\n");
        while(1) {
            bzero(str, 100);
            n = read(comm_fd, str, 100 - 1);
            if (n <= 0)
                break;
            printf("Received packet request: %s\n", str);
            n = sscanf(str, "dna%i, n=%i", &dna_number, &number_of_packets);
            if (n<2 || dna_number<0 || dna_number >15 || number_of_packets < 0 || number_of_packets > 1000000) {
                printf("Bad packet request!\n");
                continue;
            }
            sprintf(dna_name, "dna%i", dna_number);
            capture_and_send_packets(dna_name, number_of_packets, comm_fd);
        }
        printf("Connection closed\n");
        close(comm_fd);
    }
}


int capture_and_send_packets(char *interface, int number_of_packets, int target_fd) {

    // Setup the PF_RING.
    pfring *pd = NULL;
    unsigned char * pkt_buf;
    int num_packets = 0;
    char packet_header[100];
    time_t t0;

    pd = pfring_open(interface, 16384, PF_RING_PROMISC );

    if(pd == NULL) {
        fprintf(stderr, "pfring_open error [%s] (pf_ring not loaded or quick mode is enabled you and have already a socket bound to %s?)\n",
              strerror(errno), interface);
        sprintf(packet_header, "{\"error\": \"PFRING open error\"}\n");
        write(target_fd, packet_header, strlen(packet_header));
        goto fail;
    }

    pfring_set_application_name(pd, "inspect_pkt");

    if (pd->dna.dna_mapped_device == 0) {
        fprintf(stderr, "The device is not in DNA mode.?");
    }

    pfring_set_poll_duration(pd, 1000);
    pfring_set_poll_watermark(pd, 10000);

    if (pfring_enable_ring(pd) != 0) {
        fprintf(stderr, "Cannot enable the PF_RING.\n");
        sprintf(packet_header, "{\"error\": \"PFRING enable error\"}\n");
        write(target_fd, packet_header, strlen(packet_header));
        goto fail;
    }

    t0 = time(NULL);
    for (;;) {
        struct pfring_pkthdr pf_header;
        int rc;
        if (time(NULL)-t0 > 1) {
            fprintf(stderr, "Timeout\n");
            sprintf(packet_header, "{\"error\": \"Timeout while waiting for packets\"}\n");
            write(target_fd, packet_header, strlen(packet_header));
            goto fail;
        }
        rc = pfring_recv(pd, &pkt_buf, 0, &pf_header, 0);
        if (rc == 0)
            continue;
        else if (rc < 0)
            fprintf(stderr, "Error in pfring_recv! %d\n", rc);
        else {
            // Create and send a JSON header
            sprintf(packet_header, "{\"packet_number\": %i, \"packet_length\": %i, \"error\": null}\n", num_packets, pf_header.len);
            write(target_fd, packet_header, strlen(packet_header));
            write(target_fd, pkt_buf, pf_header.len);
            t0 = time(NULL);

            num_packets++;

            if (num_packets >= number_of_packets) {
                break;
            }
        }
    }
    fail:
    if (pd)
        pfring_close(pd);
}
