#include "pfring.h"

#include <errno.h>
#include <stdio.h>

void hex_dump (const int rows, void *addr, int len) {
    int i;
    unsigned char *char_buf = (unsigned char*)addr;

    for (i = 0; i < len; i++) {
        if ((i % rows) == 0) {
            // Add a new line as needed.
            if (i != 0)
                printf ("\n");

            // Print the offset.
            printf ("  %04x ", i);
        }

        // Print the hex value
        printf (" %02x", char_buf[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {

    // Setup the PF_RING.
    pfring *pd = NULL;
    unsigned char * pkt_buf;
    int num_packets = 0;

    pd = pfring_open("dna0", 10000, PF_RING_PROMISC );

    if(pd == NULL) {
        fprintf(stderr, "pfring_open error [%s] (pf_ring not loaded or quick mode is enabled you and have already a socket bound to %s?)\n",
                strerror(errno), argv[2]);
        exit(-1);
    }

    pfring_set_application_name(pd, "inspect_pkt");

    if (pd->dna.dna_mapped_device == 0) {
        fprintf(stderr, "The device is not in DNA mode.?");
    }

    pfring_set_poll_duration(pd, 100);
    pfring_set_poll_watermark(pd, 1000);

    if (pfring_enable_ring(pd) != 0) {
        fprintf(stderr, "Cannot enable the PF_RING.");
        exit(-1);
    }

    for (;;) {
        struct pfring_pkthdr pf_header;
        int rc = pfring_recv(pd, &pkt_buf, 0, &pf_header, 1);
        if (rc <= 0) {
            // No packets available.
            if (rc < 0) {
                fprintf(stderr, "Error in pfring_recv! %d", rc);
            }
            continue;
        }

        printf("Packet %d:\n", num_packets);
        hex_dump(16, pkt_buf, pf_header.len);

        num_packets++;

        if (num_packets >= 5) {
            break;
        }
    }
}