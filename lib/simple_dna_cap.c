#include "simple_dna_cap.h"

#include "network.h"
#include "buffers.h"
#include "errors.h"

#include "pfring.h"

#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>

void simple_dna_cap(void * arg) {

    fprintf(stderr, "Starting udp cap");

    struct dnaCapArgs * args;
    args = (struct dnaCapArgs *) arg;

    // Setup the PF_RING.
    pfring *pd = NULL;
    unsigned char * pkt_buf;

    char dna_name[50];
    snprintf(dna_name, 50, "dna%d", args->dna_id);

    pd = pfring_open(dna_name, 10000, PF_RING_PROMISC);

    if(pd == NULL) {
        fprintf(stderr, "pfring_open error [%s] (pf_ring not loaded or quick mode is enabled you and have already a socket bound to dna%d?)\n",
                strerror(errno), args->dna_id);
        exit(-1);
    }

    pfring_set_application_name(pd, "simple_dna_cap");

    if (pd->dna.dna_mapped_device == 0) {
        fprintf(stderr, "The device is not in DNA mode.?");
    }

    pfring_set_poll_duration(pd, 100);
    pfring_set_poll_watermark(pd, 1000);

    if (pfring_enable_ring(pd) != 0) {
        fprintf(stderr, "Cannot enable the PF_RING.");
        exit(-1);
    }


    int buffer_ID = 0;
    int buffer_location = 0;

    int first_run = 1;

    wait_for_empty_buffer(args->buf, buffer_ID);

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

        if (pf_header.len != args->packet_size) {
            fprintf(stderr, "Packet with incorrect size received, size was: %d\n", pf_header.len);
            continue;
        }

        if (first_run == 1) {
            uint32_t seq = *((uint32_t *) &pkt_buf[54]);
            if ( (seq % args->integration_edge) <= 10 && (seq % args->integration_edge) >= 0 ) {
                fprintf(stderr, "Link dna%d; got first packet with seq num: %d", args->dna_id, seq);
                first_run = 0;
            } else {
                continue;
            }
        }
        assert((buffer_location + args->packet_size) <= args->buf->buffer_size);

        memcpy( (void*) &args->buf->data[buffer_ID][buffer_location],
                (void*) pkt_buf,
                args->packet_size );
        //fprintf(stderr, "copied packet!");

        buffer_location += args->packet_size;

        if (buffer_location == args->buf->buffer_size) {

            mark_buffer_full(args->buf, buffer_ID);
            buffer_ID = (buffer_ID + 1) % args->buf->num_buffers;

            if (args->close_on_block == 1) {
                // Stop as soon as we start to block.
                if (is_buffer_empty(args->buf, buffer_ID) == 0) {
                    mark_producer_done(args->buf, 0);
                    fprintf(stderr,
                            "Network buffer is full, ending network capture on dna%d...\n",
                            args->dna_id);
                    int ret = 0;
                    pthread_exit((void *) &ret);
                }
            } else {
                wait_for_empty_buffer(args->buf, buffer_ID);
            }
            buffer_location = 0;
        }
    }

}