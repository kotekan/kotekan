#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <complex.h>
#include <math.h>
#include <assert.h>

#define MAX_NUM_LINKS (8)

// A TCP frame contains this header followed by the visibilities, and flags.
// -- HEADER:sizeof(TCP_frame_header) --
// -- VISIBILITIES:n_prod * n_freq * sizeof(complex_int_t) --
// -- FLAGS:n_prod * sizeof(uint8_t) --
#pragma pack(1)
struct stream_id {
    unsigned int link_id : 8;
    unsigned int slot_id : 8;
    unsigned int crate_id : 8;
    unsigned int reserved : 8;
};

struct tcp_frame_header {
    uint32_t fpga_seq_number;
    uint32_t num_freq;
    uint32_t num_vis; // The number of visibilities per frequency.

    struct stream_id stream_ids[MAX_NUM_LINKS];

    struct timeval cpu_timestamp; // The time stamp as set by the GPU correlator - not accurate!
};
#pragma pack(0)

#define N_PROD 32896
#define N_FREQ 64
#define PORT 40016

//! A complex integer datatype.
typedef struct {
    int32_t real; //!< The real component.
    int32_t imag; //!< The imaginary component.
} complex_int_t;

int main(int argc, char ** argv) {
    unsigned char *tcp_buf;
    uint8_t *curr_flag, err_flag;
    int i, j, k, l, tcp_listen_fd, tcp_fd, bytes_recv, buffer_size;
    int n_val;
    socklen_t tcp_addr_len;
    struct sockaddr_in acq_addr, gpu_addr;
    struct tcp_frame_header *header;
    complex_int_t * vis;

    // Buffer size is number of visibilities x number of frequencies, plus the
    // header.
    // TODO This should be expanded to handle the flags as well.
    n_val = N_PROD * N_FREQ;
    buffer_size = sizeof(struct tcp_frame_header) +
        n_val * (sizeof(complex_int_t) + sizeof(uint8_t));

    tcp_buf = malloc(buffer_size);
    if (tcp_buf == NULL) {
        fprintf(stderr, "Error creating tcp buffer.");
        return NULL;
    }

    curr_flag = (uint8_t *)malloc(n_val * sizeof(uint8_t));
    if (curr_flag == NULL) {
        fprintf(stderr, "Error creating curr_flag buffer.");
        return NULL;
    }

    tcp_listen_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (tcp_listen_fd < 0) {
        fprintf(stderr, "Error creating tcp socket.");
        return NULL;
    }

    bzero(&acq_addr, sizeof(acq_addr));
    acq_addr.sin_family = AF_INET;
    acq_addr.sin_addr.s_addr=htonl(INADDR_ANY);
    acq_addr.sin_port=htons(PORT);

    if (bind(tcp_listen_fd, (struct sockaddr *)&acq_addr,
        sizeof(struct sockaddr)) < 0) {
        fprintf(stderr, "Error binding tcp socket (errno = %d).");
        return -1;
    }

    listen(tcp_listen_fd, 16);

    tcp_addr_len = sizeof(gpu_addr);

    FILE * out_fp = fopen(argv[1], "wb");
    if (!out_fp) {
        fprintf(stderr, "Error opening file %s", argv[1]);
        return -1;
    }

    for (;;) {

        fprintf(stderr, "Waiting for connection from GPU.\n");
        tcp_fd = accept(tcp_listen_fd, (struct sockaddr *)&gpu_addr, &tcp_addr_len);

        if (tcp_fd < 0) {
            fprintf(stderr, "Error failed to accept tcp socket.\n");
            continue;
        } else {
            fprintf(stderr, "TCP connection with GPU established.\n");
        }

        for (;;) {
            bytes_recv = recv(tcp_fd, (void *)tcp_buf, buffer_size, MSG_WAITALL);

            if (bytes_recv == 0) {
                fprintf(stderr, "The GPU correlator closed its connection.\n");
                close(tcp_fd);
                fclose(out_fp);
                exit(0);
            }

            if (bytes_recv < 0) {
                fprintf(stderr, "There was an error with the GPU correlator connection.\n");
                break;
            }

            if (bytes_recv != buffer_size) {
                fprintf(stderr, "Did not get full buffer: expected %d, got %d.",
                        buffer_size, bytes_recv);
                break;
            }

            header = (struct tcp_frame_header *)tcp_buf;
            vis = (complex_int_t *)(tcp_buf + sizeof(struct tcp_frame_header));
            err_flag = (uint8_t *)(tcp_buf + sizeof(struct tcp_frame_header) +
                        n_val * sizeof(complex_int_t));

            if (header->num_freq != N_FREQ || header->num_vis != N_PROD) {
                fprintf(stderr, "bytes_recv: %d", bytes_recv);
                fprintf(stderr, "SEQ NUM: %d", header->fpga_seq_number);
                fprintf(stderr, "freq: %d", header->num_freq);
                fprintf(stderr, "vis: %d", header->num_vis);
                fprintf(stderr, "local freq: %d ", N_FREQ);
                fprintf(stderr, "local vis: %d", N_PROD);
                fprintf(stderr, "The tcp frame has a different number of frequencies %d or visibilities %d.", header->num_freq,
                header->num_vis);
                return NULL;
            }

            printf("Got GPU frame: %d\n",  header->fpga_seq_number);
            printf("     Part || Link Id || Slot ID || Crate ID || Reserved\n");
            for (int j = 0; j < MAX_NUM_LINKS; ++j) {
                printf("%8u %8u %8u %8u %8u\n", j, header->stream_ids[j].link_id,
                       header->stream_ids[j].slot_id, header->stream_ids[j].crate_id,
                       header->stream_ids[j].reserved);
            }

            size_t bytes_written = fwrite(vis, n_val * sizeof(complex_int_t), 1, out_fp);
            printf("bytes_written: %d", bytes_written);
            //assert(bytes_written == n_val * sizeof(complex_int_t));

        }
    }

    fclose(out_fp);
}