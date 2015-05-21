#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "null.h"
#include "buffers.h"
#include "errors.h"
#include "chrx.h"
#include "output_formating.h"
#include "config.h"

void null_thread(void* arg)
{
    struct NullThreadArg * args = (struct NullThreadArg *) arg;

    int buffer_ID = 0;

    // Wait for, and drop full buffers
    for (;;) {

        // This call is blocking!
        buffer_ID = get_full_buffer_from_list(args->buf, &buffer_ID, 1);

        // Check if the producer has finished, and we should exit.
        if (buffer_ID == -1) {
            INFO("Closing null thread");
            int ret;
            pthread_exit((void *) &ret);
        }

        INFO("Dropping frame in null thread.");

        mark_buffer_empty(args->buf, buffer_ID);

        buffer_ID = (buffer_ID + 1) % args->buf->num_buffers;
    }
}
