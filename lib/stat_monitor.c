#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "null.h"
#include "buffers.h"
#include "errors.h"
#include "output_formating.h"
#include "config.h"
#include "stat_monitor.h"

void* stat_monitor(void* arg)
{
    struct stat_monitor_arg * args = (struct stat_monitor_arg *) arg;

    for (;;) {
        sleep(5);
        for (int i = 0; i < args->num_buffer_objects; ++i) {
            print_buffer_status(args->bufs[i]);
        }
    }
}
