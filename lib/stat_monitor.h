#ifndef STAT_MONITOR_H
#define STAT_MONITOR_H

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BUFS 30

struct stat_monitor_arg {
    // TODO Make this into a vector when moving to c++
    struct Buffer * bufs[MAX_BUFS];
    int num_buffer_objects;
};

void* stat_monitor(void * arg);


#ifdef __cplusplus
}
#endif

#endif /* STAT_MONITOR_H */

