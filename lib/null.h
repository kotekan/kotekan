#ifndef NULL_THREAD
#define NULL_THREAD

struct NullThreadArg {
    struct Config * config;
    struct Buffer * buf;
};

void null_thread(void * arg);

#endif