#ifndef NULL_THREAD
#define NULL_THREAD

#ifdef __cplusplus
extern "C" {
#endif

struct NullThreadArg {
    struct Config * config;
    struct Buffer * buf;
};

void* null_thread(void * arg);

#ifdef __cplusplus
}
#endif

#endif