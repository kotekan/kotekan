#ifndef CALLBACKDATA_H
#define CALLBACKDATA_H

#include "gpu_command.h"
#include "buffers.h"
#include "pthread.h"

struct loopCounter {
    loopCounter(){
        CHECK_ERROR( pthread_mutex_init(&lock, NULL) );
        CHECK_ERROR( pthread_cond_init(&cond, NULL) );
    }
    ~loopCounter() {
        CHECK_ERROR( pthread_mutex_destroy(&lock) );
        CHECK_ERROR( pthread_cond_destroy(&cond) );
    }
    int iteration=0;

    pthread_mutex_t lock;  // Lock for the is_ready function.
    pthread_cond_t cond;
};
class callBackData {
public:
    callBackData();
    callBackData(cl_uint param_NumCommand);
    callBackData(callBackData &cb);
    ~callBackData();
    int buffer_id;
    int numCommands;
    int use_beamforming;

    gpu_command ** listCommands;

    // Buffer objects
    struct Buffer * in_buf;
    struct Buffer * out_buf;
    struct Buffer * beamforming_out_buf;

    struct loopCounter * cnt;
};

#endif // CALLBACKDATA_H
