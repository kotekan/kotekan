#ifndef LOOPCOUNTER_H
#define LOOPCOUNTER_H
#include "pthread.h"

struct loopCounter {
    int iteration=0;

    pthread_mutex_t lock;  // Lock for the is_ready function.
    pthread_cond_t cond;
};

#endif // LOOPCOUNTER_H
