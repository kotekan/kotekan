/**
 * @file
 * @brief Simple windowed stats functions and value storage
 * - movingStats
 * -- create_moving_stats
 * -- add_sample
 * -- get_average
 */
#ifndef MOVING_STATS_H
#define MOVING_STATS_H

#include <pthread.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct movingStats
 * @brief Simple struct to store moving pipeline stats
 *
 * @todo Add more than just average value
 * @author Andre Renard
 */
struct movingStats {
    /// The samples to do stats on.
    double * samples;

    /// The number of samples in the @c sample_times array.
    uint32_t num_samples;

    /// The maximum total
    uint32_t max_samples;

    /// The current array position
    uint32_t head_idx;

    /// The current average
    double average;

    /// Lock the updates to ensure thread thread safety
    pthread_mutex_t lock;
};

struct movingStats * create_moving_stats(uint32_t max_samples);

void delete_stats(struct movingStats * stats);

void add_sample(struct movingStats * stats, double sample);

double get_average(struct movingStats * stats);

#ifdef __cplusplus
}
#endif

#endif /* MOVING_STATS_H */
