#include "movingStats.h"
#include "errors.h"

struct movingStats * create_moving_stats(uint32_t max_samples) {
    struct movingStats * stats = malloc(sizeof(struct movingStats));
    CHECK_MEM(stats);

    stats->samples = malloc(max_samples * sizeof(double));

    stats->num_samples = 0;
    stats->max_samples = max_samples;
    stats->head_idx = 0;
    stats->average = 0;

    CHECK_ERROR( pthread_mutex_init(&stats->lock, NULL) );

    return stats;
}

void delete_stats(struct movingStats * stats) {
    if (stats != NULL) {
        free(stats->samples);
        CHECK_ERROR( pthread_mutex_destroy(&stats->lock) );
        free(stats);
    }
}

void add_sample(struct movingStats * stats, double sample) {
    CHECK_ERROR( pthread_mutex_lock(&stats->lock) );

	double last_sum = stats->num_samples*stats->average;

    if(stats->num_samples < stats->max_samples){
        stats->samples[stats->head_idx] = sample;
        stats->num_samples++;
    } else {
        last_sum -= stats->samples[stats->head_idx];
        stats->samples[stats->head_idx] = sample;
    }

    stats->head_idx = (stats->head_idx + 1) % stats->max_samples;
    stats->average = (last_sum + sample) / stats->num_samples;

    CHECK_ERROR( pthread_mutex_unlock(&stats->lock) );
}

double get_average(struct movingStats * stats) {
    return stats->average;
}