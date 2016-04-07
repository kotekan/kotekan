#ifndef TIME_TRACKING
#define TIME_TRACKING

#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The functions in this file work like a static class.
// Time is represented as nanoseconds since the j2000 epoch.

uint32_t first_fpga_seq_num;
uint64_t first_fpga_seq_num_time;

void set_fpga_num_and_time(const struct timeval * time, const uint32_t fpga_seq_num);

uint64_t get_time_from_seq_number(const uint32_t seq_number);

uint32_t get_vdif_second(const uint32_t seq_number);

uint32_t get_vdif_frame(const uint32_t seq_number);

// Gets the location within a frame in the range of [0,649]
uint32_t get_vdif_location(const uint32_t seq_number);

#ifdef __cplusplus
}
#endif

#endif