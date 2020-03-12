#ifndef GPS_TIME_H
#define GPS_TIME_H

#include <stdint.h> // for uint64_t
#include <time.h>   // for timespec

#ifdef __cplusplus
extern "C" {
#endif

#define FPGA_PERIOD_NS 2560


// Set the GPS time frame0 time (we might want to add drift tracking later)
// Currently we are assuming no drift, because if we do have drift then
// the time stamps might not line up when merging frames
// TODO This needs to be carefully addressed to support drifting
void set_global_gps_time(uint64_t frame0_time);
// Returns 1 if the GPS time is set, 0 if not
int is_gps_global_time_set();
// Returns the GPS time based on the given fpga seq number.
struct timespec compute_gps_time(uint64_t fpga_seq_num);

/// Returns the fpga seq at GPS time ``ts``
uint64_t compute_fpga_seq(struct timespec ts);
#ifdef __cplusplus
}
#endif


#endif /* GPS_TIME_H */
