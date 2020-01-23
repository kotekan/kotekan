#include "time_tracking.h"

#include <sys/time.h> // for timeval

static uint32_t first_fpga_seq_num;
static uint64_t first_fpga_seq_num_time;

void set_fpga_num_and_time(const struct timeval* time, const uint32_t fpga_seq_num) {
    // Assign the given time to first seq number, we use the j2000 epoch here.
    // HACK we cheat here and do not look at the micro seconds so that we always
    // align to interger seconds.  This will actually be true once the fpga supports
    // being locked to GPU clock and maser.  For now this is hacked.
    first_fpga_seq_num_time = (time->tv_sec - 946728000) * 1000000000;

    // First fpga seq number
    first_fpga_seq_num = fpga_seq_num;
}

uint64_t get_time_from_seq_number(const uint32_t seq_number) {
    return first_fpga_seq_num_time + (seq_number - first_fpga_seq_num) * 2560;
}

uint32_t get_vdif_second(const uint32_t seq_number) {
    return get_time_from_seq_number(seq_number) / 1000000000;
}

uint32_t get_vdif_frame(const uint32_t seq_number) {
    return (get_time_from_seq_number(seq_number) % 1000000000) / (625 * 2560);
}

uint32_t get_vdif_location(const uint32_t seq_number) {
    return (get_time_from_seq_number(seq_number) % 1000000000) % 625;
}
