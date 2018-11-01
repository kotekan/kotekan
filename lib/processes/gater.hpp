#ifndef GATER_HPP
#define GATER_HPP
#include <unistd.h>

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"
#include <fftw3.h>

#include <string>
using std::string;

/**
 *
 */
class gater : public KotekanProcess {
public:
    /// Constructor, also initializes FFTW and values from config yaml.
    gater(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~gater();
    void main_thread() override;
    virtual void apply_config(uint64_t fpga_seq) override;

private:
    /// Kotekan buffer which this process consumes from.
    /// Data should be packed as int16_t values, [r,i] in each 32b value.
    struct Buffer *in_buf;
    /// Kotekan buffer which this process produces into.
    struct Buffer *out_buf;

    /// Frame index for the input buffer.
    int frame_in;
    /// Frame index for the output buffer.
    int frame_pulsar;

    int32_t _num_gpu_frames;    
    int32_t _samples_per_frame;    
    int32_t _num_elements;    

    vector<double> _polyco_coeffs;
    double _pulse_width;
    double _f0;
    double _tmid;
    double _rphase;
    double _dm;

    timespec add_nsec(timespec time, long nsec);
    timespec get_timespec_diff(timespec t1, timespec t2);
    double gps_to_mjd(timespec gps_time);
    timespec mjd_to_gps(double time_mjd);
    double get_polyco_phase(double time_mjd);
    double get_polyco_time(double phase);
    float get_frequency(struct Buffer* buf, int frame_id);
    bool is_pulsing(timespec gps_time, float freq);
    timespec get_pulse_time(timespec time_gps, float freq);
    vector<timespec> get_pulse_times(timespec start_time_gps, timespec end_time_gps,  float freq);
};


#endif
