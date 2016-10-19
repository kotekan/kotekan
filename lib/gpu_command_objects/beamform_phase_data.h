#ifndef BEAMFORM_PHASE_DATA_H
#define BEAMFORM_PHASE_DATA_H

#include "gpu_command.h"
#include "callbackdata.h"

#include <vector>

class beamform_phase_data: public gpu_command
{
public:
    beamform_phase_data(const char* param_name, Config &config);
    ~beamform_phase_data();
    virtual void build(class device_interface &param_Device) override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface &param_Device, cl_event param_PrecedeEvent) override;
    virtual void cleanMe(int param_BufferID) override;
    virtual void freeMe() override;
protected:
    void get_delays(float * phases, time_t beamform_time);
    void apply_config(const uint64_t& fpga_seq) override;

    // phase data
    float * phases[2];
    int beamforming_do_not_track;
    double inst_lat;
    double inst_long;
    int fixed_time;
    double ra;
    double dec;
    vector<float> feed_positions;
    time_t start_beamform_time;
    int64_t last_bankID;
};

#endif


