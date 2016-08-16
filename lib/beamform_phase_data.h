#ifndef BEAMFORM_PHASE_DATA_H
#define BEAMFORM_PHASE_DATA_H

#include "gpu_command.h"
#include "callbackdata.h"
#include "config.h"

class beamform_phase_data: public gpu_command
{
public:
    beamform_phase_data(char* param_name);
    ~beamform_phase_data();
    virtual void build(Config *param_Config, class device_interface &param_Device);
    virtual cl_event execute(int param_bufferID, class device_interface &param_Device, cl_event param_PrecedeEvent);
    virtual void cleanMe(int param_BufferID);
    virtual void freeMe();
protected:
    void get_delays(float * phases, time_t beamform_time);
    
    // phase data
    float * phases;
    int beamforming_do_not_track;
    double inst_lat;
    double inst_long;
    int num_elements;
    int num_local_freq;
    int fixed_time;
    double ra;
    double dec;
    float * feed_positions;
    cl_event * data_staged_event;
    time_t start_beamform_time;
};

#endif


