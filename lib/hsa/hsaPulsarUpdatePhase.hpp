#ifndef HSA_PULSAR_UPDATE_PHASE_H
#define HSA_PULSAR_UPDATE_PHASE_H

#include "hsaCommand.hpp"
#include <mutex>
#include <thread>
#include "restServer.hpp"

class hsaPulsarUpdatePhase: public hsaCommand
{
public:
    hsaPulsarUpdatePhase( Config &config,const string &unique_name,
                        bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaPulsarUpdatePhase();

    int wait_on_precondition(int gpu_frame_id) override;

    void update_gains_callback(connectionInstance& conn, json& json_request);

    void calculate_phase(struct psrCoord psr_coord, timeval time_now, float freq_now, float *gain, float *output);

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id);

    void pulsar_grab_callback(connectionInstance& conn, json& json_request);

private:

    int32_t phase_frame_len;
    float * host_phase_0;
    float * host_phase_1;
    int32_t gain_len;
    string _gain_dir;
    vector<float> default_gains;
    float * host_gain;

    float _source_ra;
    float _source_dec;
    uint32_t _psr_scaling;

    int32_t _num_elements;
    int16_t _num_pulsar;
    int16_t _num_gpus;

    int32_t map_len;
    vector<int32_t> _reorder_map;
    int * _reorder_map_c;

    int32_t metadata_buffer_id;
    int32_t metadata_buffer_precondition_id;
    Buffer * metadata_buf;

    struct psrCoord psr_coord;
    struct psrCoord * psr_coord2;
    struct timeval time_now;

    int32_t freq_idx;
    float freq_MHz;
    float _feed_sep_NS;
    int32_t _feed_sep_EW;

    std::thread phase_thread_handle;

    uint16_t bank_read_id;
    uint16_t bank_write;
    std::mutex mtx_read;
    std::mutex _pulsar_lock;

    bool update_gains; //so gains only load on request!
    bool first_pass; //avoid re-calculating freq-specific params

};

#endif
