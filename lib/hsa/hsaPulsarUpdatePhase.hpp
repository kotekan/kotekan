#ifndef HSA_PULSAR_UPDATE_PHASE_H
#define HSA_PULSAR_UPDATE_PHASE_H

#include "hsaCommand.hpp"
#include <mutex>
#include <thread>
#include "restServer.hpp"

class hsaPulsarUpdatePhase: public hsaCommand
{
public:
    hsaPulsarUpdatePhase(const string &kernel_name, const string &kernel_file_name,
                        hsaDeviceInterface &device, Config &config,
                        bufferContainer &host_buffers,
                        const string &unique_name);

    virtual ~hsaPulsarUpdatePhase();

    void apply_config(const uint64_t& fpga_seq) override;

    int wait_on_precondition(int gpu_frame_id) override;

    void calculate_phase(struct psrCoord psr_coord, timeval time_now, float freq_now, float *output);

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id);

    void pulsar_grab_callback(connectionInstance& conn, json& json_request);

    void phase_thread();

private:

    int32_t phase_frame_len;
    float * host_phase_0;
    float * host_phase_1;

    int32_t _num_elements;
    int16_t _num_pulsar;
    int16_t _num_gpus;

    int32_t metadata_buffer_id;
    int32_t metadata_buffer_precondition_id;
    Buffer * metadata_buf;
  
    struct psrCoord psr_coord;
    struct psrCoord * psr_coord2;
    struct timeval time_now;

    float freq_now;
    int32_t * _elem_position_c = NULL;
    float _feed_sep_NS;
    int32_t _feed_sep_EW;

    std::thread phase_thread_handle;

    uint16_t bank_read_id;
    uint16_t bank_write;
    std::mutex mtx_read;
    std::mutex _pulsar_lock;

};

#endif
