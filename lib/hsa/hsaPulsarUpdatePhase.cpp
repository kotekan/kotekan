#include <math.h>
#include <time.h>
#include "hsaBase.h"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "restServer.hpp"
#include "hsaPulsarUpdatePhase.hpp"
//#include "buffer.h"
//#include "bufferContainer.hpp"
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define PI 3.14159265
#define light 3.e8
#define R2D 180./PI
#define D2R PI/180.
#define TAU 2*PI
#define inst_long -119.6175
#define inst_lat 49.3203

REGISTER_HSA_COMMAND(hsaPulsarUpdatePhase);

hsaPulsarUpdatePhase::hsaPulsarUpdatePhase(Config& config, const string &unique_name,
                           bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand("", "", config, unique_name, host_buffers, device){

    _num_elements = config.get_int(unique_name, "num_elements");
    _num_pulsar = config.get_int(unique_name, "num_pulsar");
    _num_gpus = config.get_int(unique_name, "num_gpus");
    _elem_position_c = new int32_t[_num_elements];
    for (int i = 0; i < _num_elements; ++i) {
        _elem_position_c[i] = i;
    }

    //Now assume they are really regular
    _feed_sep_NS = config.get_float(unique_name, "feed_sep_NS");
    _feed_sep_EW = config.get_int(unique_name, "feed_sep_EW");

    //Just for metadata manipulation
    metadata_buf = host_buffers.get_buffer("network_buf");
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    freq_now = 0;

    phase_frame_len = _num_elements*_num_pulsar*2*sizeof(float);
    //Two alternating banks
    host_phase_0 = (float *)hsa_host_malloc(phase_frame_len);
    host_phase_1 = (float *)hsa_host_malloc(phase_frame_len);
    int index = 0;
    for (int b=0; b < _num_pulsar*_num_elements; b++){
        host_phase_0[index++] = 0;
        host_phase_0[index++] = 0;
    }

    //Come up with an initial position, to be updated
    for (int i=0;i<_num_pulsar;i++){
        psr_coord.ra[i] = 53.51337;
        psr_coord.dec[i] = 54.6248916;
    }

    bank_read_id = 8;
    bank_write = 0;

    //Here launch a new thread to listen for updates
    phase_thread_handle = std::thread(&hsaPulsarUpdatePhase::phase_thread, std::ref(*this));
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto &i : config.get_int_array(unique_name, "cpu_affinity"))
      CPU_SET(i, &cpuset);
    pthread_setaffinity_np(phase_thread_handle.native_handle(), sizeof(cpu_set_t), &cpuset);
}

hsaPulsarUpdatePhase::~hsaPulsarUpdatePhase() {
    hsa_host_free(host_phase_0);
    hsa_host_free(host_phase_1);
    if (_elem_position_c != NULL) {
        delete _elem_position_c;
    }
}

int hsaPulsarUpdatePhase::wait_on_precondition(int gpu_frame_id)
{
    uint8_t * frame = wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == NULL) return -1;
    metadata_buffer_precondition_id = (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;
    return 0;
}

void hsaPulsarUpdatePhase::calculate_phase(struct psrCoord psr_coord, timeval time_now, float freq_now, float *output) {
  //Mostly copied from get_delays in beamform_phase_data.cpp

    for (int b=0; b < _num_pulsar; b++){
        const double one_over_c = 3.3356;
        const double phi_0 = 280.46;  //This is a PF value, using as placeholder
        const double LST_rate = 360./86164.09054;
        const double j2000_unix = 946728000; //unix timestamp
        double time = time_now.tv_sec + time_now.tv_usec/1.e6;
        //INFO("[calculate_phase] time=%.6f",time);
        double precession_offset = (time - j2000_unix) * 0.012791 / (365 * 24 * 3600);
        double LST = phi_0 + inst_long + LST_rate*(time - j2000_unix) - precession_offset;
        LST = fmod(LST, 360.);
        double hour_angle = LST - psr_coord.ra[b];
        double alt = sin(psr_coord.dec[b]*D2R)*sin(inst_lat*D2R)+cos(psr_coord.dec[b]*D2R)*cos(inst_lat*D2R)*cos(hour_angle*D2R);
        alt = asin(alt);
        double az = (sin(psr_coord.dec[b]*D2R) - sin(alt)*sin(inst_lat*D2R))/(cos(alt)*cos(inst_lat*D2R));
        az = acos(az);
        if(sin(hour_angle*D2R) >= 0){az = TAU - az;}

        double projection_angle, effective_angle, offset_distance;
        for(int i = 0; i < _num_elements; ++i){
            //Why does this not depend on the frequency? CHECK
            //Also, it seems taht elem_position has real& imag? why 2*i+1?
            projection_angle = 90*D2R - atan2(_elem_position_c[2*i+1],_elem_position_c[2*i]);
            offset_distance  = cos(alt)*sqrt(pow(_elem_position_c[2*i],2) + pow(_elem_position_c[2*i+1],2));
            effective_angle  = projection_angle - az;
            output[(b*2048+i)*2] = TAU*cos(effective_angle)*offset_distance*one_over_c; //Real
            output[(b*2048+i)*2+1] = TAU*cos(effective_angle)*offset_distance*one_over_c; //Imag
        }
    }
}

hsa_signal_t hsaPulsarUpdatePhase::execute(int gpu_frame_id, const uint64_t& fpga_seq,
                                            hsa_signal_t precede_signal) {

    //From the metadata, figure out the frequency
    stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
    freq_now = bin_number_chime(&stream_id);

    //GPS time, need ch_master
    /*struct timespec time_now_gps = get_gps_time(metadata_buf, metadata_buffer_id);
    time_now = get_gps_time(metadata_buf, metadata_buffer_id);
    if (time_now.tv_sec == 0) {
        //Sytstem time, not as accurate
        time_now = get_first_packet_recv_time(metadata_buf, metadata_buffer_id);
    }*/
    time_now = get_first_packet_recv_time(metadata_buf, metadata_buffer_id);

    char time_buf[64];
    time_t temp_time = time_now.tv_sec;
    struct tm* l_time = gmtime(&temp_time);
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", l_time);

    INFO("####Frequency is %.2f; metadata_buffer_id=%d, gpu_frame_id=%d time stamp: %ld.%06ld (%s.%06ld) device.get_gpu_id=%d",
         freq_now, metadata_buffer_id,  gpu_frame_id, time_now.tv_sec, time_now.tv_usec, time_buf, device.get_gpu_id());

    //Update phase every one second
    const uint64_t phase_update_period = 390625;
    uint64_t current_seq = get_fpga_seq_num(metadata_buf, metadata_buffer_id);
    uint bankID = (current_seq / phase_update_period) % 2;

    if(bankID == bank_write) {  //Time to update
        if (bank_write == 0) {
            calculate_phase(psr_coord, time_now, freq_now, host_phase_0);
        }
        if (bank_write == 1) {
            calculate_phase(psr_coord, time_now, freq_now, host_phase_1);
        }
        {
            std::lock_guard<std::mutex> lock(mtx_read);
            bank_read_id = bank_write;
        }
        bank_write = (bank_write + 1) % 2; //So if next time want to update, get written to the alt. bank instead to avoid overwritting
    }

    set_psr_coord(metadata_buf, metadata_buffer_id, psr_coord);
    INFO("Updating H8 to be RA[0]=%f Dec[0]=%f for  metadata_buffer_id=%d gpu_frame_id=%d", psr_coord.ra[0], psr_coord.dec[0],  metadata_buffer_id, gpu_frame_id);

    metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;

    // Do the data copy. Now I am doing async everytime there is new data
    //(i.e., when main_thread is being called, in principle I just need to copy in
    //when there is an update, which is of slower cadence. Down the road optimization

    // Get the gpu memory pointer. i will need multiple frame,
    //because while it has been sent away for async copy, the next update might be happening.
    void * gpu_memory_frame = device.get_gpu_memory("beamform_phase",
                                                    phase_frame_len);

    {
        std::lock_guard<std::mutex> lock(mtx_read); //Prevent multiple read if read_id change during execut

        //This is just for the beginning, and sending host_phase_0 which are all zeros.
        if (unlikely(bank_read_id==8)) {
            INFO("Waiting for bank_read, current id=%d",bank_read_id);
            device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_0, phase_frame_len, precede_signal, signals[gpu_frame_id]);
        }
        //as soon as it start updating bank_read_id will be either 0 or 1
        if (likely(bank_read_id == 0)) {
            INFO("Reading phase from CPU bank id=0");
            device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_0, phase_frame_len, precede_signal, signals[gpu_frame_id]);
        }
        if (likely(bank_read_id == 1)) {
            INFO("Reading phase from CPU bank id=1");
            device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_1, phase_frame_len, precede_signal, signals[gpu_frame_id]);
        }
    }
    return signals[gpu_frame_id];
}

void hsaPulsarUpdatePhase::finalize_frame(int frame_id)
{
    hsaCommand::finalize_frame(frame_id);
}

void hsaPulsarUpdatePhase::pulsar_grab_callback(connectionInstance& conn, json& json_request) {
    //Some try statement here
    int beam;
    try {
        beam = json_request["beam"];
    } catch (...) {
        conn.send_error("could not parse new pulsar beam id", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    //check beam within range
    if (beam >= _num_pulsar || beam <0) {
        conn.send_error("num_pulsar out of range", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    //update ra and dec
    {
        std::lock_guard<std::mutex> lock(_pulsar_lock);
        psr_coord.ra[beam] = json_request["ra"];
        psr_coord.dec[beam] = json_request["dec"];
        conn.send_empty_reply(HTTP_RESPONSE::OK);
        INFO("=============!!![H8]!!-------Pulsar endpoint got beam=%d, ra=%.4f dec=%.4f gpu=%d",beam,psr_coord.ra[beam],psr_coord.dec[beam], device.get_gpu_id());
    }
}

void hsaPulsarUpdatePhase::phase_thread() {

    using namespace std::placeholders;
    sleep(5);
    for(;;) {
        sleep(1);
        //Listen to RestServer for new pulsar, and update ra and dec
        restServer &rest_server = restServer::instance();
        string endpoint = unique_name + "/update_pulsar/"+std::to_string(device.get_gpu_id());
        rest_server.register_json_callback(endpoint,
                                            std::bind(&hsaPulsarUpdatePhase::pulsar_grab_callback, this, _1, _2));
    }
}
