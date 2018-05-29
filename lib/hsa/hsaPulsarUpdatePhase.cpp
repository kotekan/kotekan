#include <math.h>
#include <time.h>
#include "hsaBase.h"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "restServer.hpp"
#include "hsaPulsarUpdatePhase.hpp"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define PI 3.14159265
#define light 3.e8
#define one_over_c 0.0033356
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

    _feed_sep_NS = config.get_float(unique_name, "feed_sep_NS");
    _feed_sep_EW = config.get_int(unique_name, "feed_sep_EW");

    _gain_dir = config.get_string(unique_name, "gain_dir");
    vector<float> dg = {0.0,0.0}; //re,im
    default_gains = config.get_float_array_default(unique_name,"frb_missing_gains",dg);

    psr_coord.ra = config.get_float_array(unique_name, "source_ra");
    psr_coord.dec = config.get_float_array(unique_name, "source_dec");
    //Default scaling factor, can be changed via endpoint
    psr_coord.scaling = config.get_int_array(unique_name, "psr_scaling");

    //Just for metadata manipulation
    metadata_buf = host_buffers.get_buffer("network_buf");
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    freq_idx = -1;
    freq_MHz = -1;

    update_gains=true;
    first_pass=true;
    gain_len = 2*_num_elements*sizeof(float);
    host_gain = (float *)hsa_host_malloc(gain_len);

    phase_frame_len = _num_elements*_num_pulsar*2*sizeof(float);
    //Two alternating banks
    host_phase_0 = (float *)hsa_host_malloc(phase_frame_len);
    host_phase_1 = (float *)hsa_host_malloc(phase_frame_len);
    int index = 0;
    for (int b=0; b < _num_pulsar*_num_elements; b++){
        host_phase_0[index++] = 0;
        host_phase_0[index++] = 0;
    }

    bank_read_id = 8;
    bank_write = 0;

    // Register function to listen for new pulsar, and update ra and dec
    using namespace std::placeholders;
    restServer &rest_server = restServer::instance()
    endpoint_psrcoord = unique_name + "/update_pulsar/"+std::to_string(device.get_gpu_id());
    rest_server.register_post_callback(endpoint_psrcoord,
                                        std::bind(&hsaPulsarUpdatePhase::pulsar_grab_callback, this, _1, _2));

    //Piggy-back on FRB to listen for gain updates
    endpoint_gain = unique_name + "/frb/update_gains/"+std::to_string(device.get_gpu_id());
    rest_server.register_post_callback(endpoint_gain,
                                        std::bind(&hsaPulsarUpdatePhase::update_gains_callback, this, _1, _2));

}

hsaPulsarUpdatePhase::~hsaPulsarUpdatePhase() {
    restServer::instance().remove_json_callback(endpoint);
    hsa_host_free(host_phase_0);
    hsa_host_free(host_phase_1);
    hsa_host_free(host_gain);
}

void hsaPulsarUpdatePhase::update_gains_callback(connectionInstance& conn, json& json_request) {
    try {
        _gain_dir = json_request["gain_dir"];
    } catch (...) {
        conn.send_error("Couldn't parse new gain_dir parameter.", STATUS_BAD_REQUEST);
    return;
    }
    update_gains=true;
    INFO("Updating gains from %s", _gain_dir.c_str());
    conn.send_empty_reply(STATUS_OK);
}

int hsaPulsarUpdatePhase::wait_on_precondition(int gpu_frame_id)
{
    uint8_t * frame = wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == NULL) return -1;
    metadata_buffer_precondition_id = (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;
    return 0;
}

void hsaPulsarUpdatePhase::calculate_phase(struct psrCoord psr_coord, timeval time_now, float freq_now, float *gains, float *output) {

    float FREQ = freq_now;
    struct tm * timeinfo;
    timeinfo = localtime (&time_now.tv_sec);
    uint year = timeinfo->tm_year+1900;
    uint month = timeinfo->tm_mon+1;
    uint day = timeinfo->tm_mday;
    float JD = 2-int(year/100.)+int(int(year/100.)/4.)+int(365.25*year)+int(30.6001*(month+1))+day+1720994.5;
    double T= (JD-2451545.0)/36525.0;  //Works if time after year 2000, otherwise T is -ve and might break
    double T0=fmod((6.697374558 + (2400.051336*T) + (0.000025862*T*T)),24.);
    double UT = (timeinfo->tm_hour) + (timeinfo->tm_min/60.) + (timeinfo->tm_sec+time_now.tv_usec/1.e6)/3600.;
    double GST = fmod((T0 + UT*1.002737909),24.);
    double LST = GST + inst_long/15.;
    while (LST <0) {LST= LST+24;}
    LST = fmod(LST,24);
    for (int b=0; b < _num_pulsar; b++){
        double hour_angle = LST*15. - psr_coord.ra[b];
        double alt = sin(psr_coord.dec[b]*D2R)*sin(inst_lat*D2R)+cos(psr_coord.dec[b]*D2R)*cos(inst_lat*D2R)*cos(hour_angle*D2R);
        alt = asin(alt);
        double az = (sin(psr_coord.dec[b]*D2R) - sin(alt)*sin(inst_lat*D2R))/(cos(alt)*cos(inst_lat*D2R));
        az = acos(az);
        if(sin(hour_angle*D2R) >= 0){az = TAU - az;}
        double projection_angle, effective_angle, offset_distance;

        for (int i=0;i<4;i++){ //loop 4 cylinders
            for (int j=0;j<256;j++) { //loop 256 feeds
                float dist_y = j*_feed_sep_NS;
                float dist_x = i*_feed_sep_EW;
                projection_angle = 90*D2R - atan2(dist_y, dist_x); 
                offset_distance  = sqrt( pow(dist_y,2) + pow(dist_x,2) ) ;
                effective_angle  = projection_angle - az; 
                float delay_real = cos(TAU*cos(effective_angle)*cos(alt)*offset_distance*FREQ*one_over_c);
                float delay_imag = -sin(TAU*cos(effective_angle)*cos(-alt)*offset_distance*FREQ*one_over_c);
                for (int p=0;p<2;p++){ //loop 2 pol
                    uint elem_id = p*1024+i*256+j;
                    //Not scrembled, assume reordering kernel has been run
                    output[(b*_num_elements+elem_id)*2  ] = delay_real*gains[elem_id*2  ] - delay_imag*gains[elem_id*2+1];
                    output[(b*_num_elements+elem_id)*2+1] = delay_real*gains[elem_id*2+1] + delay_imag*gains[elem_id*2  ];
                }
            }
        }
    }
}

hsa_signal_t hsaPulsarUpdatePhase::execute(int gpu_frame_id, const uint64_t& fpga_seq,
                                            hsa_signal_t precede_signal) {

    //Update phase every one second
    const uint64_t phase_update_period = 390625;
    uint64_t current_seq = get_fpga_seq_num(metadata_buf, metadata_buffer_id);
    uint bankID = (current_seq / phase_update_period) % 2;

    if (first_pass) {
        first_pass = false;
        //From the metadata, figure out the frequency
        stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
        freq_idx = bin_number_chime(&stream_id);
        freq_MHz = freq_from_bin(freq_idx);
        bankID = bank_write;
    }

    if (update_gains) {
        //brute force wait to make sure we don't clobber memory
        if (hsa_signal_wait_scacquire(precede_signal, HSA_SIGNAL_CONDITION_LT, 1,
                      UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0) {
        ERROR("***** ERROR **** Unexpected signal value **** ERROR **** ");
        }
        update_gains=false;
        FILE *ptr_myfile;
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin",_gain_dir.c_str(),freq_idx);
        INFO("Loading gains from %s",filename);
        ptr_myfile=fopen(filename,"rb");
        if (ptr_myfile == NULL) {
            ERROR("GPU Cannot open gain file %s", filename);
            for (int i=0;i<2048;i++){
                host_gain[i*2  ] = default_gains[0];
                host_gain[i*2+1] = default_gains[1];
            }
        }
        else {
            if (_num_elements != fread(host_gain,sizeof(float)*2,_num_elements,ptr_myfile)) {
                ERROR("Gain file (%s) wasn't long enough! Something went wrong, breaking...", filename);
            }
            fclose(ptr_myfile);
        }
    }

    if(bankID == bank_write) {  //Time to update
      /*
        //GPS time, need ch_master
        struct timespec time_now_gps = get_gps_time(metadata_buf, metadata_buffer_id);
       time_now = get_gps_time(metadata_buf, metadata_buffer_id);
       if (time_now.tv_sec == 0) {
            //Sytstem time, not as accurate
        INFO("!!!!!! Going to use system time");
        time_now = get_first_packet_recv_time(metadata_buf, metadata_buffer_id);
        }*/
        time_now = get_first_packet_recv_time(metadata_buf, metadata_buffer_id);
        if (bank_write == 0) {
        calculate_phase(psr_coord, time_now, freq_MHz, host_gain, host_phase_0);
        }
        if (bank_write == 1) {
        calculate_phase(psr_coord, time_now, freq_MHz, host_gain, host_phase_1);
        }
        {
            std::lock_guard<std::mutex> lock(mtx_read);
            bank_read_id = bank_write;
        }
        bank_write = (bank_write + 1) % 2; //So if next time want to update, get written to the alt. bank instead to avoid overwritting
    }

    set_psr_coord(metadata_buf, metadata_buffer_id, psr_coord);
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
            device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_0, phase_frame_len, precede_signal, signals[gpu_frame_id]);
        }
        //as soon as it start updating bank_read_id will be either 0 or 1
        if (likely(bank_read_id == 0)) {
            device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_0, phase_frame_len, precede_signal, signals[gpu_frame_id]);
        }
        if (likely(bank_read_id == 1)) {
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
	psr_coord.scaling[beam] = json_request["scaling"];
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    }
}
