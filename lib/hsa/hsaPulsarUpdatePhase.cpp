//curl localhost:12048/pulsar_gain -X POST -H 'Content-Type: application/json' -d '{"pulsar_gain_dir":["path0","path1","path2","path3","path4","path5","path6","path7","path8","path9"]}'

#include <string>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "hsaBase.h"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "restServer.hpp"
#include "hsaPulsarUpdatePhase.hpp"
#include "configUpdater.hpp"

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
    hsaCommand(config, unique_name, host_buffers, device, "", ""){

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_beams = config.get<int16_t>(unique_name, "num_beams");

    _feed_sep_NS = config.get<float>(unique_name, "feed_sep_NS");
    _feed_sep_EW = config.get<int32_t>(unique_name, "feed_sep_EW");

    vector<float> dg = {0.0,0.0}; //re,im
    default_gains = config.get_default<std::vector<float>>(
                unique_name, "frb_missing_gains", dg);

    _source_ra = config.get<std::vector<float>>(unique_name, "source_ra");
    _source_dec = config.get<std::vector<float>>(unique_name, "source_dec");
    _source_scl = config.get<std::vector<int>>(unique_name, "psr_scaling");

    for (int i=0;i<_num_beams;i++){
        psr_coord_latest_update.ra[i] = _source_ra[i];
        psr_coord_latest_update.dec[i] = _source_dec[i];
        psr_coord_latest_update.scaling[i] = _source_scl[i];
    }

    //Just for metadata manipulation
    metadata_buf = host_buffers.get_buffer("network_buf");
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    freq_idx = -1;
    freq_MHz = -1;

    update_gains=true;
    first_pass=true;
    gain_len = 2*_num_beams*_num_elements*sizeof(float);
    host_gain = (float *)hsa_host_malloc(gain_len);

    phase_frame_len = _num_elements*_num_beams*2*sizeof(float);
    //Two alternating banks
    host_phase_0 = (float *)hsa_host_malloc(phase_frame_len);
    host_phase_1 = (float *)hsa_host_malloc(phase_frame_len);
    int index = 0;
    for (uint b=0; b < _num_beams*_num_elements; b++){
        host_phase_0[index++] = 0;
        host_phase_0[index++] = 0;
    }

    bankID = (uint *)hsa_host_malloc(device.get_gpu_buffer_depth());
    bank_use_0 = 0;
    bank_use_1 = 0;
    second_last = 0;

    config_base = "/gpu/gpu_" + std::to_string(device.get_gpu_id());

    // Register function to listen for new pulsar, and update ra and dec
    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    endpoint_psrcoord = config_base + "/update_pulsar/"+std::to_string(device.get_gpu_id());
    rest_server.register_post_callback(endpoint_psrcoord,
                                        std::bind(&hsaPulsarUpdatePhase::pulsar_grab_callback, this, _1, _2));

    //listen for gain updates
    configUpdater::instance().subscribe(config.get<std::string>(unique_name,"updatable_gain_psr"),
                                        std::bind(&hsaPulsarUpdatePhase::update_gains_callback, this, _1));
}

hsaPulsarUpdatePhase::~hsaPulsarUpdatePhase() {
    restServer::instance().remove_json_callback(endpoint_psrcoord);
    hsa_host_free(host_phase_0);
    hsa_host_free(host_phase_1);
    hsa_host_free(bankID);
    hsa_host_free(host_gain);
}

bool hsaPulsarUpdatePhase::update_gains_callback(nlohmann::json &json) {
    update_gains=true;
    try {
        _gain_dir = json.at("pulsar_gain_dir").get<std::vector<string>>();
        INFO("[PSR] Updating gains from %s %s %s %s %s %s %s %s %s %s", _gain_dir[0].c_str(), _gain_dir[1].c_str(),_gain_dir[2].c_str(),_gain_dir[3].c_str(), _gain_dir[4].c_str(), _gain_dir[5].c_str(), _gain_dir[6].c_str(), _gain_dir[7].c_str(), _gain_dir[8].c_str(), _gain_dir[9].c_str());
    } catch (std::exception const & e) {
        WARN("[PSR] Fail to read gain_dir %s", e.what());
        return false;
    }
    return true;
}

int hsaPulsarUpdatePhase::wait_on_precondition(int gpu_frame_id)
{
    (void)gpu_frame_id;
    uint8_t * frame = wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == NULL) return -1;
    metadata_buffer_precondition_id = (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;
    return 0;
}

void hsaPulsarUpdatePhase::calculate_phase(struct psrCoord psr_coord, timespec time_now, float freq_now, float *gains, float *output) {

    float FREQ = freq_now;
    struct tm * timeinfo;
    timeinfo = localtime (&time_now.tv_sec);
    uint year = timeinfo->tm_year+1900;
    uint month = timeinfo->tm_mon+1;
    uint day = timeinfo->tm_mday;
    float JD = 2-int(year/100.)+int(int(year/100.)/4.)+int(365.25*year)+int(30.6001*(month+1))+day+1720994.5;
    double T= (JD-2451545.0)/36525.0;  //Works if time after year 2000, otherwise T is -ve and might break
    double T0=fmod((6.697374558 + (2400.051336*T) + (0.000025862*T*T)),24.);
    double UT = (timeinfo->tm_hour) + (timeinfo->tm_min/60.) + (timeinfo->tm_sec+time_now.tv_nsec/1.e9)/3600.;
    double GST = fmod((T0 + UT*1.002737909),24.);
    double LST = GST + inst_long/15.;
    while (LST <0) {LST= LST+24;}
    LST = fmod(LST,24);
    for (int b=0; b < _num_beams; b++){
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
                    output[(b*_num_elements+elem_id)*2  ] = delay_real*gains[(b*_num_elements + elem_id)*2  ] - delay_imag*gains[(b*_num_elements + elem_id)*2+1];
                    output[(b*_num_elements+elem_id)*2+1] = delay_real*gains[(b*_num_elements + elem_id)*2+1] + delay_imag*gains[(b*_num_elements + elem_id)*2  ];
                }
            }
        }
    }
}

hsa_signal_t hsaPulsarUpdatePhase::execute(int gpu_frame_id,
                                           hsa_signal_t precede_signal) {
    //Update phase every one second
    const uint64_t phase_update_period = 390625;
    uint64_t current_seq = get_fpga_seq_num(metadata_buf, metadata_buffer_id);
    second_now = (current_seq / phase_update_period) % 2;
    if (second_now != second_last) {
        update_phase = true;
    }
    second_last = second_now;

    if (first_pass) {
        first_pass = false;
        //From the metadata, figure out the frequency
        stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
        freq_idx = bin_number_chime(&stream_id);
        freq_MHz = freq_from_bin(freq_idx);
        update_phase = true;
    }

    if (update_gains) {
        update_gains=false;
        FILE *ptr_myfile;
        char filename[256];
        for (int b=0;b<_num_beams;b++){
            snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin",_gain_dir[b].c_str(),freq_idx);
            INFO("Loading gains from %s",filename);
            ptr_myfile=fopen(filename,"rb");
            if (ptr_myfile == NULL) {
                ERROR("GPU Cannot open gain file %s", filename);
                for (int i=0;i<2048;i++){
                    host_gain[(b*2048+i)*2  ] = default_gains[0];
                    host_gain[(b*2048+i)*2+1] = default_gains[1];
                }
            }
            else {
                if (_num_elements != fread(&host_gain[b*2048*2],sizeof(float)*2,_num_elements,ptr_myfile)) {
                    ERROR("Gain file (%s) wasn't long enough! Something went wrong, breaking...", filename);
                    raise(SIGINT);
                    return precede_signal;
                }
                fclose(ptr_myfile);
            }
        } //end beam
    }
    if(update_phase) {
        //GPS time, need ch_master
        time_now_gps = get_gps_time(metadata_buf, metadata_buffer_id);
        if (time_now_gps.tv_sec == 0) {
            ERROR("GPS time appears to be zero, bad news for pulsar timing!");
        }
        //use whichever bank that has no lock
        if ( bank_use_0 == 0)  {  //no more outstanding async copy using bank0
            std::lock_guard<std::mutex> lock(_pulsar_lock);
            psr_coord = psr_coord_latest_update;
            calculate_phase(psr_coord, time_now_gps, freq_MHz, host_gain, host_phase_0);
            bank_active=0;
            update_phase = false;
        }
        else if (bank_use_1 == 0) { //no more outstanding async copy using bank1
            std::lock_guard<std::mutex> lock(_pulsar_lock);
            psr_coord = psr_coord_latest_update;
            calculate_phase(psr_coord, time_now_gps, freq_MHz, host_gain, host_phase_1);
            bank_active = 1;
            update_phase = false;
        }
    }

    bankID[gpu_frame_id] = bank_active; // update or not, read from the latest bank
    set_psr_coord(metadata_buf, metadata_buffer_id, psr_coord);
    metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;

    // Do the data copy. Now I am doing async everytime there is new data
    //(i.e., when main_thread is being called, in principle I just need to copy in
    //when there is an update, which is of slower cadence. Down the road optimization

    // Get the gpu memory pointer. i will need multiple frame through the use of get_gpu_mem_array,
    //because while it has been sent away for async copy, the next update might be happening.
    void * gpu_memory_frame = device.get_gpu_memory_array("beamform_phase", gpu_frame_id, phase_frame_len);

    if (bankID[gpu_frame_id] == 0) {
        device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_0, phase_frame_len, precede_signal, signals[gpu_frame_id]);
        bank_use_0 = bank_use_0 + 1;
    }
    if (bankID[gpu_frame_id] == 1) {
        device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_1, phase_frame_len, precede_signal, signals[gpu_frame_id]);
        bank_use_1 = bank_use_1 + 1;
    }
    return signals[gpu_frame_id];
}

void hsaPulsarUpdatePhase::finalize_frame(int frame_id)
{
    hsaCommand::finalize_frame(frame_id);
    if (bankID[frame_id] == 1) {
        bank_use_1 = bank_use_1 - 1;
    }
    if (bankID[frame_id] == 0) {
        bank_use_0 = bank_use_0 - 1;
    }
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
    if (beam >= _num_beams || beam <0) {
        conn.send_error("num_beams out of range", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    //update ra and dec
    {
        std::lock_guard<std::mutex> lock(_pulsar_lock);
        psr_coord_latest_update.ra[beam] = json_request["ra"];
        psr_coord_latest_update.dec[beam] = json_request["dec"];
        psr_coord_latest_update.scaling[beam] = json_request["scaling"];
        conn.send_empty_reply(HTTP_RESPONSE::OK);
        config.update_value(config_base, "source_ra/" + std::to_string(beam), json_request["ra"]);
        config.update_value(config_base, "source_dec/" + std::to_string(beam), json_request["dec"]);
        config.update_value(config_base, "psr_scaling/" + std::to_string(beam), json_request["scaling"]);
        update_phase = true;
    }
}
