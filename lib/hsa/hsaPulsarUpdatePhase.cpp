#include <math.h>
#include "hsaBase.h"
#include "restServer.hpp"
#include "hsaPulsarUpdatePhase.hpp"
//#include "buffer.h"
//#include "bufferContainer.hpp" 
# define likely(x)      __builtin_expect(!!(x), 1)
# define unlikely(x)    __builtin_expect(!!(x), 0)

#define PI 3.14159265
#define light 3.e8
#define R2D 180./PI
#define D2R PI/180.

hsaPulsarUpdatePhase::hsaPulsarUpdatePhase(const string& kernel_name, const string& kernel_file_name,
					   hsaDeviceInterface& device, Config& config,
					   bufferContainer& host_buffers, const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name){
    apply_config(0);


    in_buf = (struct Buffer **)malloc(_num_gpus * sizeof (struct Buffer *));
    for (int i = 0; i < _num_gpus; ++i) {
      in_buf[i] = host_buffers.get_buffer("gpu_input_buffer_" + std::to_string(i));
      register_consumer(in_buf[i], unique_name.c_str());
    }

    phase_frame_len = _num_elements*_num_pulsar*2*sizeof(float);
    //Two alternating banks
    host_phase_0 = (float *)hsa_host_malloc(phase_frame_len);
    host_phase_1 = (float *)hsa_host_malloc(phase_frame_len);
    int index = 0;
    for (int b=0; b < _num_pulsar*_num_elements; b++){
         host_phase_0[index++] = 0;
	 host_phase_0[index++] = 0;
    }
    
    ra = (float *)hsa_host_malloc(_num_pulsar*sizeof(float));
    dec = (float *)hsa_host_malloc(_num_pulsar*sizeof(float));

    //Come up with an initial position, to be updated
    for (int i=0;i<_num_pulsar;i++){
        ra[i] = 53.51337;
	dec[i] = 54.6248916;
    }
    bank_read_id = 8;
    //Here launch a new thread to listen for updates
    phase_thread_handle = std::thread(&hsaPulsarUpdatePhase::phase_thread, std::ref(*this));
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int j = 4; j < 12; j++){
        CPU_SET(j, &cpuset);
	pthread_setaffinity_np(phase_thread_handle.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
}

 hsaPulsarUpdatePhase::~hsaPulsarUpdatePhase() {
     free(in_buf);
     hsa_host_free(host_phase_0);
     hsa_host_free(host_phase_1);
     hsa_host_free(ra);
     hsa_host_free(dec);
     if (_elem_position_c != NULL) {
	 delete _elem_position_c;
     }

 }

void hsaPulsarUpdatePhase::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_pulsar = config.get_int(unique_name, "num_pulsar");
    _freq = config.get_float(unique_name, "freq");
    _num_gpus = config.get_int(unique_name, "num_gpus");
    //Eventually should have precisely mapped elem_position, for now, assume consecutive
    //_elem_position = config.get_int_array(unique_name,, "element_positions");
    //_elem_position_c = new int32_t[_elem_position.size()];
    //for (uint32_t i = 0; i < _elem_position.size(); ++i) {
    _elem_position_c = new int32_t[_num_elements];
    for (uint32_t i = 0; i < _num_elements; ++i) {
        _elem_position_c[i] = i;
    }
    
    //Now assume they are really regular
    _feed_sep_NS = config.get_float(unique_name, "feed_sep_NS");
    _feed_sep_EW = config.get_int(unique_name, "feed_sep_EW");
 }

 int hsaPulsarUpdatePhase::wait_on_precondition(int gpu_frame_id)
 {
 }

 void hsaPulsarUpdatePhase::calculate_phase(float *ra, float *dec, float time_now, int bank_write) {
     INFO("The function to calculate phase working on bank_write=%d",bank_write);
     if (bank_write == 0) {
	 int index = 0;
	 for (int b=0; b < _num_pulsar; b++){
	     for (int n=0; n<_num_elements; n++){
		 //Dummy for print screen
		 host_phase_0[index++] = 2;//b/float(_num_pulsar);
		 host_phase_0[index++] = 3;//b/float(_num_pulsar);
	     }
	 }
	 INFO("HOST PHASE 0 %f %f %f",host_phase_0[0],host_phase_0[1],host_phase_0[2]);
     }
     if (bank_write == 1){
	 for (int b=0; b < _num_pulsar; b++){
	     //TODO: Some coordinate conversion from RA Dec to Theta Phi
	     float THETA = 20+time_now;//= convert_coord(ra[b],dec[b],time_now);
	     float PHI = 1;//= convert_coord(ra[b],dec[b],time_now);
	     for (int n=0; n<1024; n++){
		 //Not sure how these element positions are arranged
		 int nn = _elem_position_c[n % 4];
		 int mm = _elem_position_c[int(floor(nn/4))];
		 float PHIpi = atan((_feed_sep_EW*nn) / (_feed_sep_NS*mm))*R2D;
		 if (nn == 0) {
		     PHIpi=0;
		 }
		 if (mm==0) {
		     PHIpi=90;
		 }
		 float distance_offset = sin(THETA*D2R)*cos((PHI-PHIpi)*D2R)*sqrt( pow(_feed_sep_NS*mm,2)+pow(_feed_sep_EW*nn,2) );
		 float phase_angle =  PI * 2 * (_freq*1.e6) / light * distance_offset;
		 //Pol 0
		 host_phase_1[(b*2048+nn*256+mm)*2] = cos(phase_angle); //Real
		 host_phase_1[(b*2048+nn*256+mm)*2+1] = sin(phase_angle) ;//Imag;
		 //Pol 1
		 host_phase_1[(b*2048+1024+nn*256+mm)*2] = cos(phase_angle); //Real
		 host_phase_1[(b*2048+1024+nn*256+mm)*2+1] = sin(phase_angle) ;//Imag;
	     }
	 }
	 INFO("HOST PHASE 1 %f %f %f",host_phase_1[0],host_phase_1[1],host_phase_1[2]);
     }
 }

 hsa_signal_t hsaPulsarUpdatePhase::execute(int gpu_frame_id, const uint64_t& fpga_seq,
					    hsa_signal_t precede_signal) {

     uint in_buffer_ID[_num_gpus] ;
     uint8_t * in_frame[_num_gpus];

     // Do the data copy. Now I am doing async everytime there is new data 
     //(i.e., when main_thread is being called, in principle I just need to copy in 
     //when there is an update, which is of slower cadence. Down the road optimization

     // Get the gpu memory pointer. i will need multiple frame, 
     //because while it has been sent away for async copy, the next update might be happening.
   //INFO("[pulsarUpdatePhase] phae-frame_len=%d====================",phase_frame_len);
     void * gpu_memory_frame = device.get_gpu_memory_array("beamform_phase", 
							   gpu_frame_id, phase_frame_len);

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
	     in_frame[gpu_frame_id] = wait_for_full_frame(in_buf[gpu_frame_id], unique_name.c_str(), in_buffer_ID[gpu_frame_id]);
	     if (in_frame[gpu_frame_id] == NULL) goto end_loop;
	     stream_id_t stream_id = get_stream_id_t(in_buf[gpu_frame_id], in_buffer_ID[gpu_frame_id]);
	     float freq_now = bin_number_chime(&stream_id);
	     INFO("#####The frequency is %.2f",freq_now);
	     device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_0, phase_frame_len, precede_signal, signals[gpu_frame_id]);
	 }
	 if (likely(bank_read_id == 1)) {
	     INFO("Reading phase from CPU bank id=1");
	     device.async_copy_host_to_gpu(gpu_memory_frame,(void *)host_phase_1, phase_frame_len, precede_signal, signals[gpu_frame_id]);
	 }
     }
     end_loop:;
     return signals[gpu_frame_id];
 }

 void hsaPulsarUpdatePhase::finalize_frame(int frame_id)
 {
 }

 void hsaPulsarUpdatePhase::pulsar_grab_callback(connectionInstance& conn, json& json_request) {
     //Some try statement here 

     int beam;
     try {
	 beam = json_request["beam"];
     } catch (...) {
	 conn.send_error("could not parse new pulsar beam id", STATUS_BAD_REQUEST);
	 return;
     }
     //check beam within range
     if (beam >= _num_pulsar || beam <0) {
	 conn.send_error("num_pulsar out of range", STATUS_BAD_REQUEST);
	 return;
     }
     //update ra and dec 
     {
	 std::lock_guard<std::mutex> lock(_pulsar_lock);
	 ra[beam] = json_request["ra"];
	 dec[beam] = json_request["dec"];
	 conn.send_empty_reply(STATUS_OK);
     }
 }


void hsaPulsarUpdatePhase::phase_thread() {

    using namespace std::placeholders;
    int bank_write = 0;
    float time_now = 2;
    sleep(5);
    for(;;) {
        //Check time if > 1sec, and somehow update time
        sleep(1);
      
        //Listen to RestServer for new pulsar, and update ra and dec
	restServer * rest_server = get_rest_server();
	//??? Not sure how to parse gpu_id here but i think i need it.
	string endpoint = "/update_pulsar/";//+std::to_string(gpu_id); 
	rest_server->register_json_callback(endpoint,
					    std::bind(&hsaPulsarUpdatePhase::pulsar_grab_callback, this, _1, _2));

	//std::lock_guard<std::mutex> lock(mtx_write); //Currently not locking write, since write is relatively infrequent and we have 2 banks
	calculate_phase(ra, dec, time_now, bank_write);
	INFO("[hsaPulsarUpdatePhase] updated phase_%d; array0=%f, array1=%f",bank_write,host_phase_0[2612], host_phase_1[26112]);
	INFO("[hsaPulsarUpdatePhase] updated phase_%d; array0=%f, array1=%f",bank_write,host_phase_0[1], host_phase_1[1]);
	
	{
  	    std::lock_guard<std::mutex> lock(mtx_read);
	    bank_read_id = bank_write;
	}

	//bank_read_id = bank_write;
	bank_write = (bank_write + 1) % 2;
      
	time_now = time_now + 1; //Eventually something sensible here to update time.
	}
}
