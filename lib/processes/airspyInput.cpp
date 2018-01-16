#include "airspyInput.hpp"
#include "errors.h"
#include "airspy_control.h"
#include "util.h"

#include <unistd.h>
//#include <libairspy/airspy.h>

airspyInput::airspyInput(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&airspyInput::main_thread, this)) {

   // apply_config(0);

//    buf = get_buffer("out_buf"); //Buffer

  //  register_producer(buf, unique_name.c_str()); //Mark as producer

}

airspyInput::~airspyInput() {

   // airspy_stop_rx(a_device);

   // airspy_close(a_device);

    //airspy_exit();

}

void airspyInput::apply_config(uint64_t fpga_seq) {

}

void airspyInput::main_thread() {

    buf_id = 0;

    //airspy_init();
    
   // a_device=init_device();

    //airspy_start_rx(a_device, my_callback,  static_cast<void*>(this));
}

//int my_callback(airspy_transfer_t* transfer){
    //INFO("Callback");
   // airspyInput* foo = static_cast<airspyInput*>(transfer->ctx);
    //foo->buf_id = foo->airspy_producer(transfer->samples, transfer->sample_count * BYTES_PER_SAMPLE, foo->buf_id);
    
//}

/*
int airspyInput::airspy_producer(void *in, int bt, int ID){
	buf_id = (buf_id + 1) % 10;
	//INFO("Any system call");
	//buf_id++;
	//if(buf_id == 10) buf_id = 0;
}
*/

int airspyInput::airspy_producer(void *in, int bt, int ID){
/*	
	INFO("Airspy waiting for buf_id %d",ID);
	wait_for_empty_buffer(buf, unique_name.c_str(), ID);

        unsigned char* buf_ptr = buf->data[ID];

	INFO("Filling Buffer %d With %d Data Samples",ID,bt);

	//FILL THE BUFFER
	memcpy(buf_ptr, in, bt);
	set_data_ID(buf, ID, ID);
	INFO("Marking Buffer %d Full",ID);
        mark_buffer_full(buf, unique_name.c_str(), ID);
	INFO("Buffer %d Full",ID);

	ID = (ID + 1) % buf->num_buffers;
	//usleep(0);
	//INFO("New Buffer ID %d",buf_id);
	return ID;*/
	return 0;
}
