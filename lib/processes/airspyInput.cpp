#include "airspyInput.hpp"

airspyInput::airspyInput(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&airspyInput::main_thread, this)) {

    buf = get_buffer("out_buf");
    register_producer(buf, unique_name.c_str());
//    base_dir = config.get_string(unique_name, "base_dir");
}

airspyInput::~airspyInput() {
   airspy_stop_rx(a_device);
   airspy_close(a_device);
   airspy_exit();
}

void airspyInput::apply_config(uint64_t fpga_seq) {

}

void airspyInput::main_thread() {
    frame_id = 0;
    frame_loc = 0;
    recv_busy = PTHREAD_MUTEX_INITIALIZER;

    airspy_init();
    
   a_device=init_device();

    airspy_start_rx(a_device, airspy_callback, static_cast<void*>(this));
}

int airspyInput::airspy_callback(airspy_transfer_t* transfer){
    DEBUG("Airspy Callback");
    airspyInput* proc = static_cast<airspyInput*>(transfer->ctx);
    proc->airspy_producer(transfer);
    return 0;
}
void airspyInput::airspy_producer(airspy_transfer_t* transfer){
    //first, make sure two callbacks don't do this at once
    pthread_mutex_lock(&recv_busy);

    void *in = transfer->samples;
    int bt = transfer->sample_count * BYTES_PER_SAMPLE;
    while (bt > 0){
        if (frame_loc == 0){
            DEBUG("Airspy waiting for frame_id %d",frame_id);
            buf_ptr = (unsigned char*) wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        }

        int copy_length = bt < buf->frame_size ? bt : buf->frame_size;
        DEBUG("Filling Buffer %d With %d Data Samples",frame_id,copy_length);
        //FILL THE BUFFER
        memcpy(buf_ptr, in, copy_length);
        bt-=copy_length;
        frame_loc = (frame_loc + copy_length) % buf->frame_size;
        
        if (frame_loc == 0){
            DEBUG("Airspy Buffer %d Full",frame_id);
            mark_frame_full(buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
        }
    }
    pthread_mutex_unlock(&recv_busy);
}

struct airspy_device *airspyInput::init_device(){
    int result;
    uint8_t board_id = AIRSPY_BOARD_ID_INVALID;

    struct airspy_device *dev;
    result = airspy_open(&dev);
    if( result != AIRSPY_SUCCESS ) {
        printf("airspy_open() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
        airspy_exit();
    }

    int sample_rate_val=2500000;
//  int sample_rate_val=10000000;
    result = airspy_set_samplerate(dev, sample_rate_val);
    if (result != AIRSPY_SUCCESS) {
        printf("airspy_set_samplerate() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
        airspy_close(dev);
        airspy_exit();
    }

    result = airspy_set_sample_type(dev, (enum airspy_sample_type)5);
    if (result != AIRSPY_SUCCESS) {
        printf("airspy_set_sample_type() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
        airspy_close(dev);
        airspy_exit();
    }

    result = airspy_set_vga_gain(dev, 15); //MAX:15
    if( result != AIRSPY_SUCCESS ) {
        printf("airspy_set_vga_gain() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
    }

    result = airspy_set_freq(dev, 1420000000);
    if( result != AIRSPY_SUCCESS ) {
        printf("airspy_set_freq() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
    }

    result = airspy_set_mixer_gain(dev, 15); //MAX: 15
    if( result != AIRSPY_SUCCESS ) {
        printf("airspy_set_mixer_gain() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
    }
    result = airspy_set_mixer_agc(dev, 0); //Auto gain control: 0/1
    if( result != AIRSPY_SUCCESS ) {
        printf("airspy_set_mixer_agc() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
    }

    result = airspy_set_lna_gain(dev, 14); //MAX: 14
    if( result != AIRSPY_SUCCESS ) {
        printf("airspy_set_lna_gain() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
    }


    result = airspy_set_rf_bias(dev, 0);//biast_val);
    if( result != AIRSPY_SUCCESS ) {
        printf("airspy_set_rf_bias() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
        airspy_close(dev);
        airspy_exit();
    }

    result = airspy_board_id_read(dev, &board_id);
    if (result != AIRSPY_SUCCESS) {
        fprintf(stderr, "airspy_board_id_read() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
    }
    printf("Board ID Number: %d (%s)\n", board_id, airspy_board_id_name((enum airspy_board_id)board_id));

    airspy_read_partid_serialno_t read_partid_serialno;
    result = airspy_board_partid_serialno_read(dev, &read_partid_serialno);
    if (result != AIRSPY_SUCCESS) {
        fprintf(stderr, "airspy_board_partid_serialno_read() failed: %s (%d)\n", airspy_error_name((enum airspy_error)result), result);
    }
    printf("Part ID Number: 0x%08X 0x%08X\n",
        read_partid_serialno.part_id[0],
        read_partid_serialno.part_id[1]);
    printf("Serial Number: 0x%08X%08X\n",
        read_partid_serialno.serial_no[2],
        read_partid_serialno.serial_no[3]);
//        return EXIT_FAILURE;
    return dev;
}
