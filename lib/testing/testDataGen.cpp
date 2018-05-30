#include "testDataGen.hpp"
#include <random>
#include "errors.h"
#include "chimeMetadata.h"
#include <unistd.h>
#include <sys/time.h>
// Needed for a bunch of time utilities.
#include "visUtil.hpp"
#include "gpsTime.h"


REGISTER_KOTEKAN_PROCESS(testDataGen);

testDataGen::testDataGen(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name,
                   buffer_container, std::bind(&testDataGen::main_thread, this)) {

    buf = get_buffer("network_out_buf");
    register_producer(buf, unique_name.c_str());
    type = config.get_string(unique_name, "type");
    assert(type == "const" || type == "random" || type=="ramp");
    if (type == "const")
        value = config.get_int(unique_name, "value");
    if (type=="ramp")
        value = config.get_int(unique_name, "value");
    _pathfinder_test_mode = config.get_bool_default(unique_name, "pathfinder_test_mode", false);

    samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    stream_id = config.get_int_default(unique_name, "stream_id", 0);
    num_frames = config.get_int_default(unique_name, "num_frames", -1);
    // Try to generate data based on `samples_per_dataset` cadence or else just generate it as
    // fast as possible.
    wait = config.get_bool_default(unique_name, "wait", true);
    // Whether to wait for is rest signal to start or generate next frame. Useful for testing processes
    // that must interact rest commands. Valid modes are "start", "step", and "none".
    rest_mode = config.get_string_default(unique_name, "rest_mode", "none");
    assert(rest_mode == "none" || rest_mode == "start" || rest_mode == "step");
}

testDataGen::~testDataGen() {

}

void testDataGen::apply_config(uint64_t fpga_seq) {

}

void testDataGen::main_thread() {

    int frame_id = 0;
    int frame_id_abs = 0;
    uint8_t * frame = NULL;
    uint64_t seq_num = 0;
    bool finished_seeding_consant = false;
    static struct timeval now;

    int link_id = 0;


    double start_time = current_time();
    while (!stop_thread) {
        frame = (uint8_t*)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        allocate_new_metadata_object(buf, frame_id);
        set_fpga_seq_num(buf, frame_id, seq_num);
        // TODO This should be dynamic/config controlled.
        set_stream_id(buf, frame_id, stream_id);

        gettimeofday(&now, NULL);
        set_first_packet_recv_time(buf, frame_id, now);

        //std::random_device rd;
        //std::mt19937 gen(rd());
        //std::uniform_int_distribution<> dis(0, 255);
        srand(42);
        unsigned char temp_output;
        // XXX Why sizeof(float) here? -km
        for (uint j = 0; j < buf->frame_size/sizeof(float); ++j) {
            if (type == "const") {
                if (finished_seeding_consant) break;
                frame[j] = value;
            } else if (type == "ramp"){
                frame[j] = fmod(j*value,256*value);
//                frame[j] = j*value;
            } else if (type == "random") {
                unsigned char new_real;
                unsigned char new_imaginary;
                new_real = rand()%16;
                new_imaginary = rand()%16;
                temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output;
            }
        }
        DEBUG("Generated a %s test data set in %s[%d]", type.c_str(), buf->buffer_name, frame_id);

        mark_frame_full(buf, unique_name.c_str(), frame_id);

        frame_id_abs +=1;
        if (num_frames >=0 && frame_id_abs >= num_frames) break;
        frame_id = frame_id_abs % buf->num_frames;

        if (_pathfinder_test_mode == true){
            //Test PF seq_num increment.
            if (link_id == 7){
                link_id = 0;
                seq_num += samples_per_data_set;
            } else {
                link_id++;
            }
        } else {
            seq_num += samples_per_data_set;
        }
        if (frame_id == 0) finished_seeding_consant = true;

        if (wait) {
            double time = current_time();
            double frame_end_time = (start_time + (float) samples_per_data_set
                                     * frame_id_abs * FPGA_PERIOD_NS * 1e-9);
            if (time < frame_end_time) usleep((int) (1e6 * (frame_end_time - time)));
        }
    }
}

