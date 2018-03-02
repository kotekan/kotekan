#include "testDataGenFloat.hpp"
#include <random>
#include "errors.h"
#include "chimeMetadata.h"
#include <unistd.h>
#include <sys/time.h>

REGISTER_KOTEKAN_PROCESS(testDataGenFloat);

testDataGenFloat::testDataGenFloat(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name,
                   buffer_container, std::bind(&testDataGenFloat::main_thread, this)) {

    buf = get_buffer("network_out_buf");
    register_producer(buf, unique_name.c_str());
    type = config.get_string(unique_name, "type");
    assert(type == "const" || type == "random" || type=="ramp");
    if (type == "const")
        value = config.get_int(unique_name, "value");
    if (type=="ramp")
        value = config.get_float(unique_name, "value");
    _pathfinder_test_mode = config.get_bool_default(unique_name, "pathfinder_test_mode", false);
}

testDataGenFloat::~testDataGenFloat() {

}

void testDataGenFloat::apply_config(uint64_t fpga_seq) {

}

void testDataGenFloat::main_thread() {

    int frame_id = 0;
    float * frame = NULL;
    uint64_t seq_num = 0;
    bool finished_seeding_consant = false;
    static struct timeval now;

    int link_id = 0;

    while (!stop_thread) {
        frame = (float*)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        allocate_new_metadata_object(buf, frame_id);
        set_fpga_seq_num(buf, frame_id, seq_num);
        // TODO This should be dynamic/config controlled.
        set_stream_id(buf, frame_id, 0);

        gettimeofday(&now, NULL);
        set_first_packet_recv_time(buf, frame_id, now);

        //std::random_device rd;
        //std::mt19937 gen(rd());
        //std::uniform_int_distribution<> dis(0, 255);
        srand(42);
        unsigned char temp_output;
        for (int j = 0; j < buf->frame_size/sizeof(float); ++j) {
            if (type == "const") {
                if (finished_seeding_consant) break;
                frame[j] = value;
            } else if (type == "ramp"){
                frame[j] = fmod(j*value,256*value);
            } else if (type == "random") {
                unsigned char new_real;
                unsigned char new_imaginary;
                new_real = rand()%16;
                new_imaginary = rand()%16;
                temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output;
            }
        }
        usleep(83000);
        DEBUG("Generated a %s test data set in %s[%d]", type.c_str(), buf->buffer_name, frame_id);

        mark_frame_full(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + 1) % buf->num_frames;
        
        if (_pathfinder_test_mode == true){
            //Test PF seq_num increment.
            if (link_id == 7){
                link_id = 0;
                seq_num += 32768;
            } else {
                link_id++;
            }
        } else {
            seq_num += 32768;
        }
        if (frame_id == 0) finished_seeding_consant = true;
    }
}

