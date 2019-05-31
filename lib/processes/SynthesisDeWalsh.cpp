#include "SynthesisDeWalsh.hpp"

REGISTER_KOTEKAN_STAGE(SynthesisDeWalsh);

SynthesisDeWalsh::SynthesisDeWalsh(kotekan::Config& config, const string& unique_name,
                         kotekan::bufferContainer &buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container,
          std::bind(&SynthesisDeWalsh::main_thread, this))
{
    buf = get_buffer("out_buf");
    register_producer(buf, unique_name.c_str());

    dev_name = config.get<string>(unique_name,"device");
    INFO("INIT DEWALSHER\n");

}

SynthesisDeWalsh::~SynthesisDeWalsh() {
//    close(UART);
}

int SynthesisDeWalsh::read_data(void *dest, int src_uart, int length){
    ssize_t rec = 0;
    while (rec < length) {
        int result = read(src_uart, ((char*)dest)+rec, length-rec);
        if (result == -1) {
            ERROR("RECV = -1 %i",errno);
            // Handle error ...
            break;
        }
        else if (result == 0) {
            ERROR("RECV = 0 %i",errno);
            // Handle disconnect ...
            break;
        }
        else {
//            INFO("Received: %i",result);
            rec += result;
        }
    }
    return rec;
}

#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#ifdef macos
#include <IOKit/serial/ioss.h>
#endif

void SynthesisDeWalsh::main_thread() {
    int UART = open (dev_name.c_str(), O_RDWR | O_NOCTTY);
    int frame_id = 0;
    char *frame_ptr;

    DEBUG("STARTING DEWALSHER");

    //57600, 115200, 230400, 460800, 921600, 1843200, 
    set_interface_attribs(UART,115200,0);


    char walsh[16][16] = {
        {-1,-1,-1,-1, 1, 1, 1, 1,  -1,-1,-1,-1, 1, 1, 1, 1},
        {-1,-1,-1,-1, 1, 1, 1, 1,  -1,-1,-1,-1, 1, 1, 1, 1},
        {-1,-1, 1, 1, 1, 1,-1,-1,  -1,-1, 1, 1, 1, 1,-1,-1},
        {-1,-1, 1, 1, 1, 1,-1,-1,  -1,-1, 1, 1, 1, 1,-1,-1},
        { 1, 1,-1,-1, 1, 1,-1,-1,   1, 1,-1,-1, 1, 1,-1,-1},
        { 1, 1,-1,-1, 1, 1,-1,-1,   1, 1,-1,-1, 1, 1,-1,-1},
        { 1,-1,-1, 1, 1,-1,-1, 1,   1,-1,-1, 1, 1,-1,-1, 1},
        { 1,-1,-1, 1, 1,-1,-1, 1,   1,-1,-1, 1, 1,-1,-1, 1},
        {-1, 1, 1,-1, 1,-1,-1, 1,  -1, 1, 1,-1, 1,-1,-1, 1},
        {-1, 1, 1,-1, 1,-1,-1, 1,  -1, 1, 1,-1, 1,-1,-1, 1},
        {-1, 1,-1, 1, 1,-1, 1,-1,  -1, 1,-1, 1, 1,-1, 1,-1},
        {-1, 1,-1, 1, 1,-1, 1,-1,  -1, 1,-1, 1, 1,-1, 1,-1},
        { 1,-1, 1,-1, 1,-1, 1,-1,   1,-1, 1,-1, 1,-1, 1,-1},
        { 1,-1, 1,-1, 1,-1, 1,-1,   1,-1, 1,-1, 1,-1, 1,-1},
        {-1,-1,-1,-1,-1,-1,-1,-1,  -1,-1,-1,-1,-1,-1,-1,-1},
        {-1,-1,-1,-1,-1,-1,-1,-1,  -1,-1,-1,-1,-1,-1,-1,-1}
    };

    bool od[4] = {0,0,0,0};
    unsigned char walsh_idx = 0;
    frame_ptr = (char*) wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
    if (frame_ptr == NULL) return;
    while (!stop_thread) {

        int result = write(UART, "a", 1);
        if (result != 1) ERROR("Uh-oh, wrong send length!");

        char c[4];
        int n = read_data(c,UART,4);
        if (n!=4) ERROR("Uh-oh, wrong recv length!");

        bool d[4];
        d[0] = c[0]=='1';
        d[1] = c[1]=='1';
        d[2] = c[2]=='1';
        d[3] = c[3]=='1';
        bool rise[4];
        rise[0] = !od[0] && d[0];
        rise[1] = !od[1] && d[1];
        rise[2] = !od[2] && d[2];
        rise[3] = !od[3] && d[3];
        bool fall[4];
        fall[0] = od[0] && !d[0];
        fall[1] = od[1] && !d[1];
        fall[2] = od[2] && !d[2];
        fall[3] = od[3] && !d[3];

        od[0] = d[0];
        od[1] = d[1];
        od[2] = d[2];
        od[3] = d[3];

        unsigned char SAMPLE=0;
        unsigned char RESET =1;
        unsigned char SYNC  =2;
        unsigned char MISC  =3;

        (void)MISC;
        (void)RESET;
        (void)fall;

        if (rise[SYNC]) walsh_idx=0;

        if (rise[SAMPLE]) {
            mark_frame_full(buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;

            frame_ptr = (char*) wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
            if (frame_ptr == NULL) return;

            walsh_idx = (walsh_idx+1)%16;
            for (int i=0; i<16; i++) for (int j=0; j<16; j++)
                frame_ptr[i*16+j] = walsh[i][walsh_idx] * walsh[j][walsh_idx];
            frame_ptr[16*16] = walsh_idx;
/*          //quick hack to print out the current walsh matrix, for debugging
            for (int i=0; i<16; i++){
                for (int j=0; j<16; j++){
                    printf("%2d ",frame_ptr[i*16+j]);
                }
                printf("\n");
            }
*/
        }


        DEBUG("State: %d,%d,%d,%d  Rise: %d,%d,%d,%d  Fall: %d,%d,%d,%d  Walsh Idx: %d",
                d[0],d[1],d[2],d[3],
                rise[0],rise[1],rise[2],rise[3],
                fall[0],fall[1],fall[2],fall[3],
                walsh_idx);
        sleep(0.01);
    }
}

int SynthesisDeWalsh::set_interface_attribs (int fd, int speed, int parity) {
    struct termios tty;
    memset (&tty, 0, sizeof tty);
    if (tcgetattr (fd, &tty) != 0) {
        ERROR("error %d from tcgetattr", errno);
        return -1;
    }

    cfsetospeed (&tty, speed);
    cfsetispeed (&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
    // disable IGNBRK for mismatched speed tests; otherwise receive break
    // as \000 chars
    tty.c_iflag &= ~IGNBRK;         // disable break processing
    tty.c_lflag = 0;                // no signaling chars, no echo,
                                    // no canonical processing
    tty.c_oflag = 0;                // no remapping, no delays
    tty.c_cc[VMIN]  = 1;            // read blocks
    tty.c_cc[VTIME] = 1;            // 0.1 seconds read timeout

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

    tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
                                    // enable reading
    tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
    tty.c_cflag |= parity;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr (fd, TCSANOW, &tty) != 0) {
        ERROR("error %d from tcsetattr", errno);
        return -1;
    }
    return 0;
}

