#include<stdio.h>
#include<string.h>
#include<sys/socket.h>
#include<arpa/inet.h>
#include<stdlib.h>
#include<errno.h>
#include<fcntl.h>
#include<unistd.h>

int main(int argc , char *argv[])
{
    int sock;
    struct sockaddr_in server;

    //Create socket
    sock = socket(AF_INET , SOCK_STREAM , 0);
    if (sock == -1)
    {
        printf("Could not create socket");
    }
    printf("Socket created");

    server.sin_addr.s_addr = inet_addr("10.1.1.196");
    server.sin_family = AF_INET;
    server.sin_port = htons( 10042 );

    //Connect to remote server
    if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0)
    {
        perror("connect failed. Error");
        return 1;
    }

    printf("Connected\n");

    // Open the file to write
    FILE * fd;
    const int file_name_len = 100;
    char file_name[file_name_len];

    snprintf(file_name, file_name_len, "/mnt/ram_disk/power_data.dat");

    const int head_size = 3;
    const int line_head_size = 2;
    const int num_entries = 400000;
    const int line_size = 2048 + line_head_size;
    uint32_t buf[line_size + head_size];

    uint32_t line_idx = 0;
    uint32_t num_rolls = 0;
    uint32_t num_sent = 0;

    //keep communicating with server
    while(1)
    {

        fd = fopen(file_name, "r");

        if (fd == NULL) {
            perror("Cannot open file");
            fprintf(stderr, "File name was: %s", file_name);
            exit(errno);
        }

        fseek(fd, 0, SEEK_SET);
        fread(buf, sizeof(int), head_size, fd);
        int cur_line_idx = buf[0];
        int num_samples = buf[2];
        num_sent = 0;
        //fprintf(stderr, "cur_line_idx: %u\n", cur_line_idx);

        for (;;) {
            if (line_idx == cur_line_idx)
                break;

            fseek(fd, sizeof(int) * (line_size * line_idx  + head_size), SEEK_SET);
            fread((void*)&buf[head_size], sizeof(int), line_size, fd);

            buf[0] = line_idx;
            buf[1] = num_rolls;
            buf[2] = num_samples;

            if ( line_idx > num_entries) {
                fprintf(stderr, "line_idx: %u\n", buf[0]);
            }
            if( send(sock , buf , (line_size + head_size) * sizeof(int) , 0) < 0)
            {
                printf("Send failed");
                return 1;
            }

            line_idx++;
            if (line_idx >= num_entries) {
                line_idx = 0;
                num_rolls++;
            }
            num_sent++;
        }

        if (num_sent > 1000) {
            fprintf(stderr, "Warning, over 1000 lines behind!! - line sent = %d", num_sent);
        }
        if (num_sent > 100) {
            continue;
        }
        usleep(10000);

        fclose(fd);
    }

    close(sock);
    return 0;
}