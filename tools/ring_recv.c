#include<stdint.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<errno.h>
#include<sys/socket.h>
#include<arpa/inet.h>
#include<unistd.h>

int main(int argc , char *argv[])
{
    int socket_desc , client_sock , c , read_size;
    struct sockaddr_in server , client;

    fprintf(stderr, "starting\n");
    //Create socket
    socket_desc = socket(AF_INET, SOCK_STREAM , 0);
    if (socket_desc == -1)
    {
        fprintf(stderr, "Could not create socket");
    }
    fprintf(stderr, "Socket created\n");

    //Prepare the sockaddr_in structure
    bzero(&server, sizeof(struct sockaddr_in));
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons( 10042 );

    //Bind
    if( bind(socket_desc,(struct sockaddr *)&server , sizeof(server)) < 0)
    {
        //print the error message
        perror("bind failed. Error");
        return 1;
    }
    fprintf(stderr, "bind done\n");

    //Listen
    listen(socket_desc , 3);

    //Accept and incoming connection
    fprintf(stderr, "Waiting for incoming connections...\n");
    c = sizeof(struct sockaddr_in);

    //accept connection from an incoming client
    client_sock = accept(socket_desc, (struct sockaddr *)&client, (socklen_t*)&c);
    if (client_sock < 0)
    {
        perror("accept failed");
        return 1;
    }
    printf("Connection accepted");

    // Open the file to write
    FILE * fd;
    const int file_name_len = 100;
    char file_name[file_name_len];

    snprintf(file_name, file_name_len, "/mnt/ram_disk/power_data.dat");

    const int head_size = 3;
    const int line_head_size = 2;
    const int num_entries = 400000;
    const int line_size = 2048 + line_head_size;
    uint32_t buf[line_size + 3];

    // Delete the file before first use
    unlink(file_name);

    fd = fopen(file_name, "w+");

    if (fd == NULL) {
        perror("Cannot open file");
        fprintf(stderr, "File name was: %s", file_name);
        exit(errno);
    }

    // Grow the file to full size with zeros
    memset((void *)buf, 0, line_size*sizeof(int));
    fwrite((void *)buf, sizeof(int), head_size, fd);
    for (int i = 0; i < num_entries ; ++i) {
        fwrite((void *) buf, sizeof(int), line_size, fd);
    }
    fseek(fd, 0, SEEK_SET);

    //Receive a message from client
    while( (read_size = recv(client_sock , buf , (line_size + 3)*sizeof(int) , MSG_WAITALL)) > 0 )
    {
        if ( buf[0] > num_entries)
            printf("idx: %u", buf[0]);
        fseek(fd, sizeof(int) * (line_size * buf[0] + head_size), SEEK_SET);
        ssize_t ints_written = fwrite((void *)&buf[3], sizeof(int), line_size, fd);

        if (ints_written != line_size) {
            printf("Failed to write power data to ram disk!!!");
            fclose(fd);
        }

        fseek(fd, 0, SEEK_SET);
        fwrite((void*)buf, sizeof(int), 3, fd); // index, rolls, intergration time.
        //fflush(fd);
    }

    if(read_size == 0)
    {
        printf("Client disconnected");
        fflush(stdout);
    }
    else if(read_size == -1)
    {
        perror("recv failed");
    }

    return 0;
}