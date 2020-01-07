#include "tx_utils.hpp"

#include <cstring>   // for strlen
#include <stdexcept> // for runtime_error
#include <stdlib.h>  // for atoi, malloc
#include <time.h>    // for timespec
#include <unistd.h>  // for gethostname

void parse_chime_host_name(int& my_rack, int& my_node, int& my_nos, int& my_node_id) {
    int rack = 0, node = 0, nos = 0;
    // std::stringstream temp_ip[number_of_subnets];
    char* my_host_name = (char*)malloc(sizeof(char) * 100);
    // CHECK_MEM(my_host_name);
    gethostname(my_host_name, sizeof(my_host_name));

    if (my_host_name[0] != 'c' && my_host_name[3] != 'g') {
        // INFO_NON_OO("Not a valid name \n");
        throw std::runtime_error("Invalid host name");
    }


    if (my_host_name[1] == 'n') {
        nos = 0;
        my_node_id = 0;
    } else if (my_host_name[1] == 's') {
        nos = 100;
        my_node_id = 128;
    } else {
        throw std::runtime_error("Invalid host name");
    }

    switch (my_host_name[2]) {
        case '0':
            rack = 0;
            break;
        case '1':
            rack = 1;
            break;
        case '2':
            rack = 2;
            break;
        case '3':
            rack = 3;
            break;
        case '4':
            rack = 4;
            break;
        case '5':
            rack = 5;
            break;
        case '6':
            rack = 6;
            break;
        // case '7': rack=7; break;
        case '8':
            rack = 8;
            break;
        case '9':
            rack = 9;
            break;
        case 'A':
            rack = 10;
            break;
        case 'B':
            rack = 11;
            break;
        case 'C':
            rack = 12;
            break;
        case 'D':
            rack = 13;
            break;
        default:
            throw std::runtime_error("Invalid host name");
    }

    switch (my_host_name[4]) {
        case '0':
            node = 0;
            break;
        case '1':
            node = 1;
            break;
        case '2':
            node = 2;
            break;
        case '3':
            node = 3;
            break;
        case '4':
            node = 4;
            break;
        case '5':
            node = 5;
            break;
        case '6':
            node = 6;
            break;
        case '7':
            node = 7;
            break;
        case '8':
            node = 8;
            break;
        case '9':
            node = 9;
            break;
        default:
            throw std::runtime_error("Invalid host name");
    }

    if (rack < 7)
        my_node_id += rack * 10 + (9 - node); // fix for the arrangment of nodes in the racks
    if (rack > 7)
        my_node_id += (rack - 1) * 10 + (9 - node);
    my_rack = rack;
    my_node = node;
    my_nos = nos;
}

void add_nsec(struct timespec& temp, long nsec) {
    temp.tv_nsec += nsec;
    if (temp.tv_nsec >= 1000000000) {
        long sec = temp.tv_nsec / 1000000000;

        temp.tv_sec += sec;
        temp.tv_nsec -= sec * 1000000000;
    } else if (temp.tv_nsec < 0) {
        long sec = temp.tv_nsec / 1000000000;
        sec -= 1;
        temp.tv_nsec -= sec * 1000000000;
        temp.tv_sec += sec;
    }
}

int get_vlan_from_ip(const char* ip_address) {
    int len = 0;
    int location = 0;
    char* temp;
    int vlan = 0;
    int flag = 0;
    for (unsigned int i = 0; i < std::strlen(ip_address); i++) {
        if (ip_address[i] == '.' && flag == 0) {
            location = i + 1;
            flag = 1;
        }
    }
    flag = 0;
    for (unsigned int i = location; i < std::strlen(ip_address); i++) {
        if (ip_address[i] != '.' && flag == 0)
            len++;
        else
            flag = 1;
    }

    temp = new char[len];
    flag = 0;
    for (unsigned int i = location; i < std::strlen(ip_address); i++) {
        if (ip_address[i] != '.' && flag == 0)
            temp[i - location] = ip_address[i];
        else
            flag = 1;
    }

    vlan = atoi(temp);
    return vlan;
}


#ifdef MAC_OSX

void osx_clock_abs_nanosleep(clockid_t clock, struct timespec ts) {
    timespec t0;
    clock_gettime(clock, &t0);
    long sec = ts.tv_sec - t0.tv_sec;
    long nsec = ts.tv_nsec - t0.tv_nsec;
    if (nsec < 0) {
        nsec += 1000000000;
        sec -= 1;
    }
    if (sec < 0) {
        return;
    } else {
        timespec waittime;
        waittime.tv_sec = sec;
        waittime.tv_nsec = nsec;
        nanosleep(&waittime, nullptr);
    }
    return;
}

#endif
