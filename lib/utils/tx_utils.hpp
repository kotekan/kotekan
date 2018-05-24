#ifndef TX_UTILS_HPP
#define TX_UTILS_HPP



#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <string>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include<arpa/inet.h>
#include<sys/socket.h>
#include <netinet/in.h>
#include <cmath>

void parse_host_name(int &my_rack, int &my_node, int &my_nos, int &my_node_id);
void add_nsec(struct timespec &temp, long nsec);
#endif

