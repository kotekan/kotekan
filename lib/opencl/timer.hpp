/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   timer.hpp
 * Author: iantretyakov
 *
 * Created on September 23, 2017, 12:55 AM
 */

#ifndef TIMER_HPP
#define TIMER_HPP

#ifdef __APPLE__
    #include <OpenCL/cl_platform.h>
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl_platform.h>
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <sys/time.h>
#include "errors.h"
#include <thread>
#include "pthread.h"

using std::string;
using std::vector;
using std::map;

//Not sure what this is, it throws compiler warnings
//and doesn't seem to do anything...
//#pragma(1)
struct time_interval {
    time_interval(){
        CHECK_ERROR( pthread_mutex_init(&lock, NULL) );
    }
    ~time_interval(){
        CHECK_ERROR( pthread_mutex_destroy(&lock) );
    }
    clock_t start_time;
    clock_t stop_time;
    float interval;
    pthread_mutex_t lock;  // Lock for the is_ready function.
};

class timer {
public:
    timer();
    //timer(const timer& orig);
    ~timer();
    void start(string interval_name);
    void stop(string interval_name);
    void broadcast(string interval_name);
    int get_num_interval(string interval_name);
    void print_std(string interval_name);
    void print_min(string interval_name);
    void print_max(string interval_name);
    void print_avg(string interval_name);
    void time_opencl_single_kernel(cl_command_queue commands, cl_event profile_event, string interval_name);
    void time_opencl_multi_kernel(cl_event profile_event, string interval_name);
//    void kernel_profiling_thread();
protected:
    float calculate_std(string interval_name);
    float find_min(string interval_name);
    float find_max(string interval_name);
    float calculate_avg(string interval_name);
    map<string, vector<time_interval * > > time_list;
    bool first_run = false;
//    std::thread kernel_profiling_thread_handle;
};

void CL_CALLBACK profileCallback(cl_event param_event, cl_int param_status, void *data);

#endif /* TIMER_HPP */

