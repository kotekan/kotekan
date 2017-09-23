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


#include <vector>
#include <string>
#include <sys/time.h>
#include "errors.h"

using std::string;
using std::vector;

struct time_interval {
    time_interval();
    ~time_interval();
    string name;
    clock_t start_time;
    clock_t stop_time;
};

class timer {
public:
    timer();
    //timer(const timer& orig);
    virtual ~timer();
    int start(string interval_name);
    void stop(int interval_id);
    void broadcast(int interval_id);
    void broadcast(string interval_name);
    int get_num_interval();
protected:
    vector<time_interval *> time_list;
};


#endif /* TIMER_HPP */

