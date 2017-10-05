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


#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <sys/time.h>
#include "errors.h"

using std::string;
using std::vector;
using std::map;

#pragma(1)
struct time_interval {
    clock_t start_time;
    clock_t stop_time;
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
protected:
    void populate_interval(string interval_name, float interval);
    float calculate_std(string interval_name);
    float find_min(string interval_name);
    float find_max(string interval_name);
    float calculate_avg(string interval_name);
    map<string, vector<time_interval * > > time_list;
    map<string, vector<float > > interval_list;
};


#endif /* TIMER_HPP */

