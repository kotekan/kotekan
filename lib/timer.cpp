/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "timer.hpp"

timer::timer() {
}

//timer::timer(const timer& orig) {
//}

timer::~timer() {
}

int timer::start(string interval_name) {
    
    int i;
    
    time_list.push_back(new time_interval());
    
    i = time_list.size() - 1;
    
    time_list[i]->name = interval_name;
    
    time_list[i]->start_time = clock();
    
    return i;
}

void timer::stop(int interval_id){
    
    time_list[interval_id]->stop_time = clock();
}

void timer::broadcast(int interval_id){
    string msg = time_list[interval_id]->name + " - interval: " + std::to_string((time_list[interval_id]->start_time - time_list[interval_id]->stop_time)/(double)(CLOCKS_PER_SEC / 1000)) + " ms";
    INFO(msg.c_str());
}

void timer::broadcast(string interval_name){
    string msg;
    for (uint32_t j = 0; j<time_list.size(); j++){
        if (time_list[j]->name == interval_name) {
            msg = time_list[j]->name + " - interval: " + std::to_string((time_list[j]->start_time - time_list[j]->stop_time)/(double)(CLOCKS_PER_SEC / 1000)) + " ms\n";
            INFO(msg.c_str());
        }
    }
}

int timer::get_num_interval(){
    return time_list.size();
}
