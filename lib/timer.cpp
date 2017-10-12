/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <numeric>
#include <complex>

#include "timer.hpp"

timer::timer() {
}

//timer::timer(const timer& orig) {
//}

timer::~timer() {
}

void timer::start(const string interval_name) {
    int i;

    auto it = time_list.find(interval_name);

    if (it == time_list.end()){
        time_list[interval_name];
    }
//    auto it = myMap.find("test");
//    if (it != myMap.end())
//        std::cout << "value for " << it->first << " is " << it->second << std::endl;
//    else
//        std::cout << "value not found" << std::endl;

    time_list[interval_name].push_back(new time_interval());

    i = time_list[interval_name].size() - 1;

    time_list[interval_name][i]->start_time = clock();
}

void timer::stop(string interval_name){
    int i;

    clock_t stop_time;
    float interval;

    i = time_list[interval_name].size() - 1;

    stop_time = clock();
    interval = (float)(stop_time - time_list[interval_name][i]->start_time)/(float)(CLOCKS_PER_SEC / 1000);

    time_list[interval_name][i]->stop_time = stop_time;

    populate_interval(interval_name, interval);

}

void timer::broadcast(string interval_name){
    string msg;
    for (uint32_t j = 0; j<time_list[interval_name].size(); j++){
        msg = interval_name + " - interval: " + std::to_string((time_list[interval_name][j]->stop_time - time_list[interval_name][j]->start_time)/(float)(CLOCKS_PER_SEC / 1000)) + " ms\n";
        INFO(msg.c_str());
    }
}

int timer::get_num_interval(string interval_name){
    return time_list[interval_name].size();
}

void timer::print_std(string interval_name){

    float ret;
    string msg;

    ret = calculate_std(interval_name);

    msg = interval_name + " - standard deviation: " + std::to_string(ret) + " ms\n";

    INFO(msg.c_str());
}
void timer::print_min(string interval_name){

    float ret;
    string msg;

    ret = find_min(interval_name);

    msg = interval_name + " - minimum: " + std::to_string(ret) + " ms\n";

    INFO(msg.c_str());
}
void timer::print_max(string interval_name){

    float ret;
    string msg;

    ret = find_max(interval_name);

    msg = interval_name + " - maximum: " + std::to_string(ret) + " ms\n";

    INFO(msg.c_str());
}
void timer::print_avg(string interval_name){

    float ret;
    string msg;

    ret = calculate_avg(interval_name);

    msg = interval_name + " - mean: " + std::to_string(ret) + " ms\n";

    INFO(msg.c_str());
}

void timer::populate_interval(const string interval_name, float interval){

    auto it = interval_list.find(interval_name);

    if (it == interval_list.end()){
        interval_list[interval_name];
        interval_list[interval_name].push_back(interval);
    }
    else{
        if (interval_list[interval_name].size() <= 5000)
            interval_list[interval_name].push_back(interval);
    }
}

float timer::calculate_avg(string interval_name){

    float sum = std::accumulate(interval_list[interval_name].begin(), interval_list[interval_name].end(), 0.0);
    float mean = sum / interval_list[interval_name].size();

    return mean;
}

float timer::find_min(string interval_name){

    auto result = std::min_element(interval_list[interval_name].begin(), interval_list[interval_name].end());

    return (float) *result;
}

float timer::find_max(string interval_name){

    auto result = std::max_element(interval_list[interval_name].begin(), interval_list[interval_name].end());

    return (float) *result;
}

float timer::calculate_std(string interval_name){

    float sum = std::accumulate(interval_list[interval_name].begin(), interval_list[interval_name].end(), 0.0);
    float mean = sum / interval_list[interval_name].size();

    std::vector<float> diff(interval_list[interval_name].size());
    std::transform(interval_list[interval_name].begin(), interval_list[interval_name].end(), diff.begin(), [mean](float x) { return x - mean; });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdev = std::sqrt(sq_sum / interval_list[interval_name].size());

    return stdev;
}

