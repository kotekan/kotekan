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
    //remove the opencl timer thread.
}

void timer::start(const string interval_name) {
    int i;

    auto it = time_list.find(interval_name);

    if (it == time_list.end()){
        time_list[interval_name];
    }

    if (time_list[interval_name].size() <= 5000){

        time_list[interval_name].push_back(new time_interval());

        i = time_list[interval_name].size() - 1;

        time_list[interval_name][i]->start_time = clock();

        time_list[interval_name][i]->stop_time = -1;
    }
}

void timer::stop(string interval_name){
    int i;

    clock_t stop_time;

    i = time_list[interval_name].size() - 1;

    stop_time = clock();

    //Avoid overwriting 5000th stop_time value repeatedly.
    if (time_list[interval_name][i]->stop_time < 0){
        time_list[interval_name][i]->stop_time = stop_time;
        time_list[interval_name][i]->interval = (float)(stop_time - time_list[interval_name][i]->start_time)/(float)(CLOCKS_PER_SEC / 1000);
    }

}

void timer::broadcast(string interval_name){
    string msg;
    for (uint32_t j = 0; j<1; j++){//j<time_list[interval_name].size(); j++){
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

float timer::calculate_avg(string interval_name){


    float sum = std::accumulate (std::begin(time_list[interval_name]), std::end(time_list[interval_name]), 0,
    [](float i, const time_interval* o){ return o->interval + i; });

//    float sum = std::accumulate(time_list[interval_name].begin().interval, time_list[interval_name].end().interval, 0.0);
    float mean = sum / time_list[interval_name].size();

    return mean;
}

float timer::find_min(string interval_name){

//    auto result = std::min_element(time_list[interval_name].begin(), time_list[interval_name].end());

    auto result = std::min_element( time_list[interval_name].begin(), time_list[interval_name].end(),
                         []( const time_interval *a, const time_interval *b )
                         {
                             return a->interval < b->interval;
                         } );
//    return (float) *result;
    return (*result)[0].interval;
}

float timer::find_max(string interval_name){

    auto result = std::max_element( time_list[interval_name].begin(), time_list[interval_name].end(),
                             []( const time_interval *a, const time_interval *b )
                             {
                                 return a->interval < b->interval;
                             } );

//    auto result = std::max_element(time_list[interval_name].begin(), time_list[interval_name].end());

//    return (float) *result;
    return (*result)[0].interval;
}

float timer::calculate_std(string interval_name){

//    float sum = std::accumulate(time_list[interval_name].begin()->interval, time_list[interval_name].end()->interval, 0.0);
    float sum = std::accumulate (std::begin(time_list[interval_name]), std::end(time_list[interval_name]), 0,
    [](float i, const time_interval* o){ return o->interval + i; });
    float mean = sum / time_list[interval_name].size();

    std::vector<float> diff(time_list[interval_name].size());

//    std::transform(time_list[interval_name].begin(), time_list[interval_name].end(), diff.begin()
//    , [mean](const struct time_interval *  x) { return x->interval - mean; });
//    , [mean](auto x) { return (*x)[0]->interval - mean; });
    for (uint32_t i=0; i<time_list[interval_name].size(); i++)
    {
        diff.push_back(time_list[interval_name][i]->interval - mean);
    }
//    std::for_each(time_list[interval_name].begin(), time_list[interval_name].end()
//    , [mean, diff](auto x) { diff.push_back((*x)[0].interval - mean); });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdev = std::sqrt(sq_sum / time_list[interval_name].size());

    return stdev;
}

void timer::time_opencl_single_kernel(cl_command_queue commands, cl_event profile_event, string interval_name){
    cl_int err;
    int i;
    cl_ulong ev_start_time=(cl_ulong)0;
    cl_ulong ev_end_time=(cl_ulong)0;
    size_t return_bytes;
    //float interval;

    clFinish(commands);
    err = clWaitForEvents(1, &profile_event );
    if (err != CL_SUCCESS) ERROR("Error Waiting for CL profile event!");
    err = clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &ev_start_time, &return_bytes);
    if (err != CL_SUCCESS) ERROR("Couldn't Get Profiling Start Time!");
    err = clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes);
    if (err != CL_SUCCESS) ERROR("Couldn't Get Profiling Stop Time!");

    auto it = time_list.find(interval_name);

    if (it == time_list.end()){
        time_list[interval_name];
    }

    if (time_list[interval_name].size() <= 5000){
        time_list[interval_name].push_back(new time_interval());

        i = time_list[interval_name].size() - 1;

        time_list[interval_name][i]->start_time = (clock_t)ev_start_time;

        time_list[interval_name][i]->stop_time = (clock_t)ev_end_time;

        time_list[interval_name][i]->interval = (float)(ev_end_time - ev_start_time)*1.0e-6; //convert from ns to ms
    }
}

void timer::time_opencl_multi_kernel(cl_event profile_event, string interval_name){
    cl_int err;
    int i;

//    if (first_run)
//    {
//        kernel_profiling_thread_handle = std::thread(&timer::kernel_profiling_thread, std::ref(*this));
//
//        cpu_set_t cpuset;
//        CPU_ZERO(&cpuset);
//        for (int j = 4; j < 12; j++)
//            CPU_SET(j, &cpuset);
//        pthread_setaffinity_np(kernel_profiling_thread_handle.native_handle(),
//                                sizeof(cpu_set_t), &cpuset);
//
//        first_run = false;
//    }
    auto it = time_list.find(interval_name);

    if (it == time_list.end()){
        time_list[interval_name];
    }

    if (time_list[interval_name].size() <= 5000){
        time_list[interval_name].push_back(new time_interval());

        i = time_list[interval_name].size() - 1;

        err = clSetEventCallback (profile_event, CL_COMPLETE, &profileCallback, (void *)time_list[interval_name][i]);
        if (err != CL_SUCCESS) ERROR("Couldn't Set Profiling Callback!");
    }



    // TODO Make the exiting process actually work here.
//    kernel_profiling_thread_handle.join();

}

//void clProcess::kernel_profiling_thread()
//{
//    CHECK_ERROR( pthread_mutex_lock(&cb_data[j]->buff_id_lock->lock));
//
//    while (cb_data[j]->buff_id_lock->clean == 0) {
//        pthread_cond_wait(&cb_data[j]->buff_id_lock->clean_cond, &cb_data[j]->buff_id_lock->lock);
//        //INFO("GPU_THREAD: Read Complete Buffer ID %d", cb_data[j]->buffer_id);
//    }
//
//    CHECK_ERROR( pthread_mutex_unlock(&cb_data[j]->buff_id_lock->lock));
//}

void CL_CALLBACK profileCallback(cl_event ev, cl_int event_status, void * data){
    struct time_interval * cur_profile = (struct time_interval * )data;
    cl_int err;
    cl_ulong ev_start_time=(cl_ulong)0;
    cl_ulong ev_end_time=(cl_ulong)0;
    size_t return_bytes;

    CHECK_ERROR( pthread_mutex_lock(&cur_profile->lock));

    err = clGetEventProfilingInfo( ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, &return_bytes);
    if (err != CL_SUCCESS) ERROR("Couldn't Get Profiling Start Time!");
    err = clGetEventProfilingInfo( ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes);
    if (err != CL_SUCCESS) ERROR("Couldn't Set Profiling Stop Time!");

    cur_profile->start_time = (clock_t)ev_start_time;

    cur_profile->stop_time = (clock_t)ev_end_time;

    cur_profile->interval = (float)(ev_end_time - ev_start_time)*1.0e-6; //convert from ns to ms

    CHECK_ERROR( pthread_mutex_unlock(&cur_profile->lock));
}
