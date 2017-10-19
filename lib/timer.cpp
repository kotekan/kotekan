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
/*
The profiling times are given in billionths of a second, called nanoseconds or ns. But
not every device can resolve time down to individual nanoseconds. To determine the resolution of a device, call clGetDeviceInfo with CL_DEVICE_PROFILING_TIMER_ RESOLUTION set as the second argument. This is shown in the following code:
size_t time_res;
clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(time_res), &time_res, NULL);

void time_opencl_kernel(){
    CL_QUEUE_PROFILING_ENABLE

            commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
cl_event
    // Sample Code that can be used for timing kernel execution duration
    // Using different parameters for cl_profiling_info allows us to
    // measure the wait time
    cl_event timing_event;
    cl_int err_code;
    //! We are timing the clEnqueueNDRangeKernel call and timing //information will be stored in timing_event
err_code = clEnqueueNDRangeKernel ( command_queue, kernel, work_dim,
        global_work_offset, global_work_size, local_work_size, 0,
NULL,
&timing_event);
clFinish(command_queue);
cl_ulong starttime;
cl_ulong endtime;

err_code = clGetEventProfilingInfo( timing_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);

kerneltimer = clGetEventProfilingInfo( timing_event,
CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);

unsigned long elapsed = (unsigned long)(endtime - starttime);

printf("Kernel Execution\t%ld ns\n",elapsed);
}

// set up platform, context, and devices (not shown) // Create a command-queue with profiling enabled cl_command_queue commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
// set up program, kernel, memory objects (not shown) cl_event prof_event;
err = clEnqueueNDRangeKernel(commands, kernel, nd, NULL, global, NULL, 0, NULL, prof_event);
clFinish(commands);
err = clWaitForEvents(1, &prof_event );
cl_ulong ev_start_time=(cl_ulong)0;
cl_ulong ev_end_time=(cl_ulong)0;
size_t return_bytes;
err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong), &ev_start_time, &return_bytes);
err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes);
run_time =(double)(ev_end_time - ev_start_time);
printf("\n profile data %f secs\n",run_time*1.0e-9);






        #include "mult.h"
#include "kernels.h"
void CL_CALLBACK eventCallback(cl_event ev, cl_int event_status, void * user_data)
{
int err, evID = (int)user_data;
cl_ulong ev_start_time=(cl_ulong)0;
cl_ulong ev_end_time=(cl_ulong)0;
size_t return_bytes;
double run_time;
printf(" Event callback %d %d ",(int)event_status, evID);
err = clGetEventProfilingInfo( ev, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &ev_start_time, &return_bytes);
err = clGetEventProfilingInfo( ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes);
run_time = (double)(ev_end_time - ev_start_time);
printf("\n kernel runtime %f secs\n",run_time*1.0e-9);
}
//------------------------------------------------------------------ int main(int argc, char **argv) {
// Declarations and platform definitions that are not shown.
commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
cl_event prof_event; //event to trigger the DAG
cl_event uevent = clCreateUserEvent(context, &err); // Set up the DAG of commands and profiling callbacks
err = clEnqueueNDRangeKernel(commands, kernel, nd, NULL, global, NULL, 1, &uevent, &prof_event);
int ID=0;
err = clSetEventCallback (prof_event, CL_COMPLETE, &eventCallback,(void *)ID);
// Once the DAG of commands is set up (we showed only one) // trigger the DAG using prof_event to profile execution // of the DAG
err = clSetUserEventStatus(uevent, CL_SUCCESS);
 */