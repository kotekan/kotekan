#ifndef GPU_PROCESS_H
#define GPU_PROCESS_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

#include "gpuDeviceInterface.hpp"
#include "gpuEventContainer.hpp"
#include "KotekanProcess.hpp"
#include "restServer.hpp"
#include "gpuCommand.hpp"
#include "json.hpp"

class gpuProcess : public KotekanProcess {
public:
    gpuProcess(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container);
    virtual ~gpuProcess();
    void main_thread() override;
    void profile_callback(connectionInstance& conn);
protected:
    virtual gpuCommand *create_command(const std::string &cmd_name,
                                       const std::string &unique_name) = 0;
    virtual gpuEventContainer *create_signal() = 0;
    virtual void queue_commands(int gpu_frame_id) = 0;
    void results_thread();
    void init(void);

    vector<gpuEventContainer*> final_signals;
    bufferContainer local_buffer_container;

    bool log_profiling;
    // The mean expected time between frames in seconds
    double frame_arrival_period;

    std::thread results_thread_handle;
    gpuDeviceInterface * dev;
    std::vector<gpuCommand *> commands;

    // Config variables
    uint32_t _gpu_buffer_depth;
    uint32_t gpu_id;
};

#endif //GPU_PROCESS_H
