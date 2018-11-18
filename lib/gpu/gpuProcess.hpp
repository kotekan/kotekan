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
    void results_thread();
    void profile_callback(connectionInstance& conn);
protected:
    virtual gpuCommand *create_command(json cmd) = 0;
    virtual gpuEventContainer *create_signal() = 0;
    void init(void);

    vector<gpuEventContainer*> final_signals;
    bufferContainer local_buffer_container;

    std::thread results_thread_handle;
    gpuDeviceInterface * device;
    std::vector<gpuCommand *> commands;

    // Config variables
    uint32_t _gpu_buffer_depth;
    uint32_t gpu_id;
};

#endif //GPU_PROCESS_H
