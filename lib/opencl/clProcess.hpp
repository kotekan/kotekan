#ifndef CL_PROCESS_H
#define CL_PROCESS_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

#include "gpuProcess.hpp"
#include "clDeviceInterface.hpp"
#include "clEventContainer.hpp"
#include "clCommand.hpp"

class clProcess final : public gpuProcess {
public:
    clProcess(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container);
    virtual ~clProcess();
    void main_thread() override;
    void results_thread();
    void profile_callback(connectionInstance& conn);
protected:
    gpuCommand *create_command(json cmd) override;
    gpuEventContainer *create_signal() override;

//    vector<clEventContainer> final_signals;

    std::thread results_thread_handle;
};

#endif //CL_PROCESS_H
