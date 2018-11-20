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

protected:
    gpuCommand *create_command(json cmd) override;
    gpuEventContainer *create_signal() override;
    void queue_commands(int gpu_frame_id) override;

    clDeviceInterface *device;
};

#endif //CL_PROCESS_H
