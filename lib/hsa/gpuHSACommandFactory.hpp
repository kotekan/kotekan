#ifndef GPU_HSA_COMMAND_FACTORY_H
#define GPU_HSA_COMMAND_FACTORY_H

#include "Config.hpp"
#include "gpuHSADeviceInterface.hpp"
#include "bufferContainer.hpp"
#include "gpuHSACommand.hpp"

class gpuHSACommandFactory
{
public:
    gpuHSACommandFactory(Config &config, gpuHSADeviceInterface &device,
                         bufferContainer &host_buffers);
    virtual ~gpuHSACommandFactory();

    vector<gpuHSAcommand *> &get_commands();
protected:
    Config &config;
    gpuHSADeviceInterface &device;

    bufferContainer &host_buffers;

    vector<gpuHSAcommand *> list_commands;
};

#endif // GPU_COMMAND_FACTORY_H
