#ifndef GPU_HSA_COMMAND_FACTORY_H
#define GPU_HSA_COMMAND_FACTORY_H

#include "Config.hpp"
#include "hsaDeviceInterface.hpp"
#include "bufferContainer.hpp"

class hsaCommand;

class hsaCommandMaker{
public:
    virtual hsaCommand *create(Config &config, const string &unique_name,
                bufferContainer &host_buffers, hsaDeviceInterface &device) const = 0;
 };

class hsaCommandFactory
{
public:
    hsaCommandFactory(Config &config, const string &unique_name,
                         bufferContainer &host_buffers, hsaDeviceInterface &device);
    virtual ~hsaCommandFactory();

    vector<hsaCommand *> &get_commands();
protected:
    Config &config;
    hsaDeviceInterface &device;
    bufferContainer &host_buffers;
    string unique_name;

    vector<hsaCommand *> list_commands;

private:
    std::map<std::string, hsaCommandMaker*> _hsa_commands;

    hsaCommand* create(const string &name,
                       Config& config,
                       const string &unique_name,
                       bufferContainer &host_buffers,
                       hsaDeviceInterface& device) const;
};


class hsaCommandFactoryRegistry {
public:
    static hsaCommandFactoryRegistry& Instance();
    std::map<std::string, hsaCommandMaker*> _hsa_commands;

    //Add the process to the registry.
    void hsaRegisterCommand(const std::string& key, hsaCommandMaker* cmd);

private:
    hsaCommandFactoryRegistry();
};


template<typename T>
class hsaCommandMakerTemplate : public hsaCommandMaker
{
    public:
        hsaCommandMakerTemplate(const std::string& key)
        {
            printf("Registering an HSA Command! %s\n",key.c_str());
            hsaCommandFactoryRegistry::Instance().hsaRegisterCommand(key, this);
        }
        virtual hsaCommand *create(Config &config, const string &unique_name,
                    bufferContainer &host_buffers, hsaDeviceInterface &device) const
        {
            return new T(config, unique_name, host_buffers, device);
        }
}; 
#define REGISTER_HSA_COMMAND(T) static hsaCommandMakerTemplate<T> maker(#T);

#include "hsaCommand.hpp"

#endif // GPU_COMMAND_FACTORY_H
