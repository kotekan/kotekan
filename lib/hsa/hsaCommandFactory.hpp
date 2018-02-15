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

    void hsaRegisterCommand(const std::string& key, hsaCommandMaker* cmd);
    static hsaCommandFactory& Instance();

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

template<typename T>
class hsaCommandMakerTemplate : public hsaCommandMaker
{
    public:
        hsaCommandMakerTemplate(const std::string& key)
        {
            printf("Registering! %s\n",key.c_str());
            hsaCommandFactory::Instance().hsaRegisterCommand(key, this);
        }
        virtual hsaCommand *create(Config &config, const string &unique_name,
                    bufferContainer &host_buffers, hsaDeviceInterface &device) const
        {
            return new T(config, unique_name, host_buffers, device);
        }
}; 
#define REGISTER_HSA_COMMAND(T) static hsaCommandMakerTemplate<T> maker(#T);

#endif // GPU_COMMAND_FACTORY_H
