#ifndef CL_COMMAND_FACTORY_H
#define CL_COMMAND_FACTORY_H

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

#include "Config.hpp"
#include "correlator_kernel.h"
#include "offset_kernel.h"
#include "preseed_kernel.h"
#include "clDeviceInterface.hpp"
#include "input_data_stage.h"
#include "output_data_result.h"
#include "callbackdata.h"
#include "beamform_kernel.h"
#include "beamform_phase_data.h"
#include "output_beamform_result.h"
#include "beamform_incoherent_kernel.h"
#include "output_beamform_incoh_result.h"
#include "rfi_kernel.h"
#include "output_rfi.h"
#include "timer.hpp"


class clCommand;

class clCommandMaker{
public:
    virtual clCommand *create(Config &config, const string &unique_name) const = 0;
 };

class clCommandFactory
{
public:
    clCommandFactory(class device_interface & param_Device, Config& param_Config, const string& unique_name);
    ~clCommandFactory();
    vector<clCommand *> &get_commands();
    //void initializeCommands(class device_interface & param_Device, Config& param_Config);

protected:
    cl_uint num_commands;
    cl_uint current_command_cnt;

    int use_beamforming;
    //int use_incoh_beamforming;

    Config &config;
    device_interface &device;
    string unique_name;

    vector<clCommand *> list_commands;

private:
    std::map<std::string, clCommandMaker*> _cl_commands;

    clCommand* create(const string &name,
                       Config& config,
                       const string &unique_name) const;

};



class clCommandFactoryRegistry {
public:
    //Add the process to the registry.
    static void cl_register_command(const std::string& key, clCommandMaker* cmd);
    //INFO all the known commands out.
    static std::map<std::string, clCommandMaker*> get_registered_commands();

private:
    static clCommandFactoryRegistry& instance();
    void cl_reg(const std::string& key, clCommandMaker* cmd);
    std::map<std::string, clCommandMaker*> _cl_commands;
    clCommandFactoryRegistry();
};


template<typename T>
class clCommandMakerTemplate : public clCommandMaker
{
    public:
        clCommandMakerTemplate(const std::string& key)
        {
            clCommandFactoryRegistry::cl_register_command(key, this);
        }
        virtual clCommand *create(Config &config, const string &unique_name) const
        {
            return new T(config, unique_name);
        }
}; 
#define REGISTER_CL_COMMAND(T) static clCommandMakerTemplate<T> maker##T(#T);

#include "clCommand.hpp"


#endif // CL_COMMAND_FACTORY_H
