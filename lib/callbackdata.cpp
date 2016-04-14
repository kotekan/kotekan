#include "callbackdata.h"

callBackData::callBackData()
{

}

callBackData::callBackData(cl_uint param_NumCommand)
{
    listCommands = (gpu_command**)malloc(param_NumCommand * sizeof (class gpu_command *));
}

callBackData::~callBackData()
{
    free(listCommands);
}


